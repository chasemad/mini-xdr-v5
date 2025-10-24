#!/bin/bash
# ============================================================================
# Setup Fluent Bit on Azure T-Pot to forward logs to Mini-XDR
# ============================================================================

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Configuration
TPOT_IP="74.235.242.205"
TPOT_SSH_PORT="64295"
TPOT_SSH_KEY="$HOME/.ssh/mini-xdr-tpot-azure"

echo -e "${BLUE}╔═══════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  Azure T-Pot Fluent Bit Configuration Setup          ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════╝${NC}"
echo ""

# Step 1: Get Mini-XDR public URL
echo -e "${YELLOW}📡 Step 1: Expose Mini-XDR to the Internet${NC}"
echo ""
echo "Your Mini-XDR is running at http://localhost:8000"
echo "Azure T-Pot needs a public URL to send logs to."
echo ""
echo -e "${GREEN}Choose an option:${NC}"
echo "  1) Use ngrok (recommended for testing)"
echo "  2) I have my own public URL"
echo "  3) Skip for now (configure manually later)"
echo ""
read -p "Enter choice [1-3]: " choice

case $choice in
    1)
        # Use ngrok
        echo -e "${BLUE}Installing/using ngrok...${NC}"
        if ! command -v ngrok &> /dev/null; then
            echo -e "${YELLOW}ngrok not found. Installing...${NC}"
            brew install ngrok
        fi
        
        echo -e "${BLUE}Starting ngrok tunnel...${NC}"
        echo -e "${YELLOW}⚠️  Keep this terminal open while testing!${NC}"
        
        # Start ngrok in background and capture URL
        ngrok http 8000 > /tmp/ngrok.log 2>&1 &
        NGROK_PID=$!
        sleep 3
        
        # Get the public URL
        NGROK_URL=$(curl -s http://localhost:4040/api/tunnels | jq -r '.tunnels[0].public_url' 2>/dev/null || echo "")
        
        if [ -z "$NGROK_URL" ]; then
            echo -e "${RED}❌ Failed to start ngrok${NC}"
            exit 1
        fi
        
        echo -e "${GREEN}✅ ngrok tunnel started: $NGROK_URL${NC}"
        echo -e "${YELLOW}   PID: $NGROK_PID (kill with: kill $NGROK_PID)${NC}"
        
        MINI_XDR_URL="$NGROK_URL"
        ;;
    2)
        read -p "Enter your Mini-XDR public URL (e.g., https://xdr.yourdomain.com): " MINI_XDR_URL
        ;;
    3)
        echo -e "${YELLOW}⚠️  Skipping configuration. You'll need to set this up manually.${NC}"
        exit 0
        ;;
    *)
        echo -e "${RED}Invalid choice${NC}"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}Using Mini-XDR URL: $MINI_XDR_URL${NC}"
echo ""

# Step 2: Create Fluent Bit configuration
echo -e "${YELLOW}📝 Step 2: Creating Fluent Bit configuration...${NC}"

cat > /tmp/fluent-bit-tpot.conf << EOF
[SERVICE]
    Flush         5
    Log_Level     info
    Daemon        off
    HTTP_Server   On
    HTTP_Listen   0.0.0.0
    HTTP_Port     2020

# Cowrie SSH/Telnet Honeypot
[INPUT]
    Name              tail
    Path              /data/cowrie/log/cowrie.json
    Parser            json
    Tag               tpot.cowrie
    Refresh_Interval  5
    Read_from_Head    false
    Skip_Long_Lines   On
    Buffer_Chunk_Size 32k
    Buffer_Max_Size   64k

# Dionaea Multi-Protocol Honeypot
[INPUT]
    Name              tail
    Path              /data/dionaea/log/dionaea.json
    Parser            json
    Tag               tpot.dionaea
    Refresh_Interval  5
    Read_from_Head    false
    Skip_Long_Lines   On

# Suricata Network IDS
[INPUT]
    Name              tail
    Path              /data/suricata/log/eve.json
    Parser            json
    Tag               tpot.suricata
    Refresh_Interval  5
    Read_from_Head    false
    Skip_Long_Lines   On

# Output to Mini-XDR
[OUTPUT]
    Name  http
    Match tpot.*
    Host  $(echo $MINI_XDR_URL | sed 's~https\?://~~' | cut -d/ -f1)
    Port  $(if [[ $MINI_XDR_URL == https://* ]]; then echo 443; else echo 80; fi)
    URI   /ingest/multi
    Format json_stream
    Header Content-Type application/json
    json_date_key timestamp
    json_date_format iso8601
    Retry_Limit 3
    tls $(if [[ $MINI_XDR_URL == https://* ]]; then echo On; else echo Off; fi)
EOF

echo -e "${GREEN}✅ Configuration created${NC}"

# Step 3: Deploy to Azure T-Pot
echo ""
echo -e "${YELLOW}📤 Step 3: Deploying to Azure T-Pot...${NC}"

ssh -i "$TPOT_SSH_KEY" -p "$TPOT_SSH_PORT" azureuser@"$TPOT_IP" 'bash -s' << 'ENDSSH'
# Stop existing Fluent Bit if running
sudo systemctl stop fluent-bit 2>/dev/null || true

# Backup existing config
sudo cp /etc/fluent-bit/fluent-bit.conf /etc/fluent-bit/fluent-bit.conf.backup 2>/dev/null || true

# Create service file
sudo tee /etc/systemd/system/fluent-bit.service > /dev/null << 'SERVICEEOF'
[Unit]
Description=Fluent Bit for T-Pot Log Forwarding
After=network.target

[Service]
Type=simple
ExecStart=/usr/bin/fluent-bit -c /etc/fluent-bit/fluent-bit.conf
Restart=always
RestartSec=10
User=root

[Install]
WantedBy=multi-user.target
SERVICEEOF

echo "✅ Service file created"

# Enable and reload
sudo systemctl daemon-reload
sudo systemctl enable fluent-bit
echo "✅ Service enabled"

ENDSSH

# Copy our config
scp -i "$TPOT_SSH_KEY" -P "$TPOT_SSH_PORT" /tmp/fluent-bit-tpot.conf azureuser@"$TPOT_IP":/tmp/
ssh -i "$TPOT_SSH_KEY" -p "$TPOT_SSH_PORT" azureuser@"$TPOT_IP" \
    "sudo mv /tmp/fluent-bit-tpot.conf /etc/fluent-bit/fluent-bit.conf"

echo -e "${GREEN}✅ Configuration deployed${NC}"

# Step 4: Start Fluent Bit
echo ""
echo -e "${YELLOW}🚀 Step 4: Starting Fluent Bit service...${NC}"

ssh -i "$TPOT_SSH_KEY" -p "$TPOT_SSH_PORT" azureuser@"$TPOT_IP" \
    "sudo systemctl start fluent-bit && sudo systemctl status fluent-bit --no-pager -l"

echo ""
echo -e "${GREEN}╔═══════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  ✅ Azure T-Pot Fluent Bit Setup Complete!           ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}📊 Verification:${NC}"
echo "  1. Check Fluent Bit logs:"
echo "     ssh -i $TPOT_SSH_KEY -p $TPOT_SSH_PORT azureuser@$TPOT_IP 'sudo journalctl -u fluent-bit -f'"
echo ""
echo "  2. Check Mini-XDR events:"
echo "     curl http://localhost:8000/events?limit=10 | jq"
echo ""
echo "  3. Run verification again:"
echo "     ./scripts/testing/verify-azure-honeypot-integration.sh"
echo ""

if [ "$choice" = "1" ]; then
    echo -e "${YELLOW}⚠️  IMPORTANT: Keep ngrok running!${NC}"
    echo "   To stop ngrok: kill $NGROK_PID"
    echo "   To restart: ngrok http 8000"
    echo ""
fi

echo -e "${GREEN}🎯 Logs should start flowing to Mini-XDR within 30 seconds!${NC}"


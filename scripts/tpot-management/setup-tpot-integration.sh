#!/bin/bash
# Set up Mini-XDR backend for secure T-Pot honeypot integration
# Creates API keys, configures ingestion, and prepares log forwarding

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log() { echo -e "${BLUE}[$(date +'%H:%M:%S')]${NC} $1"; }
success() { echo -e "${GREEN}✅ $1${NC}"; }
warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
error() { echo -e "${RED}❌ $1${NC}"; }

echo "🔧 Setting up Mini-XDR for T-Pot Integration"
echo ""

# 1. Generate secure API key for T-Pot
log "Generating secure API key for T-Pot..."
TPOT_API_KEY=$(openssl rand -hex 32)

# 2. Create T-Pot configuration
log "Creating T-Pot configuration..."
mkdir -p $(cd "$(dirname "$0")/../.." .. pwd)/config/tpot

cat > $(cd "$(dirname "$0")/../.." .. pwd)/config/tpot/tpot-config.json << EOF
{
  "name": "T-Pot Honeypot",
  "api_key": "$TPOT_API_KEY",
  "source_type": "tpot",
  "hostname": "tpot-honeypot",
  "ingestion_endpoint": "http://localhost:8000/ingest/multi",
  "log_sources": {
    "cowrie": "/data/cowrie/log/cowrie.json",
    "dionaea": "/data/dionaea/log/dionaea.json", 
    "suricata": "/data/suricata/log/eve.json",
    "honeytrap": "/data/honeytrap/log/honeytrap.json",
    "elasticpot": "/data/elasticpot/log/elasticpot.json",
    "heralding": "/data/heralding/log/heralding.log"
  },
  "security": {
    "validate_signatures": true,
    "rate_limit": 1000,
    "allowed_ips": ["34.193.101.171"]
  }
}
EOF

# 3. Create Fluent Bit configuration for T-Pot
log "Creating Fluent Bit configuration for T-Pot..."
cat > $(cd "$(dirname "$0")/../.." .. pwd)/config/tpot/fluent-bit-tpot.conf << 'EOF'
[SERVICE]
    Flush         5
    Log_Level     info
    Daemon        off
    Parsers_File  parsers.conf
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
    Buffer_Max_Size   32k

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

# Honeytrap Network Honeypot
[INPUT]
    Name              tail
    Path              /data/honeytrap/log/honeytrap.json
    Parser            json
    Tag               tpot.honeytrap
    Refresh_Interval  5
    Read_from_Head    false
    Skip_Long_Lines   On

# Elasticpot Elasticsearch Honeypot
[INPUT]
    Name              tail
    Path              /data/elasticpot/log/elasticpot.json
    Parser            json
    Tag               tpot.elasticpot
    Refresh_Interval  5
    Read_from_Head    false
    Skip_Long_Lines   On

# Output to Mini-XDR
[OUTPUT]
    Name  http
    Match tpot.*
    Host  YOUR_LOCAL_IP
    Port  8000
    URI   /ingest/multi
    Format json_stream
    Header Authorization Bearer $TPOT_API_KEY
    Header Content-Type application/json
    json_date_key timestamp
    json_date_format iso8601
    Retry_Limit 3
    tls Off
EOF

# 4. Create T-Pot deployment script for Fluent Bit
log "Creating T-Pot deployment script..."
cat > $(cd "$(dirname "$0")/../.." .. pwd)/deploy-tpot-logging.sh << 'DEPLOYEOF'
#!/bin/bash
# Deploy Fluent Bit configuration to T-Pot honeypot
# Run this AFTER T-Pot is started and accessible

set -e

TPOT_IP="$1"
LOCAL_IP="$2"

if [ -z "$TPOT_IP" ] || [ -z "$LOCAL_IP" ]; then
    echo "Usage: $0 <tpot-ip> <local-ip>"
    echo "Example: $0 34.193.101.171 192.168.1.100"
    exit 1
fi

echo "🔧 Deploying log forwarding to T-Pot at $TPOT_IP"
echo "📡 Logs will be sent to Mini-XDR at $LOCAL_IP:8000"

# Replace LOCAL_IP in config
sed "s/YOUR_LOCAL_IP/$LOCAL_IP/g" $(cd "$(dirname "$0")/../.." .. pwd)/config/tpot/fluent-bit-tpot.conf > /tmp/fluent-bit-tpot.conf

# Copy configuration to T-Pot
echo "📁 Copying Fluent Bit configuration..."
scp -i ~/.ssh/mini-xdr-tpot-key.pem -P 64295 /tmp/fluent-bit-tpot.conf admin@$TPOT_IP:/tmp/

# Install and configure Fluent Bit on T-Pot
echo "🔧 Installing Fluent Bit on T-Pot..."
ssh -i ~/.ssh/mini-xdr-tpot-key.pem -p 64295 admin@$TPOT_IP << 'SSHEOF'
# Install Fluent Bit
curl https://raw.githubusercontent.com/fluent/fluent-bit/master/install.sh | sudo sh

# Stop default service
sudo systemctl stop fluent-bit || true
sudo systemctl disable fluent-bit || true

# Copy our configuration
sudo cp /tmp/fluent-bit-tpot.conf /etc/fluent-bit/fluent-bit.conf

# Create systemd service for T-Pot log forwarding
sudo tee /etc/systemd/system/tpot-fluent-bit.service > /dev/null << 'SERVICEEOF'
[Unit]
Description=Fluent Bit for T-Pot Log Forwarding
After=network.target tpot.service
Requires=tpot.service

[Service]
Type=simple
ExecStart=/opt/fluent-bit/bin/fluent-bit -c /etc/fluent-bit/fluent-bit.conf
Restart=always
RestartSec=10
User=root

[Install]
WantedBy=multi-user.target
SERVICEEOF

# Enable and start the service
sudo systemctl daemon-reload
sudo systemctl enable tpot-fluent-bit
sudo systemctl start tpot-fluent-bit

echo "✅ Fluent Bit configured and started"
SSHEOF

echo "✅ T-Pot log forwarding deployed successfully!"
echo "📊 Logs should now be flowing to Mini-XDR at $LOCAL_IP:8000"
DEPLOYEOF

chmod +x $(cd "$(dirname "$0")/../.." .. pwd)/deploy-tpot-logging.sh

# 5. Update Mini-XDR environment with T-Pot settings
log "Updating Mini-XDR environment..."
if [ -f "$(cd "$(dirname "$0")/../.." .. pwd)/backend/.env" ]; then
    # Add T-Pot configuration to existing .env
    echo "" >> $(cd "$(dirname "$0")/../.." .. pwd)/backend/.env
    echo "# T-Pot Honeypot Integration" >> $(cd "$(dirname "$0")/../.." .. pwd)/backend/.env
    echo "TPOT_API_KEY=$TPOT_API_KEY" >> $(cd "$(dirname "$0")/../.." .. pwd)/backend/.env
    echo "TPOT_HOST=34.193.101.171" >> $(cd "$(dirname "$0")/../.." .. pwd)/backend/.env
    echo "TPOT_SSH_PORT=64295" >> $(cd "$(dirname "$0")/../.." .. pwd)/backend/.env
    echo "TPOT_WEB_PORT=64297" >> $(cd "$(dirname "$0")/../.." .. pwd)/backend/.env
    success "Added T-Pot configuration to existing .env file"
else
    warning ".env file not found - you may need to create it"
fi

success "Mini-XDR T-Pot integration setup complete!"
echo ""
echo "📋 Next Steps:"
echo "1. Start your Mini-XDR backend: cd backend && python -m app.main"
echo "2. Start T-Pot when ready: ./start-secure-tpot.sh"
echo "3. Deploy log forwarding: ./deploy-tpot-logging.sh <tpot-ip> <your-local-ip>"
echo "4. Test with Kali: ./kali-access.sh add <kali-ip> 22 80 443"
echo ""
echo "🔑 T-Pot API Key: $TPOT_API_KEY"
echo "📁 Configuration files saved in: config/tpot/"
echo ""
echo "🔒 Security: Only your IPs can access T-Pot management and honeypots"

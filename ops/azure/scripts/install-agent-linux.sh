#!/bin/bash
# ============================================================================
# Mini-XDR Linux Agent Installer
# ============================================================================
# Installs and configures the Mini-XDR agent on Linux systems
# ============================================================================

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

# Configuration
BACKEND_URL="${1:-}"
API_KEY="${2:-}"
AGENT_TYPE="${3:-endpoint}"
INSTALL_PATH="/opt/minixdr"

if [ -z "$BACKEND_URL" ] || [ -z "$API_KEY" ]; then
    echo -e "${RED}Usage: $0 <backend-url> <api-key> [agent-type]${NC}"
    echo -e "${YELLOW}Example: $0 https://mini-xdr.example.com api-key-here endpoint${NC}"
    exit 1
fi

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║       Mini-XDR Linux Agent Installer                           ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo -e "${RED}✗ This script must be run as root (use sudo)${NC}"
    exit 1
fi

echo "Configuration:"
echo "  Backend URL: $BACKEND_URL"
echo "  Agent Type: $AGENT_TYPE"
echo "  Install Path: $INSTALL_PATH"
echo ""

# Step 1: Create installation directory
echo -e "${YELLOW}[1/7] Creating installation directory...${NC}"
mkdir -p "$INSTALL_PATH"
echo -e "${GREEN}✓ Created: $INSTALL_PATH${NC}"

# Step 2: Install dependencies
echo -e "${YELLOW}[2/7] Installing dependencies...${NC}"
if command -v apt-get &> /dev/null; then
    apt-get update -qq
    apt-get install -y -qq python3 python3-pip curl > /dev/null 2>&1
elif command -v yum &> /dev/null; then
    yum install -y -q python3 python3-pip curl > /dev/null 2>&1
else
    echo -e "${RED}✗ Unsupported package manager${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Dependencies installed${NC}"

# Step 3: Install Python packages
echo -e "${YELLOW}[3/7] Installing Python packages...${NC}"
pip3 install --quiet requests psutil > /dev/null 2>&1
echo -e "${GREEN}✓ Python packages installed${NC}"

# Step 4: Create agent script
echo -e "${YELLOW}[4/7] Creating agent script...${NC}"

cat > "$INSTALL_PATH/agent.py" << 'PYEOF'
#!/usr/bin/env python3
import os
import sys
import time
import json
import requests
import logging
from datetime import datetime
import psutil
import socket

# Configure logging
logging.basicConfig(
    filename='/var/log/minixdr-agent.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class MiniXDRAgent:
    def __init__(self):
        self.backend_url = os.environ.get('BACKEND_URL', '__BACKEND_URL__')
        self.api_key = os.environ.get('API_KEY', '__API_KEY__')
        self.hostname = socket.gethostname()
        self.agent_type = os.environ.get('AGENT_TYPE', '__AGENT_TYPE__')
        
    def collect_metrics(self):
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                'hostname': self.hostname,
                'agent_type': self.agent_type,
                'timestamp': datetime.utcnow().isoformat(),
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available': memory.available,
                'disk_percent': disk.percent,
                'disk_free': disk.free
            }
        except Exception as e:
            logging.error(f'Error collecting metrics: {e}')
            return None
            
    def send_heartbeat(self):
        try:
            metrics = self.collect_metrics()
            if metrics:
                response = requests.post(
                    f'{self.backend_url}/api/agents/heartbeat',
                    json=metrics,
                    headers={'X-API-Key': self.api_key},
                    timeout=10,
                    verify=False  # For self-signed certs
                )
                if response.status_code == 200:
                    logging.info('Heartbeat sent successfully')
                    return True
                else:
                    logging.error(f'Heartbeat failed: {response.status_code}')
        except Exception as e:
            logging.error(f'Error sending heartbeat: {e}')
        return False
        
    def run(self):
        logging.info(f'Mini-XDR Agent started - {self.hostname}')
        print(f'Mini-XDR Agent running on {self.hostname}')
        
        while True:
            try:
                self.send_heartbeat()
                time.sleep(60)  # Send heartbeat every minute
            except KeyboardInterrupt:
                logging.info('Agent stopped by user')
                break
            except Exception as e:
                logging.error(f'Agent error: {e}')
                time.sleep(60)

if __name__ == '__main__':
    agent = MiniXDRAgent()
    agent.run()
PYEOF

# Replace placeholders
sed -i "s|__BACKEND_URL__|$BACKEND_URL|g" "$INSTALL_PATH/agent.py"
sed -i "s|__API_KEY__|$API_KEY|g" "$INSTALL_PATH/agent.py"
sed -i "s|__AGENT_TYPE__|$AGENT_TYPE|g" "$INSTALL_PATH/agent.py"

chmod +x "$INSTALL_PATH/agent.py"
echo -e "${GREEN}✓ Agent script created${NC}"

# Step 5: Create systemd service
echo -e "${YELLOW}[5/7] Creating systemd service...${NC}"

cat > /etc/systemd/system/minixdr-agent.service << EOF
[Unit]
Description=Mini-XDR Security Agent
After=network.target

[Service]
Type=simple
User=root
Environment="BACKEND_URL=$BACKEND_URL"
Environment="API_KEY=$API_KEY"
Environment="AGENT_TYPE=$AGENT_TYPE"
ExecStart=/usr/bin/python3 $INSTALL_PATH/agent.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
echo -e "${GREEN}✓ Systemd service created${NC}"

# Step 6: Enable and start service
echo -e "${YELLOW}[6/7] Enabling and starting service...${NC}"
systemctl enable minixdr-agent > /dev/null 2>&1
systemctl start minixdr-agent

sleep 2

if systemctl is-active --quiet minixdr-agent; then
    echo -e "${GREEN}✓ Service started successfully${NC}"
else
    echo -e "${RED}✗ Service failed to start${NC}"
    echo "Check logs: journalctl -u minixdr-agent -n 50"
fi

# Step 7: Configure firewall (if applicable)
echo -e "${YELLOW}[7/7] Configuring firewall...${NC}"
if command -v ufw &> /dev/null; then
    ufw allow out 443/tcp > /dev/null 2>&1
    ufw allow out 8000/tcp > /dev/null 2>&1
    echo -e "${GREEN}✓ UFW rules added${NC}"
elif command -v firewall-cmd &> /dev/null; then
    firewall-cmd --permanent --add-port=443/tcp > /dev/null 2>&1
    firewall-cmd --permanent --add-port=8000/tcp > /dev/null 2>&1
    firewall-cmd --reload > /dev/null 2>&1
    echo -e "${GREEN}✓ Firewalld rules added${NC}"
else
    echo -e "${YELLOW}⚠ No firewall detected${NC}"
fi

# Display summary
echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║        Mini-XDR Agent Installation Complete!                  ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "Installation Details:"
echo "  • Install Path: $INSTALL_PATH"
echo "  • Service Name: minixdr-agent"
echo "  • Backend URL: $BACKEND_URL"
echo "  • Agent Type: $AGENT_TYPE"
echo ""
echo "Useful Commands:"
echo "  • Check status:  systemctl status minixdr-agent"
echo "  • View logs:     journalctl -u minixdr-agent -f"
echo "  • Restart:       systemctl restart minixdr-agent"
echo "  • Stop:          systemctl stop minixdr-agent"
echo ""


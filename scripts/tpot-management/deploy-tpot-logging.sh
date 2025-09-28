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
sed "s/YOUR_LOCAL_IP/$LOCAL_IP/g" /Users/chasemad/Desktop/mini-xdr/config/tpot/fluent-bit-tpot.conf > /tmp/fluent-bit-tpot.conf

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

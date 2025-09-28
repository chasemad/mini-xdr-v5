#!/bin/bash
# Configure the AWS relay with your local IP and update TPOT

set -e

RELAY_IP="$1"
YOUR_IP="$2"

if [ -z "$RELAY_IP" ] || [ -z "$YOUR_IP" ]; then
    echo "Usage: $0 <relay-ip> <your-ip>"
    echo "Example: $0 1.2.3.4 24.11.0.176"
    exit 1
fi

echo "🔧 Configuring Mini-XDR Relay for TPOT connectivity"
echo "📡 Relay IP: $RELAY_IP"
echo "🏠 Your IP: $YOUR_IP"
echo ""

# Configure relay with your local IP
echo "⚙️  Configuring relay with your local IP..."
ssh -i ~/.ssh/mini-xdr-tpot-key.pem -o StrictHostKeyChecking=yes -o UserKnownHostsFile=~/.ssh/known_hosts ubuntu@$RELAY_IP << RELAY_CONFIG
# Update relay script with your IP
sed -i "s/YOUR_LOCAL_IP_PLACEHOLDER/$YOUR_IP/" /home/ubuntu/relay.py

# Start the relay service
sudo systemctl enable minixdr-relay
sudo systemctl start minixdr-relay

echo "✅ Relay service started"
sudo systemctl status minixdr-relay --no-pager
RELAY_CONFIG

# Update TPOT Fluent Bit to use relay
echo "📡 Updating TPOT to send logs to relay..."

# Create new Fluent Bit config with relay IP
sed "s/10\.0\.0\.222/$RELAY_IP/" config/tpot/fluent-bit-tpot.conf > /tmp/fluent-bit-relay.conf
sed -i "s/\$TPOT_API_KEY/6c49b95dd921e0003ce159e6b3c0b6eb4e126fc2b19a1530a0f72a4a9c0c1eee/" /tmp/fluent-bit-relay.conf

# Deploy to TPOT
scp -i ~/.ssh/mini-xdr-tpot-key.pem -P 64295 -o StrictHostKeyChecking=yes -o UserKnownHostsFile=~/.ssh/known_hosts /tmp/fluent-bit-relay.conf admin@34.193.101.171:/tmp/

ssh -i ~/.ssh/mini-xdr-tpot-key.pem -p 64295 -o StrictHostKeyChecking=yes -o UserKnownHostsFile=~/.ssh/known_hosts admin@34.193.101.171 << TPOT_CONFIG
# Update Fluent Bit config
sudo cp /tmp/fluent-bit-relay.conf /etc/fluent-bit/fluent-bit.conf

# Restart Fluent Bit
sudo systemctl restart tpot-fluent-bit

echo "✅ TPOT configured to use relay"
sudo systemctl status tpot-fluent-bit --no-pager
TPOT_CONFIG

echo ""
echo "🎉 TPOT → AWS RELAY → Mini-XDR CONNECTION ESTABLISHED!"
echo ""
echo "📊 Data Flow:"
echo "   TPOT (34.193.101.171) → AWS Relay ($RELAY_IP) → Your Mini-XDR ($YOUR_IP:8000)"
echo ""
echo "✅ Your ML models and agents will now receive real attack data!"
echo "✅ Globe visualization will show real global attacks!"
echo ""
echo "🔍 Monitor the connection:"
echo "   Relay logs: ssh -i ~/.ssh/mini-xdr-tpot-key.pem ubuntu@$RELAY_IP 'tail -f /home/ubuntu/relay.log'"
echo "   Backend logs: tail -f backend/backend.log"

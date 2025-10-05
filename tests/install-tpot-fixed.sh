#!/bin/bash
# Install T-Pot on Azure VM (Fixed version)

set -e

TPOT_IP="74.235.242.205"
SSH_KEY="$HOME/.ssh/mini-xdr-tpot-azure"

echo "🔧 Installing T-Pot on Azure VM (FIXED)..."
echo "IP: $TPOT_IP"
echo ""

# Create install script with correct T-Pot installer syntax
cat > /tmp/install-tpot-fixed.sh << 'TPOT_SCRIPT'
#!/bin/bash
set -e

echo "================================================"
echo "Installing T-Pot Honeypot (STANDARD Edition)"
echo "================================================"
echo ""

# Update system
echo "[1/4] Updating system..."
sudo apt-get update -qq 2>/dev/null

# Install prerequisites
echo "[2/4] Installing prerequisites..."
sudo apt-get install -y git curl wget 2>/dev/null

# Clone T-Pot
echo "[3/4] Cloning T-Pot..."
if [ -d "/opt/tpotce" ]; then
    echo "T-Pot already cloned"
else
    cd /opt
    sudo git clone https://github.com/telekom-security/tpotce
fi

# Run installer with correct syntax
echo "[4/4] Running T-Pot installer..."
echo "Installing HIVE edition (includes web UI and honeypots)"
cd /opt/tpotce

# Install HIVE edition with web interface
# Default user: tsec, password: mini-xdr-tpot
sudo ./install.sh -s -t h -u tsec -p minixdrtpot2025

echo ""
echo "✅ T-Pot installation complete!"
echo "VM will reboot shortly..."
echo ""
echo "After reboot, access T-Pot at:"
echo "  Web: https://$(curl -s ifconfig.me):64297"
echo "  Username: tsec"
echo "  Password: minixdrtpot2025"
echo ""

# Reboot
sleep 10
sudo reboot
TPOT_SCRIPT

# Upload script
echo "📤 Uploading fixed install script..."
scp -o StrictHostKeyChecking=no -o ConnectTimeout=10 -i "$SSH_KEY" /tmp/install-tpot-fixed.sh "azureuser@$TPOT_IP:/tmp/" || {
    echo "❌ Could not upload script"
    exit 1
}

echo "✅ Script uploaded"
echo ""

# Run installer
echo "🚀 Starting T-Pot installation..."
echo "   This will take 15-30 minutes"
echo "   VM will automatically reboot when done"
echo ""

ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -i "$SSH_KEY" "azureuser@$TPOT_IP" "bash /tmp/install-tpot-fixed.sh" 2>&1 || true

echo ""
echo "✅ T-Pot installation initiated!"
echo ""
echo "📋 T-Pot Credentials:"
echo "   Username: tsec"
echo "   Password: minixdrtpot2025"
echo ""
echo "Next steps:"
echo "  1. Wait 20-30 minutes for installation"
echo "  2. Check status: ./check-tpot-status.sh"
echo "  3. Access T-Pot: https://$TPOT_IP:64297"
echo ""


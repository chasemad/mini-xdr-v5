#!/bin/bash
# Install T-Pot on Azure VM

set -e

TPOT_IP="74.235.242.205"
SSH_KEY="$HOME/.ssh/mini-xdr-tpot-azure"

echo "🔧 Installing T-Pot on Azure VM..."
echo "IP: $TPOT_IP"
echo ""

# Create install script
cat > /tmp/install-tpot.sh << 'TPOT_SCRIPT'
#!/bin/bash
set -e

echo "================================================"
echo "Installing T-Pot Honeypot"
echo "================================================"
echo ""

# Update system
echo "[1/5] Updating system packages..."
sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get upgrade -y -qq

# Install prerequisites
echo "[2/5] Installing prerequisites..."
sudo apt-get install -y git curl wget

# Clone T-Pot
echo "[3/5] Cloning T-Pot repository..."
if [ -d "/opt/tpotce" ]; then
    echo "T-Pot already cloned"
else
    cd /opt
    sudo git clone https://github.com/telekom-security/tpotce
fi

# Run installer
echo "[4/5] Running T-Pot installer..."
echo "This will take 15-30 minutes..."
cd /opt/tpotce

# Install with STANDARD edition (auto mode)
sudo ./install.sh --type=auto --conf=standard

echo "[5/5] T-Pot installation complete!"
echo "VM will reboot shortly..."
echo ""
echo "After reboot, access T-Pot at:"
echo "  Web: https://$(curl -s ifconfig.me):64297"
echo "  Default user: tsec"
echo "  Password: Check /opt/tpot/etc/tpot.yml after reboot"

# Reboot in 30 seconds
sleep 30
sudo reboot
TPOT_SCRIPT

# Upload script
echo "📤 Uploading install script..."
scp -o StrictHostKeyChecking=no -o ConnectTimeout=10 -i "$SSH_KEY" /tmp/install-tpot.sh "azureuser@$TPOT_IP:/tmp/" || {
    echo "❌ Could not upload script. VM may not be accessible."
    echo "Try manually: ssh -i $SSH_KEY azureuser@$TPOT_IP"
    exit 1
}

echo "✅ Script uploaded"
echo ""

# Run installer
echo "🚀 Starting T-Pot installation (this will take 15-30 minutes)..."
echo "The VM will automatically reboot when done."
echo ""

ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -i "$SSH_KEY" "azureuser@$TPOT_IP" "bash /tmp/install-tpot.sh" 2>&1 || {
    echo ""
    echo "ℹ️  Connection closed (VM is rebooting or installing)"
    echo ""
}

echo ""
echo "✅ T-Pot installation initiated!"
echo ""
echo "Next steps:"
echo "  1. Wait 20-30 minutes for installation to complete"
echo "  2. Check status: ./check-tpot-status.sh"
echo "  3. Access T-Pot: https://$TPOT_IP:64297"
echo "  4. Run test attack: ./test-honeypot-attack.sh"
echo ""


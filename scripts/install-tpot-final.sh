#!/bin/bash
# Install T-Pot on Azure VM (Non-root version)

set -e

TPOT_IP="74.235.242.205"
SSH_KEY="$HOME/.ssh/mini-xdr-tpot-azure"

echo "🔧 Installing T-Pot on Azure VM (as regular user)..."
echo "IP: $TPOT_IP"
echo ""

# Create install script that runs as regular user
cat > /tmp/install-tpot-final.sh << 'TPOT_SCRIPT'
#!/bin/bash
set -e

echo "================================================"
echo "Installing T-Pot Honeypot"
echo "================================================"
echo ""

# Update system (needs sudo)
echo "[1/4] Updating system..."
sudo apt-get update -qq 2>/dev/null

# Install prerequisites (needs sudo)
echo "[2/4] Installing prerequisites..."
sudo apt-get install -y git curl wget 2>/dev/null

# Clone T-Pot to user's home directory first
echo "[3/4] Cloning T-Pot..."
cd ~
if [ -d "tpotce" ]; then
    echo "T-Pot already cloned"
else
    git clone https://github.com/telekom-security/tpotce
fi

# Run installer as regular user (it will ask for sudo when needed)
echo "[4/4] Running T-Pot installer..."
echo "Installing HIVE edition with web UI"
cd ~/tpotce

# Install HIVE edition - run without sudo, let installer handle it
./install.sh -s -t h -u tsec -p minixdrtpot2025

echo ""
echo "✅ T-Pot installation complete!"
echo ""
TPOT_SCRIPT

# Upload script
echo "📤 Uploading install script..."
scp -o StrictHostKeyChecking=no -o ConnectTimeout=10 -i "$SSH_KEY" /tmp/install-tpot-final.sh "azureuser@$TPOT_IP:/tmp/" || {
    echo "❌ Could not upload script"
    exit 1
}

echo "✅ Script uploaded"
echo ""

# Run installer as regular user
echo "🚀 Starting T-Pot installation (as azureuser)..."
echo "   This will take 15-30 minutes"
echo "   VM will automatically reboot when done"
echo ""

ssh -o StrictHostKeyChecking=no -o ServerAliveInterval=60 -o ConnectTimeout=10 -i "$SSH_KEY" "azureuser@$TPOT_IP" "bash /tmp/install-tpot-final.sh" 2>&1 || {
    echo ""
    echo "ℹ️  Connection may have closed (installation running or VM rebooting)"
}

echo ""
echo "✅ T-Pot installation initiated!"
echo ""
echo "📋 T-Pot Credentials:"
echo "   Web URL: https://74.235.242.205:64297"
echo "   Username: tsec"
echo "   Password: minixdrtpot2025"
echo ""
echo "⏱️  Installation Progress:"
echo "   • Installation takes 15-30 minutes"
echo "   • VM will reboot automatically when done"
echo "   • Check progress: ./check-tpot-status.sh"
echo ""


#!/bin/bash

# Configure T-Pot for Passwordless Sudo (Security Commands Only)
# This allows Mini-XDR AI agents to execute defensive actions

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  T-Pot Passwordless Sudo Configuration for Mini-XDR         ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

TPOT_HOST="203.0.113.42"
TPOT_PORT="64295"
TPOT_USER="luxieum"

echo "This script will configure T-Pot to allow passwordless sudo for:"
echo "  • UFW firewall commands (IP blocking)"
echo "  • iptables commands (IP blocking)"
echo "  • Docker commands (container management)"
echo "  • File reading for log monitoring"
echo ""
echo "Target: $TPOT_USER@$TPOT_HOST:$TPOT_PORT"
echo ""
read -p "Press Enter to continue or Ctrl+C to cancel..."
echo ""

# Create sudoers configuration file
cat > /tmp/mini-xdr-sudoers << 'EOF'
# Mini-XDR AI Agent Permissions
# Allow luxieum user to execute specific security commands without password

# UFW firewall management
luxieum ALL=(ALL) NOPASSWD: /usr/sbin/ufw status*
luxieum ALL=(ALL) NOPASSWD: /usr/sbin/ufw deny from *
luxieum ALL=(ALL) NOPASSWD: /usr/sbin/ufw delete *
luxieum ALL=(ALL) NOPASSWD: /usr/sbin/ufw allow from *
luxieum ALL=(ALL) NOPASSWD: /usr/sbin/ufw reload

# iptables firewall management
luxieum ALL=(ALL) NOPASSWD: /usr/sbin/iptables -I INPUT *
luxieum ALL=(ALL) NOPASSWD: /usr/sbin/iptables -D INPUT *
luxieum ALL=(ALL) NOPASSWD: /usr/sbin/iptables -L *
luxieum ALL=(ALL) NOPASSWD: /usr/sbin/iptables -F *

# Docker container management
luxieum ALL=(ALL) NOPASSWD: /usr/bin/docker ps*
luxieum ALL=(ALL) NOPASSWD: /usr/bin/docker stop *
luxieum ALL=(ALL) NOPASSWD: /usr/bin/docker start *
luxieum ALL=(ALL) NOPASSWD: /usr/bin/docker restart *
luxieum ALL=(ALL) NOPASSWD: /usr/bin/docker logs *
luxieum ALL=(ALL) NOPASSWD: /usr/bin/docker stats *

# Log file access
luxieum ALL=(ALL) NOPASSWD: /usr/bin/tail *
luxieum ALL=(ALL) NOPASSWD: /usr/bin/cat /home/luxieum/tpotce/data/*
luxieum ALL=(ALL) NOPASSWD: /usr/bin/wc *
luxieum ALL=(ALL) NOPASSWD: /usr/bin/test *

# System status commands
luxieum ALL=(ALL) NOPASSWD: /usr/bin/netstat *
luxieum ALL=(ALL) NOPASSWD: /usr/bin/ss *
luxieum ALL=(ALL) NOPASSWD: /usr/sbin/journalctl *
EOF

echo "Created sudoers configuration file"
echo ""

# Upload and install sudoers file
echo "Uploading configuration to T-Pot..."
scp -P $TPOT_PORT /tmp/mini-xdr-sudoers $TPOT_USER@$TPOT_HOST:/tmp/mini-xdr-sudoers

if [ $? -eq 0 ]; then
    echo "✅ File uploaded"
    echo ""

    echo "Installing sudoers configuration (requires password)..."
    ssh -p $TPOT_PORT $TPOT_USER@$TPOT_HOST << 'REMOTE_EOF'
        echo "Validating sudoers file..."
        sudo visudo -c -f /tmp/mini-xdr-sudoers

        if [ $? -eq 0 ]; then
            echo "✅ Sudoers file is valid"
            echo ""
            echo "Installing to /etc/sudoers.d/..."
            sudo cp /tmp/mini-xdr-sudoers /etc/sudoers.d/90-mini-xdr
            sudo chmod 0440 /etc/sudoers.d/90-mini-xdr
            sudo chown root:root /etc/sudoers.d/90-mini-xdr
            echo "✅ Installed successfully"
            echo ""
            echo "Testing passwordless sudo..."
            sudo -n ufw status
            if [ $? -eq 0 ]; then
                echo "✅ Passwordless sudo is working!"
            else
                echo "⚠️  Passwordless sudo test failed"
            fi
        else
            echo "❌ Sudoers file validation failed"
            exit 1
        fi
REMOTE_EOF

    if [ $? -eq 0 ]; then
        echo ""
        echo "╔══════════════════════════════════════════════════════════════╗"
        echo "║              ✅ CONFIGURATION COMPLETE! ✅                   ║"
        echo "╚══════════════════════════════════════════════════════════════╝"
        echo ""
        echo "Your AI agents can now execute:"
        echo "  • IP blocking (UFW/iptables)"
        echo "  • Container management (docker stop/start)"
        echo "  • Log monitoring (tail/cat)"
        echo ""
        echo "Test it:"
        echo "  ssh -p $TPOT_PORT $TPOT_USER@$TPOT_HOST"
        echo "  sudo -n ufw status"
        echo ""
        echo "Then restart your Mini-XDR backend to enable real blocking!"
    else
        echo "❌ Configuration failed"
        exit 1
    fi
else
    echo "❌ Failed to upload file to T-Pot"
    exit 1
fi

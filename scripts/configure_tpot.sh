#!/bin/bash
# Configuration script for T-Pot integration
# This script helps set up the T-Pot SSH password and test connectivity

set -e

BACKEND_DIR="$(cd "$(dirname "$0")/.." && pwd)/backend"
ENV_FILE="$BACKEND_DIR/.env"

echo "==============================================="
echo "T-Pot Integration Configuration"
echo "==============================================="
echo

# Check if .env exists
if [ ! -f "$ENV_FILE" ]; then
    echo "❌ Error: .env file not found at $ENV_FILE"
    echo "Please copy env.example to .env first:"
    echo "  cp $BACKEND_DIR/env.example $ENV_FILE"
    exit 1
fi

echo "Current T-Pot configuration:"
echo "----------------------------"
grep "^TPOT_" "$ENV_FILE" || echo "No T-Pot configuration found"
echo

# Prompt for T-Pot configuration
read -p "T-Pot Host IP (default: 24.11.0.176): " TPOT_HOST
TPOT_HOST=${TPOT_HOST:-24.11.0.176}

read -p "T-Pot SSH Port (default: 64295): " TPOT_SSH_PORT
TPOT_SSH_PORT=${TPOT_SSH_PORT:-64295}

read -p "T-Pot SSH Username (default: luxieum): " TPOT_USER
TPOT_USER=${TPOT_USER:-luxieum}

echo
read -sp "T-Pot SSH Password: " TPOT_PASSWORD
echo

if [ -z "$TPOT_PASSWORD" ]; then
    echo "❌ Error: Password cannot be empty"
    exit 1
fi

# Update .env file
echo
echo "Updating .env file..."

# Remove old T-Pot configuration
sed -i.bak '/^TPOT_/d' "$ENV_FILE"
sed -i.bak '/^HONEYPOT_USER=/d' "$ENV_FILE"

# Add new configuration
cat >> "$ENV_FILE" << EOF

# T-Pot Honeypot Integration (configured $(date))
TPOT_HOST=$TPOT_HOST
TPOT_SSH_PORT=$TPOT_SSH_PORT
TPOT_API_KEY=$TPOT_PASSWORD
HONEYPOT_USER=$TPOT_USER
EOF

echo "✅ Configuration saved to .env"
echo

# Test connection
echo "Testing T-Pot connection..."
echo "----------------------------"

# Test SSH connection
if ssh -p "$TPOT_SSH_PORT" -o ConnectTimeout=5 -o StrictHostKeyChecking=no "$TPOT_USER@$TPOT_HOST" "echo 'Connected successfully'" 2>/dev/null; then
    echo "✅ SSH connection successful"

    # Test UFW access
    echo
    echo "Testing UFW status..."
    if ssh -p "$TPOT_SSH_PORT" "$TPOT_USER@$TPOT_HOST" "echo '$TPOT_PASSWORD' | sudo -S ufw status" 2>/dev/null | grep -q "Status: active"; then
        echo "✅ UFW access successful"
        echo
        echo "Current UFW rules:"
        ssh -p "$TPOT_SSH_PORT" "$TPOT_USER@$TPOT_HOST" "echo '$TPOT_PASSWORD' | sudo -S ufw status numbered" 2>/dev/null
    else
        echo "⚠️  Warning: Could not access UFW (may need sudo password or permissions)"
    fi
else
    echo "⚠️  Warning: Could not connect to T-Pot at $TPOT_HOST:$TPOT_SSH_PORT"
    echo "Please verify:"
    echo "  1. T-Pot is running and accessible"
    echo "  2. Your IP is allowed in T-Pot firewall"
    echo "  3. SSH credentials are correct"
fi

echo
echo "==============================================="
echo "Configuration complete!"
echo "==============================================="
echo
echo "Next steps:"
echo "1. Restart the backend server for changes to take effect"
echo "2. Check backend logs to verify T-Pot connection"
echo "3. Test IP blocking from the incident detail page"
echo
echo "To restart backend:"
echo "  cd $BACKEND_DIR && source venv/bin/activate && uvicorn app.main:app --reload"

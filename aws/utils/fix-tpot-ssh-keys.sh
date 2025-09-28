#!/bin/bash

# FIX TPOT SSH HOST KEY VERIFICATION
# Resolves the SSH host key verification issue

set -euo pipefail

TPOT_HOST="34.193.101.171"
TPOT_SSH_PORT="64295"
TPOT_USER="admin"
KEY_NAME="${KEY_NAME:-mini-xdr-tpot-key}"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() { echo -e "${GREEN}[$(date +'%H:%M:%S')] $1${NC}"; }
warn() { echo -e "${YELLOW}[WARN] $1${NC}"; }
error() { echo -e "${RED}[ERROR] $1${NC}"; }
step() { echo -e "${BLUE}$1${NC}"; }

echo "🔑 FIXING TPOT SSH HOST KEY VERIFICATION"
echo "========================================"

step "🔍 Checking SSH Key and Known Hosts Setup"

# Check if SSH key exists
if [ ! -f "$HOME/.ssh/${KEY_NAME}.pem" ]; then
    error "SSH key not found: $HOME/.ssh/${KEY_NAME}.pem"
    echo "Please ensure your TPOT SSH key is in place."
    exit 1
fi

# Set correct permissions on SSH key
chmod 600 "$HOME/.ssh/${KEY_NAME}.pem"
log "✅ SSH key permissions set correctly"

# Create SSH config directory if it doesn't exist
mkdir -p "$HOME/.ssh"
chmod 700 "$HOME/.ssh"

# Check if known_hosts exists
if [ ! -f "$HOME/.ssh/known_hosts" ]; then
    touch "$HOME/.ssh/known_hosts"
    chmod 644 "$HOME/.ssh/known_hosts"
    log "✅ Created known_hosts file"
fi

step "🔑 Adding TPOT Host Key to Known Hosts"

# Remove any existing entries for this host
ssh-keygen -R "[$TPOT_HOST]:$TPOT_SSH_PORT" 2>/dev/null || true
log "Removed any existing host keys for $TPOT_HOST:$TPOT_SSH_PORT"

# Scan and add the host key
log "Scanning for host keys on $TPOT_HOST:$TPOT_SSH_PORT..."
ssh-keyscan -p "$TPOT_SSH_PORT" "$TPOT_HOST" >> "$HOME/.ssh/known_hosts" 2>/dev/null

if [ $? -eq 0 ]; then
    log "✅ Host key successfully added to known_hosts"
else
    warn "Host key scan failed, trying alternative method..."
    
    # Alternative: Connect once with StrictHostKeyChecking=no to add the key
    log "Connecting once to establish host key..."
    echo "yes" | ssh -i "$HOME/.ssh/${KEY_NAME}.pem" \
        -p "$TPOT_SSH_PORT" \
        -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile="$HOME/.ssh/known_hosts" \
        -o ConnectTimeout=10 \
        "$TPOT_USER@$TPOT_HOST" "echo 'Host key established'" 2>/dev/null || true
fi

step "🧪 Testing SSH Connection"

# Test the connection
if ssh -i "$HOME/.ssh/${KEY_NAME}.pem" \
       -p "$TPOT_SSH_PORT" \
       -o StrictHostKeyChecking=yes \
       -o UserKnownHostsFile="$HOME/.ssh/known_hosts" \
       -o ConnectTimeout=10 \
       "$TPOT_USER@$TPOT_HOST" "echo 'SSH connection successful'" 2>/dev/null; then
    log "✅ SSH connection test: SUCCESS"
else
    error "❌ SSH connection test: FAILED"
    
    step "🔧 Troubleshooting Information"
    echo "Host: $TPOT_HOST:$TPOT_SSH_PORT"
    echo "User: $TPOT_USER"
    echo "Key: $HOME/.ssh/${KEY_NAME}.pem"
    echo ""
    echo "Manual connection command:"
    echo "ssh -i $HOME/.ssh/${KEY_NAME}.pem -p $TPOT_SSH_PORT $TPOT_USER@$TPOT_HOST"
    echo ""
    echo "If still failing, try:"
    echo "1. Check if TPOT is running: ping $TPOT_HOST"
    echo "2. Check if SSH port is open: nc -zv $TPOT_HOST $TPOT_SSH_PORT"
    echo "3. Verify your SSH key has access to this TPOT instance"
    exit 1
fi

step "✅ SSH Configuration Complete"

log "TPOT SSH connection is now properly configured"
log "Host key verification will work correctly"
log "Your containment agent should now be able to connect to TPOT"

echo ""
echo "🤖 Testing Containment Agent Connection..."

# Test if containment agent can now work
log "The containment agent should now be able to execute commands like:"
log "sudo iptables -I INPUT -s 192.168.1.200 -j DROP"
echo ""
log "✅ TPOT SSH setup is now secure and working!"
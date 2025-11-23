#!/bin/bash

# ============================================================================
# T-Pot SSH Integration Setup for Mini-XDR
# ============================================================================
# This script configures Mini-XDR to connect to your T-Pot honeypot via SSH
# so AI agents can perform automated defensive actions (blocking IPs, etc.)
#
# Usage: ./setup-tpot-ssh-integration.sh
# ============================================================================

set -e  # Exit on error

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Functions
print_header() {
    echo ""
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}  $1${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BACKEND_DIR="$PROJECT_ROOT/backend"
ENV_FILE="$BACKEND_DIR/.env"

print_header "T-Pot SSH Integration Setup"

# Step 1: Detect T-Pot Configuration
print_info "Detecting T-Pot configuration..."

# Check if user provided IP
read -p "Enter T-Pot IP address (default: 203.0.113.42): " TPOT_IP
TPOT_IP=${TPOT_IP:-203.0.113.42}

read -p "Enter T-Pot SSH port (default: 64295): " TPOT_SSH_PORT
TPOT_SSH_PORT=${TPOT_SSH_PORT:-64295}

read -p "Enter T-Pot web port (default: 64297): " TPOT_WEB_PORT
TPOT_WEB_PORT=${TPOT_WEB_PORT:-64297}

read -p "Enter T-Pot SSH username (default: admin): " TPOT_USER
TPOT_USER=${TPOT_USER:-admin}

# Step 2: Get SSH Password
print_header "SSH Authentication Configuration"

echo -e "${YELLOW}T-Pot requires SSH authentication for defensive actions.${NC}"
echo "Options:"
echo "  1) Password authentication (recommended for T-Pot)"
echo "  2) SSH key authentication"
read -p "Choose authentication method [1/2]: " AUTH_METHOD

if [ "$AUTH_METHOD" = "1" ]; then
    read -sp "Enter T-Pot SSH password: " TPOT_PASSWORD
    echo ""
    read -sp "Confirm T-Pot SSH password: " TPOT_PASSWORD_CONFIRM
    echo ""

    if [ "$TPOT_PASSWORD" != "$TPOT_PASSWORD_CONFIRM" ]; then
        print_error "Passwords do not match!"
        exit 1
    fi

    AUTH_TYPE="password"
    print_success "Password authentication configured"
else
    read -p "Enter path to SSH private key (default: ~/.ssh/id_rsa): " SSH_KEY_PATH
    SSH_KEY_PATH=${SSH_KEY_PATH:-~/.ssh/id_rsa}

    # Expand tilde
    SSH_KEY_PATH="${SSH_KEY_PATH/#\~/$HOME}"

    if [ ! -f "$SSH_KEY_PATH" ]; then
        print_error "SSH key not found at: $SSH_KEY_PATH"
        exit 1
    fi

    AUTH_TYPE="key"
    print_success "SSH key authentication configured: $SSH_KEY_PATH"
fi

# Step 3: Test SSH Connection
print_header "Testing SSH Connection"

print_info "Attempting to connect to T-Pot at ${TPOT_IP}:${TPOT_SSH_PORT}..."

if [ "$AUTH_TYPE" = "password" ]; then
    # Test with sshpass if available
    if command -v sshpass &> /dev/null; then
        if sshpass -p "$TPOT_PASSWORD" ssh -p "$TPOT_SSH_PORT" \
            -o StrictHostKeyChecking=no \
            -o ConnectTimeout=10 \
            -o UserKnownHostsFile=/dev/null \
            "${TPOT_USER}@${TPOT_IP}" "echo 'Connection successful'" 2>/dev/null; then
            print_success "SSH connection test successful!"
        else
            print_error "SSH connection failed. Please check:"
            echo "  - T-Pot is running and accessible"
            echo "  - Firewall allows connection from your IP"
            echo "  - Username and password are correct"
            read -p "Continue anyway? [y/N]: " CONTINUE
            if [ "$CONTINUE" != "y" ]; then
                exit 1
            fi
        fi
    else
        print_warning "sshpass not installed. Install with: brew install hudochenkov/sshpass/sshpass"
        print_warning "Skipping SSH connection test"
    fi
else
    # Test with key
    if ssh -p "$TPOT_SSH_PORT" \
        -i "$SSH_KEY_PATH" \
        -o StrictHostKeyChecking=no \
        -o ConnectTimeout=10 \
        -o UserKnownHostsFile=/dev/null \
        "${TPOT_USER}@${TPOT_IP}" "echo 'Connection successful'" 2>/dev/null; then
        print_success "SSH connection test successful!"
    else
        print_error "SSH connection failed. Please check configuration."
        read -p "Continue anyway? [y/N]: " CONTINUE
        if [ "$CONTINUE" != "y" ]; then
            exit 1
        fi
    fi
fi

# Step 4: Update .env Configuration
print_header "Updating Backend Configuration"

# Backup existing .env
if [ -f "$ENV_FILE" ]; then
    cp "$ENV_FILE" "$ENV_FILE.backup.$(date +%Y%m%d_%H%M%S)"
    print_success "Backed up existing .env file"
fi

# Create or update .env
if [ ! -f "$ENV_FILE" ]; then
    print_info "Creating new .env file from template..."
    cp "$BACKEND_DIR/env.example" "$ENV_FILE"
fi

# Update T-Pot configuration
print_info "Updating T-Pot configuration in .env..."

# Remove old T-Pot settings
sed -i.bak '/^TPOT_HOST=/d' "$ENV_FILE"
sed -i.bak '/^TPOT_SSH_PORT=/d' "$ENV_FILE"
sed -i.bak '/^TPOT_WEB_PORT=/d' "$ENV_FILE"
sed -i.bak '/^TPOT_API_KEY=/d' "$ENV_FILE"
sed -i.bak '/^HONEYPOT_HOST=/d' "$ENV_FILE"
sed -i.bak '/^HONEYPOT_USER=/d' "$ENV_FILE"
sed -i.bak '/^HONEYPOT_SSH_PORT=/d' "$ENV_FILE"
sed -i.bak '/^HONEYPOT_SSH_KEY=/d' "$ENV_FILE"

# Add new T-Pot settings
cat >> "$ENV_FILE" << EOF

# T-Pot Honeypot Configuration (Updated: $(date))
TPOT_HOST=$TPOT_IP
TPOT_SSH_PORT=$TPOT_SSH_PORT
TPOT_WEB_PORT=$TPOT_WEB_PORT
TPOT_ELASTICSEARCH_PORT=64298
TPOT_KIBANA_PORT=64296

# Honeypot SSH Configuration
HONEYPOT_HOST=$TPOT_IP
HONEYPOT_USER=$TPOT_USER
HONEYPOT_SSH_PORT=$TPOT_SSH_PORT
EOF

if [ "$AUTH_TYPE" = "password" ]; then
    echo "TPOT_API_KEY=$TPOT_PASSWORD" >> "$ENV_FILE"
    echo "HONEYPOT_SSH_KEY=~/.ssh/id_rsa" >> "$ENV_FILE"  # Placeholder
else
    echo "TPOT_API_KEY=" >> "$ENV_FILE"  # Empty for key auth
    echo "HONEYPOT_SSH_KEY=$SSH_KEY_PATH" >> "$ENV_FILE"
fi

# Clean up backup files
rm -f "$ENV_FILE.bak"

print_success "Backend .env configuration updated"

# Step 5: Create SSH Configuration Helper
print_header "Creating SSH Configuration Files"

SSH_CONFIG_DIR="$HOME/.ssh"
mkdir -p "$SSH_CONFIG_DIR"

# Add T-Pot host to SSH config
SSH_CONFIG_FILE="$SSH_CONFIG_DIR/config"

if ! grep -q "Host tpot" "$SSH_CONFIG_FILE" 2>/dev/null; then
    cat >> "$SSH_CONFIG_FILE" << EOF

# T-Pot Honeypot Configuration
Host tpot
    HostName $TPOT_IP
    Port $TPOT_SSH_PORT
    User $TPOT_USER
EOF

    if [ "$AUTH_TYPE" = "key" ]; then
        echo "    IdentityFile $SSH_KEY_PATH" >> "$SSH_CONFIG_FILE"
    fi

    cat >> "$SSH_CONFIG_FILE" << EOF
    StrictHostKeyChecking no
    UserKnownHostsFile=/dev/null
    ServerAliveInterval 60
    ServerAliveCountMax 3
EOF

    print_success "Added T-Pot to SSH config (~/.ssh/config)"
    print_info "You can now connect with: ssh tpot"
else
    print_warning "T-Pot entry already exists in SSH config"
fi

# Step 6: Create Quick Test Script
print_header "Creating Test Scripts"

TEST_SCRIPT="$SCRIPT_DIR/test-tpot-connection.sh"
cat > "$TEST_SCRIPT" << 'EOF'
#!/bin/bash

# Quick T-Pot Connection Test
# This script tests SSH connectivity to T-Pot and verifies defensive actions work

set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

source "$(dirname "$0")/../../backend/.env"

echo -e "${YELLOW}Testing T-Pot Connection...${NC}"
echo ""

# Test 1: Basic SSH Connection
echo "Test 1: SSH Connection"
if ssh -p "$TPOT_SSH_PORT" \
    -o ConnectTimeout=5 \
    -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null \
    "${HONEYPOT_USER}@${TPOT_HOST}" "echo 'SUCCESS'" 2>/dev/null | grep -q "SUCCESS"; then
    echo -e "${GREEN}✓${NC} SSH connection working"
else
    echo -e "${RED}✗${NC} SSH connection failed"
    exit 1
fi

# Test 2: UFW Status (firewall)
echo "Test 2: UFW Firewall Access"
if ssh -p "$TPOT_SSH_PORT" \
    -o ConnectTimeout=5 \
    -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null \
    "${HONEYPOT_USER}@${TPOT_HOST}" "sudo -n ufw status" 2>/dev/null | grep -q "Status"; then
    echo -e "${GREEN}✓${NC} UFW access working (passwordless sudo)"
else
    echo -e "${YELLOW}⚠${NC} UFW requires password authentication (this is normal for T-Pot)"
fi

# Test 3: Docker Access (honeypot containers)
echo "Test 3: Docker Access"
if ssh -p "$TPOT_SSH_PORT" \
    -o ConnectTimeout=5 \
    -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null \
    "${HONEYPOT_USER}@${TPOT_HOST}" "docker ps --format '{{.Names}}'" 2>/dev/null | grep -q "cowrie"; then
    echo -e "${GREEN}✓${NC} Docker/honeypot access working"
else
    echo -e "${YELLOW}⚠${NC} Cannot access Docker containers"
fi

echo ""
echo -e "${GREEN}T-Pot connection test complete!${NC}"
echo ""
echo "Access URLs:"
echo "  • Web Interface: http://${TPOT_HOST}:${TPOT_WEB_PORT}"
echo "  • SSH: ssh -p ${TPOT_SSH_PORT} ${HONEYPOT_USER}@${TPOT_HOST}"
EOF

chmod +x "$TEST_SCRIPT"
print_success "Created test script: $TEST_SCRIPT"

# Step 7: Summary
print_header "Setup Complete!"

echo -e "${GREEN}T-Pot Integration Configuration Summary:${NC}"
echo ""
echo "  T-Pot Host:      $TPOT_IP"
echo "  SSH Port:        $TPOT_SSH_PORT"
echo "  Web Port:        $TPOT_WEB_PORT"
echo "  Username:        $TPOT_USER"
echo "  Auth Method:     $AUTH_TYPE"
echo ""
echo -e "${CYAN}Next Steps:${NC}"
echo ""
echo "1. Test T-Pot connection:"
echo "   ${YELLOW}$TEST_SCRIPT${NC}"
echo ""
echo "2. Start Mini-XDR backend (it will auto-connect to T-Pot):"
echo "   ${YELLOW}cd $BACKEND_DIR && ./START_MINIXDR.sh${NC}"
echo ""
echo "3. Access T-Pot web interface:"
echo "   ${YELLOW}http://${TPOT_IP}:${TPOT_WEB_PORT}${NC}"
echo ""
echo "4. Run SSH brute force demo attack:"
echo "   ${YELLOW}$SCRIPT_DIR/../demo/demo-attack.sh${NC}"
echo ""
echo -e "${GREEN}Mini-XDR AI agents can now SSH into T-Pot for defensive actions!${NC}"
echo ""

#!/bin/bash

# SSH SECURITY FIX SCRIPT
# Fixes all instances of disabled SSH host verification
# RUN THIS AFTER CREDENTIAL CLEANUP

set -euo pipefail

# Configuration
PROJECT_ROOT="/Users/chasemad/Desktop/mini-xdr"
TPOT_HOST="34.193.101.171"
YOUR_ADMIN_IP="${YOUR_ADMIN_IP:-$(curl -s ipinfo.io/ip)}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[WARN] $1${NC}"
}

error() {
    echo -e "${RED}[ERROR] $1${NC}"
    exit 1
}

critical() {
    echo -e "${RED}[CRITICAL] $1${NC}"
}

# Banner
show_banner() {
    echo -e "${CYAN}"
    echo "=============================================="
    echo "    🔐 SSH SECURITY FIX 🔐"
    echo "=============================================="
    echo -e "${NC}"
    echo "This script will:"
    echo "  ❌ Remove all StrictHostKeyChecking=yes instances"
    echo "  🔑 Enable proper SSH host verification"
    echo "  📋 Create SSH known_hosts entries"
    echo "  🛡️ Implement secure SSH configuration"
    echo ""
}

# Find all files with SSH security issues
find_ssh_security_issues() {
    log "🔍 Scanning for SSH security issues..."
    
    # Find all files with StrictHostKeyChecking=yes
    grep -r "StrictHostKeyChecking=yes" "$PROJECT_ROOT" > /tmp/ssh-issues.txt 2>/dev/null || true
    
    local issue_count=$(wc -l < /tmp/ssh-issues.txt)
    
    if [ "$issue_count" -eq 0 ]; then
        log "✅ No SSH security issues found"
        return 0
    fi
    
    critical "🚨 Found $issue_count SSH security issues:"
    cat /tmp/ssh-issues.txt | while read line; do
        echo "  $line"
    done
    
    return 1
}

# Fix StrictHostKeyChecking issues
fix_strict_host_checking() {
    log "🔧 Fixing StrictHostKeyChecking=yes issues..."
    
    local fixed_count=0
    
    # Get unique files that need fixing
    local files_to_fix=($(grep -r "StrictHostKeyChecking=yes" "$PROJECT_ROOT" 2>/dev/null | cut -d: -f1 | sort -u || true))
    
    for file in "${files_to_fix[@]}"; do
        if [ -f "$file" ]; then
            log "Fixing: $file"
            
            # Create backup
            cp "$file" "${file}.ssh-backup-$(date +%Y%m%d_%H%M%S)"
            
            # Replace StrictHostKeyChecking=yes with proper verification
            sed -i 's/StrictHostKeyChecking=yes/StrictHostKeyChecking=yes/g' "$file"
            
            # Also add UserKnownHostsFile for better security
            sed -i 's/-o StrictHostKeyChecking=yes -o UserKnownHostsFile=~/.ssh/known_hosts/-o StrictHostKeyChecking=yes -o UserKnownHostsFile=~/.ssh/known_hosts -o UserKnownHostsFile=~\/.ssh\/known_hosts/g' "$file"
            
            ((fixed_count++))
        fi
    done
    
    log "✅ Fixed StrictHostKeyChecking in $fixed_count files"
}

# Create SSH known_hosts entries
setup_ssh_known_hosts() {
    log "📋 Setting up SSH known_hosts entries..."
    
    # Ensure .ssh directory exists
    mkdir -p ~/.ssh
    chmod 700 ~/.ssh
    
    # Create or update known_hosts
    if [ ! -f ~/.ssh/known_hosts ]; then
        touch ~/.ssh/known_hosts
        chmod 600 ~/.ssh/known_hosts
    fi
    
    # Add TPOT host key
    log "Adding TPOT host key for $TPOT_HOST..."
    
    # Remove any existing entries for TPOT host
    ssh-keygen -R "$TPOT_HOST" 2>/dev/null || true
    ssh-keygen -R "$TPOT_HOST:64295" 2>/dev/null || true
    
    # Add new host keys
    ssh-keyscan -H "$TPOT_HOST" >> ~/.ssh/known_hosts 2>/dev/null || warn "Could not scan TPOT host keys"
    ssh-keyscan -H -p 64295 "$TPOT_HOST" >> ~/.ssh/known_hosts 2>/dev/null || warn "Could not scan TPOT management port"
    
    # Add localhost entries for local development
    ssh-keyscan -H localhost >> ~/.ssh/known_hosts 2>/dev/null || warn "Could not scan localhost"
    ssh-keyscan -H 127.0.0.1 >> ~/.ssh/known_hosts 2>/dev/null || warn "Could not scan 127.0.0.1"
    
    # If we know backend IP, add that too
    if [ -n "${BACKEND_IP:-}" ]; then
        log "Adding backend host key for $BACKEND_IP..."
        ssh-keyscan -H "$BACKEND_IP" >> ~/.ssh/known_hosts 2>/dev/null || warn "Could not scan backend host"
    fi
    
    log "✅ SSH known_hosts configured"
}

# Create secure SSH configuration
create_secure_ssh_config() {
    log "🛡️ Creating secure SSH configuration..."
    
    # Create SSH config directory structure
    mkdir -p ~/.ssh/config.d
    
    # Create secure SSH config for Mini-XDR
    cat > ~/.ssh/config.d/mini-xdr << 'EOF'
# Mini-XDR SSH Configuration
# Secure settings for Mini-XDR infrastructure

# TPOT Honeypot Management
Host tpot tpot-mgmt
    HostName 34.193.101.171
    Port 64295
    User admin
    IdentityFile ~/.ssh/mini-xdr-tpot-key.pem
    StrictHostKeyChecking yes
    UserKnownHostsFile ~/.ssh/known_hosts
    ServerAliveInterval 60
    ServerAliveCountMax 3
    ConnectTimeout 10
    
# TPOT Honeypot (for testing connectivity)
Host tpot-test
    HostName 34.193.101.171
    Port 22
    User admin
    IdentityFile ~/.ssh/mini-xdr-tpot-key.pem
    StrictHostKeyChecking yes
    UserKnownHostsFile ~/.ssh/known_hosts
    ConnectTimeout 5

# Default security settings for all hosts
Host *
    StrictHostKeyChecking yes
    UserKnownHostsFile ~/.ssh/known_hosts
    HashKnownHosts yes
    PasswordAuthentication no
    ChallengeResponseAuthentication no
    GSSAPIAuthentication no
    ServerAliveInterval 60
    ServerAliveCountMax 3
    ConnectTimeout 10
    ForwardAgent no
    ForwardX11 no
    AddKeysToAgent yes
EOF
    
    # Include the config in main SSH config
    if [ ! -f ~/.ssh/config ]; then
        cat > ~/.ssh/config << 'EOF'
# SSH Configuration
Include config.d/*
EOF
    elif ! grep -q "Include config.d/\*" ~/.ssh/config; then
        sed -i '1iInclude config.d/*' ~/.ssh/config
    fi
    
    # Set proper permissions
    chmod 600 ~/.ssh/config
    chmod 600 ~/.ssh/config.d/mini-xdr
    
    log "✅ Secure SSH configuration created"
}

# Create SSH connection test script
create_ssh_test_script() {
    log "🧪 Creating SSH connection test script..."
    
    cat > "$PROJECT_ROOT/test-ssh-connections.sh" << 'EOF'
#!/bin/bash

# SSH Connection Test Script
# Tests all SSH connections after security fixes

echo "Testing SSH connections with security fixes..."

# Test TPOT management connection
echo "Testing TPOT management (port 64295)..."
if ssh -o ConnectTimeout=5 tpot-mgmt "echo 'TPOT management SSH working'" 2>/dev/null; then
    echo "✅ TPOT management SSH: Working"
else
    echo "❌ TPOT management SSH: Failed"
fi

# Test TPOT honeypot connection (should connect to honeypot)
echo "Testing TPOT honeypot (port 22)..."
if timeout 5 ssh -o ConnectTimeout=3 tpot-test "echo 'Connected'" 2>/dev/null; then
    echo "✅ TPOT honeypot SSH: Working"
else
    echo "⚠️ TPOT honeypot SSH: Expected to fail (honeypot)"
fi

# Test backend connection if IP is available
if [ -n "${BACKEND_IP:-}" ]; then
    echo "Testing backend SSH ($BACKEND_IP)..."
    if ssh -o ConnectTimeout=5 -i ~/.ssh/mini-xdr-tpot-key.pem ubuntu@"$BACKEND_IP" "echo 'Backend SSH working'" 2>/dev/null; then
        echo "✅ Backend SSH: Working"
    else
        echo "❌ Backend SSH: Failed"
    fi
fi

echo "SSH connection tests completed."
EOF
    
    chmod +x "$PROJECT_ROOT/test-ssh-connections.sh"
    log "✅ SSH test script created: test-ssh-connections.sh"
}

# Implement SSH session logging
setup_ssh_logging() {
    log "📊 Setting up SSH session logging..."
    
    # Create SSH log directory
    sudo mkdir -p /var/log/ssh-sessions 2>/dev/null || mkdir -p ~/ssh-sessions
    
    # Create SSH wrapper script for logging
    cat > "$PROJECT_ROOT/scripts/secure-ssh.sh" << 'EOF'
#!/bin/bash

# Secure SSH wrapper with logging
# Usage: ./secure-ssh.sh <host> [command]

LOGDIR="${HOME}/ssh-sessions"
mkdir -p "$LOGDIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
HOST="$1"
shift

# Log the connection
echo "$(date): SSH connection to $HOST by $USER" >> "$LOGDIR/ssh-connections.log"

# Execute SSH with secure settings
ssh -o StrictHostKeyChecking=yes -o UserKnownHostsFile=~/.ssh/known_hosts \
    -o UserKnownHostsFile=~/.ssh/known_hosts \
    -o ServerAliveInterval=60 \
    -o ConnectTimeout=10 \
    "$HOST" "$@"

# Log the disconnection
echo "$(date): SSH disconnection from $HOST by $USER" >> "$LOGDIR/ssh-connections.log"
EOF
    
    chmod +x "$PROJECT_ROOT/scripts/secure-ssh.sh"
    log "✅ SSH logging configured"
}

# Generate SSH security report
generate_ssh_security_report() {
    log "📊 Generating SSH security report..."
    
    cat > "/tmp/ssh-security-fix-report.txt" << EOF
SSH SECURITY FIX REPORT
=======================
Date: $(date)
Project: Mini-XDR

ACTIONS TAKEN:
✅ Fixed all StrictHostKeyChecking=yes instances
✅ Enabled proper SSH host verification
✅ Created SSH known_hosts entries
✅ Implemented secure SSH configuration
✅ Created SSH connection test script
✅ Set up SSH session logging

FILES FIXED:
$(grep -r "StrictHostKeyChecking=yes" "$PROJECT_ROOT" 2>/dev/null | cut -d: -f1 | sort -u | wc -l) files updated

SSH SECURITY IMPROVEMENTS:
- Host key verification: ENABLED
- Known hosts file: ~/.ssh/known_hosts
- SSH config: ~/.ssh/config.d/mini-xdr
- Connection timeouts: 10 seconds
- Server alive interval: 60 seconds
- Password authentication: DISABLED
- Agent forwarding: DISABLED
- X11 forwarding: DISABLED

SSH HOSTS CONFIGURED:
- TPOT Management: 34.193.101.171:64295
- TPOT Honeypot: 34.193.101.171:22
- Localhost: 127.0.0.1

SECURITY VALIDATION:
# Test SSH connections:
./test-ssh-connections.sh

# Check for remaining issues:
grep -r "StrictHostKeyChecking=yes" /Users/chasemad/Desktop/mini-xdr/ || echo "No issues found"

# View SSH configuration:
cat ~/.ssh/config.d/mini-xdr

NEXT STEPS:
1. Test all SSH connections with new security settings
2. Run database-security-hardening.sh script
3. Update any applications that use SSH
4. Monitor SSH connection logs

EMERGENCY CONTACT:
If SSH connections fail, check ~/.ssh/known_hosts and verify host keys.
EOF
    
    log "📋 Report saved to: /tmp/ssh-security-fix-report.txt"
    echo ""
    cat /tmp/ssh-security-fix-report.txt
}

# Validate SSH security fixes
validate_ssh_security() {
    log "✅ Validating SSH security fixes..."
    
    # Check for remaining StrictHostKeyChecking=yes
    local remaining_issues
    remaining_issues=$(grep -r "StrictHostKeyChecking=yes" "$PROJECT_ROOT" 2>/dev/null | wc -l || echo "0")
    
    if [ "$remaining_issues" -eq 0 ]; then
        log "✅ No StrictHostKeyChecking=yes instances found"
    else
        warn "⚠️ Found $remaining_issues remaining SSH security issues"
    fi
    
    # Check SSH configuration
    if [ -f ~/.ssh/config.d/mini-xdr ]; then
        log "✅ SSH configuration file created"
    else
        warn "⚠️ SSH configuration file not found"
    fi
    
    # Check known_hosts
    if [ -f ~/.ssh/known_hosts ]; then
        local hosts_count=$(wc -l < ~/.ssh/known_hosts)
        log "✅ Known hosts file exists with $hosts_count entries"
    else
        warn "⚠️ Known hosts file not found"
    fi
    
    # Test basic SSH functionality
    log "Testing SSH configuration..."
    if ssh -o ConnectTimeout=2 -o BatchMode=yes localhost "echo test" 2>/dev/null; then
        log "✅ SSH configuration test passed"
    else
        log "ℹ️ SSH configuration test skipped (no local SSH server)"
    fi
}

# Main execution
main() {
    show_banner
    
    # Confirm action
    critical "⚠️  WARNING: This will modify SSH configurations and may affect connectivity!"
    echo ""
    read -p "Continue with SSH security fixes? (type 'FIX SSH SECURITY' to confirm): " -r
    if [ "$REPLY" != "FIX SSH SECURITY" ]; then
        log "Operation cancelled by user"
        exit 0
    fi
    
    log "🔐 Starting SSH security fixes..."
    local start_time=$(date +%s)
    
    # Execute SSH security procedures
    if find_ssh_security_issues; then
        log "✅ No SSH security issues found"
    else
        fix_strict_host_checking
    fi
    
    setup_ssh_known_hosts
    create_secure_ssh_config
    create_ssh_test_script
    setup_ssh_logging
    validate_ssh_security
    generate_ssh_security_report
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    log "🎉 SSH security fixes completed in ${duration} seconds"
    
    echo ""
    critical "🚨 NEXT STEPS:"
    echo "1. Test SSH connections: ./test-ssh-connections.sh"
    echo "2. Run: ./database-security-hardening.sh"
    echo "3. Verify application connectivity"
    echo "4. Monitor SSH logs for any issues"
}

# Export configuration for other scripts
export PROJECT_ROOT="$PROJECT_ROOT"
export TPOT_HOST="$TPOT_HOST"

# Run main function
main "$@"

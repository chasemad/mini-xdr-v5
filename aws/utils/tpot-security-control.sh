#!/bin/bash

# TPOT Security Mode Control Script
# Switch between testing (secure) and live (open to attackers) modes

set -euo pipefail

# Configuration
TPOT_HOST="34.193.101.171"
TPOT_SSH_PORT="64295"
TPOT_USER="admin"
KEY_NAME="${KEY_NAME:-mini-xdr-tpot-key}"
YOUR_IP="${YOUR_IP:-24.11.0.176}"
REGION="${AWS_REGION:-us-east-1}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# Banner
show_banner() {
    echo -e "${CYAN}"
    echo "=================================="
    echo "      TPOT Security Control"
    echo "=================================="
    echo -e "${NC}"
}

# Logging function
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

step() {
    echo -e "${BLUE}$1${NC}"
}

critical() {
    echo -e "${MAGENTA}[CRITICAL] $1${NC}"
}

# Show usage
show_usage() {
    echo "Usage: $0 {testing|live|status|lockdown}"
    echo ""
    echo "Security Modes:"
    echo "  testing   - Secure mode: Only accessible from your IP ($YOUR_IP)"
    echo "  live      - Live mode: Open to internet attackers (USE WITH CAUTION)"
    echo "  status    - Show current security mode and configuration"
    echo "  lockdown  - Emergency lockdown: Block all external access"
    echo ""
    echo "Examples:"
    echo "  $0 testing    # Safe for development and testing"
    echo "  $0 live       # Production honeypot mode"
    echo "  $0 status     # Check current mode"
    echo "  $0 lockdown   # Emergency shutdown"
    echo ""
    echo -e "${YELLOW}⚠️  WARNING: 'live' mode exposes honeypot to real attackers!${NC}"
}

# Get TPOT instance information
get_tpot_info() {
    log "Getting TPOT instance information..."
    
    # Find TPOT instance by IP
    TPOT_INSTANCE_ID=$(aws ec2 describe-instances \
        --filters "Name=ip-address,Values=$TPOT_HOST" \
        --query 'Reservations[0].Instances[0].InstanceId' \
        --output text \
        --region "$REGION" 2>/dev/null || echo "")
    
    if [ -z "$TPOT_INSTANCE_ID" ] || [ "$TPOT_INSTANCE_ID" = "None" ]; then
        warn "Could not find TPOT instance by IP. Trying alternative methods..."
        
        # Try to find by tag or name
        TPOT_INSTANCE_ID=$(aws ec2 describe-instances \
            --filters "Name=tag:Name,Values=*tpot*" "Name=instance-state-name,Values=running" \
            --query 'Reservations[0].Instances[0].InstanceId' \
            --output text \
            --region "$REGION" 2>/dev/null || echo "")
    fi
    
    if [ -z "$TPOT_INSTANCE_ID" ] || [ "$TPOT_INSTANCE_ID" = "None" ]; then
        error "Could not find TPOT instance. Please check the IP address and region."
    fi
    
    # Get security group ID
    TPOT_SG_ID=$(aws ec2 describe-instances \
        --instance-ids "$TPOT_INSTANCE_ID" \
        --query 'Reservations[0].Instances[0].SecurityGroups[0].GroupId' \
        --output text \
        --region "$REGION")
    
    log "TPOT Instance ID: $TPOT_INSTANCE_ID"
    log "TPOT Security Group: $TPOT_SG_ID"
}

# Test TPOT connectivity
test_tpot_connectivity() {
    log "Testing TPOT connectivity..."
    
    if ssh -i "~/.ssh/${KEY_NAME}.pem" -p "$TPOT_SSH_PORT" -o ConnectTimeout=10 -o StrictHostKeyChecking=yes -o UserKnownHostsFile=~/.ssh/known_hosts \
           "$TPOT_USER@$TPOT_HOST" "echo 'TPOT SSH accessible'" >/dev/null 2>&1; then
        log "✅ TPOT SSH accessible"
    else
        error "❌ Cannot connect to TPOT via SSH. Check connectivity and credentials."
    fi
}

# Get current security group rules
get_current_rules() {
    aws ec2 describe-security-groups \
        --group-ids "$TPOT_SG_ID" \
        --query 'SecurityGroups[0].IpPermissions' \
        --output json \
        --region "$REGION"
}

# Show current security status
show_status() {
    step "📊 TPOT Security Status"
    
    get_tpot_info
    
    echo ""
    echo "TPOT Instance Information:"
    echo "  🖥️  Instance ID: $TPOT_INSTANCE_ID"
    echo "  🌐 IP Address: $TPOT_HOST"
    echo "  🔒 Security Group: $TPOT_SG_ID"
    echo "  🔑 Management Port: $TPOT_SSH_PORT"
    echo ""
    
    # Check current rules
    local rules
    rules=$(get_current_rules)
    
    echo "Current Security Rules:"
    
    # Check for 0.0.0.0/0 rules (open to internet)
    local open_rules
    open_rules=$(echo "$rules" | jq -r '.[] | select(.IpRanges[]?.CidrIp == "0.0.0.0/0") | .FromPort' 2>/dev/null || echo "")
    
    # Check for restricted rules (your IP only)
    local restricted_rules
    restricted_rules=$(echo "$rules" | jq -r ".[] | select(.IpRanges[]?.CidrIp == \"$YOUR_IP/32\") | .FromPort" 2>/dev/null || echo "")
    
    if [ -n "$open_rules" ]; then
        critical "🚨 LIVE MODE: Honeypot is OPEN to internet attackers!"
        echo "  Open ports: $(echo "$open_rules" | tr '\n' ',' | sed 's/,$//')"
    elif [ -n "$restricted_rules" ]; then
        log "🔒 TESTING MODE: Honeypot is restricted to your IP ($YOUR_IP)"
        echo "  Restricted ports: $(echo "$restricted_rules" | tr '\n' ',' | sed 's/,$//')"
    else
        warn "⚠️  UNKNOWN MODE: Custom security configuration detected"
    fi
    
    echo ""
    
    # Check TPOT service status
    log "Checking TPOT service status..."
    ssh -i "~/.ssh/${KEY_NAME}.pem" -p "$TPOT_SSH_PORT" -o ConnectTimeout=10 -o StrictHostKeyChecking=yes -o UserKnownHostsFile=~/.ssh/known_hosts \
        "$TPOT_USER@$TPOT_HOST" << 'REMOTE_STATUS'
        echo "TPOT Container Status:"
        if command -v docker >/dev/null 2>&1; then
            RUNNING_CONTAINERS=$(sudo docker ps --format "table {{.Names}}\t{{.Status}}" | grep -E "(cowrie|dionaea|suricata|honeytrap)" | wc -l)
            echo "  🐳 Running honeypot containers: $RUNNING_CONTAINERS"
            
            if [ "$RUNNING_CONTAINERS" -gt 0 ]; then
                echo "  ✅ TPOT services are active"
            else
                echo "  ❌ TPOT services may not be running"
            fi
        else
            echo "  ⚠️  Docker not found or not accessible"
        fi
REMOTE_STATUS
    
    echo ""
}

# Set testing mode (secure, your IP only)
set_testing_mode() {
    step "🔒 Setting TPOT to TESTING MODE (Secure)"
    
    get_tpot_info
    
    critical "This will restrict TPOT access to your IP only: $YOUR_IP"
    read -p "Continue? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log "Operation cancelled"
        return
    fi
    
    log "Configuring security group for testing mode..."
    
    # Remove all existing rules (except management SSH)
    log "Removing existing honeypot rules..."
    local current_rules
    current_rules=$(get_current_rules)
    
    # Remove rules for common honeypot ports (but keep management SSH)
    for port in 22 80 443 21 23 25 53 135 139 445 993 995 1433 3306 3389 5432 8080 8443; do
        aws ec2 revoke-security-group-ingress \
            --group-id "$TPOT_SG_ID" \
            --protocol tcp \
            --port "$port" \
            --cidr 0.0.0.0/0 \
            --region "$REGION" 2>/dev/null || true
    done
    
    # Add restricted rules for your IP only
    log "Adding restricted rules for your IP ($YOUR_IP)..."
    
    # Honeypot services - restricted to your IP
    local honeypot_ports=(22 80 443 21 23 25 53 135 139 445 993 995 1433 3306 3389 5432 8080 8443)
    for port in "${honeypot_ports[@]}"; do
        aws ec2 authorize-security-group-ingress \
            --group-id "$TPOT_SG_ID" \
            --protocol tcp \
            --port "$port" \
            --cidr "$YOUR_IP/32" \
            --region "$REGION" 2>/dev/null || true
    done
    
    # UDP ports for DNS, etc.
    aws ec2 authorize-security-group-ingress \
        --group-id "$TPOT_SG_ID" \
        --protocol udp \
        --port 53 \
        --cidr "$YOUR_IP/32" \
        --region "$REGION" 2>/dev/null || true
    
    log "✅ TPOT configured for TESTING MODE"
    log "🔒 Only accessible from your IP: $YOUR_IP"
    log "🧪 Safe for development and testing"
    
    # Restart TPOT services to apply changes
    restart_tpot_services
}

# Set live mode (open to attackers)
set_live_mode() {
    step "🚨 Setting TPOT to LIVE MODE (Open to Attackers)"
    
    get_tpot_info
    
    critical "⚠️  WARNING: This will expose TPOT to REAL ATTACKERS on the internet!"
    critical "⚠️  Only do this when you're ready for production honeypot operation!"
    echo ""
    echo "This will:"
    echo "  - Open honeypot ports to 0.0.0.0/0 (entire internet)"
    echo "  - Attract real cybercriminals and malware"
    echo "  - Generate real attack data"
    echo ""
    critical "Are you absolutely sure you want to proceed?"
    read -p "Type 'I UNDERSTAND THE RISKS' to continue: " -r
    if [ "$REPLY" != "I UNDERSTAND THE RISKS" ]; then
        log "Operation cancelled - wise choice for safety!"
        return
    fi
    
    log "Configuring security group for live mode..."
    
    # Add open rules for honeypot services
    log "Opening honeypot ports to internet..."
    
    # Common attack target ports
    local honeypot_ports=(22 80 443 21 23 25 53 135 139 445 993 995 1433 3306 3389 5432 8080 8443)
    for port in "${honeypot_ports[@]}"; do
        aws ec2 authorize-security-group-ingress \
            --group-id "$TPOT_SG_ID" \
            --protocol tcp \
            --port "$port" \
            --cidr 0.0.0.0/0 \
            --region "$REGION" 2>/dev/null || true
    done
    
    # UDP ports
    aws ec2 authorize-security-group-ingress \
        --group-id "$TPOT_SG_ID" \
        --protocol udp \
        --port 53 \
        --cidr 0.0.0.0/0 \
        --region "$REGION" 2>/dev/null || true
    
    critical "🚨 TPOT is now LIVE and exposed to internet attackers!"
    critical "🚨 Monitor closely and be prepared for immediate attack activity!"
    log "📊 Real attack data will now flow to your Mini-XDR system"
    
    # Restart TPOT services
    restart_tpot_services
    
    # Show monitoring information
    echo ""
    log "📈 Monitor attack activity:"
    log "   - TPOT Logs: ssh -i ~/.ssh/${KEY_NAME}.pem -p $TPOT_SSH_PORT $TPOT_USER@$TPOT_HOST"
    log "   - Mini-XDR: Use your AWS services control script"
    log "   - Emergency lockdown: $0 lockdown"
}

# Emergency lockdown
emergency_lockdown() {
    step "🚨 Emergency Lockdown Mode"
    
    get_tpot_info
    
    critical "This will immediately block ALL external access to TPOT!"
    log "Removing all internet-facing security group rules..."
    
    # Remove all 0.0.0.0/0 rules
    local current_rules
    current_rules=$(get_current_rules)
    
    # Remove open rules for all ports
    for port in 22 80 443 21 23 25 53 135 139 445 993 995 1433 3306 3389 5432 8080 8443; do
        aws ec2 revoke-security-group-ingress \
            --group-id "$TPOT_SG_ID" \
            --protocol tcp \
            --port "$port" \
            --cidr 0.0.0.0/0 \
            --region "$REGION" 2>/dev/null || true
    done
    
    aws ec2 revoke-security-group-ingress \
        --group-id "$TPOT_SG_ID" \
        --protocol udp \
        --port 53 \
        --cidr 0.0.0.0/0 \
        --region "$REGION" 2>/dev/null || true
    
    critical "🔒 EMERGENCY LOCKDOWN COMPLETE"
    log "✅ TPOT is now isolated from internet attacks"
    log "🔧 Management access still available from your IP"
    
    # Keep management access for your IP
    aws ec2 authorize-security-group-ingress \
        --group-id "$TPOT_SG_ID" \
        --protocol tcp \
        --port "$TPOT_SSH_PORT" \
        --cidr "$YOUR_IP/32" \
        --region "$REGION" 2>/dev/null || true
}

# Restart TPOT services
restart_tpot_services() {
    log "Restarting TPOT services to apply configuration changes..."
    
    ssh -i "~/.ssh/${KEY_NAME}.pem" -p "$TPOT_SSH_PORT" -o ConnectTimeout=30 -o StrictHostKeyChecking=yes -o UserKnownHostsFile=~/.ssh/known_hosts \
        "$TPOT_USER@$TPOT_HOST" << 'REMOTE_RESTART'
        echo "Restarting TPOT Docker containers..."
        
        # Restart main TPOT services
        if command -v docker >/dev/null 2>&1; then
            # Get container names
            CONTAINERS=$(sudo docker ps --format "{{.Names}}" | grep -E "(cowrie|dionaea|suricata|honeytrap|fluent)" || true)
            
            if [ -n "$CONTAINERS" ]; then
                echo "Restarting containers: $CONTAINERS"
                echo "$CONTAINERS" | xargs -r sudo docker restart
                sleep 10
                
                # Check status
                RUNNING=$(sudo docker ps --format "{{.Names}}" | grep -E "(cowrie|dionaea|suricata|honeytrap)" | wc -l)
                echo "Running honeypot containers after restart: $RUNNING"
            else
                echo "No TPOT containers found to restart"
            fi
        else
            echo "Docker not accessible"
        fi
        
        echo "TPOT service restart completed"
REMOTE_RESTART
    
    log "✅ TPOT services restarted"
}

# Main function
main() {
    show_banner
    
    case "${1:-}" in
        testing)
            set_testing_mode
            echo ""
            show_status
            ;;
        live)
            set_live_mode
            echo ""
            show_status
            ;;
        status)
            show_status
            ;;
        lockdown)
            emergency_lockdown
            echo ""
            show_status
            ;;
        *)
            show_usage
            exit 1
            ;;
    esac
}

# Export configuration
export AWS_REGION="$REGION"
export KEY_NAME="$KEY_NAME"
export YOUR_IP="$YOUR_IP"

# Run main function
main "$@"

#!/bin/bash
# Manage Kali machine access to T-Pot honeypot for secure testing

set -e

REGION="us-east-1"
SG_ID="${AWS_SECURITY_GROUP_ID:-YOUR_SECURITY_GROUP_ID_HERE}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() { echo -e "${BLUE}[$(date +'%H:%M:%S')]${NC} $1"; }
success() { echo -e "${GREEN}✅ $1${NC}"; }
warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
error() { echo -e "${RED}❌ $1${NC}"; }

usage() {
    echo "Usage: $0 [add|remove|status] [kali-ip] [ports...]"
    echo ""
    echo "Examples:"
    echo "  $0 add 203.0.113.10 22 80 443    # Allow Kali IP to access SSH, HTTP, HTTPS"
    echo "  $0 remove 203.0.113.10 22 80 443 # Remove Kali IP access to those ports"
    echo "  $0 status                         # Show current access rules"
    echo ""
    echo "Common honeypot ports for testing:"
    echo "  22 (SSH), 23 (Telnet), 25 (SMTP), 80 (HTTP), 443 (HTTPS)"
    echo "  3306 (MySQL), 3389 (RDP), 5432 (PostgreSQL), 6379 (Redis)"
    echo ""
    exit 1
}

show_status() {
    log "Current T-Pot security group rules:"
    aws ec2 describe-security-groups \
        --group-ids $SG_ID \
        --query 'SecurityGroups[0].IpPermissions[*].[IpProtocol,FromPort,ToPort,IpRanges[0].CidrIp]' \
        --output table \
        --region $REGION
}

add_access() {
    local kali_ip="$1"
    shift
    local ports=("$@")
    
    if [ -z "$kali_ip" ] || [ ${#ports[@]} -eq 0 ]; then
        error "Must specify Kali IP and at least one port"
        usage
    fi
    
    log "Adding Kali machine access: $kali_ip"
    
    for port in "${ports[@]}"; do
        log "Allowing access to TCP port $port from $kali_ip..."
        aws ec2 authorize-security-group-ingress \
            --group-id $SG_ID \
            --protocol tcp \
            --port $port \
            --cidr "$kali_ip/32" \
            --region $REGION 2>/dev/null && success "Port $port accessible" || warning "Port $port rule may already exist"
    done
    
    success "Kali machine $kali_ip can now access ports: ${ports[*]}"
}

remove_access() {
    local kali_ip="$1"
    shift
    local ports=("$@")
    
    if [ -z "$kali_ip" ] || [ ${#ports[@]} -eq 0 ]; then
        error "Must specify Kali IP and at least one port"
        usage
    fi
    
    log "Removing Kali machine access: $kali_ip"
    
    for port in "${ports[@]}"; do
        log "Removing access to TCP port $port from $kali_ip..."
        aws ec2 revoke-security-group-ingress \
            --group-id $SG_ID \
            --protocol tcp \
            --port $port \
            --cidr "$kali_ip/32" \
            --region $REGION 2>/dev/null && success "Port $port blocked" || warning "Port $port rule may not exist"
    done
    
    success "Kali machine $kali_ip access removed from ports: ${ports[*]}"
}

# Main logic
case "${1:-status}" in
    add)
        add_access "${@:2}"
        echo ""
        show_status
        ;;
    remove)
        remove_access "${@:2}"
        echo ""
        show_status
        ;;
    status)
        show_status
        ;;
    *)
        usage
        ;;
esac

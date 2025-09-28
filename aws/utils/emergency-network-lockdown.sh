#!/bin/bash

# EMERGENCY NETWORK LOCKDOWN SCRIPT
# Removes all 0.0.0.0/0 exposures immediately
# RUN THIS IMMEDIATELY TO SECURE THE SYSTEM

set -euo pipefail

# Configuration
REGION="${AWS_REGION:-us-east-1}"
YOUR_ADMIN_IP="${YOUR_ADMIN_IP:-$(curl -s ipinfo.io/ip)}"
TPOT_IP="34.193.101.171"

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
    echo "    🚨 EMERGENCY NETWORK LOCKDOWN 🚨"
    echo "=============================================="
    echo -e "${NC}"
    echo "This script will:"
    echo "  ❌ Remove ALL 0.0.0.0/0 security group rules"
    echo "  🔒 Lock down TPOT to testing mode"
    echo "  ✅ Add secure admin access for your IP: $YOUR_ADMIN_IP"
    echo "  📊 Generate lockdown report"
    echo ""
}

# Get all security groups with 0.0.0.0/0 rules
find_exposed_security_groups() {
    log "🔍 Scanning for security groups with 0.0.0.0/0 exposures..."
    
    # Get all security groups
    aws ec2 describe-security-groups \
        --region "$REGION" \
        --query 'SecurityGroups[*].{GroupId:GroupId,GroupName:GroupName,IpPermissions:IpPermissions}' \
        --output json > /tmp/all-security-groups.json
    
    # Find groups with 0.0.0.0/0 rules
    EXPOSED_GROUPS=$(jq -r '.[] | select(.IpPermissions[]?.IpRanges[]?.CidrIp == "0.0.0.0/0") | .GroupId' /tmp/all-security-groups.json | sort -u)
    
    if [ -z "$EXPOSED_GROUPS" ]; then
        log "✅ No security groups found with 0.0.0.0/0 exposures"
        return 0
    fi
    
    critical "🚨 Found exposed security groups:"
    echo "$EXPOSED_GROUPS" | while read sg_id; do
        sg_name=$(jq -r ".[] | select(.GroupId == \"$sg_id\") | .GroupName" /tmp/all-security-groups.json)
        echo "  - $sg_id ($sg_name)"
    done
    
    return 1
}

# Remove dangerous 0.0.0.0/0 rules
remove_open_access_rules() {
    log "🔒 Removing dangerous 0.0.0.0/0 security group rules..."
    
    local fixed_count=0
    
    echo "$EXPOSED_GROUPS" | while read sg_id; do
        if [ -z "$sg_id" ]; then continue; fi
        
        log "Processing security group: $sg_id"
        
        # Get current rules for this security group
        aws ec2 describe-security-groups \
            --group-ids "$sg_id" \
            --region "$REGION" \
            --query 'SecurityGroups[0].IpPermissions' \
            --output json > /tmp/sg-rules-${sg_id}.json
        
        # Find and remove each 0.0.0.0/0 rule
        jq -c '.[] | select(.IpRanges[]?.CidrIp == "0.0.0.0/0")' /tmp/sg-rules-${sg_id}.json | while read rule; do
            protocol=$(echo "$rule" | jq -r '.IpProtocol')
            from_port=$(echo "$rule" | jq -r '.FromPort // empty')
            to_port=$(echo "$rule" | jq -r '.ToPort // empty')
            
            if [ "$protocol" = "tcp" ] || [ "$protocol" = "udp" ]; then
                if [ -n "$from_port" ] && [ -n "$to_port" ]; then
                    log "  Removing $protocol rule: ports $from_port-$to_port"
                    aws ec2 revoke-security-group-ingress \
                        --group-id "$sg_id" \
                        --protocol "$protocol" \
                        --port "$from_port-$to_port" \
                        --cidr 0.0.0.0/0 \
                        --region "$REGION" 2>/dev/null || warn "Failed to remove rule"
                    ((fixed_count++))
                fi
            elif [ "$protocol" = "icmp" ]; then
                log "  Removing ICMP rule"
                aws ec2 revoke-security-group-ingress \
                    --group-id "$sg_id" \
                    --protocol icmp \
                    --port -1 \
                    --cidr 0.0.0.0/0 \
                    --region "$REGION" 2>/dev/null || warn "Failed to remove ICMP rule"
                ((fixed_count++))
            fi
        done
    done
    
    log "✅ Removed $fixed_count dangerous security group rules"
}

# Add secure admin access
add_secure_admin_access() {
    log "🔐 Adding secure admin access for IP: $YOUR_ADMIN_IP"
    
    # Find security groups that need admin access
    local backend_sg_id=""
    local tpot_sg_id=""
    
    # Try to find backend security group
    backend_sg_id=$(aws ec2 describe-security-groups \
        --filters "Name=group-name,Values=*backend*,*mini-xdr*" \
        --region "$REGION" \
        --query 'SecurityGroups[0].GroupId' \
        --output text 2>/dev/null || echo "")
    
    if [ -n "$backend_sg_id" ] && [ "$backend_sg_id" != "None" ]; then
        log "Adding admin SSH access to backend security group: $backend_sg_id"
        aws ec2 authorize-security-group-ingress \
            --group-id "$backend_sg_id" \
            --protocol tcp \
            --port 22 \
            --cidr "$YOUR_ADMIN_IP/32" \
            --region "$REGION" 2>/dev/null || warn "Admin SSH rule may already exist"
        
        log "Adding admin API access to backend security group: $backend_sg_id"
        aws ec2 authorize-security-group-ingress \
            --group-id "$backend_sg_id" \
            --protocol tcp \
            --port 8000 \
            --cidr "$YOUR_ADMIN_IP/32" \
            --region "$REGION" 2>/dev/null || warn "Admin API rule may already exist"
    fi
}

# Lock TPOT to testing mode
lock_tpot_testing_mode() {
    log "🍯 Locking TPOT honeypot to testing mode..."
    
    if [ -f "./tpot-security-control.sh" ]; then
        ./tpot-security-control.sh testing
    elif [ -f "../utils/tpot-security-control.sh" ]; then
        ../utils/tpot-security-control.sh testing
    elif [ -f "/Users/chasemad/Desktop/mini-xdr/aws/utils/tpot-security-control.sh" ]; then
        /Users/chasemad/Desktop/mini-xdr/aws/utils/tpot-security-control.sh testing
    else
        warn "TPOT security control script not found - manual lockdown required"
        warn "SSH to TPOT and manually restrict security groups"
    fi
}

# Generate lockdown report
generate_lockdown_report() {
    log "📊 Generating security lockdown report..."
    
    cat > "/tmp/emergency-lockdown-report.txt" << EOF
EMERGENCY NETWORK LOCKDOWN REPORT
=================================
Date: $(date)
Region: $REGION
Admin IP: $YOUR_ADMIN_IP

ACTIONS TAKEN:
- Removed all 0.0.0.0/0 security group rules
- Locked TPOT honeypot to testing mode  
- Added secure admin access for $YOUR_ADMIN_IP
- Scanned $(echo "$EXPOSED_GROUPS" | wc -l) security groups

SECURITY GROUPS MODIFIED:
$(echo "$EXPOSED_GROUPS" | sed 's/^/- /')

NEXT STEPS:
1. Fix hardcoded credentials (run credential-cleanup.sh)
2. Fix SSH security issues (run ssh-security-fix.sh)  
3. Harden database security (run database-hardening.sh)
4. Review and test all applications for connectivity

VALIDATION COMMANDS:
# Verify no 0.0.0.0/0 rules remain:
aws ec2 describe-security-groups --query 'SecurityGroups[?IpPermissions[?IpRanges[?CidrIp==\`0.0.0.0/0\`]]]'

# Check TPOT status:
./tpot-security-control.sh status

EMERGENCY CONTACT:
If this lockdown breaks critical functionality, contact security team immediately.
EOF
    
    log "📋 Report saved to: /tmp/emergency-lockdown-report.txt"
    echo ""
    cat /tmp/emergency-lockdown-report.txt
}

# Validate lockdown success
validate_lockdown() {
    log "✅ Validating emergency lockdown..."
    
    # Check for remaining 0.0.0.0/0 rules
    local remaining_exposures
    remaining_exposures=$(aws ec2 describe-security-groups \
        --region "$REGION" \
        --query 'SecurityGroups[?IpPermissions[?IpRanges[?CidrIp==`0.0.0.0/0`]]]' \
        --output json | jq '. | length')
    
    if [ "$remaining_exposures" -eq 0 ]; then
        log "✅ SUCCESS: No 0.0.0.0/0 exposures remaining"
    else
        error "❌ FAILED: $remaining_exposures security groups still have 0.0.0.0/0 exposures"
    fi
    
    # Test admin connectivity
    log "Testing admin connectivity..."
    if curl -m 5 "http://$YOUR_ADMIN_IP" >/dev/null 2>&1; then
        log "✅ Network connectivity confirmed"
    else
        warn "⚠️ Network connectivity test failed (may be expected)"
    fi
}

# Main execution
main() {
    show_banner
    
    # Confirm action
    critical "⚠️  WARNING: This will immediately lock down ALL network access!"
    critical "⚠️  Only IP $YOUR_ADMIN_IP will have admin access after this operation."
    echo ""
    read -p "Continue with emergency lockdown? (type 'EMERGENCY LOCKDOWN' to confirm): " -r
    if [ "$REPLY" != "EMERGENCY LOCKDOWN" ]; then
        log "Operation cancelled by user"
        exit 0
    fi
    
    log "🚨 Starting emergency network lockdown..."
    local start_time=$(date +%s)
    
    # Execute lockdown procedures
    if find_exposed_security_groups; then
        log "✅ No immediate network exposures found"
    else
        remove_open_access_rules
    fi
    
    add_secure_admin_access
    lock_tpot_testing_mode
    validate_lockdown
    generate_lockdown_report
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    log "🎉 Emergency network lockdown completed in ${duration} seconds"
    log "📋 Review the report above and proceed with remaining security fixes"
    
    echo ""
    critical "🚨 NEXT CRITICAL STEPS:"
    echo "1. Run: ./credential-emergency-cleanup.sh"
    echo "2. Run: ./ssh-security-fix.sh" 
    echo "3. Run: ./database-security-hardening.sh"
    echo "4. Test application connectivity"
    echo "5. Monitor for any service disruptions"
}

# Export configuration for other scripts
export AWS_REGION="$REGION"
export YOUR_ADMIN_IP="$YOUR_ADMIN_IP"

# Run main function
main "$@"

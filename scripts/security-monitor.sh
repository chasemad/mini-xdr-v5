#!/bin/bash
# 🔍 Mini-XDR Security Monitoring System
# Continuous security monitoring and alerting

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
AWS_REGION="${AWS_REGION:-us-east-1}"

# Monitoring Configuration
LOG_GROUP_NAME="/aws/ec2/mini-xdr"
SECURITY_LOG_FILE="/tmp/mini-xdr-security-$(date +%Y%m%d).log"
ALERT_THRESHOLD_FAILED_AUTHS=5
ALERT_THRESHOLD_UNUSUAL_IPS=3

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

# Logging
log() {
    echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$SECURITY_LOG_FILE"
}

success() {
    log "${GREEN}✅ $1${NC}"
}

warning() {
    log "${YELLOW}⚠️  $1${NC}"
}

error() {
    log "${RED}❌ ALERT: $1${NC}"
}

info() {
    log "${BLUE}ℹ️  $1${NC}"
}

alert() {
    log "${RED}🚨 SECURITY ALERT: $1${NC}"
}

# Help function
show_help() {
    echo -e "${BLUE}Mini-XDR Security Monitoring System${NC}"
    echo
    echo "USAGE:"
    echo "  $0 [COMMAND] [OPTIONS]"
    echo
    echo "COMMANDS:"
    echo "  monitor    Start continuous security monitoring"
    echo "  scan       Run one-time security scan"
    echo "  check      Quick security status check"
    echo "  alerts     Show recent security alerts"
    echo "  report     Generate security report"
    echo
    echo "OPTIONS:"
    echo "  --interval SECONDS   Monitoring interval (default: 300)"
    echo "  --verbose           Enable verbose logging"
    echo "  --help              Show this help"
    echo
    echo "EXAMPLES:"
    echo "  $0 monitor                    # Start monitoring with default interval"
    echo "  $0 monitor --interval 60     # Monitor every minute"
    echo "  $0 scan                      # Run immediate security scan"
    echo "  $0 report                    # Generate security report"
}

# Check for unauthorized access attempts
check_failed_authentications() {
    info "Checking for failed authentication attempts..."
    
    local failed_auths=$(aws logs filter-log-events \
        --region "$AWS_REGION" \
        --log-group-name "$LOG_GROUP_NAME" \
        --filter-pattern "ERROR Authentication" \
        --start-time $(date -d "1 hour ago" +%s)000 \
        --query 'events[].message' \
        --output text 2>/dev/null | wc -l || echo "0")
    
    if [[ "$failed_auths" -gt "$ALERT_THRESHOLD_FAILED_AUTHS" ]]; then
        alert "High number of failed authentications: $failed_auths in the last hour"
        return 1
    else
        success "Failed authentication attempts: $failed_auths (within normal range)"
        return 0
    fi
}

# Monitor security group changes
check_security_group_changes() {
    info "Checking for unauthorized security group changes..."
    
    local open_groups=$(aws ec2 describe-security-groups \
        --region "$AWS_REGION" \
        --query 'SecurityGroups[?IpPermissions[?IpRanges[?CidrIp==`0.0.0.0/0`]]]' \
        --output json | jq '. | length' 2>/dev/null || echo "0")
    
    # Get T-Pot security group to check if it's intentionally open
    local tpot_instance_id="i-091156c8c15b7ece4"  # From your script
    local tpot_sg_open=$(aws ec2 describe-instances \
        --region "$AWS_REGION" \
        --instance-ids "$tpot_instance_id" \
        --query 'Reservations[0].Instances[0].SecurityGroups[0].GroupId' \
        --output text 2>/dev/null | xargs -I {} aws ec2 describe-security-groups \
        --region "$AWS_REGION" \
        --group-ids {} \
        --query 'SecurityGroups[0].IpPermissions[?IpRanges[?CidrIp==`0.0.0.0/0`]]' \
        --output json | jq '. | length' 2>/dev/null || echo "0")
    
    if [[ "$open_groups" -gt 1 ]] || ([[ "$open_groups" -eq 1 ]] && [[ "$tpot_sg_open" -eq 0 ]]); then
        alert "Unauthorized security groups open to 0.0.0.0/0 detected: $open_groups"
        aws ec2 describe-security-groups \
            --region "$AWS_REGION" \
            --query 'SecurityGroups[?IpPermissions[?IpRanges[?CidrIp==`0.0.0.0/0`]]].{GroupId:GroupId,GroupName:GroupName}' \
            --output table | tee -a "$SECURITY_LOG_FILE"
        return 1
    elif [[ "$tpot_sg_open" -gt 0 ]]; then
        warning "T-Pot honeypot is in LIVE mode (intentional exposure)"
        return 0
    else
        success "Security group configuration: SECURE (no unauthorized 0.0.0.0/0 access)"
        return 0
    fi
}

# Monitor credential access
check_credential_access() {
    info "Monitoring credential access patterns..."
    
    local secret_accesses=$(aws cloudtrail lookup-events \
        --region "$AWS_REGION" \
        --lookup-attributes AttributeKey=EventName,AttributeValue=GetSecretValue \
        --start-time $(date -d "1 hour ago" +%Y-%m-%dT%H:%M:%S) \
        --query 'Events | length(@)' \
        --output text 2>/dev/null || echo "0")
    
    if [[ "$secret_accesses" -gt 10 ]]; then
        warning "High number of secret accesses: $secret_accesses in the last hour"
        return 1
    else
        success "Secret access patterns: Normal ($secret_accesses accesses)"
        return 0
    fi
}

# Check for unusual IP addresses
check_unusual_ips() {
    info "Analyzing IP address patterns..."
    
    # This is a simplified check - in production you'd want more sophisticated analysis
    local unusual_ips=$(aws logs filter-log-events \
        --region "$AWS_REGION" \
        --log-group-name "$LOG_GROUP_NAME" \
        --filter-pattern "[timestamp, request_id, level=\"ERROR\", msg, ip_addr]" \
        --start-time $(date -d "1 hour ago" +%s)000 \
        --query 'events[].message' \
        --output text 2>/dev/null | grep -oE '\b([0-9]{1,3}\.){3}[0-9]{1,3}\b' | sort | uniq -c | sort -nr | head -5 | wc -l || echo "0")
    
    if [[ "$unusual_ips" -gt "$ALERT_THRESHOLD_UNUSUAL_IPS" ]]; then
        warning "Multiple error-generating IP addresses detected: $unusual_ips unique IPs"
        return 1
    else
        success "IP address patterns: Normal"
        return 0
    fi
}

# Check system resource usage
check_resource_usage() {
    info "Checking system resource usage..."
    
    # This would connect to your instances to check resources
    # For now, we'll just check if instances are running
    local backend_status=$(aws ec2 describe-instances \
        --region "$AWS_REGION" \
        --instance-ids "i-05ce3f39bd9c8f388" \
        --query 'Reservations[0].Instances[0].State.Name' \
        --output text 2>/dev/null || echo "unknown")
    
    local tpot_status=$(aws ec2 describe-instances \
        --region "$AWS_REGION" \
        --instance-ids "i-091156c8c15b7ece4" \
        --query 'Reservations[0].Instances[0].State.Name' \
        --output text 2>/dev/null || echo "unknown")
    
    if [[ "$backend_status" != "running" ]]; then
        error "Backend instance not running: $backend_status"
        return 1
    fi
    
    if [[ "$tpot_status" != "running" ]]; then
        error "T-Pot instance not running: $tpot_status"
        return 1
    fi
    
    success "All instances running normally"
    return 0
}

# Run comprehensive security scan
run_security_scan() {
    local scan_results=0
    
    echo -e "${PURPLE}=== Mini-XDR Security Scan $(date) ===${NC}"
    echo
    
    check_failed_authentications || ((scan_results++))
    echo
    
    check_security_group_changes || ((scan_results++))
    echo
    
    check_credential_access || ((scan_results++))
    echo
    
    check_unusual_ips || ((scan_results++))
    echo
    
    check_resource_usage || ((scan_results++))
    echo
    
    if [[ $scan_results -eq 0 ]]; then
        success "🎉 Security scan completed: No issues detected"
    else
        alert "Security scan completed: $scan_results issues detected"
    fi
    
    return $scan_results
}

# Generate security report
generate_security_report() {
    local report_file="/tmp/mini-xdr-security-report-$(date +%Y%m%d-%H%M%S).txt"
    
    info "Generating comprehensive security report..."
    
    {
        echo "=================================="
        echo "Mini-XDR Security Report"
        echo "Generated: $(date)"
        echo "=================================="
        echo
        
        echo "SYSTEM STATUS:"
        echo "--------------"
        aws ec2 describe-instances \
            --region "$AWS_REGION" \
            --instance-ids "i-05ce3f39bd9c8f388" "i-091156c8c15b7ece4" \
            --query 'Reservations[].Instances[].[InstanceId,State.Name,PublicIpAddress]' \
            --output table
        echo
        
        echo "SECURITY GROUPS:"
        echo "----------------"
        aws ec2 describe-security-groups \
            --region "$AWS_REGION" \
            --query 'SecurityGroups[?IpPermissions[?IpRanges[?CidrIp==`0.0.0.0/0`]]].{GroupId:GroupId,GroupName:GroupName,Description:Description}' \
            --output table
        echo
        
        echo "RECENT CLOUDTRAIL EVENTS:"
        echo "-------------------------"
        aws cloudtrail lookup-events \
            --region "$AWS_REGION" \
            --lookup-attributes AttributeKey=EventName,AttributeValue=GetSecretValue \
            --start-time $(date -d "24 hours ago" +%Y-%m-%dT%H:%M:%S) \
            --query 'Events[0:10].{Time:EventTime,User:Username,Event:EventName,Source:EventSource}' \
            --output table
        echo
        
        echo "RECOMMENDATIONS:"
        echo "----------------"
        echo "1. Review security group configurations regularly"
        echo "2. Monitor CloudTrail logs for unusual activity"
        echo "3. Rotate SSH keys monthly using aws/utils/ssh-key-rotation.sh"
        echo "4. Keep system and dependencies updated"
        echo "5. Review and update IAM policies quarterly"
        echo
        
    } > "$report_file"
    
    success "Security report generated: $report_file"
    
    # Show summary
    echo -e "${BLUE}Report Summary:${NC}"
    tail -20 "$report_file"
}

# Show recent alerts
show_recent_alerts() {
    info "Recent security alerts:"
    echo
    
    if [[ -f "$SECURITY_LOG_FILE" ]]; then
        grep -E "(ALERT|🚨)" "$SECURITY_LOG_FILE" | tail -10 || echo "No recent alerts found"
    else
        echo "No security log file found"
    fi
}

# Quick security status check
quick_status_check() {
    echo -e "${BLUE}Mini-XDR Security Status:${NC}"
    echo
    
    # Check if instances are running
    local backend_status=$(aws ec2 describe-instances \
        --region "$AWS_REGION" \
        --instance-ids "i-05ce3f39bd9c8f388" \
        --query 'Reservations[0].Instances[0].State.Name' \
        --output text 2>/dev/null || echo "unknown")
    
    echo -e "Backend Instance: ${GREEN}$backend_status${NC}"
    
    # Check T-Pot mode
    local tpot_sg_open=$(aws ec2 describe-instances \
        --region "$AWS_REGION" \
        --instance-ids "i-091156c8c15b7ece4" \
        --query 'Reservations[0].Instances[0].SecurityGroups[0].GroupId' \
        --output text 2>/dev/null | xargs -I {} aws ec2 describe-security-groups \
        --region "$AWS_REGION" \
        --group-ids {} \
        --query 'SecurityGroups[0].IpPermissions[?IpRanges[?CidrIp==`0.0.0.0/0`]]' \
        --output json | jq '. | length' 2>/dev/null || echo "0")
    
    if [[ "$tpot_sg_open" -gt 0 ]]; then
        echo -e "T-Pot Mode: ${RED}LIVE${NC} (accepting external traffic)"
    else
        echo -e "T-Pot Mode: ${GREEN}TESTING${NC} (restricted access)"
    fi
    
    echo -e "Last Security Scan: $(ls -la $SECURITY_LOG_FILE 2>/dev/null | awk '{print $6, $7, $8}' || echo 'Never')"
}

# Continuous monitoring loop
start_monitoring() {
    local interval="$1"
    
    echo -e "${PURPLE}🔍 Starting Mini-XDR Security Monitoring${NC}"
    echo "Monitoring interval: $interval seconds"
    echo "Log file: $SECURITY_LOG_FILE"
    echo "Press Ctrl+C to stop"
    echo
    
    while true; do
        run_security_scan
        echo
        info "Next scan in $interval seconds..."
        sleep "$interval"
    done
}

# Main function
main() {
    local command="${1:-check}"
    local interval="300"
    local verbose="false"
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            monitor|scan|check|alerts|report)
                command=$1
                ;;
            --interval)
                interval="$2"
                shift
                ;;
            --verbose)
                verbose="true"
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            *)
                if [[ "$1" != "$command" ]]; then
                    error "Unknown option: $1"
                    show_help
                    exit 1
                fi
                ;;
        esac
        shift
    done
    
    # Set verbose logging
    if [[ "$verbose" == "true" ]]; then
        set -x
    fi
    
    # Create log directory
    mkdir -p "$(dirname "$SECURITY_LOG_FILE")"
    
    # Execute command
    case $command in
        monitor)
            start_monitoring "$interval"
            ;;
        scan)
            run_security_scan
            ;;
        check)
            quick_status_check
            ;;
        alerts)
            show_recent_alerts
            ;;
        report)
            generate_security_report
            ;;
        *)
            error "Unknown command: $command"
            show_help
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"

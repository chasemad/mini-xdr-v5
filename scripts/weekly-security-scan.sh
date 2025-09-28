#!/bin/bash
# 📊 Mini-XDR Weekly Security Assessment
# Comprehensive weekly security scan and report

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
AWS_REGION="${AWS_REGION:-us-east-1}"

# Report Configuration
REPORT_FILE="/tmp/mini-xdr-weekly-security-$(date +%Y%m%d).txt"
EMAIL_REPORT="${EMAIL_REPORT:-false}"
ADMIN_EMAIL="${ADMIN_EMAIL:-admin@example.com}"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

# Logging
log() {
    echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$REPORT_FILE"
}

success() {
    log "${GREEN}✅ $1${NC}"
}

warning() {
    log "${YELLOW}⚠️  $1${NC}"
}

error() {
    log "${RED}❌ ISSUE: $1${NC}"
}

info() {
    log "${BLUE}ℹ️  $1${NC}"
}

critical() {
    log "${RED}🚨 CRITICAL: $1${NC}"
}

# Header
print_header() {
    {
        echo "======================================================="
        echo "          Mini-XDR Weekly Security Assessment"
        echo "======================================================="
        echo "Generated: $(date)"
        echo "Report Period: $(date -d '7 days ago' +%Y-%m-%d) to $(date +%Y-%m-%d)"
        echo "======================================================="
        echo
    } | tee "$REPORT_FILE"
}

# Check for hardcoded credentials in codebase
check_hardcoded_credentials() {
    info "🔍 Scanning for hardcoded credentials..."
    local issues=0
    
    # Patterns to search for
    local patterns=(
        "sk-proj"           # OpenAI API keys
        "xai-"              # X.AI API keys  
        "password.*="       # Password assignments
        "secret.*="         # Secret assignments
        "key.*="            # Key assignments
        "token.*="          # Token assignments
        "api_key.*="        # API key assignments
    )
    
    for pattern in "${patterns[@]}"; do
        local matches=$(grep -r "$pattern" "$PROJECT_ROOT" \
            --exclude-dir=.git \
            --exclude-dir=node_modules \
            --exclude-dir=venv \
            --exclude-dir=ml-training-env \
            --exclude="*.log" \
            --exclude="*security-*.txt" \
            --exclude="*audit*.md" \
            2>/dev/null | grep -v "CONFIGURE_IN_AWS_SECRETS_MANAGER" | head -10 || echo "")
        
        if [[ -n "$matches" ]]; then
            warning "Found potential credential pattern '$pattern':"
            echo "$matches" | sed 's/^/    /' | tee -a "$REPORT_FILE"
            ((issues++))
        fi
    done
    
    if [[ $issues -eq 0 ]]; then
        success "No hardcoded credentials detected"
    else
        error "$issues potential credential patterns found"
    fi
    
    return $issues
}

# Check security group configurations
check_security_groups() {
    info "🛡️  Analyzing security group configurations..."
    local issues=0
    
    # Get all security groups with 0.0.0.0/0 access
    local open_groups=$(aws ec2 describe-security-groups \
        --region "$AWS_REGION" \
        --query 'SecurityGroups[?IpPermissions[?IpRanges[?CidrIp==`0.0.0.0/0`]]]' \
        --output json 2>/dev/null || echo "[]")
    
    local open_count=$(echo "$open_groups" | jq '. | length' 2>/dev/null || echo "0")
    
    if [[ "$open_count" -gt 1 ]]; then
        error "Multiple security groups open to 0.0.0.0/0: $open_count"
        echo "$open_groups" | jq -r '.[] | "    \(.GroupId): \(.GroupName) - \(.Description)"' | tee -a "$REPORT_FILE"
        ((issues++))
    elif [[ "$open_count" -eq 1 ]]; then
        # Check if it's the T-Pot honeypot (intentional)
        local group_name=$(echo "$open_groups" | jq -r '.[0].GroupName' 2>/dev/null || echo "unknown")
        if [[ "$group_name" == *"tpot"* ]] || [[ "$group_name" == *"honeypot"* ]]; then
            warning "T-Pot honeypot security group open to 0.0.0.0/0 (intentional)"
        else
            error "Unexpected security group open to 0.0.0.0/0: $group_name"
            ((issues++))
        fi
    else
        success "All security groups properly restricted"
    fi
    
    return $issues
}

# Check for exposed secrets in git history
check_git_history() {
    info "📜 Scanning git history for exposed secrets..."
    local issues=0
    
    cd "$PROJECT_ROOT"
    
    # Search git history for potential secrets (last 100 commits)
    local secret_patterns=$(git log --all -p --since="1 week ago" 2>/dev/null | \
        grep -E "(password|key|secret|token)" | \
        grep -v "CONFIGURE_IN_AWS_SECRETS_MANAGER" | \
        grep -v "your-.*-key-here" | \
        head -20 || echo "")
    
    if [[ -n "$secret_patterns" ]]; then
        warning "Potential secrets found in recent git history:"
        echo "$secret_patterns" | sed 's/^/    /' | tee -a "$REPORT_FILE"
        ((issues++))
    else
        success "No secrets detected in recent git history"
    fi
    
    return $issues
}

# Check CloudTrail for suspicious activity
check_cloudtrail_activity() {
    info "🔍 Analyzing CloudTrail for suspicious activity..."
    local issues=0
    
    # Check for unusual API calls
    local unusual_events=$(aws cloudtrail lookup-events \
        --region "$AWS_REGION" \
        --start-time $(date -d "7 days ago" +%Y-%m-%dT%H:%M:%S) \
        --lookup-attributes AttributeKey=EventName,AttributeValue=CreateUser \
        --query 'Events | length(@)' \
        --output text 2>/dev/null || echo "0")
    
    if [[ "$unusual_events" -gt 0 ]]; then
        warning "User creation events detected: $unusual_events"
        ((issues++))
    fi
    
    # Check for secret access patterns
    local secret_accesses=$(aws cloudtrail lookup-events \
        --region "$AWS_REGION" \
        --start-time $(date -d "7 days ago" +%Y-%m-%dT%H:%M:%S) \
        --lookup-attributes AttributeKey=EventName,AttributeValue=GetSecretValue \
        --query 'Events | length(@)' \
        --output text 2>/dev/null || echo "0")
    
    if [[ "$secret_accesses" -gt 100 ]]; then
        warning "High number of secret accesses: $secret_accesses in the last week"
        ((issues++))
    else
        success "Secret access patterns normal: $secret_accesses accesses"
    fi
    
    return $issues
}

# Check system configurations
check_system_configurations() {
    info "⚙️  Checking system configurations..."
    local issues=0
    
    # Check if instances are properly configured
    local instances=$(aws ec2 describe-instances \
        --region "$AWS_REGION" \
        --instance-ids "i-05ce3f39bd9c8f388" "i-091156c8c15b7ece4" \
        --query 'Reservations[].Instances[]' \
        --output json 2>/dev/null || echo "[]")
    
    # Check for instances without proper tags
    local untagged=$(echo "$instances" | jq -r '.[] | select(.Tags | length == 0) | .InstanceId' 2>/dev/null || echo "")
    if [[ -n "$untagged" ]]; then
        warning "Instances without proper tags: $untagged"
        ((issues++))
    fi
    
    # Check for instances with public IP but no security group restrictions
    local exposed=$(echo "$instances" | jq -r '.[] | select(.PublicIpAddress != null and (.SecurityGroups[] | select(.GroupName | contains("default")))) | .InstanceId' 2>/dev/null || echo "")
    if [[ -n "$exposed" ]]; then
        error "Instances with default security groups and public IPs: $exposed"
        ((issues++))
    fi
    
    if [[ $issues -eq 0 ]]; then
        success "System configurations look good"
    fi
    
    return $issues
}

# Check for outdated dependencies
check_dependencies() {
    info "📦 Checking for outdated dependencies..."
    local issues=0
    
    # Check Python dependencies (if requirements.txt exists)
    if [[ -f "$PROJECT_ROOT/backend/requirements.txt" ]]; then
        cd "$PROJECT_ROOT/backend"
        
        # Check for known vulnerable packages (simplified check)
        local vulnerable_packages=$(grep -E "(django==1\.|flask==0\.|requests==2\.2[0-5])" requirements.txt 2>/dev/null || echo "")
        if [[ -n "$vulnerable_packages" ]]; then
            warning "Potentially vulnerable Python packages detected:"
            echo "$vulnerable_packages" | sed 's/^/    /' | tee -a "$REPORT_FILE"
            ((issues++))
        fi
    fi
    
    # Check Node.js dependencies (if package.json exists)
    if [[ -f "$PROJECT_ROOT/frontend/package.json" ]]; then
        cd "$PROJECT_ROOT/frontend"
        
        # Check for known vulnerable packages (simplified check)
        if command -v npm &> /dev/null; then
            local npm_audit=$(npm audit --audit-level=high --json 2>/dev/null | jq -r '.metadata.vulnerabilities.high // 0' || echo "0")
            if [[ "$npm_audit" -gt 0 ]]; then
                warning "High severity npm vulnerabilities detected: $npm_audit"
                ((issues++))
            fi
        fi
    fi
    
    if [[ $issues -eq 0 ]]; then
        success "No obvious dependency issues detected"
    fi
    
    return $issues
}

# Generate recommendations
generate_recommendations() {
    info "💡 Generating security recommendations..."
    
    {
        echo
        echo "SECURITY RECOMMENDATIONS:"
        echo "=========================="
        echo
        echo "1. IMMEDIATE ACTIONS:"
        echo "   - Review and address any critical issues above"
        echo "   - Rotate SSH keys using: aws/utils/ssh-key-rotation.sh rotate"
        echo "   - Update all system packages on EC2 instances"
        echo
        echo "2. WEEKLY TASKS:"
        echo "   - Review CloudTrail logs for unusual activity"
        echo "   - Check security group configurations"
        echo "   - Scan for hardcoded credentials in new code"
        echo "   - Update dependencies in backend and frontend"
        echo
        echo "3. MONTHLY TASKS:"
        echo "   - Rotate API keys in AWS Secrets Manager"
        echo "   - Review and update IAM policies"
        echo "   - Test backup and recovery procedures"
        echo "   - Conduct penetration testing"
        echo
        echo "4. QUARTERLY TASKS:"
        echo "   - Full security audit"
        echo "   - Review and update security policies"
        echo "   - Security training for team members"
        echo "   - Update incident response procedures"
        echo
        echo "5. MONITORING SETUP:"
        echo "   - Enable continuous monitoring: scripts/security-monitor.sh monitor"
        echo "   - Set up automated alerting for critical events"
        echo "   - Configure log aggregation and analysis"
        echo
    } | tee -a "$REPORT_FILE"
}

# Calculate security score
calculate_security_score() {
    local total_issues="$1"
    local max_possible_issues=20  # Adjust based on number of checks
    
    local score=$((100 - (total_issues * 100 / max_possible_issues)))
    if [[ $score -lt 0 ]]; then
        score=0
    fi
    
    {
        echo
        echo "SECURITY SCORE: $score/100"
        echo "==============="
        if [[ $score -ge 90 ]]; then
            echo "🟢 EXCELLENT - Your security posture is very strong"
        elif [[ $score -ge 75 ]]; then
            echo "🟡 GOOD - Minor security improvements needed"
        elif [[ $score -ge 60 ]]; then
            echo "🟠 FAIR - Several security issues need attention"
        else
            echo "🔴 POOR - Critical security issues require immediate attention"
        fi
        echo
    } | tee -a "$REPORT_FILE"
}

# Send email report (if configured)
send_email_report() {
    if [[ "$EMAIL_REPORT" == "true" ]] && command -v mail &> /dev/null; then
        info "📧 Sending email report to $ADMIN_EMAIL..."
        mail -s "Mini-XDR Weekly Security Report - $(date +%Y-%m-%d)" "$ADMIN_EMAIL" < "$REPORT_FILE"
        success "Email report sent"
    fi
}

# Main assessment function
run_weekly_assessment() {
    print_header
    
    local total_issues=0
    
    check_hardcoded_credentials || ((total_issues += $?))
    echo | tee -a "$REPORT_FILE"
    
    check_security_groups || ((total_issues += $?))
    echo | tee -a "$REPORT_FILE"
    
    check_git_history || ((total_issues += $?))
    echo | tee -a "$REPORT_FILE"
    
    check_cloudtrail_activity || ((total_issues += $?))
    echo | tee -a "$REPORT_FILE"
    
    check_system_configurations || ((total_issues += $?))
    echo | tee -a "$REPORT_FILE"
    
    check_dependencies || ((total_issues += $?))
    echo | tee -a "$REPORT_FILE"
    
    generate_recommendations
    calculate_security_score "$total_issues"
    
    {
        echo "SUMMARY:"
        echo "========"
        echo "Total Issues Found: $total_issues"
        echo "Report File: $REPORT_FILE"
        echo "Assessment Completed: $(date)"
        echo
    } | tee -a "$REPORT_FILE"
    
    send_email_report
    
    success "🎉 Weekly security assessment completed!"
    success "📄 Full report available at: $REPORT_FILE"
    
    return $total_issues
}

# Main function
main() {
    local command="${1:-assess}"
    
    case $command in
        assess|scan|report)
            run_weekly_assessment
            ;;
        --help|-h)
            echo "Mini-XDR Weekly Security Assessment"
            echo "Usage: $0 [assess|scan|report]"
            echo "Environment variables:"
            echo "  EMAIL_REPORT=true    Send report via email"
            echo "  ADMIN_EMAIL=email    Email address for reports"
            exit 0
            ;;
        *)
            error "Unknown command: $command"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"

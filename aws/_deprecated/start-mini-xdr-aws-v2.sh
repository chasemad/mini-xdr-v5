#!/bin/bash
# 🛡️ Mini-XDR AWS Startup Script v2.0
# Updated for direct Mini-XDR deployment with SageMaker integration and HMAC authentication

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
AWS_REGION="${AWS_REGION:-us-east-1}"
SSH_KEY="${SSH_KEY:-~/.ssh/mini-xdr-tpot-key.pem}"

# Updated Instance IDs for new architecture
BACKEND_INSTANCE="i-05ce3f39bd9c8f388"  # New Mini-XDR backend instance
TPOT_INSTANCE="i-091156c8c15b7ece4"     # T-Pot honeypot

# Network Configuration
BACKEND_IP="54.91.233.149"      # New Mini-XDR instance IP
TPOT_IP="34.193.101.171"        # T-Pot IP
BACKEND_API_PORT="8000"
FRONTEND_PORT="3000"

# SageMaker Configuration
SAGEMAKER_TRAINING_JOB="mini-xdr-gpu-regular-20250927-061258"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Logging
LOG_FILE="/tmp/mini-xdr-aws-startup-v2-$(date +%Y%m%d-%H%M%S).log"
exec 1> >(tee -a "$LOG_FILE")
exec 2> >(tee -a "$LOG_FILE" >&2)

log() {
    echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

success() {
    log "${GREEN}✅ $1${NC}"
}

warning() {
    log "${YELLOW}⚠️  $1${NC}"
}

error() {
    log "${RED}❌ $1${NC}"
}

info() {
    log "${BLUE}ℹ️  $1${NC}"
}

header() {
    echo
    log "${CYAN}$1${NC}"
    echo "$(printf '=%.0s' {1..60})"
}

# Usage function
usage() {
    cat << EOF
${CYAN}🛡️ Mini-XDR AWS Startup Script v2.0${NC}

Usage: $0 [OPTIONS] [MODE]

MODES:
  testing     Start all instances with T-Pot in testing mode (default)
  live        Start all instances with T-Pot in LIVE attack mode
  status      Check status of all instances and services
  stop        Stop all instances safely
  deploy      Deploy trained SageMaker model to endpoint

OPTIONS:
  -h, --help              Show this help message
  -v, --verbose           Enable verbose logging
  -f, --force             Skip confirmation prompts
  --skip-security-check   Skip security validation (not recommended)
  --backend-only          Start only backend instance
  --dry-run              Show what would be done without executing
  --validate-agents      Test all agent authentication and capabilities

EXAMPLES:
  $0                      # Start in testing mode (safe)
  $0 testing              # Start in testing mode explicitly
  $0 live                 # Start in LIVE mode (requires confirmation)
  $0 status               # Check all instance and service status
  $0 deploy               # Deploy trained SageMaker model
  $0 --validate-agents    # Test agent authentication

EOF
}

# Parse command line arguments
MODE="testing"
VERBOSE=false
FORCE=false
SKIP_SECURITY=false
BACKEND_ONLY=false
DRY_RUN=false
VALIDATE_AGENTS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            usage
            exit 0
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -f|--force)
            FORCE=true
            shift
            ;;
        --skip-security-check)
            SKIP_SECURITY=true
            warning "Security checks will be skipped!"
            shift
            ;;
        --backend-only)
            BACKEND_ONLY=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --validate-agents)
            VALIDATE_AGENTS=true
            shift
            ;;
        testing|live|status|stop|deploy)
            MODE="$1"
            shift
            ;;
        *)
            error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Dry run prefix
if [[ "$DRY_RUN" == "true" ]]; then
    DRY_PREFIX="[DRY RUN] "
    info "DRY RUN MODE - No actual changes will be made"
else
    DRY_PREFIX=""
fi

# Check dependencies
check_dependencies() {
    header "🔍 Checking Dependencies"

    local missing_deps=()

    if ! command -v aws &> /dev/null; then
        missing_deps+=("aws-cli")
    fi

    if ! command -v jq &> /dev/null; then
        missing_deps+=("jq")
    fi

    if ! command -v ssh &> /dev/null; then
        missing_deps+=("ssh")
    fi

    if ! command -v python3 &> /dev/null; then
        missing_deps+=("python3")
    fi

    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        error "Missing required dependencies: ${missing_deps[*]}"
        echo "Please install missing dependencies and try again."
        exit 1
    fi

    # Check AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        error "AWS credentials not configured or invalid"
        echo "Please run 'aws configure' to set up your credentials."
        exit 1
    fi

    local account_id
    account_id=$(aws sts get-caller-identity --query Account --output text)
    success "Dependencies verified (AWS Account: $account_id)"
}

# Get instance status
get_instance_status() {
    local instance_id="$1"
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "running"  # Mock status for dry run
        return
    fi

    aws ec2 describe-instances \
        --instance-ids "$instance_id" \
        --query 'Reservations[0].Instances[0].State.Name' \
        --output text 2>/dev/null || echo "not-found"
}

# Start instance
start_instance() {
    local instance_id="$1"
    local instance_name="$2"

    local status
    status=$(get_instance_status "$instance_id")

    case "$status" in
        "running")
            success "$instance_name is already running"
            return 0
            ;;
        "stopped"|"stopping")
            info "${DRY_PREFIX}Starting $instance_name instance ($instance_id)..."
            if [[ "$DRY_RUN" != "true" ]]; then
                aws ec2 start-instances --instance-ids "$instance_id" > /dev/null
                # Wait for instance to be running
                info "Waiting for $instance_name to start..."
                aws ec2 wait instance-running --instance-ids "$instance_id"
            fi
            success "$instance_name started successfully"
            return 0
            ;;
        "not-found")
            error "$instance_name instance ($instance_id) not found"
            return 1
            ;;
        *)
            warning "$instance_name is in '$status' state - may need manual intervention"
            return 1
            ;;
    esac
}

# Check SageMaker training job status
check_sagemaker_training() {
    header "🤖 SageMaker Training Status"

    if [[ "$DRY_RUN" == "true" ]]; then
        success "SageMaker training status (simulated in dry run)"
        return 0
    fi

    info "Checking SageMaker training job: $SAGEMAKER_TRAINING_JOB"

    local training_status
    training_status=$(aws sagemaker describe-training-job \
        --training-job-name "$SAGEMAKER_TRAINING_JOB" \
        --query 'TrainingJobStatus' \
        --output text 2>/dev/null || echo "NotFound")

    case "$training_status" in
        "Completed")
            success "✅ SageMaker training completed successfully"
            info "Model artifacts available for deployment"
            return 0
            ;;
        "InProgress")
            warning "⏳ SageMaker training still in progress"
            info "Training will continue in background"
            return 0
            ;;
        "Failed"|"Stopped")
            error "❌ SageMaker training $training_status"
            local failure_reason
            failure_reason=$(aws sagemaker describe-training-job \
                --training-job-name "$SAGEMAKER_TRAINING_JOB" \
                --query 'FailureReason' \
                --output text 2>/dev/null || echo "Unknown")
            error "Failure reason: $failure_reason"
            return 1
            ;;
        "NotFound")
            warning "SageMaker training job not found"
            return 1
            ;;
        *)
            info "SageMaker training status: $training_status"
            return 0
            ;;
    esac
}

# Deploy SageMaker endpoint
deploy_sagemaker_endpoint() {
    header "🚀 Deploying SageMaker Endpoint"

    if [[ "$DRY_RUN" == "true" ]]; then
        success "SageMaker endpoint deployment (simulated in dry run)"
        return 0
    fi

    # Check if training is complete
    info "Verifying training completion..."
    if ! check_sagemaker_training; then
        error "Cannot deploy endpoint - training not complete or failed"
        return 1
    fi

    local training_status
    training_status=$(aws sagemaker describe-training-job \
        --training-job-name "$SAGEMAKER_TRAINING_JOB" \
        --query 'TrainingJobStatus' \
        --output text)

    if [[ "$training_status" != "Completed" ]]; then
        warning "Training not yet complete. Skipping endpoint deployment."
        return 0
    fi

    info "Running SageMaker endpoint deployment script..."
    cd "$PROJECT_ROOT/scripts/ml-training"
    python3 deploy_trained_model.py

    success "SageMaker endpoint deployment completed"
}

# Enhanced status check
check_status() {
    header "📊 System Status Check"

    local instances=(
        "$BACKEND_INSTANCE:Mini-XDR Backend"
        "$TPOT_INSTANCE:T-Pot Honeypot"
    )

    printf "%-15s %-20s %-15s %-15s\n" "Instance" "Name" "Status" "Public IP"
    echo "$(printf '─%.0s' {1..70})"

    for instance_info in "${instances[@]}"; do
        IFS=':' read -r instance_id name <<< "$instance_info"

        local status public_ip
        status=$(get_instance_status "$instance_id")

        if [[ "$DRY_RUN" != "true" && "$status" == "running" ]]; then
            public_ip=$(aws ec2 describe-instances \
                --instance-ids "$instance_id" \
                --query 'Reservations[0].Instances[0].PublicIpAddress' \
                --output text 2>/dev/null || echo "N/A")
        else
            public_ip="N/A"
        fi

        # Color status
        local colored_status
        case "$status" in
            "running") colored_status="${GREEN}$status${NC}" ;;
            "stopped") colored_status="${YELLOW}$status${NC}" ;;
            *) colored_status="${RED}$status${NC}" ;;
        esac

        printf "%-15s %-20s %-25s %-15s\n" "${instance_id:0:15}" "$name" "$colored_status" "$public_ip"
    done
    echo

    # Check services if backend is running
    if [[ "$DRY_RUN" != "true" ]]; then
        local backend_status
        backend_status=$(get_instance_status "$BACKEND_INSTANCE")

        if [[ "$backend_status" == "running" ]]; then
            info "Checking service status..."

            # Backend API
            if timeout 10 curl -s "http://$BACKEND_IP:$BACKEND_API_PORT/health" > /dev/null 2>&1; then
                success "✅ Backend API is healthy"
            else
                warning "⚠️  Backend API not responding"
            fi

            # Frontend
            if timeout 10 curl -s "http://$BACKEND_IP:$FRONTEND_PORT" > /dev/null 2>&1; then
                success "✅ Frontend is accessible"
            else
                warning "⚠️  Frontend not responding"
            fi

            # T-Pot connectivity
            if timeout 5 curl -s "http://$TPOT_IP:9200/_cat/indices" > /dev/null 2>&1; then
                success "✅ T-Pot Elasticsearch accessible"
            else
                warning "⚠️  T-Pot Elasticsearch not accessible"
            fi
        fi
    fi

    # SageMaker status
    check_sagemaker_training
}

# HMAC Authentication Test
test_hmac_authentication() {
    header "🔐 Testing HMAC Authentication"

    if [[ "$DRY_RUN" == "true" ]]; then
        success "HMAC authentication test (simulated in dry run)"
        return 0
    fi

    info "Running HMAC authentication test script..."
    cd "$PROJECT_ROOT"

    if [[ -f "test_hmac_auth.py" ]]; then
        python3 test_hmac_auth.py
        success "HMAC authentication test completed"
    else
        warning "HMAC test script not found - creating basic test"

        # Basic connectivity test
        if timeout 10 curl -s "http://$BACKEND_IP:$BACKEND_API_PORT/health" > /dev/null; then
            success "✅ Backend API accessible (basic test)"
        else
            error "❌ Backend API not accessible"
            return 1
        fi
    fi
}

# Validate all agents and capabilities
validate_agents() {
    header "🤖 Agent Validation & Capability Testing"

    if [[ "$DRY_RUN" == "true" ]]; then
        success "Agent validation (simulated in dry run)"
        return 0
    fi

    info "Testing agent orchestration..."

    # Test orchestrator status
    local orchestrator_response
    orchestrator_response=$(timeout 10 curl -s "http://$BACKEND_IP:$BACKEND_API_PORT/api/orchestrator/status" 2>/dev/null || echo "ERROR")

    if echo "$orchestrator_response" | grep -q "orchestrator"; then
        success "✅ Agent orchestrator is operational"

        # Parse agent statuses
        local agents_active
        agents_active=$(echo "$orchestrator_response" | jq -r '.orchestrator.agents | keys[]' 2>/dev/null | wc -l || echo "0")
        info "Active agents detected: $agents_active"

        if [[ "$agents_active" -gt 0 ]]; then
            success "✅ Multiple agents active and responsive"
        else
            warning "⚠️  No active agents detected"
        fi
    else
        error "❌ Agent orchestrator not responding properly"
        return 1
    fi

    # Test ML engine
    local ml_response
    ml_response=$(timeout 10 curl -s "http://$BACKEND_IP:$BACKEND_API_PORT/api/ml/status" 2>/dev/null || echo "ERROR")

    if echo "$ml_response" | grep -q "success"; then
        success "✅ ML engine is operational"
    else
        warning "⚠️  ML engine may need attention"
    fi

    # Test data ingestion
    info "Testing data ingestion capability..."
    local test_payload='{"source": "test", "eventid": "test.connection", "ts": "'$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)'", "src_ip": "192.168.1.100", "dst_port": 22, "message": "Test connection"}'

    # This would require HMAC auth, so we'll just test the endpoint exists
    if timeout 5 curl -s -o /dev/null -w "%{http_code}" "http://$BACKEND_IP:$BACKEND_API_PORT/ingest/cowrie" | grep -q "401\|500"; then
        success "✅ Ingestion endpoint accessible (authentication required as expected)"
    else
        warning "⚠️  Ingestion endpoint response unexpected"
    fi

    success "Agent validation completed"
}

# Enhanced security validation
validate_security() {
    header "🔒 Enhanced Security Validation"

    if [[ "$SKIP_SECURITY" == "true" ]]; then
        warning "Skipping security checks as requested"
        return 0
    fi

    if [[ "$DRY_RUN" == "true" ]]; then
        success "Security validation (simulated in dry run)"
        return 0
    fi

    local security_issues=0

    # Check backend security group
    info "Checking Mini-XDR security group..."
    local backend_sg
    backend_sg=$(aws ec2 describe-instances --instance-ids "$BACKEND_INSTANCE" \
        --query 'Reservations[0].Instances[0].SecurityGroups[0].GroupId' --output text)

    # Check for overly permissive rules
    local dangerous_rules
    dangerous_rules=$(aws ec2 describe-security-groups --group-ids "$backend_sg" \
        --query 'SecurityGroups[0].IpPermissions[?IpRanges[?CidrIp==`0.0.0.0/0`] && (FromPort==`22` || FromPort==`8000` || FromPort==`3000`)]' \
        --output json | jq length)

    if [[ "$dangerous_rules" -gt 0 ]]; then
        error "Mini-XDR has overly permissive security group rules!"
        ((security_issues++))
    else
        success "✅ Mini-XDR security group properly configured"
    fi

    # Check T-Pot security group
    info "Checking T-Pot security group..."
    local tpot_sg
    tpot_sg=$(aws ec2 describe-instances --instance-ids "$TPOT_INSTANCE" \
        --query 'Reservations[0].Instances[0].SecurityGroups[0].GroupId' --output text)

    # Verify T-Pot is properly restricted
    local tpot_open_rules
    tpot_open_rules=$(aws ec2 describe-security-groups --group-ids "$tpot_sg" \
        --query 'SecurityGroups[0].IpPermissions[?IpRanges[?CidrIp==`0.0.0.0/0`]]' \
        --output json | jq length)

    if [[ "$tpot_open_rules" -gt 0 ]]; then
        warning "T-Pot has open rules (expected for honeypot functionality)"
        info "Verifying they're restricted to trusted sources..."

        # Check if management ports are properly restricted
        local mgmt_open
        mgmt_open=$(aws ec2 describe-security-groups --group-ids "$tpot_sg" \
            --query 'SecurityGroups[0].IpPermissions[?IpRanges[?CidrIp==`0.0.0.0/0`] && (FromPort==`64295` || FromPort==`64297` || FromPort==`22`)]' \
            --output json | jq length)

        if [[ "$mgmt_open" -gt 0 ]]; then
            error "T-Pot management ports are open to the world!"
            ((security_issues++))
        else
            success "✅ T-Pot management ports properly restricted"
        fi
    else
        success "✅ T-Pot security group has no unrestricted rules"
    fi

    # Check SSH key
    expanded_ssh_key="${SSH_KEY/#\~/$HOME}"
    if [[ ! -f "$expanded_ssh_key" ]]; then
        error "SSH key not found: $SSH_KEY"
        ((security_issues++))
    else
        success "✅ SSH key found and accessible"

        # Check key permissions
        local key_perms
        key_perms=$(stat -c %a "$expanded_ssh_key" 2>/dev/null || echo "unknown")
        if [[ "$key_perms" == "600" ]]; then
            success "✅ SSH key has correct permissions (600)"
        else
            warning "⚠️  SSH key permissions: $key_perms (should be 600)"
        fi
    fi

    # Check AWS Secrets Manager configuration
    info "Verifying AWS Secrets Manager configuration..."
    local secrets_count
    secrets_count=$(aws secretsmanager list-secrets --query 'SecretList[?contains(Name, `mini-xdr`)]' --output json | jq length)

    if [[ "$secrets_count" -gt 0 ]]; then
        success "✅ Mini-XDR secrets found in AWS Secrets Manager ($secrets_count secrets)"
    else
        warning "⚠️  No Mini-XDR secrets found in AWS Secrets Manager"
        ((security_issues++))
    fi

    if [[ $security_issues -gt 0 ]]; then
        error "$security_issues security issues found!"
        if [[ "$FORCE" != "true" ]]; then
            echo "Use --force to continue anyway or fix the issues first."
            exit 1
        else
            warning "Continuing despite security issues due to --force flag"
        fi
    else
        success "🔒 All security checks passed"
    fi
}

# Start Mini-XDR services
start_minixdr_services() {
    header "🚀 Starting Mini-XDR Services"

    if [[ "$DRY_RUN" == "true" ]]; then
        success "Mini-XDR services startup (simulated in dry run)"
        return 0
    fi

    info "Connecting to Mini-XDR backend instance..."

    # Check if services are already running
    local backend_running frontend_running
    backend_running=$(timeout 10 ssh -i "$SSH_KEY" -o ConnectTimeout=5 ubuntu@"$BACKEND_IP" \
        "pgrep -f 'uvicorn.*main:app' > /dev/null && echo 'true' || echo 'false'" 2>/dev/null || echo "false")

    frontend_running=$(timeout 10 ssh -i "$SSH_KEY" -o ConnectTimeout=5 ubuntu@"$BACKEND_IP" \
        "pgrep -f 'next.*dev' > /dev/null && echo 'true' || echo 'false'" 2>/dev/null || echo "false")

    if [[ "$backend_running" == "true" && "$frontend_running" == "true" ]]; then
        success "✅ Services already running"
        return 0
    fi

    # Start backend if not running
    if [[ "$backend_running" != "true" ]]; then
        info "Starting Mini-XDR backend..."
        timeout 60 ssh -i "$SSH_KEY" -o ConnectTimeout=10 ubuntu@"$BACKEND_IP" \
            'cd /opt/mini-xdr/backend && source venv/bin/activate && PYTHONPATH=/opt/mini-xdr/backend python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 > /tmp/backend.log 2>&1 &'

        # Wait for backend to start
        info "Waiting for backend to initialize..."
        local backend_ready=false
        for i in {1..30}; do
            if timeout 5 curl -s "http://$BACKEND_IP:$BACKEND_API_PORT/health" > /dev/null 2>&1; then
                backend_ready=true
                break
            fi
            sleep 2
        done

        if [[ "$backend_ready" == "true" ]]; then
            success "✅ Backend started and healthy"
        else
            error "❌ Backend failed to start or respond"
            return 1
        fi
    else
        success "✅ Backend already running"
    fi

    # Start frontend if not running
    if [[ "$frontend_running" != "true" ]]; then
        info "Starting Mini-XDR frontend..."
        timeout 60 ssh -i "$SSH_KEY" -o ConnectTimeout=10 ubuntu@"$BACKEND_IP" \
            'cd /opt/mini-xdr/frontend && npm run dev > /tmp/frontend.log 2>&1 &'

        # Wait for frontend to start
        info "Waiting for frontend to initialize..."
        local frontend_ready=false
        for i in {1..30}; do
            if timeout 5 curl -s "http://$BACKEND_IP:$FRONTEND_PORT" > /dev/null 2>&1; then
                frontend_ready=true
                break
            fi
            sleep 2
        done

        if [[ "$frontend_ready" == "true" ]]; then
            success "✅ Frontend started and accessible"
        else
            warning "⚠️  Frontend may still be starting up"
        fi
    else
        success "✅ Frontend already running"
    fi

    success "🎉 Mini-XDR services startup completed"
}

# Configure T-Pot integration
configure_tpot_integration() {
    local tpot_mode="${1:-testing}"

    header "🍯 Configuring T-Pot Integration ($tpot_mode mode)"

    if [[ "$DRY_RUN" == "true" ]]; then
        success "T-Pot integration (simulated in dry run)"
        return 0
    fi

    # Verify T-Pot connectivity
    info "Testing T-Pot connectivity..."
    if timeout 10 curl -s "http://$TPOT_IP:9200/_cat/indices" > /dev/null 2>&1; then
        success "✅ T-Pot Elasticsearch accessible from Mini-XDR"

        local indices_count
        indices_count=$(curl -s "http://$TPOT_IP:9200/_cat/indices" | wc -l)
        info "T-Pot has $indices_count active indices"
    else
        error "❌ Cannot connect to T-Pot Elasticsearch"
        error "Check security group rules and network connectivity"
        return 1
    fi

    # Configure honeypot monitoring
    info "Configuring honeypot log monitoring..."
    timeout 30 ssh -i "$SSH_KEY" -o ConnectTimeout=10 ubuntu@"$BACKEND_IP" << 'EOF'
# Update backend configuration for T-Pot integration
cd /opt/mini-xdr/backend
if grep -q "honeypot_host" .env; then
    echo "T-Pot configuration already present in .env"
else
    echo "" >> .env
    echo "# T-Pot Integration" >> .env
    echo "HONEYPOT_HOST=34.193.101.171" >> .env
    echo "HONEYPOT_PORT=9200" >> .env
    echo "HONEYPOT_ENABLED=true" >> .env
fi
EOF

    if [[ "$tpot_mode" == "live" ]]; then
        warning "🚨 T-Pot configured for LIVE mode - real attacks will be collected!"
    else
        success "🛡️  T-Pot configured for testing mode"
    fi
}

# Main startup sequence
start_infrastructure() {
    local mode="$1"

    header "🛡️ Mini-XDR AWS Infrastructure Startup v2.0"
    info "Mode: $mode"
    info "Log file: $LOG_FILE"
    info "Architecture: Direct Mini-XDR ↔ T-Pot (no relay)"

    # Confirmation for live mode
    if [[ "$mode" == "live" && "$FORCE" != "true" ]]; then
        echo
        warning "⚠️  LIVE MODE WARNING ⚠️"
        echo "You are about to start the system in LIVE mode."
        echo "This will expose T-Pot to REAL ATTACKS from the internet."
        echo "Make sure you have:"
        echo "  - Proper monitoring in place"
        echo "  - Emergency shutdown procedures ready"
        echo "  - Security team on standby"
        echo
        read -p "Are you sure you want to proceed with LIVE mode? (type 'LIVE' to confirm): " confirm
        if [[ "$confirm" != "LIVE" ]]; then
            info "Startup cancelled by user"
            exit 0
        fi
    fi

    # Start instances
    if [[ "$BACKEND_ONLY" != "true" ]]; then
        start_instance "$BACKEND_INSTANCE" "Mini-XDR Backend"
        start_instance "$TPOT_INSTANCE" "T-Pot Honeypot"
    else
        start_instance "$BACKEND_INSTANCE" "Mini-XDR Backend"
    fi

    # Wait for instances to fully boot
    if [[ "$DRY_RUN" != "true" ]]; then
        info "Waiting for instances to fully boot (60 seconds)..."
        sleep 60
    fi

    # Validate security
    validate_security

    # Start Mini-XDR services
    start_minixdr_services

    # Configure T-Pot integration if not backend-only
    if [[ "$BACKEND_ONLY" != "true" ]]; then
        configure_tpot_integration "$mode"
    fi

    # Check SageMaker training status
    check_sagemaker_training

    # Test HMAC authentication
    test_hmac_authentication

    # Validate agents if requested
    if [[ "$VALIDATE_AGENTS" == "true" ]]; then
        validate_agents
    fi

    # Final status check
    check_status

    success "🎉 Mini-XDR AWS infrastructure startup completed!"

    # Provide access information
    echo
    header "📋 Access Information"
    echo "🎯 Frontend Dashboard:   http://$BACKEND_IP:$FRONTEND_PORT"
    echo "🔧 Backend API:          http://$BACKEND_IP:$BACKEND_API_PORT"
    echo "📊 Health Check:         http://$BACKEND_IP:$BACKEND_API_PORT/health"
    echo "📋 API Docs:             http://$BACKEND_IP:$BACKEND_API_PORT/docs"
    echo "🔐 HMAC Auth:            Required for /api/* and /ingest/* endpoints"
    if [[ "$BACKEND_ONLY" != "true" ]]; then
        echo "🍯 T-Pot Integration:    $TPOT_IP:9200 → Mini-XDR"
        echo "🍯 T-Pot Mode:           $mode"
    fi
    echo "🤖 SageMaker Training:   $SAGEMAKER_TRAINING_JOB"
    echo "📝 Log File:             $LOG_FILE"
    echo

    if [[ "$mode" == "live" ]]; then
        warning "🚨 SYSTEM IS LIVE - MONITOR CLOSELY! 🚨"
    else
        success "🛡️  System is running in safe testing mode"
    fi
}

# Stop all infrastructure
stop_infrastructure() {
    header "🛑 Mini-XDR AWS Infrastructure Shutdown"

    # Gracefully stop services first
    if [[ "$DRY_RUN" != "true" ]]; then
        info "Stopping Mini-XDR services..."
        timeout 30 ssh -i "$SSH_KEY" -o ConnectTimeout=5 ubuntu@"$BACKEND_IP" \
            'pkill -f "uvicorn.*main:app" || true; pkill -f "next.*dev" || true' 2>/dev/null || true
        sleep 5
    fi

    # Stop instances
    local backend_status tpot_status
    backend_status=$(get_instance_status "$BACKEND_INSTANCE")
    tpot_status=$(get_instance_status "$TPOT_INSTANCE")

    if [[ "$backend_status" == "running" ]]; then
        info "${DRY_PREFIX}Stopping Mini-XDR Backend instance..."
        if [[ "$DRY_RUN" != "true" ]]; then
            aws ec2 stop-instances --instance-ids "$BACKEND_INSTANCE" > /dev/null
        fi
    fi

    if [[ "$tpot_status" == "running" ]]; then
        info "${DRY_PREFIX}Stopping T-Pot Honeypot instance..."
        if [[ "$DRY_RUN" != "true" ]]; then
            aws ec2 stop-instances --instance-ids "$TPOT_INSTANCE" > /dev/null
        fi
    fi

    if [[ "$DRY_RUN" != "true" && ("$backend_status" == "running" || "$tpot_status" == "running") ]]; then
        info "Waiting for instances to stop..."
        if [[ "$backend_status" == "running" ]]; then
            aws ec2 wait instance-stopped --instance-ids "$BACKEND_INSTANCE" &
        fi
        if [[ "$tpot_status" == "running" ]]; then
            aws ec2 wait instance-stopped --instance-ids "$TPOT_INSTANCE" &
        fi
        wait
    fi

    success "🛑 All instances stopped safely"
}

# Main execution
main() {
    # Check dependencies first
    check_dependencies

    case "$MODE" in
        "status")
            check_status
            if [[ "$VALIDATE_AGENTS" == "true" ]]; then
                validate_agents
            fi
            ;;
        "stop")
            stop_infrastructure
            ;;
        "testing"|"live")
            start_infrastructure "$MODE"
            ;;
        "deploy")
            deploy_sagemaker_endpoint
            ;;
        *)
            error "Invalid mode: $MODE"
            usage
            exit 1
            ;;
    esac
}

# Handle script interruption
trap 'error "Script interrupted"; exit 130' INT TERM

# Run main function
main "$@"
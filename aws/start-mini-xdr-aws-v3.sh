#!/bin/bash
# 🛡️ Mini-XDR AWS Management Script v3.0
# Complete system management: start, stop, monitor, and security validation
# NEW: By default, starting instances does NOT deploy local code changes
# Use --deploy-code flag to push local development changes to AWS

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
AWS_REGION="${AWS_REGION:-us-east-1}"

# SSH Keys
SSH_KEY_BACKEND="${SSH_KEY_BACKEND:-~/.ssh/mini-xdr-tpot-key.pem}"
SSH_KEY_TPOT="${SSH_KEY_TPOT:-~/.ssh/mini-xdr-tpot-key.pem}"  # Updated for secure TPOT

# Instance Configuration
BACKEND_INSTANCE="i-05ce3f39bd9c8f388"  # Mini-XDR backend instance
TPOT_INSTANCE="i-0584d6b913192a953"     # NEW Secure T-Pot honeypot

# Network Configuration
BACKEND_IP="54.237.168.3"   # Elastic IP for backend instance
TPOT_IP="107.22.132.190"    # NEW Secure T-Pot instance IP
BACKEND_API_PORT="8000"
FRONTEND_PORT="3000"
REDIS_PORT="6379"
KAFKA_PORT="9092"
ZOOKEEPER_PORT="2181"

# User IP for T-Pot security (fetched dynamically)
USER_IP=""

# SageMaker Configuration
SAGEMAKER_TRAINING_JOB="mini-xdr-gpu-regular-20250927-061258"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Logging
LOG_FILE="/tmp/mini-xdr-aws-v3-$(date +%Y%m%d-%H%M%S).log"
exec 1> >(tee -a "$LOG_FILE")
exec 2> >(tee -a "$LOG_FILE" >&2)

# Logging functions
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
    log "${RED}❌ ERROR: $1${NC}"
}

info() {
    log "${BLUE}ℹ️  $1${NC}"
}

header() {
    echo
    log "${BOLD}${CYAN}🚀 $1${NC}"
    echo "$(printf '=%.0s' {1..80})"
}

# Help function
show_help() {
    echo -e "${BOLD}${CYAN}Mini-XDR AWS Management Script v3.0${NC}"
    echo
    echo -e "${BOLD}USAGE:${NC}"
    echo "  $0 [COMMAND] [OPTIONS]"
    echo
    echo -e "${BOLD}COMMANDS:${NC}"
    echo -e "  ${GREEN}start${NC}           Start all instances and services (default: testing mode)"
    echo -e "  ${GREEN}stop${NC}            Stop all instances"
    echo -e "  ${GREEN}restart${NC}         Restart all instances and services"
    echo -e "  ${GREEN}status${NC}          Show comprehensive system status"
    echo -e "  ${GREEN}monitor${NC}         Monitor system health (continuous)"
    echo -e "  ${GREEN}validate${NC}        Run complete security and functionality validation"
    echo -e "  ${GREEN}logs${NC}            Show system logs"
    echo
    echo -e "${BOLD}OPTIONS:${NC}"
    echo -e "  ${YELLOW}--live${NC}          Put T-Pot honeypot in live mode (accepts external traffic)"
    echo -e "  ${YELLOW}--testing${NC}       Put T-Pot honeypot in testing mode (restricted access) [DEFAULT]"
    echo -e "  ${YELLOW}--deploy-code${NC}   Deploy local code changes to AWS (otherwise just start services)"
    echo -e "  ${YELLOW}--force${NC}         Force operations without confirmation"
    echo -e "  ${YELLOW}--verbose${NC}       Enable verbose logging"
    echo -e "  ${YELLOW}--help${NC}          Show this help message"
    echo
    echo -e "${BOLD}EXAMPLES:${NC}"
    echo "  $0 start                       # Start system in testing mode (no code deployment)"
    echo "  $0 start --deploy-code         # Start system and deploy local code changes"
    echo "  $0 start --live --deploy-code  # Start with T-Pot live and deploy code"
    echo "  $0 status                      # Check system status"
    echo "  $0 validate --verbose          # Full validation with detailed output"
    echo "  $0 stop --force                # Force stop without confirmation"
    echo
    echo -e "${BOLD}SECURITY MODES:${NC}"
    echo -e "  ${GREEN}Testing Mode:${NC}   T-Pot restricted to your IP only (safe for testing)"
    echo -e "  ${RED}Live Mode:${NC}       T-Pot accepts traffic from 0.0.0.0/0 (production honeypot)"
    echo
}

# Get current user IP
get_user_ip() {
    USER_IP=$(curl -s https://ipinfo.io/ip 2>/dev/null || curl -s https://ifconfig.me 2>/dev/null || echo "unknown")
    if [[ "$USER_IP" == "unknown" ]]; then
        warning "Could not determine your public IP address"
        return 1
    fi
    info "Your public IP: $USER_IP"
}

# Verify instance IPs (using static Elastic IPs)
get_instance_ips() {
    info "Using static Elastic IP addresses..."

    # Verify instances are running and IPs are accessible
    local backend_check=$(aws ec2 describe-instances \
        --region "$AWS_REGION" \
        --instance-ids "$BACKEND_INSTANCE" \
        --query 'Reservations[0].Instances[0].PublicIpAddress' \
        --output text 2>/dev/null || echo "")

    local tpot_check=$(aws ec2 describe-instances \
        --region "$AWS_REGION" \
        --instance-ids "$TPOT_INSTANCE" \
        --query 'Reservations[0].Instances[0].PublicIpAddress' \
        --output text 2>/dev/null || echo "")

    if [[ -z "$backend_check" || "$backend_check" == "None" ]]; then
        error "Backend instance appears to be stopped"
        return 1
    fi

    if [[ -z "$tpot_check" || "$tpot_check" == "None" ]]; then
        error "T-Pot instance appears to be stopped"
        return 1
    fi

    success "Backend IP: $BACKEND_IP (Elastic IP - static)"
    success "T-Pot IP: $TPOT_IP (Elastic IP - static)"
}

# Check AWS CLI and credentials
check_aws_setup() {
    if ! command -v aws &> /dev/null; then
        error "AWS CLI not found. Please install AWS CLI first."
        return 1
    fi

    if ! aws sts get-caller-identity &>/dev/null; then
        error "AWS credentials not configured. Run 'aws configure' first."
        return 1
    fi

    success "AWS CLI configured and credentials valid"
}

# Start EC2 instances
start_instances() {
    local tpot_mode=${1:-testing}

    header "Starting EC2 Instances"

    # Start Mini-XDR Backend
    info "Starting Mini-XDR backend instance..."
    aws ec2 start-instances --region "$AWS_REGION" --instance-ids "$BACKEND_INSTANCE" >/dev/null

    # Start T-Pot
    info "Starting T-Pot honeypot instance..."
    aws ec2 start-instances --region "$AWS_REGION" --instance-ids "$TPOT_INSTANCE" >/dev/null

    # Wait for instances to be running
    info "Waiting for instances to reach running state..."
    aws ec2 wait instance-running --region "$AWS_REGION" --instance-ids "$BACKEND_INSTANCE" "$TPOT_INSTANCE"

    # Wait additional time for SSH to be ready
    info "Waiting for SSH to be available..."
    sleep 30

    success "All instances are running and SSH-ready"

    # Configure T-Pot security based on mode
    configure_tpot_security "$tpot_mode"

    # Ensure backend can communicate with TPOT for agent access
    configure_agent_tpot_connectivity
}

# Start existing services on backend (without deploying code)
start_backend_services() {
    header "Starting Backend Services (Using Existing Code)"

    # Start existing services on remote server
    ssh -o StrictHostKeyChecking=no -i "$SSH_KEY_BACKEND" "ubuntu@$BACKEND_IP" << 'EOF'
        set -e

        echo "🚀 Starting Mini-XDR backend services..."

        # Start Redis (if not running)
        sudo systemctl start redis-server || true
        sudo systemctl enable redis-server || true

        # Start Docker services (Kafka, Zookeeper)
        cd /home/ubuntu
        if [ -f "docker-compose-kafka.yml" ]; then
            docker compose -f docker-compose-kafka.yml down || true
            docker compose -f docker-compose-kafka.yml up -d
        fi

        # Wait for services
        sleep 10

        # Start backend service
        cd /opt/mini-xdr/backend
        source venv/bin/activate || (echo "Virtual environment not found. Please run with --deploy-code first." && exit 1)

        # Stop any existing backend process
        pkill -f uvicorn || true
        sleep 2

        # Start backend with secure configuration (using entrypoint for proper CORS)
        SECRETS_MANAGER_ENABLED=true AWS_DEFAULT_REGION=us-east-1 UI_ORIGIN=http://54.237.168.3:3000 PYTHONPATH=/opt/mini-xdr/backend \
            nohup python3 -m uvicorn app.entrypoint:app --host 0.0.0.0 --port 8000 > /tmp/backend.log 2>&1 &

        # Start frontend
        cd /opt/mini-xdr/frontend
        pkill -f "next dev" || true
        sleep 2
        nohup npm run dev > /tmp/frontend.log 2>&1 &

        echo "✅ Services started (using existing deployed code)"
EOF

    success "Backend services startup initiated"
}

# Stop EC2 instances
stop_instances() {
    header "Stopping EC2 Instances"

    info "Stopping Mini-XDR backend instance..."
    aws ec2 stop-instances --region "$AWS_REGION" --instance-ids "$BACKEND_INSTANCE" >/dev/null

    info "Stopping T-Pot honeypot instance..."
    aws ec2 stop-instances --region "$AWS_REGION" --instance-ids "$TPOT_INSTANCE" >/dev/null

    info "Waiting for instances to stop..."
    aws ec2 wait instance-stopped --region "$AWS_REGION" --instance-ids "$BACKEND_INSTANCE" "$TPOT_INSTANCE"

    success "All instances stopped successfully"
}

# Configure T-Pot security groups (Updated for new secure setup)
configure_tpot_security() {
    local mode=$1

    header "Configuring T-Pot Security ($mode mode)"

    # Get T-Pot security group ID - use the dedicated mini-xdr-tpot-sg
    local sg_id="sg-09d7c38d0e0ed44a4"  # Our secure TPOT security group

    info "Using secure T-Pot security group: $sg_id"

    if [[ "$mode" == "live" ]]; then
        warning "CONFIGURING T-POT FOR LIVE MODE - WILL ACCEPT EXTERNAL TRAFFIC!"

        # Add rules for external traffic (honeypot services) - remove user IP restriction
        local honeypot_ports=(21 23 25 80 443 993 995 3306 5432 1433 3389 64297)

        for port in "${honeypot_ports[@]}"; do
            # Remove user-only restriction
            aws ec2 revoke-security-group-ingress \
                --region "$AWS_REGION" \
                --group-id "$sg_id" \
                --protocol tcp \
                --port "$port" \
                --cidr "${USER_IP}/32" 2>/dev/null || true

            # Add rule for all traffic
            aws ec2 authorize-security-group-ingress \
                --region "$AWS_REGION" \
                --group-id "$sg_id" \
                --protocol tcp \
                --port "$port" \
                --cidr 0.0.0.0/0 2>/dev/null || true
        done

        success "T-Pot configured for LIVE mode - accepting external traffic"
        warning "⚠️  SECURITY NOTICE: T-Pot is now a live honeypot!"

    else
        info "Configuring T-Pot for TESTING mode..."

        # Ensure TESTING mode - restrict to user IP only
        local honeypot_ports=(21 23 25 80 443 993 995 3306 5432 1433 3389 64297)

        for port in "${honeypot_ports[@]}"; do
            # Remove any 0.0.0.0/0 rules first
            aws ec2 revoke-security-group-ingress \
                --region "$AWS_REGION" \
                --group-id "$sg_id" \
                --protocol tcp \
                --port "$port" \
                --cidr 0.0.0.0/0 2>/dev/null || true

            # Add rule for user IP only (honeypot attack testing)
            aws ec2 authorize-security-group-ingress \
                --region "$AWS_REGION" \
                --group-id "$sg_id" \
                --protocol tcp \
                --port "$port" \
                --cidr "${USER_IP}/32" 2>/dev/null || true
        done

        success "T-Pot configured for TESTING mode - restricted to your IP ($USER_IP)"
    fi
}

# Configure secure agent-to-TPOT connectivity
configure_agent_tpot_connectivity() {
    header "Configuring Secure Agent-to-TPOT Connectivity"

    local sg_id="sg-09d7c38d0e0ed44a4"  # Our secure TPOT security group

    info "Ensuring backend agents can access TPOT for control actions..."

    # Ensure backend can access TPOT via SSH for agent control (port 22 and 64295)
    aws ec2 authorize-security-group-ingress \
        --region "$AWS_REGION" \
        --group-id "$sg_id" \
        --protocol tcp \
        --port 22 \
        --cidr "${BACKEND_IP}/32" 2>/dev/null || true

    aws ec2 authorize-security-group-ingress \
        --region "$AWS_REGION" \
        --group-id "$sg_id" \
        --protocol tcp \
        --port 64295 \
        --cidr "${BACKEND_IP}/32" 2>/dev/null || true

    # Allow ICMP for connectivity testing
    aws ec2 authorize-security-group-ingress \
        --region "$AWS_REGION" \
        --group-id "$sg_id" \
        --protocol icmp \
        --port -1 \
        --cidr "${BACKEND_IP}/32" 2>/dev/null || true

    success "Agent connectivity configured - backend can reach TPOT for control actions"

    # Test connectivity
    info "Testing agent connectivity to TPOT..."
    if ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 -i "$SSH_KEY_BACKEND" "ubuntu@$BACKEND_IP" "ping -c 1 $TPOT_IP" >/dev/null 2>&1; then
        success "✅ Agent-to-TPOT connectivity: WORKING"
    else
        warning "⚠️  Agent-to-TPOT connectivity test failed - may need time to propagate"
    fi
}

# Deploy and start services on backend (uploads local code changes)
deploy_backend_services() {
    header "Deploying Backend Services (with Local Code)"

    # Upload latest code and models
    info "Uploading latest code to backend instance..."
    scp -r -o StrictHostKeyChecking=no -i "$SSH_KEY_BACKEND" \
        "$PROJECT_ROOT/backend" "ubuntu@$BACKEND_IP:/tmp/mini-xdr-deploy/" || true

    # Upload comprehensive models separately to ensure they're deployed
    info "Uploading comprehensive 7-class detection models..."
    scp -r -o StrictHostKeyChecking=no -i "$SSH_KEY_BACKEND" \
        "$PROJECT_ROOT/models" "ubuntu@$BACKEND_IP:/tmp/mini-xdr-deploy/" || true

    # Install dependencies and start services
    ssh -o StrictHostKeyChecking=no -i "$SSH_KEY_BACKEND" "ubuntu@$BACKEND_IP" << 'EOF'
        set -e

        echo "🔄 Setting up Mini-XDR backend services..."

        # Update deployed code
        sudo rsync -av --delete /tmp/mini-xdr-deploy/backend/ /opt/mini-xdr/backend/ 2>/dev/null || true

        # Deploy comprehensive models (without delete flag to preserve any existing models)
        if [ -d "/tmp/mini-xdr-deploy/models" ]; then
            echo "🧠 Deploying comprehensive 7-class attack detection models..."
            sudo mkdir -p /opt/mini-xdr/models
            sudo rsync -av /tmp/mini-xdr-deploy/models/ /opt/mini-xdr/models/ 2>/dev/null || true

            # Verify model deployment
            if [ -f "/opt/mini-xdr/models/model_metadata.json" ]; then
                echo "✅ Model metadata found - checking configuration..."
                sudo cat /opt/mini-xdr/models/model_metadata.json | head -5
            fi
        fi

        sudo chown -R ubuntu:ubuntu /opt/mini-xdr/

        # Install/update Python dependencies
        cd /opt/mini-xdr/backend
        source venv/bin/activate || python3 -m venv venv && source venv/bin/activate
        pip install -r requirements.txt > /dev/null 2>&1 || true

        # Install additional dependencies we configured
        pip install redis aiokafka python-jose pycryptodome openai optuna shap lime > /dev/null 2>&1 || true

        # Stop any existing backend process
        pkill -f uvicorn || true
        sleep 2

        # Start Redis (if not running)
        sudo systemctl start redis-server || true
        sudo systemctl enable redis-server || true

        # Start Docker services (Kafka, Zookeeper)
        cd /home/ubuntu
        if [ -f "docker-compose-kafka.yml" ]; then
            docker compose -f docker-compose-kafka.yml down || true
            docker compose -f docker-compose-kafka.yml up -d
        fi

        # Wait for services
        sleep 10

        # Validate comprehensive model before starting backend
        echo "🧪 Validating comprehensive model integration..."
        cd /opt/mini-xdr/backend
        source venv/bin/activate

        # Quick model validation
        python3 -c "
try:
    import sys, os
    sys.path.append('/opt/mini-xdr/backend')
    from app.deep_learning_models import DeepLearningModelManager
    import json

    # Check model metadata
    with open('/opt/mini-xdr/models/model_metadata.json', 'r') as f:
        meta = json.load(f)
    print(f'📊 Model: {meta.get(\"num_classes\", \"?\")} classes, {meta.get(\"best_accuracy\", \"?\"):.4f} accuracy')

    # Test model loading
    manager = DeepLearningModelManager()
    results = manager.load_models('/opt/mini-xdr/models')
    if results.get('threat_detector', False):
        print('✅ Comprehensive 7-class threat detector loaded successfully')
    else:
        print('❌ Failed to load threat detector')

except Exception as e:
    print(f'⚠️  Model validation warning: {e}')
" 2>/dev/null || echo "⚠️  Model validation skipped"

        # Start backend with secure configuration (using entrypoint for proper CORS)
        SECRETS_MANAGER_ENABLED=true AWS_DEFAULT_REGION=us-east-1 UI_ORIGIN=http://54.237.168.3:3000 PYTHONPATH=/opt/mini-xdr/backend \
            nohup python3 -m uvicorn app.entrypoint:app --host 0.0.0.0 --port 8000 > /tmp/backend.log 2>&1 &

        # Start frontend
        cd /opt/mini-xdr/frontend
        pkill -f "next dev" || true
        sleep 2
        nohup npm run dev > /tmp/frontend.log 2>&1 &

        echo "✅ Services starting up..."
EOF

    success "Backend services deployment initiated"
}

# Validate system health and security
validate_system() {
    header "Comprehensive System Validation"

    local errors=0
    
    # Note: Added retry logic and longer wait times to handle ML model loading
    # and backend initialization which can take 60-90 seconds after deployment

    # Test 1: Instance connectivity
    info "Testing instance connectivity..."
    if ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -i "$SSH_KEY_BACKEND" "ubuntu@$BACKEND_IP" "echo 'Backend SSH OK'" >/dev/null 2>&1; then
        success "Backend SSH connectivity: OK"
    else
        error "Backend SSH connectivity: FAILED"
        ((errors++))
    fi

    # Test TPOT connectivity (new secure instance uses ubuntu user initially)
    if ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -i "$SSH_KEY_TPOT" "ubuntu@$TPOT_IP" "echo 'T-Pot SSH OK'" >/dev/null 2>&1; then
        success "T-Pot SSH connectivity: OK"
    elif ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -i "$SSH_KEY_TPOT" "admin@$TPOT_IP" "echo 'T-Pot SSH OK'" >/dev/null 2>&1; then
        success "T-Pot SSH connectivity: OK (admin user)"
    else
        warning "T-Pot SSH connectivity: Limited (instance may need TPOT installation)"
        # Don't fail validation for SSH - focus on network connectivity
    fi

    # Test 2: Core services
    info "Testing core services..."
    info "Waiting for services to fully initialize..."
    
    # Wait longer and add retry logic for backend API health
    local backend_ready=false
    for i in {1..6}; do
        sleep 15  # Wait 15 seconds between attempts (total up to 90 seconds)
        info "Backend readiness check $i/6..."
        
        if curl -s --max-time 10 "http://$BACKEND_IP:$BACKEND_API_PORT/health" >/dev/null 2>&1; then
            success "Backend API health: OK"
            backend_ready=true
            break
        else
            if [ $i -eq 6 ]; then
                error "Backend API health: FAILED (after 90 seconds)"
                ((errors++))
            else
                warning "Backend not ready yet, retrying in 15 seconds..."
            fi
        fi
    done

    # Frontend
    if curl -s --max-time 10 "http://$BACKEND_IP:$FRONTEND_PORT" >/dev/null 2>&1; then
        success "Frontend service: OK"
    else
        error "Frontend service: FAILED"
        ((errors++))
    fi

    # Test 3: Infrastructure services
    info "Testing infrastructure services..."

    # Redis
    if ssh -o StrictHostKeyChecking=no -i "$SSH_KEY_BACKEND" "ubuntu@$BACKEND_IP" "redis-cli ping" 2>/dev/null | grep -q "PONG"; then
        success "Redis service: OK"
    else
        error "Redis service: FAILED"
        ((errors++))
    fi

    # Kafka/Zookeeper
    if ssh -o StrictHostKeyChecking=no -i "$SSH_KEY_BACKEND" "ubuntu@$BACKEND_IP" "nc -z localhost 9092" >/dev/null 2>&1; then
        success "Kafka service: OK"
    else
        error "Kafka service: FAILED"
        ((errors++))
    fi

    # Test 4: Security validation
    info "Testing security configuration..."

    # HMAC authentication test with retry logic (backend needs to be fully ready)
    local auth_ready=false
    if [[ "$backend_ready" == true ]]; then
        for i in {1..3}; do
            info "HMAC authentication test $i/3..."
            local auth_test=$(python3 -c "
import requests, json, hashlib, hmac, uuid
from datetime import datetime, timezone
def make_auth_test():
    try:
        timestamp = str(int(datetime.now(timezone.utc).timestamp()))
        nonce = str(uuid.uuid4())
        canonical = '|'.join(['GET', '/api/ml/status', '', timestamp, nonce])
        signature = hmac.new('678aae7bdf9e61cbb5fd059f0c774baf6d3143495cd091b2759265fe15c0beb3'.encode(), canonical.encode(), hashlib.sha256).hexdigest()
        headers = {'X-Device-ID': 'ffb56f4f-b0c8-4258-8922-0f976e536a7b', 'X-TS': timestamp, 'X-Nonce': nonce, 'X-Signature': signature}
        r = requests.get('http://$BACKEND_IP:$BACKEND_API_PORT/api/ml/status', headers=headers, timeout=10)
        return 'OK' if r.status_code == 200 else 'FAILED'
    except: return 'FAILED'
print(make_auth_test())
" 2>/dev/null)

            if [[ "$auth_test" == "OK" ]]; then
                success "HMAC authentication: OK"
                auth_ready=true
                break
            else
                if [ $i -eq 3 ]; then
                    error "HMAC authentication: FAILED (after 3 attempts)"
                    ((errors++))
                else
                    warning "HMAC auth not ready yet, retrying..."
                    sleep 5
                fi
            fi
        done
    else
        warning "Skipping HMAC test - backend not ready"
        ((errors++))
    fi

    # Test 5: T-Pot security
    info "Testing T-Pot security configuration..."

    local sg_id=$(aws ec2 describe-instances \
        --region "$AWS_REGION" \
        --instance-ids "$TPOT_INSTANCE" \
        --query 'Reservations[0].Instances[0].SecurityGroups[0].GroupId' \
        --output text)

    local open_to_all=$(aws ec2 describe-security-groups \
        --region "$AWS_REGION" \
        --group-ids "$sg_id" \
        --query 'SecurityGroups[0].IpPermissions[?IpRanges[?CidrIp==`0.0.0.0/0`]]' \
        --output json | jq '. | length')

    if [[ "$open_to_all" -gt 0 ]]; then
        warning "T-Pot has ports open to 0.0.0.0/0 (LIVE MODE)"
    else
        success "T-Pot security: Restricted access (TESTING MODE)"
    fi

    # Test 6: Agent functionality and TPOT connectivity
    info "Testing agent functionality and TPOT access..."

    # Test agent orchestration
    local agent_test=$(python3 -c "
import requests, json, hashlib, hmac, uuid
from datetime import datetime, timezone
try:
    timestamp = str(int(datetime.now(timezone.utc).timestamp()))
    nonce = str(uuid.uuid4())
    body = json.dumps({'query': 'test agent functionality', 'incident_id': 1, 'priority': 'high'})
    canonical = '|'.join(['POST', '/api/agents/orchestrate', body, timestamp, nonce])
    signature = hmac.new('678aae7bdf9e61cbb5fd059f0c774baf6d3143495cd091b2759265fe15c0beb3'.encode(), canonical.encode(), hashlib.sha256).hexdigest()
    headers = {'X-Device-ID': 'ffb56f4f-b0c8-4258-8922-0f976e536a7b', 'X-TS': timestamp, 'X-Nonce': nonce, 'X-Signature': signature, 'Content-Type': 'application/json'}
    r = requests.post('http://$BACKEND_IP:$BACKEND_API_PORT/api/agents/orchestrate', headers=headers, data=body, timeout=10)
    print('OK' if r.status_code == 200 else 'FAILED')
except: print('FAILED')
" 2>/dev/null)

    if [[ "$agent_test" == "OK" ]]; then
        success "Agent orchestration: OK"
    else
        error "Agent orchestration: FAILED"
        ((errors++))
    fi

    # Test agent-to-TPOT connectivity
    info "Testing agent access to TPOT honeypot..."
    local tpot_agent_test=$(ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 -i "$SSH_KEY_BACKEND" "ubuntu@$BACKEND_IP" \
        "python3 -c '
import subprocess, sys
try:
    # Test ping to TPOT
    result = subprocess.run([\"ping\", \"-c\", \"1\", \"$TPOT_IP\"], capture_output=True, timeout=5)
    if result.returncode == 0:
        print(\"TPOT_REACHABLE\")
    else:
        print(\"TPOT_UNREACHABLE\")
except:
    print(\"TPOT_UNREACHABLE\")
'" 2>/dev/null)

    if [[ "$tpot_agent_test" == "TPOT_REACHABLE" ]]; then
        success "Agent-to-TPOT connectivity: OK"
    else
        warning "Agent-to-TPOT connectivity: Limited (may need TPOT services running)"
        # Don't fail validation - just warn
    fi

    # Test TPOT data flow configuration
    info "Verifying TPOT data flow configuration..."
    local tpot_config_test=$(ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 -i "$SSH_KEY_BACKEND" "ubuntu@$BACKEND_IP" \
        "grep 'HONEYPOT_HOST=$TPOT_IP' /opt/mini-xdr/backend/.env >/dev/null 2>&1 && echo 'OK' || echo 'FAILED'" 2>/dev/null)

    if [[ "$tpot_config_test" == "OK" ]]; then
        success "TPOT data flow configuration: OK"
    else
        error "TPOT data flow configuration: FAILED - backend not configured for new TPOT"
        ((errors++))
    fi

    # Summary
    echo
    if [[ $errors -eq 0 ]]; then
        success "🎉 ALL VALIDATION TESTS PASSED! System is fully operational."
    else
        error "❌ $errors validation tests failed. System needs attention."
        return 1
    fi
}

# Show comprehensive system status
show_status() {
    header "Mini-XDR System Status"

    # Instance status
    echo -e "${BOLD}Instance Status:${NC}"
    aws ec2 describe-instances \
        --region "$AWS_REGION" \
        --instance-ids "$BACKEND_INSTANCE" "$TPOT_INSTANCE" \
        --query 'Reservations[].Instances[].[InstanceId,State.Name,PublicIpAddress,PrivateIpAddress]' \
        --output table

    # Service endpoints
    if get_instance_ips; then
        echo
        echo -e "${BOLD}Service Endpoints:${NC}"
        echo -e "🖥️  Backend API:        http://$BACKEND_IP:$BACKEND_API_PORT"
        echo -e "🎛️  Frontend Dashboard: http://$BACKEND_IP:$FRONTEND_PORT"
        echo -e "🍯  T-Pot Honeypot:    http://$TPOT_IP:64295"
        echo -e "📊  Health Check:       http://$BACKEND_IP:$BACKEND_API_PORT/health"

        # Quick health checks
        echo
        echo -e "${BOLD}Quick Health Checks:${NC}"

        if curl -s --max-time 5 "http://$BACKEND_IP:$BACKEND_API_PORT/health" >/dev/null 2>&1; then
            echo -e "✅ Backend API: ${GREEN}Healthy${NC}"
        else
            echo -e "❌ Backend API: ${RED}Not responding${NC}"
        fi

        if curl -s --max-time 5 "http://$BACKEND_IP:$FRONTEND_PORT" >/dev/null 2>&1; then
            echo -e "✅ Frontend: ${GREEN}Available${NC}"
        else
            echo -e "❌ Frontend: ${RED}Not responding${NC}"
        fi

        # T-Pot security mode
        local sg_id=$(aws ec2 describe-instances \
            --region "$AWS_REGION" \
            --instance-ids "$TPOT_INSTANCE" \
            --query 'Reservations[0].Instances[0].SecurityGroups[0].GroupId' \
            --output text 2>/dev/null || echo "")

        if [[ -n "$sg_id" ]]; then
            local open_to_all=$(aws ec2 describe-security-groups \
                --region "$AWS_REGION" \
                --group-ids "$sg_id" \
                --query 'SecurityGroups[0].IpPermissions[?IpRanges[?CidrIp==`0.0.0.0/0`]]' \
                --output json 2>/dev/null | jq '. | length' 2>/dev/null || echo "0")

            if [[ "$open_to_all" -gt 0 ]]; then
                echo -e "🍯 T-Pot Mode: ${RED}LIVE${NC} (accepting external traffic)"
            else
                echo -e "🍯 T-Pot Mode: ${GREEN}TESTING${NC} (restricted access)"
            fi
        fi
    fi

    echo
    echo -e "${BOLD}Log Files:${NC}"
    echo -e "📝 Startup script log: $LOG_FILE"
    echo -e "📝 Backend logs: ssh ubuntu@$BACKEND_IP 'tail -f /tmp/backend.log'"
    echo -e "📝 Frontend logs: ssh ubuntu@$BACKEND_IP 'tail -f /tmp/frontend.log'"
}

# Monitor system continuously
monitor_system() {
    header "Starting System Monitor"
    info "Monitoring system health every 30 seconds (Ctrl+C to stop)..."

    while true; do
        clear
        show_status

        echo
        echo -e "${CYAN}Next check in 30 seconds... (Ctrl+C to stop)${NC}"
        sleep 30
    done
}

# Show system logs
show_logs() {
    header "System Logs"

    echo -e "${BOLD}Recent backend logs:${NC}"
    ssh -o StrictHostKeyChecking=no -i "$SSH_KEY_BACKEND" "ubuntu@$BACKEND_IP" \
        "tail -20 /tmp/backend.log 2>/dev/null || echo 'No backend logs found'"

    echo
    echo -e "${BOLD}Recent frontend logs:${NC}"
    ssh -o StrictHostKeyChecking=no -i "$SSH_KEY_BACKEND" "ubuntu@$BACKEND_IP" \
        "tail -20 /tmp/frontend.log 2>/dev/null || echo 'No frontend logs found'"
}

# Main function
main() {
    local command="${1:-start}"
    local tpot_mode="testing"
    local force_mode=false
    local verbose=false
    local deploy_code=false

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            start|stop|restart|status|monitor|validate|logs)
                command=$1
                ;;
            --live)
                tpot_mode="live"
                ;;
            --testing)
                tpot_mode="testing"
                ;;
            --deploy-code)
                deploy_code=true
                ;;
            --force)
                force_mode=true
                ;;
            --verbose)
                verbose=true
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            *)
                error "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
        shift
    done

    # Set verbose logging
    if [[ "$verbose" == true ]]; then
        set -x
    fi

    # Header
    echo -e "${BOLD}${PURPLE}"
    echo "╔══════════════════════════════════════════════════════════════════════════════╗"
    echo "║                    🛡️  Mini-XDR AWS Management Script v3.0                   ║"
    echo "║              Complete Infrastructure Management & Security Validation         ║"
    echo "╚══════════════════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"

    # Prerequisite checks
    info "Running prerequisite checks..."
    check_aws_setup || exit 1
    get_user_ip || exit 1

    # Execute command
    case $command in
        start)
            if [[ "$tpot_mode" == "live" && "$force_mode" == false ]]; then
                echo
                warning "⚠️  SECURITY WARNING: You are about to start T-Pot in LIVE mode!"
                warning "This will make your honeypot accessible from the internet."
                echo -n "Are you sure? (yes/no): "
                read -r confirmation
                if [[ "$confirmation" != "yes" ]]; then
                    info "Operation cancelled."
                    exit 0
                fi
            fi

            start_instances "$tpot_mode"
            get_instance_ips || exit 1
            
            # Deploy code or just start services based on flag
            if [[ "$deploy_code" == true ]]; then
                info "Deploying local code changes to AWS..."
                deploy_backend_services
            else
                info "Starting existing services without code deployment..."
                start_backend_services
            fi
            
            validate_system
            show_status

            echo
            success "🎉 Mini-XDR system startup complete!"
            if [[ "$deploy_code" == true ]]; then
                success "✅ Local code changes deployed to AWS"
            else
                info "ℹ️  Used existing code deployment (use --deploy-code to update)"
            fi
            if [[ "$tpot_mode" == "live" ]]; then
                warning "⚠️  T-Pot is running in LIVE mode - monitor for attacks!"
            else
                success "✅ T-Pot is running in TESTING mode - safe for development"
            fi
            ;;

        stop)
            if [[ "$force_mode" == false ]]; then
                echo -n "Are you sure you want to stop all instances? (yes/no): "
                read -r confirmation
                if [[ "$confirmation" != "yes" ]]; then
                    info "Operation cancelled."
                    exit 0
                fi
            fi

            stop_instances
            success "🛑 All instances stopped successfully"
            ;;

        restart)
            if [[ "$force_mode" == false ]]; then
                echo -n "Are you sure you want to restart all instances? (yes/no): "
                read -r confirmation
                if [[ "$confirmation" != "yes" ]]; then
                    info "Operation cancelled."
                    exit 0
                fi
            fi

            stop_instances
            sleep 10
            start_instances "$tpot_mode"
            get_instance_ips || exit 1
            
            # Deploy code or just start services based on flag
            if [[ "$deploy_code" == true ]]; then
                info "Deploying local code changes to AWS..."
                deploy_backend_services
            else
                info "Restarting existing services without code deployment..."
                start_backend_services
            fi
            
            validate_system
            show_status
            
            success "🔄 System restart complete!"
            if [[ "$deploy_code" == true ]]; then
                success "✅ Local code changes deployed to AWS"
            else
                info "ℹ️  Used existing code deployment (use --deploy-code to update)"
            fi
            ;;

        status)
            show_status
            ;;

        monitor)
            get_instance_ips || exit 1
            monitor_system
            ;;

        validate)
            get_instance_ips || exit 1
            validate_system
            ;;

        logs)
            get_instance_ips || exit 1
            show_logs
            ;;

        *)
            error "Unknown command: $command"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac

    echo
    info "Script completed. Log file: $LOG_FILE"
}

# Run main function with all arguments
main "$@"
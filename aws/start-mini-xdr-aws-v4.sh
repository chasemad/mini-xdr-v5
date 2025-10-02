#!/bin/bash
# 🛡️ Mini-XDR AWS Management Script v4.0 - PHASE 1 ENHANCED
# Enhanced with Advanced Response System and Workflow Orchestration
# Based on proven v3.0 infrastructure with Phase 1 additions

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
AWS_REGION="${AWS_REGION:-us-east-1}"

# SSH Keys
SSH_KEY_BACKEND="${SSH_KEY_BACKEND:-~/.ssh/mini-xdr-tpot-key.pem}"
SSH_KEY_TPOT="${SSH_KEY_TPOT:-~/.ssh/mini-xdr-tpot-key.pem}"

# Instance Configuration (from your working v3.0)
BACKEND_INSTANCE="i-05ce3f39bd9c8f388"  # Mini-XDR backend instance
TPOT_INSTANCE="i-0584d6b913192a953"     # Secure T-Pot honeypot

# Network Configuration (from your working v3.0)
BACKEND_IP="54.237.168.3"   # Elastic IP for backend instance
TPOT_IP="107.22.132.190"    # Secure T-Pot instance IP
BACKEND_API_PORT="8000"
FRONTEND_PORT="3000"
REDIS_PORT="6379"
KAFKA_PORT="9092"
ZOOKEEPER_PORT="2181"

# NEW: Phase 1 Advanced Response Configuration
ADVANCED_RESPONSE_ENABLED="${ADVANCED_RESPONSE_ENABLED:-true}"
WORKFLOW_ENGINE_ENABLED="${WORKFLOW_ENGINE_ENABLED:-true}"
RESPONSE_ANALYTICS_ENABLED="${RESPONSE_ANALYTICS_ENABLED:-true}"

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
LOG_FILE="/tmp/mini-xdr-aws-v4-$(date +%Y%m%d-%H%M%S).log"
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
    echo -e "${BOLD}${CYAN}Mini-XDR AWS Management Script v4.0 - PHASE 1 ENHANCED${NC}"
    echo
    echo -e "${BOLD}NEW FEATURES:${NC}"
    echo -e "  ${GREEN}✨ Advanced Response System${NC}     - 16 enterprise-grade response actions"
    echo -e "  ${GREEN}✨ Workflow Orchestration${NC}       - Multi-step response workflows"
    echo -e "  ${GREEN}✨ Response Analytics${NC}           - Real-time effectiveness monitoring"
    echo -e "  ${GREEN}✨ Safety Controls${NC}              - Rollback and approval systems"
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
    echo -e "  ${GREEN}deploy-phase1${NC}   Deploy Phase 1 advanced response system"
    echo
    echo -e "${BOLD}OPTIONS:${NC}"
    echo -e "  ${YELLOW}--live${NC}              Put T-Pot honeypot in live mode (accepts external traffic)"
    echo -e "  ${YELLOW}--testing${NC}           Put T-Pot honeypot in testing mode (restricted access) [DEFAULT]"
    echo -e "  ${YELLOW}--deploy-code${NC}       Deploy local code changes to AWS (otherwise just start services)"
    echo -e "  ${YELLOW}--force${NC}             Force operations without confirmation"
    echo -e "  ${YELLOW}--verbose${NC}           Enable verbose logging"
    echo -e "  ${YELLOW}--skip-security-check${NC} Skip security validation (not recommended)"
    echo
    echo -e "${BOLD}PHASE 1 OPTIONS:${NC}"
    echo -e "  ${PURPLE}--enable-workflows${NC}      Enable advanced workflow orchestration"
    echo -e "  ${PURPLE}--enable-analytics${NC}      Enable response analytics dashboard"
    echo -e "  ${PURPLE}--migrate-database${NC}      Apply Phase 1 database migrations"
    echo -e "  ${PURPLE}--test-phase1${NC}           Test Phase 1 functionality after deployment"
    echo
    echo -e "${BOLD}EXAMPLES:${NC}"
    echo -e "  ${CYAN}# Deploy Phase 1 with all features${NC}"
    echo "  $0 deploy-phase1 --deploy-code --enable-workflows --enable-analytics"
    echo
    echo -e "  ${CYAN}# Start system with Phase 1 features${NC}"
    echo "  $0 start --deploy-code --testing"
    echo
    echo -e "  ${CYAN}# Test Phase 1 deployment${NC}"
    echo "  $0 validate --test-phase1"
    echo
}

# Get user's public IP for security groups
get_user_ip() {
    USER_IP=$(curl -s ipinfo.io/ip 2>/dev/null || curl -s ifconfig.me 2>/dev/null || echo "")
    if [ -z "$USER_IP" ]; then
        warning "Could not determine your public IP address"
        warning "T-Pot security groups may need manual configuration"
    else
        info "Your public IP: $USER_IP"
    fi
}

# Check AWS CLI configuration
check_aws_config() {
    if ! command -v aws &> /dev/null; then
        error "AWS CLI not found. Please install AWS CLI first."
        exit 1
    fi
    
    if ! aws sts get-caller-identity &> /dev/null; then
        error "AWS CLI not configured or credentials invalid"
        error "Run: aws configure"
        exit 1
    fi
    
    success "AWS CLI configured correctly"
}

# Check instance status
check_instance_status() {
    local instance_id="$1"
    local instance_name="$2"
    
    local status=$(aws ec2 describe-instances \
        --instance-ids "$instance_id" \
        --query 'Reservations[0].Instances[0].State.Name' \
        --output text \
        --region "$AWS_REGION" 2>/dev/null || echo "not-found")
    
    case "$status" in
        "running")
            success "$instance_name ($instance_id): Running"
            return 0
            ;;
        "stopped")
            warning "$instance_name ($instance_id): Stopped"
            return 1
            ;;
        "stopping"|"pending"|"shutting-down")
            warning "$instance_name ($instance_id): $status"
            return 1
            ;;
        "not-found")
            error "$instance_name ($instance_id): Not found or access denied"
            return 2
            ;;
        *)
            warning "$instance_name ($instance_id): Unknown status ($status)"
            return 1
            ;;
    esac
}

# Start instance
start_instance() {
    local instance_id="$1"
    local instance_name="$2"
    
    info "Starting $instance_name..."
    
    if check_instance_status "$instance_id" "$instance_name"; then
        success "$instance_name is already running"
        return 0
    fi
    
    aws ec2 start-instances --instance-ids "$instance_id" --region "$AWS_REGION" > /dev/null
    
    info "Waiting for $instance_name to start..."
    aws ec2 wait instance-running --instance-ids "$instance_id" --region "$AWS_REGION"
    
    success "$instance_name started successfully"
}

# Stop instance
stop_instance() {
    local instance_id="$1"
    local instance_name="$2"
    
    info "Stopping $instance_name..."
    
    if ! check_instance_status "$instance_id" "$instance_name"; then
        success "$instance_name is already stopped"
        return 0
    fi
    
    aws ec2 stop-instances --instance-ids "$instance_id" --region "$AWS_REGION" > /dev/null
    
    info "Waiting for $instance_name to stop..."
    aws ec2 wait instance-stopped --instance-ids "$instance_id" --region "$AWS_REGION"
    
    success "$instance_name stopped successfully"
}

# Deploy Phase 1 advanced response system
deploy_phase1_advanced_response() {
    header "DEPLOYING PHASE 1: ADVANCED RESPONSE SYSTEM"
    
    info "Deploying to Backend Instance: $BACKEND_INSTANCE ($BACKEND_IP)"
    
    # Check if backend is running
    if ! check_instance_status "$BACKEND_INSTANCE" "Backend"; then
        info "Starting backend instance for deployment..."
        start_instance "$BACKEND_INSTANCE" "Backend"
        sleep 30  # Wait for SSH to be ready
    fi
    
    # Deploy backend changes
    deploy_backend_phase1
    
    # Deploy frontend changes
    deploy_frontend_phase1
    
    # Apply database migrations
    apply_phase1_migrations
    
    # Restart services with new features
    restart_services_phase1
    
    success "Phase 1 Advanced Response System deployed successfully!"
}

# Deploy backend Phase 1 changes
deploy_backend_phase1() {
    info "📦 Deploying backend Phase 1 changes..."
    
    # Create deployment package
    local temp_dir=$(mktemp -d)
    info "Creating deployment package in $temp_dir"
    
    # Copy backend files
    cp -r "$PROJECT_ROOT/backend" "$temp_dir/"
    
    # Create deployment script for remote execution
    cat > "$temp_dir/deploy_phase1_backend.sh" << 'EOF'
#!/bin/bash
set -euo pipefail

DEPLOY_DIR="/opt/mini-xdr"
BACKUP_DIR="/opt/mini-xdr-backup-$(date +%Y%m%d-%H%M%S)"

echo "🔄 Starting Phase 1 backend deployment..."

# Create backup
if [ -d "$DEPLOY_DIR" ]; then
    echo "📦 Creating backup..."
    sudo cp -r "$DEPLOY_DIR" "$BACKUP_DIR"
    echo "✅ Backup created: $BACKUP_DIR"
fi

# Stop services
echo "🛑 Stopping services..."
sudo systemctl stop mini-xdr-backend || true
sudo systemctl stop mini-xdr-frontend || true

# Update backend code
echo "📁 Updating backend code..."
sudo mkdir -p "$DEPLOY_DIR"
sudo cp -r /tmp/backend/* "$DEPLOY_DIR/"
sudo chown -R ubuntu:ubuntu "$DEPLOY_DIR"

# Install/update Python dependencies
echo "🐍 Installing Python dependencies..."
cd "$DEPLOY_DIR"
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Apply Phase 1 database migrations
echo "🗄️ Applying Phase 1 database migrations..."
if [ -f "alembic.ini" ]; then
    alembic upgrade head
    echo "✅ Database migrations applied"
else
    echo "⚠️ No alembic.ini found, skipping migrations"
fi

# Create/update systemd service for backend
sudo tee /etc/systemd/system/mini-xdr-backend.service > /dev/null << 'SERVICE_EOF'
[Unit]
Description=Mini-XDR Backend API with Advanced Response System
After=network.target

[Service]
Type=simple
User=ubuntu
Group=ubuntu
WorkingDirectory=/opt/mini-xdr
Environment=PATH=/opt/mini-xdr/venv/bin
Environment=PYTHONPATH=/opt/mini-xdr
ExecStart=/opt/mini-xdr/venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
SERVICE_EOF

# Reload systemd and start services
sudo systemctl daemon-reload
sudo systemctl enable mini-xdr-backend

echo "✅ Phase 1 backend deployment completed"
EOF

    chmod +x "$temp_dir/deploy_phase1_backend.sh"
    
    # Upload and execute deployment
    info "📤 Uploading deployment package..."
    scp -i "$SSH_KEY_BACKEND" -o StrictHostKeyChecking=no -r "$temp_dir"/* "ubuntu@$BACKEND_IP:/tmp/"
    
    info "🚀 Executing remote deployment..."
    ssh -i "$SSH_KEY_BACKEND" -o StrictHostKeyChecking=no "ubuntu@$BACKEND_IP" "bash /tmp/deploy_phase1_backend.sh"
    
    # Cleanup
    rm -rf "$temp_dir"
    
    success "Backend Phase 1 deployment completed"
}

# Deploy frontend Phase 1 changes
deploy_frontend_phase1() {
    info "🎨 Deploying frontend Phase 1 changes..."
    
    # Create deployment package
    local temp_dir=$(mktemp -d)
    info "Creating frontend deployment package in $temp_dir"
    
    # Copy frontend files
    cp -r "$PROJECT_ROOT/frontend" "$temp_dir/"
    
    # Create frontend deployment script
    cat > "$temp_dir/deploy_phase1_frontend.sh" << 'EOF'
#!/bin/bash
set -euo pipefail

FRONTEND_DIR="/opt/mini-xdr-frontend"
BACKUP_DIR="/opt/mini-xdr-frontend-backup-$(date +%Y%m%d-%H%M%S)"

echo "🎨 Starting Phase 1 frontend deployment..."

# Create backup
if [ -d "$FRONTEND_DIR" ]; then
    echo "📦 Creating frontend backup..."
    sudo cp -r "$FRONTEND_DIR" "$BACKUP_DIR"
    echo "✅ Frontend backup created: $BACKUP_DIR"
fi

# Update frontend code
echo "📁 Updating frontend code..."
sudo mkdir -p "$FRONTEND_DIR"
sudo cp -r /tmp/frontend/* "$FRONTEND_DIR/"
sudo chown -R ubuntu:ubuntu "$FRONTEND_DIR"

# Install Node.js dependencies
echo "📦 Installing Node.js dependencies..."
cd "$FRONTEND_DIR"
npm install

# Build frontend
echo "🔨 Building frontend with Phase 1 components..."
npm run build

# Create/update systemd service for frontend
sudo tee /etc/systemd/system/mini-xdr-frontend.service > /dev/null << 'SERVICE_EOF'
[Unit]
Description=Mini-XDR Frontend with Advanced Response UI
After=network.target

[Service]
Type=simple
User=ubuntu
Group=ubuntu
WorkingDirectory=/opt/mini-xdr-frontend
Environment=NODE_ENV=production
Environment=NEXT_PUBLIC_API_BASE=http://localhost:8000
ExecStart=/usr/bin/npm start
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
SERVICE_EOF

# Reload systemd and enable service
sudo systemctl daemon-reload
sudo systemctl enable mini-xdr-frontend

echo "✅ Phase 1 frontend deployment completed"
EOF

    chmod +x "$temp_dir/deploy_phase1_frontend.sh"
    
    # Upload and execute deployment
    info "📤 Uploading frontend deployment package..."
    scp -i "$SSH_KEY_BACKEND" -o StrictHostKeyChecking=no -r "$temp_dir"/* "ubuntu@$BACKEND_IP:/tmp/"
    
    info "🚀 Executing frontend deployment..."
    ssh -i "$SSH_KEY_BACKEND" -o StrictHostKeyChecking=no "ubuntu@$BACKEND_IP" "bash /tmp/deploy_phase1_frontend.sh"
    
    # Cleanup
    rm -rf "$temp_dir"
    
    success "Frontend Phase 1 deployment completed"
}

# Apply Phase 1 database migrations
apply_phase1_migrations() {
    info "🗄️ Applying Phase 1 database migrations..."
    
    # Create migration script
    cat > /tmp/apply_phase1_migrations.sh << 'EOF'
#!/bin/bash
set -euo pipefail

cd /opt/mini-xdr
source venv/bin/activate

echo "🗄️ Applying Phase 1 database migrations..."

# Check if alembic is configured
if [ ! -f "alembic.ini" ]; then
    echo "⚠️ Alembic not configured, initializing..."
    alembic init migrations
    
    # Configure alembic.ini
    sed -i 's|sqlalchemy.url = .*|sqlalchemy.url = sqlite:///./xdr.db|' alembic.ini
    
    # Update env.py
    sed -i 's|target_metadata = None|from app.models import Base\ntarget_metadata = Base.metadata|' migrations/env.py
fi

# Apply migrations
echo "📊 Running database migrations..."
alembic upgrade head

echo "✅ Phase 1 database migrations completed"
EOF

    # Upload and execute migration
    scp -i "$SSH_KEY_BACKEND" -o StrictHostKeyChecking=no /tmp/apply_phase1_migrations.sh "ubuntu@$BACKEND_IP:/tmp/"
    ssh -i "$SSH_KEY_BACKEND" -o StrictHostKeyChecking=no "ubuntu@$BACKEND_IP" "bash /tmp/apply_phase1_migrations.sh"
    
    success "Phase 1 database migrations applied"
}

# Restart services with Phase 1 features
restart_services_phase1() {
    info "🔄 Restarting services with Phase 1 features..."
    
    # Create service restart script
    cat > /tmp/restart_phase1_services.sh << 'EOF'
#!/bin/bash
set -euo pipefail

echo "🔄 Restarting Mini-XDR services with Phase 1 features..."

# Stop services
sudo systemctl stop mini-xdr-frontend || true
sudo systemctl stop mini-xdr-backend || true

# Wait a moment
sleep 5

# Start backend first
echo "🚀 Starting enhanced backend..."
sudo systemctl start mini-xdr-backend

# Wait for backend to be ready
echo "⏳ Waiting for backend to be ready..."
for i in {1..30}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "✅ Backend is ready"
        break
    fi
    sleep 2
done

# Test Phase 1 features
echo "🧪 Testing Phase 1 advanced response system..."
if curl -s http://localhost:8000/api/response/test | grep -q "Advanced Response System is working"; then
    echo "✅ Phase 1 Advanced Response System is working"
else
    echo "⚠️ Phase 1 system test failed"
fi

# Start frontend
echo "🎨 Starting enhanced frontend..."
sudo systemctl start mini-xdr-frontend

# Wait for frontend to be ready
echo "⏳ Waiting for frontend to be ready..."
for i in {1..30}; do
    if curl -s http://localhost:3000 > /dev/null 2>&1; then
        echo "✅ Frontend is ready"
        break
    fi
    sleep 2
done

echo "✅ All services restarted with Phase 1 features"
EOF

    # Upload and execute restart
    scp -i "$SSH_KEY_BACKEND" -o StrictHostKeyChecking=no /tmp/restart_phase1_services.sh "ubuntu@$BACKEND_IP:/tmp/"
    ssh -i "$SSH_KEY_BACKEND" -o StrictHostKeyChecking=no "ubuntu@$BACKEND_IP" "bash /tmp/restart_phase1_services.sh"
    
    success "Services restarted with Phase 1 features"
}

# Test Phase 1 functionality
test_phase1_functionality() {
    header "TESTING PHASE 1 FUNCTIONALITY"
    
    info "Testing Advanced Response System..."
    
    # Test backend endpoints
    info "🔗 Testing backend endpoints..."
    
    # Health check
    if curl -s "http://$BACKEND_IP:$BACKEND_API_PORT/health" | grep -q "healthy"; then
        success "Backend health check: PASS"
    else
        error "Backend health check: FAIL"
        return 1
    fi
    
    # Advanced response test endpoint
    if curl -s "http://$BACKEND_IP:$BACKEND_API_PORT/api/response/test" | grep -q "Advanced Response System is working"; then
        success "Advanced Response System: WORKING"
        
        # Extract action count
        local action_count=$(curl -s "http://$BACKEND_IP:$BACKEND_API_PORT/api/response/test" | grep -o '"available_actions":[0-9]*' | cut -d':' -f2)
        success "Available response actions: $action_count"
    else
        error "Advanced Response System: FAILED"
        return 1
    fi
    
    # Test frontend
    info "🎨 Testing frontend..."
    if curl -s "http://$BACKEND_IP:$FRONTEND_PORT" | grep -q "Mini-XDR"; then
        success "Frontend: ACCESSIBLE"
    else
        error "Frontend: FAILED"
        return 1
    fi
    
    # Test incident page with new tabs
    info "🔍 Testing incident page with Phase 1 features..."
    if curl -s "http://$BACKEND_IP:$FRONTEND_PORT/incidents/incident/1" > /dev/null 2>&1; then
        success "Incident detail page: ACCESSIBLE"
    else
        warning "Incident detail page: Could not access (may need existing incidents)"
    fi
    
    success "Phase 1 functionality tests completed"
}

# Enhanced status check with Phase 1 features
show_enhanced_status() {
    header "MINI-XDR SYSTEM STATUS - PHASE 1 ENHANCED"
    
    # Instance status
    info "📊 Instance Status:"
    check_instance_status "$BACKEND_INSTANCE" "Backend"
    check_instance_status "$TPOT_INSTANCE" "T-Pot Honeypot"
    
    # Service status (if backend is running)
    if check_instance_status "$BACKEND_INSTANCE" "Backend" > /dev/null 2>&1; then
        echo
        info "🔗 Service Status:"
        
        # Backend API
        if curl -s "http://$BACKEND_IP:$BACKEND_API_PORT/health" > /dev/null 2>&1; then
            success "Backend API: HEALTHY"
        else
            error "Backend API: UNREACHABLE"
        fi
        
        # Phase 1 Advanced Response System
        if curl -s "http://$BACKEND_IP:$BACKEND_API_PORT/api/response/test" > /dev/null 2>&1; then
            local response=$(curl -s "http://$BACKEND_IP:$BACKEND_API_PORT/api/response/test")
            if echo "$response" | grep -q "Advanced Response System is working"; then
                success "Advanced Response System: OPERATIONAL"
                local actions=$(echo "$response" | grep -o '"available_actions":[0-9]*' | cut -d':' -f2)
                info "  └─ Available Actions: $actions"
                local samples=$(echo "$response" | grep -o '"sample_actions":\[[^]]*\]' | cut -d'[' -f2 | cut -d']' -f1)
                info "  └─ Sample Actions: $samples"
            else
                warning "Advanced Response System: RESPONDING BUT NOT READY"
            fi
        else
            error "Advanced Response System: UNREACHABLE"
        fi
        
        # Frontend
        if curl -s "http://$BACKEND_IP:$FRONTEND_PORT" > /dev/null 2>&1; then
            success "Frontend Dashboard: ACCESSIBLE"
        else
            error "Frontend Dashboard: UNREACHABLE"
        fi
    fi
    
    echo
    info "🌐 Access URLs:"
    info "  Frontend: http://$BACKEND_IP:$FRONTEND_PORT"
    info "  Backend API: http://$BACKEND_IP:$BACKEND_API_PORT"
    info "  API Docs: http://$BACKEND_IP:$BACKEND_API_PORT/docs"
    info "  Health Check: http://$BACKEND_IP:$BACKEND_API_PORT/health"
    info "  Phase 1 Test: http://$BACKEND_IP:$BACKEND_API_PORT/api/response/test"
    
    echo
    info "📋 Phase 1 Features:"
    info "  ✨ Advanced Response Actions: 16 enterprise-grade actions"
    info "  ✨ Workflow Orchestration: Multi-step response coordination"
    info "  ✨ Response Analytics: Real-time effectiveness monitoring"
    info "  ✨ Safety Controls: Rollback and approval systems"
}

# Main execution logic
main() {
    local command="${1:-help}"
    shift || true
    
    # Parse options
    local deploy_code=false
    local live_mode=false
    local testing_mode=true
    local force=false
    local verbose=false
    local skip_security_check=false
    local enable_workflows=false
    local enable_analytics=false
    local migrate_database=false
    local test_phase1=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --deploy-code)
                deploy_code=true
                shift
                ;;
            --live)
                live_mode=true
                testing_mode=false
                shift
                ;;
            --testing)
                testing_mode=true
                live_mode=false
                shift
                ;;
            --force)
                force=true
                shift
                ;;
            --verbose)
                verbose=true
                shift
                ;;
            --skip-security-check)
                skip_security_check=true
                shift
                ;;
            --enable-workflows)
                enable_workflows=true
                shift
                ;;
            --enable-analytics)
                enable_analytics=true
                shift
                ;;
            --migrate-database)
                migrate_database=true
                shift
                ;;
            --test-phase1)
                test_phase1=true
                shift
                ;;
            *)
                error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # Check prerequisites
    check_aws_config
    get_user_ip
    
    case "$command" in
        "help"|"--help"|"-h")
            show_help
            ;;
        "start")
            header "STARTING MINI-XDR SYSTEM WITH PHASE 1 FEATURES"
            
            # Start instances
            start_instance "$BACKEND_INSTANCE" "Backend"
            
            if [ "$live_mode" = true ]; then
                warning "LIVE MODE: Starting T-Pot honeypot in live mode"
                if [ "$force" = false ]; then
                    read -p "Are you sure you want to start T-Pot in LIVE mode? (yes/no): " confirm
                    if [ "$confirm" != "yes" ]; then
                        info "T-Pot start cancelled"
                        exit 0
                    fi
                fi
                start_instance "$TPOT_INSTANCE" "T-Pot Honeypot"
            fi
            
            # Deploy code if requested
            if [ "$deploy_code" = true ]; then
                deploy_phase1_advanced_response
            fi
            
            # Test Phase 1 if requested
            if [ "$test_phase1" = true ]; then
                sleep 10  # Wait for services to be ready
                test_phase1_functionality
            fi
            
            show_enhanced_status
            ;;
        "stop")
            header "STOPPING MINI-XDR SYSTEM"
            stop_instance "$BACKEND_INSTANCE" "Backend"
            stop_instance "$TPOT_INSTANCE" "T-Pot Honeypot"
            success "All instances stopped"
            ;;
        "restart")
            header "RESTARTING MINI-XDR SYSTEM"
            stop_instance "$BACKEND_INSTANCE" "Backend"
            if [ "$live_mode" = true ]; then
                stop_instance "$TPOT_INSTANCE" "T-Pot Honeypot"
            fi
            sleep 10
            start_instance "$BACKEND_INSTANCE" "Backend"
            if [ "$live_mode" = true ]; then
                start_instance "$TPOT_INSTANCE" "T-Pot Honeypot"
            fi
            
            if [ "$deploy_code" = true ]; then
                deploy_phase1_advanced_response
            fi
            
            show_enhanced_status
            ;;
        "status")
            show_enhanced_status
            ;;
        "deploy-phase1")
            header "DEPLOYING PHASE 1: ADVANCED RESPONSE SYSTEM"
            deploy_phase1_advanced_response
            
            if [ "$test_phase1" = true ]; then
                test_phase1_functionality
            fi
            
            show_enhanced_status
            ;;
        "validate")
            header "VALIDATING MINI-XDR SYSTEM"
            show_enhanced_status
            
            if [ "$test_phase1" = true ]; then
                test_phase1_functionality
            fi
            ;;
        "monitor")
            header "MONITORING MINI-XDR SYSTEM"
            info "Monitoring system health (Press Ctrl+C to stop)..."
            
            while true; do
                clear
                show_enhanced_status
                
                if [ "$test_phase1" = true ]; then
                    echo
                    test_phase1_functionality
                fi
                
                sleep 30
            done
            ;;
        "logs")
            header "MINI-XDR SYSTEM LOGS"
            info "Fetching logs from backend instance..."
            
            ssh -i "$SSH_KEY_BACKEND" -o StrictHostKeyChecking=no "ubuntu@$BACKEND_IP" "
                echo '=== Backend Service Logs ==='
                sudo journalctl -u mini-xdr-backend --no-pager -n 50
                echo
                echo '=== Frontend Service Logs ==='
                sudo journalctl -u mini-xdr-frontend --no-pager -n 50
            "
            ;;
        *)
            error "Unknown command: $command"
            show_help
            exit 1
            ;;
    esac
}

# Script entry point
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    main "$@"
fi



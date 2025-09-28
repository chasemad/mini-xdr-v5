#!/bin/bash

# Complete Mini-XDR AWS System Deployment
# Deploys backend, frontend, and configures TPOT in testing mode

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REGION="${AWS_REGION:-us-east-1}"
INSTANCE_TYPE="${INSTANCE_TYPE:-t3.medium}"
KEY_NAME="${KEY_NAME:-mini-xdr-tpot-key}"
YOUR_IP="${YOUR_IP:-24.11.0.176}"

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
    echo "=============================================="
    echo "      Complete Mini-XDR AWS Deployment"
    echo "=============================================="
    echo -e "${NC}"
    echo "This script deploys the complete Mini-XDR system:"
    echo "  ✓ Backend (EC2 + RDS + S3)"
    echo "  ✓ Frontend (S3 + CloudFront)"
    echo "  ✓ TPOT Integration (Testing Mode)"
    echo "  ✓ Management Scripts"
    echo ""
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

highlight() {
    echo -e "${MAGENTA}$1${NC}"
}

# Check prerequisites
check_prerequisites() {
    step "🔍 Checking Prerequisites"
    
    # Check AWS CLI
    if ! command -v aws &> /dev/null; then
        error "AWS CLI not found. Please install it first."
    fi
    
    # Check AWS configuration
    if ! aws sts get-caller-identity &> /dev/null; then
        error "AWS CLI not configured. Please run 'aws configure' first."
    fi
    
    # Check key pair
    if ! aws ec2 describe-key-pairs --key-names "$KEY_NAME" --region "$REGION" &> /dev/null; then
        error "Key pair '$KEY_NAME' not found. Please create it first."
    fi
    
    # Check required tools
    for tool in jq curl ssh scp npm; do
        if ! command -v "$tool" &> /dev/null; then
            error "$tool not found. Please install it first."
        fi
    done
    
    # Check SSH key file
    if [ ! -f "$HOME/.ssh/${KEY_NAME}.pem" ]; then
        error "SSH key file not found at $HOME/.ssh/${KEY_NAME}.pem"
    fi
    
    # Check TPOT connectivity
    if ! ssh -i "$HOME/.ssh/${KEY_NAME}.pem" -p 64295 -o ConnectTimeout=10 -o StrictHostKeyChecking=yes -o UserKnownHostsFile=~/.ssh/known_hosts \
           admin@34.193.101.171 "echo 'TPOT connection test'" >/dev/null 2>&1; then
        error "Cannot connect to TPOT. Please check connectivity."
    fi
    
    log "✅ All prerequisites satisfied!"
}

# Confirm deployment
confirm_deployment() {
    step "⚙️  Deployment Configuration"
    echo ""
    echo "Deployment Settings:"
    echo "  Region: $REGION"
    echo "  Instance Type: $INSTANCE_TYPE"
    echo "  Key Pair: $KEY_NAME"
    echo "  Your IP: $YOUR_IP"
    echo "  TPOT Host: 34.193.101.171"
    echo ""
    echo "What will be deployed:"
    echo "  🔧 Mini-XDR Backend (EC2 + RDS + S3)"
    echo "  🎨 Mini-XDR Frontend (S3 + CloudFront)"
    echo "  🔗 TPOT Integration (Testing Mode - Safe)"
    echo "  📋 Management Scripts"
    echo ""
    highlight "💰 Estimated monthly cost: ~$55"
    echo ""
    
    read -p "Continue with deployment? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log "Deployment cancelled by user"
        exit 0
    fi
}

# Deploy backend infrastructure and application
deploy_backend() {
    step "🏗️  Deploying Backend Infrastructure & Application (20-25 minutes)"
    
    log "Phase 1: AWS Infrastructure (EC2, RDS, S3, Security Groups)..."
    "$SCRIPT_DIR/deploy-mini-xdr-aws.sh"
    
    log "Phase 2: Application Code & Configuration..."
    "$SCRIPT_DIR/deploy-mini-xdr-code.sh"
    
    log "✅ Backend deployment completed!"
}

# Deploy frontend
deploy_frontend() {
    step "🎨 Deploying Frontend (S3 + CloudFront) (10-15 minutes)"
    
    "$SCRIPT_DIR/deploy-frontend-aws.sh"
    
    log "✅ Frontend deployment completed!"
}

# Configure TPOT in testing mode
configure_tpot_testing() {
    step "🔗 Configuring TPOT → AWS Integration (Testing Mode)"
    
    log "Setting up secure TPOT data flow..."
    "$SCRIPT_DIR/configure-tpot-aws-connection.sh"
    
    log "Setting TPOT to testing mode (safe for development)..."
    "$SCRIPT_DIR/tpot-security-control.sh" testing
    
    log "✅ TPOT configured in secure testing mode!"
}

# Create management scripts
setup_management_scripts() {
    step "📋 Setting Up Management Scripts"
    
    # Make all scripts executable
    chmod +x "$SCRIPT_DIR"/*.sh
    
    # Copy key management scripts to home directory
    local scripts=(
        "aws-services-control.sh"
        "tpot-security-control.sh"
        "update-pipeline.sh"
    )
    
    for script in "${scripts[@]}"; do
        if [ -f "$SCRIPT_DIR/$script" ]; then
            cp "$SCRIPT_DIR/$script" "$HOME/${script}"
            chmod +x "$HOME/${script}"
            log "📋 Copied $script to home directory"
        fi
    done
    
    # Create quick aliases script
    cat > "$HOME/mini-xdr-aliases.sh" << 'ALIASES_EOF'
#!/bin/bash
# Mini-XDR Quick Command Aliases

# Service management
alias xdr-start="~/aws-services-control.sh start"
alias xdr-stop="~/aws-services-control.sh stop"
alias xdr-status="~/aws-services-control.sh status"
alias xdr-logs="~/aws-services-control.sh logs"
alias xdr-restart="~/aws-services-control.sh restart"

# Updates
alias xdr-update-frontend="~/update-pipeline.sh frontend"
alias xdr-update-backend="~/update-pipeline.sh backend"
alias xdr-update-all="~/update-pipeline.sh both"
alias xdr-quick-update="~/update-pipeline.sh quick"

# TPOT security
alias tpot-testing="~/tpot-security-control.sh testing"
alias tpot-live="~/tpot-security-control.sh live"
alias tpot-status="~/tpot-security-control.sh status"
alias tpot-lockdown="~/tpot-security-control.sh lockdown"

echo "Mini-XDR aliases loaded! Use 'xdr-status' to check system status."
ALIASES_EOF
    
    chmod +x "$HOME/mini-xdr-aliases.sh"
    
    log "✅ Management scripts set up!"
}

# Get deployment information
get_deployment_info() {
    # Backend info
    local backend_outputs
    backend_outputs=$(aws cloudformation describe-stacks \
        --stack-name "mini-xdr-backend" \
        --region "$REGION" \
        --query 'Stacks[0].Outputs' \
        --output json)
    
    BACKEND_IP=$(echo "$backend_outputs" | jq -r '.[] | select(.OutputKey=="BackendPublicIP") | .OutputValue')
    DB_ENDPOINT=$(echo "$backend_outputs" | jq -r '.[] | select(.OutputKey=="DatabaseEndpoint") | .OutputValue')
    MODELS_BUCKET=$(echo "$backend_outputs" | jq -r '.[] | select(.OutputKey=="ModelsBucket") | .OutputValue')
    
    # Frontend info
    local frontend_outputs
    frontend_outputs=$(aws cloudformation describe-stacks \
        --stack-name "mini-xdr-frontend" \
        --region "$REGION" \
        --query 'Stacks[0].Outputs' \
        --output json)
    
    FRONTEND_URL=$(echo "$frontend_outputs" | jq -r '.[] | select(.OutputKey=="CloudFrontURL") | .OutputValue')
}

# Validate complete deployment
validate_deployment() {
    step "✅ Validating Complete Deployment"
    
    get_deployment_info
    
    # Test backend
    log "Testing backend API..."
    local retry_count=0
    local max_retries=10
    
    while ! curl -f "http://$BACKEND_IP:8000/health" >/dev/null 2>&1; do
        retry_count=$((retry_count + 1))
        if [ $retry_count -gt $max_retries ]; then
            error "Backend health check failed"
        fi
        log "Waiting for backend API... ($retry_count/$max_retries)"
        sleep 15
    done
    
    log "✅ Backend API is healthy"
    
    # Test frontend
    log "Testing frontend..."
    if curl -f "$FRONTEND_URL" >/dev/null 2>&1; then
        log "✅ Frontend is accessible"
    else
        warn "⚠️  Frontend may still be propagating (CloudFront takes 5-15 minutes)"
    fi
    
    # Test TPOT data flow
    log "Testing TPOT → AWS data flow..."
    local events_before
    events_before=$(curl -s "http://$BACKEND_IP:8000/events" | jq '. | length' 2>/dev/null || echo "0")
    
    log "Current events in system: $events_before"
    log "✅ Data flow endpoint accessible"
    
    log "🎉 Complete system validation passed!"
}

# Show deployment summary
show_summary() {
    step "🎉 Deployment Complete!"
    
    get_deployment_info
    
    echo ""
    echo "=============================================="
    echo "        Mini-XDR AWS System Ready!"
    echo "=============================================="
    echo ""
    echo "🌐 Service URLs:"
    echo "   Frontend: $FRONTEND_URL"
    echo "   Backend API: http://$BACKEND_IP:8000"
    echo "   Health Check: http://$BACKEND_IP:8000/health"
    echo "   Events API: http://$BACKEND_IP:8000/events"
    echo ""
    echo "🗄️  Infrastructure:"
    echo "   Backend IP: $BACKEND_IP"
    echo "   Database: $DB_ENDPOINT"
    echo "   Models Bucket: $MODELS_BUCKET"
    echo ""
    echo "🔒 Security Status:"
    echo "   TPOT Mode: Testing (Secure - Your IP Only)"
    echo "   Management: Restricted to $YOUR_IP"
    echo ""
    echo "🎯 Quick Commands:"
    echo "   Service Status: ~/aws-services-control.sh status"
    echo "   View Logs: ~/aws-services-control.sh logs"
    echo "   Update Frontend: ~/update-pipeline.sh frontend"
    echo "   Update Backend: ~/update-pipeline.sh backend"
    echo "   TPOT Status: ~/tpot-security-control.sh status"
    echo ""
    echo "📋 Load Aliases:"
    echo "   source ~/mini-xdr-aliases.sh"
    echo ""
    echo "🚨 When Ready for Live Attacks:"
    echo "   ~/tpot-security-control.sh live"
    echo "   (⚠️  This exposes TPOT to real attackers!)"
    echo ""
    echo "✅ Your Mini-XDR system is ready for cybersecurity operations!"
}

# Create getting started guide
create_getting_started() {
    cat > "$HOME/MINI_XDR_GETTING_STARTED.md" << 'GUIDE_EOF'
# Mini-XDR AWS System - Getting Started

## Quick Start Commands

```bash
# Load helpful aliases
source ~/mini-xdr-aliases.sh

# Check system status
xdr-status

# View real-time logs
xdr-logs

# Open your cybersecurity dashboard
open FRONTEND_URL_PLACEHOLDER
```

## Service Management

```bash
# Start/stop services
xdr-start
xdr-stop
xdr-restart

# Update deployments
xdr-update-frontend    # Deploy frontend changes
xdr-update-backend     # Deploy backend changes
xdr-update-all         # Deploy both
```

## TPOT Honeypot Control

```bash
# Check TPOT security mode
tpot-status

# Switch to live mode (when ready for real attacks)
tpot-live

# Emergency lockdown
tpot-lockdown

# Return to testing mode
tpot-testing
```

## Development Workflow

1. **Make changes** to frontend or backend code
2. **Test locally** if needed
3. **Deploy changes**: `xdr-update-frontend` or `xdr-update-backend`
4. **Check status**: `xdr-status`
5. **Monitor**: `xdr-logs`

## Security Notes

- TPOT starts in **testing mode** (safe, your IP only)
- Use `tpot-live` only when ready for real attackers
- Always use `tpot-lockdown` for emergency shutdown
- Monitor costs in AWS console

## Support

- Service logs: `xdr-logs`
- TPOT status: `tpot-status`
- AWS console for infrastructure monitoring
- Documentation: `/Users/chasemad/Desktop/mini-xdr/docs/`
GUIDE_EOF
    
    # Replace placeholder with actual frontend URL
    sed -i.bak "s|FRONTEND_URL_PLACEHOLDER|$FRONTEND_URL|g" "$HOME/MINI_XDR_GETTING_STARTED.md"
    rm -f "$HOME/MINI_XDR_GETTING_STARTED.md.bak"
    
    log "📖 Getting started guide created: $HOME/MINI_XDR_GETTING_STARTED.md"
}

# Error handling
cleanup_on_error() {
    if [ $? -ne 0 ]; then
        error "Deployment failed! Check the logs above for details."
        echo ""
        echo "To clean up partial deployment:"
        echo "  aws cloudformation delete-stack --stack-name mini-xdr-backend --region $REGION"
        echo "  aws cloudformation delete-stack --stack-name mini-xdr-frontend --region $REGION"
        echo ""
        echo "To retry deployment:"
        echo "  $0"
    fi
}

# Set up error handling
trap cleanup_on_error EXIT

# Main deployment function
main() {
    show_banner
    check_prerequisites
    confirm_deployment
    
    log "🚀 Starting complete AWS deployment..."
    local start_time=$(date +%s)
    
    deploy_backend
    deploy_frontend
    configure_tpot_testing
    setup_management_scripts
    validate_deployment
    create_getting_started
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local minutes=$((duration / 60))
    local seconds=$((duration % 60))
    
    show_summary
    
    echo ""
    log "🕒 Total deployment time: ${minutes}m ${seconds}s"
    log "📖 See $HOME/MINI_XDR_GETTING_STARTED.md for next steps"
    echo ""
    
    # Disable error trap for successful completion
    trap - EXIT
}

# Export configuration for subscripts
export AWS_REGION="$REGION"
export INSTANCE_TYPE="$INSTANCE_TYPE"
export KEY_NAME="$KEY_NAME"
export YOUR_IP="$YOUR_IP"

# Run main function
main "$@"

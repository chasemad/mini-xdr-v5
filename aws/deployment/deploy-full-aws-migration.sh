#!/bin/bash

# Mini-XDR Full AWS Migration Script
# This script orchestrates the complete migration of Mini-XDR to AWS

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
NC='\033[0m' # No Color

# Banner
show_banner() {
    echo -e "${CYAN}"
    echo "=================================="
    echo "   Mini-XDR AWS Migration"
    echo "=================================="
    echo -e "${NC}"
    echo "This script will migrate your complete Mini-XDR system to AWS"
    echo ""
    echo "Components to be deployed:"
    echo "  ✓ Mini-XDR Backend (EC2 + RDS)"
    echo "  ✓ ML Models (S3)"
    echo "  ✓ Direct TPOT → AWS connection"
    echo "  ✓ Security groups & IAM roles"
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

# Check prerequisites
check_prerequisites() {
    step "🔍 Checking prerequisites..."
    
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
    for tool in jq curl ssh scp; do
        if ! command -v "$tool" &> /dev/null; then
            error "$tool not found. Please install it first."
        fi
    done
    
    # Check SSH key file
    if [ ! -f "$HOME/.ssh/${KEY_NAME}.pem" ]; then
        error "SSH key file not found at $HOME/.ssh/${KEY_NAME}.pem"
    fi
    
    log "✅ Prerequisites check passed!"
}

# Confirm configuration
confirm_configuration() {
    step "⚙️  Configuration Review"
    echo ""
    echo "Deployment Configuration:"
    echo "  Region: $REGION"
    echo "  Instance Type: $INSTANCE_TYPE"
    echo "  Key Pair: $KEY_NAME"
    echo "  Your IP: $YOUR_IP"
    echo "  TPOT Host: 34.193.101.171"
    echo ""
    
    read -p "Continue with this configuration? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log "Deployment cancelled by user"
        exit 0
    fi
}

# Deploy AWS infrastructure
deploy_infrastructure() {
    step "🏗️  Deploying AWS Infrastructure (15-20 minutes)"
    log "Creating EC2 instance, RDS database, security groups..."
    
    "$SCRIPT_DIR/deploy-mini-xdr-aws.sh"
    
    if [ $? -eq 0 ]; then
        log "✅ Infrastructure deployment completed!"
    else
        error "❌ Infrastructure deployment failed!"
    fi
}

# Deploy application code
deploy_application() {
    step "📦 Deploying Application Code (10-15 minutes)"
    log "Uploading Mini-XDR backend, models, and configuration..."
    
    "$SCRIPT_DIR/deploy-mini-xdr-code.sh"
    
    if [ $? -eq 0 ]; then
        log "✅ Application deployment completed!"
    else
        error "❌ Application deployment failed!"
    fi
}

# Configure TPOT connection
configure_tpot() {
    step "🔗 Configuring TPOT → AWS Connection"
    log "Setting up direct data flow from TPOT to AWS Mini-XDR..."
    
    "$SCRIPT_DIR/configure-tpot-aws-connection.sh"
    
    if [ $? -eq 0 ]; then
        log "✅ TPOT connection configured!"
    else
        error "❌ TPOT connection configuration failed!"
    fi
}

# Get deployment information
get_deployment_info() {
    step "📋 Getting Deployment Information"
    
    local outputs
    outputs=$(aws cloudformation describe-stacks \
        --stack-name "mini-xdr-backend" \
        --region "$REGION" \
        --query 'Stacks[0].Outputs' \
        --output json 2>/dev/null || echo "[]")
    
    if [ "$outputs" != "[]" ]; then
        BACKEND_IP=$(echo "$outputs" | jq -r '.[] | select(.OutputKey=="BackendPublicIP") | .OutputValue' 2>/dev/null || echo "")
        INSTANCE_ID=$(echo "$outputs" | jq -r '.[] | select(.OutputKey=="BackendInstanceId") | .OutputValue' 2>/dev/null || echo "")
        DB_ENDPOINT=$(echo "$outputs" | jq -r '.[] | select(.OutputKey=="DatabaseEndpoint") | .OutputValue' 2>/dev/null || echo "")
        MODELS_BUCKET=$(echo "$outputs" | jq -r '.[] | select(.OutputKey=="ModelsBucket") | .OutputValue' 2>/dev/null || echo "")
    fi
}

# Validate deployment
validate_deployment() {
    step "✅ Validating Deployment"
    
    get_deployment_info
    
    if [ -z "$BACKEND_IP" ]; then
        error "Could not retrieve backend IP address"
    fi
    
    log "Testing Mini-XDR API..."
    local health_url="http://$BACKEND_IP:8000/health"
    local retry_count=0
    local max_retries=10
    
    while ! curl -f "$health_url" >/dev/null 2>&1; do
        retry_count=$((retry_count + 1))
        if [ $retry_count -gt $max_retries ]; then
            error "Health check failed after $max_retries attempts"
        fi
        log "Waiting for API to be ready... ($retry_count/$max_retries)"
        sleep 15
    done
    
    log "✅ Mini-XDR API is responding!"
    
    # Test events endpoint
    local events_url="http://$BACKEND_IP:8000/events"
    if curl -f "$events_url" >/dev/null 2>&1; then
        log "✅ Events API is accessible!"
    else
        warn "⚠️  Events API may not be ready yet"
    fi
}

# Show deployment summary
show_summary() {
    step "🎉 Deployment Summary"
    
    get_deployment_info
    
    echo ""
    echo "=================================="
    echo "   AWS Migration Completed!"
    echo "=================================="
    echo ""
    echo "🌐 Mini-XDR Backend:"
    echo "   IP Address: $BACKEND_IP"
    echo "   API Endpoint: http://$BACKEND_IP:8000"
    echo "   Health Check: http://$BACKEND_IP:8000/health"
    echo "   Events API: http://$BACKEND_IP:8000/events"
    echo ""
    echo "🗄️  Database:"
    echo "   Type: PostgreSQL (RDS)"
    echo "   Endpoint: $DB_ENDPOINT"
    echo ""
    echo "💾 Storage:"
    echo "   Models Bucket: $MODELS_BUCKET"
    echo ""
    echo "🔗 Data Flow:"
    echo "   TPOT (34.193.101.171) → Mini-XDR ($BACKEND_IP:8000)"
    echo ""
    echo "🔧 Management Commands:"
    echo "   SSH: ssh -i ~/.ssh/${KEY_NAME}.pem ubuntu@$BACKEND_IP"
    echo "   Logs: sudo journalctl -u mini-xdr -f"
    echo "   Restart: sudo systemctl restart mini-xdr"
    echo "   Monitor: ~/monitor-tpot-connection.sh $BACKEND_IP"
    echo ""
    echo "🎯 Next Steps:"
    echo "   1. Update your local frontend to use: http://$BACKEND_IP:8000"
    echo "   2. Configure API keys in the environment file"
    echo "   3. Monitor data flow using the monitoring script"
    echo "   4. View real attack data on the globe visualization"
    echo ""
    echo "✅ Your Mini-XDR system is now fully deployed on AWS!"
}

# Create frontend configuration
create_frontend_config() {
    step "⚙️  Creating Frontend Configuration"
    
    get_deployment_info
    
    local frontend_env_file="/Users/chasemad/Desktop/mini-xdr/frontend/env.local.aws"
    
    cat > "$frontend_env_file" << EOF
# AWS Mini-XDR Backend Configuration
NEXT_PUBLIC_API_URL=http://$BACKEND_IP:8000
NEXT_PUBLIC_WS_URL=ws://$BACKEND_IP:8000/ws
NEXT_PUBLIC_ENV=aws

# Optional: For development with HTTPS
# NEXT_PUBLIC_API_URL=https://$BACKEND_IP:8000
# NEXT_PUBLIC_WS_URL=wss://$BACKEND_IP:8000/ws
EOF
    
    log "Frontend configuration created: $frontend_env_file"
    log "To use AWS backend, copy this file to frontend/.env.local"
}

# Error handling
cleanup_on_error() {
    if [ $? -ne 0 ]; then
        error "Deployment failed! Check the logs above for details."
        echo ""
        echo "To clean up partial deployment:"
        echo "  aws cloudformation delete-stack --stack-name mini-xdr-backend --region $REGION"
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
    confirm_configuration
    
    log "🚀 Starting AWS migration..."
    local start_time=$(date +%s)
    
    deploy_infrastructure
    deploy_application
    configure_tpot
    validate_deployment
    create_frontend_config
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local minutes=$((duration / 60))
    local seconds=$((duration % 60))
    
    show_summary
    
    echo ""
    log "🕒 Total deployment time: ${minutes}m ${seconds}s"
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

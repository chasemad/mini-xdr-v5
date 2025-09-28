#!/bin/bash

# SECURE Mini-XDR AWS Deployment Script
# Deploys with security built-in from the start

set -euo pipefail

# Configuration
REGION="${AWS_REGION:-us-east-1}"
STACK_NAME="mini-xdr-secure"
YOUR_IP="${YOUR_IP:-$(curl -s ipinfo.io/ip)}"
KEY_NAME="${KEY_NAME:-mini-xdr-tpot-key}"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

log() { echo -e "${GREEN}[$(date +'%H:%M:%S')] $1${NC}"; }
warn() { echo -e "${YELLOW}[WARN] $1${NC}"; }
error() { echo -e "${RED}[ERROR] $1${NC}"; exit 1; }
step() { echo -e "${BLUE}$1${NC}"; }

show_banner() {
    echo -e "${BLUE}"
    echo "=============================================="
    echo "    🛡️ SECURE Mini-XDR Deployment 🛡️"
    echo "=============================================="
    echo -e "${NC}"
    echo "This will deploy Mini-XDR with security built-in:"
    echo "  ✅ No 0.0.0.0/0 network exposures"
    echo "  ✅ Encrypted database with secure passwords"
    echo "  ✅ Least-privilege IAM policies"
    echo "  ✅ Credentials in AWS Secrets Manager"
    echo ""
}

check_prerequisites() {
    step "🔍 Checking Prerequisites"
    
    if ! command -v aws &> /dev/null; then
        error "AWS CLI not found. Please install it first."
    fi
    
    if ! aws sts get-caller-identity &> /dev/null; then
        error "AWS CLI not configured. Please run 'aws configure' first."
    fi
    
    if ! aws ec2 describe-key-pairs --key-names "$KEY_NAME" &> /dev/null; then
        error "Key pair '$KEY_NAME' not found. Please create it first."
    fi
    
    log "✅ Prerequisites check passed"
}

generate_secure_database_password() {
    step "🔐 Generating Secure Database Password"
    
    # Generate cryptographically secure password
    SECURE_DB_PASSWORD=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-25)
    SECURE_DB_PASSWORD="MiniXDR_${SECURE_DB_PASSWORD}_2025"
    
    # Store in Secrets Manager
    aws secretsmanager create-secret \
        --name "mini-xdr/database-password" \
        --description "Mini-XDR secure database password" \
        --secret-string "$SECURE_DB_PASSWORD" \
        --region "$REGION" 2>/dev/null || \
    aws secretsmanager update-secret \
        --secret-id "mini-xdr/database-password" \
        --secret-string "$SECURE_DB_PASSWORD" \
        --region "$REGION"
    
    log "✅ Secure database password generated and stored"
}

deploy_secure_infrastructure() {
    step "🏗️ Deploying Secure Infrastructure"
    
    log "Deploying secure CloudFormation stack..."
    aws cloudformation deploy \
        --template-file "$(dirname "$0")/deployment/secure-mini-xdr-aws.yaml" \
        --stack-name "$STACK_NAME" \
        --parameter-overrides \
            KeyPairName="$KEY_NAME" \
            YourPublicIP="$YOUR_IP" \
        --capabilities CAPABILITY_IAM \
        --region "$REGION"
    
    log "✅ Secure infrastructure deployed"
}

show_deployment_summary() {
    step "📊 Secure Deployment Summary"
    
    local outputs
    outputs=$(aws cloudformation describe-stacks \
        --stack-name "$STACK_NAME" \
        --region "$REGION" \
        --query 'Stacks[0].Outputs' \
        --output json)
    
    local backend_ip
    backend_ip=$(echo "$outputs" | jq -r '.[] | select(.OutputKey=="BackendPublicIP") | .OutputValue')
    
    echo ""
    echo "=============================================="
    echo "   🛡️ SECURE Mini-XDR Deployment Complete!"
    echo "=============================================="
    echo ""
    echo "🔒 Security Features Enabled:"
    echo "  ✅ Network access restricted to your IP: $YOUR_IP"
    echo "  ✅ Database encrypted with secure password"
    echo "  ✅ No 0.0.0.0/0 security group rules"
    echo "  ✅ Credentials stored in AWS Secrets Manager"
    echo "  ✅ Least-privilege IAM policies"
    echo ""
    echo "🌐 Access Information:"
    echo "  Backend IP: $backend_ip"
    echo "  SSH: ssh -i ~/.ssh/$KEY_NAME.pem ubuntu@$backend_ip"
    echo "  API: http://$backend_ip:8000 (restricted to your IP)"
    echo ""
    echo "🔑 Next Steps:"
    echo "  1. SSH to backend and configure your API keys"
    echo "  2. Deploy application code with: ./deploy-mini-xdr-code.sh"
    echo "  3. Test all functionality"
    echo "  4. Deploy frontend if needed"
    echo ""
    echo "✅ Your Mini-XDR is now SECURELY deployed!"
}

main() {
    show_banner
    
    echo "This will deploy Mini-XDR with SECURITY BUILT-IN."
    echo "Your IP ($YOUR_IP) will be the only one with access."
    echo ""
    read -p "Continue with secure deployment? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log "Deployment cancelled"
        exit 0
    fi
    
    check_prerequisites
    generate_secure_database_password
    deploy_secure_infrastructure
    show_deployment_summary
}

export AWS_REGION="$REGION"
export YOUR_IP="$YOUR_IP"
export KEY_NAME="$KEY_NAME"

main "$@"

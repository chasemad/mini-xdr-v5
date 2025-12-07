#!/bin/bash

# Mini-XDR Code Deployment Script
# Uploads and configures the Mini-XDR backend on AWS

set -euo pipefail

# Configuration
REGION="${AWS_REGION:-us-east-1}"
STACK_NAME="mini-xdr-backend"
PROJECT_DIR="."
KEY_NAME="${KEY_NAME:-mini-xdr-tpot-key}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# Get stack outputs
get_stack_info() {
    log "Getting stack information..."

    if ! aws cloudformation describe-stacks --stack-name "$STACK_NAME" --region "$REGION" >/dev/null 2>&1; then
        error "Stack '$STACK_NAME' not found. Please run './deploy-mini-xdr-aws.sh' first."
    fi

    local outputs
    outputs=$(aws cloudformation describe-stacks \
        --stack-name "$STACK_NAME" \
        --region "$REGION" \
        --query 'Stacks[0].Outputs' \
        --output json)

    BACKEND_IP=$(echo "$outputs" | jq -r '.[] | select(.OutputKey=="BackendPublicIP") | .OutputValue')
    INSTANCE_ID=$(echo "$outputs" | jq -r '.[] | select(.OutputKey=="BackendInstanceId") | .OutputValue')
    DB_ENDPOINT=$(echo "$outputs" | jq -r '.[] | select(.OutputKey=="DatabaseEndpoint") | .OutputValue')
    MODELS_BUCKET=$(echo "$outputs" | jq -r '.[] | select(.OutputKey=="ModelsBucket") | .OutputValue')

    log "Backend IP: $BACKEND_IP"
    log "Instance ID: $INSTANCE_ID"
    log "Database: $DB_ENDPOINT"
    log "Models Bucket: $MODELS_BUCKET"
}

# Wait for instance to be ready
wait_for_instance() {
    log "Waiting for instance to be ready..."

    aws ec2 wait instance-status-ok --instance-ids "$INSTANCE_ID" --region "$REGION"

    # Additional wait for SSH to be available
    local retry_count=0
    local max_retries=30

    while ! ssh -i "~/.ssh/${KEY_NAME}.pem" -o ConnectTimeout=5 -o StrictHostKeyChecking=yes -o UserKnownHostsFile=~/.ssh/known_hosts ubuntu@"$BACKEND_IP" "echo 'SSH Ready'" >/dev/null 2>&1; do
        retry_count=$((retry_count + 1))
        if [ $retry_count -gt $max_retries ]; then
            error "SSH connection timeout after $max_retries attempts"
        fi
        log "Waiting for SSH to be available... ($retry_count/$max_retries)"
        sleep 10
    done

    log "Instance is ready!"
}

# Create deployment package
create_deployment_package() {
    log "Creating deployment package..."

    local temp_dir="/tmp/mini-xdr-deploy"
    rm -rf "$temp_dir"
    mkdir -p "$temp_dir"

    # Copy backend code
    cp -r "$PROJECT_DIR/backend" "$temp_dir/"

    # Copy models if they exist
    if [ -d "$PROJECT_DIR/models" ]; then
        cp -r "$PROJECT_DIR/models" "$temp_dir/"
    fi

    # Copy policies
    if [ -d "$PROJECT_DIR/policies" ]; then
        cp -r "$PROJECT_DIR/policies" "$temp_dir/"
    fi

    # Copy datasets for training
    if [ -d "$PROJECT_DIR/datasets" ]; then
        cp -r "$PROJECT_DIR/datasets" "$temp_dir/"
    fi

    # Create tar archive
    cd "$temp_dir"
    tar -czf "/tmp/mini-xdr-backend.tar.gz" .

    log "Deployment package created: /tmp/mini-xdr-backend.tar.gz"
}

# Upload code to EC2
upload_code() {
    log "Uploading code to EC2 instance..."

    # Upload deployment package
    scp -i "~/.ssh/${KEY_NAME}.pem" -o StrictHostKeyChecking=yes -o UserKnownHostsFile=~/.ssh/known_hosts \
        "/tmp/mini-xdr-backend.tar.gz" ubuntu@"$BACKEND_IP":/tmp/

    # Extract and setup
    ssh -i "~/.ssh/${KEY_NAME}.pem" -o StrictHostKeyChecking=yes -o UserKnownHostsFile=~/.ssh/known_hosts ubuntu@"$BACKEND_IP" << 'REMOTE_SETUP'
        set -euo pipefail

        # Extract code
        cd /opt/mini-xdr
        sudo tar -xzf /tmp/mini-xdr-backend.tar.gz
        sudo chown -R ubuntu:ubuntu /opt/mini-xdr

        # Install Python dependencies
        source venv/bin/activate
        cd backend
        pip install --upgrade pip
        pip install -r requirements.txt

        echo "Code upload and setup completed!"
REMOTE_SETUP

    log "Code uploaded successfully!"
}

# Upload models to S3
upload_models_to_s3() {
    log "Uploading ML models to S3..."

    if [ -d "$PROJECT_DIR/models" ]; then
        aws s3 sync "$PROJECT_DIR/models" "s3://$MODELS_BUCKET/models/" --region "$REGION"
        log "Models uploaded to S3 bucket: $MODELS_BUCKET"
    else
        warn "No models directory found, skipping model upload"
    fi
}

# Configure environment
configure_environment() {
    log "Configuring environment variables..."

    # Get database password from CloudFormation
    local stack_id
    stack_id=$(aws cloudformation describe-stacks \
        --stack-name "$STACK_NAME" \
        --region "$REGION" \
        --query 'Stacks[0].StackId' \
        --output text)

    # Get secure database password from AWS Secrets Manager
    local db_password=$(aws secretsmanager get-secret-value \
        --secret-id mini-xdr/database-password \
        --query SecretString \
        --output text 2>/dev/null || echo "CONFIGURE_IN_SECRETS_MANAGER")
    local db_url="postgresql://postgres:${db_password}@${DB_ENDPOINT}:5432/postgres"

    # Create comprehensive environment file
    ssh -i "~/.ssh/${KEY_NAME}.pem" -o StrictHostKeyChecking=yes -o UserKnownHostsFile=~/.ssh/known_hosts ubuntu@"$BACKEND_IP" << EOF
        set -euo pipefail

        # Update environment file
        cat > /opt/mini-xdr/.env << 'ENVEOF'
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
UI_ORIGIN=http://localhost:3000,http://24.11.0.176:3000,http://$BACKEND_IP:3000
API_KEY=\$(openssl rand -hex 32)

# Database
DATABASE_URL=${db_url}

# Detection Configuration
FAIL_WINDOW_SECONDS=60
FAIL_THRESHOLD=6
AUTO_CONTAIN=false
ALLOW_PRIVATE_IP_BLOCKING=true

# Honeypot Configuration - TPOT
HONEYPOT_HOST=34.193.101.171
HONEYPOT_USER=admin
HONEYPOT_SSH_KEY=/home/ubuntu/.ssh/mini-xdr-tpot-key.pem
HONEYPOT_SSH_PORT=64295

# LLM Configuration
LLM_PROVIDER=openai
OPENAI_API_KEY=YOUR_OPENAI_API_KEY_HERE
OPENAI_MODEL=gpt-4o-mini
XAI_API_KEY=YOUR_XAI_API_KEY_HERE
XAI_MODEL=grok-beta

# External APIs
ABUSEIPDB_API_KEY=YOUR_ABUSEIPDB_KEY_HERE
VIRUSTOTAL_API_KEY=YOUR_VIRUSTOTAL_KEY_HERE

# ML Models and Storage
ML_MODELS_PATH=/opt/mini-xdr/models
POLICIES_PATH=/opt/mini-xdr/policies
MODELS_BUCKET=${MODELS_BUCKET}
AWS_REGION=${REGION}

# Agent Authentication (placeholder - update with real values)
AGENT_API_KEY=WILL_BE_GENERATED_DURING_DEPLOYMENT

# T-Pot Integration
TPOT_HOST=34.193.101.171
TPOT_SSH_PORT=64295
TPOT_WEB_PORT=64297
TPOT_API_KEY=demo-tpot-api-key
ENVEOF

        echo "Environment configuration completed!"
EOF

    log "Environment configured!"
}

# Setup database
setup_database() {
    log "Setting up database..."

    ssh -i "~/.ssh/${KEY_NAME}.pem" -o StrictHostKeyChecking=yes -o UserKnownHostsFile=~/.ssh/known_hosts ubuntu@"$BACKEND_IP" << 'DB_SETUP'
        set -euo pipefail

        cd /opt/mini-xdr/backend
        source ../venv/bin/activate

        # Initialize database schema
        python -c "
import asyncio
from app.models import init_db
asyncio.run(init_db())
print('Database initialized successfully!')
"

        echo "Database setup completed!"
DB_SETUP

    log "Database setup completed!"
}

# Start services
start_services() {
    log "Starting Mini-XDR services..."

    ssh -i "~/.ssh/${KEY_NAME}.pem" -o StrictHostKeyChecking=yes -o UserKnownHostsFile=~/.ssh/known_hosts ubuntu@"$BACKEND_IP" << 'START_SERVICES'
        set -euo pipefail

        # Start Mini-XDR service
        sudo systemctl start mini-xdr
        sudo systemctl status mini-xdr --no-pager

        # Check service logs
        echo "Recent logs:"
        sudo journalctl -u mini-xdr -n 20 --no-pager

        echo "Services started!"
START_SERVICES

    log "Services started!"
}

# Test deployment
test_deployment() {
    log "Testing deployment..."

    # Test health endpoint
    local health_url="http://$BACKEND_IP:8000/health"
    local retry_count=0
    local max_retries=10

    while ! curl -f "$health_url" >/dev/null 2>&1; do
        retry_count=$((retry_count + 1))
        if [ $retry_count -gt $max_retries ]; then
            error "Health check failed after $max_retries attempts"
        fi
        log "Waiting for API to be ready... ($retry_count/$max_retries)"
        sleep 10
    done

    # Test API endpoints
    log "Testing API endpoints..."
    curl -s "$health_url" | jq .

    # Test events endpoint
    local events_url="http://$BACKEND_IP:8000/events"
    curl -s "$events_url" | jq .

    log "✅ Deployment test passed!"
}

# Copy SSH key for TPOT access
copy_ssh_key() {
    log "Copying SSH key for TPOT access..."

    if [ -f "~/.ssh/${KEY_NAME}.pem" ]; then
        scp -i "~/.ssh/${KEY_NAME}.pem" -o StrictHostKeyChecking=yes -o UserKnownHostsFile=~/.ssh/known_hosts \
            "~/.ssh/${KEY_NAME}.pem" ubuntu@"$BACKEND_IP":/home/ubuntu/.ssh/mini-xdr-tpot-key.pem

        ssh -i "~/.ssh/${KEY_NAME}.pem" -o StrictHostKeyChecking=yes -o UserKnownHostsFile=~/.ssh/known_hosts ubuntu@"$BACKEND_IP" \
            "chmod 600 /home/ubuntu/.ssh/mini-xdr-tpot-key.pem"

        log "SSH key copied successfully!"
    else
        warn "SSH key not found, please copy manually"
    fi
}

# Main deployment function
main() {
    log "Starting Mini-XDR code deployment..."

    get_stack_info
    wait_for_instance
    create_deployment_package
    upload_code
    upload_models_to_s3
    copy_ssh_key
    configure_environment
    setup_database
    start_services
    test_deployment

    log "✅ Mini-XDR deployment completed successfully!"
    log ""
    log "🌐 API Endpoint: http://$BACKEND_IP:8000"
    log "🔗 Health Check: http://$BACKEND_IP:8000/health"
    log "📊 Events API: http://$BACKEND_IP:8000/events"
    log ""
    log "🔧 Management:"
    log "   SSH: ssh -i ~/.ssh/${KEY_NAME}.pem ubuntu@$BACKEND_IP"
    log "   Logs: sudo journalctl -u mini-xdr -f"
    log "   Restart: sudo systemctl restart mini-xdr"
    log ""
    log "Next step: Configure TPOT to send data to: http://$BACKEND_IP:8000/ingest/multi"
}

# Run main function
main "$@"

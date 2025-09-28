#!/bin/bash

# Mini-XDR Update Pipeline
# Easy deployment of frontend and backend changes to AWS

set -euo pipefail

# Configuration
REGION="${AWS_REGION:-us-east-1}"
BACKEND_STACK_NAME="mini-xdr-backend"
FRONTEND_STACK_NAME="mini-xdr-frontend"
PROJECT_DIR="/Users/chasemad/Desktop/mini-xdr"
KEY_NAME="${KEY_NAME:-mini-xdr-tpot-key}"

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
    echo "=================================="
    echo "    Mini-XDR Update Pipeline"
    echo "=================================="
    echo -e "${NC}"
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

# Show usage
show_usage() {
    echo "Usage: $0 {frontend|backend|both|quick} [options]"
    echo ""
    echo "Update Types:"
    echo "  frontend  - Deploy frontend changes to S3 + CloudFront"
    echo "  backend   - Deploy backend changes to EC2"
    echo "  both      - Deploy both frontend and backend changes"
    echo "  quick     - Quick frontend update (no cache invalidation)"
    echo ""
    echo "Options:"
    echo "  --no-build        Skip build step (use existing build)"
    echo "  --no-restart      Skip service restart (backend only)"
    echo "  --no-cache-clear  Skip CloudFront cache invalidation"
    echo "  --dry-run         Show what would be updated without deploying"
    echo ""
    echo "Examples:"
    echo "  $0 frontend                    # Standard frontend update"
    echo "  $0 backend --no-restart        # Backend update without restart"
    echo "  $0 both                        # Full update"
    echo "  $0 quick                       # Fast frontend-only update"
    echo "  $0 frontend --dry-run          # Preview frontend changes"
}

# Parse command line arguments
parse_arguments() {
    UPDATE_TYPE="${1:-}"
    shift || true
    
    NO_BUILD=false
    NO_RESTART=false
    NO_CACHE_CLEAR=false
    DRY_RUN=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --no-build)
                NO_BUILD=true
                shift
                ;;
            --no-restart)
                NO_RESTART=true
                shift
                ;;
            --no-cache-clear)
                NO_CACHE_CLEAR=true
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            *)
                error "Unknown option: $1"
                ;;
        esac
    done
}

# Get deployment info
get_deployment_info() {
    # Backend info
    if aws cloudformation describe-stacks --stack-name "$BACKEND_STACK_NAME" --region "$REGION" >/dev/null 2>&1; then
        local backend_outputs
        backend_outputs=$(aws cloudformation describe-stacks \
            --stack-name "$BACKEND_STACK_NAME" \
            --region "$REGION" \
            --query 'Stacks[0].Outputs' \
            --output json)
        
        BACKEND_IP=$(echo "$backend_outputs" | jq -r '.[] | select(.OutputKey=="BackendPublicIP") | .OutputValue' 2>/dev/null || echo "")
        BACKEND_INSTANCE_ID=$(echo "$backend_outputs" | jq -r '.[] | select(.OutputKey=="BackendInstanceId") | .OutputValue' 2>/dev/null || echo "")
        BACKEND_DEPLOYED=true
    else
        BACKEND_DEPLOYED=false
    fi
    
    # Frontend info
    if aws cloudformation describe-stacks --stack-name "$FRONTEND_STACK_NAME" --region "$REGION" >/dev/null 2>&1; then
        local frontend_outputs
        frontend_outputs=$(aws cloudformation describe-stacks \
            --stack-name "$FRONTEND_STACK_NAME" \
            --region "$REGION" \
            --query 'Stacks[0].Outputs' \
            --output json)
        
        FRONTEND_BUCKET=$(echo "$frontend_outputs" | jq -r '.[] | select(.OutputKey=="BucketName") | .OutputValue' 2>/dev/null || echo "")
        FRONTEND_CLOUDFRONT_URL=$(echo "$frontend_outputs" | jq -r '.[] | select(.OutputKey=="CloudFrontURL") | .OutputValue' 2>/dev/null || echo "")
        FRONTEND_DEPLOYED=true
        
        # Get CloudFront distribution ID
        if [ -n "$FRONTEND_BUCKET" ]; then
            CLOUDFRONT_DISTRIBUTION_ID=$(aws cloudfront list-distributions \
                --query "DistributionList.Items[?Origins.Items[0].DomainName=='$FRONTEND_BUCKET.s3.amazonaws.com'].Id" \
                --output text 2>/dev/null || echo "")
        fi
    else
        FRONTEND_DEPLOYED=false
    fi
}

# Check if changes need deployment
check_changes() {
    step "🔍 Checking for changes..."
    
    # Check git status for changes
    cd "$PROJECT_DIR"
    
    local frontend_changes=false
    local backend_changes=false
    
    # Check for uncommitted changes
    if ! git diff --quiet HEAD; then
        log "Uncommitted changes detected"
        
        # Check specific directories
        if git diff --quiet HEAD -- frontend/; then
            log "No frontend changes"
        else
            frontend_changes=true
            log "Frontend changes detected"
        fi
        
        if git diff --quiet HEAD -- backend/; then
            log "No backend changes"
        else
            backend_changes=true
            log "Backend changes detected"
        fi
    else
        log "No uncommitted changes (deploying current state)"
    fi
    
    # Show file changes if any
    if [ "$frontend_changes" = true ] || [ "$backend_changes" = true ]; then
        echo ""
        log "Changed files:"
        git diff --name-only HEAD
        echo ""
    fi
}

# Update frontend
update_frontend() {
    step "🎨 Updating Frontend"
    
    if [ "$FRONTEND_DEPLOYED" = false ]; then
        error "Frontend not deployed. Run './deploy-frontend-aws.sh' first."
    fi
    
    if [ "$DRY_RUN" = true ]; then
        log "DRY RUN: Would update frontend to S3 bucket: $FRONTEND_BUCKET"
        return
    fi
    
    cd "$PROJECT_DIR/frontend"
    
    # Build frontend
    if [ "$NO_BUILD" = false ]; then
        log "Building frontend..."
        
        # Create production environment
        cat > ".env.production" << EOF
NEXT_PUBLIC_API_URL=http://${BACKEND_IP}:8000
NEXT_PUBLIC_WS_URL=ws://${BACKEND_IP}:8000/ws
NEXT_PUBLIC_ENV=aws-production
EOF
        
        npm ci
        npm run build
        log "✅ Frontend build completed"
    else
        log "⏭️  Skipping build (using existing build)"
    fi
    
    # Upload to S3
    log "Uploading to S3..."
    aws s3 sync out/ "s3://$FRONTEND_BUCKET/" --delete --region "$REGION"
    
    # Set proper content types and cache headers
    log "Setting content types and cache headers..."
    
    # HTML files - no cache
    aws s3 cp "s3://$FRONTEND_BUCKET/" "s3://$FRONTEND_BUCKET/" \
        --recursive \
        --exclude "*" \
        --include "*.html" \
        --metadata-directive REPLACE \
        --cache-control "public, max-age=0, must-revalidate" \
        --content-type "text/html" \
        --region "$REGION"
    
    # CSS files - long cache
    aws s3 cp "s3://$FRONTEND_BUCKET/" "s3://$FRONTEND_BUCKET/" \
        --recursive \
        --exclude "*" \
        --include "*.css" \
        --metadata-directive REPLACE \
        --cache-control "public, max-age=31536000, immutable" \
        --content-type "text/css" \
        --region "$REGION"
    
    # JS files - long cache
    aws s3 cp "s3://$FRONTEND_BUCKET/" "s3://$FRONTEND_BUCKET/" \
        --recursive \
        --exclude "*" \
        --include "*.js" \
        --metadata-directive REPLACE \
        --cache-control "public, max-age=31536000, immutable" \
        --content-type "application/javascript" \
        --region "$REGION"
    
    log "✅ Frontend uploaded to S3"
    
    # Invalidate CloudFront cache
    if [ "$NO_CACHE_CLEAR" = false ] && [ -n "$CLOUDFRONT_DISTRIBUTION_ID" ]; then
        log "Invalidating CloudFront cache..."
        aws cloudfront create-invalidation \
            --distribution-id "$CLOUDFRONT_DISTRIBUTION_ID" \
            --paths "/*" >/dev/null
        log "✅ CloudFront cache invalidated"
        log "⏰ Cache propagation may take 5-15 minutes"
    else
        warn "⏭️  Skipping CloudFront cache invalidation"
    fi
    
    log "🎉 Frontend update completed!"
    log "🌐 URL: $FRONTEND_CLOUDFRONT_URL"
}

# Update backend
update_backend() {
    step "🔧 Updating Backend"
    
    if [ "$BACKEND_DEPLOYED" = false ]; then
        error "Backend not deployed. Run './deploy-mini-xdr-aws.sh' first."
    fi
    
    if [ "$DRY_RUN" = true ]; then
        log "DRY RUN: Would update backend on instance: $BACKEND_INSTANCE_ID"
        return
    fi
    
    # Check if instance is running
    local instance_state
    instance_state=$(aws ec2 describe-instances \
        --instance-ids "$BACKEND_INSTANCE_ID" \
        --region "$REGION" \
        --query 'Reservations[0].Instances[0].State.Name' \
        --output text)
    
    if [ "$instance_state" != "running" ]; then
        log "Starting backend instance..."
        aws ec2 start-instances --instance-ids "$BACKEND_INSTANCE_ID" --region "$REGION" >/dev/null
        aws ec2 wait instance-running --instance-ids "$BACKEND_INSTANCE_ID" --region "$REGION"
        log "✅ Backend instance started"
    fi
    
    # Create deployment package
    log "Creating deployment package..."
    local temp_dir="/tmp/mini-xdr-update-$(date +%s)"
    mkdir -p "$temp_dir"
    
    # Copy only backend code (exclude large datasets, models, etc.)
    rsync -av --exclude='node_modules' --exclude='__pycache__' --exclude='*.pyc' \
          --exclude='venv' --exclude='.git' \
          "$PROJECT_DIR/backend/" "$temp_dir/"
    
    # Create tar archive
    cd "$temp_dir"
    tar -czf "/tmp/backend-update.tar.gz" .
    
    log "Uploading backend code..."
    scp -i "~/.ssh/${KEY_NAME}.pem" -o StrictHostKeyChecking=yes -o UserKnownHostsFile=~/.ssh/known_hosts \
        "/tmp/backend-update.tar.gz" ubuntu@"$BACKEND_IP":/tmp/
    
    # Deploy on remote server
    log "Deploying backend changes..."
    ssh -i "~/.ssh/${KEY_NAME}.pem" -o StrictHostKeyChecking=yes -o UserKnownHostsFile=~/.ssh/known_hosts ubuntu@"$BACKEND_IP" << 'REMOTE_DEPLOY'
        set -euo pipefail
        
        echo "Extracting backend update..."
        cd /opt/mini-xdr
        sudo tar -xzf /tmp/backend-update.tar.gz -C backend/
        sudo chown -R ubuntu:ubuntu backend/
        
        echo "Installing/updating Python dependencies..."
        source venv/bin/activate
        cd backend
        pip install --upgrade pip
        pip install -r requirements.txt
        
        echo "Backend code update completed!"
REMOTE_DEPLOY
    
    # Restart services if not skipped
    if [ "$NO_RESTART" = false ]; then
        log "Restarting backend services..."
        ssh -i "~/.ssh/${KEY_NAME}.pem" -o StrictHostKeyChecking=yes -o UserKnownHostsFile=~/.ssh/known_hosts ubuntu@"$BACKEND_IP" << 'REMOTE_RESTART'
            sudo systemctl restart mini-xdr
            sleep 5
            if systemctl is-active --quiet mini-xdr; then
                echo "✅ Mini-XDR service restarted successfully"
                sudo systemctl status mini-xdr --no-pager -l
            else
                echo "❌ Mini-XDR service failed to restart"
                sudo journalctl -u mini-xdr -n 20 --no-pager
                exit 1
            fi
REMOTE_RESTART
        log "✅ Backend services restarted"
    else
        warn "⏭️  Skipping service restart"
    fi
    
    # Test backend
    log "Testing backend API..."
    local retry_count=0
    local max_retries=5
    
    while ! curl -f "http://$BACKEND_IP:8000/health" >/dev/null 2>&1; do
        retry_count=$((retry_count + 1))
        if [ $retry_count -gt $max_retries ]; then
            error "Backend health check failed after restart"
        fi
        log "Waiting for API to be ready... ($retry_count/$max_retries)"
        sleep 10
    done
    
    log "✅ Backend API is healthy"
    
    # Cleanup
    rm -f "/tmp/backend-update.tar.gz"
    rm -rf "$temp_dir"
    
    log "🎉 Backend update completed!"
    log "🌐 API: http://$BACKEND_IP:8000"
}

# Quick frontend update (minimal steps)
quick_frontend_update() {
    step "⚡ Quick Frontend Update"
    
    if [ "$FRONTEND_DEPLOYED" = false ]; then
        error "Frontend not deployed. Run './deploy-frontend-aws.sh' first."
    fi
    
    cd "$PROJECT_DIR/frontend"
    
    log "Quick build..."
    npm run build
    
    log "Quick upload..."
    aws s3 sync out/ "s3://$FRONTEND_BUCKET/" --delete --region "$REGION"
    
    log "✅ Quick frontend update completed!"
    log "🌐 URL: $FRONTEND_CLOUDFRONT_URL"
    warn "⏰ CloudFront cache not invalidated - changes may take time to appear"
}

# Show deployment summary
show_summary() {
    step "📋 Deployment Summary"
    
    echo ""
    echo "Services Updated:"
    case "$UPDATE_TYPE" in
        frontend|quick)
            echo "  ✅ Frontend: $FRONTEND_CLOUDFRONT_URL"
            ;;
        backend)
            echo "  ✅ Backend: http://$BACKEND_IP:8000"
            ;;
        both)
            echo "  ✅ Frontend: $FRONTEND_CLOUDFRONT_URL"
            echo "  ✅ Backend: http://$BACKEND_IP:8000"
            ;;
    esac
    
    echo ""
    echo "Quick Commands:"
    echo "  🔄 Update again: $0 $UPDATE_TYPE"
    echo "  📊 Check status: ./aws-services-control.sh status"
    echo "  📋 View logs: ./aws-services-control.sh logs"
    echo "  🌐 Open frontend: open $FRONTEND_CLOUDFRONT_URL"
    echo ""
}

# Main function
main() {
    show_banner
    
    parse_arguments "$@"
    
    if [ -z "$UPDATE_TYPE" ]; then
        show_usage
        exit 1
    fi
    
    get_deployment_info
    check_changes
    
    local start_time=$(date +%s)
    
    case "$UPDATE_TYPE" in
        frontend)
            update_frontend
            ;;
        backend)
            update_backend
            ;;
        both)
            update_backend
            update_frontend
            ;;
        quick)
            quick_frontend_update
            ;;
        *)
            show_usage
            exit 1
            ;;
    esac
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    show_summary
    log "⏱️  Update completed in ${duration}s"
}

# Export configuration
export AWS_REGION="$REGION"
export KEY_NAME="$KEY_NAME"

# Run main function
main "$@"

#!/bin/bash

# Mini-XDR AWS Services Control Script
# Start, stop, restart, and monitor all AWS services

set -euo pipefail

# Configuration
REGION="${AWS_REGION:-us-east-1}"
BACKEND_STACK_NAME="mini-xdr-backend"
FRONTEND_STACK_NAME="mini-xdr-frontend"
KEY_NAME="${KEY_NAME:-mini-xdr-tpot-key}"

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
    echo "    Mini-XDR AWS Services"
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

# Show usage
show_usage() {
    echo "Usage: $0 {start|stop|restart|status|logs|update}"
    echo ""
    echo "Commands:"
    echo "  start    - Start all AWS services"
    echo "  stop     - Stop AWS services (EC2 instances)"
    echo "  restart  - Restart AWS services"
    echo "  status   - Show status of all services"
    echo "  logs     - Show recent logs from backend"
    echo "  update   - Update both frontend and backend"
    echo "  urls     - Show service URLs"
    echo ""
    echo "Examples:"
    echo "  $0 start"
    echo "  $0 status"
    echo "  $0 logs"
    echo "  $0 update"
}

# Get stack information
get_stack_info() {
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
        
        FRONTEND_CLOUDFRONT_URL=$(echo "$frontend_outputs" | jq -r '.[] | select(.OutputKey=="CloudFrontURL") | .OutputValue' 2>/dev/null || echo "")
        FRONTEND_BUCKET=$(echo "$frontend_outputs" | jq -r '.[] | select(.OutputKey=="BucketName") | .OutputValue' 2>/dev/null || echo "")
        FRONTEND_DEPLOYED=true
    else
        FRONTEND_DEPLOYED=false
    fi
}

# Start services
start_services() {
    step "🚀 Starting Mini-XDR AWS Services"
    
    get_stack_info
    
    if [ "$BACKEND_DEPLOYED" = false ]; then
        error "Backend not deployed. Run './deploy-mini-xdr-aws.sh' first."
    fi
    
    # Start backend EC2 instance
    if [ -n "$BACKEND_INSTANCE_ID" ]; then
        local instance_state
        instance_state=$(aws ec2 describe-instances \
            --instance-ids "$BACKEND_INSTANCE_ID" \
            --region "$REGION" \
            --query 'Reservations[0].Instances[0].State.Name' \
            --output text)
        
        if [ "$instance_state" = "stopped" ]; then
            log "Starting backend EC2 instance..."
            aws ec2 start-instances --instance-ids "$BACKEND_INSTANCE_ID" --region "$REGION" >/dev/null
            aws ec2 wait instance-running --instance-ids "$BACKEND_INSTANCE_ID" --region "$REGION"
            log "✅ Backend instance started"
        elif [ "$instance_state" = "running" ]; then
            log "✅ Backend instance already running"
        else
            log "Backend instance state: $instance_state"
        fi
        
        # Start Mini-XDR service
        log "Starting Mini-XDR application service..."
        ssh -i "~/.ssh/${KEY_NAME}.pem" -o ConnectTimeout=30 -o StrictHostKeyChecking=yes -o UserKnownHostsFile=~/.ssh/known_hosts \
            ubuntu@"$BACKEND_IP" << 'REMOTE_START'
            sudo systemctl start mini-xdr
            sudo systemctl start nginx
            sleep 5
            if systemctl is-active --quiet mini-xdr; then
                echo "✅ Mini-XDR service started"
            else
                echo "❌ Mini-XDR service failed to start"
                sudo journalctl -u mini-xdr -n 20 --no-pager
            fi
REMOTE_START
    fi
    
    # Frontend is always available (S3 + CloudFront)
    if [ "$FRONTEND_DEPLOYED" = true ]; then
        log "✅ Frontend available via CloudFront"
    fi
    
    log "🎉 All services started successfully!"
    show_service_urls
}

# Stop services
stop_services() {
    step "🛑 Stopping Mini-XDR AWS Services"
    
    get_stack_info
    
    if [ "$BACKEND_DEPLOYED" = false ]; then
        warn "Backend not deployed, nothing to stop"
        return
    fi
    
    # Stop backend services
    if [ -n "$BACKEND_INSTANCE_ID" ]; then
        local instance_state
        instance_state=$(aws ec2 describe-instances \
            --instance-ids "$BACKEND_INSTANCE_ID" \
            --region "$REGION" \
            --query 'Reservations[0].Instances[0].State.Name' \
            --output text)
        
        if [ "$instance_state" = "running" ]; then
            log "Stopping Mini-XDR application service..."
            ssh -i "~/.ssh/${KEY_NAME}.pem" -o ConnectTimeout=30 -o StrictHostKeyChecking=yes -o UserKnownHostsFile=~/.ssh/known_hosts \
                ubuntu@"$BACKEND_IP" "sudo systemctl stop mini-xdr" || true
            
            log "Stopping backend EC2 instance..."
            aws ec2 stop-instances --instance-ids "$BACKEND_INSTANCE_ID" --region "$REGION" >/dev/null
            aws ec2 wait instance-stopped --instance-ids "$BACKEND_INSTANCE_ID" --region "$REGION"
            log "✅ Backend instance stopped"
        else
            log "✅ Backend instance already stopped"
        fi
    fi
    
    log "🎉 Services stopped successfully!"
    warn "Note: Frontend (CloudFront) remains available. S3 costs are minimal."
}

# Restart services
restart_services() {
    step "🔄 Restarting Mini-XDR AWS Services"
    
    get_stack_info
    
    if [ "$BACKEND_DEPLOYED" = false ]; then
        error "Backend not deployed. Run './deploy-mini-xdr-aws.sh' first."
    fi
    
    if [ -n "$BACKEND_INSTANCE_ID" ]; then
        log "Restarting Mini-XDR application service..."
        ssh -i "~/.ssh/${KEY_NAME}.pem" -o ConnectTimeout=30 -o StrictHostKeyChecking=yes -o UserKnownHostsFile=~/.ssh/known_hosts \
            ubuntu@"$BACKEND_IP" << 'REMOTE_RESTART'
            sudo systemctl restart mini-xdr
            sudo systemctl restart nginx
            sleep 5
            if systemctl is-active --quiet mini-xdr; then
                echo "✅ Mini-XDR service restarted"
            else
                echo "❌ Mini-XDR service failed to restart"
                sudo journalctl -u mini-xdr -n 20 --no-pager
            fi
REMOTE_RESTART
    fi
    
    log "✅ Services restarted successfully!"
}

# Show service status
show_status() {
    step "📊 Mini-XDR AWS Services Status"
    
    get_stack_info
    
    echo ""
    echo "Backend Services:"
    if [ "$BACKEND_DEPLOYED" = true ]; then
        local instance_state
        instance_state=$(aws ec2 describe-instances \
            --instance-ids "$BACKEND_INSTANCE_ID" \
            --region "$REGION" \
            --query 'Reservations[0].Instances[0].State.Name' \
            --output text)
        
        echo "  🖥️  EC2 Instance: $instance_state ($BACKEND_INSTANCE_ID)"
        echo "  🌐 IP Address: $BACKEND_IP"
        
        if [ "$instance_state" = "running" ]; then
            # Check application status
            if curl -f "http://$BACKEND_IP:8000/health" >/dev/null 2>&1; then
                echo "  ✅ Mini-XDR API: Healthy"
            else
                echo "  ❌ Mini-XDR API: Unhealthy"
            fi
            
            # Check service status via SSH
            local app_status
            app_status=$(ssh -i "~/.ssh/${KEY_NAME}.pem" -o ConnectTimeout=10 -o StrictHostKeyChecking=yes -o UserKnownHostsFile=~/.ssh/known_hosts \
                ubuntu@"$BACKEND_IP" "systemctl is-active mini-xdr" 2>/dev/null || echo "unknown")
            echo "  🔧 Application Service: $app_status"
        else
            echo "  ⏸️  Mini-XDR API: Instance stopped"
        fi
    else
        echo "  ❌ Backend: Not deployed"
    fi
    
    echo ""
    echo "Frontend Services:"
    if [ "$FRONTEND_DEPLOYED" = true ]; then
        echo "  ✅ S3 Bucket: $FRONTEND_BUCKET"
        echo "  🌐 CloudFront: Available"
        echo "  🔗 URL: $FRONTEND_CLOUDFRONT_URL"
        
        # Test CloudFront accessibility
        if curl -f "$FRONTEND_CLOUDFRONT_URL" >/dev/null 2>&1; then
            echo "  ✅ Frontend: Accessible"
        else
            echo "  ⚠️  Frontend: May be propagating"
        fi
    else
        echo "  ❌ Frontend: Not deployed"
    fi
    
    echo ""
}

# Show service logs
show_logs() {
    step "📋 Mini-XDR Backend Logs"
    
    get_stack_info
    
    if [ "$BACKEND_DEPLOYED" = false ]; then
        error "Backend not deployed."
    fi
    
    local instance_state
    instance_state=$(aws ec2 describe-instances \
        --instance-ids "$BACKEND_INSTANCE_ID" \
        --region "$REGION" \
        --query 'Reservations[0].Instances[0].State.Name' \
        --output text)
    
    if [ "$instance_state" != "running" ]; then
        error "Backend instance is not running"
    fi
    
    log "Fetching recent logs from backend..."
    ssh -i "~/.ssh/${KEY_NAME}.pem" -o ConnectTimeout=30 -o StrictHostKeyChecking=yes -o UserKnownHostsFile=~/.ssh/known_hosts \
        ubuntu@"$BACKEND_IP" << 'REMOTE_LOGS'
        echo "=== Mini-XDR Service Status ==="
        sudo systemctl status mini-xdr --no-pager
        echo ""
        echo "=== Recent Application Logs ==="
        sudo journalctl -u mini-xdr -n 50 --no-pager
        echo ""
        echo "=== Recent System Logs ==="
        sudo tail -20 /var/log/mini-xdr-setup.log 2>/dev/null || echo "Setup log not found"
REMOTE_LOGS
}

# Update services
update_services() {
    step "🔄 Updating Mini-XDR Services"
    
    log "Updating backend..."
    if [ -f "./deploy-mini-xdr-code.sh" ]; then
        ./deploy-mini-xdr-code.sh
    else
        error "Backend deployment script not found"
    fi
    
    log "Updating frontend..."
    if [ -f "$HOME/update-frontend-aws.sh" ]; then
        "$HOME/update-frontend-aws.sh"
    elif [ -f "./deploy-frontend-aws.sh" ]; then
        ./deploy-frontend-aws.sh
    else
        error "Frontend deployment script not found"
    fi
    
    log "✅ All services updated successfully!"
}

# Show service URLs
show_service_urls() {
    step "🔗 Mini-XDR Service URLs"
    
    get_stack_info
    
    echo ""
    if [ "$BACKEND_DEPLOYED" = true ]; then
        echo "Backend Services:"
        echo "  🌐 API Endpoint: http://$BACKEND_IP:8000"
        echo "  🏥 Health Check: http://$BACKEND_IP:8000/health"
        echo "  📊 Events API: http://$BACKEND_IP:8000/events"
        echo "  📈 Globe Data: http://$BACKEND_IP:8000/events/globe"
        echo ""
    fi
    
    if [ "$FRONTEND_DEPLOYED" = true ]; then
        echo "Frontend Services:"
        echo "  🌐 Main URL: $FRONTEND_CLOUDFRONT_URL"
        echo "  🎯 Dashboard: $FRONTEND_CLOUDFRONT_URL/dashboard"
        echo "  🌍 Globe View: $FRONTEND_CLOUDFRONT_URL/globe"
        echo ""
    fi
    
    echo "Management:"
    echo "  🔧 SSH Backend: ssh -i ~/.ssh/${KEY_NAME}.pem ubuntu@$BACKEND_IP"
    echo "  📋 Service Logs: $0 logs"
    echo "  🔄 Update All: $0 update"
}

# Main function
main() {
    show_banner
    
    case "${1:-}" in
        start)
            start_services
            ;;
        stop)
            stop_services
            ;;
        restart)
            restart_services
            ;;
        status)
            show_status
            ;;
        logs)
            show_logs
            ;;
        update)
            update_services
            ;;
        urls)
            show_service_urls
            ;;
        *)
            show_usage
            exit 1
            ;;
    esac
}

# Export configuration
export AWS_REGION="$REGION"
export KEY_NAME="$KEY_NAME"

# Run main function
main "$@"

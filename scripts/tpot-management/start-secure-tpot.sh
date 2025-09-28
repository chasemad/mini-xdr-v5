#!/bin/bash
# Start T-Pot honeypot with secure configuration
# All security groups are already configured - only your IPs can access

set -e

REGION="us-east-1"
INSTANCE_NAME="mini-xdr-tpot-honeypot"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() { echo -e "${BLUE}[$(date +'%H:%M:%S')]${NC} $1"; }
success() { echo -e "${GREEN}✅ $1${NC}"; }
warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }

log "Starting secure T-Pot honeypot..."

# Get instance ID
INSTANCE_ID=$(aws ec2 describe-instances \
    --region $REGION \
    --filters "Name=tag:Name,Values=$INSTANCE_NAME" "Name=instance-state-name,Values=stopped" \
    --query 'Reservations[0].Instances[0].InstanceId' \
    --output text)

if [ "$INSTANCE_ID" = "None" ] || [ -z "$INSTANCE_ID" ]; then
    warning "T-Pot instance not found or not in stopped state"
    echo "Checking current instance status..."
    aws ec2 describe-instances \
        --region $REGION \
        --filters "Name=tag:Name,Values=$INSTANCE_NAME" \
        --query 'Reservations[0].Instances[0].[InstanceId,State.Name,PublicIpAddress]' \
        --output table
    exit 1
fi

log "Found T-Pot instance: $INSTANCE_ID"

# Start the instance
log "Starting instance..."
aws ec2 start-instances --region $REGION --instance-ids $INSTANCE_ID

# Wait for instance to be running
log "Waiting for instance to start..."
aws ec2 wait instance-running --region $REGION --instance-ids $INSTANCE_ID

# Get public IP
PUBLIC_IP=$(aws ec2 describe-instances \
    --region $REGION \
    --instance-ids $INSTANCE_ID \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

success "T-Pot honeypot started successfully!"
echo ""
echo "📊 Instance Details:"
echo "   • Instance ID: $INSTANCE_ID"
echo "   • Public IP: $PUBLIC_IP"
echo "   • Region: $REGION"
echo ""
echo "🔒 Security Status:"
echo "   • Management SSH (64295): Restricted to your IP"
echo "   • Web Interface (64297): Restricted to your IP"
echo "   • Honeypot ports: BLOCKED from public internet"
echo ""
echo "⏳ T-Pot services will take 2-3 minutes to fully start"
echo ""
echo "🔧 Management Commands:"
echo "   • SSH Access: ssh -i ~/.ssh/mini-xdr-tpot-key.pem -p 64295 admin@$PUBLIC_IP"
echo "   • Web Interface: https://$PUBLIC_IP:64297/"
echo "   • Check Status: ssh -i ~/.ssh/mini-xdr-tpot-key.pem -p 64295 admin@$PUBLIC_IP 'sudo docker ps'"
echo ""
echo "🎯 For Kali Testing:"
echo "   • Get Kali IP: curl -s -4 icanhazip.com"
echo "   • Add Access: ./kali-access.sh add KALI_IP 22 80 443"
echo "   • Remove Access: ./kali-access.sh remove KALI_IP 22 80 443"

#!/bin/bash
# Create ALB Security Group for Mini-XDR
# Allows HTTPS from specified IP (or 0.0.0.0/0 for public access)

set -e

REGION="us-east-1"
VPC_ID="vpc-0d474acd38d418e98"  # From AWS deployment docs
SG_NAME="mini-xdr-alb-sg"
YOUR_IP="37.19.221.202/32"  # From AWS deployment docs

# Change to 0.0.0.0/0 for public access
ALLOWED_CIDR="${1:-$YOUR_IP}"

echo "🔒 Creating ALB Security Group for Mini-XDR"
echo "==========================================="
echo ""
echo "VPC:          $VPC_ID"
echo "Region:       $REGION"
echo "Allowed CIDR: $ALLOWED_CIDR"
echo ""

# Check if security group already exists
EXISTING_SG=$(aws ec2 describe-security-groups \
  --region $REGION \
  --filters "Name=group-name,Values=$SG_NAME" "Name=vpc-id,Values=$VPC_ID" \
  --query 'SecurityGroups[0].GroupId' \
  --output text 2>/dev/null || echo "None")

if [ "$EXISTING_SG" != "None" ] && [ -n "$EXISTING_SG" ]; then
  echo "✅ Security group already exists: $EXISTING_SG"
  echo ""
  
  # Update rules
  echo "📝 Updating security group rules..."
  
  # Remove old rules (if any)
  aws ec2 revoke-security-group-ingress \
    --group-id $EXISTING_SG \
    --region $REGION \
    --protocol tcp \
    --port 443 \
    --cidr 0.0.0.0/0 2>/dev/null || true
  
  aws ec2 revoke-security-group-ingress \
    --group-id $EXISTING_SG \
    --region $REGION \
    --protocol tcp \
    --port 80 \
    --cidr 0.0.0.0/0 2>/dev/null || true
  
  # Add new rules
  aws ec2 authorize-security-group-ingress \
    --group-id $EXISTING_SG \
    --region $REGION \
    --protocol tcp \
    --port 443 \
    --cidr $ALLOWED_CIDR
  
  aws ec2 authorize-security-group-ingress \
    --group-id $EXISTING_SG \
    --region $REGION \
    --protocol tcp \
    --port 80 \
    --cidr $ALLOWED_CIDR
  
  echo "   ✅ Updated HTTPS (443) from $ALLOWED_CIDR"
  echo "   ✅ Updated HTTP (80) from $ALLOWED_CIDR"
  
else
  # Create new security group
  echo "📦 Creating new security group..."
  
  SG_ID=$(aws ec2 create-security-group \
    --group-name $SG_NAME \
    --description "Security group for Mini-XDR ALB" \
    --vpc-id $VPC_ID \
    --region $REGION \
    --query 'GroupId' \
    --output text)
  
  echo "   ✅ Created: $SG_ID"
  echo ""
  
  # Add ingress rules
  echo "📝 Adding ingress rules..."
  
  aws ec2 authorize-security-group-ingress \
    --group-id $SG_ID \
    --region $REGION \
    --protocol tcp \
    --port 443 \
    --cidr $ALLOWED_CIDR
  
  aws ec2 authorize-security-group-ingress \
    --group-id $SG_ID \
    --region $REGION \
    --protocol tcp \
    --port 80 \
    --cidr $ALLOWED_CIDR
  
  echo "   ✅ HTTPS (443) from $ALLOWED_CIDR"
  echo "   ✅ HTTP (80) from $ALLOWED_CIDR"
  echo ""
  
  # Tag security group
  aws ec2 create-tags \
    --resources $SG_ID \
    --tags Key=Name,Value=$SG_NAME Key=Project,Value=mini-xdr \
    --region $REGION
  
  echo "   ✅ Tagged security group"
  
  EXISTING_SG=$SG_ID
fi

echo ""
echo "✅ Security Group Ready: $EXISTING_SG"
echo ""
echo "📋 Next Steps:"
echo "   1. Update k8s/ingress-alb.yaml with this security group ID"
echo "   2. Apply ingress: kubectl apply -f k8s/ingress-alb.yaml"
echo "   3. Wait 3-5 minutes for ALB provisioning"
echo "   4. Get ALB URL: kubectl get ingress -n mini-xdr"
echo ""
echo "💡 To allow public access:"
echo "   ./scripts/create-alb-security-group.sh 0.0.0.0/0"
echo ""



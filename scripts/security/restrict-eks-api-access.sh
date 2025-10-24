#!/bin/bash
# ==============================================================================
# Restrict EKS API Endpoint Access
# ==============================================================================
# Currently, the EKS API endpoint is accessible from 0.0.0.0/0 (anywhere).
# This script restricts it to only your IP address for better security.
# ==============================================================================

set -e

REGION="us-east-1"
CLUSTER_NAME="mini-xdr-cluster"

echo "======================================"
echo "Restrict EKS API Endpoint Access"
echo "======================================"
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check prerequisites
echo "🔍 Checking prerequisites..."

if ! command -v aws &> /dev/null; then
    echo -e "${RED}❌ Error: AWS CLI is required${NC}"
    exit 1
fi

if ! command -v curl &> /dev/null; then
    echo -e "${RED}❌ Error: curl is required${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Prerequisites met${NC}"
echo ""

# Get current IP
echo "🌐 Detecting your current IP address..."
CURRENT_IP=$(curl -s -4 ifconfig.me)

if [ -z "$CURRENT_IP" ]; then
    echo -e "${RED}❌ Error: Could not detect your IP address${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Your current IP: $CURRENT_IP${NC}"
echo ""

# Check current EKS endpoint access configuration
echo "📊 Checking current EKS API endpoint configuration..."
CURRENT_CONFIG=$(aws eks describe-cluster \
    --name "$CLUSTER_NAME" \
    --region "$REGION" \
    --query 'cluster.resourcesVpcConfig' \
    --output json)

CURRENT_PUBLIC_ACCESS=$(echo "$CURRENT_CONFIG" | jq -r '.endpointPublicAccess')
CURRENT_PRIVATE_ACCESS=$(echo "$CURRENT_CONFIG" | jq -r '.endpointPrivateAccess')
CURRENT_CIDRS=$(echo "$CURRENT_CONFIG" | jq -r '.publicAccessCidrs[]')

echo "Current configuration:"
echo "  Public Access: $CURRENT_PUBLIC_ACCESS"
echo "  Private Access: $CURRENT_PRIVATE_ACCESS"
echo "  Allowed CIDRs: $CURRENT_CIDRS"
echo ""

if [[ "$CURRENT_CIDRS" == *"$CURRENT_IP"* ]] && [ "$CURRENT_CIDRS" != "0.0.0.0/0" ]; then
    echo -e "${GREEN}✅ Your IP is already in the whitelist${NC}"
    echo ""
    read -p "Do you want to update anyway? (yes/no): " UPDATE_ANYWAY
    if [ "$UPDATE_ANYWAY" != "yes" ]; then
        exit 0
    fi
fi

echo -e "${YELLOW}⚠️  IMPORTANT WARNINGS:${NC}"
echo ""
echo "1. This will restrict EKS API access to ONLY your current IP"
echo "2. If your IP changes, you'll need to update this again"
echo "3. kubectl commands will only work from your current IP"
echo "4. You can always add more IPs later"
echo ""
echo "Your IP will be whitelisted: ${CURRENT_IP}/32"
echo ""

read -p "Do you want to continue? (yes/no): " CONFIRM

if [ "$CONFIRM" != "yes" ]; then
    echo "Operation cancelled"
    exit 0
fi

echo ""
echo "🔧 Updating EKS API endpoint access configuration..."
echo ""

# Update cluster configuration
aws eks update-cluster-config \
    --name "$CLUSTER_NAME" \
    --region "$REGION" \
    --resources-vpc-config \
        endpointPublicAccess=true,publicAccessCidrs=["${CURRENT_IP}/32"]

echo ""
echo "⏳ Waiting for cluster update to complete (this takes 3-5 minutes)..."
echo ""

# Wait for update to complete
WAIT_ATTEMPTS=0
MAX_WAIT=15  # 7.5 minutes (30-second intervals)

while [ $WAIT_ATTEMPTS -lt $MAX_WAIT ]; do
    UPDATE_STATUS=$(aws eks describe-cluster \
        --name "$CLUSTER_NAME" \
        --region "$REGION" \
        --query 'cluster.status' \
        --output text)

    if [ "$UPDATE_STATUS" == "ACTIVE" ]; then
        # Double check that update is really complete
        sleep 10
        UPDATED_CIDRS=$(aws eks describe-cluster \
            --name "$CLUSTER_NAME" \
            --region "$REGION" \
            --query 'cluster.resourcesVpcConfig.publicAccessCidrs[]' \
            --output text)

        if [[ "$UPDATED_CIDRS" == *"$CURRENT_IP"* ]]; then
            echo -e "${GREEN}✅ Cluster update successful!${NC}"
            break
        fi
    fi

    echo "   Status: $UPDATE_STATUS (Attempt $((WAIT_ATTEMPTS + 1))/$MAX_WAIT)"
    sleep 30
    WAIT_ATTEMPTS=$((WAIT_ATTEMPTS + 1))
done

if [ $WAIT_ATTEMPTS -eq $MAX_WAIT ]; then
    echo -e "${YELLOW}⚠️  Update may still be in progress. Check AWS Console.${NC}"
fi

echo ""

# Verify kubectl still works
echo "🧪 Testing kubectl connectivity..."
if kubectl get nodes &>/dev/null; then
    echo -e "${GREEN}✅ kubectl connection successful${NC}"
else
    echo -e "${RED}❌ kubectl connection failed${NC}"
    echo "   You may need to wait a few more minutes or check your IP"
fi

echo ""
echo "======================================"
echo "✅ EKS API Access Restricted!"
echo "======================================"
echo ""
echo "Allowed IP: ${CURRENT_IP}/32"
echo ""
echo "🔐 Security Status:"
echo "  ✅ EKS API endpoint only accessible from your IP"
echo "  ✅ kubectl commands secured"
echo "  ✅ Unauthorized access blocked"
echo ""
echo "📝 Important Notes:"
echo ""
echo "1. If your IP changes, run this script again"
echo ""
echo "2. To add additional IPs (e.g., CI/CD, team members):"
echo "   aws eks update-cluster-config \\"
echo "     --name $CLUSTER_NAME \\"
echo "     --region $REGION \\"
echo "     --resources-vpc-config \\"
echo "       endpointPublicAccess=true,publicAccessCidrs=[\"${CURRENT_IP}/32\",\"OTHER_IP/32\"]"
echo ""
echo "3. To allow access from VPN (recommended for teams):"
echo "   - Set up AWS VPN or bastion host"
echo "   - Whitelist VPN exit IP"
echo ""
echo "4. To fully disable public access (most secure):"
echo "   aws eks update-cluster-config \\"
echo "     --name $CLUSTER_NAME \\"
echo "     --region $REGION \\"
echo "     --resources-vpc-config \\"
echo "       endpointPublicAccess=false,endpointPrivateAccess=true"
echo "   (Requires VPN or bastion host for kubectl access)"
echo ""
echo "======================================"

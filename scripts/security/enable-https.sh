#!/bin/bash
# ==============================================================================
# Enable HTTPS on Mini-XDR ALB
# ==============================================================================
# This script:
# 1. Requests an AWS Certificate Manager (ACM) certificate
# 2. Validates the certificate via DNS
# 3. Updates the ALB ingress to use HTTPS
# 4. Enables SSL redirect
# ==============================================================================

set -e

REGION="us-east-1"
CLUSTER_NAME="mini-xdr-cluster"
NAMESPACE="mini-xdr"

echo "======================================"
echo "Mini-XDR HTTPS Enablement"
echo "======================================"
echo ""

# Color codes
RED='\033[0:31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check prerequisites
echo "🔍 Checking prerequisites..."

if ! command -v aws &> /dev/null; then
    echo -e "${RED}❌ Error: AWS CLI is required${NC}"
    exit 1
fi

if ! command -v kubectl &> /dev/null; then
    echo -e "${RED}❌ Error: kubectl is required${NC}"
    exit 1
fi

if ! command -v jq &> /dev/null; then
    echo -e "${RED}❌ Error: jq is required${NC}"
    exit 1
fi

echo -e "${GREEN}✅ All prerequisites met${NC}"
echo ""

# Get domain name from user
echo "📝 Domain Configuration"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Please enter your domain name for Mini-XDR"
echo "Example: mini-xdr.yourdomain.com"
echo ""
read -p "Domain name: " DOMAIN_NAME

if [ -z "$DOMAIN_NAME" ]; then
    echo -e "${RED}❌ Error: Domain name is required${NC}"
    exit 1
fi

echo ""
echo -e "${YELLOW}⚠️  Important: You will need DNS access to create validation records${NC}"
echo ""
read -p "Do you have DNS admin access for $DOMAIN_NAME? (yes/no): " HAS_DNS_ACCESS

if [ "$HAS_DNS_ACCESS" != "yes" ]; then
    echo -e "${YELLOW}⚠️  You'll need to coordinate with your DNS admin for certificate validation${NC}"
fi

echo ""

# Step 1: Request ACM Certificate
echo "📜 Step 1: Requesting ACM Certificate"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

CERT_ARN=$(aws acm request-certificate \
    --domain-name "$DOMAIN_NAME" \
    --validation-method DNS \
    --region "$REGION" \
    --query 'CertificateArn' \
    --output text)

if [ -z "$CERT_ARN" ]; then
    echo -e "${RED}❌ Error: Failed to request certificate${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Certificate requested successfully${NC}"
echo "   ARN: $CERT_ARN"
echo ""

# Step 2: Get DNS validation records
echo "📝 Step 2: DNS Validation Configuration"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Waiting for DNS validation records..."
sleep 5

DNS_VALIDATION=$(aws acm describe-certificate \
    --certificate-arn "$CERT_ARN" \
    --region "$REGION" \
    --query 'Certificate.DomainValidationOptions[0].ResourceRecord' \
    --output json)

DNS_NAME=$(echo "$DNS_VALIDATION" | jq -r '.Name')
DNS_VALUE=$(echo "$DNS_VALIDATION" | jq -r '.Value')
DNS_TYPE=$(echo "$DNS_VALIDATION" | jq -r '.Type')

echo -e "${YELLOW}⚠️  ACTION REQUIRED: Add this DNS record to validate your certificate${NC}"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Record Type: $DNS_TYPE"
echo "Record Name: $DNS_NAME"
echo "Record Value: $DNS_VALUE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Steps to add DNS record:"
echo "1. Log in to your DNS provider (Route 53, Cloudflare, etc.)"
echo "2. Add a new $DNS_TYPE record"
echo "3. Use the Name and Value shown above"
echo "4. Wait for DNS propagation (usually 5-30 minutes)"
echo ""

read -p "Press Enter after you've added the DNS record..."

echo ""
echo "⏳ Waiting for certificate validation..."
echo "   This may take 5-30 minutes depending on DNS propagation"
echo ""

# Wait for certificate to be validated
ATTEMPTS=0
MAX_ATTEMPTS=60  # 30 minutes (30-second intervals)

while [ $ATTEMPTS -lt $MAX_ATTEMPTS ]; do
    STATUS=$(aws acm describe-certificate \
        --certificate-arn "$CERT_ARN" \
        --region "$REGION" \
        --query 'Certificate.Status' \
        --output text)

    if [ "$STATUS" == "ISSUED" ]; then
        echo -e "${GREEN}✅ Certificate validated and issued!${NC}"
        break
    fi

    echo "   Status: $STATUS (Attempt $((ATTEMPTS + 1))/$MAX_ATTEMPTS)"
    sleep 30
    ATTEMPTS=$((ATTEMPTS + 1))
done

if [ "$STATUS" != "ISSUED" ]; then
    echo -e "${RED}❌ Error: Certificate validation timed out${NC}"
    echo "   Please check your DNS records and try again"
    echo "   Certificate ARN: $CERT_ARN"
    exit 1
fi

echo ""

# Step 3: Update ALB Ingress
echo "🔧 Step 3: Updating ALB Ingress for HTTPS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Create new ingress with HTTPS
kubectl annotate ingress mini-xdr-ingress \
    -n "$NAMESPACE" \
    alb.ingress.kubernetes.io/certificate-arn="$CERT_ARN" \
    alb.ingress.kubernetes.io/ssl-redirect="true" \
    alb.ingress.kubernetes.io/listen-ports='[{"HTTP": 80}, {"HTTPS": 443}]' \
    --overwrite

echo -e "${GREEN}✅ Ingress updated for HTTPS${NC}"
echo ""

# Wait for ALB to update
echo "⏳ Waiting for ALB to update (this may take 2-3 minutes)..."
sleep 120

# Get ALB DNS name
ALB_DNS=$(kubectl get ingress mini-xdr-ingress \
    -n "$NAMESPACE" \
    -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')

echo ""
echo "======================================"
echo "✅ HTTPS Configuration Complete!"
echo "======================================"
echo ""
echo "Certificate ARN: $CERT_ARN"
echo "ALB DNS: $ALB_DNS"
echo ""
echo "🔐 Next Steps:"
echo "  1. Create a DNS CNAME record:"
echo "     Name: $DOMAIN_NAME"
echo "     Type: CNAME"
echo "     Value: $ALB_DNS"
echo ""
echo "  2. Wait for DNS propagation (5-30 minutes)"
echo ""
echo "  3. Test HTTPS access:"
echo "     https://$DOMAIN_NAME"
echo ""
echo "  4. Verify HTTP→HTTPS redirect:"
echo "     curl -I http://$DOMAIN_NAME"
echo "     (Should return 301 redirect to HTTPS)"
echo ""
echo "⚠️  Security Reminder:"
echo "  - HTTPS is now enabled"
echo "  - All traffic is encrypted with TLS 1.2+"
echo "  - Auto-renewal enabled via ACM"
echo ""
echo "======================================"

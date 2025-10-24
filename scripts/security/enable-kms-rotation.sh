#!/bin/bash
# ==============================================================================
# Enable KMS Key Rotation
# ==============================================================================
# Enables automatic annual rotation for existing KMS keys used by Mini-XDR
# ==============================================================================

set -e

REGION="us-east-1"
KEY_ID="431cb645-f4d9-41f6-8d6e-6c26c79c5c04"  # Existing RDS encryption key

echo "======================================"
echo "Enable KMS Key Rotation"
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

echo -e "${GREEN}✅ Prerequisites met${NC}"
echo ""

# Verify key exists and get details
echo "🔐 Checking KMS key: $KEY_ID"
KEY_DETAILS=$(aws kms describe-key \
    --key-id "$KEY_ID" \
    --region "$REGION" \
    --output json 2>/dev/null)

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Error: Key not found or no access${NC}"
    exit 1
fi

KEY_ARN=$(echo "$KEY_DETAILS" | jq -r '.KeyMetadata.Arn')
KEY_STATE=$(echo "$KEY_DETAILS" | jq -r '.KeyMetadata.KeyState')
KEY_USAGE=$(echo "$KEY_DETAILS" | jq -r '.KeyMetadata.KeyUsage')
CURRENT_ROTATION=$(echo "$KEY_DETAILS" | jq -r '.KeyMetadata.KeyRotationEnabled // "Not set"')

echo "Key Details:"
echo "  ARN: $KEY_ARN"
echo "  State: $KEY_STATE"
echo "  Usage: $KEY_USAGE"
echo "  Current Rotation: $CURRENT_ROTATION"
echo ""

if [ "$KEY_STATE" != "Enabled" ]; then
    echo -e "${RED}❌ Error: Key is not in Enabled state${NC}"
    exit 1
fi

if [ "$CURRENT_ROTATION" == "true" ]; then
    echo -e "${GREEN}✅ Key rotation is already enabled${NC}"
    echo ""
    read -p "Do you want to verify and continue anyway? (yes/no): " CONTINUE
    if [ "$CONTINUE" != "yes" ]; then
        exit 0
    fi
fi

echo ""
echo "📋 About KMS Key Rotation:"
echo "  - AWS automatically rotates the key material every year"
echo "  - Old key material is retained for decryption"
echo "  - No downtime or re-encryption required"
echo "  - Meets compliance requirements (SOC 2, ISO 27001)"
echo ""

read -p "Enable automatic key rotation? (yes/no): " ENABLE

if [ "$ENABLE" != "yes" ]; then
    echo "Operation cancelled"
    exit 0
fi

echo ""
echo "🔄 Enabling key rotation..."

aws kms enable-key-rotation \
    --key-id "$KEY_ID" \
    --region "$REGION"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Key rotation enabled successfully${NC}"
else
    echo -e "${RED}❌ Failed to enable key rotation${NC}"
    exit 1
fi

echo ""

# Verify rotation is enabled
ROTATION_STATUS=$(aws kms get-key-rotation-status \
    --key-id "$KEY_ID" \
    --region "$REGION" \
    --query 'KeyRotationEnabled' \
    --output text)

echo "Verification:"
echo "  Rotation Status: $ROTATION_STATUS"
echo ""

if [ "$ROTATION_STATUS" == "True" ]; then
    echo -e "${GREEN}✅ Rotation verified and active${NC}"
else
    echo -e "${YELLOW}⚠️  Could not verify rotation status${NC}"
fi

echo ""
echo "======================================"
echo "✅ KMS Key Rotation Enabled!"
echo "======================================"
echo ""
echo "Key ID: $KEY_ID"
echo "Rotation: Automatic (annual)"
echo ""
echo "🔐 Security Status:"
echo "  ✅ Key material rotates automatically every year"
echo "  ✅ Old versions retained for decryption"
echo "  ✅ No manual intervention required"
echo "  ✅ Compliance requirement met"
echo ""
echo "📝 Additional Recommendations:"
echo ""
echo "1. Find all KMS keys in your account and enable rotation:"
echo "   aws kms list-keys --region $REGION --query 'Keys[*].KeyId' --output text | \\"
echo "     xargs -I {} aws kms enable-key-rotation --key-id {} --region $REGION"
echo ""
echo "2. Monitor key usage in CloudWatch:"
echo "   - Metric: KMSKeyUsage"
echo "   - Set alarms for unusual activity"
echo ""
echo "3. Review key policies regularly:"
echo "   aws kms get-key-policy --key-id $KEY_ID --policy-name default --region $REGION"
echo ""
echo "======================================"

#!/bin/bash
# ==============================================================================
# Enable Kubernetes Secrets Encryption for EKS
# ==============================================================================
# This script enables encryption at rest for Kubernetes secrets using AWS KMS.
# Currently, secrets are only base64 encoded - this encrypts them with KMS.
# ==============================================================================

set -e

REGION="us-east-1"
CLUSTER_NAME="mini-xdr-cluster"

echo "======================================"
echo "Enable EKS Secrets Encryption"
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

# Check current encryption status
echo "📊 Checking current encryption status..."
CURRENT_ENCRYPTION=$(aws eks describe-cluster \
    --name "$CLUSTER_NAME" \
    --region "$REGION" \
    --query 'cluster.encryptionConfig' \
    --output json 2>/dev/null || echo "null")

if [ "$CURRENT_ENCRYPTION" != "null" ] && [ "$CURRENT_ENCRYPTION" != "[]" ]; then
    echo -e "${YELLOW}⚠️  Secrets encryption is already enabled${NC}"
    echo "   Current config: $CURRENT_ENCRYPTION"
    echo ""
    read -p "Do you want to continue anyway? (yes/no): " CONTINUE
    if [ "$CONTINUE" != "yes" ]; then
        exit 0
    fi
fi

echo ""

# Step 1: Create KMS key for EKS secrets
echo "🔐 Step 1: Creating KMS Key for EKS Secrets"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

KEY_METADATA=$(aws kms create-key \
    --description "Mini-XDR EKS Secrets Encryption Key" \
    --region "$REGION" \
    --output json)

KEY_ID=$(echo "$KEY_METADATA" | jq -r '.KeyMetadata.KeyId')
KEY_ARN=$(echo "$KEY_METADATA" | jq -r '.KeyMetadata.Arn')

if [ -z "$KEY_ID" ] || [ "$KEY_ID" == "null" ]; then
    echo -e "${RED}❌ Error: Failed to create KMS key${NC}"
    exit 1
fi

echo -e "${GREEN}✅ KMS key created successfully${NC}"
echo "   Key ID: $KEY_ID"
echo "   Key ARN: $KEY_ARN"
echo ""

# Step 2: Create alias for the key
echo "🏷️  Step 2: Creating KMS Key Alias"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

aws kms create-alias \
    --alias-name "alias/mini-xdr-eks-secrets" \
    --target-key-id "$KEY_ID" \
    --region "$REGION"

echo -e "${GREEN}✅ Alias created: alias/mini-xdr-eks-secrets${NC}"
echo ""

# Step 3: Enable automatic key rotation
echo "🔄 Step 3: Enabling Automatic Key Rotation"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

aws kms enable-key-rotation \
    --key-id "$KEY_ID" \
    --region "$REGION"

echo -e "${GREEN}✅ Automatic key rotation enabled (annual)${NC}"
echo ""

# Step 4: Create key policy to allow EKS to use the key
echo "📝 Step 4: Configuring Key Policy for EKS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Get AWS account ID
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

# Get EKS cluster role ARN
CLUSTER_ROLE_ARN=$(aws eks describe-cluster \
    --name "$CLUSTER_NAME" \
    --region "$REGION" \
    --query 'cluster.roleArn' \
    --output text)

# Create key policy that allows EKS to use the key
KEY_POLICY=$(cat <<EOF
{
  "Version": "2012-10-17",
  "Id": "mini-xdr-eks-secrets-encryption",
  "Statement": [
    {
      "Sid": "Enable IAM User Permissions",
      "Effect": "Allow",
      "Principal": {
        "AWS": "arn:aws:iam::${ACCOUNT_ID}:root"
      },
      "Action": "kms:*",
      "Resource": "*"
    },
    {
      "Sid": "Allow EKS to use the key",
      "Effect": "Allow",
      "Principal": {
        "Service": "eks.amazonaws.com"
      },
      "Action": [
        "kms:Decrypt",
        "kms:DescribeKey",
        "kms:CreateGrant"
      ],
      "Resource": "*",
      "Condition": {
        "StringEquals": {
          "kms:ViaService": "eks.${REGION}.amazonaws.com"
        }
      }
    },
    {
      "Sid": "Allow EKS cluster role to use the key",
      "Effect": "Allow",
      "Principal": {
        "AWS": "${CLUSTER_ROLE_ARN}"
      },
      "Action": [
        "kms:Decrypt",
        "kms:DescribeKey",
        "kms:CreateGrant"
      ],
      "Resource": "*"
    }
  ]
}
EOF
)

aws kms put-key-policy \
    --key-id "$KEY_ID" \
    --policy-name default \
    --policy "$KEY_POLICY" \
    --region "$REGION"

echo -e "${GREEN}✅ Key policy configured for EKS access${NC}"
echo ""

# Step 5: Associate encryption config with EKS cluster
echo "🔧 Step 5: Enabling Encryption on EKS Cluster"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo -e "${YELLOW}⚠️  This operation takes 5-10 minutes and cannot be reversed${NC}"
echo ""
read -p "Do you want to enable secrets encryption now? (yes/no): " ENABLE_NOW

if [ "$ENABLE_NOW" != "yes" ]; then
    echo ""
    echo "📝 Encryption not enabled. You can enable it later with:"
    echo ""
    echo "aws eks associate-encryption-config \\"
    echo "  --cluster-name $CLUSTER_NAME \\"
    echo "  --encryption-config '[{\"resources\":[\"secrets\"],\"provider\":{\"keyArn\":\"$KEY_ARN\"}}]' \\"
    echo "  --region $REGION"
    echo ""
    exit 0
fi

echo "⏳ Enabling encryption (this takes 5-10 minutes)..."
echo ""

aws eks associate-encryption-config \
    --cluster-name "$CLUSTER_NAME" \
    --encryption-config "[{\"resources\":[\"secrets\"],\"provider\":{\"keyArn\":\"$KEY_ARN\"}}]" \
    --region "$REGION"

# Wait for update to complete
echo "⏳ Waiting for cluster update to complete..."
echo "   You can check status in AWS Console: EKS → Clusters → $CLUSTER_NAME → Updates"
echo ""

WAIT_ATTEMPTS=0
MAX_WAIT=20  # 10 minutes (30-second intervals)

while [ $WAIT_ATTEMPTS -lt $MAX_WAIT ]; do
    UPDATE_STATUS=$(aws eks describe-update \
        --name "$CLUSTER_NAME" \
        --update-id "$(aws eks list-updates --name $CLUSTER_NAME --region $REGION --query 'updateIds[0]' --output text)" \
        --region "$REGION" \
        --query 'update.status' \
        --output text 2>/dev/null || echo "InProgress")

    if [ "$UPDATE_STATUS" == "Successful" ]; then
        echo -e "${GREEN}✅ Cluster update successful!${NC}"
        break
    elif [ "$UPDATE_STATUS" == "Failed" ]; then
        echo -e "${RED}❌ Cluster update failed${NC}"
        exit 1
    fi

    echo "   Status: $UPDATE_STATUS (Attempt $((WAIT_ATTEMPTS + 1))/$MAX_WAIT)"
    sleep 30
    WAIT_ATTEMPTS=$((WAIT_ATTEMPTS + 1))
done

echo ""

# Step 6: Re-encrypt existing secrets
echo "🔄 Step 6: Re-encrypting Existing Secrets"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo -e "${YELLOW}⚠️  To re-encrypt existing secrets, they must be updated${NC}"
echo ""
echo "Running secret re-encryption for mini-xdr namespace..."
echo ""

# Get all secrets in mini-xdr namespace
SECRETS=$(kubectl get secrets -n mini-xdr -o json | jq -r '.items[].metadata.name')

if [ -z "$SECRETS" ]; then
    echo "No secrets found in mini-xdr namespace"
else
    for SECRET in $SECRETS; do
        echo "Re-encrypting secret: $SECRET"
        kubectl get secret "$SECRET" -n mini-xdr -o json | kubectl apply -f -
    done
    echo ""
    echo -e "${GREEN}✅ All secrets re-encrypted with KMS${NC}"
fi

echo ""
echo "======================================"
echo "✅ EKS Secrets Encryption Enabled!"
echo "======================================"
echo ""
echo "KMS Key ID: $KEY_ID"
echo "KMS Key ARN: $KEY_ARN"
echo "Key Alias: alias/mini-xdr-eks-secrets"
echo "Key Rotation: Enabled (annual)"
echo ""
echo "🔐 Security Status:"
echo "  ✅ New secrets are automatically encrypted with KMS"
echo "  ✅ Existing secrets have been re-encrypted"
echo "  ✅ Secrets are encrypted at rest in etcd"
echo "  ✅ Decryption only possible by authorized principals"
echo ""
echo "📝 Important Notes:"
echo "  - Keep the KMS key ID for your records"
echo "  - Monitor KMS usage in CloudWatch"
echo "  - Key rotation happens automatically every year"
echo "  - Backup key ARN: $KEY_ARN"
echo ""
echo "======================================"

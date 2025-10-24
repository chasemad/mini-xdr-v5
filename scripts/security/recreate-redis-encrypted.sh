#!/bin/bash

# Recreate Redis ElastiCache with Encryption Enabled
# WARNING: This will cause downtime and data loss

set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

AWS_REGION="us-east-1"
CACHE_CLUSTER_ID="mini-xdr-redis"
SUBNET_GROUP="mini-xdr-redis-subnet-group"
SECURITY_GROUP_ID="sg-0c95daec27927de46"

echo -e "${RED}================================${NC}"
echo -e "${RED}Redis Recreation with Encryption${NC}"
echo -e "${RED}================================${NC}"
echo
echo -e "${RED}WARNING: This will DELETE the existing Redis cluster!${NC}"
echo -e "${YELLOW}All data will be lost. Application must handle reconnection.${NC}"
echo
read -p "Are you sure you want to continue? (yes/no): " CONFIRM

if [ "$CONFIRM" != "yes" ]; then
    echo "Aborted."
    exit 1
fi

echo
echo -e "${YELLOW}Step 1: Deleting existing Redis cluster...${NC}"

aws elasticache delete-cache-cluster \
    --cache-cluster-id ${CACHE_CLUSTER_ID} \
    --region ${AWS_REGION} \
    --no-cli-pager

echo "Waiting for cluster deletion (this takes 5-10 minutes)..."
aws elasticache wait cache-cluster-deleted \
    --cache-cluster-id ${CACHE_CLUSTER_ID} \
    --region ${AWS_REGION}

echo -e "${GREEN}✓ Cluster deleted${NC}"

echo
echo -e "${YELLOW}Step 2: Creating new encrypted Redis cluster...${NC}"

# Generate a strong auth token
AUTH_TOKEN=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-32)

# Create new cluster with encryption
aws elasticache create-cache-cluster \
    --cache-cluster-id ${CACHE_CLUSTER_ID} \
    --engine redis \
    --engine-version 7.0 \
    --cache-node-type cache.t3.micro \
    --num-cache-nodes 1 \
    --cache-subnet-group-name ${SUBNET_GROUP} \
    --security-group-ids ${SECURITY_GROUP_ID} \
    --transit-encryption-enabled \
    --at-rest-encryption-enabled \
    --auth-token "${AUTH_TOKEN}" \
    --preferred-maintenance-window sun:03:00-sun:04:00 \
    --snapshot-retention-limit 7 \
    --auto-minor-version-upgrade \
    --tags Key=Environment,Value=production Key=Project,Value=mini-xdr \
    --region ${AWS_REGION} \
    --no-cli-pager

echo "Waiting for cluster creation (this takes 10-15 minutes)..."
aws elasticache wait cache-cluster-available \
    --cache-cluster-id ${CACHE_CLUSTER_ID} \
    --region ${AWS_REGION}

# Get the new endpoint
NEW_ENDPOINT=$(aws elasticache describe-cache-clusters \
    --cache-cluster-id ${CACHE_CLUSTER_ID} \
    --show-cache-node-info \
    --region ${AWS_REGION} \
    --query 'CacheClusters[0].CacheNodes[0].Endpoint.Address' \
    --output text)

echo -e "${GREEN}✓ New encrypted cluster created${NC}"

echo
echo -e "${YELLOW}Step 3: Updating AWS Secrets Manager...${NC}"

# Update the REDIS_URL secret with auth token
REDIS_URL="rediss://:${AUTH_TOKEN}@${NEW_ENDPOINT}:6379/0"

aws secretsmanager update-secret \
    --secret-id mini-xdr-secrets \
    --secret-string "$(aws secretsmanager get-secret-value \
        --secret-id mini-xdr-secrets \
        --region ${AWS_REGION} \
        --query SecretString \
        --output text | \
        jq --arg url "$REDIS_URL" '.REDIS_URL = $url')" \
    --region ${AWS_REGION} \
    --no-cli-pager > /dev/null

echo -e "${GREEN}✓ Secrets Manager updated${NC}"

echo
echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}Redis Recreation Complete!${NC}"
echo -e "${GREEN}================================${NC}"
echo
echo "New Redis Configuration:"
echo "  Endpoint: ${NEW_ENDPOINT}"
echo "  Port: 6379 (TLS)"
echo "  Auth Token: ${AUTH_TOKEN}"
echo
echo "  ✓ Transit Encryption: ENABLED"
echo "  ✓ At-Rest Encryption: ENABLED"
echo "  ✓ Authentication: ENABLED"
echo "  ✓ Automatic Backups: 7 days"
echo
echo -e "${YELLOW}NEXT STEPS:${NC}"
echo "1. Restart backend pods to pick up new Redis connection:"
echo "   kubectl rollout restart deployment/mini-xdr-backend -n mini-xdr"
echo
echo "2. Verify connection:"
echo "   kubectl logs -n mini-xdr deployment/mini-xdr-backend | grep -i redis"
echo
echo "3. Save auth token securely (already in AWS Secrets Manager)"
echo
echo -e "${GREEN}Security Improvement:${NC} Redis now encrypted and authenticated!"
echo

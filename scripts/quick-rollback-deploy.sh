#!/bin/bash
#===============================================================================
# Quick Rollback Deployment - Build and Deploy Old UI
#===============================================================================
# This script builds the old UI locally and deploys to AWS EKS
# Run after starting Docker Desktop: ./scripts/quick-rollback-deploy.sh
#===============================================================================

set -euo pipefail

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
AWS_REGION="us-east-1"
AWS_ACCOUNT_ID="116912495274"
ECR_REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
NAMESPACE="mini-xdr"
VERSION="1.0.2-rollback"
GIT_SHA=$(git rev-parse --short HEAD)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}🔄 Mini-XDR UI Rollback Deployment${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "${YELLOW}Version: $VERSION${NC}"
echo -e "${YELLOW}Git SHA: $GIT_SHA${NC}"
echo ""

# Check Docker
echo -e "${BLUE}[1/6] Checking Docker...${NC}"
if ! docker info &> /dev/null; then
    echo -e "${RED}❌ Docker daemon not running!${NC}"
    echo "Please start Docker Desktop and try again."
    exit 1
fi
echo -e "${GREEN}✅ Docker is running${NC}"
echo ""

# ECR Login
echo -e "${BLUE}[2/6] Logging into ECR...${NC}"
aws ecr get-login-password --region $AWS_REGION | \
    docker login --username AWS --password-stdin $ECR_REGISTRY
echo -e "${GREEN}✅ Logged into ECR${NC}"
echo ""

# Build Backend
echo -e "${BLUE}[3/6] Building Backend Image...${NC}"
cd "$PROJECT_ROOT"
docker buildx build \
    --platform linux/amd64 \
    --file backend/Dockerfile \
    --tag "$ECR_REGISTRY/mini-xdr-backend:$VERSION" \
    --tag "$ECR_REGISTRY/mini-xdr-backend:$GIT_SHA" \
    --tag "$ECR_REGISTRY/mini-xdr-backend:latest" \
    --load \
    backend/
echo -e "${GREEN}✅ Backend image built${NC}"
echo ""

# Build Frontend
echo -e "${BLUE}[4/6] Building Frontend Image (Old UI)...${NC}"
docker buildx build \
    --platform linux/amd64 \
    --file frontend/Dockerfile \
    --tag "$ECR_REGISTRY/mini-xdr-frontend:$VERSION" \
    --tag "$ECR_REGISTRY/mini-xdr-frontend:$GIT_SHA" \
    --tag "$ECR_REGISTRY/mini-xdr-frontend:latest" \
    --load \
    frontend/
echo -e "${GREEN}✅ Frontend image built (OLD UI)${NC}"
echo ""

# Push Images
echo -e "${BLUE}[5/6] Pushing Images to ECR...${NC}"
docker push "$ECR_REGISTRY/mini-xdr-backend:$VERSION"
docker push "$ECR_REGISTRY/mini-xdr-backend:latest"
docker push "$ECR_REGISTRY/mini-xdr-frontend:$VERSION"
docker push "$ECR_REGISTRY/mini-xdr-frontend:latest"
echo -e "${GREEN}✅ Images pushed to ECR${NC}"
echo ""

# Deploy to EKS
echo -e "${BLUE}[6/6] Deploying to EKS...${NC}"
aws eks update-kubeconfig --name mini-xdr-cluster --region $AWS_REGION

echo "  Updating backend deployment..."
kubectl set image deployment/mini-xdr-backend \
    backend="$ECR_REGISTRY/mini-xdr-backend:$VERSION" \
    -n $NAMESPACE

echo "  Updating frontend deployment..."
kubectl set image deployment/mini-xdr-frontend \
    frontend="$ECR_REGISTRY/mini-xdr-frontend:$VERSION" \
    -n $NAMESPACE

echo "  Waiting for rollouts..."
kubectl rollout status deployment/mini-xdr-backend -n $NAMESPACE --timeout=5m &
kubectl rollout status deployment/mini-xdr-frontend -n $NAMESPACE --timeout=5m &
wait

echo -e "${GREEN}✅ Deployment complete!${NC}"
echo ""

# Verify
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}🎉 Rollback Successful!${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "Frontend: $ECR_REGISTRY/mini-xdr-frontend:$VERSION"
echo "Backend:  $ECR_REGISTRY/mini-xdr-backend:$VERSION"
echo ""
echo "ALB URL: http://k8s-minixdr-minixdri-dc5fc1df8b-1132128475.us-east-1.elb.amazonaws.com"
echo ""
echo -e "${YELLOW}Note: May take 1-2 minutes for pods to fully restart${NC}"
echo ""

# Show pod status
kubectl get pods -n $NAMESPACE -o wide

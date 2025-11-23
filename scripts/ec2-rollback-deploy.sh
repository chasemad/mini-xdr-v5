#!/bin/bash
#===============================================================================
# EC2 Remote Build & Deploy - UI Rollback
#===============================================================================
# Builds images on EC2 instance and deploys to EKS
# Run from local machine: ./scripts/ec2-rollback-deploy.sh
#===============================================================================

set -euo pipefail

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
EC2_HOST="54.82.186.21"
EC2_USER="ec2-user"
SSH_KEY="$HOME/.ssh/mini-xdr-eks-key.pem"
AWS_REGION="us-east-1"
AWS_ACCOUNT_ID="116912495274"
NAMESPACE="mini-xdr"
VERSION="1.0.2-rollback"
GITHUB_REPO="https://github.com/chasemad/mini-xdr-v5.git"
GITHUB_BRANCH="main"

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}🔄 EC2 Remote Build & Deploy (UI Rollback)${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "${YELLOW}EC2 Instance: $EC2_HOST${NC}"
echo -e "${YELLOW}Version: $VERSION${NC}"
echo ""

# Check SSH key exists
if [ ! -f "$SSH_KEY" ]; then
    echo -e "${RED}❌ SSH key not found: $SSH_KEY${NC}"
    exit 1
fi
echo -e "${GREEN}✅ SSH key found${NC}"

# Test EC2 connectivity
echo -e "${BLUE}[1/5] Testing EC2 connectivity...${NC}"
if ! ssh -i "$SSH_KEY" -o ConnectTimeout=10 -o StrictHostKeyChecking=no "$EC2_USER@$EC2_HOST" "echo 'Connection successful'" &> /dev/null; then
    echo -e "${RED}❌ Cannot connect to EC2 instance${NC}"
    echo "Please ensure:"
    echo "  1. EC2 instance is running"
    echo "  2. Security group allows SSH from your IP"
    echo "  3. SSH key has correct permissions (chmod 400)"
    exit 1
fi
echo -e "${GREEN}✅ Connected to EC2 instance${NC}"
echo ""

# Create remote build script
echo -e "${BLUE}[2/5] Creating remote build script...${NC}"
cat > /tmp/ec2-build-script.sh << 'EOFREMOTE'
#!/bin/bash
set -euo pipefail

# Colors for remote output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

AWS_REGION="us-east-1"
AWS_ACCOUNT_ID="116912495274"
ECR_REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
VERSION="1.0.2-rollback"
PROJECT_DIR="/home/ec2-user/mini-xdr-v2"
GITHUB_REPO="https://github.com/chasemad/mini-xdr-v5.git"

echo -e "${BLUE}=== Remote Build Script on EC2 ===${NC}"
echo ""

# Update or clone repository
echo -e "${BLUE}Step 1: Updating code from GitHub...${NC}"
if [ -d "$PROJECT_DIR" ]; then
    cd "$PROJECT_DIR"
    echo "Repository exists, pulling latest changes..."
    git fetch origin
    git reset --hard origin/main
    git pull origin main
else
    echo "Cloning repository..."
    cd /home/ec2-user
    git clone "$GITHUB_REPO"
    cd "$PROJECT_DIR"
fi

GIT_SHA=$(git rev-parse --short HEAD)
echo -e "${GREEN}✅ Code updated (commit: $GIT_SHA)${NC}"
echo ""

# ECR Login
echo -e "${BLUE}Step 2: Logging into ECR...${NC}"
aws ecr get-login-password --region $AWS_REGION | \
    docker login --username AWS --password-stdin $ECR_REGISTRY
echo -e "${GREEN}✅ Logged into ECR${NC}"
echo ""

# Build Backend
echo -e "${BLUE}Step 3: Building Backend Image...${NC}"
docker build \
    --platform linux/amd64 \
    --build-arg BUILD_DATE="$(date -u +"%Y-%m-%dT%H:%M:%SZ")" \
    --build-arg VCS_REF="$GIT_SHA" \
    --build-arg VERSION="$VERSION" \
    -t "$ECR_REGISTRY/mini-xdr-backend:$VERSION" \
    -t "$ECR_REGISTRY/mini-xdr-backend:$GIT_SHA" \
    -t "$ECR_REGISTRY/mini-xdr-backend:latest" \
    -f backend/Dockerfile \
    backend/
echo -e "${GREEN}✅ Backend image built${NC}"
echo ""

# Build Frontend
echo -e "${BLUE}Step 4: Building Frontend Image (OLD UI)...${NC}"
docker build \
    --platform linux/amd64 \
    --build-arg BUILD_DATE="$(date -u +"%Y-%m-%dT%H:%M:%SZ")" \
    --build-arg VCS_REF="$GIT_SHA" \
    --build-arg VERSION="$VERSION" \
    -t "$ECR_REGISTRY/mini-xdr-frontend:$VERSION" \
    -t "$ECR_REGISTRY/mini-xdr-frontend:$GIT_SHA" \
    -t "$ECR_REGISTRY/mini-xdr-frontend:latest" \
    -f frontend/Dockerfile \
    frontend/
echo -e "${GREEN}✅ Frontend image built (OLD UI restored)${NC}"
echo ""

# Push Images
echo -e "${BLUE}Step 5: Pushing images to ECR...${NC}"
echo "Pushing backend..."
docker push "$ECR_REGISTRY/mini-xdr-backend:$VERSION"
docker push "$ECR_REGISTRY/mini-xdr-backend:$GIT_SHA"
docker push "$ECR_REGISTRY/mini-xdr-backend:latest"

echo "Pushing frontend..."
docker push "$ECR_REGISTRY/mini-xdr-frontend:$VERSION"
docker push "$ECR_REGISTRY/mini-xdr-frontend:$GIT_SHA"
docker push "$ECR_REGISTRY/mini-xdr-frontend:latest"
echo -e "${GREEN}✅ All images pushed to ECR${NC}"
echo ""

echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}🎉 EC2 Build Complete!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "Backend:  $ECR_REGISTRY/mini-xdr-backend:$VERSION"
echo "Frontend: $ECR_REGISTRY/mini-xdr-frontend:$VERSION"
echo ""
EOFREMOTE

# Upload and execute build script on EC2
echo -e "${BLUE}[3/5] Uploading build script to EC2...${NC}"
scp -i "$SSH_KEY" -o StrictHostKeyChecking=no /tmp/ec2-build-script.sh "$EC2_USER@$EC2_HOST:/tmp/build.sh"
echo -e "${GREEN}✅ Build script uploaded${NC}"
echo ""

echo -e "${BLUE}[4/5] Executing build on EC2 (this may take 5-10 minutes)...${NC}"
echo -e "${YELLOW}───────────────────────────────────────────${NC}"
ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no "$EC2_USER@$EC2_HOST" "chmod +x /tmp/build.sh && /tmp/build.sh"
echo -e "${YELLOW}───────────────────────────────────────────${NC}"
echo -e "${GREEN}✅ EC2 build complete${NC}"
echo ""

# Deploy to EKS (from local machine)
echo -e "${BLUE}[5/5] Deploying to EKS from local machine...${NC}"
aws eks update-kubeconfig --name mini-xdr-cluster --region $AWS_REGION 2>/dev/null

echo "  Updating backend deployment..."
kubectl set image deployment/mini-xdr-backend \
    backend="$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/mini-xdr-backend:$VERSION" \
    -n $NAMESPACE

echo "  Updating frontend deployment..."
kubectl set image deployment/mini-xdr-frontend \
    frontend="$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/mini-xdr-frontend:$VERSION" \
    -n $NAMESPACE

echo "  Waiting for rollouts..."
kubectl rollout status deployment/mini-xdr-backend -n $NAMESPACE --timeout=5m &
BACKEND_PID=$!
kubectl rollout status deployment/mini-xdr-frontend -n $NAMESPACE --timeout=5m &
FRONTEND_PID=$!

wait $BACKEND_PID
wait $FRONTEND_PID

echo -e "${GREEN}✅ Deployment complete!${NC}"
echo ""

# Final verification
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}🎉 UI ROLLBACK SUCCESSFUL!${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "${GREEN}Deployed Images:${NC}"
echo "  Backend:  $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/mini-xdr-backend:$VERSION"
echo "  Frontend: $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/mini-xdr-frontend:$VERSION"
echo ""
echo -e "${GREEN}ALB URL:${NC}"
echo "  http://k8s-minixdr-minixdri-dc5fc1df8b-1132128475.us-east-1.elb.amazonaws.com"
echo ""
echo -e "${YELLOW}Note: Pods may take 1-2 minutes to fully restart${NC}"
echo ""

echo -e "${BLUE}Current Pod Status:${NC}"
kubectl get pods -n $NAMESPACE -o wide

echo ""
echo -e "${GREEN}Old UI is now live! 🎊${NC}"

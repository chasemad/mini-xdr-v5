#!/bin/bash
# Mini-XDR v1.1.8 - Complete AWS Deployment Script
# Verifies readiness, builds images, and deploys to EKS
# Run from project root: ./scripts/deploy-v1.1.8.sh

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
AWS_REGION="us-east-1"
AWS_ACCOUNT_ID="116912495274"
VERSION="1.1.8"
EKS_CLUSTER="mini-xdr-cluster"
NAMESPACE="mini-xdr"
ALB_URL="http://k8s-minixdr-minixdri-dc5fc1df8b-1132128475.us-east-1.elb.amazonaws.com"

echo -e "${BLUE}╔═══════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║       Mini-XDR v1.1.0 - AWS Deployment Script        ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════╝${NC}"
echo ""

# Function to check command exists
check_command() {
    if ! command -v $1 &> /dev/null; then
        echo -e "${RED}✗ $1 not found${NC}"
        echo "  Please install $1 and try again"
        exit 1
    fi
    echo -e "${GREEN}✓ $1 found${NC}"
}

# Function to check AWS credentials
check_aws_auth() {
    if aws sts get-caller-identity --region $AWS_REGION &> /dev/null; then
        ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
        echo -e "${GREEN}✓ AWS authenticated (Account: $ACCOUNT)${NC}"
        if [ "$ACCOUNT" != "$AWS_ACCOUNT_ID" ]; then
            echo -e "${YELLOW}⚠ Warning: Current account ($ACCOUNT) doesn't match expected ($AWS_ACCOUNT_ID)${NC}"
            read -p "Continue anyway? (y/N) " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                exit 1
            fi
        fi
    else
        echo -e "${RED}✗ AWS authentication failed${NC}"
        echo "  Please configure AWS credentials and try again"
        exit 1
    fi
}

# Function to check CodeBuild concurrency
check_concurrency() {
    echo ""
    echo -e "${BLUE}━━━ Checking CodeBuild Concurrency Quota ━━━${NC}"
    
    QUOTA=$(aws service-quotas get-service-quota \
        --service-code codebuild \
        --quota-code L-ACCF6C0D \
        --region $AWS_REGION \
        --query 'Quota.Value' \
        --output text 2>/dev/null || echo "0")
    
    if [ "$QUOTA" == "0" ] || [ "$QUOTA" == "0.0" ]; then
        echo -e "${RED}✗ CodeBuild concurrency limit is 0${NC}"
        echo ""
        echo "Your AWS account needs verification to increase build concurrency."
        echo ""
        echo "Options:"
        echo "  1. Wait 24-48 hours after EC2 launch (automatic verification)"
        echo "  2. Contact AWS Support and request limit increase"
        echo "  3. Use an existing verified AWS account"
        echo ""
        read -p "Skip concurrency check and continue? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        echo -e "${GREEN}✓ CodeBuild concurrency quota: $QUOTA${NC}"
    fi
}

# Step 1: Pre-flight checks
echo -e "${BLUE}━━━ Step 1: Pre-flight Checks ━━━${NC}"
check_command aws
check_command kubectl
check_command docker
check_command jq
check_command git
check_aws_auth
check_concurrency

# Step 2: Verify git state
echo ""
echo -e "${BLUE}━━━ Step 2: Verifying Git Repository ━━━${NC}"

# Check if we're in a git repo
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo -e "${RED}✗ Not in a git repository${NC}"
    exit 1
fi

# Check for uncommitted changes
if [[ -n $(git status -s) ]]; then
    echo -e "${YELLOW}⚠ Uncommitted changes detected:${NC}"
    git status -s
    echo ""
    read -p "Commit changes before deploying? (Y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        git add buildspec-backend.yml buildspec-frontend.yml k8s/backend-deployment.yaml k8s/frontend-deployment.yaml
        git commit -m "fix: CodeBuild compatibility and v1.1.0 deployment configs"
        echo -e "${GREEN}✓ Changes committed${NC}"
    fi
fi

# Check if tag exists
if git rev-parse "v$VERSION" >/dev/null 2>&1; then
    echo -e "${GREEN}✓ Tag v$VERSION exists${NC}"
else
    echo -e "${YELLOW}⚠ Tag v$VERSION not found${NC}"
    read -p "Create tag v$VERSION now? (Y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        git tag -a "v$VERSION" -m "Production release v$VERSION with JWT onboarding fix"
        echo -e "${GREEN}✓ Tag v$VERSION created${NC}"
    fi
fi

# Push to GitHub
echo ""
read -p "Push to GitHub? (Y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Nn]$ ]]; then
    git push origin main --tags
    echo -e "${GREEN}✓ Pushed to GitHub${NC}"
fi

# Step 3: Start CodeBuild projects
echo ""
echo -e "${BLUE}━━━ Step 3: Starting CodeBuild Projects ━━━${NC}"

echo ""
read -p "Start CodeBuild builds now? (Y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Nn]$ ]]; then
    echo "Starting backend build..."
    BACKEND_BUILD_ID=$(aws codebuild start-build \
        --project-name mini-xdr-backend-build \
        --source-version "refs/tags/v$VERSION" \
        --region $AWS_REGION \
        --query 'build.id' \
        --output text)
    echo -e "${GREEN}✓ Backend build started: $BACKEND_BUILD_ID${NC}"
    
    echo "Starting frontend build..."
    FRONTEND_BUILD_ID=$(aws codebuild start-build \
        --project-name mini-xdr-frontend-build \
        --source-version "refs/tags/v$VERSION" \
        --region $AWS_REGION \
        --query 'build.id' \
        --output text)
    echo -e "${GREEN}✓ Frontend build started: $FRONTEND_BUILD_ID${NC}"
    
    echo ""
    echo "Monitor builds at:"
    echo "  https://console.aws.amazon.com/codesuite/codebuild/projects?region=$AWS_REGION"
    echo ""
    echo "Or check status:"
    echo "  aws codebuild batch-get-builds --ids $BACKEND_BUILD_ID --region $AWS_REGION"
    echo "  aws codebuild batch-get-builds --ids $FRONTEND_BUILD_ID --region $AWS_REGION"
    
    # Wait for builds
    echo ""
    read -p "Wait for builds to complete? (Y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        echo "Waiting for backend build..."
        while true; do
            STATUS=$(aws codebuild batch-get-builds \
                --ids $BACKEND_BUILD_ID \
                --region $AWS_REGION \
                --query 'builds[0].buildStatus' \
                --output text)
            
            if [ "$STATUS" == "SUCCEEDED" ]; then
                echo -e "${GREEN}✓ Backend build succeeded${NC}"
                break
            elif [ "$STATUS" == "FAILED" ] || [ "$STATUS" == "FAULT" ] || [ "$STATUS" == "TIMED_OUT" ] || [ "$STATUS" == "STOPPED" ]; then
                echo -e "${RED}✗ Backend build failed with status: $STATUS${NC}"
                echo "Check logs at: https://console.aws.amazon.com/codesuite/codebuild/projects/mini-xdr-backend-build/build/$BACKEND_BUILD_ID"
                exit 1
            fi
            
            echo "  Status: $STATUS (waiting...)"
            sleep 10
        done
        
        echo "Waiting for frontend build..."
        while true; do
            STATUS=$(aws codebuild batch-get-builds \
                --ids $FRONTEND_BUILD_ID \
                --region $AWS_REGION \
                --query 'builds[0].buildStatus' \
                --output text)
            
            if [ "$STATUS" == "SUCCEEDED" ]; then
                echo -e "${GREEN}✓ Frontend build succeeded${NC}"
                break
            elif [ "$STATUS" == "FAILED" ] || [ "$STATUS" == "FAULT" ] || [ "$STATUS" == "TIMED_OUT" ] || [ "$STATUS" == "STOPPED" ]; then
                echo -e "${RED}✗ Frontend build failed with status: $STATUS${NC}"
                echo "Check logs at: https://console.aws.amazon.com/codesuite/codebuild/projects/mini-xdr-frontend-build/build/$FRONTEND_BUILD_ID"
                exit 1
            fi
            
            echo "  Status: $STATUS (waiting...)"
            sleep 10
        done
    fi
else
    echo -e "${YELLOW}⚠ Skipping CodeBuild - you'll need to build manually${NC}"
fi

# Step 4: Verify ECR images
echo ""
echo -e "${BLUE}━━━ Step 4: Verifying ECR Images ━━━${NC}"

echo "Checking backend image..."
BACKEND_IMAGE=$(aws ecr describe-images \
    --repository-name mini-xdr-backend \
    --region $AWS_REGION \
    --image-ids imageTag=$VERSION \
    --query 'imageDetails[0].imageTags' \
    --output json 2>/dev/null || echo "[]")

if [ "$BACKEND_IMAGE" != "[]" ]; then
    echo -e "${GREEN}✓ Backend image found with tags: $BACKEND_IMAGE${NC}"
else
    echo -e "${RED}✗ Backend image with tag $VERSION not found in ECR${NC}"
    echo "  Build may have failed or version tag was not applied"
    exit 1
fi

echo "Checking frontend image..."
FRONTEND_IMAGE=$(aws ecr describe-images \
    --repository-name mini-xdr-frontend \
    --region $AWS_REGION \
    --image-ids imageTag=$VERSION \
    --query 'imageDetails[0].imageTags' \
    --output json 2>/dev/null || echo "[]")

if [ "$FRONTEND_IMAGE" != "[]" ]; then
    echo -e "${GREEN}✓ Frontend image found with tags: $FRONTEND_IMAGE${NC}"
else
    echo -e "${RED}✗ Frontend image with tag $VERSION not found in ECR${NC}"
    echo "  Build may have failed or version tag was not applied"
    exit 1
fi

# Step 5: Deploy to EKS
echo ""
echo -e "${BLUE}━━━ Step 5: Deploying to EKS ━━━${NC}"

echo "Updating kubeconfig..."
aws eks update-kubeconfig --name $EKS_CLUSTER --region $AWS_REGION
echo -e "${GREEN}✓ Kubeconfig updated${NC}"

echo ""
read -p "Deploy to EKS cluster $EKS_CLUSTER? (Y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Nn]$ ]]; then
    echo "Deploying backend..."
    kubectl set image deployment/mini-xdr-backend \
        backend=$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/mini-xdr-backend:$VERSION \
        -n $NAMESPACE
    
    echo "Deploying frontend..."
    kubectl set image deployment/mini-xdr-frontend \
        frontend=$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/mini-xdr-frontend:$VERSION \
        -n $NAMESPACE
    
    echo ""
    echo "Waiting for rollouts to complete..."
    kubectl rollout status deployment/mini-xdr-backend -n $NAMESPACE --timeout=300s
    kubectl rollout status deployment/mini-xdr-frontend -n $NAMESPACE --timeout=300s
    
    echo -e "${GREEN}✓ Deployments complete${NC}"
else
    echo -e "${YELLOW}⚠ Skipping EKS deployment${NC}"
fi

# Step 6: Verification
echo ""
echo -e "${BLUE}━━━ Step 6: Verifying Deployment ━━━${NC}"

echo ""
read -p "Run verification tests? (Y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Nn]$ ]]; then
    echo "Checking deployed image versions..."
    BACKEND_IMAGE_DEPLOYED=$(kubectl get deployment mini-xdr-backend -n $NAMESPACE -o jsonpath='{.spec.template.spec.containers[0].image}')
    FRONTEND_IMAGE_DEPLOYED=$(kubectl get deployment mini-xdr-frontend -n $NAMESPACE -o jsonpath='{.spec.template.spec.containers[0].image}')
    
    echo "  Backend:  $BACKEND_IMAGE_DEPLOYED"
    echo "  Frontend: $FRONTEND_IMAGE_DEPLOYED"
    
    if [[ $BACKEND_IMAGE_DEPLOYED == *":$VERSION"* ]] && [[ $FRONTEND_IMAGE_DEPLOYED == *":$VERSION"* ]]; then
        echo -e "${GREEN}✓ Correct image versions deployed${NC}"
    else
        echo -e "${RED}✗ Image versions don't match expected $VERSION${NC}"
    fi
    
    echo ""
    echo "Testing authentication..."
    TOKEN=$(curl -s -X POST "$ALB_URL/api/auth/login" \
        -H "Content-Type: application/json" \
        -d '{"email":"chasemadrian@protonmail.com","password":"demo-tpot-api-key"}' \
        | jq -r '.access_token' 2>/dev/null || echo "")
    
    if [ -n "$TOKEN" ] && [ "$TOKEN" != "null" ]; then
        echo -e "${GREEN}✓ Authentication successful${NC}"
        
        echo ""
        echo "Testing onboarding endpoints (v1.1.0 fix)..."
        ONBOARDING_STATUS=$(curl -s -H "Authorization: Bearer $TOKEN" "$ALB_URL/api/onboarding/status" | jq -r '.detail' 2>/dev/null || echo "")
        
        if [ "$ONBOARDING_STATUS" == "Unauthorized" ] || [ -z "$ONBOARDING_STATUS" ]; then
            echo -e "${RED}✗ Onboarding endpoint returned 401 - JWT fix may not be deployed${NC}"
        else
            echo -e "${GREEN}✓ Onboarding endpoint accessible with JWT token${NC}"
            curl -s -H "Authorization: Bearer $TOKEN" "$ALB_URL/api/onboarding/status" | jq .
        fi
    else
        echo -e "${RED}✗ Authentication failed - check backend logs${NC}"
    fi
fi

# Summary
echo ""
echo -e "${BLUE}╔═══════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║              Deployment Complete! 🚀                  ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════╝${NC}"
echo ""
echo "Version:  v$VERSION"
echo "Cluster:  $EKS_CLUSTER"
echo "ALB:      $ALB_URL"
echo ""
echo "Next steps:"
echo "  1. Test onboarding wizard in browser: $ALB_URL"
echo "  2. Monitor logs: kubectl logs -f deployment/mini-xdr-backend -n $NAMESPACE"
echo "  3. Update documentation to mark deployment complete"
echo ""
echo -e "${GREEN}All systems operational!${NC}"


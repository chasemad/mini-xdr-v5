#!/bin/bash
# ============================================================================
# Mini-XDR MCP Server Deployment Script
# ============================================================================
# Builds, pushes, and deploys the MCP server to AWS EKS
# Usage: ./ops/deploy-mcp-server.sh [--build] [--push] [--deploy] [--all]
# ============================================================================

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AWS_REGION="${AWS_REGION:-us-east-1}"
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
ECR_REPO="mini-xdr-mcp-server"
IMAGE_TAG="${IMAGE_TAG:-latest}"
FULL_IMAGE="${ECR_REGISTRY}/${ECR_REPO}:${IMAGE_TAG}"

# Kubernetes configuration
K8S_NAMESPACE="mini-xdr"
DEPLOYMENT_NAME="mini-xdr-mcp-server"

# Logging functions
log() {
    echo -e "${BLUE}[$(date +'%H:%M:%S')]${NC} $1"
}

success() {
    echo -e "${GREEN}✅ $1${NC}"
}

warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

error() {
    echo -e "${RED}❌ $1${NC}"
}

# Parse command line arguments
DO_BUILD=false
DO_PUSH=false
DO_DEPLOY=false

if [ $# -eq 0 ]; then
    warning "No arguments provided. Use --all to build, push, and deploy."
    echo "Usage: $0 [--build] [--push] [--deploy] [--all]"
    exit 1
fi

for arg in "$@"; do
    case $arg in
        --build)
            DO_BUILD=true
            ;;
        --push)
            DO_PUSH=true
            ;;
        --deploy)
            DO_DEPLOY=true
            ;;
        --all)
            DO_BUILD=true
            DO_PUSH=true
            DO_DEPLOY=true
            ;;
        *)
            error "Unknown argument: $arg"
            exit 1
            ;;
    esac
done

# ============================================================================
# Step 1: Create ECR repository if it doesn't exist
# ============================================================================
log "Checking ECR repository..."
if ! aws ecr describe-repositories --repository-names "$ECR_REPO" --region "$AWS_REGION" &>/dev/null; then
    log "Creating ECR repository: $ECR_REPO"
    aws ecr create-repository \
        --repository-name "$ECR_REPO" \
        --region "$AWS_REGION" \
        --image-scanning-configuration scanOnPush=true \
        --encryption-configuration encryptionType=AES256
    success "ECR repository created"
else
    success "ECR repository exists"
fi

# ============================================================================
# Step 2: Build Docker image
# ============================================================================
if [ "$DO_BUILD" = true ]; then
    log "Building MCP server Docker image..."
    cd "$PROJECT_ROOT"

    # Build with buildkit for better caching
    export DOCKER_BUILDKIT=1

    docker build \
        -f ops/Dockerfile.mcp-server \
        -t "${ECR_REPO}:${IMAGE_TAG}" \
        -t "${FULL_IMAGE}" \
        --build-arg BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
        --build-arg VCS_REF="$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')" \
        --build-arg VERSION="${IMAGE_TAG}" \
        .

    success "Docker image built: ${FULL_IMAGE}"

    # Display image size
    IMAGE_SIZE=$(docker images --format "{{.Size}}" "${ECR_REPO}:${IMAGE_TAG}" | head -1)
    log "Image size: ${IMAGE_SIZE}"
fi

# ============================================================================
# Step 3: Push to ECR
# ============================================================================
if [ "$DO_PUSH" = true ]; then
    log "Authenticating with ECR..."
    aws ecr get-login-password --region "$AWS_REGION" | \
        docker login --username AWS --password-stdin "$ECR_REGISTRY"

    log "Pushing image to ECR..."
    docker push "${FULL_IMAGE}"

    success "Image pushed to ECR: ${FULL_IMAGE}"

    # Tag as latest if not already
    if [ "$IMAGE_TAG" != "latest" ]; then
        log "Tagging as latest..."
        docker tag "${FULL_IMAGE}" "${ECR_REGISTRY}/${ECR_REPO}:latest"
        docker push "${ECR_REGISTRY}/${ECR_REPO}:latest"
    fi
fi

# ============================================================================
# Step 4: Deploy to Kubernetes
# ============================================================================
if [ "$DO_DEPLOY" = true ]; then
    log "Deploying MCP server to Kubernetes..."

    # Check kubectl context
    CURRENT_CONTEXT=$(kubectl config current-context)
    log "Current kubectl context: $CURRENT_CONTEXT"

    # Verify we're targeting the right cluster
    if ! echo "$CURRENT_CONTEXT" | grep -q "mini-xdr"; then
        warning "Current context doesn't seem to be mini-xdr cluster"
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            error "Deployment cancelled"
            exit 1
        fi
    fi

    # Create namespace if it doesn't exist
    kubectl create namespace "$K8S_NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -

    # Apply manifests with image substitution
    log "Applying deployment manifest..."
    cat "$PROJECT_ROOT/ops/k8s/mcp-server-deployment.yaml" | \
        sed "s|\${AWS_ACCOUNT_ID}|${AWS_ACCOUNT_ID}|g" | \
        sed "s|\${AWS_REGION}|${AWS_REGION}|g" | \
        kubectl apply -f -

    log "Applying service manifest..."
    kubectl apply -f "$PROJECT_ROOT/ops/k8s/mcp-server-service.yaml"

    success "Manifests applied"

    # Wait for deployment to be ready
    log "Waiting for deployment to be ready..."
    kubectl rollout status deployment/"$DEPLOYMENT_NAME" -n "$K8S_NAMESPACE" --timeout=5m

    success "Deployment ready!"

    # Display pod status
    log "Pod status:"
    kubectl get pods -n "$K8S_NAMESPACE" -l app=mini-xdr-mcp-server

    # Display service info
    log "Service info:"
    kubectl get svc -n "$K8S_NAMESPACE" mcp-server-service

    # Check health
    log "Checking MCP server health..."
    MCP_POD=$(kubectl get pods -n "$K8S_NAMESPACE" -l app=mini-xdr-mcp-server -o jsonpath='{.items[0].metadata.name}')

    if kubectl exec -n "$K8S_NAMESPACE" "$MCP_POD" -- curl -sf http://localhost:3001/health > /dev/null; then
        success "MCP server is healthy!"
    else
        warning "MCP server health check failed"
        log "Check logs with: kubectl logs -n $K8S_NAMESPACE $MCP_POD"
    fi

    # Display logs
    log "Recent logs:"
    kubectl logs -n "$K8S_NAMESPACE" "$MCP_POD" --tail=20
fi

# ============================================================================
# Summary
# ============================================================================
echo ""
log "🎉 MCP Server Deployment Complete!"
echo ""
echo "Next steps:"
echo "  1. Update ALB ingress to route /mcp traffic:"
echo "     kubectl edit ingress mini-xdr-ingress -n mini-xdr"
echo ""
echo "  2. Connect Claude Code to MCP server:"
echo "     claude mcp add --transport http mini-xdr \\"
echo "       http://your-alb-url/mcp"
echo ""
echo "  3. Test MCP server:"
echo "     kubectl port-forward -n mini-xdr svc/mcp-server-service 3001:3001"
echo "     curl http://localhost:3001/health"
echo ""
echo "  4. View logs:"
echo "     kubectl logs -n mini-xdr -l app=mini-xdr-mcp-server -f"
echo ""

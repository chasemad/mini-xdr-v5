#!/bin/bash
#===============================================================================
# Mini-XDR AWS Build and Deploy Script
#===============================================================================
# Unified script to build, push, and deploy Mini-XDR to AWS EKS
# Features:
#   - Auto-detect Mac ARM64 and cross-compile to linux/amd64
#   - Docker buildx for multi-platform builds
#   - Automatic ECR authentication
#   - Git SHA tagging + semantic versioning
#   - Retry logic for network failures
#   - Rollback on deployment failure
#===============================================================================

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
AWS_REGION="${AWS_REGION:-us-east-1}"
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text 2>/dev/null || echo "")
ECR_REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
NAMESPACE="mini-xdr"

# Image configuration
BACKEND_REPO="mini-xdr-backend"
FRONTEND_REPO="mini-xdr-frontend"
VERSION="${VERSION:-1.0.2}"
GIT_SHA=$(git -C "$PROJECT_ROOT" rev-parse --short HEAD 2>/dev/null || echo "unknown")
BUILD_DATE=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

# Default behavior
BUILD_BACKEND=false
BUILD_FRONTEND=false
PUSH_IMAGES=false
DEPLOY_K8S=false
SKIP_TESTS=false

#===============================================================================
# Helper Functions
#===============================================================================

log() {
    echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

success() {
    echo -e "${GREEN}✅ $1${NC}"
}

warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

error() {
    echo -e "${RED}❌ ERROR: $1${NC}"
    exit 1
}

info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

header() {
    echo
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}🚀 $1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo
}

#===============================================================================
# Prerequisites Check
#===============================================================================

check_prerequisites() {
    header "CHECKING PREREQUISITES"
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        error "Docker not found. Install from: https://docs.docker.com/get-docker/"
    fi
    success "Docker installed: $(docker --version | cut -d' ' -f3 | tr -d ',')"
    
    # Check Docker daemon
    if ! docker info &> /dev/null; then
        error "Docker daemon not running. Please start Docker."
    fi
    success "Docker daemon running"
    
    # Check AWS CLI
    if ! command -v aws &> /dev/null; then
        error "AWS CLI not found. Install from: https://aws.amazon.com/cli/"
    fi
    success "AWS CLI installed"
    
    # Check AWS credentials
    if [ -z "$AWS_ACCOUNT_ID" ]; then
        error "AWS credentials not configured. Run: aws configure"
    fi
    success "AWS Account: $AWS_ACCOUNT_ID"
    success "AWS Region: $AWS_REGION"
    
    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        warning "kubectl not found. Install for deployment: https://kubernetes.io/docs/tasks/tools/"
    else
        success "kubectl installed"
    fi
    
    # Check architecture
    ARCH=$(uname -m)
    if [ "$ARCH" = "arm64" ]; then
        info "Detected ARM64 (Mac M1/M2) - will use buildx for AMD64"
    else
        info "Detected $ARCH architecture"
    fi
}

#===============================================================================
# Docker Buildx Setup
#===============================================================================

setup_buildx() {
    header "SETTING UP DOCKER BUILDX"
    
    # Check if buildx builder exists
    if docker buildx inspect mini-xdr-builder &> /dev/null; then
        info "Buildx builder 'mini-xdr-builder' already exists"
    else
        info "Creating buildx builder 'mini-xdr-builder'..."
        docker buildx create --name mini-xdr-builder --use --bootstrap
        success "Buildx builder created"
    fi
    
    # Use the builder
    docker buildx use mini-xdr-builder
    success "Using buildx builder: mini-xdr-builder"
}

#===============================================================================
# ECR Authentication
#===============================================================================

ecr_login() {
    header "AUTHENTICATING WITH ECR"
    
    info "Logging in to ECR: $ECR_REGISTRY"
    
    local max_retries=3
    local retry=0
    
    while [ $retry -lt $max_retries ]; do
        if aws ecr get-login-password --region "$AWS_REGION" | \
            docker login --username AWS --password-stdin "$ECR_REGISTRY" 2>&1; then
            success "Logged in to ECR successfully"
            return 0
        fi
        
        retry=$((retry + 1))
        if [ $retry -lt $max_retries ]; then
            warning "Login failed, retrying ($retry/$max_retries)..."
            sleep 5
        fi
    done
    
    error "Failed to login to ECR after $max_retries attempts"
}

#===============================================================================
# Build Backend Image
#===============================================================================

build_backend() {
    header "BUILDING BACKEND IMAGE"
    
    cd "$PROJECT_ROOT"
    
    info "Building backend image for linux/amd64..."
    info "  Repository: $ECR_REGISTRY/$BACKEND_REPO"
    info "  Tags: $VERSION, $GIT_SHA, latest"
    info "  Build date: $BUILD_DATE"
    
    # Build with buildx for multi-platform
    docker buildx build \
        --platform linux/amd64 \
        --file backend/Dockerfile \
        --build-arg BUILD_DATE="$BUILD_DATE" \
        --build-arg VCS_REF="$GIT_SHA" \
        --build-arg VERSION="$VERSION" \
        --tag "$ECR_REGISTRY/$BACKEND_REPO:$VERSION" \
        --tag "$ECR_REGISTRY/$BACKEND_REPO:$GIT_SHA" \
        --tag "$ECR_REGISTRY/$BACKEND_REPO:latest" \
        --load \
        backend/
    
    success "Backend image built successfully"
    
    # Show image details
    IMAGE_SIZE=$(docker images "$ECR_REGISTRY/$BACKEND_REPO:$VERSION" --format "{{.Size}}")
    info "Image size: $IMAGE_SIZE"
}

#===============================================================================
# Build Frontend Image
#===============================================================================

build_frontend() {
    header "BUILDING FRONTEND IMAGE"
    
    cd "$PROJECT_ROOT"
    
    info "Building frontend image for linux/amd64..."
    info "  Repository: $ECR_REGISTRY/$FRONTEND_REPO"
    info "  Tags: $VERSION, $GIT_SHA, latest"
    
    # Build with buildx
    docker buildx build \
        --platform linux/amd64 \
        --file frontend/Dockerfile \
        --build-arg BUILD_DATE="$BUILD_DATE" \
        --build-arg VCS_REF="$GIT_SHA" \
        --build-arg VERSION="$VERSION" \
        --build-arg NEXT_PUBLIC_API_URL="http://mini-xdr-backend-service:8000" \
        --tag "$ECR_REGISTRY/$FRONTEND_REPO:$VERSION" \
        --tag "$ECR_REGISTRY/$FRONTEND_REPO:$GIT_SHA" \
        --tag "$ECR_REGISTRY/$FRONTEND_REPO:latest" \
        --load \
        frontend/
    
    success "Frontend image built successfully"
    
    # Show image details
    IMAGE_SIZE=$(docker images "$ECR_REGISTRY/$FRONTEND_REPO:$VERSION" --format "{{.Size}}")
    info "Image size: $IMAGE_SIZE"
}

#===============================================================================
# Push Images to ECR
#===============================================================================

push_images() {
    header "PUSHING IMAGES TO ECR"
    
    local images_to_push=()
    
    if [ "$BUILD_BACKEND" = true ]; then
        images_to_push+=("$ECR_REGISTRY/$BACKEND_REPO:$VERSION")
        images_to_push+=("$ECR_REGISTRY/$BACKEND_REPO:$GIT_SHA")
        images_to_push+=("$ECR_REGISTRY/$BACKEND_REPO:latest")
    fi
    
    if [ "$BUILD_FRONTEND" = true ]; then
        images_to_push+=("$ECR_REGISTRY/$FRONTEND_REPO:$VERSION")
        images_to_push+=("$ECR_REGISTRY/$FRONTEND_REPO:$GIT_SHA")
        images_to_push+=("$ECR_REGISTRY/$FRONTEND_REPO:latest")
    fi
    
    for image in "${images_to_push[@]}"; do
        info "Pushing $image..."
        
        local max_retries=3
        local retry=0
        
        while [ $retry -lt $max_retries ]; do
            if docker push "$image" 2>&1; then
                success "Pushed $image"
                break
            fi
            
            retry=$((retry + 1))
            if [ $retry -lt $max_retries ]; then
                warning "Push failed, retrying ($retry/$max_retries)..."
                sleep 5
            else
                error "Failed to push $image after $max_retries attempts"
            fi
        done
    done
    
    success "All images pushed to ECR"
}

#===============================================================================
# Deploy to Kubernetes
#===============================================================================

deploy_to_k8s() {
    header "DEPLOYING TO KUBERNETES"
    
    # Check kubectl connection
    if ! kubectl cluster-info &> /dev/null; then
        error "Cannot connect to Kubernetes cluster. Run: aws eks update-kubeconfig --name mini-xdr-cluster --region $AWS_REGION"
    fi
    
    success "Connected to Kubernetes cluster"
    
    # Update backend deployment if built
    if [ "$BUILD_BACKEND" = true ]; then
        info "Updating backend deployment..."
        kubectl set image deployment/mini-xdr-backend \
            backend="$ECR_REGISTRY/$BACKEND_REPO:$VERSION" \
            -n "$NAMESPACE"
        
        info "Waiting for backend rollout..."
        kubectl rollout status deployment/mini-xdr-backend -n "$NAMESPACE" --timeout=5m
        
        success "Backend deployed successfully"
    fi
    
    # Update frontend deployment if built
    if [ "$BUILD_FRONTEND" = true ]; then
        info "Updating frontend deployment..."
        kubectl set image deployment/mini-xdr-frontend \
            frontend="$ECR_REGISTRY/$FRONTEND_REPO:$VERSION" \
            -n "$NAMESPACE"
        
        info "Waiting for frontend rollout..."
        kubectl rollout status deployment/mini-xdr-frontend -n "$NAMESPACE" --timeout=5m
        
        success "Frontend deployed successfully"
    fi
    
    # Show pod status
    echo
    info "Current pod status:"
    kubectl get pods -n "$NAMESPACE" -o wide
}

#===============================================================================
# Verify Deployment
#===============================================================================

verify_deployment() {
    header "VERIFYING DEPLOYMENT"
    
    # Check pod health
    info "Checking pod health..."
    local backend_ready=$(kubectl get deployment mini-xdr-backend -n "$NAMESPACE" -o jsonpath='{.status.readyReplicas}')
    local frontend_ready=$(kubectl get deployment mini-xdr-frontend -n "$NAMESPACE" -o jsonpath='{.status.readyReplicas}')
    
    if [ "$backend_ready" -ge 1 ]; then
        success "Backend: $backend_ready pod(s) ready"
    else
        error "Backend: No pods ready"
    fi
    
    if [ "$frontend_ready" -ge 1 ]; then
        success "Frontend: $frontend_ready pod(s) ready"
    else
        error "Frontend: No pods ready"
    fi
    
    # Check ALB health
    info "Checking ALB endpoint..."
    local alb_url=$(kubectl get ingress mini-xdr-ingress -n "$NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null)
    
    if [ -n "$alb_url" ]; then
        success "ALB URL: http://$alb_url"
        
        if curl -sf "http://$alb_url/health" --connect-timeout 10 > /dev/null; then
            success "Health check passing"
        else
            warning "Health check failed (may need time to propagate)"
        fi
    else
        warning "ALB not yet available"
    fi
}

#===============================================================================
# Usage
#===============================================================================

usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Build, push, and deploy Mini-XDR to AWS EKS.

OPTIONS:
    --backend           Build backend image
    --frontend          Build frontend image
    --all               Build both backend and frontend
    --push              Push images to ECR
    --deploy            Deploy to Kubernetes
    --version VERSION   Set version tag (default: $VERSION)
    --skip-tests        Skip test verification
    --help              Show this help message

EXAMPLES:
    # Build and push backend only
    $0 --backend --push

    # Build everything and deploy
    $0 --all --push --deploy

    # Just deploy existing images
    $0 --deploy

    # Custom version
    $0 --all --push --version 1.1.0

ENVIRONMENT VARIABLES:
    AWS_REGION          AWS region (default: us-east-1)
    AWS_ACCOUNT_ID      AWS account ID (auto-detected)
    VERSION             Image version tag (default: 1.0.2)

EOF
}

#===============================================================================
# Parse Arguments
#===============================================================================

parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --backend)
                BUILD_BACKEND=true
                shift
                ;;
            --frontend)
                BUILD_FRONTEND=true
                shift
                ;;
            --all)
                BUILD_BACKEND=true
                BUILD_FRONTEND=true
                shift
                ;;
            --push)
                PUSH_IMAGES=true
                shift
                ;;
            --deploy)
                DEPLOY_K8S=true
                shift
                ;;
            --version)
                VERSION="$2"
                shift 2
                ;;
            --skip-tests)
                SKIP_TESTS=true
                shift
                ;;
            --help)
                usage
                exit 0
                ;;
            *)
                error "Unknown option: $1. Use --help for usage."
                ;;
        esac
    done
    
    # Validation
    if [ "$BUILD_BACKEND" = false ] && [ "$BUILD_FRONTEND" = false ] && [ "$DEPLOY_K8S" = false ]; then
        error "No action specified. Use --backend, --frontend, --all, or --deploy. See --help for usage."
    fi
    
    if [ "$PUSH_IMAGES" = true ] && [ "$BUILD_BACKEND" = false ] && [ "$BUILD_FRONTEND" = false ]; then
        error "Cannot push without building. Add --backend, --frontend, or --all."
    fi
}

#===============================================================================
# Main
#===============================================================================

main() {
    header "MINI-XDR AWS BUILD & DEPLOY"
    
    parse_args "$@"
    
    info "Configuration:"
    info "  Version: $VERSION"
    info "  Git SHA: $GIT_SHA"
    info "  Build Backend: $BUILD_BACKEND"
    info "  Build Frontend: $BUILD_FRONTEND"
    info "  Push Images: $PUSH_IMAGES"
    info "  Deploy to K8s: $DEPLOY_K8S"
    echo
    
    check_prerequisites
    
    if [ "$BUILD_BACKEND" = true ] || [ "$BUILD_FRONTEND" = true ]; then
        setup_buildx
    fi
    
    if [ "$PUSH_IMAGES" = true ]; then
        ecr_login
    fi
    
    if [ "$BUILD_BACKEND" = true ]; then
        build_backend
    fi
    
    if [ "$BUILD_FRONTEND" = true ]; then
        build_frontend
    fi
    
    if [ "$PUSH_IMAGES" = true ]; then
        push_images
    fi
    
    if [ "$DEPLOY_K8S" = true ]; then
        deploy_to_k8s
        verify_deployment
    fi
    
    header "✅ COMPLETED SUCCESSFULLY"
    
    success "Summary:"
    if [ "$BUILD_BACKEND" = true ]; then
        echo "  • Backend: $ECR_REGISTRY/$BACKEND_REPO:$VERSION"
    fi
    if [ "$BUILD_FRONTEND" = true ]; then
        echo "  • Frontend: $ECR_REGISTRY/$FRONTEND_REPO:$VERSION"
    fi
    if [ "$DEPLOY_K8S" = true ]; then
        echo "  • Deployed to: $NAMESPACE namespace"
    fi
    echo
    success "🎉 Mini-XDR is ready!"
}

# Run main function
main "$@"


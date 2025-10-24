#!/bin/bash
# ============================================================================
# Mini-XDR Docker Build and Push Script
# ============================================================================
# Builds optimized Docker images and pushes to AWS ECR
# - Creates ECR repositories if they don't exist
# - Builds production images (CPU-only, no training data)
# - Tags with git commit SHA and 'latest'
# - Pushes to ECR with proper authentication
# ============================================================================

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
AWS_REGION="${AWS_REGION:-us-east-1}"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REGISTRY="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

# Image names
BACKEND_IMAGE="mini-xdr-backend"
FRONTEND_IMAGE="mini-xdr-frontend"

# Get git commit SHA for tagging
GIT_SHA=$(git -C "$PROJECT_ROOT" rev-parse --short HEAD 2>/dev/null || echo "unknown")
BUILD_DATE=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

# Logging
LOG_FILE="/tmp/mini-xdr-build-$(date +%Y%m%d-%H%M%S).log"
exec 1> >(tee -a "$LOG_FILE")
exec 2> >(tee -a "$LOG_FILE" >&2)

# ============================================================================
# Helper Functions
# ============================================================================

log() {
    echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

success() {
    log "${GREEN}✅ $1${NC}"
}

warning() {
    log "${YELLOW}⚠️  $1${NC}"
}

error() {
    log "${RED}❌ ERROR: $1${NC}"
    exit 1
}

info() {
    log "${BLUE}ℹ️  $1${NC}"
}

header() {
    echo
    log "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    log "${BLUE}🚀 $1${NC}"
    log "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo
}

# ============================================================================
# Prerequisites
# ============================================================================

check_prerequisites() {
    header "CHECKING PREREQUISITES"

    # Check Docker
    if ! command -v docker &> /dev/null; then
        error "Docker not found. Please install Docker: https://docs.docker.com/get-docker/"
    fi
    success "Docker installed: $(docker --version)"

    # Check Docker daemon
    if ! docker info &> /dev/null; then
        error "Docker daemon not running. Please start Docker."
    fi
    success "Docker daemon running"

    # Check AWS CLI
    if ! command -v aws &> /dev/null; then
        error "AWS CLI not found. Please install: https://aws.amazon.com/cli/"
    fi
    success "AWS CLI installed"

    # Check AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        error "AWS credentials not configured. Run: aws configure"
    fi
    success "AWS Account ID: $ACCOUNT_ID"
    success "AWS Region: $AWS_REGION"

    # Check git (for commit SHA)
    if ! command -v git &> /dev/null; then
        warning "Git not found. Will use 'unknown' as version tag"
    else
        success "Git SHA: $GIT_SHA"
    fi
}

# ============================================================================
# ECR Repository Setup
# ============================================================================

setup_ecr_repositories() {
    header "SETTING UP ECR REPOSITORIES"

    for repo in "$BACKEND_IMAGE" "$FRONTEND_IMAGE"; do
        info "Checking if ECR repository '$repo' exists..."

        if aws ecr describe-repositories --repository-names "$repo" --region "$AWS_REGION" &> /dev/null; then
            success "Repository '$repo' already exists"
        else
            info "Creating ECR repository '$repo'..."

            aws ecr create-repository \
                --repository-name "$repo" \
                --region "$AWS_REGION" \
                --image-scanning-configuration scanOnPush=true \
                --encryption-configuration encryptionType=AES256 \
                --tags Key=Project,Value=mini-xdr Key=Environment,Value=production \
                > /dev/null

            success "Repository '$repo' created"

            # Set lifecycle policy to keep only 10 most recent images
            info "Setting lifecycle policy for '$repo'..."

            aws ecr put-lifecycle-policy \
                --repository-name "$repo" \
                --region "$AWS_REGION" \
                --lifecycle-policy-text '{
                    "rules": [{
                        "rulePriority": 1,
                        "description": "Keep only 10 most recent images",
                        "selection": {
                            "tagStatus": "any",
                            "countType": "imageCountMoreThan",
                            "countNumber": 10
                        },
                        "action": {
                            "type": "expire"
                        }
                    }]
                }' > /dev/null

            success "Lifecycle policy set for '$repo'"
        fi
    done

    success "All ECR repositories ready"
}

# ============================================================================
# ECR Authentication
# ============================================================================

ecr_login() {
    header "AUTHENTICATING WITH ECR"

    info "Logging in to ECR registry: $ECR_REGISTRY"

    aws ecr get-login-password --region "$AWS_REGION" | \
        docker login --username AWS --password-stdin "$ECR_REGISTRY"

    success "Logged in to ECR successfully"
}

# ============================================================================
# Build Docker Images
# ============================================================================

build_backend_image() {
    header "BUILDING BACKEND IMAGE (CPU-Only, No Training Data)"

    cd "$PROJECT_ROOT"

    info "Building backend image..."
    info "  Context: $PROJECT_ROOT"
    info "  Dockerfile: ops/Dockerfile.backend.production"
    info "  Image: $ECR_REGISTRY/$BACKEND_IMAGE:$GIT_SHA"
    info "  Using default .dockerignore (excludes frontend)"

    # Build with production Dockerfile (uses default .dockerignore)
    docker build \
        --no-cache \
        --file ops/Dockerfile.backend.production \
        --build-arg BUILD_DATE="$BUILD_DATE" \
        --build-arg VCS_REF="$GIT_SHA" \
        --build-arg VERSION="1.0.0" \
        --tag "$ECR_REGISTRY/$BACKEND_IMAGE:$GIT_SHA" \
        --tag "$ECR_REGISTRY/$BACKEND_IMAGE:latest" \
        .

    success "Backend image built successfully"

    # Show image size
    IMAGE_SIZE=$(docker images "$ECR_REGISTRY/$BACKEND_IMAGE:$GIT_SHA" --format "{{.Size}}")
    info "Image size: $IMAGE_SIZE"

    # Verify no training data in image
    info "Verifying no training data in image..."
    if docker run --rm "$ECR_REGISTRY/$BACKEND_IMAGE:$GIT_SHA" \
        sh -c "find /app -name '*training_data*.csv' 2>/dev/null | wc -l" | grep -q "^0$"; then
        success "✅ No training data found in image"
    else
        error "❌ Training data detected in image! Build failed."
    fi
}

build_frontend_image() {
    header "BUILDING FRONTEND IMAGE (Optimized Production)"

    cd "$PROJECT_ROOT"

    info "Building frontend image..."
    info "  Context: $PROJECT_ROOT"
    info "  Dockerfile: ops/Dockerfile.frontend.production"
    info "  Image: $ECR_REGISTRY/$FRONTEND_IMAGE:$GIT_SHA"

    # Temporarily swap .dockerignore to include frontend
    info "Swapping to frontend .dockerignore (includes frontend directory)..."
    if [ -f .dockerignore ]; then
        mv .dockerignore .dockerignore.backend.tmp
    fi
    if [ -f .dockerignore.frontend ]; then
        cp .dockerignore.frontend .dockerignore
    fi

    # Build with production Dockerfile
    docker build \
        --no-cache \
        --file ops/Dockerfile.frontend.production \
        --build-arg BUILD_DATE="$BUILD_DATE" \
        --build-arg VCS_REF="$GIT_SHA" \
        --build-arg VERSION="1.0.0" \
        --build-arg NEXT_PUBLIC_API_URL="http://mini-xdr-backend-service:8000" \
        --tag "$ECR_REGISTRY/$FRONTEND_IMAGE:$GIT_SHA" \
        --tag "$ECR_REGISTRY/$FRONTEND_IMAGE:latest" \
        .

    # Restore original .dockerignore
    info "Restoring original .dockerignore..."
    rm -f .dockerignore
    if [ -f .dockerignore.backend.tmp ]; then
        mv .dockerignore.backend.tmp .dockerignore
    fi

    success "Frontend image built successfully"

    # Show image size
    IMAGE_SIZE=$(docker images "$ECR_REGISTRY/$FRONTEND_IMAGE:$GIT_SHA" --format "{{.Size}}")
    info "Image size: $IMAGE_SIZE"
}

# ============================================================================
# Push Images to ECR
# ============================================================================

push_images() {
    header "PUSHING IMAGES TO ECR"

    # Push backend
    info "Pushing backend image..."
    docker push "$ECR_REGISTRY/$BACKEND_IMAGE:$GIT_SHA"
    docker push "$ECR_REGISTRY/$BACKEND_IMAGE:latest"
    success "Backend image pushed: $ECR_REGISTRY/$BACKEND_IMAGE:$GIT_SHA"

    # Push frontend
    info "Pushing frontend image..."
    docker push "$ECR_REGISTRY/$FRONTEND_IMAGE:$GIT_SHA"
    docker push "$ECR_REGISTRY/$FRONTEND_IMAGE:latest"
    success "Frontend image pushed: $ECR_REGISTRY/$FRONTEND_IMAGE:$GIT_SHA"
}

# ============================================================================
# Cleanup (optional)
# ============================================================================

cleanup_local_images() {
    header "CLEANUP (Optional)"

    read -p "Do you want to remove local images to free up space? (y/N): " -n 1 -r
    echo

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        info "Removing local images..."

        docker rmi "$ECR_REGISTRY/$BACKEND_IMAGE:$GIT_SHA" 2>/dev/null || true
        docker rmi "$ECR_REGISTRY/$BACKEND_IMAGE:latest" 2>/dev/null || true
        docker rmi "$ECR_REGISTRY/$FRONTEND_IMAGE:$GIT_SHA" 2>/dev/null || true
        docker rmi "$ECR_REGISTRY/$FRONTEND_IMAGE:latest" 2>/dev/null || true

        success "Local images removed"
    else
        info "Keeping local images"
    fi
}

# ============================================================================
# Summary
# ============================================================================

print_summary() {
    header "BUILD AND PUSH COMPLETE ✅"

    success "Images pushed to ECR:"
    echo ""
    echo "  Backend:"
    echo "    $ECR_REGISTRY/$BACKEND_IMAGE:$GIT_SHA"
    echo "    $ECR_REGISTRY/$BACKEND_IMAGE:latest"
    echo ""
    echo "  Frontend:"
    echo "    $ECR_REGISTRY/$FRONTEND_IMAGE:$GIT_SHA"
    echo "    $ECR_REGISTRY/$FRONTEND_IMAGE:latest"
    echo ""

    info "📋 Next Steps:"
    echo ""
    echo "  1. Update K8s manifests with ECR image URLs:"
    echo "     export AWS_ACCOUNT_ID=$ACCOUNT_ID"
    echo "     export AWS_REGION=$AWS_REGION"
    echo "     envsubst < ops/k8s/backend-deployment-production.yaml | kubectl apply -f -"
    echo ""
    echo "  2. Deploy to EKS:"
    echo "     ./infrastructure/aws/deploy-to-eks.sh"
    echo ""

    info "📝 Log file saved to: $LOG_FILE"
}

# ============================================================================
# Main Execution
# ============================================================================

main() {
    header "MINI-XDR DOCKER BUILD AND PUSH"

    info "This script will build and push Docker images to AWS ECR"
    info "Git SHA: $GIT_SHA"
    info "Build Date: $BUILD_DATE"
    echo ""

    check_prerequisites
    setup_ecr_repositories
    ecr_login
    build_backend_image
    build_frontend_image
    push_images
    cleanup_local_images
    print_summary

    success "🎉 Build and push completed successfully!"
}

# Run main function
main "$@"

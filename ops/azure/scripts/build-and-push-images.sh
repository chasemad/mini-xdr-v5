#!/bin/bash
# ============================================================================
# Mini-XDR Docker Image Build and Push Script
# ============================================================================
# Builds Docker images and pushes them to Azure Container Registry
# ============================================================================

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
TERRAFORM_DIR="$PROJECT_ROOT/ops/azure/terraform"

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║       Mini-XDR Docker Image Build & Push                       ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Get ACR name from Terraform or argument
if [ -n "$1" ]; then
    ACR_NAME="$1"
elif [ -f "$TERRAFORM_DIR/terraform.tfstate" ]; then
    ACR_LOGIN_SERVER=$(terraform -chdir="$TERRAFORM_DIR" output -raw acr_login_server)
    ACR_NAME="${ACR_LOGIN_SERVER%%.*}"
else
    echo -e "${YELLOW}Usage: $0 <acr-name>${NC}"
    echo -e "${YELLOW}Or run after Terraform deployment${NC}"
    exit 1
fi

# Version tag (default to latest if not specified)
VERSION="${VERSION:-latest}"

echo "ACR Name: $ACR_NAME"
echo "ACR Login Server: ${ACR_NAME}.azurecr.io"
echo "Version Tag: $VERSION"
echo ""

# Login to ACR
echo "Logging into Azure Container Registry..."
az acr login --name "$ACR_NAME"
echo -e "${GREEN}✅ Logged in successfully${NC}"
echo ""

cd "$PROJECT_ROOT"

# Build backend image
echo "Building backend image..."
docker build \
    -f ops/Dockerfile.backend \
    -t "${ACR_NAME}.azurecr.io/mini-xdr-backend:${VERSION}" \
    -t "${ACR_NAME}.azurecr.io/mini-xdr-backend:latest" \
    --build-arg BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
    --build-arg VCS_REF="$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')" \
    .
echo -e "${GREEN}✅ Backend image built${NC}"

# Build frontend image
echo "Building frontend image..."
docker build \
    -f ops/Dockerfile.frontend \
    -t "${ACR_NAME}.azurecr.io/mini-xdr-frontend:${VERSION}" \
    -t "${ACR_NAME}.azurecr.io/mini-xdr-frontend:latest" \
    --build-arg BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
    --build-arg VCS_REF="$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')" \
    .
echo -e "${GREEN}✅ Frontend image built${NC}"

# Build ingestion agent image
echo "Building ingestion agent image..."
docker build \
    -f ops/Dockerfile.ingestion-agent \
    -t "${ACR_NAME}.azurecr.io/mini-xdr-agent:${VERSION}" \
    -t "${ACR_NAME}.azurecr.io/mini-xdr-agent:latest" \
    --build-arg BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
    --build-arg VCS_REF="$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')" \
    .
echo -e "${GREEN}✅ Ingestion agent image built${NC}"
echo ""

# Push images
echo "Pushing images to ACR..."
echo "This may take several minutes..."
echo ""

echo "Pushing backend images..."
docker push "${ACR_NAME}.azurecr.io/mini-xdr-backend:${VERSION}"
docker push "${ACR_NAME}.azurecr.io/mini-xdr-backend:latest"
echo -e "${GREEN}✅ Backend images pushed${NC}"

echo "Pushing frontend images..."
docker push "${ACR_NAME}.azurecr.io/mini-xdr-frontend:${VERSION}"
docker push "${ACR_NAME}.azurecr.io/mini-xdr-frontend:latest"
echo -e "${GREEN}✅ Frontend images pushed${NC}"

echo "Pushing ingestion agent images..."
docker push "${ACR_NAME}.azurecr.io/mini-xdr-agent:${VERSION}"
docker push "${ACR_NAME}.azurecr.io/mini-xdr-agent:latest"
echo -e "${GREEN}✅ Ingestion agent images pushed${NC}"
echo ""

# Display summary
echo -e "${GREEN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║              Images Built and Pushed Successfully!             ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "Images available in ACR:"
echo "  • ${ACR_NAME}.azurecr.io/mini-xdr-backend:${VERSION}"
echo "  • ${ACR_NAME}.azurecr.io/mini-xdr-frontend:${VERSION}"
echo "  • ${ACR_NAME}.azurecr.io/mini-xdr-agent:${VERSION}"
echo ""
echo "List all images:"
echo "  az acr repository list --name $ACR_NAME --output table"
echo ""


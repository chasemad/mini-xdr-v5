#!/bin/bash
# ============================================================================
# Mini-XDR AKS Deployment Script
# ============================================================================
# Deploys Mini-XDR application to Azure Kubernetes Service
# ============================================================================

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
TERRAFORM_DIR="$PROJECT_ROOT/ops/azure/terraform"
K8S_DIR="$PROJECT_ROOT/ops/k8s"

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║       Mini-XDR Application Deployment to AKS                   ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Get Terraform outputs
if [ ! -f "$TERRAFORM_DIR/terraform.tfstate" ]; then
    echo -e "${RED}❌ Terraform state not found. Run infrastructure deployment first.${NC}"
    exit 1
fi

ACR_LOGIN_SERVER=$(terraform -chdir="$TERRAFORM_DIR" output -raw acr_login_server)
KEY_VAULT_NAME=$(terraform -chdir="$TERRAFORM_DIR" output -raw key_vault_name)
POSTGRES_CONNECTION=$(az keyvault secret show --vault-name "$KEY_VAULT_NAME" --name "postgres-connection-string" --query value -o tsv 2>/dev/null || echo "")
REDIS_CONNECTION=$(az keyvault secret show --vault-name "$KEY_VAULT_NAME" --name "redis-connection-string" --query value -o tsv 2>/dev/null || echo "")

echo "Deployment Configuration:"
echo "  • ACR: $ACR_LOGIN_SERVER"
echo "  • Key Vault: $KEY_VAULT_NAME"
echo "  • PostgreSQL: $([ -n "$POSTGRES_CONNECTION" ] && echo 'Configured' || echo 'Not configured')"
echo "  • Redis: $([ -n "$REDIS_CONNECTION" ] && echo 'Configured' || echo 'Not configured')"
echo ""

# Create namespace
echo "Creating mini-xdr namespace..."
kubectl create namespace mini-xdr --dry-run=client -o yaml | kubectl apply -f -
echo -e "${GREEN}✅ Namespace created${NC}"

# Create ConfigMap
echo "Creating ConfigMap..."
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: ConfigMap
metadata:
  name: mini-xdr-config
  namespace: mini-xdr
data:
  API_HOST: "0.0.0.0"
  API_PORT: "8000"
  DATABASE_URL: "${POSTGRES_CONNECTION:-sqlite:///./xdr.db}"
  REDIS_URL: "${REDIS_CONNECTION:-redis://localhost:6379}"
EOF
echo -e "${GREEN}✅ ConfigMap created${NC}"

# Create Secrets from Key Vault
echo "Creating Kubernetes secrets from Azure Key Vault..."

# Get secrets from Key Vault
OPENAI_KEY=$(az keyvault secret show --vault-name "$KEY_VAULT_NAME" --name "openai-api-key" --query value -o tsv 2>/dev/null || echo "")
XAI_KEY=$(az keyvault secret show --vault-name "$KEY_VAULT_NAME" --name "xai-api-key" --query value -o tsv 2>/dev/null || echo "")
MINI_XDR_API_KEY=$(az keyvault secret show --vault-name "$KEY_VAULT_NAME" --name "mini-xdr-api-key" --query value -o tsv 2>/dev/null || echo "")

kubectl create secret generic mini-xdr-secrets \
  --from-literal=openai-api-key="${OPENAI_KEY}" \
  --from-literal=xai-api-key="${XAI_KEY}" \
  --from-literal=mini-xdr-api-key="${MINI_XDR_API_KEY}" \
  --namespace=mini-xdr \
  --dry-run=client -o yaml | kubectl apply -f -

echo -e "${GREEN}✅ Secrets created${NC}"

# Update deployment manifests with ACR images
echo "Updating deployment manifests..."

# Backend deployment
sed "s|image: mini-xdr-backend:latest|image: ${ACR_LOGIN_SERVER}/mini-xdr-backend:latest|g" \
  "$K8S_DIR/backend-deployment.yaml" | kubectl apply -f -

# Frontend deployment
sed "s|image: mini-xdr-frontend:latest|image: ${ACR_LOGIN_SERVER}/mini-xdr-frontend:latest|g" \
  "$K8S_DIR/frontend-deployment.yaml" | kubectl apply -f -

echo -e "${GREEN}✅ Deployments created${NC}"

# Wait for deployments to be ready
echo "Waiting for deployments to be ready..."
kubectl wait --for=condition=available --timeout=300s \
  deployment/mini-xdr-backend \
  deployment/mini-xdr-frontend \
  -n mini-xdr

echo -e "${GREEN}✅ All deployments are ready${NC}"

# Display pod status
echo ""
echo "Pod Status:"
kubectl get pods -n mini-xdr
echo ""

# Display service information
echo "Service Information:"
kubectl get svc -n mini-xdr
echo ""

# Get Application Gateway IP
APPGW_IP=$(terraform -chdir="$TERRAFORM_DIR" output -raw appgw_public_ip)

echo -e "${GREEN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║           Mini-XDR Deployed Successfully to AKS!               ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "Access the application:"
echo "  • Application Gateway IP: $APPGW_IP"
echo "  • Frontend: https://$APPGW_IP"
echo "  • Backend API: https://$APPGW_IP/api"
echo ""
echo "Useful commands:"
echo "  • View logs: kubectl logs -n mini-xdr -l app=mini-xdr-backend --tail=100 -f"
echo "  • Port forward: kubectl port-forward -n mini-xdr svc/mini-xdr-backend-service 8000:8000"
echo "  • Shell access: kubectl exec -it -n mini-xdr deployment/mini-xdr-backend -- /bin/bash"
echo ""


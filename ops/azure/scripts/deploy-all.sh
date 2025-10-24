#!/bin/bash
# ============================================================================
# Mini-XDR Complete Azure Deployment Script
# ============================================================================
# This script automates the entire deployment process:
# 1. Infrastructure (Terraform)
# 2. Container images (Docker + ACR)
# 3. Application deployment (Kubernetes)
# 4. Mini corporate network setup
# ============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
TERRAFORM_DIR="$PROJECT_ROOT/ops/azure/terraform"

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║       Mini-XDR Complete Azure Deployment                       ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Function to print step headers
print_step() {
    echo ""
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}$1${NC}"
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
}

# Function to check prerequisites
check_prerequisites() {
    print_step "Step 1/7: Checking Prerequisites"
    
    # Check Azure CLI
    if ! command -v az &> /dev/null; then
        echo -e "${RED}❌ Azure CLI not found. Please install: https://docs.microsoft.com/en-us/cli/azure/install-azure-cli${NC}"
        exit 1
    fi
    echo -e "${GREEN}✅ Azure CLI installed${NC}"
    
    # Check if logged in
    if ! az account show &> /dev/null; then
        echo -e "${RED}❌ Not logged into Azure. Run: az login${NC}"
        exit 1
    fi
    echo -e "${GREEN}✅ Azure CLI authenticated${NC}"
    
    # Check Terraform
    if ! command -v terraform &> /dev/null; then
        echo -e "${RED}❌ Terraform not found. Please install: https://www.terraform.io/downloads${NC}"
        exit 1
    fi
    echo -e "${GREEN}✅ Terraform installed${NC}"
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}❌ Docker not found. Please install: https://docs.docker.com/get-docker/${NC}"
        exit 1
    fi
    echo -e "${GREEN}✅ Docker installed${NC}"
    
    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        echo -e "${RED}❌ kubectl not found. Please install: https://kubernetes.io/docs/tasks/tools/${NC}"
        exit 1
    fi
    echo -e "${GREEN}✅ kubectl installed${NC}"
    
    # Display Azure subscription
    SUBSCRIPTION=$(az account show --query name -o tsv)
    echo -e "${GREEN}✅ Subscription: $SUBSCRIPTION${NC}"
}

# Function to deploy infrastructure with Terraform
deploy_infrastructure() {
    print_step "Step 2/7: Deploying Azure Infrastructure with Terraform"
    
    cd "$TERRAFORM_DIR"
    
    echo "Initializing Terraform..."
    terraform init
    
    echo "Validating Terraform configuration..."
    terraform validate
    
    echo "Planning infrastructure deployment..."
    terraform plan -out=tfplan
    
    echo ""
    read -p "Apply this Terraform plan? (yes/no): " APPLY_PLAN
    
    if [ "$APPLY_PLAN" != "yes" ]; then
        echo -e "${YELLOW}⚠️  Deployment cancelled${NC}"
        exit 0
    fi
    
    echo "Applying Terraform configuration..."
    terraform apply tfplan
    
    echo -e "${GREEN}✅ Infrastructure deployed successfully${NC}"
    
    # Save outputs
    terraform output -json > "$PROJECT_ROOT/terraform-outputs.json"
    echo -e "${GREEN}✅ Terraform outputs saved to terraform-outputs.json${NC}"
}

# Function to build and push Docker images
build_and_push_images() {
    print_step "Step 3/7: Building and Pushing Docker Images"
    
    cd "$PROJECT_ROOT"
    
    # Get ACR login server from Terraform outputs
    ACR_LOGIN_SERVER=$(terraform -chdir="$TERRAFORM_DIR" output -raw acr_login_server)
    
    echo "Logging into Azure Container Registry..."
    az acr login --name "${ACR_LOGIN_SERVER%%.*}"
    
    echo "Building backend image..."
    docker build -f ops/Dockerfile.backend -t "$ACR_LOGIN_SERVER/mini-xdr-backend:v1.0" .
    docker build -f ops/Dockerfile.backend -t "$ACR_LOGIN_SERVER/mini-xdr-backend:latest" .
    
    echo "Building frontend image..."
    docker build -f ops/Dockerfile.frontend -t "$ACR_LOGIN_SERVER/mini-xdr-frontend:v1.0" .
    docker build -f ops/Dockerfile.frontend -t "$ACR_LOGIN_SERVER/mini-xdr-frontend:latest" .
    
    echo "Building ingestion agent image..."
    docker build -f ops/Dockerfile.ingestion-agent -t "$ACR_LOGIN_SERVER/mini-xdr-agent:v1.0" .
    docker build -f ops/Dockerfile.ingestion-agent -t "$ACR_LOGIN_SERVER/mini-xdr-agent:latest" .
    
    echo "Pushing images to ACR..."
    docker push "$ACR_LOGIN_SERVER/mini-xdr-backend:v1.0"
    docker push "$ACR_LOGIN_SERVER/mini-xdr-backend:latest"
    docker push "$ACR_LOGIN_SERVER/mini-xdr-frontend:v1.0"
    docker push "$ACR_LOGIN_SERVER/mini-xdr-frontend:latest"
    docker push "$ACR_LOGIN_SERVER/mini-xdr-agent:v1.0"
    docker push "$ACR_LOGIN_SERVER/mini-xdr-agent:latest"
    
    echo -e "${GREEN}✅ Images built and pushed successfully${NC}"
}

# Function to configure kubectl
configure_kubectl() {
    print_step "Step 4/7: Configuring kubectl"
    
    RESOURCE_GROUP=$(terraform -chdir="$TERRAFORM_DIR" output -raw resource_group_name)
    AKS_CLUSTER=$(terraform -chdir="$TERRAFORM_DIR" output -raw aks_cluster_name)
    
    echo "Getting AKS credentials..."
    az aks get-credentials --resource-group "$RESOURCE_GROUP" --name "$AKS_CLUSTER" --overwrite-existing
    
    echo "Testing kubectl connection..."
    kubectl cluster-info
    
    echo -e "${GREEN}✅ kubectl configured successfully${NC}"
}

# Function to deploy application to AKS
deploy_application() {
    print_step "Step 5/7: Deploying Application to AKS"
    
    "$SCRIPT_DIR/deploy-mini-xdr-to-aks.sh"
    
    echo -e "${GREEN}✅ Application deployed successfully${NC}"
}

# Function to setup mini corporate network
setup_mini_corp_network() {
    print_step "Step 6/7: Setting up Mini Corporate Network"
    
    if terraform -chdir="$TERRAFORM_DIR" output -raw domain_controller_private_ip &> /dev/null; then
        echo "Running mini corporate network setup..."
        "$SCRIPT_DIR/setup-mini-corp-network.sh"
        echo -e "${GREEN}✅ Mini corporate network configured${NC}"
    else
        echo -e "${YELLOW}⚠️  Mini corporate network VMs not deployed (enable_mini_corp_network = false)${NC}"
    fi
}

# Function to display deployment summary
display_summary() {
    print_step "Step 7/7: Deployment Summary"
    
    echo -e "${GREEN}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║              Deployment Completed Successfully!                ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    
    # Get outputs
    ACR_LOGIN_SERVER=$(terraform -chdir="$TERRAFORM_DIR" output -raw acr_login_server)
    APPGW_IP=$(terraform -chdir="$TERRAFORM_DIR" output -raw appgw_public_ip)
    KEY_VAULT=$(terraform -chdir="$TERRAFORM_DIR" output -raw key_vault_name)
    POSTGRES_FQDN=$(terraform -chdir="$TERRAFORM_DIR" output -raw postgres_fqdn 2>/dev/null || echo "Not deployed")
    REDIS_HOSTNAME=$(terraform -chdir="$TERRAFORM_DIR" output -raw redis_hostname 2>/dev/null || echo "Not deployed")
    DC_IP=$(terraform -chdir="$TERRAFORM_DIR" output -raw domain_controller_private_ip 2>/dev/null || echo "Not deployed")
    
    echo -e "${BLUE}📦 Infrastructure:${NC}"
    echo "  • Resource Group: $(terraform -chdir="$TERRAFORM_DIR" output -raw resource_group_name)"
    echo "  • Location: $(terraform -chdir="$TERRAFORM_DIR" output -raw resource_group_location)"
    echo "  • AKS Cluster: $(terraform -chdir="$TERRAFORM_DIR" output -raw aks_cluster_name)"
    echo ""
    
    echo -e "${BLUE}🐳 Container Registry:${NC}"
    echo "  • ACR: $ACR_LOGIN_SERVER"
    echo "  • Backend Image: $ACR_LOGIN_SERVER/mini-xdr-backend:v1.0"
    echo "  • Frontend Image: $ACR_LOGIN_SERVER/mini-xdr-frontend:v1.0"
    echo "  • Agent Image: $ACR_LOGIN_SERVER/mini-xdr-agent:v1.0"
    echo ""
    
    echo -e "${BLUE}🗄️  Databases:${NC}"
    echo "  • PostgreSQL: $POSTGRES_FQDN"
    echo "  • Redis: $REDIS_HOSTNAME"
    echo ""
    
    echo -e "${BLUE}🔐 Security:${NC}"
    echo "  • Key Vault: $KEY_VAULT"
    echo "  • Application Gateway: $APPGW_IP"
    echo "  • Your IP (NSG whitelist): $(terraform -chdir="$TERRAFORM_DIR" output -raw your_ip_address)"
    echo ""
    
    echo -e "${BLUE}🏢 Mini Corporate Network:${NC}"
    echo "  • Domain Controller: $DC_IP"
    if [ "$DC_IP" != "Not deployed" ]; then
        echo "  • Windows Endpoints: $(terraform -chdir="$TERRAFORM_DIR" output -json windows_endpoint_private_ips | jq -r '.[]' | wc -l) VMs"
        echo "  • Linux Servers: $(terraform -chdir="$TERRAFORM_DIR" output -json linux_server_private_ips | jq -r '.[]' | wc -l) VMs"
    fi
    echo ""
    
    echo -e "${BLUE}🌐 Access:${NC}"
    echo "  • Application Gateway: https://$APPGW_IP"
    echo "  • kubectl: Configured for AKS cluster"
    echo ""
    
    echo -e "${YELLOW}📝 Next Steps:${NC}"
    echo "  1. Update DNS or /etc/hosts to point mini-xdr.local to $APPGW_IP"
    echo "  2. Access the application at https://mini-xdr.local or https://$APPGW_IP"
    echo "  3. Connect to VMs via Azure Bastion in the Azure Portal"
    echo "  4. Run attack simulations: ./ops/azure/attacks/run-all-tests.sh"
    echo "  5. Monitor logs: kubectl logs -n mini-xdr -l app=mini-xdr-backend"
    echo ""
    
    echo -e "${GREEN}✨ Mini-XDR is ready for production use! ✨${NC}"
}

# Main execution
main() {
    check_prerequisites
    deploy_infrastructure
    build_and_push_images
    configure_kubectl
    deploy_application
    setup_mini_corp_network
    display_summary
}

# Run main function
main


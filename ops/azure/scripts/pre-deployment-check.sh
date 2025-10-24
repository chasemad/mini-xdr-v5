#!/bin/bash
# ============================================================================
# Pre-Deployment Validation Script
# ============================================================================
# Validates environment before Azure deployment
# ============================================================================

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
TERRAFORM_DIR="$PROJECT_ROOT/ops/azure/terraform"

# Counters
TOTAL_CHECKS=0
PASSED_CHECKS=0
FAILED_CHECKS=0
WARNING_CHECKS=0

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║       Mini-XDR Azure Pre-Deployment Validation                 ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Function to run check
check() {
    local NAME="$1"
    local COMMAND="$2"
    local REQUIRED="${3:-true}"
    
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    echo -n "[$TOTAL_CHECKS] $NAME... "
    
    if eval "$COMMAND" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ PASS${NC}"
        PASSED_CHECKS=$((PASSED_CHECKS + 1))
        return 0
    else
        if [ "$REQUIRED" = "true" ]; then
            echo -e "${RED}✗ FAIL${NC}"
            FAILED_CHECKS=$((FAILED_CHECKS + 1))
        else
            echo -e "${YELLOW}⚠ WARN${NC}"
            WARNING_CHECKS=$((WARNING_CHECKS + 1))
        fi
        return 1
    fi
}

# ============================================================================
# Phase 1: Prerequisites
# ============================================================================

echo -e "${CYAN}Phase 1/7: Prerequisites${NC}"
echo ""

check "Azure CLI installed" "command -v az"
check "Terraform installed" "command -v terraform"
check "Docker installed" "command -v docker"
check "kubectl installed" "command -v kubectl"
check "jq installed" "command -v jq" false
check "git installed" "command -v git" false

# ============================================================================
# Phase 2: Azure Authentication
# ============================================================================

echo ""
echo -e "${CYAN}Phase 2/7: Azure Authentication${NC}"
echo ""

check "Azure CLI authenticated" "az account show"

if az account show > /dev/null 2>&1; then
    SUBSCRIPTION=$(az account show --query name -o tsv 2>/dev/null || echo "Unknown")
    SUBSCRIPTION_ID=$(az account show --query id -o tsv 2>/dev/null || echo "Unknown")
    echo -e "  ${BLUE}→ Subscription:${NC} $SUBSCRIPTION"
    echo -e "  ${BLUE}→ Subscription ID:${NC} $SUBSCRIPTION_ID"
fi

# ============================================================================
# Phase 3: Azure Permissions
# ============================================================================

echo ""
echo -e "${CYAN}Phase 3/7: Azure Permissions${NC}"
echo ""

check "Can create resource groups" "az group create --name mini-xdr-test-rg --location eastus --dry-run" false

if az account show > /dev/null 2>&1; then
    USER_OBJECT_ID=$(az ad signed-in-user show --query id -o tsv 2>/dev/null || echo "")
    if [ -n "$USER_OBJECT_ID" ]; then
        echo -e "  ${BLUE}→ User Object ID:${NC} $USER_OBJECT_ID"
        PASSED_CHECKS=$((PASSED_CHECKS + 1))
    else
        echo -e "  ${YELLOW}⚠ Could not determine user object ID${NC}"
        WARNING_CHECKS=$((WARNING_CHECKS + 1))
    fi
fi

# ============================================================================
# Phase 4: Project Structure
# ============================================================================

echo ""
echo -e "${CYAN}Phase 4/7: Project Structure${NC}"
echo ""

check "Backend directory exists" "test -d $PROJECT_ROOT/backend"
check "Frontend directory exists" "test -d $PROJECT_ROOT/frontend"
check "Terraform directory exists" "test -d $TERRAFORM_DIR"
check "Terraform provider.tf exists" "test -f $TERRAFORM_DIR/provider.tf"
check "Terraform variables.tf exists" "test -f $TERRAFORM_DIR/variables.tf"
check "Kubernetes manifests exist" "test -d $PROJECT_ROOT/ops/k8s"
check "Deployment scripts exist" "test -f $SCRIPT_DIR/deploy-all.sh"

# ============================================================================
# Phase 5: Docker Configuration
# ============================================================================

echo ""
echo -e "${CYAN}Phase 5/7: Docker Configuration${NC}"
echo ""

check "Docker daemon running" "docker info"
check "Backend Dockerfile exists" "test -f $PROJECT_ROOT/ops/Dockerfile.backend"
check "Frontend Dockerfile exists" "test -f $PROJECT_ROOT/ops/Dockerfile.frontend"
check "Ingestion agent Dockerfile exists" "test -f $PROJECT_ROOT/ops/Dockerfile.ingestion-agent"

# ============================================================================
# Phase 6: Terraform Validation
# ============================================================================

echo ""
echo -e "${CYAN}Phase 6/7: Terraform Validation${NC}"
echo ""

cd "$TERRAFORM_DIR"

check "Terraform initialized" "test -d .terraform" false

if [ -d .terraform ]; then
    check "Terraform validate" "terraform validate"
else
    echo -e "  ${YELLOW}→ Run 'terraform init' in $TERRAFORM_DIR${NC}"
    WARNING_CHECKS=$((WARNING_CHECKS + 1))
fi

# Check for tfvars file
if [ -f terraform.tfvars ]; then
    echo -e "  ${GREEN}✓ terraform.tfvars found${NC}"
    PASSED_CHECKS=$((PASSED_CHECKS + 1))
else
    echo -e "  ${YELLOW}⚠ terraform.tfvars not found (will use defaults)${NC}"
    echo -e "    ${BLUE}→ Copy terraform.tfvars.example to terraform.tfvars to customize${NC}"
    WARNING_CHECKS=$((WARNING_CHECKS + 1))
fi

# ============================================================================
# Phase 7: Network Connectivity
# ============================================================================

echo ""
echo -e "${CYAN}Phase 7/7: Network Connectivity${NC}"
echo ""

check "Can reach Azure Portal" "curl -s -o /dev/null -w '%{http_code}' https://portal.azure.com | grep -q 200"
check "Can reach Docker Hub" "curl -s -o /dev/null -w '%{http_code}' https://hub.docker.com | grep -q 200" false

# Get public IP
if command -v curl > /dev/null 2>&1; then
    PUBLIC_IP=$(curl -s ifconfig.me 2>/dev/null || echo "Unable to detect")
    if [ "$PUBLIC_IP" != "Unable to detect" ]; then
        echo -e "  ${BLUE}→ Your public IP:${NC} $PUBLIC_IP"
        echo -e "    ${BLUE}(This will be used for NSG whitelisting)${NC}"
        PASSED_CHECKS=$((PASSED_CHECKS + 1))
    else
        echo -e "  ${YELLOW}⚠ Could not detect public IP${NC}"
        WARNING_CHECKS=$((WARNING_CHECKS + 1))
    fi
fi

# ============================================================================
# Summary
# ============================================================================

echo ""
echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                    Validation Summary                          ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

echo -e "Total Checks: $TOTAL_CHECKS"
echo -e "${GREEN}Passed:${NC} $PASSED_CHECKS"
if [ $FAILED_CHECKS -gt 0 ]; then
    echo -e "${RED}Failed:${NC} $FAILED_CHECKS"
fi
if [ $WARNING_CHECKS -gt 0 ]; then
    echo -e "${YELLOW}Warnings:${NC} $WARNING_CHECKS"
fi

echo ""

# ============================================================================
# Recommendations
# ============================================================================

if [ $FAILED_CHECKS -eq 0 ]; then
    echo -e "${GREEN}✨ All critical checks passed! Ready to deploy.${NC}"
    echo ""
    echo -e "${BLUE}Next Steps:${NC}"
    echo "  1. Review configuration in ops/azure/terraform/terraform.tfvars"
    echo "  2. Run deployment: ./ops/azure/scripts/deploy-all.sh"
    echo "  3. Estimated time: ~90 minutes"
    echo "  4. Estimated cost: \$800-1,400/month"
    echo ""
    exit 0
else
    echo -e "${RED}✗ Some critical checks failed. Please fix the issues above.${NC}"
    echo ""
    echo -e "${YELLOW}Common Solutions:${NC}"
    
    if ! command -v az &> /dev/null; then
        echo "  • Install Azure CLI: https://docs.microsoft.com/en-us/cli/azure/install-azure-cli"
    fi
    
    if ! az account show &> /dev/null; then
        echo "  • Login to Azure: az login"
    fi
    
    if ! command -v terraform &> /dev/null; then
        echo "  • Install Terraform: https://www.terraform.io/downloads"
    fi
    
    if ! command -v docker &> /dev/null; then
        echo "  • Install Docker: https://docs.docker.com/get-docker/"
    fi
    
    if ! command -v kubectl &> /dev/null; then
        echo "  • Install kubectl: https://kubernetes.io/docs/tasks/tools/"
    fi
    
    echo ""
    exit 1
fi


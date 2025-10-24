#!/bin/bash
# ============================================================================
# Azure Deployment Status Monitor
# ============================================================================
# Shows real-time status of Mini-XDR Azure deployment
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

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║       Mini-XDR Azure Deployment Status                         ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if deployed
if [ ! -f "$TERRAFORM_DIR/terraform.tfstate" ]; then
    echo -e "${YELLOW}⚠️  No deployment found${NC}"
    echo ""
    echo "To deploy Mini-XDR to Azure, run:"
    echo "  ./ops/azure/scripts/deploy-all.sh"
    echo ""
    exit 0
fi

# ============================================================================
# Infrastructure Status
# ============================================================================

echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${CYAN}Infrastructure Status${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

RG_NAME=$(terraform -chdir="$TERRAFORM_DIR" output -raw resource_group_name 2>/dev/null || echo "")

if [ -n "$RG_NAME" ]; then
    echo -e "${GREEN}✓ Resource Group:${NC} $RG_NAME"
    
    # Get resource counts
    TOTAL_RESOURCES=$(az resource list -g "$RG_NAME" --query "length(@)" -o tsv 2>/dev/null || echo "0")
    echo -e "  Total Resources: $TOTAL_RESOURCES"
    
    # Show key resources
    echo ""
    echo -e "${BLUE}Key Resources:${NC}"
    
    # AKS
    AKS_NAME=$(terraform -chdir="$TERRAFORM_DIR" output -raw aks_cluster_name 2>/dev/null || echo "")
    if [ -n "$AKS_NAME" ]; then
        AKS_STATUS=$(az aks show -g "$RG_NAME" -n "$AKS_NAME" --query "powerState.code" -o tsv 2>/dev/null || echo "Unknown")
        if [ "$AKS_STATUS" = "Running" ]; then
            echo -e "  ${GREEN}✓${NC} AKS Cluster: $AKS_NAME (${GREEN}$AKS_STATUS${NC})"
        else
            echo -e "  ${YELLOW}⚠${NC} AKS Cluster: $AKS_NAME (${YELLOW}$AKS_STATUS${NC})"
        fi
    fi
    
    # ACR
    ACR_SERVER=$(terraform -chdir="$TERRAFORM_DIR" output -raw acr_login_server 2>/dev/null || echo "")
    if [ -n "$ACR_SERVER" ]; then
        ACR_NAME="${ACR_SERVER%%.*}"
        echo -e "  ${GREEN}✓${NC} Container Registry: $ACR_SERVER"
        
        # Count images
        IMAGE_COUNT=$(az acr repository list --name "$ACR_NAME" --query "length(@)" -o tsv 2>/dev/null || echo "0")
        echo -e "    Images: $IMAGE_COUNT"
    fi
    
    # PostgreSQL
    POSTGRES_FQDN=$(terraform -chdir="$TERRAFORM_DIR" output -raw postgres_fqdn 2>/dev/null || echo "")
    if [ -n "$POSTGRES_FQDN" ] && [ "$POSTGRES_FQDN" != "null" ]; then
        echo -e "  ${GREEN}✓${NC} PostgreSQL: $POSTGRES_FQDN"
    fi
    
    # Redis
    REDIS_HOST=$(terraform -chdir="$TERRAFORM_DIR" output -raw redis_hostname 2>/dev/null || echo "")
    if [ -n "$REDIS_HOST" ] && [ "$REDIS_HOST" != "null" ]; then
        echo -e "  ${GREEN}✓${NC} Redis: $REDIS_HOST"
    fi
    
    # Application Gateway
    APPGW_IP=$(terraform -chdir="$TERRAFORM_DIR" output -raw appgw_public_ip 2>/dev/null || echo "")
    if [ -n "$APPGW_IP" ]; then
        echo -e "  ${GREEN}✓${NC} Application Gateway: $APPGW_IP"
    fi
    
    # Key Vault
    KV_NAME=$(terraform -chdir="$TERRAFORM_DIR" output -raw key_vault_name 2>/dev/null || echo "")
    if [ -n "$KV_NAME" ]; then
        SECRET_COUNT=$(az keyvault secret list --vault-name "$KV_NAME" --query "length(@)" -o tsv 2>/dev/null || echo "0")
        echo -e "  ${GREEN}✓${NC} Key Vault: $KV_NAME ($SECRET_COUNT secrets)"
    fi
    
else
    echo -e "${RED}✗ Resource group not found${NC}"
fi

# ============================================================================
# Kubernetes Status
# ============================================================================

echo ""
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${CYAN}Kubernetes Status${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

if kubectl cluster-info > /dev/null 2>&1; then
    echo -e "${GREEN}✓ kubectl configured${NC}"
    
    # Check namespace
    if kubectl get namespace mini-xdr > /dev/null 2>&1; then
        echo -e "${GREEN}✓ mini-xdr namespace exists${NC}"
        echo ""
        
        # Pod status
        echo -e "${BLUE}Pods:${NC}"
        BACKEND_PODS=$(kubectl get pods -n mini-xdr -l app=mini-xdr-backend -o json 2>/dev/null | jq -r '.items | length' || echo "0")
        BACKEND_RUNNING=$(kubectl get pods -n mini-xdr -l app=mini-xdr-backend -o json 2>/dev/null | jq -r '[.items[] | select(.status.phase=="Running")] | length' || echo "0")
        
        if [ "$BACKEND_PODS" -gt 0 ]; then
            if [ "$BACKEND_RUNNING" -eq "$BACKEND_PODS" ]; then
                echo -e "  ${GREEN}✓${NC} Backend: $BACKEND_RUNNING/$BACKEND_PODS running"
            else
                echo -e "  ${YELLOW}⚠${NC} Backend: $BACKEND_RUNNING/$BACKEND_PODS running"
            fi
        else
            echo -e "  ${RED}✗${NC} Backend: No pods found"
        fi
        
        FRONTEND_PODS=$(kubectl get pods -n mini-xdr -l app=mini-xdr-frontend -o json 2>/dev/null | jq -r '.items | length' || echo "0")
        FRONTEND_RUNNING=$(kubectl get pods -n mini-xdr -l app=mini-xdr-frontend -o json 2>/dev/null | jq -r '[.items[] | select(.status.phase=="Running")] | length' || echo "0")
        
        if [ "$FRONTEND_PODS" -gt 0 ]; then
            if [ "$FRONTEND_RUNNING" -eq "$FRONTEND_PODS" ]; then
                echo -e "  ${GREEN}✓${NC} Frontend: $FRONTEND_RUNNING/$FRONTEND_PODS running"
            else
                echo -e "  ${YELLOW}⚠${NC} Frontend: $FRONTEND_RUNNING/$FRONTEND_PODS running"
            fi
        else
            echo -e "  ${RED}✗${NC} Frontend: No pods found"
        fi
        
    else
        echo -e "${YELLOW}⚠ mini-xdr namespace not found${NC}"
    fi
else
    echo -e "${YELLOW}⚠ kubectl not configured or cluster not accessible${NC}"
fi

# ============================================================================
# Virtual Machines Status
# ============================================================================

echo ""
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${CYAN}Mini Corporate Network${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

if [ -n "$RG_NAME" ]; then
    VM_LIST=$(az vm list -g "$RG_NAME" -d --query "[].{name:name, powerState:powerState, ip:privateIps}" -o json 2>/dev/null || echo "[]")
    VM_COUNT=$(echo "$VM_LIST" | jq -r 'length' || echo "0")
    
    if [ "$VM_COUNT" -gt 0 ]; then
        echo -e "${BLUE}Virtual Machines: $VM_COUNT${NC}"
        echo ""
        
        # Domain Controller
        DC_STATUS=$(echo "$VM_LIST" | jq -r '.[] | select(.name | contains("dc01")) | .powerState' || echo "")
        if [ -n "$DC_STATUS" ]; then
            if [[ "$DC_STATUS" == *"running"* ]]; then
                echo -e "  ${GREEN}✓${NC} Domain Controller (DC01): ${GREEN}Running${NC}"
            else
                echo -e "  ${YELLOW}⚠${NC} Domain Controller (DC01): ${YELLOW}$DC_STATUS${NC}"
            fi
        fi
        
        # Windows Endpoints
        WIN_RUNNING=$(echo "$VM_LIST" | jq -r '[.[] | select(.name | contains("ws")) | select(.powerState | contains("running"))] | length' || echo "0")
        WIN_TOTAL=$(echo "$VM_LIST" | jq -r '[.[] | select(.name | contains("ws"))] | length' || echo "0")
        if [ "$WIN_TOTAL" -gt 0 ]; then
            echo -e "  ${GREEN}✓${NC} Windows Endpoints: $WIN_RUNNING/$WIN_TOTAL running"
        fi
        
        # Linux Servers
        LIN_RUNNING=$(echo "$VM_LIST" | jq -r '[.[] | select(.name | contains("srv")) | select(.powerState | contains("running"))] | length' || echo "0")
        LIN_TOTAL=$(echo "$VM_LIST" | jq -r '[.[] | select(.name | contains("srv"))] | length' || echo "0")
        if [ "$LIN_TOTAL" -gt 0 ]; then
            echo -e "  ${GREEN}✓${NC} Linux Servers: $LIN_RUNNING/$LIN_TOTAL running"
        fi
        
    else
        echo -e "${YELLOW}⚠ No VMs found (Mini corporate network not deployed)${NC}"
    fi
else
    echo -e "${RED}✗ Cannot check VMs (resource group not found)${NC}"
fi

# ============================================================================
# Access Information
# ============================================================================

echo ""
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${CYAN}Access Information${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

if [ -n "$APPGW_IP" ]; then
    echo -e "${BLUE}Application:${NC}"
    echo "  URL: https://$APPGW_IP"
    echo ""
    
    # Test connectivity
    HTTP_CODE=$(curl -k -s -o /dev/null -w "%{http_code}" "https://$APPGW_IP" --max-time 5 2>/dev/null || echo "000")
    if [ "$HTTP_CODE" = "200" ] || [ "$HTTP_CODE" = "302" ]; then
        echo -e "  ${GREEN}✓ Application is responding (HTTP $HTTP_CODE)${NC}"
    else
        echo -e "  ${YELLOW}⚠ Application not responding (HTTP $HTTP_CODE)${NC}"
    fi
fi

if [ -n "$KV_NAME" ]; then
    echo ""
    echo -e "${BLUE}Credentials (in Key Vault):${NC}"
    echo "  Vault: $KV_NAME"
    echo "  View secrets: az keyvault secret list --vault-name $KV_NAME"
fi

# ============================================================================
# Cost Estimate
# ============================================================================

echo ""
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${CYAN}Cost Information${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

echo -e "${BLUE}Estimated Monthly Cost:${NC} \$800-1,400"
echo ""
echo "Current running VMs: $WIN_RUNNING Windows + $LIN_RUNNING Linux"
echo ""
echo -e "${YELLOW}Cost Savings Tips:${NC}"
echo "  • Stop VMs when not testing"
echo "  • Auto-shutdown is enabled at 10 PM daily"
echo "  • Monitor costs: az consumption usage list"
echo ""

# ============================================================================
# Quick Actions
# ============================================================================

echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${CYAN}Quick Actions${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

echo "View logs:"
echo "  kubectl logs -n mini-xdr -l app=mini-xdr-backend -f"
echo ""
echo "View pods:"
echo "  kubectl get pods -n mini-xdr"
echo ""
echo "Stop all VMs:"
echo "  az vm deallocate --ids \$(az vm list -g $RG_NAME --query \"[].id\" -o tsv)"
echo ""
echo "Run attack simulations:"
echo "  ./ops/azure/attacks/run-all-tests.sh"
echo ""
echo "Full validation:"
echo "  ./ops/azure/tests/e2e-azure-test.sh"
echo ""


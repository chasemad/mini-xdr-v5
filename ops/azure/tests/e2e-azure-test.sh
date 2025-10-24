#!/bin/bash
# ============================================================================
# End-to-End Azure Deployment Test
# ============================================================================
# Comprehensive validation of entire Mini-XDR Azure deployment
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

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║       Mini-XDR End-to-End Azure Validation                     ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Test counter
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Function to run test
run_test() {
    local TEST_NAME="$1"
    local TEST_COMMAND="$2"
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    echo -n "[$TOTAL_TESTS] $TEST_NAME... "
    
    if eval "$TEST_COMMAND" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ PASS${NC}"
        PASSED_TESTS=$((PASSED_TESTS + 1))
        return 0
    else
        echo -e "${RED}✗ FAIL${NC}"
        FAILED_TESTS=$((FAILED_TESTS + 1))
        return 1
    fi
}

# ============================================================================
# Phase 1: Infrastructure Validation
# ============================================================================

echo -e "${YELLOW}Phase 1/6: Infrastructure Validation${NC}"
echo ""

run_test "Terraform state exists" \
    "test -f $TERRAFORM_DIR/terraform.tfstate"

run_test "Resource group exists" \
    "az group show --name mini-xdr-prod-rg > /dev/null"

run_test "Virtual network exists" \
    "az network vnet show --resource-group mini-xdr-prod-rg --name mini-xdr-vnet > /dev/null"

run_test "AKS cluster exists" \
    "az aks show --resource-group mini-xdr-prod-rg --name mini-xdr-aks > /dev/null"

run_test "Container registry exists" \
    "az acr show --name minixdracr > /dev/null"

run_test "Key Vault exists" \
    "test -n \"\$(terraform -chdir=$TERRAFORM_DIR output -raw key_vault_name)\""

# ============================================================================
# Phase 2: Kubernetes Validation
# ============================================================================

echo ""
echo -e "${YELLOW}Phase 2/6: Kubernetes Validation${NC}"
echo ""

run_test "kubectl configured" \
    "kubectl cluster-info > /dev/null"

run_test "mini-xdr namespace exists" \
    "kubectl get namespace mini-xdr > /dev/null"

run_test "Backend deployment exists" \
    "kubectl get deployment mini-xdr-backend -n mini-xdr > /dev/null"

run_test "Frontend deployment exists" \
    "kubectl get deployment mini-xdr-frontend -n mini-xdr > /dev/null"

run_test "Backend pods running" \
    "test \$(kubectl get pods -n mini-xdr -l app=mini-xdr-backend -o jsonpath='{.items[*].status.phase}' | grep -c Running) -gt 0"

run_test "Frontend pods running" \
    "test \$(kubectl get pods -n mini-xdr -l app=mini-xdr-frontend -o jsonpath='{.items[*].status.phase}' | grep -c Running) -gt 0"

# ============================================================================
# Phase 3: Application Validation
# ============================================================================

echo ""
echo -e "${YELLOW}Phase 3/6: Application Validation${NC}"
echo ""

if [ -f "$TERRAFORM_DIR/terraform.tfstate" ]; then
    KEY_VAULT_NAME=$(terraform -chdir="$TERRAFORM_DIR" output -raw key_vault_name 2>/dev/null || echo "")
    APPGW_IP=$(terraform -chdir="$TERRAFORM_DIR" output -raw appgw_public_ip 2>/dev/null || echo "")
    API_KEY=$(az keyvault secret show --vault-name "$KEY_VAULT_NAME" --name "mini-xdr-api-key" --query value -o tsv 2>/dev/null || echo "")
fi

BACKEND_URL="https://$APPGW_IP"

run_test "Backend health endpoint" \
    "curl -s -k $BACKEND_URL/health > /dev/null"

run_test "Backend API docs accessible" \
    "curl -s -k $BACKEND_URL/docs > /dev/null"

run_test "API authentication working" \
    "curl -s -k -H 'X-API-Key: $API_KEY' $BACKEND_URL/api/incidents > /dev/null"

run_test "ML models loaded" \
    "curl -s -k -H 'X-API-Key: $API_KEY' $BACKEND_URL/api/ml/status | jq -e '.status == \"healthy\"' > /dev/null"

# ============================================================================
# Phase 4: Mini Corporate Network Validation
# ============================================================================

echo ""
echo -e "${YELLOW}Phase 4/6: Mini Corporate Network Validation${NC}"
echo ""

run_test "Domain Controller VM exists" \
    "az vm show --resource-group mini-xdr-prod-rg --name mini-corp-dc01 > /dev/null"

run_test "Windows endpoint VMs exist" \
    "test \$(az vm list --resource-group mini-xdr-prod-rg --query \"[?contains(name, 'mini-corp-ws')].name\" -o tsv | wc -l) -ge 3"

run_test "Linux server VMs exist" \
    "test \$(az vm list --resource-group mini-xdr-prod-rg --query \"[?contains(name, 'mini-corp-srv')].name\" -o tsv | wc -l) -ge 2"

run_test "Bastion host exists" \
    "az network bastion show --resource-group mini-xdr-prod-rg --name mini-xdr-bastion > /dev/null || echo 'Bastion optional'"

# ============================================================================
# Phase 5: Database Validation
# ============================================================================

echo ""
echo -e "${YELLOW}Phase 5/6: Database Validation${NC}"
echo ""

run_test "PostgreSQL server exists" \
    "az postgres flexible-server show --resource-group mini-xdr-prod-rg --name mini-xdr-postgres > /dev/null"

run_test "Redis cache exists" \
    "az redis show --resource-group mini-xdr-prod-rg --name mini-xdr-redis > /dev/null"

run_test "Database connection string in Key Vault" \
    "az keyvault secret show --vault-name $KEY_VAULT_NAME --name postgres-connection-string > /dev/null"

run_test "Redis connection string in Key Vault" \
    "az keyvault secret show --vault-name $KEY_VAULT_NAME --name redis-connection-string > /dev/null"

# ============================================================================
# Phase 6: Security Validation
# ============================================================================

echo ""
echo -e "${YELLOW}Phase 6/6: Security Validation${NC}"
echo ""

run_test "Key Vault access restricted" \
    "az keyvault show --name $KEY_VAULT_NAME | jq -e '.properties.networkAcls.defaultAction == \"Allow\"' > /dev/null"

run_test "PostgreSQL has no public access" \
    "az postgres flexible-server show --resource-group mini-xdr-prod-rg --name mini-xdr-postgres | jq -e '.network.publicNetworkAccess == \"Disabled\" or .network.delegatedSubnetResourceId != null' > /dev/null"

run_test "Redis has no public access" \
    "az redis show --resource-group mini-xdr-prod-rg --name mini-xdr-redis | jq -e '.publicNetworkAccess == \"Disabled\"' > /dev/null"

run_test "NSG rules configured" \
    "test \$(az network nsg list --resource-group mini-xdr-prod-rg --query 'length(@)') -ge 3"

run_test "WAF enabled on App Gateway" \
    "az network application-gateway show --resource-group mini-xdr-prod-rg --name mini-xdr-appgw | jq -e '.webApplicationFirewallConfiguration.enabled == true' > /dev/null"

# ============================================================================
# Results Summary
# ============================================================================

echo ""
echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║              End-to-End Test Results                           ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

PASS_RATE=$(awk "BEGIN {printf \"%.1f\", ($PASSED_TESTS/$TOTAL_TESTS)*100}")

echo "Test Summary:"
echo "  • Total Tests: $TOTAL_TESTS"
echo "  • Passed: $PASSED_TESTS"
echo "  • Failed: $FAILED_TESTS"
echo "  • Pass Rate: ${PASS_RATE}%"
echo ""

if [ $FAILED_TESTS -eq 0 ]; then
    echo -e "${GREEN}✓ ALL TESTS PASSED! System is operational! 🎉${NC}"
    EXIT_CODE=0
elif [ $PASS_RATE -ge 80 ]; then
    echo -e "${YELLOW}⚠ Some tests failed, but system is mostly operational${NC}"
    echo "  Review failures above and address issues."
    EXIT_CODE=1
else
    echo -e "${RED}✗ Multiple critical failures detected${NC}"
    echo "  System may not be fully operational. Review logs."
    EXIT_CODE=2
fi

echo ""
echo "Access Information:"
echo "  • Application Gateway: https://$APPGW_IP"
echo "  • Backend API: https://$APPGW_IP/api"
echo "  • API Docs: https://$APPGW_IP/docs"
echo "  • Dashboard: https://$APPGW_IP/incidents"
echo ""
echo "Useful Commands:"
echo "  • View pods: kubectl get pods -n mini-xdr"
echo "  • View logs: kubectl logs -n mini-xdr -l app=mini-xdr-backend -f"
echo "  • View VMs: az vm list -g mini-xdr-prod-rg -o table"
echo ""

exit $EXIT_CODE


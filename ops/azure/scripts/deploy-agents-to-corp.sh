#!/bin/bash
# ============================================================================
# Deploy Mini-XDR Agents to Corporate Network VMs
# ============================================================================
# Deploys agents to all Windows and Linux VMs in mini corporate network
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
echo -e "${BLUE}║       Deploy Agents to Mini Corporate Network                  ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Get configuration
if [ ! -f "$TERRAFORM_DIR/terraform.tfstate" ]; then
    echo -e "${RED}❌ Terraform state not found. Deploy infrastructure first.${NC}"
    exit 1
fi

RESOURCE_GROUP=$(terraform -chdir="$TERRAFORM_DIR" output -raw resource_group_name)
KEY_VAULT_NAME=$(terraform -chdir="$TERRAFORM_DIR" output -raw key_vault_name)
APPGW_IP=$(terraform -chdir="$TERRAFORM_DIR" output -raw appgw_public_ip)
ADMIN_USERNAME=$(terraform -chdir="$TERRAFORM_DIR" output -raw vm_admin_username)

API_KEY=$(az keyvault secret show --vault-name "$KEY_VAULT_NAME" --name "mini-xdr-api-key" --query value -o tsv)
BACKEND_URL="https://$APPGW_IP"

echo "Configuration:"
echo "  • Resource Group: $RESOURCE_GROUP"
echo "  • Backend URL: $BACKEND_URL"
echo "  • Admin Username: $ADMIN_USERNAME"
echo ""

# ============================================================================
# Deploy to Windows VMs
# ============================================================================

echo -e "${YELLOW}Deploying to Windows VMs...${NC}"
echo ""

# Get Windows VM names
WINDOWS_VMS=$(az vm list --resource-group "$RESOURCE_GROUP" --query "[?storageProfile.osDisk.osType=='Windows'].name" -o tsv)

WINDOWS_INSTALL_SCRIPT=$(cat "$SCRIPT_DIR/install-agent-windows.ps1" | sed "s/__BACKEND_URL__/$BACKEND_URL/g" | sed "s/__API_KEY__/$API_KEY/g")

WINDOWS_COUNT=0
for VM_NAME in $WINDOWS_VMS; do
    echo "Installing agent on $VM_NAME..."
    
    # Upload and execute PowerShell script
    az vm run-command invoke \
        --resource-group "$RESOURCE_GROUP" \
        --name "$VM_NAME" \
        --command-id RunPowerShellScript \
        --scripts "$WINDOWS_INSTALL_SCRIPT" \
        > /tmp/agent-install-$VM_NAME.log 2>&1
    
    if [ $? -eq 0 ]; then
        echo -e "  ${GREEN}✓ Agent installed on $VM_NAME${NC}"
        WINDOWS_COUNT=$((WINDOWS_COUNT + 1))
    else
        echo -e "  ${RED}✗ Failed to install on $VM_NAME${NC}"
        echo "  See: /tmp/agent-install-$VM_NAME.log"
    fi
done

# ============================================================================
# Deploy to Linux VMs
# ============================================================================

echo ""
echo -e "${YELLOW}Deploying to Linux VMs...${NC}"
echo ""

# Get Linux VM names
LINUX_VMS=$(az vm list --resource-group "$RESOURCE_GROUP" --query "[?storageProfile.osDisk.osType=='Linux'].name" -o tsv)

LINUX_INSTALL_SCRIPT=$(cat "$SCRIPT_DIR/install-agent-linux.sh")

LINUX_COUNT=0
for VM_NAME in $LINUX_VMS; do
    echo "Installing agent on $VM_NAME..."
    
    # Upload and execute bash script
    az vm run-command invoke \
        --resource-group "$RESOURCE_GROUP" \
        --name "$VM_NAME" \
        --command-id RunShellScript \
        --scripts "$LINUX_INSTALL_SCRIPT" \
        --parameters "backendUrl=$BACKEND_URL" "apiKey=$API_KEY" "agentType=endpoint" \
        > /tmp/agent-install-$VM_NAME.log 2>&1
    
    if [ $? -eq 0 ]; then
        echo -e "  ${GREEN}✓ Agent installed on $VM_NAME${NC}"
        LINUX_COUNT=$((LINUX_COUNT + 1))
    else
        echo -e "  ${RED}✗ Failed to install on $VM_NAME${NC}"
        echo "  See: /tmp/agent-install-$VM_NAME.log"
    fi
done

# ============================================================================
# Verify Agent Connectivity
# ============================================================================

echo ""
echo -e "${YELLOW}Verifying agent connectivity...${NC}"
sleep 10  # Wait for agents to send first heartbeat

AGENT_STATUS=$(curl -s -H "X-API-Key: $API_KEY" "$BACKEND_URL/api/agents/status" 2>/dev/null || echo "{}")
ACTIVE_AGENTS=$(echo "$AGENT_STATUS" | jq -r '.agents | length' 2>/dev/null || echo "0")

echo "Active Agents: $ACTIVE_AGENTS"

# ============================================================================
# Display Summary
# ============================================================================

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║        Agent Deployment Complete!                              ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "Deployment Summary:"
echo "  • Windows VMs: $WINDOWS_COUNT/$(($(echo "$WINDOWS_VMS" | wc -l))) agents installed"
echo "  • Linux VMs: $LINUX_COUNT/$(($(echo "$LINUX_VMS" | wc -l))) agents installed"
echo "  • Total Active Agents: $ACTIVE_AGENTS"
echo ""
echo "Verification:"
echo "  • Check dashboard: https://$APPGW_IP/agents"
echo "  • View agent logs: kubectl logs -n mini-xdr -l app=mini-xdr-backend | grep 'heartbeat'"
echo "  • Test agent action: curl -X POST -H 'X-API-Key: $API_KEY' $BACKEND_URL/api/agents/iam/execute ..."
echo ""
echo "Next Steps:"
echo "  1. Verify all agents reporting: curl -H 'X-API-Key: $API_KEY' $BACKEND_URL/api/agents/status"
echo "  2. Run attack simulations: ./ops/azure/attacks/run-all-tests.sh"
echo "  3. Monitor detections in dashboard: https://$APPGW_IP"
echo ""


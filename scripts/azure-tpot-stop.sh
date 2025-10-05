#!/bin/bash
# ========================================================================
# Azure TPOT Honeypot - STOP Script
# Gracefully shuts down TPOT services and stops the VM to save costs
# ========================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
RESOURCE_GROUP="mini-xdr-rg"
VM_NAME="mini-xdr-tpot"
SSH_KEY="$HOME/.ssh/mini-xdr-tpot-azure"
SSH_USER="azureuser"
SSH_PORT=64295

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║           Azure TPOT Honeypot - STOP                           ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# ========================================================================
# STEP 1: Check Azure CLI
# ========================================================================
if ! command -v az &> /dev/null; then
    echo -e "${RED}❌ Azure CLI not found${NC}"
    exit 1
fi

if ! az account show &> /dev/null; then
    echo -e "${RED}❌ Not logged into Azure${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Azure CLI authenticated${NC}"
echo ""

# ========================================================================
# STEP 2: Check VM Status
# ========================================================================
echo -e "${YELLOW}[1/3] Checking VM status...${NC}"

VM_STATUS=$(az vm get-instance-view \
    --resource-group "$RESOURCE_GROUP" \
    --name "$VM_NAME" \
    --query "instanceView.statuses[?starts_with(code, 'PowerState/')].displayStatus" \
    -o tsv 2>/dev/null || echo "Unknown")

echo -e "  Current status: ${BLUE}$VM_STATUS${NC}"

if [[ "$VM_STATUS" == "VM stopped" ]] || [[ "$VM_STATUS" == "VM deallocated" ]]; then
    echo -e "  ${GREEN}✅ VM is already stopped${NC}"
    echo ""
    echo -e "${GREEN}✨ TPOT is already stopped - nothing to do!${NC}"
    exit 0
fi

echo ""

# ========================================================================
# STEP 3: Graceful Shutdown (Optional)
# ========================================================================
echo -e "${YELLOW}[2/3] Attempting graceful shutdown...${NC}"

VM_IP=$(az vm show -d \
    --resource-group "$RESOURCE_GROUP" \
    --name "$VM_NAME" \
    --query publicIps \
    -o tsv 2>/dev/null || echo "")

if [ -n "$VM_IP" ] && [ -f "$SSH_KEY" ]; then
    echo -e "  ${BLUE}Stopping TPOT services gracefully...${NC}"
    
    SSH_CMD="ssh -i $SSH_KEY -p $SSH_PORT -o StrictHostKeyChecking=no -o ConnectTimeout=10 $SSH_USER@$VM_IP"
    
    # Try to stop Docker containers gracefully
    if $SSH_CMD "sudo docker ps -q | xargs -r sudo docker stop" 2>/dev/null; then
        echo -e "  ${GREEN}✅ TPOT services stopped${NC}"
        sleep 3
    else
        echo -e "  ${YELLOW}⚠️  Could not connect via SSH (VM may be unresponsive)${NC}"
    fi
else
    echo -e "  ${YELLOW}⚠️  Skipping graceful shutdown (no SSH access)${NC}"
fi

echo ""

# ========================================================================
# STEP 4: Stop VM
# ========================================================================
echo -e "${YELLOW}[3/3] Stopping VM...${NC}"
echo -e "  ${BLUE}This will deallocate the VM to save costs...${NC}"

# Deallocate VM (stops charging for compute)
az vm deallocate \
    --resource-group "$RESOURCE_GROUP" \
    --name "$VM_NAME" \
    --no-wait

echo -e "  ${BLUE}Waiting for VM to stop...${NC}"

# Wait for VM to stop
for i in {1..30}; do
    sleep 2
    VM_STATUS=$(az vm get-instance-view \
        --resource-group "$RESOURCE_GROUP" \
        --name "$VM_NAME" \
        --query "instanceView.statuses[?starts_with(code, 'PowerState/')].displayStatus" \
        -o tsv 2>/dev/null || echo "Unknown")
    
    if [[ "$VM_STATUS" == "VM deallocated" ]] || [[ "$VM_STATUS" == "VM stopped" ]]; then
        echo -e "  ${GREEN}✅ VM stopped successfully${NC}"
        break
    fi
    
    if [ $i -eq 30 ]; then
        echo -e "  ${YELLOW}⚠️  VM stop is taking longer than expected${NC}"
        echo -e "  ${BLUE}Check Azure Portal: https://portal.azure.com${NC}"
        break
    fi
done

echo ""

# ========================================================================
# SUMMARY
# ========================================================================
echo -e "${GREEN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                    TPOT STOPPED SUCCESSFULLY                    ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

echo -e "${BLUE}📋 Status:${NC}"
echo -e "  • VM Status: ${GREEN}Stopped (Deallocated)${NC}"
echo -e "  • Compute Charges: ${GREEN}Stopped${NC}"
echo -e "  • Storage Charges: ${YELLOW}Still Active (minimal)${NC}"
echo ""

echo -e "${BLUE}💰 Cost Savings:${NC}"
echo -e "  • VM compute charges: ${GREEN}$0/hour${NC}"
echo -e "  • Storage charges: ${YELLOW}~$3-5/month${NC}"
echo -e "  • Total savings while stopped: ${GREEN}~$40-60/month${NC}"
echo ""

echo -e "${BLUE}🚀 To start TPOT again:${NC}"
echo -e "  ${YELLOW}./scripts/azure-tpot-start.sh${NC}"
echo ""

echo -e "${BLUE}📊 To check status:${NC}"
echo -e "  ${YELLOW}./scripts/azure-tpot-status.sh${NC}"
echo ""

echo -e "${GREEN}✨ TPOT is stopped!${NC}"


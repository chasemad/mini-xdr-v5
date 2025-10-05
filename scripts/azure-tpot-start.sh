#!/bin/bash
# ========================================================================
# Azure TPOT Honeypot - START Script
# Starts the TPOT VM and ensures all services are running
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
echo -e "${BLUE}║           Azure TPOT Honeypot - START                          ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# ========================================================================
# STEP 1: Check Azure CLI
# ========================================================================
if ! command -v az &> /dev/null; then
    echo -e "${RED}❌ Azure CLI not found${NC}"
    echo -e "${YELLOW}Install: https://docs.microsoft.com/en-us/cli/azure/install-azure-cli${NC}"
    exit 1
fi

if ! az account show &> /dev/null; then
    echo -e "${RED}❌ Not logged into Azure${NC}"
    echo -e "${YELLOW}Run: az login${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Azure CLI authenticated${NC}"
echo ""

# ========================================================================
# STEP 2: Check VM Status
# ========================================================================
echo -e "${YELLOW}[1/4] Checking VM status...${NC}"

VM_STATUS=$(az vm get-instance-view \
    --resource-group "$RESOURCE_GROUP" \
    --name "$VM_NAME" \
    --query "instanceView.statuses[?starts_with(code, 'PowerState/')].displayStatus" \
    -o tsv 2>/dev/null || echo "Unknown")

echo -e "  Current status: ${BLUE}$VM_STATUS${NC}"

# ========================================================================
# STEP 3: Start VM if Not Running
# ========================================================================
if [[ "$VM_STATUS" == "VM running" ]]; then
    echo -e "  ${GREEN}✅ VM is already running${NC}"
else
    echo -e "${YELLOW}[2/4] Starting VM...${NC}"
    echo -e "  ${BLUE}This may take 30-60 seconds...${NC}"
    
    az vm start \
        --resource-group "$RESOURCE_GROUP" \
        --name "$VM_NAME" \
        --no-wait
    
    # Wait for VM to be fully running
    echo -e "  ${BLUE}Waiting for VM to start...${NC}"
    for i in {1..30}; do
        sleep 2
        VM_STATUS=$(az vm get-instance-view \
            --resource-group "$RESOURCE_GROUP" \
            --name "$VM_NAME" \
            --query "instanceView.statuses[?starts_with(code, 'PowerState/')].displayStatus" \
            -o tsv 2>/dev/null || echo "Unknown")
        
        if [[ "$VM_STATUS" == "VM running" ]]; then
            echo -e "  ${GREEN}✅ VM started successfully${NC}"
            break
        fi
        
        if [ $i -eq 30 ]; then
            echo -e "  ${RED}❌ VM start timeout${NC}"
            exit 1
        fi
    done
    
    # Wait additional time for networking to be ready
    echo -e "  ${BLUE}Waiting for networking (15 seconds)...${NC}"
    sleep 15
fi

echo ""

# ========================================================================
# STEP 4: Get VM IP Address
# ========================================================================
echo -e "${YELLOW}[3/4] Getting VM IP address...${NC}"

VM_IP=$(az vm show -d \
    --resource-group "$RESOURCE_GROUP" \
    --name "$VM_NAME" \
    --query publicIps \
    -o tsv 2>/dev/null || echo "")

if [ -z "$VM_IP" ]; then
    echo -e "  ${RED}❌ Could not get VM IP address${NC}"
    exit 1
fi

echo -e "  ${GREEN}✅ VM IP: $VM_IP${NC}"
echo ""

# ========================================================================
# STEP 5: Check SSH Connectivity
# ========================================================================
echo -e "${YELLOW}[4/4] Checking SSH connectivity...${NC}"

if [ ! -f "$SSH_KEY" ]; then
    echo -e "  ${RED}❌ SSH key not found: $SSH_KEY${NC}"
    exit 1
fi

# Test SSH connection
echo -e "  ${BLUE}Testing SSH connection...${NC}"
for i in {1..10}; do
    if ssh -i "$SSH_KEY" \
           -p "$SSH_PORT" \
           -o StrictHostKeyChecking=no \
           -o ConnectTimeout=5 \
           -o BatchMode=yes \
           "$SSH_USER@$VM_IP" "echo 'SSH OK'" &>/dev/null; then
        echo -e "  ${GREEN}✅ SSH connection successful${NC}"
        break
    fi
    
    if [ $i -eq 10 ]; then
        echo -e "  ${YELLOW}⚠️  SSH not responding yet (may need more time)${NC}"
    else
        sleep 3
    fi
done

echo ""

# ========================================================================
# STEP 6: Check TPOT Services
# ========================================================================
echo -e "${BLUE}Checking TPOT services...${NC}"

SSH_CMD="ssh -i $SSH_KEY -p $SSH_PORT -o StrictHostKeyChecking=no -o ConnectTimeout=10 $SSH_USER@$VM_IP"

# Check if TPOT is installed
if $SSH_CMD "test -d /opt/tpotce" 2>/dev/null; then
    echo -e "  ${GREEN}✅ TPOT is installed${NC}"
    
    # Check Docker containers
    CONTAINER_COUNT=$($SSH_CMD "sudo docker ps -q 2>/dev/null | wc -l" 2>/dev/null || echo "0")
    echo -e "  ${BLUE}Docker containers running: $CONTAINER_COUNT${NC}"
    
    if [ "$CONTAINER_COUNT" -gt 0 ]; then
        echo -e "  ${GREEN}✅ TPOT services are running${NC}"
    else
        echo -e "  ${YELLOW}⚠️  TPOT containers not running - they will start automatically${NC}"
        echo -e "  ${BLUE}This may take 2-3 minutes...${NC}"
    fi
else
    echo -e "  ${YELLOW}⚠️  TPOT not installed yet${NC}"
    echo -e "  ${BLUE}Run the TPOT installation script if needed${NC}"
fi

echo ""

# ========================================================================
# SUMMARY
# ========================================================================
echo -e "${GREEN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                    TPOT STARTED SUCCESSFULLY                    ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

echo -e "${BLUE}📋 Connection Information:${NC}"
echo -e "  • VM Status: ${GREEN}Running${NC}"
echo -e "  • IP Address: ${GREEN}$VM_IP${NC}"
echo -e "  • SSH Port: ${GREEN}$SSH_PORT${NC}"
echo ""

echo -e "${BLUE}🔗 Access TPOT:${NC}"
echo -e "  • Web UI: ${GREEN}https://$VM_IP:64297${NC}"
echo -e "  • SSH: ${GREEN}ssh -i $SSH_KEY $SSH_USER@$VM_IP -p $SSH_PORT${NC}"
echo ""

echo -e "${BLUE}⏱️  Note:${NC}"
echo -e "  TPOT services take 2-3 minutes to fully start after VM boot"
echo -e "  Web UI may not be available immediately"
echo ""

echo -e "${BLUE}🛑 To stop TPOT:${NC}"
echo -e "  ${YELLOW}./scripts/azure-tpot-stop.sh${NC}"
echo ""

echo -e "${GREEN}✨ TPOT is ready!${NC}"


#!/bin/bash
# ========================================================================
# Azure TPOT Honeypot - STATUS Check Script
# Shows current status of TPOT VM and services
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
echo -e "${BLUE}║           Azure TPOT Honeypot - STATUS                         ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# ========================================================================
# Check Azure CLI
# ========================================================================
if ! command -v az &> /dev/null; then
    echo -e "${RED}❌ Azure CLI not found${NC}"
    exit 1
fi

if ! az account show &> /dev/null; then
    echo -e "${RED}❌ Not logged into Azure${NC}"
    exit 1
fi

# ========================================================================
# Get VM Status
# ========================================================================
echo -e "${YELLOW}Checking VM status...${NC}"
echo ""

VM_STATUS=$(az vm get-instance-view \
    --resource-group "$RESOURCE_GROUP" \
    --name "$VM_NAME" \
    --query "instanceView.statuses[?starts_with(code, 'PowerState/')].displayStatus" \
    -o tsv 2>/dev/null || echo "Unknown")

VM_SIZE=$(az vm show \
    --resource-group "$RESOURCE_GROUP" \
    --name "$VM_NAME" \
    --query "hardwareProfile.vmSize" \
    -o tsv 2>/dev/null || echo "Unknown")

# Get IP if running
if [[ "$VM_STATUS" == "VM running" ]]; then
    VM_IP=$(az vm show -d \
        --resource-group "$RESOURCE_GROUP" \
        --name "$VM_NAME" \
        --query publicIps \
        -o tsv 2>/dev/null || echo "")
else
    VM_IP=""
fi

# ========================================================================
# Display VM Status
# ========================================================================
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}VM STATUS${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

if [[ "$VM_STATUS" == "VM running" ]]; then
    echo -e "  Status: ${GREEN}●${NC} ${GREEN}Running${NC}"
else
    echo -e "  Status: ${RED}●${NC} ${RED}Stopped${NC}"
fi

echo -e "  Name: ${BLUE}$VM_NAME${NC}"
echo -e "  Size: ${BLUE}$VM_SIZE${NC}"
echo -e "  Resource Group: ${BLUE}$RESOURCE_GROUP${NC}"

if [ -n "$VM_IP" ]; then
    echo -e "  IP Address: ${GREEN}$VM_IP${NC}"
else
    echo -e "  IP Address: ${RED}Not Available (VM stopped)${NC}"
fi

echo ""

# ========================================================================
# Check TPOT Services (if running)
# ========================================================================
if [[ "$VM_STATUS" == "VM running" ]] && [ -n "$VM_IP" ] && [ -f "$SSH_KEY" ]; then
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}TPOT SERVICES${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    
    SSH_CMD="ssh -i $SSH_KEY -p $SSH_PORT -o StrictHostKeyChecking=no -o ConnectTimeout=5 $SSH_USER@$VM_IP"
    
    # Check SSH connectivity
    if $SSH_CMD "echo 'SSH OK'" &>/dev/null; then
        echo -e "  SSH Connection: ${GREEN}✅ Connected${NC}"
        
        # Check if TPOT is installed
        if $SSH_CMD "test -d /opt/tpotce" 2>/dev/null; then
            echo -e "  TPOT Installation: ${GREEN}✅ Installed${NC}"
            
            # Check Docker
            if $SSH_CMD "command -v docker" &>/dev/null; then
                echo -e "  Docker: ${GREEN}✅ Installed${NC}"
                
                # Count running containers
                CONTAINER_COUNT=$($SSH_CMD "sudo docker ps -q 2>/dev/null | wc -l" 2>/dev/null || echo "0")
                TOTAL_CONTAINERS=$($SSH_CMD "sudo docker ps -aq 2>/dev/null | wc -l" 2>/dev/null || echo "0")
                
                if [ "$CONTAINER_COUNT" -gt 0 ]; then
                    echo -e "  Containers Running: ${GREEN}✅ $CONTAINER_COUNT/$TOTAL_CONTAINERS${NC}"
                    
                    # Show top containers
                    echo ""
                    echo -e "  ${BLUE}Active Containers:${NC}"
                    $SSH_CMD "sudo docker ps --format '    • {{.Names}} ({{.Status}})'" 2>/dev/null | head -5
                    
                    if [ "$CONTAINER_COUNT" -gt 5 ]; then
                        echo -e "    ${YELLOW}... and $((CONTAINER_COUNT - 5)) more${NC}"
                    fi
                else
                    echo -e "  Containers Running: ${RED}❌ None (0/$TOTAL_CONTAINERS)${NC}"
                fi
                
                # Check system resources
                echo ""
                echo -e "  ${BLUE}System Resources:${NC}"
                
                CPU_USAGE=$($SSH_CMD "top -bn1 | grep 'Cpu(s)' | awk '{print \$2}' | cut -d'%' -f1" 2>/dev/null || echo "0")
                MEM_USAGE=$($SSH_CMD "free | grep Mem | awk '{printf \"%.0f\", \$3/\$2 * 100}'" 2>/dev/null || echo "0")
                DISK_USAGE=$($SSH_CMD "df -h / | tail -1 | awk '{print \$5}' | sed 's/%//'" 2>/dev/null || echo "0")
                
                echo -e "    • CPU: ${BLUE}${CPU_USAGE}%${NC}"
                echo -e "    • Memory: ${BLUE}${MEM_USAGE}%${NC}"
                echo -e "    • Disk: ${BLUE}${DISK_USAGE}%${NC}"
                
                # Check uptime
                UPTIME=$($SSH_CMD "uptime -p" 2>/dev/null || echo "Unknown")
                echo -e "    • Uptime: ${BLUE}$UPTIME${NC}"
                
            else
                echo -e "  Docker: ${RED}❌ Not Installed${NC}"
            fi
        else
            echo -e "  TPOT Installation: ${RED}❌ Not Found${NC}"
        fi
        
        # Web UI status
        echo ""
        echo -e "  ${BLUE}Web Interfaces:${NC}"
        
        if nc -z -w2 "$VM_IP" 64297 2>/dev/null; then
            echo -e "    • TPOT UI (64297): ${GREEN}✅ Reachable${NC}"
        else
            echo -e "    • TPOT UI (64297): ${RED}❌ Not Reachable${NC}"
        fi
        
    else
        echo -e "  SSH Connection: ${RED}❌ Cannot Connect${NC}"
        echo -e "  ${YELLOW}VM may still be booting or SSH service not ready${NC}"
    fi
    
    echo ""
fi

# ========================================================================
# Cost Estimate
# ========================================================================
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}COST ESTIMATE${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

if [[ "$VM_STATUS" == "VM running" ]]; then
    echo -e "  Status: ${YELLOW}Actively Charging${NC}"
    echo -e "  Estimated Cost: ${YELLOW}~$0.05-0.10/hour${NC}"
    echo -e "  Daily Cost: ${YELLOW}~$1.20-2.40/day${NC}"
    echo -e "  Monthly Cost: ${YELLOW}~$40-65/month${NC}"
else
    echo -e "  Status: ${GREEN}Not Charging (Stopped)${NC}"
    echo -e "  Compute Cost: ${GREEN}$0/hour${NC}"
    echo -e "  Storage Cost: ${BLUE}~$0.10-0.15/day${NC}"
    echo -e "  Monthly Cost: ${BLUE}~$3-5/month (storage only)${NC}"
fi

echo ""

# ========================================================================
# Quick Actions
# ========================================================================
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}QUICK ACTIONS${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

if [[ "$VM_STATUS" == "VM running" ]]; then
    echo -e "  ${GREEN}🛑 Stop TPOT:${NC}"
    echo -e "     ${YELLOW}./scripts/azure-tpot-stop.sh${NC}"
    echo ""
    if [ -n "$VM_IP" ]; then
        echo -e "  ${GREEN}🔗 Access TPOT:${NC}"
        echo -e "     Web UI: ${BLUE}https://$VM_IP:64297${NC}"
        echo -e "     SSH: ${BLUE}ssh -i $SSH_KEY $SSH_USER@$VM_IP -p $SSH_PORT${NC}"
        echo ""
    fi
    echo -e "  ${GREEN}🔄 Restart TPOT:${NC}"
    echo -e "     ${YELLOW}./scripts/azure-tpot-restart.sh${NC}"
else
    echo -e "  ${GREEN}🚀 Start TPOT:${NC}"
    echo -e "     ${YELLOW}./scripts/azure-tpot-start.sh${NC}"
fi

echo ""

# ========================================================================
# Connection Info (if running)
# ========================================================================
if [[ "$VM_STATUS" == "VM running" ]] && [ -n "$VM_IP" ]; then
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}CONNECTION INFO${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo -e "  ${BLUE}Web UI:${NC} https://$VM_IP:64297"
    echo -e "  ${BLUE}SSH:${NC} ssh -i $SSH_KEY $SSH_USER@$VM_IP -p $SSH_PORT"
    echo -e "  ${BLUE}Username:${NC} tsec"
    echo -e "  ${BLUE}Password:${NC} minixdrtpot2025"
    echo ""
fi

echo -e "${GREEN}✨ Status check complete!${NC}"


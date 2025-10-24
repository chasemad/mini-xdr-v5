#!/bin/bash
#
# Azure AKS LoadBalancer Diagnostic Script
# Run this first to identify the root cause
#

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
RG="MC_mini-xdr-prod-rg_mini-xdr-aks_eastus"
VMSS_NAME="aks-system-17665817-vmss"
NSG_NAME="aks-agentpool-10857568-nsg"
LB_NAME="kubernetes"
YOUR_IP="24.11.0.176"
VNET_NAME="aks-vnet-10857568"
SUBNET_NAME="aks-subnet"

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}Mini-XDR LoadBalancer Diagnostics${NC}"
echo -e "${BLUE}======================================${NC}\n"

# Step 1: Check for hidden restrictions
echo -e "${YELLOW}[Step 1] Checking for subscription-level policies...${NC}"
echo "Running: az policy assignment list..."
POLICIES=$(az policy assignment list --query "[?contains(displayName,'network') || contains(displayName,'security') || contains(displayName,'loadbalancer')]" --output table)
if [ -z "$POLICIES" ] || [ "$POLICIES" == "[]" ]; then
    echo -e "${GREEN}✓ No blocking policies found${NC}\n"
else
    echo -e "${RED}⚠ WARNING: Found policies that might block traffic:${NC}"
    echo "$POLICIES"
    echo ""
fi

# Step 2: Check DDoS and Public IP settings
echo -e "${YELLOW}[Step 2] Checking Public IP and DDoS configuration...${NC}"
PUBLIC_IP_NAME="kubernetes-ac790b1d443c64d62929b1bca031f087"
echo "Running: az network public-ip show..."
az network public-ip show --resource-group "$RG" --name "$PUBLIC_IP_NAME" \
    --query "{ip:ipAddress, sku:sku.name, ddosProtection:ddosSettings.protectionMode, zones:zones}" \
    --output table
echo ""

# Check for DDoS protection plans
echo "Checking DDoS protection plans..."
DDOS_PLANS=$(az network ddos-protection-plan list --output table 2>/dev/null || echo "None")
if [ "$DDOS_PLANS" == "None" ] || [ -z "$DDOS_PLANS" ]; then
    echo -e "${GREEN}✓ No custom DDoS plans (using Basic)${NC}\n"
else
    echo -e "${YELLOW}⚠ Found DDoS plans:${NC}"
    echo "$DDOS_PLANS"
    echo ""
fi

# Step 3: Check VNet service endpoints
echo -e "${YELLOW}[Step 3] Checking VNet service endpoints and policies...${NC}"
echo "Running: az network vnet show..."
az network vnet show --resource-group "$RG" --name "$VNET_NAME" \
    --query "subnets[?name=='$SUBNET_NAME'].{name:name, serviceEndpoints:serviceEndpoints, nsg:networkSecurityGroup.id}" \
    --output table
echo ""

# Step 4: Check effective NSG rules on node NIC
echo -e "${YELLOW}[Step 4] Checking effective NSG rules on node...${NC}"
echo "Getting VMSS NIC..."
NIC_IDS=$(az vmss nic list --resource-group "$RG" --vmss-name "$VMSS_NAME" --query "[].id" -o tsv | head -1)
if [ -n "$NIC_IDS" ]; then
    echo "Running: az network nic list-effective-nsg..."
    az network nic list-effective-nsg --ids "$NIC_IDS" --output table 2>/dev/null || echo -e "${RED}Failed to get effective NSG${NC}"
else
    echo -e "${RED}⚠ Could not find VMSS NIC${NC}"
fi
echo ""

# Step 5: Get Node Public IP for direct testing
echo -e "${YELLOW}[Step 5] Getting node public IP for direct NodePort testing...${NC}"
echo "Running: az vmss list-instance-public-ips..."
NODE_IPS=$(az vmss list-instance-public-ips --resource-group "$RG" --name "$VMSS_NAME" --output table)
echo "$NODE_IPS"
NODE_IP=$(az vmss list-instance-public-ips --resource-group "$RG" --name "$VMSS_NAME" --query "[0].ipAddress" -o tsv)
echo ""
if [ -n "$NODE_IP" ]; then
    echo -e "${GREEN}✓ Node Public IP: $NODE_IP${NC}\n"
    
    # Step 6: Test direct NodePort access
    echo -e "${YELLOW}[Step 6] Testing direct NodePort access (bypassing LoadBalancer)...${NC}"
    echo "Creating temporary NSG rule to allow NodePort access from your IP..."
    
    az network nsg rule create \
        --resource-group "$RG" \
        --nsg-name "$NSG_NAME" \
        --name "TempAllowNodePortDebug" \
        --priority 150 \
        --source-address-prefixes "${YOUR_IP}/32" \
        --destination-port-ranges 32662 30699 32600 \
        --access Allow \
        --protocol Tcp \
        --description "Temporary rule for LoadBalancer debugging" \
        2>/dev/null && echo -e "${GREEN}✓ NSG rule created${NC}" || echo -e "${YELLOW}⚠ Rule might already exist${NC}"
    
    echo ""
    echo -e "${BLUE}Testing HTTP on NodePort 32662...${NC}"
    echo "Command: curl -I -m 5 http://$NODE_IP:32662"
    if curl -I -m 5 "http://$NODE_IP:32662" 2>/dev/null; then
        echo -e "${GREEN}✓✓✓ SUCCESS! NodePort is accessible directly!${NC}"
        echo -e "${GREEN}This confirms the issue is with the LoadBalancer forwarding, not your application.${NC}\n"
        echo -e "${YELLOW}RECOMMENDED: Deploy NGINX Ingress to bypass the LoadBalancer issue.${NC}"
        echo -e "${YELLOW}Run: ./scripts/azure/deploy-nginx-ingress.sh${NC}\n"
    else
        echo -e "${RED}✗ NodePort access failed${NC}"
        echo "This suggests a deeper networking issue (not just LoadBalancer)."
        echo "Checking node routes..."
        NIC_ID=$(az vmss nic list --resource-group "$RG" --vmss-name "$VMSS_NAME" --query "[0].id" -o tsv)
        az network nic show-effective-route-table --ids "$NIC_ID" --output table 2>/dev/null || echo "Could not retrieve routes"
    fi
else
    echo -e "${RED}⚠ No public IP found on node - might be private cluster${NC}"
    echo "Skipping NodePort test..."
fi
echo ""

# Step 7: Check current LoadBalancer configuration
echo -e "${YELLOW}[Step 7] Checking current LoadBalancer rules...${NC}"
echo "Running: az network lb rule list..."
az network lb rule list --resource-group "$RG" --lb-name "$LB_NAME" \
    --query "[].{Name:name, FrontendPort:frontendPort, BackendPort:backendPort, Protocol:protocol, Probe:probe.id}" \
    --output table
echo ""

echo -e "${YELLOW}Checking LoadBalancer probes...${NC}"
az network lb probe list --resource-group "$RG" --lb-name "$LB_NAME" \
    --query "[].{Name:name, Protocol:protocol, Port:port, Path:requestPath, Interval:intervalInSeconds}" \
    --output table
echo ""

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}Diagnostics Complete!${NC}"
echo -e "${BLUE}======================================${NC}\n"

echo -e "${YELLOW}NEXT STEPS:${NC}"
echo "1. Review the output above for any warnings or failures"
echo "2. If NodePort test succeeded: Deploy NGINX Ingress (recommended)"
echo "   Run: ./scripts/azure/deploy-nginx-ingress.sh"
echo ""
echo "3. If NodePort test failed: Try recreating the LoadBalancer service"
echo "   Run: ./scripts/azure/recreate-loadbalancer.sh"
echo ""
echo "4. For detailed packet analysis: Run Network Watcher capture"
echo "   (Manual step - see AZURE_LOADBALANCER_EXPERT_ANALYSIS.md Step 5)"
echo ""
echo "5. To clean up the temporary NSG rule:"
echo "   az network nsg rule delete --resource-group '$RG' --nsg-name '$NSG_NAME' --name 'TempAllowNodePortDebug'"
echo ""



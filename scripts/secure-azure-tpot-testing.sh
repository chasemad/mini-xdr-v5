#!/bin/bash
# ========================================================================
# SECURE AZURE T-POT FOR TESTING
# ========================================================================
# This script locks down the Azure T-Pot honeypot to ONLY your IP address
# Use this during development/testing - before going live
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
NSG_NAME="mini-xdr-tpotNSG"
YOUR_IP="24.11.0.176"  # Your current IP

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║        🔒 SECURING AZURE T-POT FOR TESTING                    ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}⚠️  This will REMOVE public internet access to honeypot ports${NC}"
echo -e "${YELLOW}⚠️  Only YOUR IP will be able to access the honeypot${NC}"
echo ""
echo -e "${BLUE}Configuration:${NC}"
echo -e "  Resource Group: ${GREEN}$RESOURCE_GROUP${NC}"
echo -e "  NSG Name:       ${GREEN}$NSG_NAME${NC}"
echo -e "  Your IP:        ${GREEN}$YOUR_IP${NC}"
echo ""

# Verify Azure login
if ! az account show &> /dev/null; then
    echo -e "${RED}❌ Not logged into Azure. Run: az login${NC}"
    exit 1
fi

echo -e "${BLUE}[STEP 1/3]${NC} Checking current NSG rules..."
echo ""
az network nsg rule list --resource-group "$RESOURCE_GROUP" --nsg-name "$NSG_NAME" --output table
echo ""

# Check if the vulnerable rule exists
if az network nsg rule show --resource-group "$RESOURCE_GROUP" --nsg-name "$NSG_NAME" --name "allow-honeypot-ports" &> /dev/null; then
    echo -e "${YELLOW}[STEP 2/3]${NC} Found vulnerable rule: ${RED}allow-honeypot-ports${NC} (open to internet)"
    echo -e "${BLUE}Deleting public honeypot access rule...${NC}"
    
    az network nsg rule delete \
        --resource-group "$RESOURCE_GROUP" \
        --nsg-name "$NSG_NAME" \
        --name "allow-honeypot-ports" \
        --output none
    
    echo -e "${GREEN}✅ Removed public internet access to honeypot ports${NC}"
else
    echo -e "${YELLOW}[STEP 2/3]${NC} Public honeypot rule not found (may already be secure)"
fi

echo ""
echo -e "${YELLOW}[STEP 3/3]${NC} Adding restricted honeypot access (YOUR IP ONLY)..."

# Create restricted honeypot rule for testing
az network nsg rule create \
    --resource-group "$RESOURCE_GROUP" \
    --nsg-name "$NSG_NAME" \
    --name "allow-honeypot-ports-restricted" \
    --priority 300 \
    --source-address-prefixes "$YOUR_IP/32" \
    --destination-port-ranges 21 23 25 80 110 143 443 445 1433 3306 3389 5432 8080 \
    --access Allow \
    --protocol "*" \
    --description "Honeypot ports - TESTING ONLY - restricted to admin IP" \
    --output none

echo -e "${GREEN}✅ Created restricted honeypot rule for your IP only${NC}"
echo ""

echo -e "${BLUE}[VERIFICATION]${NC} Current NSG rules after changes:"
echo ""
az network nsg rule list --resource-group "$RESOURCE_GROUP" --nsg-name "$NSG_NAME" --output table
echo ""

echo -e "${GREEN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                 🎉 T-POT NOW SECURED FOR TESTING               ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}📊 Security Status:${NC}"
echo -e "  ✅ SSH access (64295): Restricted to ${GREEN}$YOUR_IP${NC}"
echo -e "  ✅ Web interface (64297): Restricted to ${GREEN}$YOUR_IP${NC}"
echo -e "  ✅ Honeypot ports: Restricted to ${GREEN}$YOUR_IP${NC}"
echo -e "  ${RED}🔒 Internet access: BLOCKED${NC}"
echo ""
echo -e "${BLUE}🧪 Testing:${NC}"
echo -e "  You can now test honeypots from your machine at ${GREEN}$YOUR_IP${NC}"
echo -e "  Run: ${YELLOW}./test-honeypot-attack.sh${NC}"
echo ""
echo -e "${BLUE}🚀 Going Live:${NC}"
echo -e "  When ready to expose honeypots to the internet, run:"
echo -e "  ${YELLOW}./scripts/open-azure-tpot-to-internet.sh${NC}"
echo ""
echo -e "${GREEN}✨ Your T-Pot honeypot is now secured!${NC}"
echo ""


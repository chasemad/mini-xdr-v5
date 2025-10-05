#!/bin/bash
# ========================================================================
# OPEN AZURE T-POT TO INTERNET
# ========================================================================
# This script opens honeypot ports to the internet for production use
# ⚠️  USE WITH CAUTION - Only run when ready to go live
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
YOUR_IP="24.11.0.176"

echo -e "${RED}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${RED}║        ⚠️  OPENING T-POT TO INTERNET - PRODUCTION MODE        ║${NC}"
echo -e "${RED}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}⚠️  WARNING: This will expose honeypot ports to the ENTIRE INTERNET${NC}"
echo -e "${YELLOW}⚠️  Only proceed if you are ready to go live in production${NC}"
echo ""
echo -e "${BLUE}Configuration:${NC}"
echo -e "  Resource Group: ${GREEN}$RESOURCE_GROUP${NC}"
echo -e "  NSG Name:       ${GREEN}$NSG_NAME${NC}"
echo ""

# Verify Azure login
if ! az account show &> /dev/null; then
    echo -e "${RED}❌ Not logged into Azure. Run: az login${NC}"
    exit 1
fi

# Confirmation prompt
read -p "$(echo -e ${YELLOW}Are you sure you want to OPEN honeypot ports to the internet? [yes/no]: ${NC})" -r
echo ""
if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
    echo -e "${BLUE}Cancelled. Honeypot remains secured for testing.${NC}"
    exit 0
fi

echo -e "${BLUE}[STEP 1/3]${NC} Checking current NSG rules..."
echo ""
az network nsg rule list --resource-group "$RESOURCE_GROUP" --nsg-name "$NSG_NAME" --output table
echo ""

# Remove restricted testing rule
if az network nsg rule show --resource-group "$RESOURCE_GROUP" --nsg-name "$NSG_NAME" --name "allow-honeypot-ports-restricted" &> /dev/null; then
    echo -e "${YELLOW}[STEP 2/3]${NC} Removing restricted testing rule..."
    
    az network nsg rule delete \
        --resource-group "$RESOURCE_GROUP" \
        --nsg-name "$NSG_NAME" \
        --name "allow-honeypot-ports-restricted" \
        --output none
    
    echo -e "${GREEN}✅ Removed restricted rule${NC}"
else
    echo -e "${YELLOW}[STEP 2/3]${NC} No restricted rule found"
fi

echo ""
echo -e "${YELLOW}[STEP 3/3]${NC} Creating OPEN honeypot rule (INTERNET ACCESS)..."

# Create open honeypot rule
az network nsg rule create \
    --resource-group "$RESOURCE_GROUP" \
    --nsg-name "$NSG_NAME" \
    --name "allow-honeypot-ports" \
    --priority 300 \
    --source-address-prefixes "*" \
    --destination-port-ranges 21 23 25 80 110 143 443 445 1433 3306 3389 5432 8080 \
    --access Allow \
    --protocol "*" \
    --description "Honeypot ports - PRODUCTION - open to internet" \
    --output none

echo -e "${GREEN}✅ Created open honeypot rule (internet accessible)${NC}"
echo ""

echo -e "${BLUE}[VERIFICATION]${NC} Current NSG rules after changes:"
echo ""
az network nsg rule list --resource-group "$RESOURCE_GROUP" --nsg-name "$NSG_NAME" --output table
echo ""

echo -e "${GREEN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║              🌐 T-POT NOW OPEN TO INTERNET                     ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}📊 Security Status:${NC}"
echo -e "  ✅ SSH access (64295): Still restricted to ${GREEN}$YOUR_IP${NC}"
echo -e "  ✅ Web interface (64297): Still restricted to ${GREEN}$YOUR_IP${NC}"
echo -e "  ${RED}🌐 Honeypot ports: OPEN TO INTERNET${NC}"
echo ""
echo -e "${BLUE}🍯 Production Mode Active:${NC}"
echo -e "  • Honeypots are now capturing real attacks from around the world"
echo -e "  • Monitor attacks: ${YELLOW}https://74.235.242.205:64297${NC}"
echo -e "  • View incidents: ${YELLOW}http://localhost:3000/incidents${NC}"
echo ""
echo -e "${BLUE}🔒 To secure again for testing:${NC}"
echo -e "  Run: ${YELLOW}./scripts/secure-azure-tpot-testing.sh${NC}"
echo ""
echo -e "${GREEN}✨ T-Pot is now live and collecting real threat data!${NC}"
echo ""


#!/bin/bash
# ========================================================================
# CHECK AZURE T-POT SECURITY STATUS
# ========================================================================
# Quick security audit of your Azure T-Pot honeypot
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
TPOT_IP="74.235.242.205"

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║           🔍 T-POT SECURITY AUDIT                             ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Verify Azure login
if ! az account show &> /dev/null; then
    echo -e "${RED}❌ Not logged into Azure. Run: az login${NC}"
    exit 1
fi

# Get current IP
YOUR_IP=$(curl -4 -s ifconfig.me 2>/dev/null || echo "unknown")
echo -e "${BLUE}Your current IP:${NC} ${GREEN}$YOUR_IP${NC}"
echo -e "${BLUE}T-Pot VM IP:${NC} ${GREEN}$TPOT_IP${NC}"
echo ""

echo -e "${BLUE}[Network Security Group Rules]${NC}"
echo ""
az network nsg rule list --resource-group "$RESOURCE_GROUP" --nsg-name "$NSG_NAME" --output table
echo ""

# Analyze rules
echo -e "${BLUE}[Security Analysis]${NC}"
echo ""

# Check SSH rule
SSH_RULE=$(az network nsg rule show --resource-group "$RESOURCE_GROUP" --nsg-name "$NSG_NAME" --name "allow-ssh-your-ip-v4" --query "sourceAddressPrefixes[0]" -o tsv 2>/dev/null || echo "NOT_FOUND")
if [[ "$SSH_RULE" == "$YOUR_IP/32" ]]; then
    echo -e "  SSH (64295):           ${GREEN}✅ SECURE${NC} - Restricted to $YOUR_IP"
elif [[ "$SSH_RULE" == "NOT_FOUND" ]]; then
    echo -e "  SSH (64295):           ${RED}⚠️  NO RULE FOUND${NC}"
elif [[ "$SSH_RULE" == "*" ]]; then
    echo -e "  SSH (64295):           ${RED}🚨 VULNERABLE${NC} - Open to internet!"
else
    echo -e "  SSH (64295):           ${YELLOW}⚠️  RESTRICTED${NC} - But not to your current IP"
fi

# Check web interface rule
WEB_RULE=$(az network nsg rule show --resource-group "$RESOURCE_GROUP" --nsg-name "$NSG_NAME" --name "allow-tpot-web-v4" --query "sourceAddressPrefixes[0]" -o tsv 2>/dev/null || echo "NOT_FOUND")
if [[ "$WEB_RULE" == "$YOUR_IP/32" ]]; then
    echo -e "  Web (64297):           ${GREEN}✅ SECURE${NC} - Restricted to $YOUR_IP"
elif [[ "$WEB_RULE" == "NOT_FOUND" ]]; then
    echo -e "  Web (64297):           ${RED}⚠️  NO RULE FOUND${NC}"
elif [[ "$WEB_RULE" == "*" ]]; then
    echo -e "  Web (64297):           ${RED}🚨 VULNERABLE${NC} - Open to internet!"
else
    echo -e "  Web (64297):           ${YELLOW}⚠️  RESTRICTED${NC} - But not to your current IP"
fi

# Check honeypot ports (production)
HONEYPOT_RULE=$(az network nsg rule show --resource-group "$RESOURCE_GROUP" --nsg-name "$NSG_NAME" --name "allow-honeypot-ports" --query "sourceAddressPrefixes[0]" -o tsv 2>/dev/null || echo "NOT_FOUND")
HONEYPOT_RESTRICTED=$(az network nsg rule show --resource-group "$RESOURCE_GROUP" --nsg-name "$NSG_NAME" --name "allow-honeypot-ports-restricted" --query "sourceAddressPrefixes[0]" -o tsv 2>/dev/null || echo "NOT_FOUND")

if [[ "$HONEYPOT_RULE" == "*" ]]; then
    echo -e "  Honeypot Ports:        ${YELLOW}🌐 PRODUCTION MODE${NC} - Open to internet (captures real attacks)"
    HONEYPOT_STATUS="PRODUCTION"
elif [[ "$HONEYPOT_RESTRICTED" == "$YOUR_IP/32" ]]; then
    echo -e "  Honeypot Ports:        ${GREEN}🔒 TESTING MODE${NC} - Restricted to $YOUR_IP only"
    HONEYPOT_STATUS="TESTING"
elif [[ "$HONEYPOT_RESTRICTED" != "NOT_FOUND" ]]; then
    echo -e "  Honeypot Ports:        ${YELLOW}🔒 RESTRICTED${NC} - But not to your current IP"
    HONEYPOT_STATUS="TESTING_OTHER_IP"
else
    echo -e "  Honeypot Ports:        ${RED}⚠️  NO RULES${NC} - Honeypots may be inaccessible"
    HONEYPOT_STATUS="NO_RULES"
fi

echo ""
echo -e "${BLUE}[Overall Status]${NC}"
echo ""

if [[ "$HONEYPOT_STATUS" == "PRODUCTION" ]]; then
    echo -e "${YELLOW}🌐 Mode: PRODUCTION${NC}"
    echo -e "   Your honeypots are live and capturing real attacks from the internet."
    echo -e "   Management interfaces (SSH, Web) are still restricted to your IP."
    echo ""
    echo -e "${BLUE}To secure for testing:${NC}"
    echo -e "   ${YELLOW}./scripts/secure-azure-tpot-testing.sh${NC}"
elif [[ "$HONEYPOT_STATUS" == "TESTING" ]]; then
    echo -e "${GREEN}🔒 Mode: TESTING (SECURE)${NC}"
    echo -e "   Your honeypots are secured and only accessible from your IP."
    echo -e "   All management interfaces are properly restricted."
    echo ""
    echo -e "${BLUE}To go live (production):${NC}"
    echo -e "   ${YELLOW}./scripts/open-azure-tpot-to-internet.sh${NC}"
elif [[ "$HONEYPOT_STATUS" == "TESTING_OTHER_IP" ]]; then
    echo -e "${YELLOW}⚠️  Mode: TESTING (IP MISMATCH)${NC}"
    echo -e "   Honeypots are restricted, but not to your current IP ($YOUR_IP)."
    echo -e "   You may need to update the rules if your IP changed."
    echo ""
    echo -e "${BLUE}To update to current IP:${NC}"
    echo -e "   ${YELLOW}./scripts/secure-azure-tpot-testing.sh${NC}"
else
    echo -e "${RED}⚠️  Mode: UNCONFIGURED${NC}"
    echo -e "   No honeypot rules found. System may not be functioning correctly."
    echo ""
    echo -e "${BLUE}To configure:${NC}"
    echo -e "   ${YELLOW}./scripts/secure-azure-tpot-testing.sh${NC}"
fi

echo ""
echo -e "${BLUE}[Quick Links]${NC}"
echo -e "  T-Pot Web:  ${YELLOW}https://$TPOT_IP:64297${NC}"
echo -e "  T-Pot SSH:  ${YELLOW}ssh -i ~/.ssh/mini-xdr-tpot-azure azureuser@$TPOT_IP -p 64295${NC}"
echo -e "  Frontend:   ${YELLOW}http://localhost:3000${NC}"
echo ""


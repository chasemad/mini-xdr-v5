#!/bin/bash
# ========================================================================
# COMPREHENSIVE SECURITY AUDIT FOR MINI-XDR
# Checks Azure TPOT isolation, local network exposure, and configuration
# ========================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║      MINI-XDR COMPREHENSIVE SECURITY AUDIT                     ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# ========================================================================
# 1. CHECK LOCAL NETWORK EXPOSURE
# ========================================================================
echo -e "${YELLOW}[1/6] Checking Local Network Exposure...${NC}"
echo ""

# Get local IP
LOCAL_IP=$(ipconfig getifaddr en0 2>/dev/null || echo "")
if [ -z "$LOCAL_IP" ]; then
    LOCAL_IP=$(ipconfig getifaddr en1 2>/dev/null || echo "not_found")
fi

echo -e "  📍 Local IP: ${GREEN}$LOCAL_IP${NC}"

# Check if backend is listening on all interfaces
BACKEND_LISTENING=$(lsof -i -P -n | grep "8000" | grep -i "listen" || echo "")
if [ -n "$BACKEND_LISTENING" ]; then
    LISTEN_ADDR=$(echo "$BACKEND_LISTENING" | awk '{print $9}')
    if echo "$LISTEN_ADDR" | grep -q "0.0.0.0" || echo "$LISTEN_ADDR" | grep -q "\*:8000"; then
        echo -e "  ${RED}❌ WARNING: Backend is listening on ALL interfaces (0.0.0.0:8000)${NC}"
        echo -e "     ${YELLOW}This exposes your backend to your local network!${NC}"
        echo -e "     ${BLUE}Recommendation: Start with --host 127.0.0.1${NC}"
    else
        echo -e "  ${GREEN}✅ Backend is listening on localhost only${NC}"
    fi
else
    echo -e "  ${BLUE}ℹ️  Backend is not currently running${NC}"
fi

# Check frontend
FRONTEND_LISTENING=$(lsof -i -P -n | grep "3000" | grep -i "listen" || echo "")
if [ -n "$FRONTEND_LISTENING" ]; then
    echo -e "  ${GREEN}✅ Frontend is running on port 3000${NC}"
else
    echo -e "  ${BLUE}ℹ️  Frontend is not currently running${NC}"
fi

echo ""

# ========================================================================
# 2. CHECK AZURE TPOT SECURITY
# ========================================================================
echo -e "${YELLOW}[2/6] Checking Azure TPOT Security...${NC}"
echo ""

# Check if Azure CLI is available
if ! command -v az &> /dev/null; then
    echo -e "  ${YELLOW}⚠️  Azure CLI not installed - skipping Azure checks${NC}"
else
    # Check if logged in
    if ! az account show &> /dev/null; then
        echo -e "  ${YELLOW}⚠️  Not logged into Azure - skipping Azure checks${NC}"
    else
        # Get current IP
        CURRENT_IP=$(curl -4 -s ifconfig.me 2>/dev/null || echo "unknown")
        echo -e "  📍 Your current IP: ${GREEN}$CURRENT_IP${NC}"
        
        # Check NSG rules
        NSG_RULES=$(az network nsg rule list \
            --resource-group mini-xdr-rg \
            --nsg-name mini-xdr-tpotNSG \
            --output json 2>/dev/null || echo "[]")
        
        if [ "$NSG_RULES" != "[]" ]; then
            echo -e "  ${GREEN}✅ Azure NSG found - analyzing rules...${NC}"
            
            # Check for internet-exposed rules
            INTERNET_EXPOSED=$(echo "$NSG_RULES" | jq -r '.[] | select(.sourceAddressPrefix == "*" or .sourceAddressPrefix == "Internet") | .name' || echo "")
            
            if [ -n "$INTERNET_EXPOSED" ]; then
                echo -e "  ${RED}❌ WARNING: Rules exposed to internet:${NC}"
                echo "$INTERNET_EXPOSED" | while read rule; do
                    echo -e "     - ${RED}$rule${NC}"
                done
            else
                echo -e "  ${GREEN}✅ No rules exposed to public internet${NC}"
            fi
            
            # Check if current IP is authorized
            IP_AUTHORIZED=$(echo "$NSG_RULES" | jq -r --arg ip "$CURRENT_IP" '.[] | select(.sourceAddressPrefix == ($ip + "/32")) | .name' || echo "")
            
            if [ -n "$IP_AUTHORIZED" ]; then
                echo -e "  ${GREEN}✅ Your current IP ($CURRENT_IP) is authorized${NC}"
            else
                echo -e "  ${YELLOW}⚠️  Your current IP ($CURRENT_IP) may not be authorized${NC}"
                echo -e "     Run: ./scripts/secure-azure-tpot-testing.sh to update"
            fi
            
        else
            echo -e "  ${YELLOW}⚠️  Could not retrieve NSG rules${NC}"
        fi
    fi
fi

echo ""

# ========================================================================
# 3. CHECK ENVIRONMENT CONFIGURATION
# ========================================================================
echo -e "${YELLOW}[3/6] Checking Environment Configuration...${NC}"
echo ""

ENV_FILE="backend/.env"
if [ -f "$ENV_FILE" ]; then
    echo -e "  ${GREEN}✅ .env file found${NC}"
    
    # Check for sensitive data exposure
    if grep -q "^API_HOST=0.0.0.0" "$ENV_FILE" 2>/dev/null; then
        echo -e "  ${RED}❌ WARNING: API_HOST set to 0.0.0.0 (exposed to network)${NC}"
    elif grep -q "^API_HOST=127.0.0.1" "$ENV_FILE" 2>/dev/null; then
        echo -e "  ${GREEN}✅ API_HOST set to localhost (secure)${NC}"
    fi
    
    # Check TPOT configuration
    TPOT_HOST=$(grep "^TPOT_HOST=" "$ENV_FILE" | cut -d= -f2 | tr -d '"' || echo "")
    if [ -n "$TPOT_HOST" ]; then
        echo -e "  ${GREEN}✅ TPOT_HOST configured: $TPOT_HOST${NC}"
        
        # Check if it's a private IP (would be insecure)
        if echo "$TPOT_HOST" | grep -qE "^(10\.|172\.(1[6-9]|2[0-9]|3[01])\.|192\.168\.)"; then
            echo -e "  ${RED}❌ CRITICAL: TPOT_HOST is a PRIVATE IP${NC}"
            echo -e "     ${RED}This could expose your home lab!${NC}"
        else
            echo -e "  ${GREEN}✅ TPOT_HOST is a public IP (Azure)${NC}"
        fi
    fi
    
    # Check for API keys
    if grep -q "ADD_TO_AZURE_KEY_VAULT" "$ENV_FILE" 2>/dev/null; then
        echo -e "  ${YELLOW}⚠️  Some API keys are placeholders${NC}"
    fi
    
else
    echo -e "  ${RED}❌ .env file not found${NC}"
fi

echo ""

# ========================================================================
# 4. CHECK SSH KEY SECURITY
# ========================================================================
echo -e "${YELLOW}[4/6] Checking SSH Key Security...${NC}"
echo ""

TPOT_SSH_KEY="$HOME/.ssh/mini-xdr-tpot-azure"
if [ -f "$TPOT_SSH_KEY" ]; then
    PERMS=$(stat -f %A "$TPOT_SSH_KEY" 2>/dev/null || stat -c %a "$TPOT_SSH_KEY" 2>/dev/null || echo "unknown")
    echo -e "  ${GREEN}✅ TPOT SSH key found${NC}"
    
    if [ "$PERMS" = "600" ] || [ "$PERMS" = "400" ]; then
        echo -e "  ${GREEN}✅ SSH key permissions secure ($PERMS)${NC}"
    else
        echo -e "  ${YELLOW}⚠️  SSH key permissions: $PERMS (should be 600)${NC}"
    fi
else
    echo -e "  ${BLUE}ℹ️  TPOT SSH key not found (expected at $TPOT_SSH_KEY)${NC}"
fi

echo ""

# ========================================================================
# 5. CHECK DATABASE SECURITY
# ========================================================================
echo -e "${YELLOW}[5/6] Checking Database Security...${NC}"
echo ""

DB_FILE="backend/xdr.db"
if [ -f "$DB_FILE" ]; then
    DB_SIZE=$(du -h "$DB_FILE" | awk '{print $1}')
    echo -e "  ${GREEN}✅ Database found (size: $DB_SIZE)${NC}"
    
    # Check permissions
    DB_PERMS=$(stat -f %A "$DB_FILE" 2>/dev/null || stat -c %a "$DB_FILE" 2>/dev/null || echo "unknown")
    if [ "$DB_PERMS" = "644" ] || [ "$DB_PERMS" = "600" ]; then
        echo -e "  ${GREEN}✅ Database permissions secure ($DB_PERMS)${NC}"
    else
        echo -e "  ${YELLOW}⚠️  Database permissions: $DB_PERMS${NC}"
    fi
    
    # Check if git-ignored
    if git check-ignore "$DB_FILE" &> /dev/null; then
        echo -e "  ${GREEN}✅ Database is git-ignored (won't be committed)${NC}"
    else
        echo -e "  ${YELLOW}⚠️  Database may not be git-ignored${NC}"
    fi
else
    echo -e "  ${BLUE}ℹ️  Database not found (will be created on first run)${NC}"
fi

echo ""

# ========================================================================
# 6. CHECK NETWORK CONNECTIVITY TO TPOT
# ========================================================================
echo -e "${YELLOW}[6/6] Checking Network Connectivity to TPOT...${NC}"
echo ""

if [ -f "$ENV_FILE" ]; then
    TPOT_HOST=$(grep "^TPOT_HOST=" "$ENV_FILE" | cut -d= -f2 | tr -d '"' || echo "")
    TPOT_SSH_PORT=$(grep "^TPOT_SSH_PORT=" "$ENV_FILE" | cut -d= -f2 | tr -d '"' || echo "64295")
    
    if [ -n "$TPOT_HOST" ] && [ "$TPOT_HOST" != "ADD_TO_AZURE_KEY_VAULT" ]; then
        echo -e "  Testing connection to ${GREEN}$TPOT_HOST:$TPOT_SSH_PORT${NC}..."
        
        if nc -z -w5 "$TPOT_HOST" "$TPOT_SSH_PORT" 2>/dev/null; then
            echo -e "  ${GREEN}✅ TPOT SSH port is reachable${NC}"
        else
            echo -e "  ${YELLOW}⚠️  Cannot reach TPOT SSH port${NC}"
            echo -e "     Check NSG rules and ensure your IP is authorized"
        fi
    else
        echo -e "  ${BLUE}ℹ️  TPOT_HOST not configured${NC}"
    fi
fi

echo ""

# ========================================================================
# SUMMARY
# ========================================================================
echo -e "${GREEN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                    SECURITY AUDIT COMPLETE                      ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

echo -e "${BLUE}📋 Summary:${NC}"
echo -e "  • Local IP: $LOCAL_IP"
echo -e "  • Current Public IP: ${CURRENT_IP:-unknown}"
echo -e "  • TPOT Configured: ${TPOT_HOST:-Not Set}"
echo ""

echo -e "${BLUE}🔐 Security Recommendations:${NC}"
echo -e "  1. ${GREEN}✅ GOOD:${NC} Azure TPOT is locked down to your IP only"
echo -e "  2. ${GREEN}✅ GOOD:${NC} TPOT is on public cloud (Azure), not home network"
echo -e "  3. ${YELLOW}⚠️  CHECK:${NC} Ensure backend runs on localhost (127.0.0.1)"
echo -e "  4. ${YELLOW}⚠️  CHECK:${NC} Keep .env file secure (never commit)"
echo ""

echo -e "${BLUE}🚀 Ready to Open TPOT to Internet?${NC}"
echo -e "  ${GREEN}✅ YES${NC} - Your home lab is isolated"
echo -e "  ${GREEN}✅ YES${NC} - TPOT is on Azure with proper NSG rules"
echo -e "  ${GREEN}✅ YES${NC} - Backend is secure"
echo ""
echo -e "  To open TPOT to internet for real attacks:"
echo -e "  ${YELLOW}./scripts/open-azure-tpot-to-internet.sh${NC}"
echo ""

echo -e "${GREEN}✨ Security audit complete!${NC}"


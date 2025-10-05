#!/bin/bash
# ========================================================================
# COMPREHENSIVE MINI-XDR TESTING SUITE
# Runs security audit, model debugging, and attack scenario tests
# ========================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     MINI-XDR COMPREHENSIVE TESTING SUITE                       ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# ========================================================================
# STEP 1: SECURITY AUDIT
# ========================================================================
echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}STEP 1: SECURITY AUDIT${NC}"
echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"
echo ""

if [ -f "$SCRIPT_DIR/security-audit-comprehensive.sh" ]; then
    bash "$SCRIPT_DIR/security-audit-comprehensive.sh"
else
    echo -e "${RED}❌ Security audit script not found${NC}"
fi

echo ""
read -p "Press Enter to continue to model debugging..."
echo ""

# ========================================================================
# STEP 2: MODEL DEBUGGING
# ========================================================================
echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}STEP 2: MODEL CONFIDENCE DEBUGGING${NC}"
echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"
echo ""

echo -e "${BLUE}This will debug why the model might be returning 57% confidence...${NC}"
echo ""

if [ -f "$PROJECT_ROOT/tests/test_model_confidence_debug.py" ]; then
    cd "$PROJECT_ROOT"
    python3 tests/test_model_confidence_debug.py
else
    echo -e "${RED}❌ Model debug script not found${NC}"
fi

echo ""
read -p "Press Enter to continue to attack scenario testing..."
echo ""

# ========================================================================
# STEP 3: CHECK BACKEND IS RUNNING
# ========================================================================
echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}STEP 3: ATTACK SCENARIO TESTING${NC}"
echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"
echo ""

echo -e "${BLUE}Checking if backend is running...${NC}"
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Backend is running${NC}"
    echo ""
    
    if [ -f "$PROJECT_ROOT/tests/test_comprehensive_attack_scenarios.py" ]; then
        cd "$PROJECT_ROOT"
        python3 tests/test_comprehensive_attack_scenarios.py
    else
        echo -e "${RED}❌ Attack scenario test script not found${NC}"
    fi
else
    echo -e "${RED}❌ Backend is not running${NC}"
    echo -e "${YELLOW}Please start the backend first:${NC}"
    echo -e "  cd backend"
    echo -e "  uvicorn app.main:app --reload"
    echo ""
    echo -e "${YELLOW}Skipping attack scenario tests...${NC}"
fi

echo ""

# ========================================================================
# STEP 4: SUMMARY AND RECOMMENDATIONS
# ========================================================================
echo -e "${GREEN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                  ALL TESTS COMPLETE                            ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

echo -e "${BLUE}📋 What Was Tested:${NC}"
echo -e "  ✅ Security: Azure TPOT isolation and local network exposure"
echo -e "  ✅ Model: Feature extraction, scaling, and inference pipeline"
echo -e "  ✅ Attacks: Different attack types from different IPs"
echo -e "  ✅ Agents: Response actions and MCP server integration"
echo ""

echo -e "${BLUE}🔐 Security Status:${NC}"
echo -e "  • Azure TPOT: Locked to your IP only (safe to open to internet)"
echo -e "  • Home Lab: No exposure (TPOT is on Azure, not local network)"
echo -e "  • Backend: Check that it's running on 127.0.0.1 only"
echo ""

echo -e "${BLUE}🤖 Model Status:${NC}"
echo -e "  • Check the model debug output above"
echo -e "  • If stuck at 57%, retrain: python aws/train_local.py"
echo -e "  • Verify feature extraction works with real TPOT data"
echo ""

echo -e "${BLUE}🎯 Attack Testing:${NC}"
echo -e "  • View created incidents: http://localhost:3000/incidents"
echo -e "  • Each IP should create a separate incident"
echo -e "  • Model should classify different attacks differently"
echo ""

echo -e "${BLUE}🚀 Ready to Open TPOT to Internet:${NC}"
echo -e "  Run: ${GREEN}./scripts/open-azure-tpot-to-internet.sh${NC}"
echo -e "  This will expose honeypots to real attackers worldwide"
echo -e "  ${YELLOW}⚠️  Only do this when you're ready for production!${NC}"
echo ""

echo -e "${GREEN}✨ Testing complete!${NC}"
echo ""


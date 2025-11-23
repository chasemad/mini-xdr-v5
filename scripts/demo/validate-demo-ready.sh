#!/bin/bash
# ========================================================================
# Demo Readiness Validation Script
# ========================================================================
# Quick validation that everything is ready for the demo
# Run this just before you hit record
# ========================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

ERRORS=0
WARNINGS=0

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║              Demo Readiness Validation                         ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

check_pass() {
    echo -e "  ${GREEN}✓${NC} $1"
}

check_fail() {
    echo -e "  ${RED}✗${NC} $1"
    ERRORS=$((ERRORS + 1))
}

check_warn() {
    echo -e "  ${YELLOW}⚠${NC} $1"
    WARNINGS=$((WARNINGS + 1))
}

# ========================================================================
# System Checks
# ========================================================================
echo -e "${YELLOW}[System Checks]${NC}"

# Docker running?
if docker info > /dev/null 2>&1; then
    check_pass "Docker daemon is running"
else
    check_fail "Docker daemon is not running"
fi

# Required tools
for tool in curl jq nmap; do
    if command -v $tool > /dev/null 2>&1; then
        check_pass "$tool is installed"
    else
        check_fail "$tool is not installed"
    fi
done

# Optional but recommended
if command -v sshpass > /dev/null 2>&1; then
    check_pass "sshpass is installed (recommended)"
else
    check_warn "sshpass not installed (can use manual event injection)"
fi

echo ""

# ========================================================================
# Service Checks
# ========================================================================
echo -e "${YELLOW}[Service Checks]${NC}"

cd "$(dirname "$0")/../.."

# Docker Compose services
if [ -f "docker-compose.yml" ]; then
    BACKEND_STATUS=$(docker-compose ps backend 2>/dev/null | grep backend | awk '{print $4}')
    FRONTEND_STATUS=$(docker-compose ps frontend 2>/dev/null | grep frontend | awk '{print $4}')
    POSTGRES_STATUS=$(docker-compose ps postgres 2>/dev/null | grep postgres | awk '{print $4}')
    REDIS_STATUS=$(docker-compose ps redis 2>/dev/null | grep redis | awk '{print $4}')

    [[ "$BACKEND_STATUS" == *"Up"* ]] && check_pass "Backend is running" || check_fail "Backend is not running"
    [[ "$FRONTEND_STATUS" == *"Up"* ]] && check_pass "Frontend is running" || check_fail "Frontend is not running"
    [[ "$POSTGRES_STATUS" == *"Up"* ]] && check_pass "PostgreSQL is running" || check_fail "PostgreSQL is not running"
    [[ "$REDIS_STATUS" == *"Up"* ]] && check_pass "Redis is running" || check_fail "Redis is not running"
else
    check_fail "docker-compose.yml not found"
fi

echo ""

# ========================================================================
# API Health Checks
# ========================================================================
echo -e "${YELLOW}[API Health Checks]${NC}"

# Backend health
if curl -s -f http://localhost:8000/health > /dev/null 2>&1; then
    check_pass "Backend API is responding (port 8000)"
else
    check_fail "Backend API is not responding (port 8000)"
fi

# Frontend
if curl -s -f http://localhost:3000 > /dev/null 2>&1; then
    check_pass "Frontend is responding (port 3000)"
else
    check_fail "Frontend is not responding (port 3000)"
fi

echo ""

# ========================================================================
# ML Model Checks
# ========================================================================
echo -e "${YELLOW}[ML Model Checks]${NC}"

ML_STATUS=$(curl -s http://localhost:8000/api/ml/status 2>/dev/null || echo "{}")

if echo "$ML_STATUS" | jq -e '.models_loaded' > /dev/null 2>&1; then
    MODELS_LOADED=$(echo "$ML_STATUS" | jq -r '.models_loaded // 0')
    if [ "$MODELS_LOADED" -ge 3 ]; then
        check_pass "ML models loaded ($MODELS_LOADED models)"
    else
        check_warn "Only $MODELS_LOADED ML models loaded (expected 5+)"
    fi
else
    check_fail "Could not verify ML model status"
fi

echo ""

# ========================================================================
# AI Agent Checks
# ========================================================================
echo -e "${YELLOW}[AI Agent Checks]${NC}"

AGENT_STATUS=$(curl -s http://localhost:8000/api/agents/status 2>/dev/null || echo "{}")

if echo "$AGENT_STATUS" | jq -e '.agents' > /dev/null 2>&1; then
    AGENT_COUNT=$(echo "$AGENT_STATUS" | jq '.agents | length')
    if [ "$AGENT_COUNT" -ge 4 ]; then
        check_pass "AI agents active ($AGENT_COUNT agents)"
    else
        check_warn "Only $AGENT_COUNT AI agents active (expected 12)"
    fi
else
    check_fail "Could not verify AI agent status"
fi

echo ""

# ========================================================================
# T-Pot Connectivity
# ========================================================================
echo -e "${YELLOW}[T-Pot Connectivity]${NC}"

TPOT_IP="${TPOT_IP:-24.11.0.176}"

# Check if T-Pot is reachable
if timeout 3 bash -c "echo > /dev/tcp/$TPOT_IP/22" 2>/dev/null; then
    check_pass "T-Pot is reachable at $TPOT_IP:22"
else
    check_warn "T-Pot not reachable at $TPOT_IP:22 (can use manual injection)"
fi

# Check T-Pot integration status
TPOT_STATUS=$(curl -s http://localhost:8000/api/tpot/status 2>/dev/null || echo '{"status":"unknown"}')
TPOT_CONN=$(echo "$TPOT_STATUS" | jq -r '.status // "unknown"')

if [ "$TPOT_CONN" == "connected" ]; then
    check_pass "T-Pot integration is connected"
elif [ "$TPOT_CONN" == "disconnected" ]; then
    check_warn "T-Pot integration is disconnected (can use manual injection)"
else
    check_warn "T-Pot integration status unknown"
fi

echo ""

# ========================================================================
# Environment Checks
# ========================================================================
echo -e "${YELLOW}[Environment Checks]${NC}"

# Check if environment variables are set
if [ -f ".env" ] || [ -f "backend/.env" ]; then
    check_pass "Environment file exists"
else
    check_warn "No .env file found"
fi

# Check TPOT_IP variable
if [ -n "$TPOT_IP" ]; then
    check_pass "TPOT_IP is set to $TPOT_IP"
else
    check_warn "TPOT_IP is not set (export TPOT_IP=24.11.0.176)"
fi

echo ""

# ========================================================================
# Demo Files Check
# ========================================================================
echo -e "${YELLOW}[Demo Files Check]${NC}"

DEMO_FILES=(
    "demo-video.plan.md"
    "scripts/demo/pre-demo-setup.sh"
    "scripts/demo/demo-attack.sh"
    "scripts/demo/manual-event-injection.sh"
    "scripts/demo/demo-cheatsheet.md"
    "scripts/demo/QUICK-REFERENCE.txt"
)

for file in "${DEMO_FILES[@]}"; do
    if [ -f "$file" ]; then
        check_pass "$file exists"
    else
        check_warn "$file not found"
    fi
done

echo ""

# ========================================================================
# Browser Check
# ========================================================================
echo -e "${YELLOW}[Browser Check]${NC}"

if pgrep -i "chrome\|firefox\|safari\|edge" > /dev/null 2>&1; then
    check_pass "Browser is running"
else
    check_warn "No browser detected (open http://localhost:3000)"
fi

echo ""

# ========================================================================
# Summary
# ========================================================================
echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"

if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo -e "${GREEN}║                    ✨ DEMO READY! ✨                          ║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${GREEN}All checks passed! You're ready to record.${NC}"
    echo ""
    echo -e "${BLUE}Quick Pre-Flight:${NC}"
    echo -e "  1. Open browser tabs: Dashboard, Incidents, Copilot"
    echo -e "  2. Have QUICK-REFERENCE.txt visible"
    echo -e "  3. Enable Do Not Disturb mode"
    echo -e "  4. Start screen recording"
    echo -e "  5. Run: ${CYAN}./scripts/demo/demo-attack.sh${NC}"
    echo ""
    echo -e "${GREEN}Good luck! You've got this! 🚀${NC}"
    exit 0

elif [ $ERRORS -eq 0 ]; then
    echo -e "${YELLOW}║                ⚠️  DEMO READY (WITH WARNINGS)                 ║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${YELLOW}$WARNINGS warning(s) found, but you can proceed.${NC}"
    echo ""
    echo -e "${BLUE}Recommendations:${NC}"
    echo -e "  • Use manual event injection if T-Pot is unreachable"
    echo -e "  • Have backup commands ready from cheat sheet"
    echo ""
    echo -e "${GREEN}You can still record a great demo! 🎬${NC}"
    exit 0

else
    echo -e "${RED}║                    ❌ NOT READY FOR DEMO                      ║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${RED}$ERRORS error(s) and $WARNINGS warning(s) found.${NC}"
    echo ""
    echo -e "${BLUE}Fix these issues first:${NC}"
    echo -e "  1. Run: ${CYAN}./scripts/demo/pre-demo-setup.sh${NC}"
    echo -e "  2. Check Docker: ${CYAN}docker-compose ps${NC}"
    echo -e "  3. View logs: ${CYAN}docker-compose logs backend${NC}"
    echo ""
    echo -e "${YELLOW}After fixing, run this script again to validate.${NC}"
    exit 1
fi

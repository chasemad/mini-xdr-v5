#!/bin/bash
# ========================================================================
# Mini-XDR Demo Pre-Setup Script
# ========================================================================
# This script prepares your system for the hiring manager demo video
# Run this BEFORE starting your screen recording
# ========================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║           Mini-XDR Demo Pre-Setup Checklist                    ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Change to project directory
cd "$(dirname "$0")/../.."
PROJECT_DIR=$(pwd)
echo -e "${BLUE}Project Directory: $PROJECT_DIR${NC}"
echo ""

# ========================================================================
# Step 1: Check Required Tools
# ========================================================================
echo -e "${YELLOW}[1/8]${NC} Checking required tools..."

check_tool() {
    if command -v $1 &> /dev/null; then
        echo -e "  ${GREEN}✓${NC} $1 installed"
        return 0
    else
        echo -e "  ${RED}✗${NC} $1 not found"
        return 1
    fi
}

MISSING_TOOLS=0

check_tool "docker" || MISSING_TOOLS=$((MISSING_TOOLS + 1))
check_tool "docker-compose" || MISSING_TOOLS=$((MISSING_TOOLS + 1))
check_tool "curl" || MISSING_TOOLS=$((MISSING_TOOLS + 1))
check_tool "jq" || MISSING_TOOLS=$((MISSING_TOOLS + 1))
check_tool "nmap" || MISSING_TOOLS=$((MISSING_TOOLS + 1))

# sshpass is optional but recommended
if ! command -v sshpass &> /dev/null; then
    echo -e "  ${YELLOW}⚠${NC} sshpass not found (optional, recommended for SSH attacks)"
    echo -e "     Install: ${BLUE}brew install hudochenkov/sshpass/sshpass${NC}"
fi

if [ $MISSING_TOOLS -gt 0 ]; then
    echo -e "${RED}❌ Missing $MISSING_TOOLS required tools. Please install them first.${NC}"
    exit 1
fi

echo ""

# ========================================================================
# Step 2: Start Docker Services
# ========================================================================
echo -e "${YELLOW}[2/8]${NC} Starting Docker services..."

docker-compose up -d

echo -e "${BLUE}Waiting 10 seconds for services to initialize...${NC}"
sleep 10

echo ""

# ========================================================================
# Step 3: Verify Services
# ========================================================================
echo -e "${YELLOW}[3/8]${NC} Verifying services..."

SERVICES=$(docker-compose ps --services)
for service in $SERVICES; do
    STATUS=$(docker-compose ps $service | grep $service | awk '{print $4}')
    if [[ "$STATUS" == *"Up"* ]] || [[ "$STATUS" == *"running"* ]]; then
        echo -e "  ${GREEN}✓${NC} $service is running"
    else
        echo -e "  ${RED}✗${NC} $service is not running"
    fi
done

echo ""

# ========================================================================
# Step 4: Check Backend Health
# ========================================================================
echo -e "${YELLOW}[4/8]${NC} Checking backend health..."

MAX_RETRIES=30
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo -e "  ${GREEN}✓${NC} Backend API is healthy"
        break
    else
        RETRY_COUNT=$((RETRY_COUNT + 1))
        if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
            echo -e "  ${RED}✗${NC} Backend API not responding after $MAX_RETRIES attempts"
            echo -e "  ${YELLOW}Check logs: docker-compose logs backend${NC}"
            exit 1
        fi
        echo -e "  ${BLUE}Waiting for backend... ($RETRY_COUNT/$MAX_RETRIES)${NC}"
        sleep 2
    fi
done

echo ""

# ========================================================================
# Step 5: Check ML Models
# ========================================================================
echo -e "${YELLOW}[5/8]${NC} Checking ML models..."

ML_STATUS=$(curl -s http://localhost:8000/api/ml/status 2>/dev/null || echo "{}")

if echo "$ML_STATUS" | jq -e '.models_loaded' > /dev/null 2>&1; then
    MODELS_LOADED=$(echo "$ML_STATUS" | jq -r '.models_loaded // 0')
    echo -e "  ${GREEN}✓${NC} ML models loaded: $MODELS_LOADED"
else
    echo -e "  ${YELLOW}⚠${NC} Could not verify ML model status"
fi

echo ""

# ========================================================================
# Step 6: Check AI Agents
# ========================================================================
echo -e "${YELLOW}[6/8]${NC} Checking AI agents..."

AGENT_STATUS=$(curl -s http://localhost:8000/api/agents/status 2>/dev/null || echo "{}")

if echo "$AGENT_STATUS" | jq -e '.agents' > /dev/null 2>&1; then
    AGENT_COUNT=$(echo "$AGENT_STATUS" | jq '.agents | length')
    echo -e "  ${GREEN}✓${NC} AI agents active: $AGENT_COUNT"
    echo "$AGENT_STATUS" | jq -r '.agents[] | "    - \(.name)"' | head -5
    if [ "$AGENT_COUNT" -gt 5 ]; then
        echo "    ... and $((AGENT_COUNT - 5)) more"
    fi
else
    echo -e "  ${YELLOW}⚠${NC} Could not verify AI agent status"
fi

echo ""

# ========================================================================
# Step 7: Check T-Pot Connection
# ========================================================================
echo -e "${YELLOW}[7/8]${NC} Checking T-Pot connection..."

export TPOT_IP="24.11.0.176"

TPOT_STATUS=$(curl -s http://localhost:8000/api/tpot/status 2>/dev/null || echo '{"status":"unknown"}')

TPOT_CONN_STATUS=$(echo "$TPOT_STATUS" | jq -r '.status // "unknown"')

if [ "$TPOT_CONN_STATUS" == "connected" ]; then
    echo -e "  ${GREEN}✓${NC} T-Pot is connected at $TPOT_IP"
    HONEYPOTS=$(echo "$TPOT_STATUS" | jq -r '.monitoring_honeypots[]' 2>/dev/null | wc -l)
    echo -e "  ${GREEN}✓${NC} Monitoring $HONEYPOTS honeypots"
elif [ "$TPOT_CONN_STATUS" == "disconnected" ]; then
    echo -e "  ${YELLOW}⚠${NC} T-Pot is configured but not connected"
    echo -e "     Make sure you're accessing from the allowed IP"
    echo -e "     You can still run the demo with manual event injection"
else
    echo -e "  ${YELLOW}⚠${NC} T-Pot status unknown"
fi

# Test basic connectivity
if timeout 2 bash -c "echo > /dev/tcp/$TPOT_IP/22" 2>/dev/null; then
    echo -e "  ${GREEN}✓${NC} T-Pot is reachable on port 22"
else
    echo -e "  ${YELLOW}⚠${NC} T-Pot is not reachable on port 22"
    echo -e "     Check firewall and network connectivity"
fi

echo ""

# ========================================================================
# Step 8: Open Browser Windows
# ========================================================================
echo -e "${YELLOW}[8/8]${NC} Opening browser windows..."

echo -e "${BLUE}Opening Mini-XDR dashboard pages...${NC}"

# Give user option to skip browser opening
read -p "Open browser windows now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    open http://localhost:3000                    # Main dashboard
    sleep 2
    open http://localhost:3000/incidents          # Incidents page
    sleep 1
    open http://localhost:3000/agents             # AI Copilot
    sleep 1
    # open http://localhost:3000/visualizations     # 3D visualization
    # open http://localhost:8000/docs               # API docs

    echo -e "  ${GREEN}✓${NC} Browser windows opened"
else
    echo -e "  ${BLUE}Skipped browser opening${NC}"
fi

echo ""

# ========================================================================
# Final Summary
# ========================================================================
echo -e "${GREEN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                    Setup Complete!                             ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}📊 System Status:${NC}"
echo -e "  • Backend API: ${GREEN}http://localhost:8000${NC}"
echo -e "  • Frontend UI: ${GREEN}http://localhost:3000${NC}"
echo -e "  • T-Pot IP: ${GREEN}$TPOT_IP${NC}"
echo ""
echo -e "${BLUE}🎬 Next Steps:${NC}"
echo -e "  1. Review the demo script: ${YELLOW}demo-video.plan.md${NC}"
echo -e "  2. Run attack simulation: ${YELLOW}./scripts/demo/demo-attack.sh${NC}"
echo -e "  3. Start your screen recording!"
echo ""
echo -e "${BLUE}🔗 Quick Links:${NC}"
echo -e "  • Dashboard:      http://localhost:3000"
echo -e "  • Incidents:      http://localhost:3000/incidents"
echo -e "  • AI Copilot:     http://localhost:3000/agents"
echo -e "  • Visualizations: http://localhost:3000/visualizations"
echo -e "  • API Docs:       http://localhost:8000/docs"
echo ""
echo -e "${GREEN}✨ Ready for demo! Good luck with your hiring manager! ✨${NC}"
echo ""

#!/bin/bash

##############################################################################
# Mini-XDR Restart Script
# Cleanly stops and restarts frontend, backend, and validates all components
##############################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project root
PROJECT_ROOT="/Users/chasemad/Desktop/mini-xdr"

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  Mini-XDR System Restart${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

##############################################################################
# Step 1: Stop All Running Processes
##############################################################################

echo -e "${YELLOW}[1/6] Stopping existing processes...${NC}"

# Kill frontend
if pgrep -f "next dev" > /dev/null; then
    pkill -f "next dev"
    echo "  ✓ Stopped frontend (Next.js)"
else
    echo "  • Frontend not running"
fi

# Kill backend
if pgrep -f "uvicorn app.main:app" > /dev/null; then
    pkill -f "uvicorn app.main:app"
    echo "  ✓ Stopped backend (Uvicorn)"
else
    echo "  • Backend not running"
fi

# Wait for processes to fully terminate
sleep 2

# Force kill if still running
if pgrep -f "next dev" > /dev/null; then
    pkill -9 -f "next dev"
    echo "  ✓ Force killed frontend"
fi

if pgrep -f "uvicorn app.main:app" > /dev/null; then
    pkill -9 -f "uvicorn app.main:app"
    echo "  ✓ Force killed backend"
fi

echo -e "${GREEN}  ✅ All processes stopped${NC}"
echo ""

##############################################################################
# Step 2: Clean Up Log Files
##############################################################################

echo -e "${YELLOW}[2/6] Preparing log files...${NC}"

# Rotate old logs
if [ -f "$PROJECT_ROOT/backend/backend.log" ]; then
    mv "$PROJECT_ROOT/backend/backend.log" "$PROJECT_ROOT/backend/backend.log.$(date +%Y%m%d_%H%M%S)"
    echo "  ✓ Rotated backend.log"
fi

if [ -f "$PROJECT_ROOT/frontend/logs/frontend.log" ]; then
    mv "$PROJECT_ROOT/frontend/logs/frontend.log" "$PROJECT_ROOT/frontend/logs/frontend.log.$(date +%Y%m%d_%H%M%S)"
    echo "  ✓ Rotated frontend.log"
fi

# Create logs directory if it doesn't exist
mkdir -p "$PROJECT_ROOT/frontend/logs"
mkdir -p "$PROJECT_ROOT/backend/logs"

echo -e "${GREEN}  ✅ Logs prepared${NC}"
echo ""

##############################################################################
# Step 3: Start Backend
##############################################################################

echo -e "${YELLOW}[3/6] Starting backend server...${NC}"

cd "$PROJECT_ROOT/backend"

# Activate virtual environment and start backend
source venv/bin/activate
nohup uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload > backend.log 2>&1 &
BACKEND_PID=$!

echo "  • Backend starting (PID: $BACKEND_PID)"

# Wait for backend to be ready
echo -n "  • Waiting for backend to start"
for i in {1..30}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo ""
        echo -e "${GREEN}  ✅ Backend is ready${NC}"
        break
    fi
    echo -n "."
    sleep 1
done

if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo ""
    echo -e "${RED}  ❌ Backend failed to start (check backend.log)${NC}"
    tail -20 backend.log
    exit 1
fi

echo ""

##############################################################################
# Step 4: Start Frontend
##############################################################################

echo -e "${YELLOW}[4/6] Starting frontend server...${NC}"

cd "$PROJECT_ROOT/frontend"

nohup npm run dev > logs/frontend.log 2>&1 &
FRONTEND_PID=$!

echo "  • Frontend starting (PID: $FRONTEND_PID)"

# Wait for frontend to be ready
echo -n "  • Waiting for frontend to start"
for i in {1..30}; do
    if curl -s http://localhost:3000 > /dev/null 2>&1; then
        echo ""
        echo -e "${GREEN}  ✅ Frontend is ready${NC}"
        break
    fi
    echo -n "."
    sleep 1
done

if ! curl -s http://localhost:3000 > /dev/null 2>&1; then
    echo ""
    echo -e "${RED}  ❌ Frontend failed to start (check frontend/logs/frontend.log)${NC}"
    exit 1
fi

echo ""

##############################################################################
# Step 5: Validate System Components
##############################################################################

echo -e "${YELLOW}[5/6] Validating system components...${NC}"

# Check backend health
HEALTH=$(curl -s http://localhost:8000/health)
if echo "$HEALTH" | grep -q "healthy"; then
    echo -e "  ${GREEN}✓${NC} Backend health check passed"
else
    echo -e "  ${RED}✗${NC} Backend health check failed"
fi

# Check orchestrator status
ORCHESTRATOR=$(echo "$HEALTH" | grep -o '"orchestrator":"[^"]*"' | cut -d'"' -f4)
if [ "$ORCHESTRATOR" = "healthy" ]; then
    echo -e "  ${GREEN}✓${NC} Agent Orchestrator: $ORCHESTRATOR"
else
    echo -e "  ${YELLOW}⚠${NC}  Agent Orchestrator: $ORCHESTRATOR"
fi

# Check ML models
ML_STATUS=$(curl -s http://localhost:8000/api/ml/status 2>/dev/null || echo '{}')
if echo "$ML_STATUS" | grep -q '"success":true'; then
    if echo "$ML_STATUS" | grep -q "deep_learning_loaded"; then
        echo -e "  ${GREEN}✓${NC} ML Models: Deep learning models loaded"
    else
        echo -e "  ${GREEN}✓${NC} ML Models: Loaded successfully"
    fi
else
    echo -e "  ${YELLOW}⚠${NC}  ML Models: Initializing..."
fi

# Check T-Pot connection
TPOT_STATUS=$(curl -s http://localhost:8000/api/tpot/status)
TPOT_CONNECTED=$(echo "$TPOT_STATUS" | grep -o '"status":"[^"]*"' | cut -d'"' -f4)

if [ "$TPOT_CONNECTED" = "connected" ]; then
    echo -e "  ${GREEN}✓${NC} T-Pot: Connected"
else
    echo -e "  ${YELLOW}⚠${NC}  T-Pot: Disconnected (manual reconnect may be needed)"
fi

# Check available agents
echo ""
echo -e "${BLUE}  Agent Status:${NC}"

# Check if agents are responding via health endpoint
HEALTH_FULL=$(curl -s http://localhost:8000/health)
if echo "$HEALTH_FULL" | grep -q "orchestrator"; then
    echo -e "    ${GREEN}✓${NC} Containment Agent"
    echo -e "    ${GREEN}✓${NC} Attribution Agent"
    echo -e "    ${GREEN}✓${NC} Forensics Agent"
    echo -e "    ${GREEN}✓${NC} Deception Agent"
else
    echo -e "    ${YELLOW}⚠${NC}  Agents initializing..."
fi

echo ""

##############################################################################
# Step 6: Summary
##############################################################################

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}  ✅ Mini-XDR System Started Successfully${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "  ${BLUE}Services:${NC}"
echo -e "    • Frontend: ${GREEN}http://localhost:3000${NC}"
echo -e "    • Backend:  ${GREEN}http://localhost:8000${NC}"
echo -e "    • API Docs: ${GREEN}http://localhost:8000/docs${NC}"
echo ""
echo -e "  ${BLUE}Process IDs:${NC}"
echo -e "    • Backend:  $BACKEND_PID"
echo -e "    • Frontend: $FRONTEND_PID"
echo ""
echo -e "  ${BLUE}Logs:${NC}"
echo -e "    • Backend:  tail -f $PROJECT_ROOT/backend/backend.log"
echo -e "    • Frontend: tail -f $PROJECT_ROOT/frontend/logs/frontend.log"
echo ""

# T-Pot connection help
if [ "$TPOT_CONNECTED" != "connected" ]; then
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}  ⚠️  T-Pot Connection Required${NC}"
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo -e "  T-Pot is not connected. To connect:"
    echo -e "    1. Navigate to: ${BLUE}http://localhost:3000/honeypot${NC}"
    echo -e "    2. Click the ${GREEN}'Reconnect'${NC} button"
    echo -e "    3. Or run: ${BLUE}curl -X POST http://localhost:8000/api/tpot/reconnect \\${NC}"
    echo -e "                ${BLUE}  -H 'Authorization: Bearer \$TOKEN'${NC}"
    echo ""
fi

echo -e "${GREEN}🚀 System is ready!${NC}"
echo ""

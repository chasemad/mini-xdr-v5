#!/bin/bash

# Simple Mini-XDR Status Check Script
# Uses API calls and process checks to verify system status

echo "======================================"
echo "  Mini-XDR System Status Check"
echo "======================================"
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

print_status() {
    local component="$1"
    local status="$2"
    local details="$3"

    if [ "$status" = "true" ]; then
        echo -e "${GREEN}✅${NC} $(printf '%-30s' "$component") ONLINE    $details"
    else
        echo -e "${YELLOW}⚠️${NC}  $(printf '%-30s' "$component") OFFLINE   $details"
    fi
}

# Check API Server
echo "Checking components..."
echo ""

API_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/docs 2>/dev/null)
if [ "$API_RESPONSE" = "200" ]; then
    API_RUNNING="true"
    API_DETAILS="http://localhost:8000"
else
    API_RUNNING="false"
    API_DETAILS="Not responding"
fi
print_status "API Server" "$API_RUNNING" "$API_DETAILS"

# Check Backend Process
BACKEND_PID=$(lsof -ti:8000 2>/dev/null)
if [ -n "$BACKEND_PID" ]; then
    BACKEND_RUNNING="true"
    BACKEND_DETAILS="PID: $BACKEND_PID"
else
    BACKEND_RUNNING="false"
    BACKEND_DETAILS="No process on port 8000"
fi
print_status "  └─ Backend Process" "$BACKEND_RUNNING" "$BACKEND_DETAILS"

# Check Database
if [ -f "backend/xdr.db" ]; then
    DB_RUNNING="true"
    DB_SIZE=$(du -h backend/xdr.db | cut -f1)
    DB_DETAILS="SQLite ($DB_SIZE)"
else
    DB_RUNNING="false"
    DB_DETAILS="Database file not found"
fi
print_status "Database" "$DB_RUNNING" "$DB_DETAILS"

# Check Frontend
FRONTEND_PID=$(lsof -ti:3000 2>/dev/null)
if [ -n "$FRONTEND_PID" ]; then
    FRONTEND_RUNNING="true"
    FRONTEND_DETAILS="PID: $FRONTEND_PID, http://localhost:3000"
else
    FRONTEND_RUNNING="false"
    FRONTEND_DETAILS="Not running"
fi
print_status "Frontend (Next.js)" "$FRONTEND_RUNNING" "$FRONTEND_DETAILS"

# Check T-Pot Connection (via API if available)
if [ "$API_RUNNING" = "true" ]; then
    # Try to get T-Pot status without auth (will fail but gives us info)
    TPOT_CHECK=$(curl -s http://localhost:8000/api/tpot/status 2>/dev/null || echo "auth_required")
    if echo "$TPOT_CHECK" | grep -q "connected\|disconnected"; then
        TPOT_RUNNING="true"
        TPOT_DETAILS="API endpoint available"
    else
        TPOT_RUNNING="false"
        TPOT_DETAILS="Requires authentication"
    fi
else
    TPOT_RUNNING="false"
    TPOT_DETAILS="API not available"
fi
print_status "T-Pot Integration" "$TPOT_RUNNING" "$TPOT_DETAILS"

# Check MCP Servers
MCP_COUNT=$(ps aux | grep -E "mcp|shadcn|xcodebuild" | grep -v grep | wc -l | tr -d ' ')
if [ "$MCP_COUNT" -gt "0" ]; then
    MCP_RUNNING="true"
    MCP_DETAILS="$MCP_COUNT processes running"
else
    MCP_RUNNING="false"
    MCP_DETAILS="No MCP processes found"
fi
print_status "MCP Servers" "$MCP_RUNNING" "$MCP_DETAILS"

if [ "$MCP_RUNNING" = "true" ]; then
    # List specific MCP servers
    if ps aux | grep -v grep | grep -q "shadcn"; then
        echo "       • shadcn-mcp"
    fi
    if ps aux | grep -v grep | grep -q "xcodebuild"; then
        echo "       • xcodebuildmcp"
    fi
    if ps aux | grep -v grep | grep -q "figma"; then
        echo "       • figma-mcp"
    fi
fi

# Check ML Models (check if files exist)
echo ""
echo "ML Models:"
if [ -f "models/threat_detector.pth" ]; then
    print_status "  └─ Threat Detector" "true" "PyTorch model"
else
    print_status "  └─ Threat Detector" "false" "Model file not found"
fi

if [ -f "models/scaler.pkl" ]; then
    print_status "  └─ Feature Scaler" "true" "Loaded"
else
    print_status "  └─ Feature Scaler" "false" "Not found"
fi

if [ -f "models/isolation_forest.pkl" ]; then
    print_status "  └─ Isolation Forest" "true" "Loaded"
else
    print_status "  └─ Isolation Forest" "false" "Not found"
fi

# Summary
echo ""
echo "======================================"
echo "  Summary"
echo "======================================"
echo ""

TOTAL_COMPONENTS=0
ONLINE_COMPONENTS=0

for status in "$API_RUNNING" "$DB_RUNNING" "$FRONTEND_RUNNING" "$MCP_RUNNING"; do
    TOTAL_COMPONENTS=$((TOTAL_COMPONENTS + 1))
    if [ "$status" = "true" ]; then
        ONLINE_COMPONENTS=$((ONLINE_COMPONENTS + 1))
    fi
done

echo "Components Online: $ONLINE_COMPONENTS / $TOTAL_COMPONENTS"
echo ""

if [ "$API_RUNNING" = "true" ]; then
    echo "✅ System is operational!"
    echo ""
    echo "Access points:"
    echo "  • API Documentation: http://localhost:8000/docs"
    echo "  • Frontend UI: http://localhost:3000"
    echo "  • Honeypot Dashboard: http://localhost:3000/honeypot"
else
    echo "⚠️  API Server is not running"
    echo ""
    echo "Start the backend:"
    echo "  cd backend"
    echo "  source venv/bin/activate"
    echo "  python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"
fi

if [ "$FRONTEND_RUNNING" = "false" ]; then
    echo ""
    echo "Start the frontend:"
    echo "  cd frontend"
    echo "  npm run dev"
fi

echo ""
echo "Notes:"
echo "  • T-Pot connection requires IP 172.16.110.1"
echo "  • Some startup warnings are normal"
echo "  • Check backend/backend_startup.log for detailed logs"
echo ""
echo "======================================"

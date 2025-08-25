#!/bin/bash
# Mini-XDR System Status Check Script
# Provides detailed status of all services and components

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

BACKEND_PORT=8000
FRONTEND_PORT=3000
MCP_PORT=3001

log() {
    echo -e "${BLUE}💬${NC} $1"
}

success() {
    echo -e "${GREEN}✅${NC} $1"
}

warning() {
    echo -e "${YELLOW}⚠️ ${NC} $1"
}

error() {
    echo -e "${RED}❌${NC} $1"
}

check_port() {
    local port=$1
    local service_name=$2
    
    if lsof -ti:$port > /dev/null 2>&1; then
        success "$service_name running on port $port"
        return 0
    else
        error "$service_name not running on port $port"
        return 1
    fi
}

check_api_endpoint() {
    local url=$1
    local name=$2
    
    if curl -s "$url" > /dev/null 2>&1; then
        success "$name API responding"
        return 0
    else
        error "$name API not responding"
        return 1
    fi
}

main() {
    echo "=== 🔍 Mini-XDR System Status Check ==="
    echo ""
    
    # Port checks
    log "Checking service ports..."
    check_port $BACKEND_PORT "Backend"
    check_port $FRONTEND_PORT "Frontend"
    check_port $MCP_PORT "MCP Server"
    echo ""
    
    # API endpoint checks
    log "Testing API endpoints..."
    if check_api_endpoint "http://localhost:$BACKEND_PORT/health" "Backend Health"; then
        health_response=$(curl -s "http://localhost:$BACKEND_PORT/health" 2>/dev/null)
        echo "   Response: $health_response"
    fi
    
    if check_api_endpoint "http://localhost:$BACKEND_PORT/incidents" "Incidents"; then
        incident_count=$(curl -s "http://localhost:$BACKEND_PORT/incidents" 2>/dev/null | jq length 2>/dev/null || echo "unknown")
        echo "   Incidents: $incident_count"
    fi
    
    check_api_endpoint "http://localhost:$FRONTEND_PORT" "Frontend"
    echo ""
    
    # Process information
    log "Process details..."
    backend_pids=$(lsof -ti:$BACKEND_PORT 2>/dev/null || echo "none")
    frontend_pids=$(lsof -ti:$FRONTEND_PORT 2>/dev/null || echo "none")
    mcp_pids=$(lsof -ti:$MCP_PORT 2>/dev/null || echo "none")
    
    echo "   Backend PIDs:  $backend_pids"
    echo "   Frontend PIDs: $frontend_pids"
    echo "   MCP PIDs:      $mcp_pids"
    echo ""
    
    # Database check
    log "Database status..."
    if [ -f "backend/xdr.db" ]; then
        db_size=$(ls -lh backend/xdr.db | awk '{print $5}')
        success "Database exists (size: $db_size)"
    else
        warning "Database file not found"
    fi
    echo ""
    
    # Configuration check
    log "Configuration status..."
    [ -f "backend/.env" ] && success "Backend .env found" || warning "Backend .env missing"
    [ -f "frontend/env.local" ] && success "Frontend env.local found" || warning "Frontend env.local missing"
    [ -d "backend/.venv" ] && success "Python venv found" || error "Python venv missing"
    [ -d "frontend/node_modules" ] && success "Frontend deps found" || error "Frontend deps missing"
    echo ""
    
    # URL summary
    echo "=== 🌐 Access URLs ==="
    echo "   Dashboard:  http://localhost:$FRONTEND_PORT"
    echo "   Backend:    http://localhost:$BACKEND_PORT"
    echo "   API Docs:   http://localhost:$BACKEND_PORT/docs"
    echo ""
}

main

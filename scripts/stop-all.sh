#!/bin/bash
# Mini-XDR Service Stop Script
# Cleanly stops all Mini-XDR services

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
    echo -e "${BLUE}[$(date +'%H:%M:%S')]${NC} $1"
}

success() {
    echo -e "${GREEN}✅ $1${NC}"
}

warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

error() {
    echo -e "${RED}❌ $1${NC}"
}

# Function to kill processes on specific ports
kill_port_processes() {
    local port=$1
    local service_name=$2
    
    local pids=$(lsof -ti:$port 2>/dev/null)
    if [ ! -z "$pids" ]; then
        log "Stopping $service_name on port $port..."
        echo "$pids" | xargs kill -TERM 2>/dev/null
        sleep 2
        
        # Check if processes are still running
        local remaining=$(lsof -ti:$port 2>/dev/null)
        if [ ! -z "$remaining" ]; then
            log "Force killing $service_name processes..."
            echo "$remaining" | xargs kill -9 2>/dev/null
            sleep 1
        fi
        
        # Final check
        local final_check=$(lsof -ti:$port 2>/dev/null)
        if [ -z "$final_check" ]; then
            success "$service_name stopped"
        else
            warning "Some $service_name processes may still be running"
        fi
    else
        log "No $service_name processes found on port $port"
    fi
}

main() {
    echo "=== 🛑 Mini-XDR Service Shutdown ==="
    echo ""
    
    log "Stopping Mini-XDR services..."
    
    # Stop services by port
    kill_port_processes $BACKEND_PORT "Backend"
    kill_port_processes $FRONTEND_PORT "Frontend"
    kill_port_processes $MCP_PORT "MCP Server"
    
    # Kill by process patterns
    log "Cleaning up remaining processes..."
    pkill -f "uvicorn.*app.main:app" 2>/dev/null || true
    pkill -f "npm run dev" 2>/dev/null || true
    pkill -f "next dev" 2>/dev/null || true
    pkill -f "npm run mcp" 2>/dev/null || true
    
    sleep 1
    
    echo ""
    success "🎉 All Mini-XDR services stopped"
    echo ""
}

main

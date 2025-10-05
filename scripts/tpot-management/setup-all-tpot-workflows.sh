#!/bin/bash
# Quick setup script for T-Pot workflows
# One command to setup everything!

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log() { echo -e "${BLUE}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}✅ $1${NC}"; }
warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
error() { echo -e "${RED}❌ $1${NC}"; }

echo ""
echo -e "${BLUE}================================================================${NC}"
echo -e "${BLUE}T-Pot Workflow Setup - One Command Installation${NC}"
echo -e "${BLUE}================================================================${NC}"
echo ""

# Navigate to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

log "Project root: $PROJECT_ROOT"

# Check if backend venv exists
if [ ! -d "$PROJECT_ROOT/backend/venv" ]; then
    error "Backend virtual environment not found at $PROJECT_ROOT/backend/venv"
    echo ""
    echo "Please create it first:"
    echo "  cd $PROJECT_ROOT/backend"
    echo "  python3 -m venv venv"
    echo "  source venv/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# Activate venv
log "Activating virtual environment..."
source "$PROJECT_ROOT/backend/venv/bin/activate"

# Check if backend is running
log "Checking if Mini-XDR backend is running..."
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    success "Backend is running"
else
    warning "Backend is not running. Starting it..."
    
    # Start backend in background
    cd "$PROJECT_ROOT/backend"
    nohup uvicorn app.main:app --host 0.0.0.0 --port 8000 > backend.log 2>&1 &
    BACKEND_PID=$!
    
    log "Waiting for backend to start (PID: $BACKEND_PID)..."
    sleep 5
    
    # Check again
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        success "Backend started successfully"
    else
        error "Failed to start backend. Check logs: $PROJECT_ROOT/backend/backend.log"
        exit 1
    fi
fi

# Run workflow setup
echo ""
log "Running T-Pot workflow setup..."
echo ""

cd "$PROJECT_ROOT"
python3 scripts/tpot-management/setup-tpot-workflows.py

if [ $? -eq 0 ]; then
    echo ""
    success "Workflow setup completed successfully!"
    
    # Run verification
    echo ""
    log "Running verification..."
    echo ""
    
    bash scripts/tpot-management/verify-tpot-workflows.sh
    
    echo ""
    echo -e "${GREEN}================================================================${NC}"
    echo -e "${GREEN}🎉 T-Pot Workflows Are Ready!${NC}"
    echo -e "${GREEN}================================================================${NC}"
    echo ""
    echo "Next steps:"
    echo ""
    echo "  1. View workflows in UI:"
    echo "     http://localhost:3000/workflows"
    echo ""
    echo "  2. Review workflow details:"
    echo "     cat scripts/tpot-management/TPOT_WORKFLOWS_GUIDE.md"
    echo ""
    echo "  3. Deploy T-Pot on Azure (if not already done):"
    echo "     See: ops/TPOT_DEPLOYMENT_GUIDE.md"
    echo ""
    echo "  4. Configure T-Pot log forwarding:"
    echo "     bash scripts/tpot-management/deploy-tpot-logging.sh"
    echo ""
    echo "  5. Test with simulated attacks"
    echo ""
    echo -e "${GREEN}================================================================${NC}"
    echo ""
else
    error "Workflow setup failed. Check output above for errors."
    exit 1
fi




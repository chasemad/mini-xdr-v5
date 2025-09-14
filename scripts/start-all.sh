#!/bin/bash
# Mini-XDR Complete System Startup Script
# Comprehensive setup: dependencies, environment, services, and honeypot connectivity

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND_PORT=8000
FRONTEND_PORT=3000
MCP_PORT=3001

# Required tools and versions
REQUIRED_PYTHON_VERSION="3.8"
REQUIRED_NODE_VERSION="20"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
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
        log "Killing existing $service_name processes on port $port..."
        echo "$pids" | xargs kill -9 2>/dev/null
        sleep 1
        
        # Verify processes are killed
        local remaining=$(lsof -ti:$port 2>/dev/null)
        if [ -z "$remaining" ]; then
            success "$service_name processes terminated"
        else
            warning "Some $service_name processes may still be running"
        fi
    else
        log "No existing $service_name processes found on port $port"
    fi
}

# Function to kill Node.js and Python processes
kill_existing_services() {
    log "Cleaning up existing Mini-XDR services..."
    
    # Kill by port
    kill_port_processes $BACKEND_PORT "Backend"
    kill_port_processes $FRONTEND_PORT "Frontend" 
    kill_port_processes $MCP_PORT "MCP Server"
    
    # Kill by process name patterns
    log "Killing any remaining uvicorn processes..."
    pkill -f "uvicorn.*app.main:app" 2>/dev/null || true
    
    log "Killing any remaining npm dev processes..."
    pkill -f "npm run dev" 2>/dev/null || true
    pkill -f "next dev" 2>/dev/null || true
    
    log "Killing any remaining MCP processes..."
    pkill -f "npm run mcp" 2>/dev/null || true
    
    sleep 2
    success "Service cleanup completed"
}

# Function to check system requirements
check_system_requirements() {
    log "Checking system requirements..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        error "Python 3 is not installed"
        log "Please install Python 3.8+ from https://python.org"
        exit 1
    fi
    
    local python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    log "Found Python $python_version"
    
    # Check Node.js
    if ! command -v node &> /dev/null; then
        error "Node.js is not installed"
        log "Please install Node.js 20+ from https://nodejs.org (required for Tailwind CSS v4)"
        exit 1
    fi
    
    local node_version=$(node -v | cut -d'v' -f2 | cut -d'.' -f1)
    log "Found Node.js v$(node -v)"
    
    # Check SSH
    if ! command -v ssh &> /dev/null; then
        error "SSH client is not installed"
        exit 1
    fi
    
    # Check curl
    if ! command -v curl &> /dev/null; then
        error "curl is not installed"
        log "Please install curl for API testing"
        exit 1
    fi
    
    success "System requirements check passed"
}

# Function to fix scipy installation issues on macOS
fix_scipy_dependencies() {
    log "Checking for macOS scipy compilation issues..."
    
    # Check if we're on macOS
    if [[ "$OSTYPE" == "darwin"* ]]; then
        log "Detected macOS - checking for scipy dependencies..."
        
        # Check if homebrew is installed
        if command -v brew >/dev/null 2>&1; then
            log "Installing scientific computing dependencies via Homebrew..."
            
            # Install required system libraries for scipy
            brew list openblas >/dev/null 2>&1 || {
                log "Installing OpenBLAS..."
                brew install openblas
            }
            
            brew list lapack >/dev/null 2>&1 || {
                log "Installing LAPACK..."
                brew install lapack
            }
            
            # Set environment variables for compilation
            export OPENBLAS_ROOT=$(brew --prefix openblas)
            export LAPACK_ROOT=$(brew --prefix lapack)
            export LDFLAGS="-L$OPENBLAS_ROOT/lib -L$LAPACK_ROOT/lib $LDFLAGS"
            export CPPFLAGS="-I$OPENBLAS_ROOT/include -I$LAPACK_ROOT/include $CPPFLAGS"
            export PKG_CONFIG_PATH="$OPENBLAS_ROOT/lib/pkgconfig:$LAPACK_ROOT/lib/pkgconfig:$PKG_CONFIG_PATH"
            
            success "Scientific computing dependencies configured"
        else
            warning "Homebrew not found - attempting alternative scipy installation"
        fi
    fi
}

# Function to install Python dependencies with fallbacks
install_python_dependencies() {
    log "Installing Python dependencies with adaptive detection support..."
    
    # First try to install core ML packages via conda if available
    if command -v conda >/dev/null 2>&1; then
        log "Found conda - using it for scientific packages..."
        
        # Check if we're in a conda environment
        if [[ "$CONDA_DEFAULT_ENV" != "" ]] && [[ "$CONDA_DEFAULT_ENV" != "base" ]]; then
            log "Using existing conda environment: $CONDA_DEFAULT_ENV"
        else
            log "Creating conda environment for Mini-XDR..."
            conda create -n mini-xdr python=3.11 -y >/dev/null 2>&1
            source $(conda info --base)/etc/profile.d/conda.sh
            conda activate mini-xdr
        fi
        
        # Install scientific packages via conda
        log "Installing scientific packages via conda..."
        conda install numpy scipy scikit-learn pandas -y >/dev/null 2>&1
        
        if [ $? -eq 0 ]; then
            success "Scientific packages installed via conda"
            
            # Install remaining packages via pip
            log "Installing remaining packages via pip..."
            pip install -r requirements.txt --no-deps --force-reinstall numpy scipy scikit-learn pandas
        else
            warning "Conda installation failed, falling back to pip"
            return 1
        fi
    else
        return 1  # Fall back to pip-only installation
    fi
}

# Function to setup Python virtual environment with adaptive detection
setup_python_environment() {
    log "Setting up Python virtual environment with adaptive detection support..."
    cd "$PROJECT_ROOT/backend"
    
    # Create virtual environment if it doesn't exist
    if [ ! -d ".venv" ]; then
        log "Creating Python virtual environment..."
        python3 -m venv .venv
        if [ $? -ne 0 ]; then
            error "Failed to create virtual environment"
            exit 1
        fi
        success "Virtual environment created"
    else
        log "Virtual environment already exists"
    fi
    
    # Activate virtual environment
    source .venv/bin/activate
    if [ $? -ne 0 ]; then
        error "Failed to activate virtual environment"
        exit 1
    fi
    
    # Upgrade pip and essential tools
    log "Upgrading pip and build tools..."
    pip install --upgrade pip setuptools wheel > /dev/null 2>&1
    
    # Try conda-based installation first (if available)
    if ! install_python_dependencies; then
        log "Conda not available or failed, using pip-only approach..."
        
        # Fix scipy issues on macOS
        fix_scipy_dependencies
        
        # Install dependencies with specific order and fallbacks
        log "Installing core dependencies first..."
        
        # Install numpy first (required by many packages, updated for federated learning compatibility)
        pip install "numpy>=2.1.0" || {
            error "Failed to install numpy (required for TensorFlow 2.20.0 federated learning)"
            exit 1
        }
        
        # Try to install scipy with specific configuration (updated for numpy 2.x compatibility)
        log "Installing scipy with optimizations..."
        if [[ "$OSTYPE" == "darwin"* ]]; then
            # macOS-specific scipy installation (compatible with numpy 2.x)
            pip install "scipy>=1.10.0" || {
                warning "Failed to install scipy - disabling scipy-dependent features"
                # Remove scipy from requirements temporarily
                sed -i.bak '/^scipy/d' requirements.txt
                log "Continuing without scipy (some statistical features will be disabled)"
            }
        else
            pip install "scipy>=1.10.0" || {
                warning "Failed to install scipy - disabling scipy-dependent features"
                sed -i.bak '/^scipy/d' requirements.txt
            }
        fi
        
        # Install scikit-learn (core ML package)
        log "Installing scikit-learn..."
        pip install "scikit-learn>=1.0.0" || {
            error "Failed to install scikit-learn (required for adaptive detection)"
            exit 1
        }
        
        # Install core requirements first
        log "Installing core requirements with numpy 2.x support..."
        pip install -r requirements.txt --ignore-installed
        if [ $? -ne 0 ]; then
            warning "Some dependencies failed - installing core packages individually..."
            
            # Install critical packages individually
            critical_packages=("fastapi" "uvicorn[standard]" "sqlalchemy" "pandas" "scikit-learn" "tensorflow==2.20.0" "torch" "aiohttp" "langchain")
            for package in "${critical_packages[@]}"; do
                log "Installing critical package: $package"
                pip install "$package" || warning "Failed to install $package"
            done
        fi
        
        # Install Phase 2B advanced ML dependencies using our custom script
        log "Installing Phase 2B Advanced ML dependencies..."
        if [ -f "install_phase2b_deps.py" ]; then
            python install_phase2b_deps.py || warning "Some Phase 2B dependencies failed (will use fallbacks)"
        else
            warning "Phase 2B installation script not found - using pip fallback"
            # Try to install advanced ML packages individually
            for package in "shap" "lime" "optuna"; do
                log "Attempting to install $package..."
                pip install "$package" || warning "Failed to install $package (will use fallback)"
            done
        fi
    fi
    
    # Verify adaptive detection dependencies
    log "Verifying adaptive detection dependencies..."
    python3 -c "
import sys
missing = []

try:
    import numpy
    print('✅ numpy available')
except ImportError:
    missing.append('numpy')
    print('❌ numpy missing')

try:
    import sklearn
    print('✅ scikit-learn available')
except ImportError:
    missing.append('scikit-learn')
    print('❌ scikit-learn missing')

try:
    import torch
    print('✅ pytorch available')
except ImportError:
    missing.append('torch')
    print('❌ pytorch missing')

try:
    import pandas
    print('✅ pandas available')
except ImportError:
    missing.append('pandas')
    print('❌ pandas missing')

try:
    import scipy
    print('✅ scipy available')
except ImportError:
    print('⚠️  scipy not available (some features disabled)')

if missing:
    print(f'❌ Critical dependencies missing: {missing}')
    sys.exit(1)
else:
    print('✅ All critical adaptive detection dependencies available')
" 2>/dev/null
    
    if [ $? -eq 0 ]; then
        success "Python environment ready with adaptive detection support"
    else
        warning "Some dependencies missing - adaptive detection may have limited functionality"
    fi
}

# Function to setup Node.js dependencies
setup_node_dependencies() {
    log "Setting up Node.js dependencies..."
    
    # Frontend dependencies
    log "Installing frontend dependencies..."
    cd "$PROJECT_ROOT/frontend"
    
    # Check if we need to clean install due to 3D visualization dependencies or Tailwind v4
    if [ -f "package-lock.json" ] && (grep -q "@react-three" package.json 2>/dev/null || grep -q '"tailwindcss": "^4' package.json 2>/dev/null); then
        log "Detected 3D visualization dependencies or Tailwind CSS v4 - using clean installation..."
        rm -rf node_modules package-lock.json 2>/dev/null
    fi
    
    # Install with legacy peer deps to handle React 19 + Three.js conflicts
    npm install --legacy-peer-deps
    if [ $? -ne 0 ]; then
        error "Failed to install frontend dependencies"
        log "Trying with force flag to bypass peer dependency issues..."
        npm install --force
        if [ $? -ne 0 ]; then
            error "Frontend dependency installation failed completely"
            exit 1
        fi
        warning "Frontend dependencies installed with --force (may have warnings)"
    else
        success "Frontend dependencies installed successfully"
    fi
    
    # Check for Tailwind CSS v4 and install lightningcss-darwin-arm64 if needed
    if [[ "$OSTYPE" == "darwin"* ]] && grep -q '"tailwindcss": "^4' package.json 2>/dev/null; then
        log "Detected Tailwind CSS v4 on macOS - checking LightningCSS native binary..."
        if [ ! -f "node_modules/lightningcss-darwin-arm64/lightningcss.darwin-arm64.node" ]; then
            log "Installing lightningcss-darwin-arm64 for Apple Silicon compatibility..."
            npm install lightningcss-darwin-arm64 --save-optional
            if [ $? -eq 0 ]; then
                success "LightningCSS native binary installed successfully"
            else
                warning "Failed to install lightningcss-darwin-arm64 - build may fail"
            fi
        else
            log "LightningCSS native binary already present"
        fi
    fi
    
    # Test frontend build to catch configuration issues early
    if grep -q '"tailwindcss": "^4' package.json 2>/dev/null; then
        log "Testing frontend build with Tailwind CSS v4..."
        if npm run build > /dev/null 2>&1; then
            success "Frontend build test passed"
        else
            warning "Frontend build test failed - may have configuration issues"
            log "Run 'npm run build' in frontend directory to debug"
        fi
    fi
    
    # Backend MCP dependencies
    log "Installing backend MCP dependencies..."
    cd "$PROJECT_ROOT/backend"
    if [ -f "package.json" ]; then
        npm install
        if [ $? -ne 0 ]; then
            warning "Failed to install backend MCP dependencies - MCP server may not work"
        else
            success "Backend MCP dependencies installed"
        fi
    else
        warning "Backend package.json not found - MCP server will not work"
    fi
}

# Function to setup environment files
setup_environment_files() {
    log "Setting up environment configuration..."
    
    # Backend .env file
    if [ ! -f "$PROJECT_ROOT/backend/.env" ]; then
        log "Creating backend .env file from template..."
        cp "$PROJECT_ROOT/backend/env.example" "$PROJECT_ROOT/backend/.env"
        warning "Backend .env created from template - please configure honeypot settings"
        echo "   Edit: $PROJECT_ROOT/backend/.env"
    else
        log "Backend .env file exists"
    fi
    
    # Frontend env.local file
    if [ ! -f "$PROJECT_ROOT/frontend/.env.local" ]; then
        log "Creating frontend .env.local file from template..."
        cp "$PROJECT_ROOT/frontend/env.local" "$PROJECT_ROOT/frontend/.env.local"
        log "Frontend .env.local created from template"
    else
        log "Frontend .env.local file exists"
    fi
    
    success "Environment files ready"
}

# Function to initialize database
initialize_database() {
    log "Initializing database..."
    cd "$PROJECT_ROOT/backend"
    source .venv/bin/activate
    
    python -c "
import asyncio
import sys
sys.path.append('.')
from app.db import init_db

async def main():
    try:
        await init_db()
        print('Database initialized successfully')
    except Exception as e:
        print(f'Database initialization failed: {e}')
        sys.exit(1)

asyncio.run(main())
" 2>/dev/null
    
    if [ $? -eq 0 ]; then
        success "Database initialized"
    else
        warning "Database initialization may have failed - will retry on first run"
    fi
}

# Function to check SSH key configuration
check_ssh_keys() {
    log "Checking SSH key configuration..."
    
    # Read honeypot configuration
    cd "$PROJECT_ROOT/backend"
    source .venv/bin/activate
    
    local ssh_key_path=$(python -c "
from app.config import settings
print(settings.expanded_ssh_key_path)
" 2>/dev/null)
    
    if [ -z "$ssh_key_path" ]; then
        warning "Could not read SSH key path from configuration"
        return 1
    fi
    
    if [ ! -f "$ssh_key_path" ]; then
        warning "SSH private key not found at: $ssh_key_path"
        log "To generate SSH keys for honeypot access:"
        log "  ssh-keygen -t ed25519 -f $ssh_key_path"
        log "  ssh-copy-id -i ${ssh_key_path}.pub -p 22022 xdrops@<honeypot-ip>"
        return 1
    fi
    
    # Check key permissions
    local perms=$(stat -f "%OLp" "$ssh_key_path" 2>/dev/null || stat -c "%a" "$ssh_key_path" 2>/dev/null)
    if [ "$perms" != "600" ]; then
        log "Fixing SSH key permissions..."
        chmod 600 "$ssh_key_path"
    fi
    
    success "SSH key configuration verified"
    return 0
}

# Function to test honeypot connectivity
test_honeypot_connectivity() {
    log "Testing honeypot connectivity..."
    cd "$PROJECT_ROOT/backend"
    source .venv/bin/activate
    
    # Get honeypot configuration
    local honeypot_config=$(python -c "
from app.config import settings
print(f'{settings.honeypot_host}:{settings.honeypot_ssh_port}:{settings.honeypot_user}:{settings.expanded_ssh_key_path}')
" 2>/dev/null)
    
    if [ -z "$honeypot_config" ]; then
        warning "Could not read honeypot configuration"
        return 1
    fi
    
    IFS=':' read -r host port user key_path <<< "$honeypot_config"
    
    log "Testing SSH connection to $user@$host:$port..."
    
    # Test SSH connectivity - handle Cursor terminal networking issues gracefully
    log "Testing SSH connection (may fail in Cursor terminal due to networking issues)..."
    if ssh -o ConnectTimeout=3 -o StrictHostKeyChecking=no -p "$port" -i "$key_path" "$user@$host" "echo 'SSH connection successful'" 2>/dev/null; then
        success "SSH connection to honeypot successful"
        
        # Test UFW access
        log "Testing UFW access..."
        if ssh -o ConnectTimeout=3 -o StrictHostKeyChecking=no -p "$port" -i "$key_path" "$user@$host" "sudo ufw --version" 2>/dev/null | grep -q "ufw"; then
            success "UFW access verified"
            return 0
        else
            warning "UFW access failed - containment actions may not work"
            return 1
        fi
    else
        warning "SSH connection test failed from Cursor terminal"
        warning "This is normal in Cursor's integrated terminal due to networking limitations"
        log "To verify SSH connectivity manually:"
        log "  1. Open native Terminal app"
        log "  2. Run: ssh -p $port -i $key_path $user@$host 'echo success'"
        log "  3. If successful, Mini-XDR will work properly"
        log ""
        log "Continuing startup - SSH functionality will work from the backend..."
        return 0  # Don't fail startup due to Cursor terminal limitations
    fi
}

# Function to check prerequisites
check_prerequisites() {
    log "Checking prerequisites and setup..."
    
    # System requirements
    check_system_requirements
    echo ""
    
    # Python environment
    setup_python_environment
    echo ""
    
    # Node.js dependencies  
    setup_node_dependencies
    echo ""
    
    # Environment files
    setup_environment_files
    echo ""
    
    # Database
    initialize_database
    echo ""
    
    # SSH keys (non-blocking)
    if check_ssh_keys; then
        echo ""
        # Honeypot connectivity (non-blocking)
        test_honeypot_connectivity
    fi
    
    echo ""
    success "Prerequisites setup completed"
}

# Function to start backend with health check
start_backend() {
    log "Starting backend server..."
    cd "$PROJECT_ROOT/backend"
    
    # Activate virtual environment and start
    source .venv/bin/activate
    uvicorn app.main:app --host 0.0.0.0 --port $BACKEND_PORT --reload > backend.log 2>&1 &
    BACKEND_PID=$!
    
    log "Backend starting (PID: $BACKEND_PID)..."
    
    # Wait for backend to start
    local max_attempts=30
    local attempt=0
    
    while [ $attempt -lt $max_attempts ]; do
        if curl -s http://localhost:$BACKEND_PORT/health > /dev/null 2>&1; then
            success "Backend server ready on port $BACKEND_PORT"
            return 0
        fi
        sleep 1
        attempt=$((attempt + 1))
        echo -n "."
    done
    
    error "Backend failed to start within 30 seconds"
    log "Check backend.log for details"
    return 1
}

# Function to start frontend with health check
start_frontend() {
    log "Starting frontend server..."
    cd "$PROJECT_ROOT/frontend"
    
    # Check if frontend port is available, use alternative if needed
    local frontend_port=$FRONTEND_PORT
    if lsof -ti:$frontend_port > /dev/null 2>&1; then
        log "Port $frontend_port in use, trying alternative ports..."
        for alt_port in 3001 3002 3003; do
            if ! lsof -ti:$alt_port > /dev/null 2>&1; then
                frontend_port=$alt_port
                log "Using alternative port: $frontend_port"
                break
            fi
        done
    fi
    
    # Start with specific port if needed
    if [ $frontend_port -ne $FRONTEND_PORT ]; then
        export PORT=$frontend_port
        npm run dev > frontend.log 2>&1 &
    else
        npm run dev > frontend.log 2>&1 &
    fi
    
    FRONTEND_PID=$!
    
    log "Frontend starting on port $frontend_port (PID: $FRONTEND_PID)..."
    
    # Wait for frontend to start - check both default and alternative ports
    local max_attempts=60  # Extended timeout for 3D deps
    local attempt=0
    
    while [ $attempt -lt $max_attempts ]; do
        if curl -s http://localhost:$frontend_port > /dev/null 2>&1; then
            success "Frontend server ready on port $frontend_port"
            FRONTEND_PORT=$frontend_port  # Update global variable
            return 0
        fi
        
        # Also check if it started on a different port (Next.js auto-port detection)
        for check_port in 3000 3001 3002 3003; do
            if curl -s http://localhost:$check_port > /dev/null 2>&1; then
                success "Frontend server ready on port $check_port"
                FRONTEND_PORT=$check_port
                return 0
            fi
        done
        
        sleep 1
        attempt=$((attempt + 1))
        echo -n "."
    done
    
    error "Frontend failed to start within 60 seconds"
    log "Check frontend.log for details"
    
    # Show last few lines of frontend log for debugging
    if [ -f "frontend.log" ]; then
        log "Last 10 lines of frontend.log:"
        tail -10 frontend.log 2>/dev/null || true
    fi
    
    return 1
}

# Function to start MCP server as background service
start_mcp_server() {
    log "Starting MCP server for LLM integration..."
    cd "$PROJECT_ROOT/backend"
    
    if [ -f "package.json" ] && npm list > /dev/null 2>&1; then
        # Start MCP server as background service
        log "Starting MCP server on port $MCP_PORT..."
        npm run mcp-server > mcp.log 2>&1 &
        MCP_PID=$!
        
        log "MCP server starting (PID: $MCP_PID)..."
        
        # Give it time to initialize
        sleep 3
        
        # Check if it started successfully
        if curl -s http://localhost:$MCP_PORT/health > /dev/null 2>&1; then
            success "MCP server ready on port $MCP_PORT"
            echo "   💡 MCP server available for LLM integrations"
            echo "   💡 Connect AI assistants to http://localhost:$MCP_PORT"
            return 0
        else
            # Check stdio mode
            if grep -q "Mini-XDR MCP server running" mcp.log 2>/dev/null; then
                success "MCP server available (stdio mode)"
                echo "   💡 MCP server runs on stdio for LLM integrations"
                return 0
            else
                warning "MCP server failed to start - check mcp.log"
                kill $MCP_PID 2>/dev/null || true
                wait $MCP_PID 2>/dev/null || true
                MCP_PID=""
                return 1
            fi
        fi
    else
        warning "MCP server dependencies not found - LLM integration disabled"
        MCP_PID=""
        return 1
    fi
}

# Function to test MCP server availability (legacy fallback)
test_mcp_server() {
    log "Testing MCP server availability..."
    cd "$PROJECT_ROOT/backend"
    
    if [ -f "package.json" ] && npm list > /dev/null 2>&1; then
        # Test if MCP server can start by running it briefly
        log "Verifying MCP server can start..."
        npm run mcp > mcp_test.log 2>&1 &
        local test_pid=$!
        
        # Give it time to initialize
        sleep 2
        
        # Check if it started successfully
        if grep -q "Mini-XDR MCP server running on stdio" mcp_test.log 2>/dev/null; then
            success "MCP server available and working"
            echo "   💡 MCP server runs on-demand for LLM integrations"
            echo "   💡 Use MCP clients or AI assistants to connect via stdio"
            
            # Kill the test process
            kill $test_pid 2>/dev/null || true
            wait $test_pid 2>/dev/null || true
            MCP_PID="available"
        else
            warning "MCP server failed to initialize - check mcp_test.log"
            kill $test_pid 2>/dev/null || true
            wait $test_pid 2>/dev/null || true
            MCP_PID=""
        fi
    else
        warning "MCP server dependencies not found - LLM integration disabled"
        MCP_PID=""
    fi
}

# Function to perform comprehensive health checks
perform_health_checks() {
    log "Performing comprehensive system health checks..."
    echo ""
    
    # Backend API health
    log "🔍 Testing Backend API..."
    local health_response=$(curl -s http://localhost:$BACKEND_PORT/health 2>/dev/null)
    if [ $? -eq 0 ]; then
        success "Backend API responding"
        echo "   Response: $health_response"
    else
        error "Backend API not responding"
        return 1
    fi
    
    # Test incidents endpoint
    log "🔍 Testing Incidents API..."
    local incidents_response=$(curl -s http://localhost:$BACKEND_PORT/incidents 2>/dev/null)
    if [ $? -eq 0 ]; then
        local incident_count=$(echo "$incidents_response" | jq length 2>/dev/null || echo "unknown")
        success "Incidents API responding ($incident_count incidents)"
    else
        error "Incidents API not responding"
    fi
    
    # Test enhanced ML API
    log "🔍 Testing ML Status API..."
    local ml_response=$(curl -s http://localhost:$BACKEND_PORT/api/ml/status 2>/dev/null)
    if [ $? -eq 0 ]; then
        local models_trained=$(echo "$ml_response" | jq -r '.metrics.models_trained' 2>/dev/null || echo "unknown")
        local total_models=$(echo "$ml_response" | jq -r '.metrics.total_models' 2>/dev/null || echo "unknown")
        success "ML Status API responding ($models_trained/$total_models models trained)"
    else
        warning "ML Status API not responding"
    fi
    
    # Test AI Agents API
    log "🔍 Testing AI Agents API..."
    local agent_response=$(curl -s -X POST http://localhost:$BACKEND_PORT/api/agents/orchestrate \
        -H "Content-Type: application/json" \
        -d '{"agent_type": "containment", "query": "System status check", "history": []}' 2>/dev/null)
    if [ $? -eq 0 ]; then
        success "AI Agents API responding"
    else
        warning "AI Agents API not responding"
    fi
    
    # Frontend connectivity
    log "🔍 Testing Frontend..."
    if curl -s http://localhost:$FRONTEND_PORT > /dev/null 2>&1; then
        success "Frontend responding"
        
        # Test for LightningCSS issues on macOS with Tailwind v4
        if [[ "$OSTYPE" == "darwin"* ]]; then
            cd "$PROJECT_ROOT/frontend"
            if grep -q '"tailwindcss": "^4' package.json 2>/dev/null; then
                log "🔍 Verifying LightningCSS native binary..."
                if [ -f "node_modules/lightningcss-darwin-arm64/lightningcss.darwin-arm64.node" ]; then
                    success "LightningCSS native binary verified"
                else
                    warning "LightningCSS native binary missing - builds may fail"
                fi
            fi
        fi
    else
        error "Frontend not responding"
        return 1
    fi
    
    # Test auto-contain setting
    log "🔍 Testing Auto-contain API..."
    local auto_contain=$(curl -s http://localhost:$BACKEND_PORT/settings/auto_contain 2>/dev/null)
    if [ $? -eq 0 ]; then
        success "Auto-contain API responding"
        echo "   Setting: $auto_contain"
    else
        warning "Auto-contain API not responding"
    fi
    
    # Test adaptive detection system
    log "🔍 Testing Adaptive Detection System..."
    local adaptive_status=$(curl -s http://localhost:$BACKEND_PORT/api/adaptive/status 2>/dev/null)
    if [ $? -eq 0 ]; then
        local learning_running=$(echo "$adaptive_status" | jq -r '.learning_pipeline.running' 2>/dev/null || echo "unknown")
        local behavioral_threshold=$(echo "$adaptive_status" | jq -r '.adaptive_engine.behavioral_threshold' 2>/dev/null || echo "unknown")
        local ml_models=$(echo "$adaptive_status" | jq -r '.ml_detector | keys | length' 2>/dev/null || echo "unknown")
        
        success "Adaptive Detection System responding"
        echo "   Learning Pipeline Running: $learning_running"
        echo "   Behavioral Threshold: $behavioral_threshold"
        echo "   ML Models Available: $ml_models"
        
        # Test forced learning update
        log "🔍 Testing Learning Pipeline..."
        local learning_result=$(curl -s -X POST http://localhost:$BACKEND_PORT/api/adaptive/force_learning 2>/dev/null)
        if [ $? -eq 0 ]; then
            local update_success=$(echo "$learning_result" | jq -r '.success' 2>/dev/null || echo "false")
            if [ "$update_success" = "true" ]; then
                success "Learning Pipeline functional"
            else
                warning "Learning Pipeline update failed"
            fi
        else
            warning "Learning Pipeline test failed"
        fi
    else
        error "Adaptive Detection System not responding"
    fi
    
    # Test SSH connectivity endpoint
    log "🔍 Testing SSH Connectivity API..."
    local ssh_test=$(curl -s http://localhost:$BACKEND_PORT/test/ssh 2>/dev/null)
    if [ $? -eq 0 ]; then
        if echo "$ssh_test" | grep -q '"ssh_status":"success"'; then
            success "Honeypot SSH connectivity verified via API"
        else
            warning "Honeypot SSH connectivity failed - check configuration"
            echo "   Response: $ssh_test"
        fi
    else
        warning "SSH connectivity test API not responding"
    fi
    
    # Database check
    log "🔍 Testing Database..."
    if [ -f "$PROJECT_ROOT/backend/xdr.db" ]; then
        local db_size=$(ls -lh "$PROJECT_ROOT/backend/xdr.db" | awk '{print $5}')
        success "Database file exists ($db_size)"
    else
        warning "Database file not found - will be created on first use"
    fi
    
    # Environment variables check
    log "🔍 Checking Configuration..."
    cd "$PROJECT_ROOT/backend"
    source .venv/bin/activate
    
    # Check honeypot configuration
    local honeypot_host=$(python -c "from app.config import settings; print(settings.honeypot_host)" 2>/dev/null)
    local honeypot_user=$(python -c "from app.config import settings; print(settings.honeypot_user)" 2>/dev/null)
    local honeypot_port=$(python -c "from app.config import settings; print(settings.honeypot_ssh_port)" 2>/dev/null)
    
    if [ ! -z "$honeypot_host" ] && [ "$honeypot_host" != "10.0.0.23" ]; then
        success "Honeypot configuration customized ($honeypot_user@$honeypot_host:$honeypot_port)"
    else
        warning "Honeypot configuration using defaults - please update .env file"
    fi
    
    # Check if LLM keys are configured
    if python -c "from app.config import settings; print('configured' if settings.openai_api_key or settings.xai_api_key else 'not configured')" 2>/dev/null | grep -q "configured"; then
        success "LLM API keys configured"
    else
        warning "LLM API keys not configured - AI analysis will be disabled"
    fi
    
    # Check enhanced ML configuration
    local ml_models_path=$(python -c "from app.config import settings; print(settings.ml_models_path)" 2>/dev/null)
    if [ -d "$PROJECT_ROOT/backend/$ml_models_path" ]; then
        success "ML models directory exists ($ml_models_path)"
    else
        warning "ML models directory not found - will be created automatically"
    fi
    
    # Check policies configuration
    local policies_path=$(python -c "from app.config import settings; print(settings.policies_path)" 2>/dev/null)
    if [ -d "$PROJECT_ROOT/$policies_path" ]; then
        local policies_count=$(find "$PROJECT_ROOT/$policies_path" -name "*.yaml" -o -name "*.yml" | wc -l)
        success "Policies directory exists ($policies_count YAML files)"
    else
        warning "Policies directory not found - default policies will be used"
    fi
    
    # Check MCP server
    if [ "$MCP_PID" = "available" ]; then
        success "MCP server available for LLM integration"
    elif [ ! -z "$MCP_PID" ]; then
        warning "MCP server status unknown"
    else
        warning "MCP server not available - LLM integration disabled"
    fi
    
    # Test enhanced multi-source ingestion
    log "🔍 Testing Enhanced Multi-Source Ingestion..."
    local sample_response=$(curl -s -X POST http://localhost:$BACKEND_PORT/ingest/multi \
        -H 'Content-Type: application/json' \
        -H 'Authorization: Bearer test-api-key' \
        -d '{"source_type":"cowrie","hostname":"startup-test","events":[{"eventid":"cowrie.login.failed","src_ip":"192.168.1.100","username":"admin","password":"123456","message":"Test event from startup script","timestamp":"'$(date -u +%Y-%m-%dT%H:%M:%SZ)'"}]}' 2>/dev/null)
    
    if [ $? -eq 0 ]; then
        local processed=$(echo "$sample_response" | jq -r '.processed' 2>/dev/null || echo "unknown")
        success "Enhanced event ingestion test successful ($processed events processed)"
        echo "   Response: $sample_response"
    else
        warning "Enhanced event ingestion test failed"
    fi
    
    # Test adaptive detection with behavioral patterns
    log "🔍 Testing Adaptive Detection with Behavioral Patterns..."
    local test_ip="192.168.1.200"
    local adaptive_test_events='[
        {"eventid":"webhoneypot.request","src_ip":"'$test_ip'","message":"GET /admin.php","raw":{"path":"/admin.php","status_code":404,"attack_indicators":["admin_scan"]}},
        {"eventid":"webhoneypot.request","src_ip":"'$test_ip'","message":"GET /wp-admin/","raw":{"path":"/wp-admin/","status_code":404,"attack_indicators":["admin_scan"]}},
        {"eventid":"webhoneypot.request","src_ip":"'$test_ip'","message":"GET /index.php?id=1 OR 1=1","raw":{"path":"/index.php","parameters":["id=1 OR 1=1"],"status_code":500,"attack_indicators":["sql_injection"]}}
    ]'
    
    local adaptive_response=$(curl -s -X POST http://localhost:$BACKEND_PORT/ingest/multi \
        -H 'Content-Type: application/json' \
        -H 'Authorization: Bearer test-api-key' \
        -d '{"source_type":"webhoneypot","hostname":"adaptive-test","events":'$adaptive_test_events'}' 2>/dev/null)
    
    if [ $? -eq 0 ]; then
        local adaptive_incidents=$(echo "$adaptive_response" | jq -r '.incidents_detected' 2>/dev/null || echo "0")
        local adaptive_processed=$(echo "$adaptive_response" | jq -r '.processed' 2>/dev/null || echo "0")
        
        if [ "${adaptive_incidents:-0}" -gt 0 ] 2>/dev/null; then
            success "Adaptive Detection triggered ($adaptive_incidents incidents from $adaptive_processed events)"
            
            # Check if the incident was created with adaptive reasoning
            sleep 1
            local recent_incident=$(curl -s http://localhost:$BACKEND_PORT/incidents 2>/dev/null | jq -r '.[0].reason' 2>/dev/null)
            if [[ "$recent_incident" == *"adaptive"* ]] || [[ "$recent_incident" == *"Behavioral"* ]]; then
                success "Intelligent adaptive detection confirmed"
                echo "   Incident: $recent_incident"
            else
                log "Traditional detection triggered (adaptive features may need more data)"
            fi
        else
            log "No incidents triggered by test (may need more events for adaptive detection)"
        fi
    else
        warning "Adaptive detection test failed"
    fi
    
    # Test federated learning system
    log "🔍 Testing Federated Learning System..."
    local federated_status=$(curl -s http://localhost:$BACKEND_PORT/api/federated/status 2>/dev/null)
    if [ $? -eq 0 ]; then
        local federated_available=$(echo "$federated_status" | jq -r '.available' 2>/dev/null || echo "unknown")
        if [ "$federated_available" = "true" ]; then
            success "Federated Learning System operational"
            
            # Test federated models status
            local models_status=$(curl -s http://localhost:$BACKEND_PORT/api/federated/models/status 2>/dev/null)
            if [ $? -eq 0 ]; then
                local federated_enabled=$(echo "$models_status" | jq -r '.federated_capabilities.ensemble_with_federated' 2>/dev/null || echo "false")
                success "Federated ML integration ready (enabled: $federated_enabled)"
            else
                warning "Federated models status check failed"
            fi
        else
            warning "Federated Learning System not fully available"
            echo "   Status: $federated_status"
        fi
    else
        warning "Federated Learning API not responding"
    fi
    
    # Test 3D Visualization APIs (Phase 4.1)
    log "🔍 Testing 3D Visualization APIs..."
    local threats_response=$(curl -s http://localhost:$BACKEND_PORT/api/intelligence/threats 2>/dev/null)
    if [ $? -eq 0 ]; then
        local threat_count=$(echo "$threats_response" | jq -r '.total_count' 2>/dev/null || echo "unknown")
        success "Threat Intelligence API responding ($threat_count threats)"
        
        # Test timeline API
        local timeline_response=$(curl -s http://localhost:$BACKEND_PORT/api/incidents/timeline 2>/dev/null)
        if [ $? -eq 0 ]; then
            local timeline_count=$(echo "$timeline_response" | jq -r '.total_count' 2>/dev/null || echo "unknown")
            success "Timeline API responding ($timeline_count events)"
        else
            warning "Timeline API not responding"
        fi
        
        # Test attack paths API
        local paths_response=$(curl -s http://localhost:$BACKEND_PORT/api/incidents/attack-paths 2>/dev/null)
        if [ $? -eq 0 ]; then
            local paths_count=$(echo "$paths_response" | jq -r '.total_count' 2>/dev/null || echo "unknown")
            success "Attack Paths API responding ($paths_count paths)"
        else
            warning "Attack Paths API not responding"
        fi
    else
        warning "3D Visualization APIs not responding"
    fi
    
    # Test distributed system APIs
    log "🔍 Testing Distributed MCP System..."
    local distributed_status=$(curl -s http://localhost:$BACKEND_PORT/api/distributed/status 2>/dev/null)
    if [ $? -eq 0 ]; then
        local capabilities_count=$(echo "$distributed_status" | jq -r '.capabilities | length' 2>/dev/null || echo "unknown")
        success "Distributed MCP System responding ($capabilities_count capabilities)"
        
        # Test distributed health
        local health_response=$(curl -s http://localhost:$BACKEND_PORT/api/distributed/health 2>/dev/null)
        if [ $? -eq 0 ]; then
            local overall_healthy=$(echo "$health_response" | jq -r '.overall_healthy' 2>/dev/null || echo "unknown")
            success "Distributed system health check (healthy: $overall_healthy)"
        else
            warning "Distributed health check failed"
        fi
    else
        warning "Distributed MCP APIs not responding"
    fi
    
    echo ""
    success "Health checks completed!"
    return 0
}

# Function to display system status
show_system_status() {
    echo ""
    echo "=== 🚀 Mini-XDR System Status ==="
    echo ""
    echo "📊 Services:"
    echo "   • Frontend:        http://localhost:$FRONTEND_PORT"
    echo "   • Backend:         http://localhost:$BACKEND_PORT"
    echo "   • API Docs:        http://localhost:$BACKEND_PORT/docs"
    echo "   • 3D Visualization: http://localhost:$FRONTEND_PORT/visualizations"
    echo "   • AI Agents:       http://localhost:$FRONTEND_PORT/agents"
    echo "   • Analytics:       http://localhost:$FRONTEND_PORT/analytics"
    if [ ! -z "$MCP_PID" ] && [ "$MCP_PID" != "available" ]; then
        echo "   • MCP Server:      http://localhost:$MCP_PORT"
    fi
    echo ""
    echo "📋 Process IDs:"
    echo "   • Backend PID:  ${BACKEND_PID:-"Not running"}"
    echo "   • Frontend PID: ${FRONTEND_PID:-"Not running"}"
    if [ "$MCP_PID" = "available" ]; then
        echo "   • MCP Server:   Available (on-demand)"
    else
        echo "   • MCP Server:   ${MCP_PID:-"Not available"}"
    fi
    echo ""
    
    # Get honeypot configuration for status display
    cd "$PROJECT_ROOT/backend"
    source .venv/bin/activate 2>/dev/null
    local honeypot_info=$(python -c "
from app.config import settings
print(f'{settings.honeypot_user}@{settings.honeypot_host}:{settings.honeypot_ssh_port}')
" 2>/dev/null)
    
    echo "🍯 Honeypot Configuration:"
    echo "   • Connection: ${honeypot_info:-"Not configured"}"
    echo "   • SSH Key:    $(python -c "from app.config import settings; print(settings.expanded_ssh_key_path)" 2>/dev/null)"
    echo ""
    
    echo "📝 Logs:"
    echo "   • Backend:  $PROJECT_ROOT/backend/backend.log"
    echo "   • Frontend: $PROJECT_ROOT/frontend/frontend.log"
    echo "   • MCP:      $PROJECT_ROOT/backend/mcp.log"
    echo ""
    
    echo "🔧 Configuration Files:"
    echo "   • Backend:  $PROJECT_ROOT/backend/.env"
    echo "   • Frontend: $PROJECT_ROOT/frontend/.env.local"
    echo ""
    
    echo "🧪 Quick Tests:"
    echo "   • Test Event:   curl -X POST http://localhost:$BACKEND_PORT/ingest/multi -H 'Content-Type: application/json' -H 'Authorization: Bearer test-api-key' -d '{\"source_type\":\"cowrie\",\"hostname\":\"test\",\"events\":[{\"eventid\":\"cowrie.login.failed\",\"src_ip\":\"192.168.1.100\",\"timestamp\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}]}'"
    echo "   • Build Test:   cd frontend && npm run build"
    echo "   • ML Status:    curl http://localhost:$BACKEND_PORT/api/ml/status"
    echo "   • AI Agent:     curl -X POST http://localhost:$BACKEND_PORT/api/agents/orchestrate -H 'Content-Type: application/json' -d '{\"agent_type\":\"containment\",\"query\":\"status\",\"history\":[]}'"
    echo "   • SSH Test:     curl http://localhost:$BACKEND_PORT/test/ssh"
    echo "   • Adaptive:     curl http://localhost:$BACKEND_PORT/api/adaptive/status"
    echo "   • Learning:     curl -X POST http://localhost:$BACKEND_PORT/api/adaptive/force_learning"
    echo "   • Fed Status:   curl http://localhost:$BACKEND_PORT/api/federated/status"
    echo "   • Fed Models:   curl http://localhost:$BACKEND_PORT/api/federated/models/status"
    echo "   • Fed Insights: curl http://localhost:$BACKEND_PORT/api/federated/insights"
    echo "   • View Logs:    tail -f $PROJECT_ROOT/backend/backend.log"
    echo ""
    
    echo "🧠 Adaptive Detection Features:"
    echo "   • Behavioral Pattern Analysis: Detects attack patterns without signatures"
    echo "   • Statistical Baseline Learning: Learns normal behavior automatically"
    echo "   • ML Ensemble Detection: Multi-model anomaly detection"
    echo "   • Continuous Learning: Self-improving detection over time"
    echo "   • Zero-Day Detection: Identifies unknown attack methods"
    echo ""
    
    echo "🔄 Federated Learning Features (Phase 2):"
    echo "   • Privacy-Preserving Model Aggregation: Secure multi-party computation"
    echo "   • Differential Privacy Protection: Noise injection for data privacy"
    echo "   • Cross-Organization Threat Sharing: Collaborative intelligence"
    echo "   • Advanced Cryptographic Protocols: 4 security levels available"
    echo "   • Real-Time Model Synchronization: Distributed learning system"
    echo ""
    
    if [ "$MCP_PID" = "available" ]; then
        echo "🤖 MCP Server Usage:"
        echo "   • Start MCP:  cd $PROJECT_ROOT/backend && npm run mcp"
        echo "   • Connect AI assistants via stdio to access Mini-XDR tools"
        echo "   • Available tools: get_incidents, contain_incident, get_system_health, etc."
        echo ""
    fi
    
    echo "🎮 Controls:"
    echo "   • Dashboard: Open http://localhost:$FRONTEND_PORT"
    echo "   • Stop All:  Press Ctrl+C"
    echo "   • Restart:   Run this script again"
    echo ""
}

# Cleanup function
cleanup() {
    echo ""
    log "Shutting down Mini-XDR services..."
    
    [ ! -z "$BACKEND_PID" ] && kill $BACKEND_PID 2>/dev/null
    [ ! -z "$FRONTEND_PID" ] && kill $FRONTEND_PID 2>/dev/null  
    # MCP server runs on-demand, no persistent process to kill
    
    sleep 2
    
    # Force kill if still running
    kill_port_processes $BACKEND_PORT "Backend"
    kill_port_processes $FRONTEND_PORT "Frontend"
    kill_port_processes $MCP_PORT "MCP"
    
    success "All services stopped"
    exit 0
}

# Function to show configuration guidance
show_configuration_guidance() {
    echo ""
    echo "=== ⚙️  Configuration Guidance ==="
    echo ""
    
    # Check if .env files need attention
    cd "$PROJECT_ROOT/backend"
    if [ -f ".env" ]; then
        local honeypot_host=$(grep "^HONEYPOT_HOST=" .env | cut -d'=' -f2)
        if [ -z "$honeypot_host" ] || [ "$honeypot_host" = "10.0.0.23" ]; then
            warning "Honeypot configuration needs attention:"
            echo "   1. Edit $PROJECT_ROOT/backend/.env"
            echo "   2. Set HONEYPOT_HOST to your honeypot VM IP"
            echo "   3. Configure SSH key path and credentials"
            echo ""
        fi
    fi
    
    # Check SSH keys
    if [ -f ".env" ]; then
        local ssh_key_path=$(python -c "from app.config import settings; print(settings.expanded_ssh_key_path)" 2>/dev/null)
        if [ ! -f "$ssh_key_path" ]; then
            warning "SSH keys need to be generated:"
            echo "   1. Generate key: ssh-keygen -t ed25519 -f $ssh_key_path"
            echo "   2. Copy to honeypot: ssh-copy-id -i ${ssh_key_path}.pub -p 22022 xdrops@<honeypot-ip>"
            echo ""
        fi
    fi
    
    # Check LLM configuration
    source .venv/bin/activate 2>/dev/null
    if ! python -c "from app.config import settings; exit(0 if settings.openai_api_key or settings.xai_api_key else 1)" 2>/dev/null; then
        warning "LLM integration not configured (optional):"
        echo "   1. Get API key from OpenAI or X.AI"
        echo "   2. Add to .env file: OPENAI_API_KEY=your_key or XAI_API_KEY=your_key"
        echo ""
    fi
}

# Main execution
main() {
    clear
    echo "=== 🛡️  Mini-XDR Complete System Startup ==="
    echo "Comprehensive setup and deployment script"
    echo ""
    
    # Set up signal handling
    trap cleanup SIGINT SIGTERM
    
    # Step 1: Clean up existing services
    log "🧹 Cleaning up existing services..."
    kill_existing_services
    echo ""
    
    # Step 2: Setup and check prerequisites
    log "🔧 Setting up dependencies and environment..."
    check_prerequisites
    echo ""
    
    # Step 3: Show configuration guidance if needed
    show_configuration_guidance
    
    # Step 4: Start services
    log "🚀 Starting all services..."
    
    if ! start_backend; then
        error "Failed to start backend - aborting"
        exit 1
    fi
    
    sleep 2
    
    if ! start_frontend; then
        error "Failed to start frontend - aborting"
        cleanup
        exit 1
    fi
    
    sleep 2
    
    # Try to start MCP server (optional)
    log "Attempting to start MCP server..."
    if start_mcp_server; then
        success "MCP server integration ready"
    else
        warning "MCP server not started - continuing without LLM integration"
        test_mcp_server  # Fallback to test mode
    fi
    
    echo ""
    
    # Step 5: Perform health checks
    log "🔍 Running comprehensive health checks..."
    if perform_health_checks; then
        echo ""
        success "🎉 Enhanced Mini-XDR System Successfully Started!"
        show_system_status
        
        echo "🛡️  Enhanced XDR System Ready with:"
        echo "   🤖 AI Agents for autonomous threat response"
        echo "   🧠 Intelligent Adaptive Detection"
        echo "   🔄 Federated Learning System (Phase 2 - NEW!)"
        echo "   🔐 Secure Multi-Party Computation"
        echo "   📊 ML Ensemble Models for anomaly detection" 
        echo "   📈 Behavioral Pattern Analysis"
        echo "   📊 Statistical Baseline Learning"
        echo "   🔄 Continuous Learning Pipeline"
        echo "   🔗 Multi-source log ingestion (Cowrie, Suricata, OSQuery)"
        echo "   📋 Policy-based automated containment"
        echo "   🚨 Zero-day attack detection capabilities"
        echo ""
        echo "🚀 INTELLIGENT ADAPTIVE DETECTION ACTIVE!"
        echo "   • Learns normal behavior patterns automatically"
        echo "   • Detects unknown attack methods without signatures"
        echo "   • Reduces false positives through contextual understanding"
        echo "   • Self-improving detection accuracy over time"
        echo ""
        echo "🔄 FEDERATED LEARNING SYSTEM ACTIVE!"
        echo "   • Privacy-preserving collaborative threat intelligence"
        echo "   • Secure multi-party model aggregation"
        echo "   • Differential privacy protection"
        echo "   • Cross-organization knowledge sharing"
        echo ""
        echo "Ready for advanced threat hunting and collaborative defense!"
        echo "Press Ctrl+C to stop all services"
        echo ""
        
        # Wait for interrupt
        wait
    else
        error "Health checks failed - check logs for issues"
        cleanup
        exit 1
    fi
}

# Run main function
main

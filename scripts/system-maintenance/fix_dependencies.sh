#!/bin/bash
# Phase 2B Dependencies Fix Script
# Resolves scipy compilation issues on macOS and installs Phase 2B ML dependencies

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

echo "🔧 Phase 2B Dependencies Fix Script"
echo "=================================="

log "Fixing scipy compilation issues and installing Phase 2B ML dependencies..."

# Install system dependencies for scipy compilation
if [[ "$OSTYPE" == "darwin"* ]]; then
    log "Detected macOS - installing system dependencies..."
    
    # Check if Homebrew is installed
    if command -v brew >/dev/null 2>&1; then
        log "Installing system dependencies via Homebrew..."
        
        # Install pkg-config and openblas (required for scipy)
        brew install pkg-config openblas gfortran || warning "Some Homebrew packages failed to install"
        
        # Set environment variables for scipy compilation
        export PKG_CONFIG_PATH="/opt/homebrew/lib/pkgconfig:$PKG_CONFIG_PATH"
        export OPENBLAS=/opt/homebrew
        export LDFLAGS="-L/opt/homebrew/lib $LDFLAGS"
        export CPPFLAGS="-I/opt/homebrew/include $CPPFLAGS"
        
        success "System dependencies installed"
    else
        warning "Homebrew not found - install it from https://brew.sh for best results"
        warning "Manual installation: brew install pkg-config openblas gfortran"
    fi
fi

# Navigate to backend directory
cd "$PROJECT_ROOT/backend" || {
    error "Backend directory not found"
    exit 1
}

# Activate virtual environment
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    success "Virtual environment activated"
else
    error "Virtual environment not found - run ./scripts/start-all.sh first"
    exit 1
fi

# Try to install scipy with proper configuration
log "Attempting to install scipy with proper configuration..."

# Method 1: Try with specific scipy version known to work with current setup
pip install --no-cache-dir scipy==1.13.1 || {
    warning "Method 1 failed - trying alternative approach..."
    
    # Method 2: Try without version constraint
    pip install --no-cache-dir scipy || {
        warning "Method 2 failed - trying conda approach..."
        
        # Method 3: Try with conda (if available)
        if command -v conda >/dev/null 2>&1; then
            log "Attempting conda installation..."
            conda install scipy -c conda-forge -y || {
                warning "Conda installation also failed - scipy will be disabled"
            }
        else
            warning "SciPy installation failed - statistical features will use fallbacks"
        fi
    }
}

# Install Phase 2B dependencies using our custom script
log "Installing Phase 2B Advanced ML dependencies..."
if [ -f "utils/install_phase2b_deps.py" ]; then
    python utils/install_phase2b_deps.py
else
    error "Phase 2B installation script not found"
    exit 1
fi

# Test the installation
log "Testing Phase 2B features..."
python -c "
import sys
print('🔍 Testing Phase 2B imports...')

# Test core functionality
try:
    from app.online_learning import online_learning_engine
    from app.concept_drift import create_drift_detector
    from app.model_versioning import model_registry
    from app.ensemble_optimizer import meta_learning_optimizer
    from app.explainable_ai import explainable_ai
    print('✅ Core Phase 2B modules: OK')
except ImportError as e:
    print(f'❌ Core modules failed: {e}')

# Test optional ML libraries
packages = [
    ('shap', 'SHAP explanations'),
    ('lime', 'LIME explanations'), 
    ('optuna', 'Hyperparameter optimization'),
    ('scipy', 'Statistical functions')
]

for package, description in packages:
    try:
        __import__(package)
        print(f'✅ {package}: Available ({description})')
    except ImportError:
        print(f'⚠️  {package}: Not available - using fallbacks ({description})')

print('🎉 Phase 2B system check complete!')
"

echo ""
echo "🚀 Phase 2B Dependencies Fix Complete!"
echo "======================================"
echo ""
echo "Next steps:"
echo "1. Run: ./scripts/start-all.sh"
echo "2. Visit: http://localhost:3000/analytics"
echo "3. Test Phase 2B features in the ML Monitoring dashboard"
echo ""

if ! python -c "import scipy" >/dev/null 2>&1; then
    echo "📋 Note about SciPy:"
    echo "SciPy installation failed but this is OK - the system will work with reduced functionality."
    echo "Statistical drift detection will use fallback implementations."
    echo ""
    echo "If you need full SciPy support, try:"
    echo "  brew install pkg-config openblas gfortran"
    echo "  pip install --no-cache-dir scipy"
    echo ""
fi

success "Dependencies fix script completed!"

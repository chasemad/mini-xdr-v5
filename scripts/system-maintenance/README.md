# 🛠️ System Maintenance Scripts

This directory contains scripts for maintaining and troubleshooting your Mini-XDR system, including dependency management, performance optimization, and system health checks.

## Scripts Overview

### 🔧 Dependency Management

#### `fix_dependencies.sh`
**Phase 2B dependencies fix and installation**
- **Purpose**: Resolve scipy compilation issues and install advanced ML dependencies
- **Features**: macOS-specific fixes, virtual environment management, dependency testing
- **Usage**: `./fix_dependencies.sh`
- **When to use**: After system updates or when ML features aren't working

## Detailed Script Information

### `fix_dependencies.sh`

#### What it does:
1. **System Dependencies**: Installs required macOS packages via Homebrew
   - `pkg-config` - Required for scipy compilation
   - `openblas` - Optimized linear algebra library
   - `gfortran` - Fortran compiler for scipy

2. **Python Environment**: Manages virtual environment and Python packages
   - Activates Mini-XDR virtual environment
   - Installs scipy with proper configuration
   - Runs Phase 2B dependency installer

3. **Validation**: Tests all installed components
   - Core Phase 2B modules
   - Optional ML libraries (SHAP, LIME, Optuna)
   - Statistical functions

#### When to run:
- After macOS system updates
- When seeing scipy import errors
- Before using advanced ML features
- After Python version changes

#### Expected output:
```
✅ Core Phase 2B modules: OK
✅ shap: Available (SHAP explanations)  
✅ lime: Available (LIME explanations)
✅ optuna: Available (Hyperparameter optimization)
⚠️  scipy: Not available - using fallbacks (Statistical functions)
```

## Usage Examples

### Basic Maintenance
```bash
# Fix dependencies after system update
./fix_dependencies.sh

# Check if fix was successful
cd ../backend && python -c "from app.online_learning import online_learning_engine; print('✅ Advanced ML ready')"
```

### Troubleshooting Workflow
```bash
# 1. Run dependency fix
./fix_dependencies.sh

# 2. Test Mini-XDR startup
cd ../backend && python -m app.main

# 3. Check advanced features
# Visit http://localhost:3000/analytics
```

## Common Issues and Solutions

### SciPy Compilation Errors
**Problem**: SciPy fails to compile on macOS
```
ERROR: Failed building wheel for scipy
```

**Solution**: 
```bash
# Install system dependencies
brew install pkg-config openblas gfortran

# Run fix script
./fix_dependencies.sh
```

### Virtual Environment Issues
**Problem**: Script can't find virtual environment
```
❌ Virtual environment not found
```

**Solution**:
```bash
# Ensure Mini-XDR is set up first
cd ../../scripts && ./start-all.sh

# Then run maintenance
cd ../scripts/system-maintenance && ./fix_dependencies.sh
```

### Phase 2B Import Errors
**Problem**: Advanced ML features not working
```
ImportError: cannot import name 'online_learning_engine'
```

**Solution**:
```bash
# Run dependency fix
./fix_dependencies.sh

# Restart Mini-XDR
cd ../../scripts && ./stop-all.sh && ./start-all.sh
```

## System Requirements

### macOS Dependencies
- **Homebrew**: Package manager for macOS
- **Xcode Command Line Tools**: For compilation
- **Python 3.9+**: Core Python runtime

### Required Packages
- **pkg-config**: Configuration helper
- **OpenBLAS**: Linear algebra library  
- **gfortran**: Fortran compiler
- **scipy**: Scientific computing (optional)

## Integration with Mini-XDR

### Phase 2B Features Supported
- **Online Learning**: Adaptive ML model updates
- **Concept Drift Detection**: Model performance monitoring
- **Model Versioning**: ML model lifecycle management
- **Ensemble Optimization**: Multi-model coordination
- **Explainable AI**: ML decision interpretation

### Fallback Behavior
When optional dependencies aren't available:
- **SciPy**: Uses numpy-based statistical fallbacks
- **SHAP**: Disables advanced explainability features
- **LIME**: Uses basic explanation methods
- **Optuna**: Uses grid search for hyperparameter tuning

## Maintenance Schedule

### Regular Maintenance (Monthly)
```bash
# Update system packages
brew update && brew upgrade

# Fix any dependency issues
./fix_dependencies.sh

# Restart Mini-XDR to pick up changes
cd ../../scripts && ./stop-all.sh && ./start-all.sh
```

### After System Updates
```bash
# Always run after macOS updates
./fix_dependencies.sh

# Test all functionality
cd ../../scripts && ./system-status.sh
```

### Before Important Demos
```bash
# Ensure everything is working
./fix_dependencies.sh

# Verify all features
cd ../backend && python -c "
import sys
print('🔍 Testing all Phase 2B features...')

try:
    from app.online_learning import online_learning_engine
    from app.concept_drift import create_drift_detector  
    from app.model_versioning import model_registry
    from app.ensemble_optimizer import meta_learning_optimizer
    from app.explainable_ai import explainable_ai
    print('✅ All Phase 2B modules working!')
except Exception as e:
    print(f'❌ Issue found: {e}')
    print('🔧 Run fix_dependencies.sh to resolve')
"
```

## Future Maintenance Scripts

### Planned Additions
- **`system-health-check.sh`**: Comprehensive system diagnostics
- **`performance-optimization.sh`**: Database and ML model optimization
- **`backup-restore.sh`**: System backup and restore procedures
- **`security-audit.sh`**: Security configuration validation
- **`log-cleanup.sh`**: Log rotation and cleanup automation

---

**Status**: Production Ready  
**Last Updated**: September 16, 2025  
**Maintained by**: Mini-XDR System Team



#!/usr/bin/env python3
"""
Phase 2B Dependencies Installation Script
Safely installs Phase 2B dependencies with fallbacks for problematic packages
"""

import subprocess
import sys
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Phase 2B dependencies with NumPy 2.0 compatibility
PHASE_2B_DEPS = [
    {
        'name': 'shap',
        'version': None,  # Try latest first for NumPy 2.0 compatibility
        'description': 'SHAP explanations for model interpretability',
        'required': False,
        'fallback': True,
        'alternatives': ['shap>=0.43.0', 'shap==0.42.1', 'shap==0.41.0'],
        'numpy_compatible': True
    },
    {
        'name': 'lime',
        'version': None,  # Try latest first
        'description': 'LIME explanations for local interpretability',
        'required': False,
        'fallback': True,
        'alternatives': ['lime>=0.2.0.1', 'lime==0.2.0.1'],
        'numpy_compatible': False  # Known issues with NumPy 2.0
    },
    {
        'name': 'optuna',
        'version': None,  # Latest should be compatible
        'description': 'Hyperparameter optimization for ensemble models',
        'required': False,
        'fallback': True,
        'alternatives': ['optuna>=3.4.0', 'optuna==3.4.0', 'optuna==3.3.0'],
        'numpy_compatible': True
    },
    {
        'name': 'scipy',
        'version': None,  # Latest should work with NumPy 2.0
        'description': 'Statistical functions for drift detection',
        'required': False,
        'fallback': True,
        'alternatives': ['scipy>=1.14.0', 'scipy==1.13.1', 'scipy==1.12.0'],
        'numpy_compatible': True
    }
]

def run_command(cmd, timeout=300):
    """Run a shell command with timeout"""
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=True, 
            text=True, 
            timeout=timeout
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        logger.error(f"Command timed out: {cmd}")
        return False, "", "Timeout"
    except Exception as e:
        logger.error(f"Command failed: {cmd} - {e}")
        return False, "", str(e)

def install_package(package_spec, description):
    """Try to install a package"""
    logger.info(f"Installing {package_spec}: {description}")
    
    # Try pip install
    success, stdout, stderr = run_command(f"pip install {package_spec}")
    
    if success:
        logger.info(f"✅ Successfully installed {package_spec}")
        return True
    else:
        logger.warning(f"❌ Failed to install {package_spec}")
        logger.warning(f"Error: {stderr}")
        return False

def install_with_alternatives(dep):
    """Try to install a package with alternative versions"""
    name = dep['name']
    version = dep['version']
    alternatives = dep.get('alternatives', [])
    
    # Try main version first
    if version:
        package_spec = f"{name}=={version}"
    else:
        package_spec = name
        
    if install_package(package_spec, dep['description']):
        return True
    
    # Try alternatives
    for alt in alternatives:
        logger.info(f"Trying alternative: {alt}")
        if install_package(alt, f"{dep['description']} (alternative version)"):
            return True
    
    # Try without version constraint
    if version:
        logger.info(f"Trying {name} without version constraint")
        if install_package(name, f"{dep['description']} (any version)"):
            return True
    
    return False

def check_virtual_env():
    """Check if we're in a virtual environment"""
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        return True
    return False

def check_numpy_version():
    """Check NumPy version for compatibility"""
    try:
        import numpy as np
        version = np.__version__
        major_version = int(version.split('.')[0])
        
        logger.info(f"Found NumPy version: {version}")
        
        if major_version >= 2:
            logger.warning("NumPy 2.0+ detected - some packages may have compatibility issues")
            return 2
        else:
            return 1
    except ImportError:
        logger.error("NumPy not found - this is required!")
        return 0

def install_numpy_compatible_packages(numpy_version):
    """Install packages with NumPy compatibility handling"""
    
    if numpy_version >= 2:
        logger.info("🔧 Using NumPy 2.0+ compatible installation strategy")
        
        # For packages known to have NumPy 2.0 issues, try with --no-build-isolation
        problematic_packages = ['lime']
        
        for dep in PHASE_2B_DEPS:
            name = dep['name']
            
            if not dep.get('numpy_compatible', True) and name in problematic_packages:
                logger.warning(f"⚠️ {name} may have NumPy 2.0 compatibility issues - trying workaround")
                
                # Try installing with --no-build-isolation and fallbacks
                success = False
                
                # Method 1: Try latest with no-build-isolation
                cmd = f"pip install --no-build-isolation --no-deps {name}"
                success, stdout, stderr = run_command(cmd)
                
                if not success:
                    # Method 2: Try with downgraded numpy temporarily
                    logger.info(f"Trying {name} with temporary NumPy downgrade...")
                    
                    # Save current numpy version
                    run_command("pip freeze | grep numpy > /tmp/numpy_version.txt")
                    
                    # Temporarily downgrade numpy
                    run_command("pip install 'numpy<2.0'")
                    
                    # Install the package
                    success, _, _ = run_command(f"pip install {name}")
                    
                    if success:
                        # Restore numpy 2.0+
                        run_command("pip install 'numpy>=2.0'")
                        logger.info(f"✅ {name} installed with NumPy compatibility workaround")
                        return True
                    else:
                        # Restore original numpy
                        run_command("pip install 'numpy>=2.0'")
                
                return success
    
    return False

def main():
    """Main installation function with NumPy compatibility handling"""
    print("🚀 Phase 2B Dependencies Installation")
    print("=" * 50)
    
    # Check virtual environment
    if not check_virtual_env():
        logger.warning("⚠️  Not in a virtual environment. Consider activating venv first.")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # Check NumPy version
    numpy_version = check_numpy_version()
    if numpy_version == 0:
        logger.error("NumPy is required but not found!")
        sys.exit(1)
    
    # Track installation results
    installed = []
    failed = []
    
    # Try to install each dependency
    for dep in PHASE_2B_DEPS:
        name = dep['name']
        
        print(f"\n📦 Processing {name}...")
        
        # Special handling for NumPy 2.0 incompatible packages
        if numpy_version >= 2 and not dep.get('numpy_compatible', True):
            logger.info(f"⚠️ {name} may have NumPy 2.0 compatibility issues - using special handling")
            
            if install_numpy_compatible_packages(numpy_version):
                installed.append(name)
                continue
            else:
                logger.warning(f"NumPy compatibility workaround failed for {name}")
        
        # Standard installation
        if install_with_alternatives(dep):
            installed.append(name)
        else:
            failed.append(name)
            if not dep.get('fallback', False):
                logger.error(f"Critical dependency {name} failed to install!")
                if not dep.get('required', True):
                    logger.info("This dependency is optional - continuing...")
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Installation Summary:")
    print(f"✅ Successfully installed: {len(installed)}")
    for pkg in installed:
        print(f"   - {pkg}")
    
    if failed:
        print(f"❌ Failed to install: {len(failed)}")
        for pkg in failed:
            print(f"   - {pkg}")
        
        print("\n💡 Don't worry! The system includes fallbacks for failed packages.")
        print("   Your Mini-XDR Phase 2B features will still work with reduced functionality.")
    
    print("\n✨ Phase 2B dependencies installation complete!")
    
    # Test imports
    print("\n🔍 Testing imports...")
    test_imports()

def test_imports():
    """Test that we can import the packages with detailed error handling"""
    test_packages = {
        'shap': 'SHAP explanations',
        'lime': 'LIME explanations', 
        'optuna': 'Hyperparameter optimization',
        'scipy': 'Statistical functions'
    }
    
    print("Testing package imports with error details...")
    
    for package, description in test_packages.items():
        try:
            # Try to import the package
            imported_module = __import__(package)
            
            # Get version if available
            version = getattr(imported_module, '__version__', 'unknown')
            print(f"✅ {package} v{version}: Available ({description})")
            
        except ImportError as e:
            error_msg = str(e)
            
            # Provide specific guidance for common errors
            if 'obj2sctype' in error_msg:
                print(f"❌ {package}: NumPy 2.0 compatibility issue - {error_msg}")
                print(f"   💡 Fallback will be used for {description}")
            elif 'No module named' in error_msg:
                print(f"❌ {package}: Not installed - using fallbacks ({description})")
            else:
                print(f"❌ {package}: Import error - {error_msg}")
                print(f"   💡 Fallback will be used for {description}")
                
        except Exception as e:
            print(f"❌ {package}: Unexpected error - {str(e)}")
            print(f"   💡 Fallback will be used for {description}")
    
    print("\n📋 Import test summary:")
    print("- ✅ Available packages will provide full functionality")
    print("- ❌ Missing packages will use built-in fallbacks")
    print("- 💡 The system is designed to work with partial dependencies")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Installation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Installation failed: {e}")
        sys.exit(1)

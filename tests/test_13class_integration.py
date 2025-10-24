#!/usr/bin/env python3
"""
Test 13-Class Windows Specialist Integration
Verify ensemble detector can load and use the new model
"""

import sys
import os
import numpy as np
import logging

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.app.ensemble_ml_detector import EnsembleMLDetector

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_model_loading():
    """Test that the 13-class model loads successfully"""
    logger.info("=" * 70)
    logger.info("🧪 TEST: Windows 13-Class Model Loading")
    logger.info("=" * 70)
    
    try:
        detector = EnsembleMLDetector()
        
        # Check models loaded
        assert detector.windows_specialist is not None, "Windows specialist failed to load"
        
        logger.info("✅ Models loaded successfully")
        
        # Check class mapping
        assert len(detector.windows_classes) == 13, f"Expected 13 classes, got {len(detector.windows_classes)}"
        logger.info(f"✅ Windows classes: {len(detector.windows_classes)} classes")
        
        for class_id, class_name in detector.windows_classes.items():
            logger.info(f"   Class {class_id:2d}: {class_name}")
        
        return True
    
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_inference():
    """Test inference with synthetic Windows attack samples"""
    logger.info("\n" + "=" * 70)
    logger.info("🧪 TEST: Windows Attack Detection")
    logger.info("=" * 70)
    
    try:
        detector = EnsembleMLDetector()
        
        # Test cases
        test_cases = [
            ("Normal Activity", np.random.rand(79) * 0.2),  # Low values = normal
            ("Kerberos Attack", _generate_kerberos_attack()),
            ("Lateral Movement", _generate_lateral_movement()),
            ("Credential Theft", _generate_credential_theft()),
            ("Privilege Escalation", _generate_priv_esc()),
        ]
        
        results = []
        for name, features in test_cases:
            import asyncio
            result = asyncio.run(detector.detect_threat(features))
            
            logger.info(f"\n{name}:")
            logger.info(f"  Windows prediction: {result.get('windows_prediction', {}).get('threat_type', 'N/A')}")
            logger.info(f"  Confidence: {result.get('confidence', 0):.3f}")
            logger.info(f"  Model used: {result.get('model_used', 'N/A')}")
            
            results.append(result)
        
        logger.info("\n✅ Inference tests passed")
        return True
    
    except Exception as e:
        logger.error(f"❌ Inference test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def _generate_kerberos_attack():
    """Generate synthetic Kerberos attack features"""
    features = np.random.rand(79) * 0.3
    # Kerberos features (65-72)
    features[65] = 0.66  # TGS request
    features[66] = 0.5   # RC4 (weak encryption)
    features[67] = 0.8   # Ticket options
    features[75] = 0.9   # High anomaly score
    return features


def _generate_lateral_movement():
    """Generate synthetic lateral movement features"""
    features = np.random.rand(79) * 0.3
    features[3] = 445/65535  # SMB port
    features[4] = 0.5        # SMB protocol
    features[75] = 0.85      # Anomaly score
    features[78] = 0.8       # Baseline deviation
    return features


def _generate_credential_theft():
    """Generate synthetic credential theft features"""
    features = np.random.rand(79) * 0.3
    features[20] = 0.9  # High process ID indicator
    features[22] = 0.8  # LSASS process
    features[24] = 1.0  # System privileges
    features[75] = 0.95 # Very high anomaly
    return features


def _generate_priv_esc():
    """Generate synthetic privilege escalation features"""
    features = np.random.rand(79) * 0.3
    features[24] = 0.9  # Privilege indicator
    features[75] = 0.9  # Anomaly score
    features[46] = 0.8  # Elevation type
    return features


def test_model_info():
    """Test model info retrieval"""
    logger.info("\n" + "=" * 70)
    logger.info("🧪 TEST: Model Information")
    logger.info("=" * 70)
    
    try:
        detector = EnsembleMLDetector()
        info = detector.get_model_info()
        
        logger.info(f"\nNetwork Model:")
        logger.info(f"  Loaded: {info['network_model']['loaded']}")
        logger.info(f"  Classes: {len(info['network_model']['classes'])}")
        
        logger.info(f"\nWindows Specialist:")
        logger.info(f"  Loaded: {info['windows_specialist']['loaded']}")
        logger.info(f"  Classes: {len(info['windows_specialist']['classes'])}")
        
        logger.info(f"\nDevice: {info['device']}")
        logger.info(f"Ensemble Mode: {info['ensemble_mode']}")
        
        logger.info("\n✅ Model info test passed")
        return True
    
    except Exception as e:
        logger.error(f"❌ Model info test failed: {e}")
        return False


def main():
    """Run all tests"""
    logger.info("\n")
    logger.info("=" * 70)
    logger.info("🚀 WINDOWS 13-CLASS INTEGRATION TESTS")
    logger.info("=" * 70)
    
    tests = [
        ("Model Loading", test_model_loading),
        ("Inference", test_inference),
        ("Model Info", test_model_info)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            logger.error(f"Test '{name}' crashed: {e}")
            results.append((name, False))
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("📊 TEST SUMMARY")
    logger.info("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"  {status}: {name}")
    
    logger.info(f"\n{passed}/{total} tests passed")
    
    if passed == total:
        logger.info("\n🎉 All tests passed!")
        return 0
    else:
        logger.error(f"\n❌ {total - passed} test(s) failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())


#!/usr/bin/env python3
"""
Test the comprehensive 7-class attack detection model integration
"""

import sys
import os
import torch
import numpy as np
sys.path.append('/Users/chasemad/Desktop/mini-xdr/backend')

from app.deep_learning_models import DeepLearningModelManager

def test_comprehensive_model():
    """Test the comprehensive attack detection model"""

    print("🔍 Testing Comprehensive Attack Detection Model")
    print("=" * 60)

    # Initialize the deep learning model manager
    try:
        manager = DeepLearningModelManager()
        print("✅ Deep learning manager initialized")
    except Exception as e:
        print(f"❌ Failed to initialize manager: {e}")
        return False

    # Load models
    try:
        model_dir = "/Users/chasemad/Desktop/mini-xdr/models"
        results = manager.load_models(model_dir)
        print(f"📊 Model loading results: {results}")

        if not results.get('threat_detector', False):
            print("❌ Threat detector failed to load")
            return False

        print("✅ Models loaded successfully")

    except Exception as e:
        print(f"❌ Failed to load models: {e}")
        return False

    # Test with synthetic attack samples
    test_cases = [
        {
            'name': 'Normal Traffic',
            'features': np.random.normal(0.1, 0.05, 79),  # Low activity
            'expected_class': 0
        },
        {
            'name': 'DDoS Attack Pattern',
            'features': create_ddos_pattern(),
            'expected_class': 1
        },
        {
            'name': 'Port Scan Pattern',
            'features': create_scan_pattern(),
            'expected_class': 2
        },
        {
            'name': 'Brute Force Pattern',
            'features': create_bruteforce_pattern(),
            'expected_class': 3
        },
        {
            'name': 'Web Attack Pattern',
            'features': create_web_attack_pattern(),
            'expected_class': 4
        }
    ]

    print("\n🧪 Testing Attack Classification:")
    print("-" * 40)

    for test_case in test_cases:
        try:
            # Create mock event objects (simplified)
            import asyncio
            from types import SimpleNamespace

            mock_events = [SimpleNamespace(**{'src_ip': '192.168.1.100', 'data': test_case['features']})]

            # Get prediction using async method
            result = asyncio.run(manager.calculate_threat_score('192.168.1.100', mock_events))

            print(f"\n📊 {test_case['name']}:")
            print(f"   Ensemble Score: {result.get('ensemble_score', 'N/A'):.3f}")
            print(f"   Confidence: {result.get('confidence', 'N/A'):.3f}")
            print(f"   Attack Type: {result.get('attack_type', 'N/A')}")
            print(f"   Attack Confidence: {result.get('attack_confidence', 'N/A'):.3f}")
            print(f"   Threat Probability: {result.get('threat_detector', 'N/A'):.3f}")

        except Exception as e:
            print(f"❌ {test_case['name']} failed: {e}")
            return False

    print("\n🎉 All tests completed successfully!")
    return True

def create_ddos_pattern():
    """Create synthetic DDoS attack pattern"""
    features = np.random.normal(0.1, 0.05, 79)

    # High packet rates and bandwidth (typical DDoS signatures)
    features[1] = np.random.uniform(1000, 5000)    # total_fwd_packets
    features[11] = np.random.uniform(100, 1000)    # flow_bytes_s
    features[12] = np.random.uniform(50, 500)      # flow_packets_s
    features[38] = np.random.uniform(10, 50)       # syn_flag_count

    return features

def create_scan_pattern():
    """Create synthetic port scanning pattern"""
    features = np.random.normal(0.1, 0.05, 79)

    # Low packets, high connection attempts
    features[1] = np.random.uniform(1, 10)         # total_fwd_packets (low)
    features[2] = np.random.uniform(0, 5)          # total_backward_packets (very low)
    features[38] = np.random.uniform(1, 5)         # syn_flag_count
    features[40] = np.random.uniform(1, 3)         # rst_flag_count

    return features

def create_bruteforce_pattern():
    """Create synthetic brute force attack pattern"""
    features = np.random.normal(0.1, 0.05, 79)

    # Many connection attempts, small packets
    features[1] = np.random.uniform(50, 200)       # total_fwd_packets
    features[2] = np.random.uniform(20, 100)       # total_backward_packets
    features[11] = np.random.uniform(0.1, 2.0)     # flow_bytes_s (low)
    features[25] = np.random.uniform(5, 20)        # fwd_psh_flags

    return features

def create_web_attack_pattern():
    """Create synthetic web attack pattern"""
    features = np.random.normal(0.1, 0.05, 79)

    # HTTP-based attack patterns
    features[3] = np.random.uniform(500, 2000)     # total_length_of_fwd_packets
    features[33] = np.random.uniform(100, 500)     # packet_length_mean
    features[25] = np.random.uniform(1, 10)        # fwd_psh_flags
    features[42] = np.random.uniform(5, 20)        # ack_flag_count

    return features

if __name__ == "__main__":
    success = test_comprehensive_model()
    sys.exit(0 if success else 1)
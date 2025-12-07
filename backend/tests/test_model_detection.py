#!/usr/bin/env python3
"""
Quick test to verify enhanced ML models are working for attack detection
"""
import sys
import asyncio
import numpy as np
sys.path.append('./aws')

from local_inference import local_ml_client

async def test_model_detection():
    """Test that models can detect various attack types"""
    print("🔍 Testing Enhanced ML Model Detection")
    print("=" * 60)
    
    # Check if models are loaded
    print("\n1. Checking model health...")
    health = await local_ml_client.health_check()
    if health:
        print(f"✅ Models loaded successfully!")
        print(f"   Available models: {list(local_ml_client.models.keys())}")
        for model_type, metadata in local_ml_client.metadata.items():
            acc = metadata.get('best_val_accuracy', 0) * 100
            print(f"   - {model_type}: {acc:.2f}% accuracy")
    else:
        print("❌ Models not loaded!")
        return False
    
    # Create test attack patterns
    print("\n2. Testing attack detection...")
    
    test_cases = [
        {
            'name': 'DDoS Attack Pattern',
            'event': {
                'src_ip': '192.168.1.100',
                'dst_port': 80,
                'eventid': 'test_ddos',
                'message': 'High volume TCP SYN packets',
                'timestamp': '2025-10-05T10:00:00',
                'raw': {},
                'features': create_ddos_pattern()
            },
            'expected': 'DDoS'
        },
        {
            'name': 'Brute Force Attack Pattern',
            'event': {
                'src_ip': '192.168.1.101',
                'dst_port': 22,
                'eventid': 'test_brute',
                'message': 'Multiple failed SSH login attempts',
                'timestamp': '2025-10-05T10:00:00',
                'raw': {},
                'features': create_bruteforce_pattern()
            },
            'expected': 'Brute Force'
        },
        {
            'name': 'Web Attack Pattern',
            'event': {
                'src_ip': '192.168.1.102',
                'dst_port': 443,
                'eventid': 'test_web',
                'message': 'SQL injection attempt detected',
                'timestamp': '2025-10-05T10:00:00',
                'raw': {},
                'features': create_web_attack_pattern()
            },
            'expected': 'Web Attack'
        },
        {
            'name': 'Normal Traffic Pattern',
            'event': {
                'src_ip': '192.168.1.103',
                'dst_port': 443,
                'eventid': 'test_normal',
                'message': 'Normal HTTPS request',
                'timestamp': '2025-10-05T10:00:00',
                'raw': {},
                'features': create_normal_pattern()
            },
            'expected': 'Normal'
        }
    ]
    
    for test in test_cases:
        print(f"\n   Testing: {test['name']}")
        try:
            result = await local_ml_client.detect_threats([test['event']])
            if result and len(result) > 0:
                detection = result[0]
                predicted = detection.get('predicted_class', 'Unknown')
                confidence = detection.get('confidence', 0) * 100
                score = detection.get('anomaly_score', 0)
                
                is_correct = predicted.lower() == test['expected'].lower()
                status = "✅" if is_correct else "⚠️"
                
                print(f"   {status} Predicted: {predicted} (confidence: {confidence:.1f}%, score: {score:.3f})")
                print(f"      Expected: {test['expected']}")
            else:
                print(f"   ❌ No detection result")
        except Exception as e:
            print(f"   ❌ Detection failed: {e}")
    
    print("\n" + "=" * 60)
    print("✅ Model detection test complete!")
    return True


def create_ddos_pattern():
    """Create synthetic features representing a DDoS attack"""
    features = np.zeros(79, dtype=np.float32)
    # High packet rate features
    features[0] = 0.95  # Very high packet rate
    features[1] = 0.92  # High bytes per second
    features[2] = 0.88  # High flow count
    features[5] = 0.05  # Low average packet size (SYN flood)
    features[10] = 0.98  # Very high connection rate
    # Add some random noise
    features += np.random.normal(0, 0.02, 79)
    return features.clip(0, 1).tolist()


def create_bruteforce_pattern():
    """Create synthetic features representing a brute force attack"""
    features = np.zeros(79, dtype=np.float32)
    # High failed login features
    features[3] = 0.95  # Very high failed login rate
    features[15] = 0.90  # Multiple authentication attempts
    features[18] = 0.85  # Password variation
    features[25] = 0.02  # Short session duration (failed logins)
    features[30] = 0.92  # High unique username count
    # Add some random noise
    features += np.random.normal(0, 0.02, 79)
    return features.clip(0, 1).tolist()


def create_web_attack_pattern():
    """Create synthetic features representing a web attack"""
    features = np.zeros(79, dtype=np.float32)
    # Web attack characteristics
    features[7] = 0.90  # High HTTP request rate
    features[12] = 0.85  # Unusual URL patterns
    features[20] = 0.92  # SQL injection indicators
    features[22] = 0.88  # XSS patterns
    features[28] = 0.80  # Suspicious user agent
    # Add some random noise
    features += np.random.normal(0, 0.02, 79)
    return features.clip(0, 1).tolist()


def create_normal_pattern():
    """Create synthetic features representing normal traffic"""
    features = np.random.normal(0.2, 0.1, 79).astype(np.float32)
    return features.clip(0, 1).tolist()


if __name__ == "__main__":
    asyncio.run(test_model_detection())



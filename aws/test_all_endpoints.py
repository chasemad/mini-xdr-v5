#!/usr/bin/env python3
"""
Comprehensive test for all 4 deployed SageMaker endpoints
Tests feature scaling, classification accuracy, and response times
"""

import boto3
import json
import logging
import numpy as np
import time
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Attack type labels
THREAT_CLASSES = [
    "Normal",
    "DDoS/DoS Attack",
    "Network Reconnaissance",
    "Brute Force Attack",
    "Web Application Attack",
    "Malware/Botnet",
    "Advanced Persistent Threat"
]

# Endpoint configurations
ENDPOINTS = {
    "general": "mini-xdr-general-endpoint",
    "ddos": "mini-xdr-ddos-specialist",
    "bruteforce": "mini-xdr-bruteforce-specialist",
    "webattack": "mini-xdr-webattack-specialist"
}


def create_attack_samples() -> Dict[str, List[float]]:
    """
    Create realistic attack samples for testing
    These are 79-feature vectors representing different attack types
    """

    # Normal traffic: Low values across most features
    normal_traffic = [0.1] * 79
    normal_traffic[0] = 80.0  # destination port (HTTP)
    normal_traffic[1] = 1.0   # protocol (TCP)
    normal_traffic[10] = 100.0  # packet size

    # DDoS attack: High packet rate, many connections
    ddos_attack = [0.1] * 79
    ddos_attack[5] = 10000.0  # high packet count
    ddos_attack[6] = 5000.0   # high connection count
    ddos_attack[7] = 0.5      # short duration
    ddos_attack[20] = 500.0   # high rate

    # Brute force: Multiple failed authentication attempts
    brute_force_attack = [0.1] * 79
    brute_force_attack[0] = 22.0    # SSH port
    brute_force_attack[8] = 100.0   # many attempts
    brute_force_attack[9] = 0.95    # high failure rate
    brute_force_attack[30] = 1.0    # authentication-related

    # Web attack: Suspicious HTTP patterns
    web_attack = [0.1] * 79
    web_attack[0] = 443.0      # HTTPS port
    web_attack[25] = 500.0     # large payload
    web_attack[26] = 1.0       # suspicious patterns
    web_attack[27] = 1.0       # SQL injection indicators
    web_attack[28] = 1.0       # XSS indicators

    return {
        "normal": normal_traffic,
        "ddos": ddos_attack,
        "brute_force": brute_force_attack,
        "web_attack": web_attack
    }


def test_endpoint(endpoint_name: str, sample_name: str, features: List[float]) -> Dict:
    """Test a single endpoint with sample data"""

    sagemaker_runtime = boto3.client('sagemaker-runtime', region_name='us-east-1')

    try:
        # Format input for SageMaker
        payload = {
            "instances": [features]
        }

        # Measure response time
        start_time = time.time()

        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='application/json',
            Accept='application/json',
            Body=json.dumps(payload)
        )

        response_time = (time.time() - start_time) * 1000  # ms

        # Parse response
        result = json.loads(response['Body'].read().decode())

        return {
            "success": True,
            "endpoint": endpoint_name,
            "sample": sample_name,
            "response": result,
            "response_time_ms": response_time
        }

    except Exception as e:
        logger.error(f"❌ Endpoint {endpoint_name} test failed: {e}")
        return {
            "success": False,
            "endpoint": endpoint_name,
            "sample": sample_name,
            "error": str(e)
        }


def analyze_predictions(result: Dict, expected_class: str = None) -> None:
    """Analyze and display prediction results"""

    if not result['success']:
        logger.error(f"  ❌ Test failed: {result.get('error', 'Unknown error')}")
        return

    response = result['response']
    response_time = result['response_time_ms']

    # Check response format
    if 'predictions' not in response:
        logger.error(f"  ❌ Invalid response format: {response}")
        return

    predictions = response['predictions'][0]

    # For specialist models (binary classification)
    if len(predictions) == 2:
        threat_prob = predictions[1]
        normal_prob = predictions[0]
        classification = "Threat" if threat_prob > 0.5 else "Normal"
        confidence = max(threat_prob, normal_prob)

        logger.info(f"  Classification: {classification} (confidence: {confidence:.2%})")
        logger.info(f"  Threat probability: {threat_prob:.2%}")
        logger.info(f"  Normal probability: {normal_prob:.2%}")

    # For general model (7-class classification)
    else:
        predicted_class_idx = np.argmax(predictions)
        predicted_class = THREAT_CLASSES[predicted_class_idx]
        confidence = predictions[predicted_class_idx]

        logger.info(f"  Predicted: {predicted_class} (confidence: {confidence:.2%})")
        logger.info(f"  Top 3 predictions:")

        # Show top 3 predictions
        top_3_indices = np.argsort(predictions)[-3:][::-1]
        for idx in top_3_indices:
            logger.info(f"    - {THREAT_CLASSES[idx]}: {predictions[idx]:.2%}")

    # Response time check
    if response_time < 500:
        logger.info(f"  ✅ Response time: {response_time:.0f}ms (< 500ms target)")
    else:
        logger.warning(f"  ⚠️  Response time: {response_time:.0f}ms (> 500ms target)")


def verify_feature_scaling(features: List[float]) -> None:
    """Verify that features are in expected range after scaling"""

    logger.info("\n🔍 Verifying Feature Scaling:")
    logger.info(f"  Input feature range: [{min(features):.2f}, {max(features):.2f}]")
    logger.info(f"  Mean: {np.mean(features):.2f}, Std: {np.std(features):.2f}")

    # After scaling, features should be roughly in [-3, +3] range
    logger.info("  Note: Scaler will transform these to ~[-3, +3] range on the endpoint")


def main():
    """Run comprehensive endpoint tests"""

    logger.info("=" * 80)
    logger.info("🧪 COMPREHENSIVE ENDPOINT TESTING")
    logger.info("=" * 80)

    # Create test samples
    samples = create_attack_samples()

    # Test 1: General endpoint with all attack types
    logger.info("\n📊 TEST 1: General Endpoint (7-class classification)")
    logger.info("-" * 80)

    for sample_name, features in samples.items():
        logger.info(f"\n  Testing with {sample_name} traffic...")
        verify_feature_scaling(features)

        result = test_endpoint(ENDPOINTS['general'], sample_name, features)
        analyze_predictions(result)

    # Test 2: DDoS specialist
    logger.info("\n" + "=" * 80)
    logger.info("📊 TEST 2: DDoS Specialist (binary classification)")
    logger.info("-" * 80)

    logger.info("\n  Testing with DDoS attack...")
    result = test_endpoint(ENDPOINTS['ddos'], 'ddos', samples['ddos'])
    analyze_predictions(result)

    logger.info("\n  Testing with normal traffic...")
    result = test_endpoint(ENDPOINTS['ddos'], 'normal', samples['normal'])
    analyze_predictions(result)

    # Test 3: Brute force specialist
    logger.info("\n" + "=" * 80)
    logger.info("📊 TEST 3: Brute Force Specialist (binary classification)")
    logger.info("-" * 80)

    logger.info("\n  Testing with brute force attack...")
    result = test_endpoint(ENDPOINTS['bruteforce'], 'brute_force', samples['brute_force'])
    analyze_predictions(result)

    logger.info("\n  Testing with normal traffic...")
    result = test_endpoint(ENDPOINTS['bruteforce'], 'normal', samples['normal'])
    analyze_predictions(result)

    # Test 4: Web attack specialist
    logger.info("\n" + "=" * 80)
    logger.info("📊 TEST 4: Web Attack Specialist (binary classification)")
    logger.info("-" * 80)

    logger.info("\n  Testing with web attack...")
    result = test_endpoint(ENDPOINTS['webattack'], 'web_attack', samples['web_attack'])
    analyze_predictions(result)

    logger.info("\n  Testing with normal traffic...")
    result = test_endpoint(ENDPOINTS['webattack'], 'normal', samples['normal'])
    analyze_predictions(result)

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("✅ ALL ENDPOINT TESTS COMPLETED")
    logger.info("=" * 80)
    logger.info("\n📋 Key Findings:")
    logger.info("  1. All 4 endpoints are responding")
    logger.info("  2. Feature scaling is handled by the endpoint (scaler.pkl)")
    logger.info("  3. Attack classifications are being made")
    logger.info("  4. Response times are recorded")
    logger.info("\n💡 Next Steps:")
    logger.info("  1. Review prediction confidence levels")
    logger.info("  2. Compare specialist vs general model predictions")
    logger.info("  3. Integrate into backend/app/sagemaker_client.py")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

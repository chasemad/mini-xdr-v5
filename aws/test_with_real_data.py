#!/usr/bin/env python3
"""
Test endpoints with actual training data samples
"""

import boto3
import json
import logging
import numpy as np
import pandas as pd
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

THREAT_CLASSES = [
    "Normal",
    "DDoS/DoS Attack",
    "Network Reconnaissance",
    "Brute Force Attack",
    "Web Application Attack",
    "Malware/Botnet",
    "Advanced Persistent Threat"
]

ENDPOINTS = {
    "general": "mini-xdr-general-endpoint",
    "ddos": "mini-xdr-ddos-specialist",
    "bruteforce": "mini-xdr-bruteforce-specialist",
    "webattack": "mini-xdr-webattack-specialist"
}


def load_sample_data():
    """Load actual samples from training data"""

    # Load the npy files
    features_path = Path("aws/training_data/training_features_20250929_062520.npy")
    labels_path = Path("aws/training_data/training_labels_20250929_062520.npy")

    if not features_path.exists():
        logger.error(f"Features file not found: {features_path}")
        return None

    logger.info("Loading training data...")
    features = np.load(features_path, mmap_mode='r')  # Memory-mapped for large files
    labels = np.load(labels_path)

    logger.info(f"Loaded {len(features)} samples with {features.shape[1]} features")

    # Get 5 samples from each class
    samples_by_class = {}

    for class_idx in range(7):
        # Find samples for this class
        class_indices = np.where(labels == class_idx)[0]

        if len(class_indices) > 0:
            # Get 5 random samples
            sample_indices = np.random.choice(class_indices, min(5, len(class_indices)), replace=False)
            samples_by_class[class_idx] = [features[i].tolist() for i in sample_indices]
            logger.info(f"  Class {class_idx} ({THREAT_CLASSES[class_idx]}): {len(class_indices)} samples, using {len(sample_indices)}")

    return samples_by_class


def test_endpoint(endpoint_name: str, features: list, expected_class: str = None):
    """Test an endpoint with real data"""

    sagemaker_runtime = boto3.client('sagemaker-runtime', region_name='us-east-1')

    try:
        payload = {
            "instances": [features]
        }

        start_time = time.time()

        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='application/json',
            Accept='application/json',
            Body=json.dumps(payload)
        )

        response_time = (time.time() - start_time) * 1000

        result = json.loads(response['Body'].read().decode())
        predictions = result['predictions'][0]

        return {
            "success": True,
            "predictions": predictions,
            "response_time_ms": response_time,
            "expected_class": expected_class
        }

    except Exception as e:
        logger.error(f"Test failed: {e}")
        return {"success": False, "error": str(e)}


def main():
    """Run tests with real training data"""

    logger.info("=" * 80)
    logger.info("🧪 TESTING WITH REAL TRAINING DATA")
    logger.info("=" * 80)

    # Load samples
    samples = load_sample_data()

    if not samples:
        logger.error("Failed to load sample data")
        return

    # Test General Endpoint with all classes
    logger.info("\n" + "=" * 80)
    logger.info("📊 TEST 1: General Endpoint (7-class classification)")
    logger.info("=" * 80)

    correct = 0
    total = 0

    for class_idx, class_samples in samples.items():
        class_name = THREAT_CLASSES[class_idx]
        logger.info(f"\n  Testing {class_name} samples...")

        for i, sample in enumerate(class_samples[:3], 1):  # Test 3 samples per class
            result = test_endpoint(ENDPOINTS['general'], sample, class_name)

            if result['success']:
                predictions = result['predictions']
                predicted_idx = np.argmax(predictions)
                predicted_class = THREAT_CLASSES[predicted_idx]
                confidence = predictions[predicted_idx]

                is_correct = predicted_idx == class_idx
                correct += is_correct
                total += 1

                status = "✅" if is_correct else "❌"
                logger.info(f"    Sample {i}: {status} Predicted: {predicted_class} ({confidence:.1%}) | Expected: {class_name}")

    accuracy = (correct / total * 100) if total > 0 else 0
    logger.info(f"\n  📊 General Model Accuracy: {correct}/{total} = {accuracy:.1f}%")

    # Test DDoS Specialist
    logger.info("\n" + "=" * 80)
    logger.info("📊 TEST 2: DDoS Specialist")
    logger.info("=" * 80)

    if 1 in samples:  # DDoS class
        logger.info("\n  Testing DDoS attacks...")
        for i, sample in enumerate(samples[1][:3], 1):
            result = test_endpoint(ENDPOINTS['ddos'], sample, "DDoS")
            if result['success']:
                threat_prob = result['predictions'][1]
                classification = "Threat" if threat_prob > 0.5 else "Normal"
                status = "✅" if threat_prob > 0.5 else "❌"
                logger.info(f"    Sample {i}: {status} {classification} (threat: {threat_prob:.1%})")

    if 0 in samples:  # Normal class
        logger.info("\n  Testing Normal traffic...")
        for i, sample in enumerate(samples[0][:3], 1):
            result = test_endpoint(ENDPOINTS['ddos'], sample, "Normal")
            if result['success']:
                normal_prob = result['predictions'][0]
                classification = "Normal" if normal_prob > 0.5 else "Threat"
                status = "✅" if normal_prob > 0.5 else "❌"
                logger.info(f"    Sample {i}: {status} {classification} (normal: {normal_prob:.1%})")

    # Test Brute Force Specialist
    logger.info("\n" + "=" * 80)
    logger.info("📊 TEST 3: Brute Force Specialist")
    logger.info("=" * 80)

    if 3 in samples:  # Brute Force class
        logger.info("\n  Testing Brute Force attacks...")
        for i, sample in enumerate(samples[3][:3], 1):
            result = test_endpoint(ENDPOINTS['bruteforce'], sample, "Brute Force")
            if result['success']:
                threat_prob = result['predictions'][1]
                classification = "Threat" if threat_prob > 0.5 else "Normal"
                status = "✅" if threat_prob > 0.5 else "❌"
                logger.info(f"    Sample {i}: {status} {classification} (threat: {threat_prob:.1%})")

    if 0 in samples:
        logger.info("\n  Testing Normal traffic...")
        for i, sample in enumerate(samples[0][:3], 1):
            result = test_endpoint(ENDPOINTS['bruteforce'], sample, "Normal")
            if result['success']:
                normal_prob = result['predictions'][0]
                classification = "Normal" if normal_prob > 0.5 else "Threat"
                status = "✅" if normal_prob > 0.5 else "❌"
                logger.info(f"    Sample {i}: {status} {classification} (normal: {normal_prob:.1%})")

    # Test Web Attack Specialist
    logger.info("\n" + "=" * 80)
    logger.info("📊 TEST 4: Web Attack Specialist")
    logger.info("=" * 80)

    if 4 in samples:  # Web Attack class
        logger.info("\n  Testing Web Application attacks...")
        for i, sample in enumerate(samples[4][:3], 1):
            result = test_endpoint(ENDPOINTS['webattack'], sample, "Web Attack")
            if result['success']:
                threat_prob = result['predictions'][1]
                classification = "Threat" if threat_prob > 0.5 else "Normal"
                status = "✅" if threat_prob > 0.5 else "❌"
                logger.info(f"    Sample {i}: {status} {classification} (threat: {threat_prob:.1%})")

    if 0 in samples:
        logger.info("\n  Testing Normal traffic...")
        for i, sample in enumerate(samples[0][:3], 1):
            result = test_endpoint(ENDPOINTS['webattack'], sample, "Normal")
            if result['success']:
                normal_prob = result['predictions'][0]
                classification = "Normal" if normal_prob > 0.5 else "Threat"
                status = "✅" if normal_prob > 0.5 else "❌"
                logger.info(f"    Sample {i}: {status} {classification} (normal: {normal_prob:.1%})")

    logger.info("\n" + "=" * 80)
    logger.info("✅ TESTING COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

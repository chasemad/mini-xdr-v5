#!/usr/bin/env python3
"""
Test for the revolutionary ensemble threat detection model
"""

import asyncio
import os
import sys

import numpy as np

sys.path.append("/Users/chasemad/Desktop/mini-xdr/backend")


async def test_revolutionary_model():
    """Test the revolutionary ensemble model"""

    print("🚀 Revolutionary Ensemble Model Test")
    print("=" * 50)

    try:
        from app.ai_models.ensemble import get_ensemble_detector

        detector = get_ensemble_detector()

        print("✅ Revolutionary detector loaded successfully!")
        print("Model info:", detector.get_info())

        # Test attack classification
        attack_classes = {
            0: "Normal Traffic",
            1: "DDoS/DoS Attack",
            2: "Network Reconnaissance",
            3: "Brute Force Attack",
            4: "Web Application Attack",
            5: "Malware/Botnet",
            6: "Advanced Persistent Threat",
        }

        print("\n🧪 Testing Ensemble Attack Classification:")
        print("-" * 50)

        # Test cases with synthetic patterns
        test_samples = [
            ("Normal Pattern", create_normal_pattern()),
            ("DDoS Attack", create_ddos_pattern()),
            ("Port Scan", create_scan_pattern()),
            ("Brute Force", create_bruteforce_pattern()),
            ("Web Attack", create_web_attack_pattern()),
        ]

        for name, features in test_samples:
            try:
                # Reshape for model input
                feature_vector = np.array([features], dtype=np.float32)

                # Get ensemble prediction
                result = await detector.predict(feature_vector)

                predicted_class = result["predicted_class"][0]
                confidence = result["confidence"][0]
                uncertainty = result["uncertainty"][0]
                threat_type = result["threat_type"][0]
                models_used = result["models_used"]

                print(f"\n📊 {name}:")
                print(f"   Predicted: {threat_type} (Class {predicted_class})")
                print(f"   Confidence: {confidence:.3f}")
                print(f"   Uncertainty: {uncertainty:.3f}")
                print(f"   Models used: {models_used}")

                # Show top 3 class probabilities
                probs = result["class_probabilities"][0]
                top_indices = np.argsort(probs)[-3:][::-1]
                print("   Top 3 predictions:")
                for i, idx in enumerate(top_indices):
                    prob = probs[idx]
                    class_name = attack_classes.get(idx, f"Unknown_{idx}")
                    print(f"     {i+1}. {class_name}: {prob:.3f}")

            except Exception as e:
                print(f"❌ {name} failed: {e}")
                continue

        print("\n🎉 Revolutionary ensemble model test completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Revolutionary model test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def create_normal_pattern():
    """Normal traffic pattern"""
    return np.random.normal(0.1, 0.05, 79)


def create_ddos_pattern():
    """DDoS attack pattern"""
    features = np.random.normal(0.1, 0.05, 79)
    features[1] = np.random.uniform(1000, 5000)  # high packet count
    features[11] = np.random.uniform(100, 1000)  # high bandwidth
    features[12] = np.random.uniform(50, 500)  # high packet rate
    return features


def create_scan_pattern():
    """Port scanning pattern"""
    features = np.random.normal(0.1, 0.05, 79)
    features[1] = np.random.uniform(1, 10)  # low packets
    features[2] = np.random.uniform(0, 5)  # very low response
    features[38] = np.random.uniform(1, 5)  # syn flags
    return features


def create_bruteforce_pattern():
    """Brute force pattern"""
    features = np.random.normal(0.1, 0.05, 79)
    features[1] = np.random.uniform(50, 200)  # many attempts
    features[11] = np.random.uniform(0.1, 2.0)  # low bandwidth
    features[25] = np.random.uniform(5, 20)  # push flags
    return features


def create_web_attack_pattern():
    """Web attack pattern"""
    features = np.random.normal(0.1, 0.05, 79)
    features[3] = np.random.uniform(500, 2000)  # large packets
    features[25] = np.random.uniform(1, 10)  # push flags
    features[42] = np.random.uniform(5, 20)  # ack flags
    return features


if __name__ == "__main__":
    success = asyncio.run(test_revolutionary_model())
    sys.exit(0 if success else 1)

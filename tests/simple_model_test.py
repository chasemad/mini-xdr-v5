#!/usr/bin/env python3
"""
Simple test for the comprehensive 7-class threat detection model
"""

import sys
import os
import torch
import numpy as np
sys.path.append('./backend')

from app.deep_learning_models import XDRThreatDetector
import joblib

def test_model_directly():
    """Test the model directly without the manager wrapper"""

    print("🔍 Direct Model Test - 7-Class Attack Detection")
    print("=" * 55)

    # Load model metadata
    try:
        with open('./models/model_metadata.json', 'r') as f:
            import json
            metadata = json.load(f)

        print(f"📊 Model Info:")
        print(f"   Classes: {metadata.get('num_classes', 'N/A')}")
        print(f"   Features: {metadata.get('features', 'N/A')}")
        print(f"   Accuracy: {metadata.get('best_accuracy', 'N/A'):.4f}")
        print(f"   Samples Trained: {metadata.get('total_samples', 'N/A'):,}")

    except Exception as e:
        print(f"⚠️  Could not load metadata: {e}")
        metadata = {'num_classes': 7, 'features': 79}

    # Load scaler
    try:
        scaler = joblib.load('./models/scaler.pkl')
        print("✅ Scaler loaded successfully")
    except Exception as e:
        print(f"❌ Scaler failed: {e}")
        return False

    # Initialize model with correct architecture
    try:
        model = XDRThreatDetector(
            input_dim=79,
            hidden_dims=[256, 128, 64],  # Small model architecture
            num_classes=7,               # 7-class model
            dropout_rate=0.2
        )

        # Load trained weights
        model.load_state_dict(torch.load(
            './models/threat_detector.pth',
            map_location='cpu',
            weights_only=True
        ))
        model.eval()

        print("✅ Threat detector loaded successfully")

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"📊 Model parameters: {total_params:,}")

    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

    # Test attack classification
    attack_classes = {
        0: "Normal Traffic",
        1: "DDoS/DoS Attack",
        2: "Network Reconnaissance",
        3: "Brute Force Attack",
        4: "Web Application Attack",
        5: "Malware/Botnet",
        6: "Advanced Persistent Threat"
    }

    print("\n🧪 Testing Attack Classification:")
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
            # Scale features
            features_scaled = scaler.transform([features])
            input_tensor = torch.tensor(features_scaled, dtype=torch.float32)

            # Get prediction
            with torch.no_grad():
                logits = model(input_tensor)
                probabilities = torch.softmax(logits, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0, predicted_class].item()
                normal_prob = probabilities[0, 0].item()
                threat_prob = 1.0 - normal_prob

            print(f"\n📊 {name}:")
            print(f"   Predicted: {attack_classes.get(predicted_class, 'Unknown')} (Class {predicted_class})")
            print(f"   Confidence: {confidence:.3f}")
            print(f"   Threat Probability: {threat_prob:.3f}")

            # Show top 3 class probabilities
            top_probs, top_indices = torch.topk(probabilities[0], 3)
            print("   Top 3 predictions:")
            for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
                print(f"     {i+1}. {attack_classes.get(idx.item(), 'Unknown')}: {prob.item():.3f}")

        except Exception as e:
            print(f"❌ {name} failed: {e}")
            return False

    print("\n🎉 Comprehensive model test completed successfully!")
    return True

def create_normal_pattern():
    """Normal traffic pattern"""
    return np.random.normal(0.1, 0.05, 79)

def create_ddos_pattern():
    """DDoS attack pattern"""
    features = np.random.normal(0.1, 0.05, 79)
    features[1] = np.random.uniform(1000, 5000)    # high packet count
    features[11] = np.random.uniform(100, 1000)    # high bandwidth
    features[12] = np.random.uniform(50, 500)      # high packet rate
    return features

def create_scan_pattern():
    """Port scanning pattern"""
    features = np.random.normal(0.1, 0.05, 79)
    features[1] = np.random.uniform(1, 10)         # low packets
    features[2] = np.random.uniform(0, 5)          # very low response
    features[38] = np.random.uniform(1, 5)         # syn flags
    return features

def create_bruteforce_pattern():
    """Brute force pattern"""
    features = np.random.normal(0.1, 0.05, 79)
    features[1] = np.random.uniform(50, 200)       # many attempts
    features[11] = np.random.uniform(0.1, 2.0)     # low bandwidth
    features[25] = np.random.uniform(5, 20)        # push flags
    return features

def create_web_attack_pattern():
    """Web attack pattern"""
    features = np.random.normal(0.1, 0.05, 79)
    features[3] = np.random.uniform(500, 2000)     # large packets
    features[25] = np.random.uniform(1, 10)        # push flags
    features[42] = np.random.uniform(5, 20)        # ack flags
    return features

if __name__ == "__main__":
    success = test_model_directly()
    sys.exit(0 if success else 1)
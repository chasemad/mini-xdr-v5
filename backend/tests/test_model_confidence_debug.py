#!/usr/bin/env python3
"""
DEBUG: Why is the model returning 57% confidence for everything?
Tests model loading, feature extraction, and inference pipeline
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

import torch
import numpy as np
import json
import logging
from typing import Dict, List
import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import from backend
from app.deep_learning_models import deep_learning_manager, XDRThreatDetector
from app.models import Event
from datetime import datetime, timedelta

class ModelConfidenceDebugger:
    """Debug model confidence issues"""
    
    def __init__(self):
        self.issues_found = []
        self.warnings = []
        self.model_dir = Path(__file__).parent.parent / "models"
        
    async def run_all_diagnostics(self):
        """Run all diagnostic tests"""
        print("\n" + "="*80)
        print("🔍 MODEL CONFIDENCE DEBUGGER")
        print("="*80 + "\n")
        
        # Test 1: Check model files
        print("📁 [1/7] Checking model files...")
        await self.test_model_files()
        
        # Test 2: Check model architecture
        print("\n🏗️  [2/7] Checking model architecture...")
        await self.test_model_architecture()
        
        # Test 3: Check feature extraction
        print("\n🔧 [3/7] Checking feature extraction...")
        await self.test_feature_extraction()
        
        # Test 4: Test with synthetic data
        print("\n🧪 [4/7] Testing with synthetic attack data...")
        await self.test_synthetic_attacks()
        
        # Test 5: Check scaler
        print("\n📊 [5/7] Checking feature scaling...")
        await self.test_feature_scaling()
        
        # Test 6: Check model weights
        print("\n⚖️  [6/7] Checking model weights...")
        await self.test_model_weights()
        
        # Test 7: Test with varied inputs
        print("\n🎯 [7/7] Testing with varied attack types...")
        await self.test_varied_attacks()
        
        # Print summary
        self.print_summary()
        
    async def test_model_files(self):
        """Check which model files exist"""
        models_to_check = [
            "threat_detector.pth",
            "anomaly_detector.pth", 
            "lstm_autoencoder.pth",
            "scaler.pkl",
            "local_trained_enhanced/general/threat_detector.pth",
            "local_trained_enhanced/general/model_metadata.json"
        ]
        
        for model_file in models_to_check:
            path = self.model_dir / model_file
            if path.exists():
                size = path.stat().st_size / (1024 * 1024)  # MB
                print(f"  ✅ {model_file} - {size:.2f} MB")
            else:
                print(f"  ❌ {model_file} - NOT FOUND")
                self.issues_found.append(f"Missing model file: {model_file}")
    
    async def test_model_architecture(self):
        """Check model architecture matches metadata"""
        metadata_path = self.model_dir / "local_trained_enhanced/general/model_metadata.json"
        
        if not metadata_path.exists():
            print("  ❌ Metadata file not found")
            self.issues_found.append("Missing metadata file")
            return
            
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        print(f"  📋 Model Metadata:")
        print(f"     - Features: {metadata.get('features', 'unknown')}")
        print(f"     - Hidden dims: {metadata.get('hidden_dims', 'unknown')}")
        print(f"     - Classes: {metadata.get('num_classes', 'unknown')}")
        print(f"     - Accuracy: {metadata.get('best_val_accuracy', 0)*100:.2f}%")
        print(f"     - Training date: {metadata.get('training_date', 'unknown')}")
        
        # Load actual model
        model_path = self.model_dir / "local_trained_enhanced/general/threat_detector.pth"
        if model_path.exists():
            try:
                # Try loading with current architecture
                features = metadata.get('features', 79)
                hidden_dims = metadata.get('hidden_dims', [512, 256, 128, 64])
                num_classes = metadata.get('num_classes', 7)
                
                model = XDRThreatDetector(
                    input_dim=features,
                    hidden_dims=hidden_dims,
                    num_classes=num_classes,
                    dropout_rate=0.3
                )
                
                state_dict = torch.load(model_path, map_location='cpu')
                model.load_state_dict(state_dict, strict=False)
                
                print(f"  ✅ Model loaded successfully")
                print(f"     - Parameters: {sum(p.numel() for p in model.parameters()):,}")
                
            except Exception as e:
                print(f"  ❌ Error loading model: {e}")
                self.issues_found.append(f"Model loading error: {e}")
        
    async def test_feature_extraction(self):
        """Test feature extraction from events"""
        # Create test events
        src_ip = "192.168.1.100"
        test_events = []
        
        base_time = datetime.utcnow()
        for i in range(10):
            event = Event(
                src_ip=src_ip,
                dst_ip="10.0.0.1",
                dst_port=22,
                eventid="cowrie.login.failed",
                message=f"Failed login attempt {i}",
                ts=base_time + timedelta(seconds=i*10),
                raw={"username": f"admin{i}", "password": "test123"}
            )
            test_events.append(event)
        
        # Extract features
        features = deep_learning_manager._extract_features(src_ip, test_events)
        
        print(f"  📊 Extracted {len(features)} features")
        print(f"     Sample features:")
        for i, (key, value) in enumerate(list(features.items())[:5]):
            print(f"       - {key}: {value:.4f}")
        
        # Check for constant features (could cause 57% issue)
        feature_values = list(features.values())
        if len(set(feature_values)) == 1:
            print(f"  ⚠️  WARNING: All features have the same value!")
            self.issues_found.append("All features are identical")
        
        # Check for NaN or inf
        if any(np.isnan(v) or np.isinf(v) for v in feature_values):
            print(f"  ❌ ERROR: Features contain NaN or Inf values")
            self.issues_found.append("Features contain invalid values")
        else:
            print(f"  ✅ No NaN or Inf values in features")
        
        return features
    
    async def test_synthetic_attacks(self):
        """Test with different synthetic attack patterns"""
        attack_scenarios = [
            {
                "name": "SSH Brute Force",
                "src_ip": "203.0.113.50",
                "events": [
                    {"eventid": "cowrie.login.failed", "dst_port": 22, "count": 20}
                ]
            },
            {
                "name": "DDoS Attack",
                "src_ip": "198.51.100.25",
                "events": [
                    {"eventid": "high_volume", "dst_port": 80, "count": 1000}
                ]
            },
            {
                "name": "Port Scan",
                "src_ip": "192.0.2.100",
                "events": [
                    {"eventid": "syn_scan", "dst_port": p, "count": 1} 
                    for p in range(20, 100, 5)
                ]
            }
        ]
        
        results = []
        
        for scenario in attack_scenarios:
            # Create events
            events = []
            base_time = datetime.utcnow()
            
            event_idx = 0
            for event_spec in scenario["events"]:
                for _ in range(event_spec.get("count", 1)):
                    event = Event(
                        src_ip=scenario["src_ip"],
                        dst_ip="10.0.0.1",
                        dst_port=event_spec.get("dst_port", 80),
                        eventid=event_spec["eventid"],
                        message=f"Attack event {event_idx}",
                        ts=base_time + timedelta(seconds=event_idx),
                        raw={}
                    )
                    events.append(event)
                    event_idx += 1
            
            # Get threat score
            score = await deep_learning_manager.calculate_threat_score(
                scenario["src_ip"], 
                events[:100]  # Limit to 100 events
            )
            
            confidence = score.get('attack_confidence', score.get('ensemble_score', 0.0))
            attack_type = score.get('attack_type', 'Unknown')
            
            print(f"\n  🎯 {scenario['name']}:")
            print(f"     IP: {scenario['src_ip']}")
            print(f"     Events: {len(events)}")
            print(f"     Confidence: {confidence*100:.2f}%")
            print(f"     Type: {attack_type}")
            
            results.append({
                "scenario": scenario['name'],
                "confidence": confidence,
                "type": attack_type
            })
            
            # Check if all confidences are the same
            if abs(confidence - 0.57) < 0.01:
                print(f"     ⚠️  WARNING: Confidence is ~57% (suspicious!)")
                self.warnings.append(f"{scenario['name']} returned 57% confidence")
        
        # Check for variance in results
        confidences = [r['confidence'] for r in results]
        variance = np.var(confidences)
        
        print(f"\n  📈 Confidence variance across attacks: {variance:.6f}")
        if variance < 0.001:
            print(f"  ❌ CRITICAL: All attacks return same confidence!")
            self.issues_found.append("Model returns constant confidence regardless of input")
        else:
            print(f"  ✅ Model shows variance in confidence")
    
    async def test_feature_scaling(self):
        """Check if feature scaling is causing issues"""
        scaler_path = self.model_dir / "scaler.pkl"
        
        if not scaler_path.exists():
            print("  ❌ Scaler file not found")
            self.issues_found.append("Missing scaler.pkl file")
            return
        
        import joblib
        scaler = joblib.load(scaler_path)
        
        print(f"  📊 Scaler type: {type(scaler).__name__}")
        
        if hasattr(scaler, 'mean_'):
            print(f"     - Mean shape: {scaler.mean_.shape}")
            print(f"     - Mean range: [{scaler.mean_.min():.4f}, {scaler.mean_.max():.4f}]")
        
        if hasattr(scaler, 'scale_'):
            print(f"     - Scale shape: {scaler.scale_.shape}")
            print(f"     - Scale range: [{scaler.scale_.min():.4f}, {scaler.scale_.max():.4f}]")
            
            # Check for zero scales
            zero_scales = (scaler.scale_ == 0).sum()
            if zero_scales > 0:
                print(f"     ⚠️  WARNING: {zero_scales} features have zero scale!")
                self.warnings.append(f"{zero_scales} features have zero variance")
        
        # Test scaling with dummy data
        dummy_features = np.random.randn(1, 79)
        try:
            scaled = scaler.transform(dummy_features)
            print(f"  ✅ Scaler works correctly")
            print(f"     - Scaled range: [{scaled.min():.4f}, {scaled.max():.4f}]")
        except Exception as e:
            print(f"  ❌ Scaler error: {e}")
            self.issues_found.append(f"Scaler transformation error: {e}")
    
    async def test_model_weights(self):
        """Check if model weights are properly loaded"""
        model_path = self.model_dir / "local_trained_enhanced/general/threat_detector.pth"
        
        if not model_path.exists():
            print("  ❌ Model file not found")
            return
        
        state_dict = torch.load(model_path, map_location='cpu')
        
        # Check weight statistics
        all_weights = []
        for name, tensor in state_dict.items():
            if 'weight' in name:
                weights = tensor.cpu().numpy().flatten()
                all_weights.extend(weights)
                
                print(f"  📊 {name}:")
                print(f"     - Shape: {tensor.shape}")
                print(f"     - Mean: {weights.mean():.6f}")
                print(f"     - Std: {weights.std():.6f}")
                print(f"     - Range: [{weights.min():.6f}, {weights.max():.6f}]")
        
        all_weights = np.array(all_weights)
        
        # Check for untrained weights (close to initialization)
        if abs(all_weights.mean()) < 0.01 and all_weights.std() < 0.1:
            print(f"  ⚠️  WARNING: Weights look like they might be untrained!")
            self.warnings.append("Model weights appear untrained")
        else:
            print(f"  ✅ Weights appear to be trained")
    
    async def test_varied_attacks(self):
        """Test with very different attack patterns to see if confidence varies"""
        test_cases = [
            {
                "name": "No Attack (Normal)",
                "event_count": 5,
                "expected_low": True
            },
            {
                "name": "Heavy Attack (1000 events)",
                "event_count": 1000,
                "expected_low": False
            },
            {
                "name": "Moderate Attack (50 events)",
                "event_count": 50,
                "expected_low": False
            }
        ]
        
        results = []
        
        for test in test_cases:
            src_ip = f"10.{np.random.randint(0,255)}.{np.random.randint(0,255)}.{np.random.randint(0,255)}"
            
            events = []
            base_time = datetime.utcnow()
            for i in range(test["event_count"]):
                event = Event(
                    src_ip=src_ip,
                    dst_ip="10.0.0.1",
                    dst_port=22,
                    eventid="cowrie.login.failed",
                    message=f"Event {i}",
                    ts=base_time + timedelta(seconds=i),
                    raw={}
                )
                events.append(event)
            
            score = await deep_learning_manager.calculate_threat_score(src_ip, events)
            confidence = score.get('attack_confidence', score.get('ensemble_score', 0.0))
            
            print(f"\n  🎯 {test['name']}:")
            print(f"     Events: {test['event_count']}")
            print(f"     Confidence: {confidence*100:.2f}%")
            
            results.append(confidence)
        
        # Check variance
        variance = np.var(results)
        print(f"\n  📈 Confidence variance: {variance:.6f}")
        
        if variance < 0.01:
            print(f"  ❌ CRITICAL: Confidence doesn't change with input!")
            self.issues_found.append("Model output is static regardless of input")
        else:
            print(f"  ✅ Model responds to different inputs")
    
    def print_summary(self):
        """Print diagnostic summary"""
        print("\n" + "="*80)
        print("📊 DIAGNOSTIC SUMMARY")
        print("="*80 + "\n")
        
        if not self.issues_found and not self.warnings:
            print("✅ No critical issues found!")
            print("\n💡 Possible reasons for 57% confidence:")
            print("  1. Model is correctly uncertain about ambiguous data")
            print("  2. Feature extraction needs tuning for your specific attacks")
            print("  3. Model needs retraining with more diverse data")
        else:
            if self.issues_found:
                print("❌ CRITICAL ISSUES:")
                for issue in self.issues_found:
                    print(f"  • {issue}")
                print()
            
            if self.warnings:
                print("⚠️  WARNINGS:")
                for warning in self.warnings:
                    print(f"  • {warning}")
                print()
        
        print("\n🔧 RECOMMENDED ACTIONS:")
        print("  1. Retrain model with fresh data: python aws/train_local.py")
        print("  2. Check feature extraction with real TPOT data")
        print("  3. Verify scaler was saved during training")
        print("  4. Test with actual honeypot events from TPOT")
        print("\n")

async def main():
    debugger = ModelConfidenceDebugger()
    await debugger.run_all_diagnostics()

if __name__ == "__main__":
    asyncio.run(main())


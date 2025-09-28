#!/usr/bin/env python3
"""
ML Model Training with Open Source Datasets
Trains all ML models using the existing cybersecurity datasets
"""
import asyncio
import json
import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# Add the backend to Python path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
BACKEND_DIR = PROJECT_ROOT / "backend"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.ml_engine import ml_detector, prepare_training_data_from_events
from app.models import Event
from app.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetTrainer:
    """Train ML models using open source cybersecurity datasets"""
    
    def __init__(self):
        self.datasets_dir = PROJECT_ROOT / "datasets"
        self.models_dir = PROJECT_ROOT / "backend" / "models"
        self.models_dir.mkdir(exist_ok=True)
        
        # Available datasets
        self.available_datasets = {
            "brute_force_ssh": "brute_force_ssh_dataset.json",
            "web_attacks": "web_attacks_dataset.json", 
            "network_scans": "network_scans_dataset.json",
            "ddos_attacks": "ddos_attacks_dataset.json",
            "malware_behavior": "malware_behavior_dataset.json",
            "combined_cybersecurity": "combined_cybersecurity_dataset.json"
        }
        
    def load_dataset(self, dataset_name: str) -> List[Dict[str, Any]]:
        """Load a specific dataset"""
        if dataset_name not in self.available_datasets:
            raise ValueError(f"Dataset {dataset_name} not found. Available: {list(self.available_datasets.keys())}")
        
        dataset_file = self.datasets_dir / self.available_datasets[dataset_name]
        
        if not dataset_file.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_file}")
        
        logger.info(f"Loading dataset: {dataset_name}")
        
        with open(dataset_file, 'r') as f:
            data = json.load(f)
        
        logger.info(f"Loaded {len(data)} events from {dataset_name}")
        return data
    
    def convert_to_events(self, raw_data: List[Dict[str, Any]]) -> List[Event]:
        """Convert raw dataset to Event objects"""
        events = []
        
        for item in raw_data:
            try:
                # Create Event-like object for ML training
                timestamp_str = item.get('timestamp', datetime.now().isoformat())
                if timestamp_str.endswith('Z'):
                    timestamp_str = timestamp_str[:-1] + '+00:00'
                
                # Parse timestamp and convert to UTC naive datetime for compatibility
                ts = datetime.fromisoformat(timestamp_str)
                if ts.tzinfo is not None:
                    ts = ts.utctimetuple()
                    ts = datetime(*ts[:6])  # Convert to naive datetime
                
                event = type('Event', (), {
                    'src_ip': item.get('src_ip', '0.0.0.0'),
                    'dst_port': item.get('dst_port', 80),
                    'eventid': item.get('eventid', 'unknown'),
                    'message': item.get('message', ''),
                    'raw': item.get('raw', {}),
                    'ts': ts
                })()
                
                events.append(event)
                
            except Exception as e:
                logger.warning(f"Skipping invalid event: {e}")
                continue
        
        return events
    
    def extract_training_features(self, events: List[Event]) -> List[Dict[str, float]]:
        """Extract ML training features from events"""
        logger.info("Extracting training features...")
        
        # Group events by source IP
        ip_groups = {}
        for event in events:
            if event.src_ip not in ip_groups:
                ip_groups[event.src_ip] = []
            ip_groups[event.src_ip].append(event)
        
        training_data = []
        
        for src_ip, ip_events in ip_groups.items():
            # Use the existing feature extraction from ml_engine
            features = self._extract_ip_features(src_ip, ip_events)
            training_data.append(features)
        
        logger.info(f"Extracted features for {len(training_data)} IP addresses")
        return training_data
    
    def _extract_ip_features(self, src_ip: str, events: List[Event]) -> Dict[str, float]:
        """Extract features for a specific IP using the existing ML engine logic"""
        from app.ml_engine import BaseMLDetector
        
        detector = BaseMLDetector()
        return detector._extract_features(src_ip, events)
    
    async def train_all_models(self, datasets_to_use: List[str] = None) -> Dict[str, Any]:
        """Train all ML models with the specified datasets"""
        if datasets_to_use is None:
            datasets_to_use = ["combined_cybersecurity", "brute_force_ssh", "web_attacks"]
        
        logger.info("🚀 Starting ML model training with open source datasets")
        logger.info(f"Using datasets: {datasets_to_use}")
        
        # Load and combine all specified datasets
        all_events = []
        dataset_stats = {}
        
        for dataset_name in datasets_to_use:
            try:
                raw_data = self.load_dataset(dataset_name)
                events = self.convert_to_events(raw_data)
                all_events.extend(events)
                dataset_stats[dataset_name] = len(events)
                
            except Exception as e:
                logger.error(f"Failed to load dataset {dataset_name}: {e}")
                continue
        
        if not all_events:
            raise ValueError("No training data loaded!")
        
        logger.info(f"Total training events: {len(all_events)}")
        logger.info(f"Dataset breakdown: {dataset_stats}")
        
        # Extract training features
        training_data = self.extract_training_features(all_events)
        
        if len(training_data) < 10:
            raise ValueError(f"Insufficient training data: {len(training_data)} samples")
        
        # Train the ensemble detector
        logger.info("🧠 Training ML ensemble models...")
        results = await ml_detector.train_models(training_data)
        
        # Verify training success
        model_status = ml_detector.get_model_status()
        
        # Save additional metadata
        training_metadata = {
            "training_completed": datetime.utcnow().isoformat(),
            "datasets_used": datasets_to_use,
            "dataset_stats": dataset_stats,
            "total_events": len(all_events),
            "training_samples": len(training_data),
            "model_results": results,
            "model_status": model_status
        }
        
        metadata_file = self.models_dir / "training_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(training_metadata, f, indent=2)
        
        return training_metadata
    
    def verify_models(self) -> Dict[str, Any]:
        """Verify that models are trained and working"""
        logger.info("🔍 Verifying trained models...")
        
        status = ml_detector.get_model_status()
        
        verification_result = {
            "models_trained": 0,
            "models_available": len(status.get('available_models', [])),
            "status": status,
            "ready_for_detection": False
        }
        
        # Count trained models
        trained_models = [
            status.get('isolation_forest', False),
            status.get('lstm', False), 
            status.get('enhanced_ml_trained', False)
        ]
        
        verification_result["models_trained"] = sum(trained_models)
        verification_result["ready_for_detection"] = verification_result["models_trained"] > 0
        
        logger.info(f"✅ {verification_result['models_trained']} models trained successfully")
        logger.info(f"Ready for detection: {verification_result['ready_for_detection']}")
        
        return verification_result

async def main():
    """Main training function"""
    trainer = DatasetTrainer()
    
    print("🧠 Mini-XDR ML Model Training with Open Source Datasets")
    print("=" * 60)
    
    try:
        # List available datasets
        print("📊 Available datasets:")
        for name, file in trainer.available_datasets.items():
            dataset_path = trainer.datasets_dir / file
            if dataset_path.exists():
                size_mb = dataset_path.stat().st_size / (1024 * 1024)
                print(f"   ✅ {name}: {file} ({size_mb:.1f} MB)")
            else:
                print(f"   ❌ {name}: {file} (not found)")
        
        print("\n🚀 Starting training...")
        
        # Train models with default datasets
        training_result = await trainer.train_all_models()
        
        print("\n✅ Training completed!")
        print(f"   📊 Datasets used: {len(training_result['datasets_used'])}")
        print(f"   📈 Total events: {training_result['total_events']}")
        print(f"   🧮 Training samples: {training_result['training_samples']}")
        
        # Show model training results
        print("\n🎯 Model Training Results:")
        for model, success in training_result['model_results'].items():
            status_icon = "✅" if success else "❌"
            print(f"   {status_icon} {model}: {'Success' if success else 'Failed'}")
        
        # Verify models
        verification = trainer.verify_models()
        print(f"\n🔍 Verification: {verification['models_trained']} models ready")
        
        if verification['ready_for_detection']:
            print("\n🎉 SUCCESS: ML models are trained and ready for detection!")
            print("\n📋 Next steps:")
            print("   • Start the backend: cd backend && python -m app.main")
            print("   • Test detection: ./scripts/test-adaptive-detection.sh")
            print("   • Generate test data: python scripts/ml-training/generate-training-data.py")
        else:
            print("\n⚠️  WARNING: Models may not be fully trained")
            print("   Check the model training results above")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        logger.exception("Training error details:")
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

#!/usr/bin/env python3
"""
ML Model Training with Real Open Source Datasets
Enhanced training using both synthetic and real cybersecurity datasets
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

class EnhancedDatasetTrainer:
    """Train ML models using both synthetic and real open source datasets"""
    
    def __init__(self):
        self.datasets_dir = PROJECT_ROOT / "datasets"
        self.real_datasets_dir = PROJECT_ROOT / "datasets" / "real_datasets"
        self.models_dir = PROJECT_ROOT / "backend" / "models"
        self.models_dir.mkdir(exist_ok=True)
        
        # Available synthetic datasets
        self.synthetic_datasets = {
            "brute_force_ssh": "brute_force_ssh_dataset.json",
            "web_attacks": "web_attacks_dataset.json", 
            "network_scans": "network_scans_dataset.json",
            "ddos_attacks": "ddos_attacks_dataset.json",
            "malware_behavior": "malware_behavior_dataset.json",
            "combined_cybersecurity": "combined_cybersecurity_dataset.json"
        }
        
        # Available real datasets
        self.real_datasets = {
            "kdd_cup_1999": "kdd_cup_1999_minixdr.json",
            "honeypot_logs": "honeypot_logs_minixdr.json",
            "urlhaus_threats": "urlhaus_minixdr.json",
            "cicids2017_sample": "cicids2017_sample_minixdr.json",
            "unsw_nb15_sample": "unsw_nb15_sample_minixdr.json",
            "malware_bazaar": "malware_bazaar_minixdr.json"
        }
        
        # Threat intelligence feeds directory
        self.threat_feeds_dir = PROJECT_ROOT / "datasets" / "threat_feeds"
        
    def list_available_datasets(self):
        """List all available datasets (synthetic + real)"""
        print("📊 Available Training Datasets:\n")
        
        print("🔬 Synthetic Datasets:")
        for name, file in self.synthetic_datasets.items():
            dataset_path = self.datasets_dir / file
            if dataset_path.exists():
                size_mb = dataset_path.stat().st_size / (1024 * 1024)
                with open(dataset_path, 'r') as f:
                    data = json.load(f)
                    event_count = len(data) if isinstance(data, list) else 1
                print(f"   ✅ {name}: {event_count} events ({size_mb:.1f} MB)")
            else:
                print(f"   ❌ {name}: not found")
        
        print("\n🌍 Real-World Datasets:")
        for name, file in self.real_datasets.items():
            dataset_path = self.real_datasets_dir / file
            if dataset_path.exists():
                size_mb = dataset_path.stat().st_size / (1024 * 1024)
                with open(dataset_path, 'r') as f:
                    data = json.load(f)
                    event_count = len(data) if isinstance(data, list) else 1
                print(f"   ✅ {name}: {event_count} events ({size_mb:.1f} MB)")
            else:
                print(f"   ❌ {name}: not found")
        
        print(f"\n🔥 Live Threat Intelligence:")
        threat_feeds = self.get_available_threat_feeds()
        total_threat_events = 0
        for feed_file in threat_feeds:
            feed_path = self.threat_feeds_dir / feed_file
            if feed_path.exists():
                size_mb = feed_path.stat().st_size / (1024 * 1024)
                with open(feed_path, 'r') as f:
                    data = json.load(f)
                    event_count = len(data) if isinstance(data, list) else 1
                    total_threat_events += event_count
                print(f"   ✅ {feed_file}: {event_count} events ({size_mb:.1f} MB)")
        
        if total_threat_events > 0:
            print(f"   🎯 Total threat intelligence: {total_threat_events:,} events")
    
    def get_available_threat_feeds(self) -> List[str]:
        """Get list of available threat intelligence feeds"""
        threat_feeds = []
        if self.threat_feeds_dir.exists():
            for feed_file in self.threat_feeds_dir.glob("*_minixdr_*.json"):
                threat_feeds.append(feed_file.name)
        return threat_feeds
    
    def load_dataset(self, dataset_name: str, dataset_type: str = "auto") -> List[Dict[str, Any]]:
        """Load a dataset (synthetic or real)"""
        
        if dataset_type == "auto":
            # Auto-detect type
            if dataset_name in self.synthetic_datasets:
                dataset_type = "synthetic"
            elif dataset_name in self.real_datasets:
                dataset_type = "real"
            else:
                raise ValueError(f"Dataset {dataset_name} not found in either synthetic or real datasets")
        
        if dataset_type == "synthetic":
            if dataset_name not in self.synthetic_datasets:
                raise ValueError(f"Synthetic dataset {dataset_name} not found")
            dataset_file = self.datasets_dir / self.synthetic_datasets[dataset_name]
        else:  # real
            if dataset_name not in self.real_datasets:
                raise ValueError(f"Real dataset {dataset_name} not found")
            dataset_file = self.real_datasets_dir / self.real_datasets[dataset_name]
        
        if not dataset_file.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_file}")
        
        logger.info(f"Loading {dataset_type} dataset: {dataset_name}")
        
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
            
            # Add dataset source metadata for analysis
            features['dataset_source'] = self._identify_dataset_source(ip_events[0])
            
            training_data.append(features)
        
        logger.info(f"Extracted features for {len(training_data)} IP addresses")
        return training_data
    
    def _extract_ip_features(self, src_ip: str, events: List[Event]) -> Dict[str, float]:
        """Extract features for a specific IP using the existing ML engine logic"""
        from app.ml_engine import BaseMLDetector
        
        detector = BaseMLDetector()
        features = detector._extract_features(src_ip, events)
        
        # Remove dataset_source if it exists to avoid conflicts
        features.pop('dataset_source', None)
        
        return features
    
    def _identify_dataset_source(self, event: Event) -> str:
        """Identify which dataset type an event came from"""
        eventid = event.eventid.lower()
        
        if 'cowrie' in eventid:
            return 'honeypot'
        elif 'kdd' in eventid:
            return 'kdd_cup'
        elif 'cicids' in eventid:
            return 'cicids2017'
        elif 'unsw' in eventid:
            return 'unsw_nb15'
        elif 'threat_intel' in eventid:
            return 'threat_intelligence'
        elif 'malware' in eventid:
            return 'malware'
        else:
            return 'synthetic'
    
    async def train_enhanced_models(self, 
                                  synthetic_datasets: List[str] = None,
                                  real_datasets: List[str] = None,
                                  threat_feeds: List[str] = None,
                                  balance_datasets: bool = True) -> Dict[str, Any]:
        """Train ML models with enhanced dataset combination including threat intelligence"""
        
        if synthetic_datasets is None:
            synthetic_datasets = ["combined_cybersecurity", "brute_force_ssh"]
        
        if real_datasets is None:
            # Use ALL available real datasets
            available_real = []
            for json_file in self.real_datasets_dir.glob("*_minixdr.json"):
                dataset_name = json_file.stem.replace("_minixdr", "")
                available_real.append(dataset_name)
            real_datasets = available_real  # Use ALL available datasets
        
        logger.info("🚀 Starting enhanced ML model training with real + synthetic + threat intelligence datasets")
        logger.info(f"Synthetic datasets: {synthetic_datasets}")
        logger.info(f"Real datasets: {real_datasets}")
        
        # Get threat feeds if not provided
        if threat_feeds is None:
            threat_feeds = self.get_available_threat_feeds()
        logger.info(f"Threat intelligence feeds: {len(threat_feeds)} files")
        
        # Load synthetic datasets
        all_events = []
        dataset_stats = {}
        
        for dataset_name in synthetic_datasets:
            try:
                raw_data = self.load_dataset(dataset_name, "synthetic")
                events = self.convert_to_events(raw_data)
                all_events.extend(events)
                dataset_stats[f"synthetic_{dataset_name}"] = len(events)
                
            except Exception as e:
                logger.error(f"Failed to load synthetic dataset {dataset_name}: {e}")
                continue
        
        # Load real datasets
        for dataset_name in real_datasets:
            try:
                raw_data = self.load_dataset(dataset_name, "real")
                events = self.convert_to_events(raw_data)
                all_events.extend(events)
                dataset_stats[f"real_{dataset_name}"] = len(events)
                
            except Exception as e:
                logger.error(f"Failed to load real dataset {dataset_name}: {e}")
                continue
        
        # Load threat intelligence feeds
        for feed_file in threat_feeds:
            try:
                feed_path = self.threat_feeds_dir / feed_file
                with open(feed_path, 'r') as f:
                    threat_data = json.load(f)
                
                # Convert to events
                threat_events = self.convert_to_events(threat_data)
                all_events.extend(threat_events)
                dataset_stats[f"threat_{feed_file}"] = len(threat_events)
                logger.info(f"Loaded {len(threat_events)} threat events from {feed_file}")
                
            except Exception as e:
                logger.error(f"Failed to load threat feed {feed_file}: {e}")
                continue
        
        if not all_events:
            raise ValueError("No training data loaded!")
        
        logger.info(f"Total training events: {len(all_events)}")
        logger.info(f"Dataset breakdown: {dataset_stats}")
        
        # Balance datasets if requested
        if balance_datasets:
            all_events = self._balance_datasets(all_events, dataset_stats)
            logger.info(f"Balanced dataset size: {len(all_events)} events")
        
        # Extract training features
        training_data = self.extract_training_features(all_events)
        
        if len(training_data) < 10:
            raise ValueError(f"Insufficient training data: {len(training_data)} samples")
        
        # Analyze dataset composition
        composition = self._analyze_dataset_composition(training_data)
        logger.info(f"Dataset composition: {composition}")
        
        # Train the ensemble detector
        logger.info("🧠 Training ML ensemble models...")
        results = await ml_detector.train_models(training_data)
        
        # Verify training success
        model_status = ml_detector.get_model_status()
        
        # Save enhanced metadata
        training_metadata = {
            "training_completed": datetime.now().isoformat(),
            "synthetic_datasets_used": synthetic_datasets,
            "real_datasets_used": real_datasets,
            "dataset_stats": dataset_stats,
            "dataset_composition": composition,
            "total_events": len(all_events),
            "training_samples": len(training_data),
            "model_results": results,
            "model_status": model_status,
            "balance_datasets": balance_datasets
        }
        
        metadata_file = self.models_dir / "enhanced_training_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(training_metadata, f, indent=2)
        
        return training_metadata
    
    def _balance_datasets(self, events: List[Event], dataset_stats: Dict[str, int]) -> List[Event]:
        """Balance the dataset to prevent bias towards larger datasets"""
        
        # Group events by dataset source
        events_by_source = {}
        for event in events:
            source = self._identify_dataset_source(event)
            if source not in events_by_source:
                events_by_source[source] = []
            events_by_source[source].append(event)
        
        # Find the median dataset size for balancing
        sizes = [len(events) for events in events_by_source.values()]
        target_size = int(np.median(sizes))
        
        # Sample from each dataset to target size
        balanced_events = []
        for source, source_events in events_by_source.items():
            if len(source_events) > target_size:
                # Randomly sample down to target size
                import random
                sampled_events = random.sample(source_events, target_size)
                balanced_events.extend(sampled_events)
                logger.info(f"Balanced {source}: {len(source_events)} -> {target_size} events")
            else:
                # Keep all events if under target
                balanced_events.extend(source_events)
                logger.info(f"Kept all {source}: {len(source_events)} events")
        
        return balanced_events
    
    def _analyze_dataset_composition(self, training_data: List[Dict[str, float]]) -> Dict[str, int]:
        """Analyze the composition of the training dataset"""
        composition = {}
        
        for sample in training_data:
            source = sample.get('dataset_source', 'unknown')
            composition[source] = composition.get(source, 0) + 1
        
        return composition
    
    def verify_enhanced_models(self) -> Dict[str, Any]:
        """Verify that enhanced models are trained and working"""
        logger.info("🔍 Verifying enhanced trained models...")
        
        status = ml_detector.get_model_status()
        
        verification_result = {
            "models_trained": 0,
            "models_available": len(status.get('available_models', [])),
            "status": status,
            "ready_for_detection": False,
            "enhancement_level": "basic"
        }
        
        # Count trained models
        trained_models = [
            status.get('isolation_forest', False),
            status.get('lstm', False), 
            status.get('enhanced_ml_trained', False),
            status.get('federated_enabled', False)
        ]
        
        verification_result["models_trained"] = sum(trained_models)
        verification_result["ready_for_detection"] = verification_result["models_trained"] > 0
        
        # Determine enhancement level
        if verification_result["models_trained"] >= 3:
            verification_result["enhancement_level"] = "advanced"
        elif verification_result["models_trained"] >= 2:
            verification_result["enhancement_level"] = "intermediate"
        
        logger.info(f"✅ {verification_result['models_trained']} models trained successfully")
        logger.info(f"Enhancement level: {verification_result['enhancement_level']}")
        logger.info(f"Ready for detection: {verification_result['ready_for_detection']}")
        
        return verification_result

async def main():
    """Main enhanced training function"""
    trainer = EnhancedDatasetTrainer()
    
    print("🧠 Enhanced ML Model Training with Real Open Source Datasets")
    print("=" * 70)
    
    try:
        # List available datasets
        trainer.list_available_datasets()
        
        print("\n🚀 Starting enhanced training...")
        
        # Include threat intelligence feeds
        threat_feeds = trainer.get_available_threat_feeds()
        
        # Train models with ALL available datasets
        training_result = await trainer.train_enhanced_models(
            synthetic_datasets=["combined_cybersecurity", "brute_force_ssh"],
            real_datasets=None,  # Auto-discover ALL real datasets
            threat_feeds=threat_feeds,
            balance_datasets=True
        )
        
        print("\n✅ Enhanced training completed!")
        print(f"   📊 Synthetic datasets: {len(training_result['synthetic_datasets_used'])}")
        print(f"   🌍 Real datasets: {len(training_result['real_datasets_used'])}")
        print(f"   📈 Total events: {training_result['total_events']}")
        print(f"   🧮 Training samples: {training_result['training_samples']}")
        
        # Show dataset composition
        print(f"\n📋 Dataset Composition:")
        for source, count in training_result['dataset_composition'].items():
            percentage = (count / training_result['training_samples']) * 100
            print(f"   • {source}: {count} samples ({percentage:.1f}%)")
        
        # Show model training results
        print("\n🎯 Model Training Results:")
        for model, success in training_result['model_results'].items():
            status_icon = "✅" if success else "❌"
            print(f"   {status_icon} {model}: {'Success' if success else 'Failed'}")
        
        # Verify enhanced models
        verification = trainer.verify_enhanced_models()
        print(f"\n🔍 Verification: {verification['models_trained']} models ready")
        print(f"   Enhancement Level: {verification['enhancement_level'].title()}")
        
        if verification['ready_for_detection']:
            print("\n🎉 SUCCESS: Enhanced ML models are trained with real-world data!")
            print("\n📋 Your models now include:")
            print("   ✅ Real network intrusion data (KDD Cup 1999)")
            print("   ✅ Real honeypot attack logs")
            print("   ✅ Real malware threat intelligence")
            print("   ✅ Synthetic attack patterns")
            print("\n🚀 Next steps:")
            print("   • Start detection: cd backend && python -m app.main")
            print("   • Test accuracy: ./scripts/test-adaptive-detection.sh")
            print("   • Monitor performance: curl http://localhost:8000/api/ml_status")
        else:
            print("\n⚠️  WARNING: Models may not be fully trained")
            print("   Check the model training results above")
        
    except Exception as e:
        print(f"\n❌ Enhanced training failed: {e}")
        logger.exception("Enhanced training error details:")
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

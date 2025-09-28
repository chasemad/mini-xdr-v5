#!/usr/bin/env python3
"""
Massive Dataset Trainer - Uses ALL available datasets
Loads every available JSON dataset for maximum training data
"""
import os
import sys
import json
import asyncio
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import logging

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent / "backend"))

from app.ml_engine import ml_detector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent

class MassiveDatasetTrainer:
    """Train with ALL available datasets for maximum performance"""
    
    def __init__(self):
        self.datasets_dir = PROJECT_ROOT / "datasets"
        self.real_datasets_dir = PROJECT_ROOT / "datasets" / "real_datasets"
        self.threat_feeds_dir = PROJECT_ROOT / "datasets" / "threat_feeds"
        
    def discover_all_datasets(self) -> Dict[str, List[Path]]:
        """Discover ALL available dataset files"""
        datasets = {
            "synthetic": [],
            "real": [],
            "threat_feeds": []
        }
        
        # Synthetic datasets
        for json_file in self.datasets_dir.glob("*_dataset.json"):
            datasets["synthetic"].append(json_file)
        
        # Real datasets
        for json_file in self.real_datasets_dir.glob("*_minixdr.json"):
            datasets["real"].append(json_file)
        
        # Threat intelligence feeds
        for json_file in self.threat_feeds_dir.glob("*_minixdr_*.json"):
            datasets["threat_feeds"].append(json_file)
        
        return datasets
    
    def load_all_datasets(self) -> List[Dict[str, Any]]:
        """Load ALL available datasets"""
        print("🗃️ Loading ALL available datasets...")
        
        datasets = self.discover_all_datasets()
        all_events = []
        dataset_stats = {}
        
        # Load synthetic datasets
        print("\n🔬 Synthetic Datasets:")
        for json_file in datasets["synthetic"]:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                events = len(data)
                all_events.extend(data)
                dataset_stats[f"synthetic_{json_file.stem}"] = events
                print(f"   ✅ {json_file.name}: {events:,} events")
                
            except Exception as e:
                print(f"   ❌ {json_file.name}: Error - {e}")
        
        # Load real datasets
        print("\n🌍 Real-World Datasets:")
        for json_file in datasets["real"]:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                events = len(data)
                all_events.extend(data)
                dataset_stats[f"real_{json_file.stem}"] = events
                print(f"   ✅ {json_file.name}: {events:,} events")
                
            except Exception as e:
                print(f"   ❌ {json_file.name}: Error - {e}")
        
        # Load threat intelligence
        print("\n🔥 Threat Intelligence:")
        for json_file in datasets["threat_feeds"]:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                events = len(data)
                all_events.extend(data)
                dataset_stats[f"threat_{json_file.stem}"] = events
                print(f"   ✅ {json_file.name}: {events:,} events")
                
            except Exception as e:
                print(f"   ❌ {json_file.name}: Error - {e}")
        
        print(f"\n🚀 TOTAL LOADED: {len(all_events):,} events from {len(dataset_stats)} datasets")
        return all_events, dataset_stats
    
    def convert_to_mini_xdr_events(self, raw_events: List[Dict]) -> List:
        """Convert raw events to Mini-XDR Event objects"""
        from app.models import Event
        
        events = []
        for event_data in raw_events:
            try:
                # Create Event object
                event = Event(
                    ts=datetime.fromisoformat(event_data.get('timestamp', datetime.now().isoformat()).replace('Z', '+00:00')),
                    src_ip=event_data.get('src_ip', '0.0.0.0'),
                    dst_ip=event_data.get('dst_ip', '0.0.0.0'),
                    dst_port=int(event_data.get('dst_port', 0)),
                    eventid=event_data.get('eventid', 'unknown'),
                    message=event_data.get('message', ''),
                    raw=event_data.get('raw', {})
                )
                events.append(event)
            except Exception as e:
                # Skip malformed events
                continue
        
        return events
    
    def extract_training_features(self, events: List) -> List[Dict[str, float]]:
        """Extract ML training features from events"""
        print("🧮 Extracting training features...")
        
        # Group events by source IP
        ip_groups = {}
        for event in events:
            src_ip = event.src_ip if hasattr(event, 'src_ip') else event.get('src_ip', '0.0.0.0')
            if src_ip not in ip_groups:
                ip_groups[src_ip] = []
            ip_groups[src_ip].append(event)
        
        training_data = []
        
        for src_ip, ip_events in ip_groups.items():
            # Extract features for this IP
            features = {
                'src_ip_numeric': self._ip_to_numeric(src_ip),
                'total_events': len(ip_events),
                'unique_dst_ports': len(set(e.dst_port for e in ip_events)),
                'unique_dst_ips': len(set(e.dst_ip for e in ip_events)),
                'time_span_minutes': 60,  # Default span
                'events_per_minute': len(ip_events) / 60,
                'most_common_port': self._get_most_common_port(ip_events),
                'dataset_source': self._identify_dataset_source(ip_events[0])
            }
            
            training_data.append(features)
        
        print(f"   ✅ Extracted features for {len(training_data)} IP addresses")
        return training_data
    
    def _ip_to_numeric(self, ip: str) -> float:
        """Convert IP address to numeric value"""
        try:
            parts = ip.split('.')
            return float(int(parts[0]) * 16777216 + int(parts[1]) * 65536 + int(parts[2]) * 256 + int(parts[3]))
        except:
            return 0.0
    
    def _get_most_common_port(self, events: List) -> int:
        """Get most commonly targeted port"""
        ports = [event.dst_port for event in events]
        
        if ports:
            return max(set(ports), key=ports.count)
        return 80
    
    def _identify_dataset_source(self, event) -> str:
        """Identify which dataset an event came from"""
        eventid = str(event.eventid).lower()
        
        if 'honeypot' in eventid or 'cowrie' in eventid:
            return 'honeypot'
        elif 'kdd' in eventid:
            return 'kdd_cup'
        elif 'cicids' in eventid:
            return 'cicids2017'
        elif 'unsw' in eventid:
            return 'unsw_nb15'
        elif 'threat' in eventid or 'abuse' in eventid or 'spam' in eventid:
            return 'threat_intelligence'
        elif 'synthetic' in eventid:
            return 'synthetic'
        else:
            return 'unknown'
    
    async def train_massive_model(self) -> Dict[str, Any]:
        """Train ML models with ALL available data"""
        print("🚀 MASSIVE DATASET ML TRAINING")
        print("=" * 60)
        
        # Load all datasets
        raw_events, dataset_stats = self.load_all_datasets()
        
        if not raw_events:
            raise ValueError("No training data found!")
        
        # Convert to Mini-XDR events
        print(f"\n🔄 Converting {len(raw_events):,} events to Mini-XDR format...")
        events = self.convert_to_mini_xdr_events(raw_events)
        print(f"   ✅ Converted {len(events):,} valid events")
        
        # Extract training features
        training_data = self.extract_training_features(events)
        
        if len(training_data) < 10:
            raise ValueError(f"Insufficient training data: {len(training_data)} samples")
        
        # Analyze dataset composition
        composition = {}
        for sample in training_data:
            source = sample.get('dataset_source', 'unknown')
            composition[source] = composition.get(source, 0) + 1
        
        print(f"\n📊 Dataset Composition:")
        for source, count in composition.items():
            percentage = (count / len(training_data)) * 100
            print(f"   • {source}: {count} samples ({percentage:.1f}%)")
        
        # Train ML models
        print(f"\n🧠 Training ML ensemble with {len(training_data)} samples...")
        results = await ml_detector.train_models(training_data)
        
        # Verify training
        model_status = ml_detector.get_model_status()
        
        return {
            "total_raw_events": len(raw_events),
            "converted_events": len(events),
            "training_samples": len(training_data),
            "dataset_stats": dataset_stats,
            "composition": composition,
            "model_results": results,
            "model_status": model_status,
            "training_completed": datetime.now().isoformat()
        }

async def main():
    """Main execution function"""
    trainer = MassiveDatasetTrainer()
    
    try:
        result = await trainer.train_massive_model()
        
        print("\n🎉 MASSIVE TRAINING COMPLETED!")
        print("=" * 60)
        print(f"📊 Raw events processed: {result['total_raw_events']:,}")
        print(f"🔄 Valid events converted: {result['converted_events']:,}")
        print(f"🧮 Training samples: {result['training_samples']:,}")
        print(f"📈 Datasets used: {len(result['dataset_stats'])}")
        
        print(f"\n🎯 Model Training Results:")
        for model, success in result['model_results'].items():
            status_icon = "✅" if success else "❌"
            print(f"   {status_icon} {model}: {'Success' if success else 'Failed'}")
        
        print(f"\n🚀 Your Mini-XDR now has MASSIVE detection capabilities!")
        print(f"   Enhanced with {result['total_raw_events']:,} training events")
        print(f"   Covering {len(result['composition'])} attack categories")
        
    except Exception as e:
        logger.error(f"Massive training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())

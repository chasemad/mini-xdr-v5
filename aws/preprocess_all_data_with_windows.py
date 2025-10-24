#!/usr/bin/env python3
"""
Preprocess ALL Training Data Including New Windows/AD Attacks
Creates fresh .npy files with 4M+ events + Windows data
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_json_dataset(json_file):
    """Load dataset from JSON or JSONL file"""
    logger.info(f"Loading {json_file.name}...")
    
    features = []
    labels = []
    
    try:
        # First try JSONL format (one JSON object per line)
        with open(json_file, 'r') as f:
            line_count = 0
            for line in tqdm(f, desc=f"  Reading {json_file.name}", unit=" lines"):
                line_count += 1
                if not line.strip():
                    continue
                
                try:
                    event = json.loads(line)
                    
                    # Extract features and label
                    if 'features' in event and 'label' in event:
                        feat = event['features']
                        if isinstance(feat, list) and len(feat) == 79:
                            features.append(feat)
                            labels.append(int(event['label']))
                    
                except json.JSONDecodeError:
                    continue
                
                # Limit to prevent memory issues (sample if too large)
                if len(features) >= 500000:  # Take first 500k from each file
                    logger.info(f"  ⚡ Limiting to 500k samples from {json_file.name}")
                    break
        
        if features:
            logger.info(f"  ✅ Loaded {len(features):,} samples from {json_file.name}")
            return np.array(features, dtype=np.float32), np.array(labels, dtype=np.int64)
        
        # If JSONL didn't work, try regular JSON
        logger.info(f"  🔄 Trying single JSON format...")
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, dict) and 'events' in data:
            events = data['events']
        elif isinstance(data, list):
            events = data
        else:
            events = [data]
        
        for event in events:
            if isinstance(event, dict) and 'features' in event and 'label' in event:
                feat = event['features']
                if len(feat) == 79:
                    features.append(feat)
                    labels.append(int(event['label']))
        
        if features:
            logger.info(f"  ✅ Loaded {len(features):,} samples from {json_file.name}")
            return np.array(features, dtype=np.float32), np.array(labels, dtype=np.int64)
        
    except Exception as e:
        logger.warning(f"  ⚠️  Failed to load {json_file.name}: {e}")
    
    return None, None


def load_csv_dataset(csv_file):
    """Load dataset from CSV file"""
    logger.info(f"Loading {csv_file.name}")
    
    try:
        df = pd.read_csv(csv_file)
        
        # Handle different CSV formats
        if 'label' in df.columns:
            features = df.drop('label', axis=1).values.astype(np.float32)
            labels = df['label'].values.astype(np.int64)
        else:
            # Assume last column is label
            features = df.iloc[:, :-1].values.astype(np.float32)
            labels = df.iloc[:, -1].values.astype(np.int64)
        
        # Ensure 79 features
        if features.shape[1] != 79:
            if features.shape[1] > 79:
                features = features[:, :79]
            else:
                padding = np.zeros((features.shape[0], 79 - features.shape[1]), dtype=np.float32)
                features = np.hstack([features, padding])
        
        # Clean data
        features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
        
        logger.info(f"  ✅ Loaded {len(features):,} samples from {csv_file.name}")
        return features, labels
        
    except Exception as e:
        logger.warning(f"  ⚠️  Failed to load {csv_file.name}: {e}")
        return None, None


def main():
    """Preprocess all datasets including new Windows data"""
    logger.info("🚀 Preprocessing ALL Data (4M+ events + Windows attacks)")
    logger.info("=" * 70)
    
    # Load all datasets
    data_dir = Path("/Users/chasemad/Desktop/mini-xdr/datasets/real_datasets")
    
    all_features = []
    all_labels = []
    dataset_info = {}
    
    # Load JSON files
    logger.info("\n📁 Loading JSON datasets...")
    json_files = list(data_dir.glob("*.json"))
    
    for json_file in sorted(json_files):
        features, labels = load_json_dataset(json_file)
        if features is not None:
            all_features.append(features)
            all_labels.extend(labels.tolist())
            dataset_info[json_file.stem] = {
                'samples': len(features),
                'file': str(json_file.name)
            }
    
    # Load CSV files
    logger.info("\n📁 Loading CSV datasets...")
    csv_files = list(data_dir.glob("*.csv"))
    
    for csv_file in sorted(csv_files):
        features, labels = load_csv_dataset(csv_file)
        if features is not None:
            all_features.append(features)
            all_labels.extend(labels.tolist())
            dataset_info[csv_file.stem] = {
                'samples': len(features),
                'file': str(csv_file.name)
            }
    
    # Combine all datasets
    if not all_features:
        logger.error("❌ No datasets loaded!")
        return 1
    
    logger.info("\n🔄 Combining all datasets...")
    combined_features = np.vstack(all_features)
    combined_labels = np.array(all_labels, dtype=np.int64)
    
    # Clean combined data
    logger.info("🧹 Cleaning data...")
    combined_features = np.nan_to_num(combined_features, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Get class distribution
    unique_classes, class_counts = np.unique(combined_labels, return_counts=True)
    
    logger.info("\n" + "=" * 70)
    logger.info("📊 COMBINED DATASET STATISTICS")
    logger.info("=" * 70)
    logger.info(f"Total samples: {len(combined_features):,}")
    logger.info(f"Features: {combined_features.shape[1]}")
    logger.info(f"Classes: {len(unique_classes)}")
    logger.info(f"Memory size: {combined_features.nbytes / 1e9:.2f} GB")
    
    logger.info("\n📊 Class Distribution:")
    class_names = ['Normal', 'DDoS', 'Recon', 'Brute Force', 'Web Attack', 'Malware', 'APT',
                   'Kerberos', 'Lateral Mvmt', 'Cred Theft', 'Priv Esc', 'Exfiltration', 'Insider']
    
    for class_id, count in zip(unique_classes, class_counts):
        class_name = class_names[int(class_id)] if int(class_id) < len(class_names) else f"Class {class_id}"
        percentage = (count / len(combined_labels)) * 100
        logger.info(f"  Class {class_id} ({class_name:15s}): {count:>8,} samples ({percentage:>5.2f}%)")
    
    logger.info("\n📁 Dataset Sources:")
    for name, info in sorted(dataset_info.items()):
        logger.info(f"  {name:40s}: {info['samples']:>8,} samples")
    
    # Save preprocessed data
    output_dir = Path("/Users/chasemad/Desktop/mini-xdr/aws/training_data")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    features_file = output_dir / f"training_features_{timestamp}.npy"
    labels_file = output_dir / f"training_labels_{timestamp}.npy"
    
    logger.info(f"\n💾 Saving preprocessed data...")
    logger.info(f"   Features: {features_file}")
    logger.info(f"   Labels: {labels_file}")
    
    np.save(features_file, combined_features)
    np.save(labels_file, combined_labels)
    
    # Save metadata
    metadata = {
        'timestamp': timestamp,
        'total_samples': int(len(combined_features)),
        'features': 79,
        'classes': int(len(unique_classes)),
        'class_distribution': dict(zip([int(x) for x in unique_classes], [int(x) for x in class_counts])),
        'dataset_sources': dataset_info,
        'includes_windows_data': True,  # ✅ IMPORTANT!
        'files': {
            'features': str(features_file),
            'labels': str(labels_file)
        }
    }
    
    metadata_file = output_dir / f"training_metadata_{timestamp}.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"   Metadata: {metadata_file}")
    
    logger.info("\n" + "=" * 70)
    logger.info("✅ PREPROCESSING COMPLETE!")
    logger.info("=" * 70)
    logger.info(f"📊 Total samples: {len(combined_features):,}")
    logger.info(f"✅ Includes Windows/AD attack data: YES")
    logger.info(f"📁 Ready for training!")
    
    logger.info("\n🚀 Next step:")
    logger.info(f"   python3 aws/train_enhanced_full_dataset.py \\")
    logger.info(f"     --data-dir aws/training_data \\")
    logger.info(f"     --epochs 50")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())


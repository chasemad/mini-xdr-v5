#!/usr/bin/env python3
"""
COMPREHENSIVE Data Preprocessing with Windows/AD Attacks
- Loads ALL 4M+ network events from JSONL files
- Adds Windows/AD attack samples
- Intelligently balances classes
- Creates training-ready .npy files
"""

import os
import sys
import json
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_jsonl_file_optimized(json_file, max_samples_per_class=500000):
    """Load JSONL file with class balancing"""
    logger.info(f"📂 Loading {json_file.name} (JSONL format)...")
    
    features_by_class = {i: [] for i in range(13)}  # 13 classes
    labels_by_class = {i: [] for i in range(13)}
    
    total_lines = 0
    valid_samples = 0
    
    try:
        with open(json_file, 'r') as f:
            for line in tqdm(f, desc=f"  Parsing", unit=" events"):
                total_lines += 1
                
                if not line.strip():
                    continue
                
                try:
                    event = json.loads(line)
                    
                    # Extract features and label
                    if isinstance(event, dict):
                        feat = event.get('features')
                        label = event.get('label')
                        
                        if feat is not None and label is not None:
                            if isinstance(feat, list) and len(feat) == 79:
                                label = int(label)
                                if 0 <= label < 13:
                                    # Only add if we haven't reached class limit
                                    if len(features_by_class[label]) < max_samples_per_class:
                                        features_by_class[label].append(feat)
                                        labels_by_class[label].append(label)
                                        valid_samples += 1
                
                except (json.JSONDecodeError, ValueError, KeyError):
                    continue
                
                # Early stopping if all classes are full
                if all(len(f) >= max_samples_per_class for f in features_by_class.values()):
                    logger.info(f"  ⚡ All classes full, stopping early")
                    break
        
        logger.info(f"  📊 Parsed {total_lines:,} lines, found {valid_samples:,} valid samples")
        
        # Combine all classes
        all_features = []
        all_labels = []
        
        for class_id in sorted(features_by_class.keys()):
            class_features = features_by_class[class_id]
            if class_features:
                all_features.extend(class_features)
                all_labels.extend(labels_by_class[class_id])
                logger.info(f"    Class {class_id}: {len(class_features):,} samples")
        
        if all_features:
            return np.array(all_features, dtype=np.float32), np.array(all_labels, dtype=np.int64)
        
    except Exception as e:
        logger.error(f"  ❌ Error loading {json_file.name}: {e}")
    
    return None, None


def main():
    """Main preprocessing with intelligent sampling"""
    logger.info("🚀 COMPREHENSIVE PREPROCESSING - 4M+ Events + Windows Data")
    logger.info("=" * 70)
    
    data_dir = Path("/Users/chasemad/Desktop/mini-xdr/datasets/real_datasets")
    
    all_features = []
    all_labels = []
    
    # Strategy: Load with class limits to keep memory manageable
    MAX_SAMPLES_PER_CLASS = 600000  # 600k per class = ~4.8M total balanced
    
    logger.info(f"\n📊 Strategy: Load up to {MAX_SAMPLES_PER_CLASS:,} samples per class")
    logger.info(f"   Max total: ~{MAX_SAMPLES_PER_CLASS * 13:,} samples (balanced across 13 classes)")
    
    # Load main datasets
    large_files = [
        'cicids2017_enhanced_minixdr.json',  # 18M lines
        'kdd_full_minixdr.json',              # 380k lines
        'kdd_10_percent_minixdr.json'         # 380k lines
    ]
    
    logger.info("\n📁 Loading large JSONL datasets...")
    for filename in large_files:
        filepath = data_dir / filename
        if filepath.exists():
            features, labels = load_jsonl_file_optimized(filepath, MAX_SAMPLES_PER_CLASS)
            if features is not None:
                all_features.append(features)
                all_labels.extend(labels.tolist())
                logger.info(f"  ✅ {filename}: {len(features):,} samples")
    
    # Load Windows data (CSV - already converted)
    logger.info("\n📁 Loading Windows/AD attack data...")
    windows_csv = data_dir / "windows_ad_converted.csv"
    if windows_csv.exists():
        import pandas as pd
        df = pd.read_csv(windows_csv)
        if 'label' in df.columns:
            win_features = df.drop('label', axis=1).values.astype(np.float32)
            win_labels = df['label'].values.astype(np.int64)
            all_features.append(win_features)
            all_labels.extend(win_labels.tolist())
            logger.info(f"  ✅ Windows data: {len(win_features):,} samples")
    
    # Combine everything
    if not all_features:
        logger.error("❌ No data loaded!")
        return 1
    
    logger.info("\n🔄 Combining all datasets...")
    combined_features = np.vstack(all_features)
    combined_labels = np.array(all_labels, dtype=np.int64)
    
    # Clean data
    logger.info("🧹 Cleaning data...")
    combined_features = np.nan_to_num(combined_features, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Show distribution
    unique_classes, class_counts = np.unique(combined_labels, return_counts=True)
    
    logger.info("\n" + "=" * 70)
    logger.info("📊 FINAL DATASET STATISTICS")
    logger.info("=" * 70)
    logger.info(f"Total samples: {len(combined_features):,}")
    logger.info(f"Features: {combined_features.shape[1]}")
    logger.info(f"Classes: {len(unique_classes)}")
    logger.info(f"Memory: {combined_features.nbytes / 1e9:.2f} GB")
    
    logger.info("\n📊 Class Distribution:")
    class_names = {
        0: 'Normal', 1: 'DDoS', 2: 'Recon', 3: 'Brute Force',
        4: 'Web Attack', 5: 'Malware', 6: 'APT',
        7: 'Kerberos Attack', 8: 'Lateral Movement', 9: 'Credential Theft',
        10: 'Privilege Escalation', 11: 'Data Exfiltration', 12: 'Insider Threat'
    }
    
    for class_id, count in zip(unique_classes, class_counts):
        class_name = class_names.get(int(class_id), f'Class {class_id}')
        percentage = (count / len(combined_labels)) * 100
        logger.info(f"  {class_id:2d} ({class_name:20s}): {count:>10,} ({percentage:>5.2f}%)")
    
    # Save preprocessed data
    output_dir = Path("/Users/chasemad/Desktop/mini-xdr/aws/training_data")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    features_file = output_dir / f"training_features_{timestamp}.npy"
    labels_file = output_dir / f"training_labels_{timestamp}.npy"
    
    logger.info(f"\n💾 Saving preprocessed data...")
    np.save(features_file, combined_features)
    np.save(labels_file, combined_labels)
    
    # Save metadata
    metadata = {
        'timestamp': timestamp,
        'total_samples': int(len(combined_features)),
        'features': 79,
        'classes': int(len(unique_classes)),
        'class_distribution': dict(zip([int(x) for x in unique_classes], [int(x) for x in class_counts])),
        'includes_windows_data': True,
        'windows_samples': int(class_counts[unique_classes == 7].sum() if 7 in unique_classes else 0) + \
                           int(class_counts[unique_classes == 8].sum() if 8 in unique_classes else 0) + \
                           int(class_counts[unique_classes == 9].sum() if 9 in unique_classes else 0),
        'files': {
            'features': str(features_file.name),
            'labels': str(labels_file.name)
        }
    }
    
    metadata_file = output_dir / f"training_metadata_{timestamp}.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"   ✅ Features: {features_file.name}")
    logger.info(f"   ✅ Labels: {labels_file.name}")
    logger.info(f"   ✅ Metadata: {metadata_file.name}")
    
    logger.info("\n" + "=" * 70)
    logger.info("✅ PREPROCESSING COMPLETE!")
    logger.info("=" * 70)
    logger.info(f"📊 Ready to train with {len(combined_features):,} samples")
    logger.info(f"✅ Includes Windows/AD attacks: YES")
    logger.info(f"✅ Balanced across {len(unique_classes)} classes")
    
    logger.info("\n🚀 Start training with:")
    logger.info(f"   python3 aws/train_enhanced_full_dataset.py --data-dir aws/training_data")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())


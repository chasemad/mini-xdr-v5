#!/usr/bin/env python3
"""
Balance Windows Dataset with Data Augmentation
Addresses class imbalance by generating synthetic variants
"""

import numpy as np
import json
import logging
from pathlib import Path
from collections import Counter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WindowsDataBalancer:
    """Balance Windows attack dataset using augmentation"""
    
    def __init__(self, input_dir="$(cd "$(dirname "$0")/../.." ${PROJECT_ROOT:-$(dirname $(dirname $(dirname $(realpath "$0"))))}${PROJECT_ROOT:-$(dirname $(dirname $(dirname $(realpath "$0"))))} pwd)/datasets/windows_converted"):
        self.input_dir = Path(input_dir)
        self.target_samples_per_class = 30000  # Target for balanced training
        
    def load_data(self):
        """Load converted Windows data"""
        logger.info("📂 Loading Windows dataset...")
        
        X = np.load(self.input_dir / "windows_features.npy")
        y = np.load(self.input_dir / "windows_labels.npy")
        
        logger.info(f"   Loaded: X={X.shape}, y={y.shape}")
        return X, y
    
    def augment_sample(self, features: np.ndarray, noise_level: float = 0.05) -> np.ndarray:
        """Augment a single sample with controlled noise"""
        # Add Gaussian noise
        noise = np.random.normal(0, noise_level, features.shape)
        augmented = features + noise
        
        # Ensure values stay in [0, 1] range
        augmented = np.clip(augmented, 0, 1)
        
        # Random feature dropout (slight variation)
        if np.random.rand() > 0.9:
            dropout_idx = np.random.choice(len(augmented), size=int(len(augmented) * 0.1), replace=False)
            augmented[dropout_idx] *= np.random.uniform(0.5, 1.5)
        
        return augmented
    
    def balance_dataset(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """Balance dataset using SMOTE-like augmentation"""
        logger.info("\n" + "=" * 70)
        logger.info("⚖️  Balancing Dataset")
        logger.info("=" * 70)
        
        # Count samples per class
        class_counts = Counter(y)
        logger.info("\nOriginal distribution:")
        for class_id, count in sorted(class_counts.items()):
            logger.info(f"  Class {class_id:2d}: {count:6,} samples")
        
        X_balanced = []
        y_balanced = []
        
        # For each class
        for class_id in range(13):
            class_mask = (y == class_id)
            class_samples = X[class_mask]
            
            if len(class_samples) == 0:
                # Generate synthetic samples if class is missing
                logger.info(f"\n  Class {class_id}: No samples - generating {self.target_samples_per_class:,} synthetic")
                for _ in range(self.target_samples_per_class):
                    synthetic = self._generate_class_specific_sample(class_id)
                    X_balanced.append(synthetic)
                    y_balanced.append(class_id)
            
            elif len(class_samples) < self.target_samples_per_class:
                # Augment to reach target
                needed = self.target_samples_per_class - len(class_samples)
                logger.info(f"  Class {class_id}: {len(class_samples):6,} → {self.target_samples_per_class:6,} (augmenting {needed:,})")
                
                # Keep original samples
                X_balanced.extend(class_samples)
                y_balanced.extend([class_id] * len(class_samples))
                
                # Generate augmented samples
                for _ in range(needed):
                    # Randomly select a sample to augment
                    idx = np.random.randint(len(class_samples))
                    augmented = self.augment_sample(class_samples[idx])
                    X_balanced.append(augmented)
                    y_balanced.append(class_id)
            
            else:
                # Downsample if too many
                logger.info(f"  Class {class_id}: {len(class_samples):6,} → {self.target_samples_per_class:6,} (downsampling)")
                indices = np.random.choice(len(class_samples), self.target_samples_per_class, replace=False)
                X_balanced.extend(class_samples[indices])
                y_balanced.extend([class_id] * self.target_samples_per_class)
        
        X_balanced = np.array(X_balanced, dtype=np.float32)
        y_balanced = np.array(y_balanced, dtype=np.int64)
        
        logger.info("\n" + "=" * 70)
        logger.info(f"✅ Balanced dataset: {len(X_balanced):,} samples")
        logger.info("\nFinal distribution:")
        
        final_counts = Counter(y_balanced)
        for class_id in range(13):
            count = final_counts.get(class_id, 0)
            pct = (count / len(y_balanced)) * 100
            logger.info(f"  Class {class_id:2d}: {count:6,} samples ({pct:5.1f}%)")
        
        return X_balanced, y_balanced
    
    def _generate_class_specific_sample(self, class_id: int) -> np.ndarray:
        """Generate synthetic sample for specific class"""
        features = np.random.rand(79) * 0.3  # Base features
        
        if class_id == 0:  # Normal
            features *= 0.2
            features[3] = np.random.choice([80, 443, 53, 22]) / 65535
        
        elif class_id == 1:  # DDoS
            features[5:9] = np.random.rand(4) * 0.9  # High bytes/packets
            features[75] = 0.85
        
        elif class_id == 2:  # Reconnaissance
            features[3] = np.random.randint(0, 1024) / 65535  # Low ports
            features[75] = 0.6
        
        elif class_id == 3:  # Brute force
            features[39] = 0.8  # Failed auth
            features[75] = 0.9
        
        elif class_id == 4:  # Web attack
            features[3] = np.random.choice([80, 443, 8080]) / 65535
            features[75] = 0.85
        
        elif class_id == 5:  # Malware
            features[20:25] = np.random.rand(5) * 0.9  # Process features
            features[75] = 0.95
        
        elif class_id == 6:  # APT
            features[73:79] = np.random.rand(6) * 0.8  # High behavioral
            features[75] = 0.8
        
        elif class_id == 7:  # Kerberos attack
            features[65:73] = np.random.rand(8) * 0.9  # Kerberos features
            features[75] = 0.9
        
        elif class_id == 8:  # Lateral movement
            features[3] = np.random.choice([445, 3389, 135]) / 65535
            features[75] = 0.85
        
        elif class_id == 9:  # Credential theft
            features[20] = 0.9
            features[24] = 1.0  # High privilege
            features[75] = 0.95
        
        elif class_id == 10:  # Privilege escalation
            features[24] = 0.9
            features[75] = 0.9
        
        elif class_id == 11:  # Data exfiltration
            features[5] = 0.9  # High bytes sent
            features[75] = 0.85
        
        elif class_id == 12:  # Insider threat
            features[73:79] = np.random.rand(6) * 0.9
            features[75] = 0.8
        
        return features
    
    def save_balanced_data(self, X: np.ndarray, y: np.ndarray):
        """Save balanced dataset"""
        logger.info("\n💾 Saving balanced dataset...")
        
        np.save(self.input_dir / "windows_features_balanced.npy", X)
        np.save(self.input_dir / "windows_labels_balanced.npy", y)
        
        logger.info(f"✅ Saved: {self.input_dir}/windows_features_balanced.npy")
        logger.info(f"✅ Saved: {self.input_dir}/windows_labels_balanced.npy")
        logger.info(f"   Shape: X={X.shape}, y={y.shape}")


def main():
    logger.info("=" * 70)
    logger.info("🎯 WINDOWS DATASET BALANCER")
    logger.info("=" * 70)
    
    balancer = WindowsDataBalancer()
    
    # Load original data
    X, y = balancer.load_data()
    
    # Balance dataset
    X_balanced, y_balanced = balancer.balance_dataset(X, y)
    
    # Save balanced data
    balancer.save_balanced_data(X_balanced, y_balanced)
    
    logger.info("\n✅ Dataset balancing complete!")
    logger.info(f"\n📊 Ready for training: {len(X_balanced):,} samples")
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())


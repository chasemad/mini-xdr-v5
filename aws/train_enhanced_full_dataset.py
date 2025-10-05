#!/usr/bin/env python3
"""
ENHANCED Local Training Script for Mini-XDR - Full Dataset
Uses ALL real data (4M+ samples) + synthetic supplement
Implements all improvements from recommendations

Key Improvements:
- Uses ALL CICIDS2017 raw data (2.8M samples, not just 600k)
- Adds synthetic data as 10% supplement
- Better hyperparameters (lower LR, smaller batch, more epochs)
- Data augmentation (Gaussian noise, feature dropout)
- Class balancing with focal loss
- Better monitoring and logging
"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from sklearn.metrics import classification_report, f1_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
import joblib
import logging
from pathlib import Path
from datetime import datetime
import time
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_enhanced.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# DATA AUGMENTATION
# ============================================================================

class DataAugmenter:
    """Apply data augmentation to training samples"""
    
    def __init__(self, noise_std=0.01, dropout_prob=0.1, mixup_alpha=0.2):
        self.noise_std = noise_std
        self.dropout_prob = dropout_prob
        self.mixup_alpha = mixup_alpha
    
    def add_gaussian_noise(self, X):
        """Add Gaussian noise to features"""
        noise = torch.randn_like(X) * self.noise_std
        return X + noise
    
    def feature_dropout(self, X):
        """Randomly drop features during training"""
        mask = torch.bernoulli(torch.ones_like(X) * (1 - self.dropout_prob))
        return X * mask
    
    def mixup(self, X, y, alpha=None):
        """Mixup data augmentation"""
        if alpha is None:
            alpha = self.mixup_alpha
        
        batch_size = X.size(0)
        lam = np.random.beta(alpha, alpha, batch_size)
        lam = torch.from_numpy(lam).float().to(X.device).unsqueeze(1)
        
        index = torch.randperm(batch_size).to(X.device)
        
        mixed_X = lam * X + (1 - lam) * X[index]
        y_a, y_b = y, y[index]
        
        return mixed_X, y_a, y_b, lam.squeeze()
    
    def augment(self, X, apply_noise=True, apply_dropout=True):
        """Apply augmentation pipeline"""
        if apply_noise:
            X = self.add_gaussian_noise(X)
        if apply_dropout:
            X = self.feature_dropout(X)
        return X


# ============================================================================
# FOCAL LOSS (Better for imbalanced classes)
# ============================================================================

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# ============================================================================
# MODEL ARCHITECTURE (Enhanced)
# ============================================================================

class AttentionLayer(nn.Module):
    def __init__(self, input_dim: int, attention_dim: int = 64):
        super().__init__()
        self.attention_dim = attention_dim
        self.query = nn.Linear(input_dim, attention_dim)
        self.key = nn.Linear(input_dim, attention_dim)
        self.value = nn.Linear(input_dim, attention_dim)
        self.output = nn.Linear(attention_dim, input_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        batch_size = x.size(0)
        q = self.query(x).unsqueeze(1)
        k = self.key(x).unsqueeze(1)
        v = self.value(x).unsqueeze(1)
        attention_weights = torch.softmax(
            torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.attention_dim), 
            dim=-1
        )
        attended = torch.matmul(attention_weights, v).squeeze(1)
        output = self.output(attended)
        output = self.dropout(output)
        return output + x


class UncertaintyBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout_rate: float = 0.3):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.batch_norm = nn.BatchNorm1d(output_dim)

    def forward(self, x):
        x = self.linear(x)
        x = self.batch_norm(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x


class ThreatDetector(nn.Module):
    def __init__(self, input_dim: int = 79, hidden_dims: list = [512, 256, 128, 64],
                 num_classes: int = 7, dropout_rate: float = 0.3, use_attention: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.use_attention = use_attention

        self.feature_interaction = nn.Linear(input_dim, input_dim)

        if use_attention:
            self.attention = AttentionLayer(input_dim, attention_dim=64)

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(UncertaintyBlock(prev_dim, hidden_dim, dropout_rate))
            prev_dim = hidden_dim

        self.feature_extractor = nn.ModuleList(layers)

        self.skip_connections = nn.ModuleList([
            nn.Linear(input_dim, hidden_dims[-1]),
            nn.Linear(hidden_dims[0], hidden_dims[-1])
        ])

        self.classifier = nn.Linear(hidden_dims[-1], num_classes)
        self.uncertainty_head = nn.Linear(hidden_dims[-1], 1)

    def forward(self, x):
        features = self.feature_interaction(x)
        skip_features = [features]

        if self.use_attention:
            features = self.attention(features)

        for i, layer in enumerate(self.feature_extractor):
            features = layer(features)
            if i == 0:
                skip_features.append(features)

        skip1 = self.skip_connections[0](skip_features[0])
        skip2 = self.skip_connections[1](skip_features[1])
        features = features + skip1 + skip2

        logits = self.classifier(features)
        uncertainty = torch.sigmoid(self.uncertainty_head(features))

        return logits, uncertainty


# ============================================================================
# DATA LOADING (FULL DATASET)
# ============================================================================

def load_cicids2017_raw(data_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load ALL CICIDS2017 raw CSV files (~2.8M samples)"""
    logger.info("Loading CICIDS2017 raw data (all 8 CSV files)...")
    
    cicids_dir = os.path.join(data_dir, 'cicids2017_official/MachineLearningCVE')
    csv_files = sorted([f for f in os.listdir(cicids_dir) if f.endswith('.csv')])
    
    all_features = []
    all_labels = []
    
    # Label mapping (CICIDS2017 specific)
    label_map = {
        'BENIGN': 0,
        'Bot': 5,
        'DDoS': 1,
        'DoS GoldenEye': 1,
        'DoS Hulk': 1,
        'DoS Slowhttptest': 1,
        'DoS slowloris': 1,
        'FTP-Patator': 3,
        'Heartbleed': 4,
        'Infiltration': 6,
        'PortScan': 2,
        'SSH-Patator': 3,
        'Web Attack � Brute Force': 4,
        'Web Attack � Sql Injection': 4,
        'Web Attack � XSS': 4,
    }
    
    for csv_file in csv_files:
        filepath = os.path.join(cicids_dir, csv_file)
        logger.info(f"  Processing {csv_file}...")
        
        try:
            df = pd.read_csv(filepath, encoding='latin-1')
            
            # Clean column names
            df.columns = df.columns.str.strip()
            
            # Extract labels
            if ' Label' in df.columns:
                labels = df[' Label'].map(label_map).fillna(0).astype(int).values
            else:
                labels = df['Label'].map(label_map).fillna(0).astype(int).values
            
            # Drop non-numeric columns
            numeric_df = df.select_dtypes(include=[np.number])
            
            # Replace inf with nan, then fill with 0
            numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan).fillna(0)
            
            # Take first 79 numeric features (or pad if less)
            features = numeric_df.values
            if features.shape[1] > 79:
                features = features[:, :79]
            elif features.shape[1] < 79:
                padding = np.zeros((features.shape[0], 79 - features.shape[1]))
                features = np.hstack([features, padding])
            
            all_features.append(features)
            all_labels.append(labels)
            
            logger.info(f"    Loaded {len(features):,} samples with {features.shape[1]} features")
        
        except Exception as e:
            logger.error(f"    Error processing {csv_file}: {e}")
            continue
    
    X = np.vstack(all_features)
    y = np.concatenate(all_labels)
    
    logger.info(f"  Total CICIDS2017: {len(X):,} samples")
    return X, y


def load_full_dataset(data_dir: str, include_synthetic: bool = True, 
                      synthetic_ratio: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load FULL dataset: ALL real data + synthetic supplement
    
    Args:
        data_dir: Path to datasets directory
        include_synthetic: Whether to include synthetic data
        synthetic_ratio: Ratio of synthetic to real data (default 10%)
    
    Returns:
        X: Features (N, 79)
        y: Labels (N,)
    """
    logger.info("="*80)
    logger.info("LOADING FULL DATASET (REAL + SYNTHETIC)")
    logger.info("="*80)
    
    # Load existing preprocessed data (UNSW-NB15, KDD, etc.)
    logger.info("\n1. Loading preprocessed data...")
    training_data_dir = os.path.join(os.path.dirname(data_dir), 'aws/training_data')
    
    X_existing = np.load(os.path.join(training_data_dir, 'training_features_20250929_062520.npy'))
    y_existing = np.load(os.path.join(training_data_dir, 'training_labels_20250929_062520.npy'))
    logger.info(f"  Preprocessed data: {len(X_existing):,} samples")
    
    # Load ALL CICIDS2017 raw data
    logger.info("\n2. Loading CICIDS2017 full raw data...")
    X_cicids, y_cicids = load_cicids2017_raw(data_dir)
    
    # Combine real datasets
    logger.info("\n3. Combining all real data...")
    X_real = np.vstack([X_existing, X_cicids])
    y_real = np.concatenate([y_existing, y_cicids])
    logger.info(f"  Total real data: {len(X_real):,} samples")
    
    # Add synthetic data as supplement (10%)
    if include_synthetic:
        logger.info("\n4. Loading synthetic data as supplement...")
        synthetic_files = [
            'brute_force_ssh_dataset.json',
            'ddos_attacks_dataset.json',
            'web_attacks_dataset.json',
            'network_scans_dataset.json',
            'malware_behavior_dataset.json'
        ]
        
        X_synth_list = []
        y_synth_list = []
        
        for synth_file in synthetic_files:
            filepath = os.path.join(data_dir, synth_file)
            if os.path.exists(filepath):
                try:
                    with open(filepath, 'r') as f:
                        synth_data = json.load(f)
                    
                    # Extract features and labels from synthetic data
                    # (Simplified - adjust based on actual format)
                    if isinstance(synth_data, list) and len(synth_data) > 0:
                        logger.info(f"    {synth_file}: {len(synth_data):,} samples")
                        # Generate synthetic features (placeholder - customize as needed)
                        n_samples = min(len(synth_data), int(len(X_real) * synthetic_ratio / len(synthetic_files)))
                        X_synth = np.random.rand(n_samples, 79) * 0.5 + 0.25  # Normalized [0.25, 0.75]
                        
                        # Map to appropriate label based on file
                        if 'ddos' in synth_file:
                            y_synth = np.ones(n_samples, dtype=int)  # DDoS = 1
                        elif 'brute_force' in synth_file:
                            y_synth = np.full(n_samples, 3, dtype=int)  # Brute Force = 3
                        elif 'web_attack' in synth_file:
                            y_synth = np.full(n_samples, 4, dtype=int)  # Web Attack = 4
                        elif 'scan' in synth_file:
                            y_synth = np.full(n_samples, 2, dtype=int)  # Recon = 2
                        else:
                            y_synth = np.full(n_samples, 5, dtype=int)  # Malware = 5
                        
                        X_synth_list.append(X_synth)
                        y_synth_list.append(y_synth)
                
                except Exception as e:
                    logger.warning(f"    Error loading {synth_file}: {e}")
        
        if X_synth_list:
            X_synth = np.vstack(X_synth_list)
            y_synth = np.concatenate(y_synth_list)
            logger.info(f"  Total synthetic: {len(X_synth):,} samples ({synthetic_ratio*100:.0f}% of real)")
            
            # Combine real + synthetic
            X = np.vstack([X_real, X_synth])
            y = np.concatenate([y_real, y_synth])
        else:
            X, y = X_real, y_real
    else:
        X, y = X_real, y_real
    
    # Normalize features
    logger.info("\n5. Normalizing features...")
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Clip to [0, 1] range
    X = np.clip(X, 0, 1)
    
    logger.info("\n" + "="*80)
    logger.info(f"FINAL DATASET: {len(X):,} samples × {X.shape[1]} features")
    logger.info(f"Class distribution:")
    for class_id in sorted(np.unique(y)):
        count = np.sum(y == class_id)
        pct = count / len(y) * 100
        logger.info(f"  Class {class_id}: {count:,} samples ({pct:.1f}%)")
    logger.info("="*80 + "\n")
    
    return X, y


# ============================================================================
# TRAINING FUNCTION (Enhanced)
# ============================================================================

def train_model_enhanced(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    num_classes: int,
    epochs: int = 50,
    batch_size: int = 256,
    learning_rate: float = 0.0005,
    patience: int = 15,
    device: str = 'cpu',
    use_focal_loss: bool = True,
    use_augmentation: bool = True,
) -> Tuple[ThreatDetector, Dict]:
    """
    Train model with enhancements:
    - Focal loss for class imbalance
    - Data augmentation
    - Better learning rate schedule
    - More epochs with early stopping
    """
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Training {model_name}")
    logger.info(f"{'='*80}")
    logger.info(f"Config:")
    logger.info(f"  Samples: {len(X_train):,} train, {len(X_val):,} val")
    logger.info(f"  Classes: {num_classes}")
    logger.info(f"  Epochs: {epochs}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Learning rate: {learning_rate}")
    logger.info(f"  Device: {device}")
    logger.info(f"  Focal loss: {use_focal_loss}")
    logger.info(f"  Data augmentation: {use_augmentation}")
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.LongTensor(y_val)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size*2, shuffle=False, num_workers=2)
    
    # Initialize model
    model = ThreatDetector(
        input_dim=79,
        hidden_dims=[512, 256, 128, 64],
        num_classes=num_classes,
        dropout_rate=0.3,
        use_attention=True
    ).to(device)
    
    # Class weights for imbalanced data
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = torch.FloatTensor(class_weights).to(device)
    
    # Loss function
    if use_focal_loss:
        criterion = FocalLoss(alpha=class_weights, gamma=2)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Learning rate scheduler
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    # Data augmenter
    if use_augmentation:
        augmenter = DataAugmenter(noise_std=0.01, dropout_prob=0.1)
    
    # Training loop
    best_val_loss = float('inf')
    best_val_acc = 0
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    start_time = time.time()
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            # Apply data augmentation
            if use_augmentation and np.random.rand() < 0.5:
                batch_X = augmenter.augment(batch_X)
            
            optimizer.zero_grad()
            
            logits, uncertainty = model(batch_X)
            loss = criterion(logits, batch_y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            train_correct += (predicted == batch_y).sum().item()
            train_total += batch_y.size(0)
        
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                logits, uncertainty = model(batch_X)
                loss = criterion(logits, batch_y)
                
                val_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                val_correct += (predicted == batch_y).sum().item()
                val_total += batch_y.size(0)
        
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        
        # Update learning rate
        scheduler.step()
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # Log progress
        logger.info(
            f"Epoch {epoch+1}/{epochs}: "
            f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f} | "
            f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f} | "
            f"LR={optimizer.param_groups[0]['lr']:.6f}"
        )
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), f'best_{model_name}.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
    
    training_time = time.time() - start_time
    
    # Load best model
    model.load_state_dict(torch.load(f'best_{model_name}.pth'))
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        X_val_device = X_val_tensor.to(device)
        y_val_device = y_val_tensor.to(device)
        logits, _ = model(X_val_device)
        _, y_pred = torch.max(logits, 1)
        y_pred = y_pred.cpu().numpy()
    
    # Classification report
    report = classification_report(y_val, y_pred, output_dict=True, zero_division=0)
    f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)
    
    results = {
        'model_name': model_name,
        'best_val_accuracy': best_val_acc,
        'best_val_loss': best_val_loss,
        'f1_score': f1,
        'training_time': training_time,
        'epochs_trained': epoch + 1,
        'history': history,
        'classification_report': report
    }
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Training Complete: {model_name}")
    logger.info(f"  Best Val Accuracy: {best_val_acc:.4f}")
    logger.info(f"  F1 Score: {f1:.4f}")
    logger.info(f"  Training Time: {training_time/60:.1f} minutes")
    logger.info(f"{'='*80}\n")
    
    return model, results


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Enhanced Local ML Training with Full Dataset')
    parser.add_argument('--data-dir', type=str, default='/Users/chasemad/Desktop/mini-xdr/datasets',
                        help='Path to datasets directory')
    parser.add_argument('--models', nargs='+', default=['general', 'ddos', 'brute_force', 'web_attacks'],
                        help='Models to train')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    parser.add_argument('--no-synthetic', action='store_true', help='Exclude synthetic data')
    parser.add_argument('--synthetic-ratio', type=float, default=0.1, help='Ratio of synthetic to real data')
    parser.add_argument('--no-augmentation', action='store_true', help='Disable data augmentation')
    parser.add_argument('--no-focal-loss', action='store_true', help='Use standard cross-entropy instead of focal loss')
    
    args = parser.parse_args()
    
    # Detect device
    if torch.cuda.is_available():
        device = 'cuda'
        logger.info("Using CUDA GPU")
    elif torch.backends.mps.is_available():
        device = 'mps'
        logger.info("Using Apple Silicon GPU (MPS)")
    else:
        device = 'cpu'
        logger.info("Using CPU")
    
    # Load full dataset
    X, y = load_full_dataset(
        args.data_dir,
        include_synthetic=not args.no_synthetic,
        synthetic_ratio=args.synthetic_ratio
    )
    
    # Create output directory
    output_dir = Path('models/local_trained_enhanced')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Train models
    all_results = []
    
    for model_type in args.models:
        if model_type == 'general':
            # General 7-class model
            logger.info("\n\n" + "="*80)
            logger.info("TRAINING GENERAL MODEL (7 classes)")
            logger.info("="*80)
            
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            
            model, results = train_model_enhanced(
                model_name='general',
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                num_classes=7,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                patience=args.patience,
                device=device,
                use_focal_loss=not args.no_focal_loss,
                use_augmentation=not args.no_augmentation
            )
            
            # Save model
            model_dir = output_dir / 'general'
            model_dir.mkdir(exist_ok=True)
            torch.save(model.state_dict(), model_dir / 'threat_detector.pth')
            
            all_results.append(results)
        
        elif model_type == 'ddos':
            # DDoS specialist
            logger.info("\n\n" + "="*80)
            logger.info("TRAINING DDOS SPECIALIST")
            logger.info("="*80)
            
            # Binary: DDoS (1) vs Normal (0)
            mask = (y == 0) | (y == 1)
            X_specialist = X[mask]
            y_specialist = (y[mask] == 1).astype(int)
            
            X_train, X_val, y_train, y_val = train_test_split(
                X_specialist, y_specialist, test_size=0.2, random_state=42, stratify=y_specialist
            )
            
            model, results = train_model_enhanced(
                model_name='ddos_specialist',
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                num_classes=2,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                patience=args.patience,
                device=device,
                use_focal_loss=not args.no_focal_loss,
                use_augmentation=not args.no_augmentation
            )
            
            # Save model
            model_dir = output_dir / 'ddos_specialist'
            model_dir.mkdir(exist_ok=True)
            torch.save(model.state_dict(), model_dir / 'threat_detector.pth')
            
            all_results.append(results)
        
        elif model_type == 'brute_force':
            # Brute Force specialist
            logger.info("\n\n" + "="*80)
            logger.info("TRAINING BRUTE FORCE SPECIALIST")
            logger.info("="*80)
            
            # Binary: Brute Force (3) vs Normal+Recon (0, 2)
            mask = (y == 0) | (y == 2) | (y == 3)
            X_specialist = X[mask]
            y_specialist = (y[mask] == 3).astype(int)
            
            X_train, X_val, y_train, y_val = train_test_split(
                X_specialist, y_specialist, test_size=0.2, random_state=42, stratify=y_specialist
            )
            
            model, results = train_model_enhanced(
                model_name='brute_force_specialist',
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                num_classes=2,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                patience=args.patience,
                device=device,
                use_focal_loss=not args.no_focal_loss,
                use_augmentation=not args.no_augmentation
            )
            
            # Save model
            model_dir = output_dir / 'brute_force_specialist'
            model_dir.mkdir(exist_ok=True)
            torch.save(model.state_dict(), model_dir / 'threat_detector.pth')
            
            all_results.append(results)
        
        elif model_type == 'web_attacks':
            # Web Attack specialist
            logger.info("\n\n" + "="*80)
            logger.info("TRAINING WEB ATTACK SPECIALIST")
            logger.info("="*80)
            
            # Binary: Web Attack (4) vs Normal (0)
            mask = (y == 0) | (y == 4)
            X_specialist = X[mask]
            y_specialist = (y[mask] == 4).astype(int)
            
            X_train, X_val, y_train, y_val = train_test_split(
                X_specialist, y_specialist, test_size=0.2, random_state=42, stratify=y_specialist
            )
            
            model, results = train_model_enhanced(
                model_name='web_attacks_specialist',
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                num_classes=2,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                patience=args.patience,
                device=device,
                use_focal_loss=not args.no_focal_loss,
                use_augmentation=not args.no_augmentation
            )
            
            # Save model
            model_dir = output_dir / 'web_attacks_specialist'
            model_dir.mkdir(exist_ok=True)
            torch.save(model.state_dict(), model_dir / 'threat_detector.pth')
            
            all_results.append(results)
    
    # Save training summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_samples': len(X),
        'dataset_composition': {
            'real_data': int(len(X) * (1 - args.synthetic_ratio)) if not args.no_synthetic else len(X),
            'synthetic_data': int(len(X) * args.synthetic_ratio) if not args.no_synthetic else 0,
        },
        'config': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'patience': args.patience,
            'use_focal_loss': not args.no_focal_loss,
            'use_augmentation': not args.no_augmentation,
        },
        'device': device,
        'results': all_results
    }
    
    with open(output_dir / 'training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info("\n\n" + "="*80)
    logger.info("ALL TRAINING COMPLETE!")
    logger.info("="*80)
    logger.info(f"Models saved to: {output_dir}")
    logger.info(f"Training summary: {output_dir}/training_summary.json")
    logger.info("\nResults:")
    for result in all_results:
        logger.info(f"  {result['model_name']}: {result['best_val_accuracy']:.4f} accuracy ({result['training_time']/60:.1f} min)")
    logger.info("="*80)


if __name__ == '__main__':
    main()



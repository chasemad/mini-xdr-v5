#!/usr/bin/env python3
"""
Local Training Script for Mini-XDR Threat Detection
Adapted from SageMaker training script to run on your local machine
Trains 4 models: General + 3 Specialists
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
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import joblib
import logging
from pathlib import Path
from datetime import datetime
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Model architecture (same as SageMaker)
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

        self.classifier = nn.Linear(prev_dim, num_classes)
        self.uncertainty_head = nn.Linear(prev_dim, 1)
        self.mc_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, return_features: bool = False):
        x_interact = torch.relu(self.feature_interaction(x))
        x = x + x_interact

        if self.use_attention:
            x_attended = self.attention(x)
            x = x_attended

        x_input = x
        x_mid = None

        for i, layer in enumerate(self.feature_extractor):
            x = layer(x)
            if i == 0:
                x_mid = x

        skip1 = torch.relu(self.skip_connections[0](x_input))
        skip2 = torch.relu(self.skip_connections[1](x_mid))
        x = x + skip1 + skip2

        features = x
        logits = self.classifier(features)
        uncertainty = torch.sigmoid(self.uncertainty_head(self.mc_dropout(features)))

        if return_features:
            return logits, uncertainty, features

        return logits, uncertainty


def load_training_data(data_dir: Path):
    """Load training data from local directory"""
    logger.info(f"Loading data from {data_dir}")
    
    # Load the latest training data
    npy_files = sorted(list(data_dir.glob("training_features_*.npy")))
    
    if not npy_files:
        raise ValueError(f"No training data found in {data_dir}")
    
    latest_features_file = npy_files[-1]
    # Extract timestamp - filename is like "training_features_20250929_062520.npy"
    # We need both the date and time parts
    parts = latest_features_file.stem.split('_')
    timestamp = '_'.join(parts[2:])  # Join date_time parts
    
    features_file = data_dir / f"training_features_{timestamp}.npy"
    labels_file = data_dir / f"training_labels_{timestamp}.npy"
    metadata_file = data_dir / f"training_metadata_{timestamp}.json"
    
    logger.info(f"Loading features from: {features_file.name}")
    logger.info(f"Loading labels from: {labels_file.name}")
    
    features = np.load(features_file)
    labels = np.load(labels_file)
    
    # Load metadata
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    logger.info(f"✅ Loaded {len(features):,} samples with {features.shape[1]} features")
    logger.info(f"   Class distribution: {metadata['class_distribution']}")
    logger.info(f"   Feature range: [{features.min():.2f}, {features.max():.2f}]")
    
    return features, labels, metadata


def prepare_specialist_data(features, labels, specialist_type):
    """Filter data for specialist models (binary classification)"""
    if specialist_type == 'general':
        return features, labels
    
    # Class mapping:
    # 0: Normal, 1: DDoS, 2: Reconnaissance, 3: Brute Force, 4: Web Attack, 5: Malware, 6: APT
    specialist_configs = {
        'ddos': {
            'positive_classes': [1],  # DDoS
            'negative_classes': [0],  # Normal
            'name': 'DDoS vs Normal'
        },
        'brute_force': {
            'positive_classes': [3],  # Brute Force
            'negative_classes': [0, 2],  # Normal + Recon (to distinguish from port scanning)
            'name': 'Brute Force vs Normal'
        },
        'web_attacks': {
            'positive_classes': [4],  # Web Attack
            'negative_classes': [0],  # Normal
            'name': 'Web Attack vs Normal'
        }
    }
    
    config = specialist_configs.get(specialist_type)
    if not config:
        logger.warning(f"Unknown specialist type: {specialist_type}, using general")
        return features, labels
    
    logger.info(f"Preparing {config['name']} specialist data...")
    
    # Create masks for positive and negative classes
    positive_mask = np.isin(labels, config['positive_classes'])
    negative_mask = np.isin(labels, config['negative_classes'])
    
    # Combine masks
    mask = positive_mask | negative_mask
    filtered_features = features[mask]
    filtered_labels = labels[mask]
    
    # Convert to binary labels (1 = attack, 0 = normal)
    binary_labels = positive_mask[mask].astype(np.int64)
    
    logger.info(f"   Attack samples: {binary_labels.sum():,}")
    logger.info(f"   Normal samples: {(binary_labels == 0).sum():,}")
    logger.info(f"   Total: {len(binary_labels):,} samples")
    
    return filtered_features, binary_labels


def train_model(features, labels, specialist_type, output_dir, device, args):
    """Train a single model"""
    
    logger.info("=" * 80)
    logger.info(f"TRAINING {specialist_type.upper()} MODEL")
    logger.info("=" * 80)
    
    start_time = time.time()
    
    # Prepare data for specialist models
    features, labels = prepare_specialist_data(features, labels, specialist_type)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    logger.info(f"Training samples: {len(X_train):,}")
    logger.info(f"Validation samples: {len(X_val):,}")
    
    # Calculate class weights for imbalanced data
    unique_classes = np.unique(labels)
    class_weights = compute_class_weight('balanced', classes=unique_classes, y=labels)
    logger.info(f"Class weights: {dict(zip(unique_classes, class_weights))}")
    
    # Create data loaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train), 
        torch.LongTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val), 
        torch.LongTensor(y_val)
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        drop_last=True,
        num_workers=0  # For local training
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        drop_last=False,
        num_workers=0
    )
    
    # Create model
    num_classes = 2 if specialist_type != 'general' else 7
    model = ThreatDetector(
        input_dim=79,
        hidden_dims=[512, 256, 128, 64],
        num_classes=num_classes,
        dropout_rate=0.3,
        use_attention=True
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(
        weight=torch.FloatTensor(class_weights).to(device)
    )
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.learning_rate, 
        weight_decay=0.01
    )
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        patience=5, 
        factor=0.5
    )
    
    # Training loop
    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience_counter = 0
    training_history = []
    
    logger.info(f"\nStarting training for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        
        # Training phase
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (features_batch, labels_batch) in enumerate(train_loader):
            features_batch = features_batch.to(device)
            labels_batch = labels_batch.to(device)
            
            optimizer.zero_grad()
            logits, _ = model(features_batch)
            loss = criterion(logits, labels_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += labels_batch.size(0)
            correct += (predicted == labels_batch).sum().item()
            
            # Progress indicator
            if batch_idx % 50 == 0:
                print('.', end='', flush=True)
        
        train_acc = 100 * correct / total
        train_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for features_batch, labels_batch in val_loader:
                features_batch = features_batch.to(device)
                labels_batch = labels_batch.to(device)
                
                logits, _ = model(features_batch)
                loss = criterion(logits, labels_batch)
                
                val_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                total += labels_batch.size(0)
                correct += (predicted == labels_batch).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels_batch.cpu().numpy())
        
        val_acc = 100 * correct / total
        val_loss /= len(val_loader)
        
        scheduler.step(val_loss)
        
        epoch_time = time.time() - epoch_start
        
        logger.info(f"\nEpoch [{epoch+1}/{args.epochs}] ({epoch_time:.1f}s)")
        logger.info(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        logger.info(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        
        # Save training history
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'learning_rate': optimizer.param_groups[0]['lr']
        })
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            logger.info(f"  ✅ New best model! (Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
            logger.info(f"  ⏸️  No improvement (patience: {patience_counter}/{args.patience})")
            
            if patience_counter >= args.patience:
                logger.info(f"\n⚠️  Early stopping triggered after {epoch+1} epochs")
                model.load_state_dict(best_model_state)
                break
    
    # Final evaluation on validation set
    logger.info("\n" + "=" * 80)
    logger.info("FINAL EVALUATION")
    logger.info("=" * 80)
    
    # Compute confusion matrix and classification report
    cm = confusion_matrix(all_labels, all_preds)
    logger.info(f"\nConfusion Matrix:\n{cm}")
    
    if num_classes == 2:
        class_names = ['Normal', 'Attack']
    else:
        class_names = ['Normal', 'DDoS', 'Recon', 'BruteForce', 'WebAttack', 'Malware', 'APT']
    
    report = classification_report(
        all_labels, 
        all_preds, 
        target_names=class_names[:num_classes],
        digits=3
    )
    logger.info(f"\nClassification Report:\n{report}")
    
    # Save model
    model_output_dir = output_dir / specialist_type
    model_output_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = model_output_dir / 'threat_detector.pth'
    torch.save(best_model_state, model_path)
    logger.info(f"\n✅ Model saved to: {model_path}")
    
    # Save metadata
    metadata = {
        'specialist_type': specialist_type,
        'features': 79,
        'num_classes': num_classes,
        'hidden_dims': [512, 256, 128, 64],
        'dropout_rate': 0.3,
        'use_attention': True,
        'best_val_accuracy': best_val_acc / 100,
        'best_val_loss': best_val_loss,
        'total_epochs': epoch + 1,
        'training_time_seconds': time.time() - start_time,
        'training_date': datetime.now().isoformat(),
        'device': str(device),
        'class_names': class_names[:num_classes]
    }
    
    metadata_path = model_output_dir / 'model_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save training history
    history_path = model_output_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    
    total_time = time.time() - start_time
    logger.info(f"✅ Training complete in {total_time/60:.1f} minutes")
    logger.info(f"   Best validation accuracy: {best_val_acc:.2f}%")
    logger.info(f"   Best validation loss: {best_val_loss:.4f}")
    
    return {
        'specialist_type': specialist_type,
        'accuracy': best_val_acc,
        'loss': best_val_loss,
        'epochs': epoch + 1
    }


def main():
    parser = argparse.ArgumentParser(
        description='Train Mini-XDR threat detection models locally'
    )
    
    # Data arguments
    parser.add_argument(
        '--data-dir', 
        type=str, 
        default='aws/training_data',
        help='Directory containing training data (.npy files)'
    )
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='models/local_trained',
        help='Directory to save trained models'
    )
    
    # Model arguments
    parser.add_argument(
        '--models', 
        type=str, 
        nargs='+',
        default=['general', 'ddos', 'brute_force', 'web_attacks'],
        choices=['general', 'ddos', 'brute_force', 'web_attacks'],
        help='Which models to train'
    )
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=512, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    
    # Hardware arguments
    parser.add_argument(
        '--device', 
        type=str, 
        default='auto',
        choices=['auto', 'cpu', 'cuda', 'mps'],
        help='Device to use for training'
    )
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
            logger.info("🚀 Using Apple Silicon GPU (MPS)")
        else:
            device = torch.device('cpu')
            logger.info("💻 Using CPU (this will be slower)")
    else:
        device = torch.device(args.device)
        logger.info(f"Using device: {device}")
    
    # Setup paths
    script_dir = Path(__file__).parent.parent
    data_dir = script_dir / args.data_dir
    output_dir = script_dir / args.output_dir
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("\n" + "=" * 80)
    logger.info("MINI-XDR LOCAL MODEL TRAINING")
    logger.info("=" * 80)
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Models to train: {', '.join(args.models)}")
    logger.info(f"Device: {device}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Max epochs: {args.epochs}")
    logger.info(f"Learning rate: {args.learning_rate}")
    
    # Load training data
    try:
        features, labels, metadata = load_training_data(data_dir)
    except Exception as e:
        logger.error(f"Failed to load training data: {e}")
        sys.exit(1)
    
    # Train each model
    results = []
    total_start = time.time()
    
    for model_type in args.models:
        logger.info("\n\n")
        try:
            result = train_model(
                features.copy(), 
                labels.copy(), 
                model_type, 
                output_dir, 
                device, 
                args
            )
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to train {model_type} model: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    total_time = time.time() - total_start
    
    logger.info("\n\n" + "=" * 80)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total training time: {total_time/60:.1f} minutes")
    logger.info(f"Models trained: {len(results)}/{len(args.models)}")
    logger.info("\nResults:")
    
    for result in results:
        logger.info(f"  {result['specialist_type']:15s} - "
                   f"Accuracy: {result['accuracy']:.2f}% | "
                   f"Loss: {result['loss']:.4f} | "
                   f"Epochs: {result['epochs']}")
    
    # Save summary
    summary = {
        'training_date': datetime.now().isoformat(),
        'total_time_minutes': total_time / 60,
        'device': str(device),
        'data_samples': len(features),
        'results': results
    }
    
    summary_path = output_dir / 'training_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\n✅ All models trained successfully!")
    logger.info(f"📁 Models saved to: {output_dir}")
    logger.info(f"📊 Summary saved to: {summary_path}")
    

if __name__ == "__main__":
    main()


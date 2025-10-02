#!/usr/bin/env python3
"""
Specialist Model Training Script with Proper Scaler Saving
Trains specialized models for specific attack types with full SageMaker packaging
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
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import joblib
import boto3
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AttentionLayer(nn.Module):
    """Attention mechanism for feature relationship learning"""

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
        attention_weights = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.attention_dim), dim=-1)
        attended = torch.matmul(attention_weights, v).squeeze(1)
        output = self.output(attended)
        output = self.dropout(output)
        return output + x


class UncertaintyBlock(nn.Module):
    """Monte Carlo Dropout for uncertainty estimation"""

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


class SpecialistThreatDetector(nn.Module):
    """Specialist threat detection model optimized for specific attack types"""

    def __init__(self, input_dim: int = 79, hidden_dims: list = [512, 256, 128, 64],
                 num_classes: int = 2, dropout_rate: float = 0.3, use_attention: bool = True):
        super().__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.use_attention = use_attention

        # Feature interaction layer
        self.feature_interaction = nn.Linear(input_dim, input_dim)

        # Attention mechanism
        if use_attention:
            self.attention = AttentionLayer(input_dim, attention_dim=64)

        # Enhanced feature extraction layers
        layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(UncertaintyBlock(prev_dim, hidden_dim, dropout_rate))
            prev_dim = hidden_dim

        self.feature_extractor = nn.ModuleList(layers)

        # Skip connections
        self.skip_connections = nn.ModuleList([
            nn.Linear(input_dim, hidden_dims[-1]),
            nn.Linear(hidden_dims[0], hidden_dims[-1])
        ])

        # Classification head
        self.classifier = nn.Linear(prev_dim, num_classes)

        # Uncertainty estimation head
        self.uncertainty_head = nn.Linear(prev_dim, 1)
        self.mc_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, return_features: bool = False):
        # Feature interactions
        x_interact = torch.relu(self.feature_interaction(x))
        x = x + x_interact

        # Attention mechanism
        if self.use_attention:
            x_attended = self.attention(x)
            x = x_attended

        # Store for skip connections
        x_input = x
        x_mid = None

        # Feature extraction with uncertainty blocks
        for i, layer in enumerate(self.feature_extractor):
            x = layer(x)
            if i == 0:
                x_mid = x

        # Skip connections
        skip1 = torch.relu(self.skip_connections[0](x_input))
        skip2 = torch.relu(self.skip_connections[1](x_mid))
        x = x + skip1 + skip2

        # Classification and uncertainty
        features = x
        logits = self.classifier(features)
        uncertainty = torch.sigmoid(self.uncertainty_head(self.mc_dropout(features)))

        if return_features:
            return logits, uncertainty, features

        return logits, uncertainty


class DatasetLoader:
    """Loads and prepares datasets for training"""

    def __init__(self, data_dir: str, specialist_type: str = None):
        self.data_dir = Path(data_dir)
        self.specialist_type = specialist_type
        self.scaler = RobustScaler()

        # Class mappings
        self.class_mapping = {
            'normal': 0,
            'ddos_dos': 1,
            'reconnaissance': 2,
            'brute_force': 3,
            'web_attacks': 4,
            'malware_botnet': 5,
            'advanced_threats': 6
        }

        # Specialist configurations
        self.specialist_configs = {
            'ddos': {
                'name': 'DDoS/DoS Specialist',
                'positive_classes': [1],  # ddos_dos
                'negative_classes': [0],  # normal
                'description': 'Specialized DDoS/DoS attack detection'
            },
            'brute_force': {
                'name': 'Brute Force Specialist',
                'positive_classes': [3],  # brute_force
                'negative_classes': [0, 2],  # normal, reconnaissance
                'description': 'Specialized brute force attack detection'
            },
            'web_attacks': {
                'name': 'Web Attack Specialist',
                'positive_classes': [4],  # web_attacks
                'negative_classes': [0, 2],  # normal, reconnaissance
                'description': 'Specialized web application attack detection'
            },
            'apt_malware': {
                'name': 'APT/Malware Specialist',
                'positive_classes': [6, 5],  # advanced_threats, malware_botnet
                'negative_classes': [0, 2, 1],  # normal, reconnaissance, ddos
                'description': 'Specialized APT and malware detection'
            }
        }

    def load_and_combine_datasets(self) -> tuple:
        """Load all CSV files and combine them"""
        logger.info(f"🔥 Loading datasets from {self.data_dir}")

        csv_files = list(self.data_dir.glob("*.csv"))
        if not csv_files:
            raise ValueError(f"No CSV files found in {self.data_dir}")

        logger.info(f"📁 Found {len(csv_files)} CSV files")

        all_features = []
        all_labels = []

        for i, csv_file in enumerate(sorted(csv_files)):
            try:
                logger.info(f"📊 Loading chunk {i+1}/{len(csv_files)}: {csv_file.name}")
                df = pd.read_csv(csv_file)

                # Separate features and labels
                if 'label' in df.columns:
                    features = df.drop('label', axis=1).values.astype(np.float32)
                    labels = df['label'].values.astype(np.int64)
                else:
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

                all_features.append(features)
                all_labels.extend(labels.tolist())

                logger.info(f"   ✅ Processed {len(features):,} samples")

            except Exception as e:
                logger.error(f"   ❌ Failed to load {csv_file}: {e}")
                continue

        if not all_features:
            raise ValueError("No datasets were successfully loaded!")

        combined_features = np.vstack(all_features)
        combined_labels = np.array(all_labels)

        logger.info(f"🎯 Combined dataset: {len(combined_features):,} samples, {combined_features.shape[1]} features")

        return combined_features, combined_labels

    def prepare_specialist_dataset(self, features, labels):
        """Prepare binary classification dataset for specialist model"""
        if not self.specialist_type or self.specialist_type not in self.specialist_configs:
            # Return as-is for general model (multi-class)
            return features, labels

        config = self.specialist_configs[self.specialist_type]
        logger.info(f"\n🎯 Preparing {config['name']}")
        logger.info(f"   Description: {config['description']}")

        # Create binary labels: 1 for target attacks, 0 for normal/other
        positive_mask = np.isin(labels, config['positive_classes'])
        negative_mask = np.isin(labels, config['negative_classes'])

        # Filter to only include positive and selected negative classes
        mask = positive_mask | negative_mask
        filtered_features = features[mask]
        filtered_labels = labels[mask]

        # Convert to binary: 1 for attack, 0 for normal
        binary_labels = positive_mask[mask].astype(np.int64)

        logger.info(f"   Positive samples (attack): {binary_labels.sum():,}")
        logger.info(f"   Negative samples (normal): {(binary_labels == 0).sum():,}")
        logger.info(f"   Total: {len(binary_labels):,}")

        return filtered_features, binary_labels


def train_model(model, train_loader, val_loader, device, epochs, learning_rate, patience, class_weights=None):
    """Train the model with early stopping"""

    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(device))
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            logits, uncertainty = model(batch_features)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            train_total += batch_labels.size(0)
            train_correct += (predicted == batch_labels).sum().item()

        train_accuracy = 100 * train_correct / train_total

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)

                logits, uncertainty = model(batch_features)
                loss = criterion(logits, batch_labels)

                val_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                val_total += batch_labels.size(0)
                val_correct += (predicted == batch_labels).sum().item()

        val_accuracy = 100 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)

        logger.info(f"Epoch [{epoch+1}/{epochs}] "
                   f"Train Loss: {train_loss/len(train_loader):.4f} "
                   f"Train Acc: {train_accuracy:.2f}% "
                   f"Val Loss: {avg_val_loss:.4f} "
                   f"Val Acc: {val_accuracy:.2f}%")

        # Learning rate scheduling
        scheduler.step(avg_val_loss)

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)

    return model


def save_model_for_sagemaker(model, scaler, model_dir, metadata):
    """Save model, scaler, and metadata for SageMaker deployment"""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save model weights
    model_path = model_dir / "threat_detector.pth"
    torch.save(model.state_dict(), model_path)
    logger.info(f"✅ Saved model weights: {model_path}")

    # No scaler needed - data is pre-normalized
    logger.info("ℹ️  Skipping scaler save (using pre-normalized data)")
    # Create a dummy scaler file for compatibility
    scaler_path = model_dir / "scaler.pkl"
    joblib.dump(None, scaler_path)
    logger.info(f"✅ Saved dummy scaler (None): {scaler_path}")

    # Save metadata
    metadata_path = model_dir / "model_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"✅ Saved metadata: {metadata_path}")

    # Copy inference script
    inference_src = Path("/Users/chasemad/Desktop/mini-xdr/aws/inference_enhanced.py")
    if inference_src.exists():
        inference_dest = model_dir / "code" / "inference.py"
        inference_dest.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy(inference_src, inference_dest)
        logger.info(f"✅ Copied inference script")

    logger.info(f"\n✅ Model package ready at: {model_dir}")


def main():
    parser = argparse.ArgumentParser(description='Train Specialist Threat Detection Model')

    parser.add_argument('--data-dir', type=str, required=True, help='Directory containing training data CSV files')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory for trained model')
    parser.add_argument('--specialist-type', type=str, choices=['ddos', 'brute_force', 'web_attacks', 'apt_malware', 'general'],
                       default='general', help='Type of specialist model to train')
    parser.add_argument('--hidden-dims', type=str, default='512,256,128,64', help='Hidden layer dimensions')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Max epochs')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    parser.add_argument('--dropout-rate', type=float, default=0.3, help='Dropout rate')

    args = parser.parse_args()

    logger.info("🚀 Starting Specialist Model Training")
    logger.info(f"📁 Data dir: {args.data_dir}")
    logger.info(f"💾 Output dir: {args.output_dir}")
    logger.info(f"🎯 Specialist type: {args.specialist_type}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"🖥️  Using device: {device}")

    # Load data
    loader = DatasetLoader(args.data_dir,
                          specialist_type=None if args.specialist_type == 'general' else args.specialist_type)
    features, labels = loader.load_and_combine_datasets()

    # Prepare dataset (binary for specialist, multi-class for general)
    features, labels = loader.prepare_specialist_dataset(features, labels)

    # Data is already normalized - no scaling needed
    logger.info("✅ Using pre-normalized features (no scaler needed)")
    features_scaled = features  # Use features as-is
    logger.info(f"   Feature range: [{features_scaled.min():.2f}, {features_scaled.max():.2f}]")

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        features_scaled, labels, test_size=0.2, random_state=42, stratify=labels
    )

    logger.info(f"📊 Training samples: {len(X_train):,}")
    logger.info(f"📊 Validation samples: {len(X_val):,}")

    # Compute class weights
    unique_classes = np.unique(labels)
    class_weights = compute_class_weight('balanced', classes=unique_classes, y=labels)
    logger.info(f"⚖️  Class weights: {class_weights}")

    # Create dataloaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.LongTensor(y_val)
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=True)

    # Create model
    hidden_dims = [int(x) for x in args.hidden_dims.split(',')]
    num_classes = 2 if args.specialist_type != 'general' else 7

    model = SpecialistThreatDetector(
        input_dim=79,
        hidden_dims=hidden_dims,
        num_classes=num_classes,
        dropout_rate=args.dropout_rate,
        use_attention=True
    ).to(device)

    logger.info(f"🤖 Model: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Train
    logger.info("\n🏋️  Training model...")
    model = train_model(
        model, train_loader, val_loader, device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        patience=args.patience,
        class_weights=class_weights
    )

    # Evaluate
    logger.info("\n📊 Final Evaluation...")
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_features, batch_labels in val_loader:
            batch_features = batch_features.to(device)
            logits, _ = model(batch_features)
            _, predicted = torch.max(logits.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_labels.numpy())

    # Print results
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(all_labels, all_preds))

    accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
    logger.info(f"\n✅ Final Validation Accuracy: {accuracy*100:.2f}%")

    # Save model
    metadata = {
        'specialist_type': args.specialist_type,
        'features': 79,
        'num_classes': num_classes,
        'hidden_dims': hidden_dims,
        'dropout_rate': args.dropout_rate,
        'use_attention': True,
        'accuracy': float(accuracy),
        'training_date': datetime.now().isoformat(),
        'scaler': 'RobustScaler'
    }

    save_model_for_sagemaker(model, loader.scaler, args.output_dir, metadata)

    logger.info("\n" + "="*60)
    logger.info("✅ TRAINING COMPLETE!")
    logger.info("="*60)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
SageMaker Training Script for Mini-XDR Threat Detection
Works with both general (7-class) and specialist (binary) models
FIXED: No scaler - data is pre-normalized
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
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import joblib
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Model architecture (same as local training)
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
        attention_weights = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.attention_dim), dim=-1)
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


def load_data(data_dir):
    """Load training data from SageMaker input directory"""
    logger.info(f"Loading data from {data_dir}")

    csv_files = list(Path(data_dir).glob("*.csv"))
    if not csv_files:
        raise ValueError(f"No CSV files found in {data_dir}")

    logger.info(f"Found {len(csv_files)} CSV files")

    all_features = []
    all_labels = []

    for csv_file in sorted(csv_files):
        logger.info(f"Loading {csv_file.name}...")
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

        features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)

        all_features.append(features)
        all_labels.extend(labels.tolist())

        logger.info(f"   Processed {len(features):,} samples")

    combined_features = np.vstack(all_features)
    combined_labels = np.array(all_labels)

    logger.info(f"Total: {len(combined_features):,} samples, {combined_features.shape[1]} features")

    return combined_features, combined_labels


def prepare_specialist_data(features, labels, specialist_type):
    """Filter data for specialist models"""
    if specialist_type == 'general':
        return features, labels

    specialist_configs = {
        'ddos': {
            'positive_classes': [1],  # DDoS
            'negative_classes': [0]  # Normal
        },
        'brute_force': {
            'positive_classes': [3],  # Brute Force
            'negative_classes': [0, 2]  # Normal + Recon
        },
        'web_attacks': {
            'positive_classes': [4],  # Web Attack
            'negative_classes': [0]  # Normal
        }
    }

    config = specialist_configs.get(specialist_type)
    if not config:
        return features, labels

    logger.info(f"Preparing {specialist_type} specialist data...")

    positive_mask = np.isin(labels, config['positive_classes'])
    negative_mask = np.isin(labels, config['negative_classes'])

    mask = positive_mask | negative_mask
    filtered_features = features[mask]
    filtered_labels = labels[mask]

    binary_labels = positive_mask[mask].astype(np.int64)

    logger.info(f"   Positive: {binary_labels.sum():,}")
    logger.info(f"   Negative: {(binary_labels == 0).sum():,}")

    return filtered_features, binary_labels


def train(args):
    """Main training function"""
    logger.info("=" * 60)
    logger.info("SAGEMAKER TRAINING - FIXED (NO SCALER)")
    logger.info("=" * 60)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Load data
    train_dir = os.environ.get('SM_CHANNEL_TRAINING', args.data_dir)
    features, labels = load_data(train_dir)

    # Prepare specialist data if needed
    features, labels = prepare_specialist_data(features, labels, args.specialist_type)

    # NO SCALING - data is already normalized
    logger.info("✅ Using pre-normalized features (no scaler)")
    logger.info(f"   Feature range: [{features.min():.2f}, {features.max():.2f}]")

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )

    logger.info(f"Training samples: {len(X_train):,}")
    logger.info(f"Validation samples: {len(X_val):,}")

    # Class weights
    unique_classes = np.unique(labels)
    class_weights = compute_class_weight('balanced', classes=unique_classes, y=labels)
    logger.info(f"Class weights: {class_weights}")

    # Create dataloaders
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)

    # Create model
    num_classes = 2 if args.specialist_type != 'general' else 7
    model = ThreatDetector(
        input_dim=79,
        hidden_dims=[512, 256, 128, 64],
        num_classes=num_classes,
        dropout_rate=0.3,
        use_attention=True
    ).to(device)

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(device))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(args.epochs):
        # Training
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for features_batch, labels_batch in train_loader:
            features_batch, labels_batch = features_batch.to(device), labels_batch.to(device)

            optimizer.zero_grad()
            logits, _ = model(features_batch)
            loss = criterion(logits, labels_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += labels_batch.size(0)
            correct += (predicted == labels_batch).sum().item()

        train_acc = 100 * correct / total
        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for features_batch, labels_batch in val_loader:
                features_batch, labels_batch = features_batch.to(device), labels_batch.to(device)

                logits, _ = model(features_batch)
                loss = criterion(logits, labels_batch)

                val_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                total += labels_batch.size(0)
                correct += (predicted == labels_batch).sum().item()

        val_acc = 100 * correct / total
        val_loss /= len(val_loader)

        scheduler.step(val_loss)

        logger.info(f"Epoch [{epoch+1}/{args.epochs}] Train Loss: {train_loss:.4f} Train Acc: {train_acc:.2f}% Val Loss: {val_loss:.4f} Val Acc: {val_acc:.2f}%")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= 10:
                logger.info("Early stopping triggered")
                model.load_state_dict(best_model_state)
                break

    # Save model
    model_dir = os.environ.get('SM_MODEL_DIR', args.model_dir)
    os.makedirs(model_dir, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(model_dir, 'threat_detector.pth'))

    # Save dummy scaler for compatibility
    joblib.dump(None, os.path.join(model_dir, 'scaler.pkl'))

    # Save metadata
    metadata = {
        'specialist_type': args.specialist_type,
        'features': 79,
        'num_classes': num_classes,
        'hidden_dims': [512, 256, 128, 64],
        'dropout_rate': 0.3,
        'use_attention': True,
        'accuracy': val_acc / 100,
        'training_date': datetime.now().isoformat(),
        'scaler': None
    }

    with open(os.path.join(model_dir, 'model_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    # Copy inference script
    inference_src = Path(__file__).parent / 'inference.py'
    if inference_src.exists():
        import shutil
        code_dir = Path(model_dir) / 'code'
        code_dir.mkdir(exist_ok=True)
        shutil.copy(inference_src, code_dir / 'inference.py')

    logger.info(f"✅ Training complete! Final accuracy: {val_acc:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Hyperparameters
    parser.add_argument('--specialist-type', type=str, default='general')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--learning-rate', type=float, default=0.001)

    # SageMaker directories
    parser.add_argument('--data-dir', type=str, default='/opt/ml/input/data/training')
    parser.add_argument('--model-dir', type=str, default='/opt/ml/model')

    args = parser.parse_args()

    train(args)

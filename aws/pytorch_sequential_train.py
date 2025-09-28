#!/usr/bin/env python3
"""
Sequential/Temporal PyTorch Training for Mini-XDR
LSTM + Transformer models for attack sequence detection
"""

import os
import json
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib
import boto3
import logging
import time
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AttackSequenceDataset(Dataset):
    """
    Dataset for sequential attack pattern detection
    Creates time series from network flow data
    """
    def __init__(self, X, y=None, sequence_length=50, stride=10):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y) if y is not None else None
        self.sequence_length = sequence_length
        self.stride = stride

        # Create sequences
        self.sequences = []
        self.labels = []

        for i in range(0, len(self.X) - sequence_length + 1, stride):
            seq = self.X[i:i + sequence_length]
            self.sequences.append(seq)

            if self.y is not None:
                # Use majority vote or last label in sequence
                label = self.y[i + sequence_length - 1]
                self.labels.append(label)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        if self.labels:
            return self.sequences[idx], self.labels[idx]
        return self.sequences[idx]

class LSTMAttentionDetector(nn.Module):
    """
    LSTM with attention mechanism for sequential threat detection
    """
    def __init__(self, input_dim, hidden_dim=256, num_layers=3, num_classes=2, dropout=0.3):
        super(LSTMAttentionDetector, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout,
            bidirectional=True
        )

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            hidden_dim * 2, num_heads=8,
            dropout=dropout, batch_first=True
        )

        # Position encoding for temporal awareness
        self.position_encoding = PositionalEncoding(hidden_dim * 2, dropout)

        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

        # Global attention pooling
        self.attention_pooling = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        # LSTM processing
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Add positional encoding
        lstm_out = self.position_encoding(lstm_out)

        # Self-attention
        attn_out, attention_weights = self.attention(lstm_out, lstm_out, lstm_out)

        # Attention pooling
        attention_scores = torch.softmax(self.attention_pooling(attn_out), dim=1)
        pooled = torch.sum(attn_out * attention_scores, dim=1)

        # Classification
        output = self.classifier(pooled)

        return output, attention_weights

class TransformerThreatDetector(nn.Module):
    """
    Pure Transformer model for attack sequence detection
    """
    def __init__(self, input_dim, d_model=512, nhead=8, num_layers=6, num_classes=2, dropout=0.1):
        super(TransformerThreatDetector, self).__init__()

        self.d_model = d_model
        self.input_projection = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # Transformer encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=2048,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers)

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # Project to model dimension
        x = self.input_projection(x)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer processing
        transformer_out = self.transformer(x)

        # Global pooling
        pooled = self.global_pool(transformer_out.transpose(1, 2)).squeeze(-1)

        # Classification
        output = self.classifier(pooled)

        return output

class PositionalEncoding(nn.Module):
    """
    Positional encoding for temporal sequences
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)

def load_sequential_data(input_path, sequence_length=50):
    """Load data and prepare for sequential modeling"""
    logger.info(f"🔄 Loading data for sequential modeling (seq_len={sequence_length})")

    # Load the dataset (reuse from deep learning script)
    if input_path.startswith('s3://'):
        s3 = boto3.client('s3')
        parts = input_path.replace('s3://', '').split('/')
        bucket = parts[0]
        prefix = '/'.join(parts[1:]) if len(parts) > 1 else ''

        response = s3.list_objects_v2(Bucket=bucket, Prefix=f"{prefix}/train_chunk_")
        train_files = [obj['Key'] for obj in response.get('Contents', []) if obj['Key'].endswith('.csv')]

        chunks = []
        for i, s3_key in enumerate(sorted(train_files[:5])):  # Limit for demo
            local_file = f"/tmp/train_chunk_{i:03d}.csv"
            s3.download_file(bucket, s3_key, local_file)
            chunk = pd.read_csv(local_file, header=None, low_memory=False)
            chunks.append(chunk)
            os.remove(local_file)

        combined_df = pd.concat(chunks, ignore_index=True)
    else:
        # Local loading
        train_files = [f for f in os.listdir(input_path) if f.startswith('train_chunk_') and f.endswith('.csv')]
        chunks = []
        for file_path in sorted([os.path.join(input_path, f) for f in train_files[:5]]):
            chunk = pd.read_csv(file_path, header=None, low_memory=False)
            chunks.append(chunk)
        combined_df = pd.concat(chunks, ignore_index=True)

    # Sort by timestamp if available (assume first column is timestamp-like)
    if combined_df.shape[1] > 10:
        combined_df = combined_df.sort_values(by=combined_df.columns[0]).reset_index(drop=True)

    logger.info(f"📊 Sequential dataset: {combined_df.shape[0]:,} samples, {combined_df.shape[1]} features")

    # Extract features and labels
    if combined_df.shape[1] == 84:
        X = combined_df.iloc[:, :-1].values.astype(np.float32)
        y = combined_df.iloc[:, -1].values
    else:
        X = combined_df.iloc[:, 1:].values.astype(np.float32)
        y = combined_df.iloc[:, 0].values

    return X, y

def train_sequential_models(X, y, device, gpu_count, epochs=100, batch_size=64, sequence_length=50):
    """Train sequential models for temporal threat detection"""
    logger.info(f"🔄 SEQUENTIAL THREAT DETECTION TRAINING")
    logger.info(f"📊 Dataset: {X.shape[0]:,} samples → sequences of length {sequence_length}")

    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Handle labels
    label_encoder = None
    if isinstance(y[0], str):
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        num_classes = len(label_encoder.classes_)
    else:
        y_encoded = y.astype(int)
        num_classes = len(np.unique(y_encoded))

    logger.info(f"🎯 {num_classes} classes detected")

    # Create sequential dataset
    dataset = AttackSequenceDataset(X_scaled, y_encoded, sequence_length=sequence_length, stride=10)

    # Split into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    logger.info(f"🔄 Created {len(train_dataset)} training sequences, {len(val_dataset)} validation sequences")

    models = {}
    training_history = {}

    # 1. Train LSTM with Attention
    logger.info("🔄 Training LSTM + Attention Model...")

    lstm_model = LSTMAttentionDetector(
        input_dim=X.shape[1],
        hidden_dim=256,
        num_layers=3,
        num_classes=num_classes,
        dropout=0.3
    ).to(device)

    if gpu_count > 1:
        lstm_model = nn.DataParallel(lstm_model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(lstm_model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, epochs=epochs, steps_per_epoch=len(train_loader))

    lstm_history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0

    for epoch in range(epochs):
        # Training
        lstm_model.train()
        train_loss = 0.0

        for batch_X, batch_y in tqdm(train_loader, desc=f"LSTM Epoch {epoch+1}/{epochs}"):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs, _ = lstm_model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

        # Validation
        lstm_model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs, _ = lstm_model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_acc = correct / total

        lstm_history['train_loss'].append(train_loss)
        lstm_history['val_loss'].append(val_loss)
        lstm_history['val_acc'].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(lstm_model.state_dict(), '/tmp/best_lstm_model.pth')

        if epoch % 10 == 0:
            logger.info(f"LSTM Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # Load best LSTM model
    lstm_model.load_state_dict(torch.load('/tmp/best_lstm_model.pth'))
    models['lstm_attention'] = lstm_model
    training_history['lstm_attention'] = lstm_history

    logger.info(f"✅ LSTM + Attention: Best Val Acc: {best_val_acc:.4f}")

    # 2. Train Transformer Model
    logger.info("🤖 Training Transformer Model...")

    transformer_model = TransformerThreatDetector(
        input_dim=X.shape[1],
        d_model=512,
        nhead=8,
        num_layers=6,
        num_classes=num_classes,
        dropout=0.1
    ).to(device)

    if gpu_count > 1:
        transformer_model = nn.DataParallel(transformer_model)

    optimizer_t = optim.AdamW(transformer_model.parameters(), lr=0.0001, weight_decay=1e-5)
    scheduler_t = optim.lr_scheduler.OneCycleLR(optimizer_t, max_lr=0.001, epochs=epochs//2, steps_per_epoch=len(train_loader))

    transformer_history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    best_val_acc_t = 0.0

    for epoch in range(epochs // 2):  # Fewer epochs for Transformer
        # Training
        transformer_model.train()
        train_loss = 0.0

        for batch_X, batch_y in tqdm(train_loader, desc=f"Transformer Epoch {epoch+1}/{epochs//2}"):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer_t.zero_grad()
            outputs = transformer_model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer_t.step()
            scheduler_t.step()

            train_loss += loss.item()

        # Validation
        transformer_model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = transformer_model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_acc = correct / total

        transformer_history['train_loss'].append(train_loss)
        transformer_history['val_loss'].append(val_loss)
        transformer_history['val_acc'].append(val_acc)

        if val_acc > best_val_acc_t:
            best_val_acc_t = val_acc
            torch.save(transformer_model.state_dict(), '/tmp/best_transformer_model.pth')

        if epoch % 5 == 0:
            logger.info(f"Transformer Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # Load best Transformer model
    transformer_model.load_state_dict(torch.load('/tmp/best_transformer_model.pth'))
    models['transformer'] = transformer_model
    training_history['transformer'] = transformer_history

    logger.info(f"✅ Transformer: Best Val Acc: {best_val_acc_t:.4f}")

    # Metadata
    metadata = {
        'total_samples': int(X.shape[0]),
        'features': int(X.shape[1]),
        'sequence_length': sequence_length,
        'num_sequences': len(dataset),
        'num_classes': int(num_classes),
        'gpu_count': gpu_count,
        'device': str(device),
        'sequential_modeling': True,
        'lstm_best_accuracy': float(best_val_acc),
        'transformer_best_accuracy': float(best_val_acc_t),
        'epochs_trained': epochs,
        'batch_size': batch_size,
        'timestamp': time.time()
    }

    return models, scaler, label_encoder, metadata, training_history

def save_sequential_models(models, scaler, label_encoder, metadata, training_history, model_dir):
    """Save sequential models"""
    logger.info(f"💾 Saving sequential models to {model_dir}")

    os.makedirs(model_dir, exist_ok=True)

    # Save PyTorch models
    for model_name, model in models.items():
        if isinstance(model, nn.DataParallel):
            torch.save(model.module.state_dict(), os.path.join(model_dir, f'{model_name}.pth'))
        else:
            torch.save(model.state_dict(), os.path.join(model_dir, f'{model_name}.pth'))
        logger.info(f"   ✅ Saved {model_name}.pth")

    # Save preprocessing
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))
    if label_encoder:
        joblib.dump(label_encoder, os.path.join(model_dir, 'label_encoder.pkl'))

    # Save metadata
    metadata_enhanced = {
        **metadata,
        'model_type': 'sequential_deep_learning',
        'architecture': 'lstm_transformer_ensemble',
        'temporal_modeling': True,
        'production_ready': True,
        'mini_xdr_version': '2.0_sequential'
    }

    with open(os.path.join(model_dir, 'model_metadata.json'), 'w') as f:
        json.dump(metadata_enhanced, f, indent=2)

    # Save training history
    with open(os.path.join(model_dir, 'training_history.json'), 'w') as f:
        history_serializable = {}
        for model_name, history in training_history.items():
            history_serializable[model_name] = {}
            for key, values in history.items():
                history_serializable[model_name][key] = [float(v) for v in values]
        json.dump(history_serializable, f, indent=2)

    logger.info("✅ All sequential models saved")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAINING', '/opt/ml/input/data/training'))
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--sequence-length', type=int, default=50)

    args = parser.parse_args()
    start_time = time.time()

    try:
        logger.info("🔄 MINI-XDR SEQUENTIAL DEEP LEARNING TRAINING")
        logger.info("=" * 60)

        # Setup GPU
        if torch.cuda.is_available():
            device = torch.device('cuda')
            gpu_count = torch.cuda.device_count()
            logger.info(f"🚀 GPU acceleration: {gpu_count} GPUs")
        else:
            device = torch.device('cpu')
            gpu_count = 0
            logger.info("⚠️  Using CPU")

        # Load data
        X, y = load_sequential_data(args.train, args.sequence_length)

        # Train models
        models, scaler, label_encoder, metadata, training_history = train_sequential_models(
            X, y, device, gpu_count, args.epochs, args.batch_size, args.sequence_length
        )

        # Save models
        save_sequential_models(models, scaler, label_encoder, metadata, training_history, args.model_dir)

        # Summary
        duration = time.time() - start_time
        logger.info("🎉 SEQUENTIAL TRAINING COMPLETED!")
        logger.info("=" * 60)
        logger.info(f"⏱️  Total time: {duration/60:.2f} minutes")
        logger.info(f"📊 Sequences: {metadata['num_sequences']:,}")
        logger.info(f"🔄 Sequence length: {metadata['sequence_length']}")
        logger.info(f"🧠 LSTM accuracy: {metadata['lstm_best_accuracy']:.4f}")
        logger.info(f"🤖 Transformer accuracy: {metadata['transformer_best_accuracy']:.4f}")
        logger.info("🚀 Sequential models ready for temporal threat detection!")

    except Exception as e:
        logger.error(f"❌ SEQUENTIAL TRAINING FAILED: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == '__main__':
    main()
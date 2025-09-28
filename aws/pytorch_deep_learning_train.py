#!/usr/bin/env python3
"""
Deep Learning PyTorch Training Script for Mini-XDR
Proper neural network architectures for threat detection
Uses full dataset with efficient batching
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
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
import joblib
import boto3
import logging
import time
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class XDRThreatDetector(nn.Module):
    """
    Deep neural network for multi-class threat detection
    Optimized for cybersecurity features
    """
    def __init__(self, input_dim, hidden_dims=[512, 256, 128, 64], num_classes=2, dropout_rate=0.3):
        super(XDRThreatDetector, self).__init__()

        layers = []
        prev_dim = input_dim

        # Build deep architecture
        for i, dim in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = dim

        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)

    def forward(self, x):
        return self.network(x)

class XDRAnomalyDetector(nn.Module):
    """
    Autoencoder for unsupervised anomaly detection
    Detects novel attack patterns
    """
    def __init__(self, input_dim, encoding_dims=[256, 128, 64], latent_dim=32):
        super(XDRAnomalyDetector, self).__init__()

        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for dim in encoding_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for dim in reversed(encoding_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def get_reconstruction_error(self, x):
        with torch.no_grad():
            reconstructed = self.forward(x)
            error = torch.mean((x - reconstructed) ** 2, dim=1)
            return error

class SequentialThreatDetector(nn.Module):
    """
    LSTM-based model for temporal threat pattern detection
    """
    def __init__(self, feature_dim, hidden_dim=256, num_layers=2, num_classes=2):
        super(SequentialThreatDetector, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            feature_dim, hidden_dim, num_layers,
            batch_first=True, dropout=0.2, bidirectional=True
        )

        # Attention mechanism
        self.attention = nn.MultiheadAttention(hidden_dim * 2, num_heads=8, batch_first=True)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # LSTM processing
        lstm_out, _ = self.lstm(x)

        # Self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)

        # Global average pooling
        pooled = torch.mean(attn_out, dim=1)

        # Classification
        output = self.classifier(pooled)
        return output

class XDRDataset(Dataset):
    """
    Custom dataset for efficient loading of large XDR data
    """
    def __init__(self, X, y=None, sequence_length=10):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y) if y is not None else None
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]

    def create_sequences(self):
        """Create sequences for LSTM training"""
        if len(self.X) < self.sequence_length:
            return self.X.unsqueeze(1), self.y

        sequences = []
        labels = []

        for i in range(len(self.X) - self.sequence_length + 1):
            seq = self.X[i:i + self.sequence_length]
            sequences.append(seq)
            if self.y is not None:
                labels.append(self.y[i + self.sequence_length - 1])

        sequences = torch.stack(sequences)
        labels = torch.stack(labels) if labels else None

        return sequences, labels

def setup_gpu():
    """Setup GPU acceleration with proper memory management"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_count = torch.cuda.device_count()
        logger.info(f"🚀 DEEP LEARNING GPU ACCELERATION ENABLED!")
        logger.info(f"💎 Available GPUs: {gpu_count}")

        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / 1e9
            logger.info(f"   GPU {i}: {props.name} ({memory_gb:.1f}GB)")

        # Set memory growth and optimization
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True

        # Very conservative memory management for single V100
        if gpu_count > 0:
            torch.cuda.set_per_process_memory_fraction(0.7)  # Use only 70% of GPU memory
            logger.info("🔧 Set GPU memory fraction to 70% for stability")

            # Additional memory optimizations
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False  # Disable for memory consistency

        return device, gpu_count
    else:
        logger.warning("⚠️  No GPU available, using CPU (will be slow)")
        return torch.device('cpu'), 0

def load_full_dataset(input_path, max_samples=None):
    """Load the complete dataset efficiently"""
    logger.info(f"🚀 Loading FULL DATASET from {input_path}")

    if input_path.startswith('s3://'):
        # S3 loading
        s3 = boto3.client('s3')
        parts = input_path.replace('s3://', '').split('/')
        bucket = parts[0]
        prefix = '/'.join(parts[1:]) if len(parts) > 1 else ''

        response = s3.list_objects_v2(Bucket=bucket, Prefix=f"{prefix}/train_chunk_")
        train_files = [obj['Key'] for obj in response.get('Contents', []) if obj['Key'].endswith('.csv')]

        if not train_files:
            raise ValueError(f"No training files found in {input_path}")

        logger.info(f"📊 Found {len(train_files)} training chunks")

        chunks = []
        total_samples = 0

        for i, s3_key in enumerate(sorted(train_files)):
            local_file = f"/tmp/train_chunk_{i:03d}.csv"
            logger.info(f"📂 Loading {s3_key}")
            s3.download_file(bucket, s3_key, local_file)

            chunk = pd.read_csv(local_file, header=None, low_memory=False)
            chunks.append(chunk)
            total_samples += len(chunk)

            os.remove(local_file)

            if max_samples and total_samples >= max_samples:
                break

        combined_df = pd.concat(chunks, ignore_index=True)

    else:
        # Local loading
        train_files = [f for f in os.listdir(input_path) if f.startswith('train_chunk_') and f.endswith('.csv')]
        if not train_files:
            raise ValueError(f"No training files found in {input_path}")

        chunks = []
        total_samples = 0

        for file_path in sorted([os.path.join(input_path, f) for f in train_files]):
            logger.info(f"📂 Loading {file_path}")
            chunk = pd.read_csv(file_path, header=None, low_memory=False)
            chunks.append(chunk)
            total_samples += len(chunk)

            if max_samples and total_samples >= max_samples:
                break

        combined_df = pd.concat(chunks, ignore_index=True)

    # Limit to max_samples if specified
    if max_samples and len(combined_df) > max_samples:
        combined_df = combined_df.sample(n=max_samples, random_state=42).reset_index(drop=True)
        logger.info(f"📊 Sampled {max_samples:,} from {total_samples:,} total samples")

    logger.info(f"✅ LOADED DATASET: {combined_df.shape[0]:,} samples, {combined_df.shape[1]} features")
    logger.info(f"📊 Memory usage: {combined_df.memory_usage(deep=True).sum()/1e9:.2f}GB")

    # Handle data format
    if combined_df.shape[1] == 84:  # 83 features + 1 label
        X = combined_df.iloc[:, :-1].values.astype(np.float32)
        y = combined_df.iloc[:, -1].values
    elif combined_df.shape[1] == 83:  # Only features
        X = combined_df.values.astype(np.float32)
        y = None
    else:
        # Assume last column is labels for any other format
        X = combined_df.iloc[:, :-1].values.astype(np.float32)
        y = combined_df.iloc[:, -1].values

    return X, y

def train_deep_models(X, y, device, gpu_count, epochs=50, batch_size=1024):
    """Train deep learning models on full dataset"""
    logger.info(f"🧠 DEEP LEARNING TRAINING")
    logger.info(f"📊 Full Dataset: {X.shape[0]:,} samples, {X.shape[1]} features")
    logger.info(f"💎 Using {gpu_count} GPUs with batch size {batch_size}")

    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Handle labels
    label_encoder = None
    if y is not None and isinstance(y[0], str):
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        num_classes = len(label_encoder.classes_)
        logger.info(f"🎯 Detected {num_classes} classes: {label_encoder.classes_}")
    elif y is not None:
        # Convert to int and validate class labels
        y_int = y.astype(int)
        unique_classes = np.unique(y_int)

        logger.info(f"🔍 Raw label analysis: {len(y_int)} samples")
        logger.info(f"🔍 Unique labels found: {unique_classes}")
        logger.info(f"🔍 Label distribution: {np.bincount(y_int)}")

        # Ensure labels are contiguous starting from 0
        if np.min(unique_classes) != 0 or np.max(unique_classes) != len(unique_classes) - 1:
            logger.info("🔧 Remapping labels to contiguous range [0, num_classes-1]")
            label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_classes)}
            y_encoded = np.array([label_mapping[label] for label in y_int])
            logger.info(f"🔧 Label mapping: {label_mapping}")
        else:
            y_encoded = y_int

        num_classes = len(unique_classes)
        logger.info(f"🎯 Final: {num_classes} classes, range [0, {num_classes-1}]")
        logger.info(f"🎯 Final distribution: {np.bincount(y_encoded)}")

        # Validate no invalid indices
        assert np.min(y_encoded) >= 0, f"Invalid negative class index: {np.min(y_encoded)}"
        assert np.max(y_encoded) < num_classes, f"Class index {np.max(y_encoded)} >= num_classes {num_classes}"

    else:
        y_encoded = None
        num_classes = 2  # Binary by default

    models = {}
    training_history = {}

    # 1. Train Deep Threat Detector (Supervised)
    if y_encoded is not None:
        logger.info("🌟 Training Deep Threat Detection Network...")

        # Split data
        try:
            X_train, X_val, y_train, y_val = train_test_split(
                X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
            )
        except ValueError:
            # Fallback without stratification if classes are too small
            logger.warning("⚠️  Using non-stratified split due to small class sizes")
            X_train, X_val, y_train, y_val = train_test_split(
                X_scaled, y_encoded, test_size=0.2, random_state=42
            )

        # Create datasets
        train_dataset = XDRDataset(X_train, y_train)
        val_dataset = XDRDataset(X_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        # Progressive model sizing based on available memory
        if gpu_count >= 4:
            # 4x V100 setup - can handle medium model
            hidden_dims = [512, 256, 128]  # ~150K parameters
            logger.info("🎯 Using MEDIUM model for 4x V100 GPUs")
        elif gpu_count >= 1:
            # Single V100 - smaller model
            hidden_dims = [256, 128, 64]   # ~60K parameters
            logger.info("🎯 Using SMALL model for single GPU")
        else:
            # CPU fallback - tiny model
            hidden_dims = [128, 64]        # ~20K parameters
            logger.info("🎯 Using TINY model for CPU")

        model = XDRThreatDetector(
            input_dim=X.shape[1],
            hidden_dims=hidden_dims,
            num_classes=num_classes,
            dropout_rate=0.2
        ).to(device)

        # Log model size
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"🧠 Model parameters: {total_params:,}")

        # DISABLED: DataParallel causes NCCL hangs on SageMaker ml.p3.8xlarge
        # Use single GPU with gradient accumulation instead
        if gpu_count > 1:
            logger.warning(f"⚠️  Multi-GPU DataParallel disabled due to NCCL issues on SageMaker")
            logger.info(f"🔧 Using single GPU with {gpu_count}x gradient accumulation instead")
            accumulation_steps *= gpu_count  # Increase accumulation to simulate multi-GPU

        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

        # Gradient accumulation for effective larger batch size
        accumulation_steps = 4  # Simulate 4x larger batch
        logger.info(f"🔧 Using gradient accumulation: {accumulation_steps} steps")

        # Temporarily disable mixed precision for stability
        scaler = None

        # Training loop
        train_history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0

            # Gradient accumulation training
            for batch_idx, (batch_X, batch_y) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
                batch_X, batch_y = batch_X.to(device, non_blocking=True), batch_y.to(device, non_blocking=True)

                # CUDA safety: Validate batch labels
                if torch.any(batch_y < 0) or torch.any(batch_y >= num_classes):
                    logger.error(f"❌ Invalid labels in batch {batch_idx}: min={torch.min(batch_y)}, max={torch.max(batch_y)}, num_classes={num_classes}")
                    logger.error(f"❌ Batch labels: {batch_y.cpu().numpy()}")
                    raise ValueError(f"Invalid class indices detected")

                # Forward pass
                outputs = model(batch_X)

                # Additional safety check for output dimensions
                if outputs.shape[1] != num_classes:
                    logger.error(f"❌ Model output shape mismatch: got {outputs.shape[1]}, expected {num_classes}")
                    raise ValueError(f"Model output dimension mismatch")

                loss = criterion(outputs, batch_y)

                # Scale loss by accumulation steps
                loss = loss / accumulation_steps

                # Backward pass
                loss.backward()

                train_loss += loss.item() * accumulation_steps

                # Update weights every accumulation_steps
                if (batch_idx + 1) % accumulation_steps == 0:
                    # Gradient clipping for stability
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    optimizer.step()
                    optimizer.zero_grad()

                    # Clear cache after each weight update
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            # Final optimizer step if needed
            if len(train_loader) % accumulation_steps != 0:
                optimizer.step()
                optimizer.zero_grad()

            # Validation phase
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                    # CUDA safety: Validate validation batch labels
                    if torch.any(batch_y < 0) or torch.any(batch_y >= num_classes):
                        logger.error(f"❌ Invalid validation labels: min={torch.min(batch_y)}, max={torch.max(batch_y)}, num_classes={num_classes}")
                        continue

                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()

                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()

            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            val_acc = correct / total

            train_history['train_loss'].append(train_loss)
            train_history['val_loss'].append(val_loss)
            train_history['val_acc'].append(val_acc)

            scheduler.step(val_loss)

            logger.info(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), '/tmp/best_threat_detector.pth')
            else:
                patience_counter += 1
                if patience_counter >= 10:
                    logger.info("Early stopping triggered")
                    break

        # Load best model
        model.load_state_dict(torch.load('/tmp/best_threat_detector.pth'))
        models['threat_detector'] = model
        training_history['threat_detector'] = train_history

        logger.info(f"✅ Deep Threat Detector: Best Val Acc: {max(train_history['val_acc']):.4f}")

    # 2. Train Autoencoder for Anomaly Detection
    logger.info("🔍 Training Deep Anomaly Detection Autoencoder...")

    # Use full dataset for unsupervised training
    dataset = XDRDataset(X_scaled)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Smaller autoencoder for memory efficiency
    autoencoder = XDRAnomalyDetector(
        input_dim=X.shape[1],
        encoding_dims=[128, 64],  # Smaller encoder
        latent_dim=16             # Smaller latent space
    ).to(device)

    # Single GPU only
    ae_params = sum(p.numel() for p in autoencoder.parameters())
    logger.info(f"🔍 Autoencoder parameters: {ae_params:,}")

    criterion_ae = nn.MSELoss()
    optimizer_ae = optim.AdamW(autoencoder.parameters(), lr=0.001)

    ae_history = {'train_loss': []}

    for epoch in range(epochs // 2):  # Fewer epochs for autoencoder
        autoencoder.train()
        epoch_loss = 0.0

        for batch_X in tqdm(dataloader, desc=f"AE Epoch {epoch+1}/{epochs//2}"):
            if isinstance(batch_X, tuple):
                batch_X = batch_X[0]
            batch_X = batch_X.to(device)

            optimizer_ae.zero_grad()
            reconstructed = autoencoder(batch_X)
            loss = criterion_ae(reconstructed, batch_X)
            loss.backward()
            optimizer_ae.step()

            epoch_loss += loss.item()

        epoch_loss /= len(dataloader)
        ae_history['train_loss'].append(epoch_loss)

        logger.info(f"AE Epoch {epoch+1}: Loss: {epoch_loss:.6f}")

    models['anomaly_detector'] = autoencoder
    training_history['anomaly_detector'] = ae_history

    logger.info("✅ Deep Anomaly Detector trained")

    # Prepare metadata
    metadata = {
        'total_samples': int(X.shape[0]),
        'features': int(X.shape[1]),
        'gpu_count': gpu_count,
        'device': str(device),
        'deep_learning': True,
        'epochs_trained': epochs,
        'batch_size': batch_size,
        'architecture': 'multi_model_deep_learning',
        'timestamp': time.time()
    }

    if y_encoded is not None:
        metadata['num_classes'] = int(num_classes)
        metadata['best_accuracy'] = float(max(train_history['val_acc']))

    return models, scaler, label_encoder, metadata, training_history

def save_deep_models(models, scaler, label_encoder, metadata, training_history, model_dir):
    """Save all deep learning models and artifacts"""
    logger.info(f"💾 Saving deep learning models to {model_dir}")

    os.makedirs(model_dir, exist_ok=True)

    # Save PyTorch models
    for model_name, model in models.items():
        if isinstance(model, nn.DataParallel):
            torch.save(model.module.state_dict(), os.path.join(model_dir, f'{model_name}.pth'))
        else:
            torch.save(model.state_dict(), os.path.join(model_dir, f'{model_name}.pth'))
        logger.info(f"   ✅ Saved {model_name}.pth")

    # Also save to backend models directory for immediate use
    backend_model_dir = "/Users/chasemad/Desktop/mini-xdr/models"
    if os.path.exists(backend_model_dir):
        logger.info(f"💾 Also saving to backend models directory: {backend_model_dir}")
        for model_name, model in models.items():
            if isinstance(model, nn.DataParallel):
                torch.save(model.module.state_dict(), os.path.join(backend_model_dir, f'{model_name}.pth'))
            else:
                torch.save(model.state_dict(), os.path.join(backend_model_dir, f'{model_name}.pth'))

    # Save preprocessing
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))
    if label_encoder:
        joblib.dump(label_encoder, os.path.join(model_dir, 'label_encoder.pkl'))

    # Also save preprocessing to backend models directory
    if os.path.exists(backend_model_dir):
        joblib.dump(scaler, os.path.join(backend_model_dir, 'scaler.pkl'))
        if label_encoder:
            joblib.dump(label_encoder, os.path.join(backend_model_dir, 'label_encoder.pkl'))

    # Save metadata
    metadata_enhanced = {
        **metadata,
        'model_type': 'pytorch_deep_learning',
        'full_dataset_training': True,
        'production_ready': True,
        'mini_xdr_version': '2.0_deep_learning'
    }

    with open(os.path.join(model_dir, 'model_metadata.json'), 'w') as f:
        json.dump(metadata_enhanced, f, indent=2)

    # Save training history
    with open(os.path.join(model_dir, 'training_history.json'), 'w') as f:
        # Convert numpy types to native Python types
        history_serializable = {}
        for model_name, history in training_history.items():
            history_serializable[model_name] = {}
            for key, values in history.items():
                history_serializable[model_name][key] = [float(v) for v in values]
        json.dump(history_serializable, f, indent=2)

    # Also save metadata and history to backend models directory
    if os.path.exists(backend_model_dir):
        with open(os.path.join(backend_model_dir, 'model_metadata.json'), 'w') as f:
            json.dump(metadata_enhanced, f, indent=2)

        with open(os.path.join(backend_model_dir, 'training_history.json'), 'w') as f:
            json.dump(history_serializable, f, indent=2)

    logger.info("✅ All deep learning models and artifacts saved")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAINING', '/opt/ml/input/data/training'))
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=1024, help='Batch size')
    parser.add_argument('--max-samples', type=int, default=None, help='Max samples to use (None = all)')

    args = parser.parse_args()
    start_time = time.time()

    try:
        print("🚀 MINI-XDR DEEP LEARNING TRAINING STARTED")
        print("=" * 60)
        print(f"📊 Arguments: {vars(args)}")
        print(f"🐍 Python version: {sys.version}")
        print(f"🔥 PyTorch version: {torch.__version__}")
        print(f"💎 CUDA available: {torch.cuda.is_available()}")

        logger.info("🚀 MINI-XDR DEEP LEARNING TRAINING")
        logger.info("=" * 60)

        # Setup GPU
        device, gpu_count = setup_gpu()

        # Load full dataset
        X, y = load_full_dataset(args.train, max_samples=args.max_samples)

        # Train deep learning models
        models, scaler, label_encoder, metadata, training_history = train_deep_models(
            X, y, device, gpu_count, epochs=args.epochs, batch_size=args.batch_size
        )

        # Save models
        save_deep_models(models, scaler, label_encoder, metadata, training_history, args.model_dir)

        # Final summary
        duration = time.time() - start_time
        logger.info("🎉 DEEP LEARNING TRAINING COMPLETED!")
        logger.info("=" * 60)
        logger.info(f"⏱️  Total time: {duration/60:.2f} minutes")
        logger.info(f"📊 Total samples: {metadata['total_samples']:,}")
        logger.info(f"🧠 Features: {metadata['features']}")
        logger.info(f"💎 GPU acceleration: {gpu_count} GPUs")
        logger.info(f"🔥 Deep learning epochs: {metadata['epochs_trained']}")

        if 'best_accuracy' in metadata:
            logger.info(f"🎯 Best accuracy: {metadata['best_accuracy']:.4f}")

        logger.info("🚀 Deep learning models ready for production!")

    except Exception as e:
        print(f"❌ DEEP LEARNING TRAINING FAILED: {e}")
        logger.error(f"❌ DEEP LEARNING TRAINING FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
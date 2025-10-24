#!/usr/bin/env python3
"""
Train 13-Class Windows Attack Specialist Model
Enhanced model with full Windows/AD attack coverage
"""

import os
import sys
import numpy as np
import logging
import json
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Windows13ClassSpecialist(nn.Module):
    """Neural network for 13-class Windows attack detection"""
    
    def __init__(self, input_dim=79, num_classes=13):
        super(Windows13ClassSpecialist, self).__init__()
        
        self.features = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(512, 384),
            nn.BatchNorm1d(384),
            nn.ReLU(),
            nn.Dropout(0.25),
            
            nn.Linear(384, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.classifier = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    
    def __init__(self, alpha=None, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


def load_data(data_dir="/Users/chasemad/Desktop/mini-xdr/datasets/windows_converted"):
    """Load balanced Windows dataset"""
    logger.info("📂 Loading Windows dataset...")
    
    data_dir = Path(data_dir)
    
    X = np.load(data_dir / "windows_features_balanced.npy")
    y = np.load(data_dir / "windows_labels_balanced.npy")
    
    logger.info(f"   Loaded: {len(X):,} samples, {X.shape[1]} features")
    
    # Check class distribution
    unique, counts = np.unique(y, return_counts=True)
    logger.info("\n📊 Class distribution:")
    for class_id, count in zip(unique, counts):
        logger.info(f"   Class {class_id:2d}: {count:,} samples")
    
    return X, y


def train_model(model, train_loader, val_loader, device, epochs=30, lr=0.001):
    """Train the model"""
    logger.info("\n🎯 Training Windows specialist...")
    
    # Setup loss and optimizer
    criterion = FocalLoss(gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )
    
    best_val_acc = 0.0
    best_epoch = 0
    patience = 10
    patience_counter = 0
    
    for epoch in range(1, epochs + 1):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()
        
        train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                _, predicted = torch.max(outputs, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()
        
        val_acc = val_correct / val_total
        
        logger.info(f"Epoch {epoch}/{epochs}: Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
        
        # Learning rate scheduling
        scheduler.step(val_acc)
        
        # Early stopping and model saving
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_windows_13class.pth')
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            logger.info(f"\nEarly stopping at epoch {epoch}")
            break
    
    logger.info(f"\n✅ Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
    
    # Load best model
    model.load_state_dict(torch.load('best_windows_13class.pth'))
    
    return model


def evaluate_model(model, test_loader, device):
    """Evaluate the model"""
    logger.info("\n📊 Final Evaluation...")
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)
    
    class_names = [
        'Normal', 'DDoS', 'Reconnaissance', 'Brute Force', 'Web Attack',
        'Malware', 'APT', 'Kerberos Attack', 'Lateral Movement',
        'Credential Theft', 'Privilege Escalation', 'Data Exfiltration',
        'Insider Threat'
    ]
    
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    
    logger.info("\n" + "=" * 70)
    logger.info("✅ WINDOWS 13-CLASS SPECIALIST TRAINING COMPLETE!")
    logger.info("=" * 70)
    logger.info(f"🎯 Final Accuracy: {accuracy:.4f}")
    logger.info(f"🎯 F1 Score: {f1:.4f}")
    
    logger.info("\n📊 Per-Class Performance:")
    for i, class_name in enumerate(class_names):
        if str(i) in report:
            precision = report[str(i)]['precision']
            recall = report[str(i)]['recall']
            logger.info(f"  {class_name:20s}: Precision={precision:.3f}, Recall={recall:.3f}")
    
    return {
        'accuracy': float(accuracy),
        'f1_score': float(f1),
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'class_names': class_names
    }


def save_model_artifacts(model, scaler, metrics, output_dir="/Users/chasemad/Desktop/mini-xdr/models/windows_specialist_13class"):
    """Save model and artifacts"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\n💾 Saving model artifacts to: {output_dir}")
    
    # Save model
    model_path = output_dir / "windows_13class_specialist.pth"
    torch.save(model.state_dict(), model_path)
    logger.info(f"✅ Model saved: {model_path}")
    
    # Save scaler
    scaler_path = output_dir / "windows_13class_scaler.pkl"
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    logger.info(f"✅ Scaler saved: {scaler_path}")
    
    # Save metadata
    metadata = {
        'model_type': 'windows_13class_specialist',
        'classes': 13,
        'class_names': metrics['class_names'],
        'input_features': 79,
        'accuracy': metrics['accuracy'],
        'f1_score': metrics['f1_score'],
        'training_date': datetime.now().isoformat(),
        'training_samples': 390000,
        'architecture': 'Deep Neural Network (79 -> 256 -> 512 -> 384 -> 256 -> 128 -> 13)'
    }
    
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"✅ Metadata saved: {metadata_path}")
    
    # Save detailed metrics
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"✅ Metrics saved: {metrics_path}")
    
    logger.info("\n✅ All artifacts saved!")


def main():
    logger.info("=" * 70)
    logger.info("🚀 TRAINING WINDOWS 13-CLASS ATTACK SPECIALIST MODEL")
    logger.info("=" * 70)
    
    # Setup device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info("🖥️  Using CUDA GPU")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        logger.info("🖥️  Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device('cpu')
        logger.info("🖥️  Using CPU")
    
    # Load data
    X, y = load_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info(f"\n📊 Train: {len(X_train):,}, Test: {len(X_test):,}")
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Create data loaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test),
        torch.LongTensor(y_test)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0)
    
    # Create model
    model = Windows13ClassSpecialist(input_dim=79, num_classes=13)
    model = model.to(device)
    
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"\n🧠 Model: {param_count:,} parameters")
    
    # Train model
    model = train_model(model, train_loader, test_loader, device, epochs=30, lr=0.001)
    
    # Evaluate model
    metrics = evaluate_model(model, test_loader, device)
    
    # Save artifacts
    save_model_artifacts(model, scaler, metrics)
    
    logger.info("\n🚀 Next steps:")
    logger.info("   1. Integrate with ensemble: backend/app/ensemble_ml_detector.py")
    logger.info("   2. Run regression tests: scripts/testing/test_enterprise_detection.py")
    logger.info("   3. Update TRAINING_STATUS.md with metrics")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())


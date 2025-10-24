#!/usr/bin/env python3
"""
Train Windows Attack Specialist Model
7-class Windows-specific threat detector
Works as ensemble with existing network models
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
import joblib
import logging
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class WindowsSpecialistModel(nn.Module):
    """Windows Attack Specialist Neural Network"""
    
    def __init__(self, input_dim=79, hidden_dims=[256, 128, 64], num_classes=7, dropout=0.3):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, num_classes)
    
    def forward(self, x):
        features = self.feature_extractor(x)
        return self.classifier(features)


def train_windows_specialist():
    """Train Windows specialist model"""
    logger.info("🚀 TRAINING WINDOWS ATTACK SPECIALIST MODEL")
    logger.info("=" * 70)
    
    # Device setup
    if torch.backends.mps.is_available():
        device = torch.device('mps')  # Apple Silicon GPU
        logger.info("🖥️  Using Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"🖥️  Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        logger.info("🖥️  Using CPU")
    
    # Load Windows specialist data
    data_dir = Path("/Users/chasemad/Desktop/mini-xdr/models/windows_specialist")
    
    # Find latest data files
    feature_files = sorted(data_dir.glob("windows_features_*.npy"))
    label_files = sorted(data_dir.glob("windows_labels_*.npy"))
    
    if not feature_files or not label_files:
        logger.error("❌ Windows specialist data not found!")
        logger.info("   Run: python3 aws/build_windows_specialist_model.py")
        return 1
    
    features = np.load(feature_files[-1])
    labels = np.load(label_files[-1])
    
    logger.info(f"📊 Loaded: {len(features):,} samples, {features.shape[1]} features")
    
    # Scale features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        features_scaled, labels, test_size=0.2, stratify=labels, random_state=42
    )
    
    logger.info(f"📊 Train: {len(X_train):,}, Test: {len(X_test):,}")
    
    # Class weights for balancing
    unique_classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=unique_classes, y=y_train)
    sample_weights = [class_weights[label] for label in y_train]
    
    # Create data loaders
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long)
    )
    
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=512, sampler=sampler)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)
    
    # Create model
    model = WindowsSpecialistModel(
        input_dim=79,
        hidden_dims=[256, 128, 64],
        num_classes=7,
        dropout=0.3
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"🧠 Model: {total_params:,} parameters")
    
    # Training setup
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Training loop
    logger.info("\n🎯 Training Windows specialist...")
    best_val_acc = 0.0
    epochs = 30  # Fewer epochs for specialist
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_features, batch_labels in train_loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += batch_labels.size(0)
            train_correct += (predicted == batch_labels).sum().item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_features, batch_labels in test_loader:
                batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += batch_labels.size(0)
                val_correct += (predicted == batch_labels).sum().item()
        
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        
        logger.info(f"Epoch {epoch+1}/{epochs}: Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
        
        scheduler.step(val_loss / len(test_loader))
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Save best model
            torch.save({
                'model_state_dict': model.state_dict(),
                'scaler': scaler,
                'epoch': epoch,
                'val_acc': val_acc
            }, data_dir / "windows_specialist.pth")
    
    # Final evaluation
    logger.info("\n📊 Final Evaluation...")
    model.eval()
    y_pred = []
    y_true = []
    
    with torch.no_grad():
        for batch_features, batch_labels in test_loader:
            batch_features = batch_features.to(device)
            outputs = model(batch_features)
            predicted = torch.max(outputs, 1)[1].cpu().numpy()
            y_pred.extend(predicted)
            y_true.extend(batch_labels.numpy())
    
    # Metrics
    report = classification_report(y_true, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    
    logger.info("\n" + "=" * 70)
    logger.info("✅ WINDOWS SPECIALIST TRAINING COMPLETE!")
    logger.info("=" * 70)
    logger.info(f"🎯 Final Accuracy: {report['accuracy']:.4f}")
    logger.info(f"🎯 F1 Score: {f1:.4f}")
    logger.info(f"💾 Model saved: {data_dir}/windows_specialist.pth")
    
    logger.info("\n📊 Per-Class Performance:")
    class_names = ['Normal', 'Kerberos', 'Lateral Mvmt', 'Cred Theft', 'Priv Esc', 'Exfiltration', 'Insider']
    for i, name in enumerate(class_names):
        if str(i) in report:
            logger.info(f"  {name:15s}: Precision={report[str(i)]['precision']:.3f}, Recall={report[str(i)]['recall']:.3f}")
    
    # Save final scaler
    joblib.dump(scaler, data_dir / "windows_scaler.pkl")
    
    # Save metadata
    metadata = {
        'model_type': 'windows_specialist',
        'classes': 7,
        'class_names': class_names,
        'accuracy': float(report['accuracy']),
        'f1_score': float(f1),
        'training_samples': int(len(X_train)),
        'test_samples': int(len(X_test)),
        'confusion_matrix': conf_matrix.tolist(),
        'per_class_metrics': {
            name: {
                'precision': float(report[str(i)]['precision']),
                'recall': float(report[str(i)]['recall']),
                'f1-score': float(report[str(i)]['f1-score'])
            }
            for i, name in enumerate(class_names) if str(i) in report
        }
    }
    
    with open(data_dir / "windows_specialist_metrics.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info("\n✅ All artifacts saved!")
    logger.info("\n🚀 Next: Integrate with existing models for ensemble detection")
    
    return 0


if __name__ == '__main__':
    sys.exit(train_windows_specialist())


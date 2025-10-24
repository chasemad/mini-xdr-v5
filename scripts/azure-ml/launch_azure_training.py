#!/usr/bin/env python3
"""
Launch Azure ML Training Job for Mini-XDR
Fast GPU-accelerated training for 4M+ events
"""

import os
import sys
import json
import logging
from pathlib import Path
from azure.ai.ml import MLClient, command, Input
from azure.ai.ml.entities import Environment
from azure.identity import DefaultAzureCredential, AzureCliCredential

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AzureMLTrainingLauncher:
    """Launch fast GPU training on Azure ML"""
    
    def __init__(self, config_file="/Users/chasemad/Desktop/mini-xdr/scripts/azure-ml/workspace_config.json"):
        # Load workspace config
        with open(config_file, 'r') as f:
            self.config = json.load(f)
        
        # Create ML client
        try:
            credential = AzureCliCredential()
        except:
            credential = DefaultAzureCredential()
        
        self.ml_client = MLClient(
            credential=credential,
            subscription_id=self.config['subscription_id'],
            resource_group_name=self.config['resource_group'],
            workspace_name=self.config['workspace_name']
        )
        
        logger.info(f"✅ Connected to workspace: {self.config['workspace_name']}")
    
    def create_training_script(self):
        """Create optimized training script for Azure"""
        script_content = '''#!/usr/bin/env python3
"""
Azure ML Training Script - Optimized for 4M+ Events
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
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import joblib
import logging
from pathlib import Path
from tqdm import tqdm
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model Architecture
class EnhancedThreatDetector(nn.Module):
    def __init__(self, input_dim=79, hidden_dims=[512, 256, 128, 64], num_classes=13, dropout=0.3):
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

def load_data(data_dir):
    """Load and combine all training data"""
    logger.info(f"Loading data from {data_dir}")
    
    data_path = Path(data_dir)
    csv_files = list(data_path.glob("*.csv"))
    json_files = list(data_path.glob("*.json"))
    
    all_features = []
    all_labels = []
    
    # Load CSV files
    for csv_file in csv_files:
        logger.info(f"Loading {csv_file.name}")
        df = pd.read_csv(csv_file)
        
        # Handle different formats
        if 'label' in df.columns:
            features = df.drop('label', axis=1).values
            labels = df['label'].values
        else:
            features = df.iloc[:, :-1].values
            labels = df.iloc[:, -1].values
        
        # Ensure 79 features
        if features.shape[1] != 79:
            if features.shape[1] > 79:
                features = features[:, :79]
            else:
                padding = np.zeros((features.shape[0], 79 - features.shape[1]))
                features = np.hstack([features, padding])
        
        all_features.append(features)
        all_labels.extend(labels)
    
    # Load JSON files
    for json_file in json_files:
        try:
            logger.info(f"Loading {json_file.name}")
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            if isinstance(data, dict) and 'events' in data:
                events = data['events']
            elif isinstance(data, list):
                events = data
            else:
                continue
            
            for event in events:
                if 'features' in event and 'label' in event:
                    features = event['features']
                    if len(features) == 79:
                        all_features.append([features])
                        all_labels.append(event['label'])
        except Exception as e:
            logger.warning(f"Failed to load {json_file.name}: {e}")
    
    if not all_features:
        raise ValueError("No data loaded!")
    
    features = np.vstack(all_features).astype(np.float32)
    labels = np.array(all_labels).astype(np.int64)
    
    # Clean data
    features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
    
    logger.info(f"Loaded {len(features):,} samples with {features.shape[1]} features")
    logger.info(f"Classes: {np.unique(labels)}")
    
    return features, labels

def train_model(args):
    """Main training function"""
    logger.info("🚀 Starting Azure ML Training")
    logger.info(f"Data dir: {args.data_dir}")
    logger.info(f"Output dir: {args.output_dir}")
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")
    
    # Load data
    features, labels = load_data(args.data_dir)
    
    # Scale features
    scaler = RobustScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features_scaled, labels, test_size=0.2, stratify=labels, random_state=42
    )
    
    logger.info(f"Train: {len(X_train):,}, Test: {len(X_test):,}")
    
    # Create tensors
    train_features = torch.tensor(X_train, dtype=torch.float32)
    train_labels = torch.tensor(y_train, dtype=torch.long)
    test_features = torch.tensor(X_test, dtype=torch.float32)
    test_labels = torch.tensor(y_test, dtype=torch.long)
    
    # Class balancing
    unique_classes = np.unique(y_train)
    num_classes = len(unique_classes)
    class_weights = compute_class_weight('balanced', classes=unique_classes, y=y_train)
    sample_weights = [class_weights[label] for label in y_train]
    
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    
    # Data loaders
    train_dataset = TensorDataset(train_features, train_labels)
    test_dataset = TensorDataset(test_features, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Model
    model = EnhancedThreatDetector(
        input_dim=79,
        hidden_dims=[512, 256, 128, 64],
        num_classes=num_classes,
        dropout=0.3
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")
    
    # Training setup
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Training loop
    logger.info(f"Starting training for {args.epochs} epochs")
    best_val_loss = float('inf')
    patience_counter = 0
    start_time = time.time()
    
    for epoch in range(args.epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_features, batch_labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
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
        
        train_loss /= len(train_loader)
        val_loss /= len(test_loader)
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        
        logger.info(f"Epoch {epoch+1}/{args.epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best model
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'scaler': scaler,
                'num_classes': num_classes,
                'epoch': epoch,
                'val_loss': val_loss,
                'val_acc': val_acc
            }, f"{args.output_dir}/best_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
    
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time/60:.1f} minutes")
    
    # Final evaluation
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
    
    # Save metrics
    report = classification_report(y_true, y_pred, output_dict=True)
    f1 = f1_score(y_true, y_pred, average='macro')
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    logger.info(f"Final Test Accuracy: {report['accuracy']:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    
    # Save artifacts
    results = {
        'accuracy': float(report['accuracy']),
        'f1_score': float(f1),
        'training_time_minutes': training_time / 60,
        'num_samples': int(len(features)),
        'num_classes': int(num_classes),
        'confusion_matrix': conf_matrix.tolist()
    }
    
    with open(f"{args.output_dir}/metrics.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    joblib.dump(scaler, f"{args.output_dir}/scaler.pkl")
    
    logger.info("✅ Training complete!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--patience', type=int, default=10)
    
    args = parser.parse_args()
    train_model(args)
'''
        
        script_path = Path("/Users/chasemad/Desktop/mini-xdr/scripts/azure-ml/azure_train.py")
        script_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        logger.info(f"📝 Created training script: {script_path}")
        return script_path
    
    def launch_training_job(self, compute_target='gpu-cluster-t4', use_spot=True):
        """Launch training job on Azure ML"""
        logger.info(f"🚀 Launching training job on {compute_target}")
        
        # Create training script
        script_path = self.create_training_script()
        
        # Create job
        job = command(
            code=str(script_path.parent),
            command="python azure_train.py --data-dir ${{inputs.training_data}} --output-dir ${{outputs.model_output}} --batch-size 512 --epochs 50",
            inputs={
                "training_data": Input(
                    type="uri_folder",
                    path="/Users/chasemad/Desktop/mini-xdr/datasets/real_datasets"
                )
            },
            outputs={
                "model_output": {"type": "uri_folder"}
            },
            environment="mini-xdr-training-env:1",
            compute=compute_target,
            display_name="mini-xdr-fast-training",
            description="Fast GPU training for Mini-XDR threat detection (4M+ events)",
            experiment_name="mini-xdr-enterprise-training"
        )
        
        # Submit job
        logger.info("📤 Submitting training job...")
        submitted_job = self.ml_client.jobs.create_or_update(job)
        
        logger.info(f"✅ Job submitted: {submitted_job.name}")
        logger.info(f"📊 Status: {submitted_job.status}")
        logger.info(f"🔗 Studio URL: {submitted_job.studio_url}")
        
        logger.info("\n💡 Monitor your job:")
        logger.info(f"   Azure ML Studio: {submitted_job.studio_url}")
        logger.info(f"   Or run: az ml job show -n {submitted_job.name} -w {self.config['workspace_name']} -g {self.config['resource_group']}")
        
        return submitted_job


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Launch Azure ML Training')
    parser.add_argument('--config', type=str, default='/Users/chasemad/Desktop/mini-xdr/scripts/azure-ml/workspace_config.json')
    parser.add_argument('--compute', type=str, default='gpu-cluster-t4', help='Compute target (gpu-cluster-t4, gpu-cluster-v100, cpu-cluster)')
    parser.add_argument('--spot', action='store_true', help='Use spot instances for lower cost')
    
    args = parser.parse_args()
    
    try:
        launcher = AzureMLTrainingLauncher(config_file=args.config)
        job = launcher.launch_training_job(compute_target=args.compute, use_spot=args.spot)
        
        logger.info("\n🎉 Training job launched successfully!")
        logger.info(f"💰 Estimated cost: $0.30-3.00 (depending on GPU type and duration)")
        logger.info(f"⏱️  Expected duration: 30 minutes - 2 hours")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Failed to launch training: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())


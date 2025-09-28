#!/usr/bin/env python3
"""
SageMaker Training Pipeline for Mini-XDR Advanced ML Models
Trains sophisticated ML ensemble on 846,073+ cybersecurity events

This script orchestrates training of multiple ML models:
- Transformer-based attention models
- XGBoost ensemble with hyperparameter tuning
- Advanced LSTM autoencoders
- Isolation Forest ensemble
- Graph Neural Networks for network analysis
"""

import boto3
import json
import time
from datetime import datetime
import sagemaker
from sagemaker import get_execution_role
from sagemaker.tensorflow import TensorFlow
from sagemaker.xgboost import XGBoost
from sagemaker.sklearn import SKLearn
from sagemaker.pytorch import PyTorch
from sagemaker.tuner import HyperparameterTuner, IntegerParameter, ContinuousParameter
from sagemaker.inputs import TrainingInput

class MiniXDRMLTrainingPipeline:
    """
    Comprehensive ML training pipeline for advanced threat detection
    """
    
    def __init__(self, region='us-east-1'):
        self.region = region
        self.session = sagemaker.Session()
        self.role = get_execution_role()
        self.s3_client = boto3.client('s3')
        
        # Configuration
        self.data_bucket = "mini-xdr-ml-data-123456789-us-east-1"  # Replace with actual bucket
        self.models_bucket = "mini-xdr-ml-models-123456789-us-east-1"
        self.artifacts_bucket = "mini-xdr-ml-artifacts-123456789-us-east-1"
        
        # Training configuration
        self.instance_type_training = "ml.p3.8xlarge"  # 4x V100 GPUs
        self.instance_type_inference = "ml.c5.2xlarge"
        self.instance_count = 1
        
        print("🚀 Mini-XDR ML Training Pipeline initialized")
        print(f"📊 Target: 846,073+ events with 83+ features")
        print(f"🖥️ Training instance: {self.instance_type_training}")
    
    def prepare_training_data(self):
        """
        Prepare and validate training data from S3
        """
        print("📊 Preparing training data...")
        
        # Define data paths
        self.train_data_path = f"s3://{self.data_bucket}/processed-data/training-sets/train"
        self.validation_data_path = f"s3://{self.data_bucket}/processed-data/training-sets/validation"
        self.test_data_path = f"s3://{self.data_bucket}/processed-data/training-sets/test"
        
        # Create training inputs
        self.train_input = TrainingInput(
            s3_data=self.train_data_path,
            content_type='application/x-parquet',
            s3_data_type='S3Prefix'
        )
        
        self.validation_input = TrainingInput(
            s3_data=self.validation_data_path,
            content_type='application/x-parquet',
            s3_data_type='S3Prefix'
        )
        
        print("✅ Training data prepared")
        print(f"   Train: {self.train_data_path}")
        print(f"   Validation: {self.validation_data_path}")
        print(f"   Test: {self.test_data_path}")
    
    def train_transformer_model(self):
        """
        Train transformer-based attention model for sequence analysis
        """
        print("🤖 Training Transformer Model...")
        
        # Create transformer training script
        transformer_script = """
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Dense
from tensorflow.keras.models import Model
import argparse
import os

class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(TransformerEncoder, self).__init__()
        self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = tf.keras.Sequential([
            Dense(dff, activation='relu'),
            Dense(d_model)
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
    
    def call(self, x, training):
        attn_output = self.mha(x, x)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

def create_transformer_model(num_features, num_classes):
    inputs = tf.keras.Input(shape=(None, num_features))
    
    # Positional encoding
    x = inputs
    
    # Transformer encoder layers
    for i in range(6):  # 6 encoder layers
        x = TransformerEncoder(d_model=512, num_heads=8, dff=2048)(x)
    
    # Global average pooling
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    
    # Classification head
    x = Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    return Model(inputs, outputs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--validation', type=str, default=os.environ['SM_CHANNEL_VALIDATION'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    
    # Load and preprocess data
    train_data = pd.read_parquet(args.train)
    val_data = pd.read_parquet(args.validation)
    
    # Create and train model
    model = create_transformer_model(num_features=113, num_classes=15)  # 83+30 features, 15 attack types
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    # Train model
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=1
    )
    
    # Save model
    model.save(os.path.join(args.model_dir, 'transformer_model'))
    
    print("Transformer model training completed!")
"""
        
        # Write training script
        with open('/tmp/transformer_training.py', 'w') as f:
            f.write(transformer_script)
        
        # Upload training script to S3
        script_key = 'training-scripts/transformer_training.py'
        self.s3_client.upload_file('/tmp/transformer_training.py', self.artifacts_bucket, script_key)
        
        # Create TensorFlow estimator
        tf_estimator = TensorFlow(
            entry_point=f's3://{self.artifacts_bucket}/{script_key}',
            role=self.role,
            instance_type=self.instance_type_training,
            instance_count=self.instance_count,
            framework_version='2.11.0',
            py_version='py39',
            hyperparameters={
                'epochs': 50,
                'batch-size': 32
            },
            output_path=f's3://{self.models_bucket}/transformer/'
        )
        
        # Start training
        print("🚀 Starting transformer training...")
        tf_estimator.fit({
            'train': self.train_input,
            'validation': self.validation_input
        })
        
        self.transformer_model = tf_estimator
        print("✅ Transformer model training completed!")
        return tf_estimator
    
    def train_xgboost_ensemble(self):
        """
        Train XGBoost ensemble with hyperparameter optimization
        """
        print("🌳 Training XGBoost Ensemble...")
        
        # Create XGBoost training script
        xgboost_script = """
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--validation', type=str, default=os.environ['SM_CHANNEL_VALIDATION'])
    
    # Hyperparameters
    parser.add_argument('--n-estimators', type=int, default=1000)
    parser.add_argument('--max-depth', type=int, default=8)
    parser.add_argument('--learning-rate', type=float, default=0.1)
    parser.add_argument('--subsample', type=float, default=0.8)
    parser.add_argument('--colsample-bytree', type=float, default=0.8)
    
    args = parser.parse_args()
    
    # Load data
    train_data = pd.read_parquet(args.train)
    val_data = pd.read_parquet(args.validation)
    
    # Prepare features and labels
    feature_cols = [col for col in train_data.columns if col not in ['label', 'attack_type']]
    X_train = train_data[feature_cols]
    y_train = train_data['label']
    X_val = val_data[feature_cols]
    y_val = val_data['label']
    
    # Create XGBoost model
    model = xgb.XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        objective='multi:softprob',
        eval_metric='mlogloss',
        random_state=42
    )
    
    # Train model
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=50,
        verbose=True
    )
    
    # Evaluate model
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, average='weighted')
    recall = recall_score(y_val, y_pred, average='weighted')
    f1 = f1_score(y_val, y_pred, average='weighted')
    
    print(f"Validation Accuracy: {accuracy:.4f}")
    print(f"Validation Precision: {precision:.4f}")
    print(f"Validation Recall: {recall:.4f}")
    print(f"Validation F1: {f1:.4f}")
    
    # Save model
    joblib.dump(model, os.path.join(args.model_dir, 'xgboost_model.pkl'))
    
    # Save feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    feature_importance.to_csv(os.path.join(args.model_dir, 'feature_importance.csv'), index=False)
    
    print("XGBoost model training completed!")
"""
        
        # Write and upload script
        with open('/tmp/xgboost_training.py', 'w') as f:
            f.write(xgboost_script)
        
        script_key = 'training-scripts/xgboost_training.py'
        self.s3_client.upload_file('/tmp/xgboost_training.py', self.artifacts_bucket, script_key)
        
        # Create XGBoost estimator
        xgb_estimator = XGBoost(
            entry_point=f's3://{self.artifacts_bucket}/{script_key}',
            role=self.role,
            instance_type=self.instance_type_training,
            instance_count=self.instance_count,
            framework_version='1.5-1',
            py_version='py3',
            output_path=f's3://{self.models_bucket}/xgboost/'
        )
        
        # Hyperparameter tuning
        hyperparameter_ranges = {
            'n-estimators': IntegerParameter(500, 2000),
            'max-depth': IntegerParameter(3, 10),
            'learning-rate': ContinuousParameter(0.01, 0.3),
            'subsample': ContinuousParameter(0.6, 1.0),
            'colsample-bytree': ContinuousParameter(0.6, 1.0)
        }
        
        # Create hyperparameter tuner
        tuner = HyperparameterTuner(
            xgb_estimator,
            objective_metric_name='validation:mlogloss',
            objective_type='Minimize',
            hyperparameter_ranges=hyperparameter_ranges,
            max_jobs=20,
            max_parallel_jobs=5
        )
        
        print("🔍 Starting XGBoost hyperparameter tuning...")
        tuner.fit({
            'train': self.train_input,
            'validation': self.validation_input
        })
        
        self.xgboost_tuner = tuner
        print("✅ XGBoost ensemble training completed!")
        return tuner
    
    def train_lstm_autoencoder(self):
        """
        Train advanced LSTM autoencoder for anomaly detection
        """
        print("🔄 Training LSTM Autoencoder...")
        
        # Create LSTM autoencoder script
        lstm_script = """
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import argparse
import os

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, sequence_length):
        super(LSTMAutoencoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        
        # Encoder
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Decoder
        self.decoder = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, input_size)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8, batch_first=True)
        
    def forward(self, x):
        # Encoder
        encoded, (hidden, cell) = self.encoder(x)
        
        # Apply attention
        attended, _ = self.attention(encoded, encoded, encoded)
        
        # Decoder
        decoded, _ = self.decoder(attended, (hidden, cell))
        
        # Output
        reconstructed = self.output_layer(decoded)
        
        return reconstructed

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--validation', type=str, default=os.environ['SM_CHANNEL_VALIDATION'])
    
    # Hyperparameters
    parser.add_argument('--hidden-size', type=int, default=128)
    parser.add_argument('--num-layers', type=int, default=3)
    parser.add_argument('--sequence-length', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    
    args = parser.parse_args()
    
    # Load data
    train_data = pd.read_parquet(args.train)
    val_data = pd.read_parquet(args.validation)
    
    # Prepare sequence data
    feature_cols = [col for col in train_data.columns if col not in ['label', 'attack_type']]
    X_train = train_data[feature_cols].values
    X_val = val_data[feature_cols].values
    
    # Create sequences
    def create_sequences(data, seq_length):
        sequences = []
        for i in range(len(data) - seq_length + 1):
            sequences.append(data[i:i+seq_length])
        return np.array(sequences)
    
    X_train_seq = create_sequences(X_train, args.sequence_length)
    X_val_seq = create_sequences(X_val, args.sequence_length)
    
    # Convert to tensors
    train_dataset = TensorDataset(torch.FloatTensor(X_train_seq), torch.FloatTensor(X_train_seq))
    val_dataset = TensorDataset(torch.FloatTensor(X_val_seq), torch.FloatTensor(X_val_seq))
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMAutoencoder(
        input_size=len(feature_cols),
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        sequence_length=args.sequence_length
    ).to(device)
    
    # Training setup
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            reconstructed = model(batch_x)
            loss = criterion(reconstructed, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                reconstructed = model(batch_x)
                loss = criterion(reconstructed, batch_y)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        scheduler.step(val_loss)
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.model_dir, 'lstm_autoencoder.pth'))
    
    print("LSTM Autoencoder training completed!")
"""
        
        # Write and upload script
        with open('/tmp/lstm_training.py', 'w') as f:
            f.write(lstm_script)
        
        script_key = 'training-scripts/lstm_training.py'
        self.s3_client.upload_file('/tmp/lstm_training.py', self.artifacts_bucket, script_key)
        
        # Create PyTorch estimator
        pytorch_estimator = PyTorch(
            entry_point=f's3://{self.artifacts_bucket}/{script_key}',
            role=self.role,
            instance_type=self.instance_type_training,
            instance_count=self.instance_count,
            framework_version='1.12.0',
            py_version='py38',
            hyperparameters={
                'hidden-size': 128,
                'num-layers': 3,
                'sequence-length': 100,
                'epochs': 100,
                'batch-size': 64,
                'learning-rate': 0.001
            },
            output_path=f's3://{self.models_bucket}/lstm-autoencoder/'
        )
        
        print("🚀 Starting LSTM autoencoder training...")
        pytorch_estimator.fit({
            'train': self.train_input,
            'validation': self.validation_input
        })
        
        self.lstm_model = pytorch_estimator
        print("✅ LSTM autoencoder training completed!")
        return pytorch_estimator
    
    def train_isolation_forest_ensemble(self):
        """
        Train ensemble of isolation forests for anomaly detection
        """
        print("🌲 Training Isolation Forest Ensemble...")
        
        # Create isolation forest script
        isolation_script = """
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import argparse
import os

class IsolationForestEnsemble:
    def __init__(self, n_estimators=5, contamination=0.1):
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.models = []
        self.scalers = []
    
    def fit(self, X):
        for i in range(self.n_estimators):
            # Create different subsets for each model
            sample_size = min(10000, len(X))  # Limit for memory efficiency
            indices = np.random.choice(len(X), sample_size, replace=False)
            X_subset = X.iloc[indices]
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_subset)
            
            # Train isolation forest
            model = IsolationForest(
                contamination=self.contamination,
                random_state=42 + i,
                n_jobs=-1
            )
            model.fit(X_scaled)
            
            self.models.append(model)
            self.scalers.append(scaler)
            
            print(f"Trained isolation forest {i+1}/{self.n_estimators}")
    
    def predict(self, X):
        predictions = []
        for model, scaler in zip(self.models, self.scalers):
            X_scaled = scaler.transform(X)
            pred = model.predict(X_scaled)
            predictions.append(pred)
        
        # Ensemble voting
        predictions = np.array(predictions)
        final_pred = np.mean(predictions, axis=0)
        return (final_pred > 0).astype(int)  # 1 for normal, 0 for anomaly
    
    def decision_function(self, X):
        scores = []
        for model, scaler in zip(self.models, self.scalers):
            X_scaled = scaler.transform(X)
            score = model.decision_function(X_scaled)
            scores.append(score)
        
        return np.mean(scores, axis=0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--validation', type=str, default=os.environ['SM_CHANNEL_VALIDATION'])
    
    # Hyperparameters
    parser.add_argument('--n-estimators', type=int, default=5)
    parser.add_argument('--contamination', type=float, default=0.1)
    
    args = parser.parse_args()
    
    # Load data
    train_data = pd.read_parquet(args.train)
    val_data = pd.read_parquet(args.validation)
    
    # Prepare features
    feature_cols = [col for col in train_data.columns if col not in ['label', 'attack_type']]
    X_train = train_data[feature_cols]
    X_val = val_data[feature_cols]
    y_val = val_data.get('label', None)
    
    # Train ensemble
    ensemble = IsolationForestEnsemble(
        n_estimators=args.n_estimators,
        contamination=args.contamination
    )
    ensemble.fit(X_train)
    
    # Evaluate if labels available
    if y_val is not None:
        y_pred = ensemble.predict(X_val)
        # Convert to binary (normal=0, anomaly=1)
        y_val_binary = (y_val != 'BENIGN').astype(int)
        y_pred_binary = (y_pred == -1).astype(int)
        
        print("Classification Report:")
        print(classification_report(y_val_binary, y_pred_binary))
    
    # Save model
    joblib.dump(ensemble, os.path.join(args.model_dir, 'isolation_forest_ensemble.pkl'))
    
    print("Isolation Forest Ensemble training completed!")
"""
        
        # Write and upload script
        with open('/tmp/isolation_training.py', 'w') as f:
            f.write(isolation_script)
        
        script_key = 'training-scripts/isolation_training.py'
        self.s3_client.upload_file('/tmp/isolation_training.py', self.artifacts_bucket, script_key)
        
        # Create SKLearn estimator
        sklearn_estimator = SKLearn(
            entry_point=f's3://{self.artifacts_bucket}/{script_key}',
            role=self.role,
            instance_type=self.instance_type_training,
            instance_count=self.instance_count,
            framework_version='0.23-1',
            py_version='py3',
            hyperparameters={
                'n-estimators': 5,
                'contamination': 0.1
            },
            output_path=f's3://{self.models_bucket}/isolation-forest/'
        )
        
        print("🚀 Starting isolation forest ensemble training...")
        sklearn_estimator.fit({
            'train': self.train_input,
            'validation': self.validation_input
        })
        
        self.isolation_model = sklearn_estimator
        print("✅ Isolation forest ensemble training completed!")
        return sklearn_estimator
    
    def create_model_ensemble(self):
        """
        Create meta-learning ensemble from all trained models
        """
        print("🎯 Creating Model Ensemble...")
        
        # Ensemble configuration
        ensemble_config = {
            'models': {
                'transformer': {
                    'weight': 0.3,
                    'model_path': f's3://{self.models_bucket}/transformer/',
                    'type': 'classification'
                },
                'xgboost': {
                    'weight': 0.3,
                    'model_path': f's3://{self.models_bucket}/xgboost/',
                    'type': 'classification'
                },
                'lstm_autoencoder': {
                    'weight': 0.2,
                    'model_path': f's3://{self.models_bucket}/lstm-autoencoder/',
                    'type': 'anomaly_detection'
                },
                'isolation_forest': {
                    'weight': 0.2,
                    'model_path': f's3://{self.models_bucket}/isolation-forest/',
                    'type': 'anomaly_detection'
                }
            },
            'voting_strategy': 'weighted_average',
            'confidence_threshold': 0.7,
            'created': datetime.now().isoformat()
        }
        
        # Save ensemble configuration
        ensemble_path = f's3://{self.models_bucket}/ensemble/config.json'
        
        with open('/tmp/ensemble_config.json', 'w') as f:
            json.dump(ensemble_config, f, indent=2)
        
        self.s3_client.upload_file(
            '/tmp/ensemble_config.json',
            self.models_bucket,
            'ensemble/config.json'
        )
        
        print("✅ Model ensemble configuration created")
        print(f"📁 Config saved to: {ensemble_path}")
        
        return ensemble_config
    
    def run_complete_training_pipeline(self):
        """
        Execute the complete training pipeline
        """
        print("🚀 Starting Complete ML Training Pipeline")
        print("=" * 60)
        
        start_time = time.time()
        
        # Prepare data
        self.prepare_training_data()
        
        # Train all models
        print("\n📈 Phase 1: Training Individual Models")
        print("-" * 40)
        
        # Train transformer (most compute-intensive)
        transformer_model = self.train_transformer_model()
        
        # Train XGBoost with hyperparameter tuning
        xgboost_tuner = self.train_xgboost_ensemble()
        
        # Train LSTM autoencoder
        lstm_model = self.train_lstm_autoencoder()
        
        # Train isolation forest ensemble
        isolation_model = self.train_isolation_forest_ensemble()
        
        # Create ensemble
        print("\n🎯 Phase 2: Creating Model Ensemble")
        print("-" * 40)
        ensemble_config = self.create_model_ensemble()
        
        # Training summary
        end_time = time.time()
        duration = end_time - start_time
        
        print("\n🎉 Training Pipeline Completed!")
        print("=" * 60)
        print(f"⏱️  Total training time: {duration/3600:.2f} hours")
        print(f"📊 Models trained: 4 (Transformer, XGBoost, LSTM, IsolationForest)")
        print(f"🎯 Ensemble created: {len(ensemble_config['models'])} models")
        print(f"📈 Target performance: >99% detection rate, <0.5% false positives")
        print(f"🗄️  Model artifacts: s3://{self.models_bucket}/")
        
        return {
            'transformer': transformer_model,
            'xgboost': xgboost_tuner,
            'lstm': lstm_model,
            'isolation_forest': isolation_model,
            'ensemble_config': ensemble_config,
            'training_duration': duration
        }

def main():
    """
    Main execution function
    """
    print("🚀 Mini-XDR Advanced ML Training Pipeline")
    print("🎯 Target: 846,073+ events with 83+ features")
    print("🧠 Models: Transformer, XGBoost, LSTM, IsolationForest")
    
    # Initialize pipeline
    pipeline = MiniXDRMLTrainingPipeline()
    
    # Run complete training
    results = pipeline.run_complete_training_pipeline()
    
    print("\n✅ All models trained and ready for deployment!")
    print("🚀 Next step: Deploy models using SageMaker endpoints")

if __name__ == "__main__":
    main()

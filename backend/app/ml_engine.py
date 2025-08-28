"""
ML-Based Anomaly Detection Engine with Ensemble Models
Supports Isolation Forest, LSTM Autoencoder, XGBoost, and Federated Learning
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import joblib
import logging
import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
import os

from .models import Event, Incident, MLModel
from .config import settings

logger = logging.getLogger(__name__)


class LSTMAutoencoder(nn.Module):
    """LSTM-based autoencoder for sequence anomaly detection"""
    
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, sequence_length: int = 10):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        
        # Encoder
        self.encoder = nn.LSTM(
            input_size, hidden_size, num_layers, 
            batch_first=True, dropout=0.2
        )
        
        # Decoder
        self.decoder = nn.LSTM(
            hidden_size, input_size, num_layers, 
            batch_first=True, dropout=0.2
        )
        
        # Output layer
        self.output_layer = nn.Linear(input_size, input_size)
        
    def forward(self, x):
        # Encode
        encoded, (hidden, cell) = self.encoder(x)
        
        # Take the last hidden state and repeat for sequence length
        hidden_repeated = hidden[-1].unsqueeze(1).repeat(1, self.sequence_length, 1)
        
        # Decode
        decoded, _ = self.decoder(hidden_repeated)
        
        # Output
        output = self.output_layer(decoded)
        return output


class BaseMLDetector:
    """Base class for ML anomaly detectors"""
    
    def __init__(self):
        self.feature_columns = [
            'event_count_1h', 'event_count_24h', 'unique_ports',
            'failed_login_count', 'session_duration_avg', 'password_diversity',
            'username_diversity', 'event_rate_per_minute', 'time_of_day',
            'is_weekend', 'unique_usernames', 'password_length_avg',
            'command_diversity', 'download_attempts', 'upload_attempts'
        ]
        self.scaler = StandardScaler()
        self.model_dir = Path("models")
        self.model_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def _extract_features(self, src_ip: str, events: List[Event]) -> Dict[str, float]:
        """Extract features from events for ML analysis"""
        if not events:
            return {col: 0.0 for col in self.feature_columns}
        
        features = {}
        
        # Time-based features
        now = datetime.utcnow()
        events_1h = [e for e in events if (now - e.ts).total_seconds() <= 3600]
        events_24h = [e for e in events if (now - e.ts).total_seconds() <= 86400]
        
        features['event_count_1h'] = len(events_1h)
        features['event_count_24h'] = len(events_24h)
        
        # Port diversity
        unique_ports = len(set(e.dst_port for e in events if e.dst_port))
        features['unique_ports'] = unique_ports
        
        # Failed login analysis
        failed_logins = [e for e in events if e.eventid == "cowrie.login.failed"]
        features['failed_login_count'] = len(failed_logins)
        
        # Session analysis
        if events:
            time_span = (events[0].ts - events[-1].ts).total_seconds()
            features['session_duration_avg'] = time_span / max(len(events), 1)
            features['event_rate_per_minute'] = len(events) / max(time_span / 60, 1)
        else:
            features['session_duration_avg'] = 0
            features['event_rate_per_minute'] = 0
        
        # Credential analysis
        usernames = set()
        passwords = set()
        password_lengths = []
        
        for event in failed_logins:
            if hasattr(event, 'raw') and event.raw:
                raw_data = event.raw if isinstance(event.raw, dict) else {}
                if 'username' in raw_data:
                    usernames.add(raw_data['username'])
                if 'password' in raw_data:
                    passwords.add(raw_data['password'])
                    password_lengths.append(len(str(raw_data['password'])))
        
        features['unique_usernames'] = len(usernames)
        features['password_diversity'] = len(passwords)
        features['username_diversity'] = len(usernames)
        features['password_length_avg'] = np.mean(password_lengths) if password_lengths else 0
        
        # Command analysis
        commands = set()
        download_count = 0
        upload_count = 0
        
        for event in events:
            if event.eventid == "cowrie.command.input":
                if hasattr(event, 'raw') and event.raw:
                    raw_data = event.raw if isinstance(event.raw, dict) else {}
                    if 'input' in raw_data:
                        commands.add(raw_data['input'].split()[0] if raw_data['input'] else '')
            elif event.eventid in ["cowrie.session.file_download", "cowrie.session.file_upload"]:
                if "download" in event.eventid:
                    download_count += 1
                else:
                    upload_count += 1
        
        features['command_diversity'] = len(commands)
        features['download_attempts'] = download_count
        features['upload_attempts'] = upload_count
        
        # Time-based features
        if events:
            avg_hour = np.mean([e.ts.hour for e in events])
            features['time_of_day'] = avg_hour / 24.0  # Normalize to 0-1
            features['is_weekend'] = float(any(e.ts.weekday() >= 5 for e in events))
        else:
            features['time_of_day'] = 0.5
            features['is_weekend'] = 0.0
        
        # Ensure all features are present
        for col in self.feature_columns:
            if col not in features:
                features[col] = 0.0
        
        return features


class IsolationForestDetector(BaseMLDetector):
    """Isolation Forest for unsupervised anomaly detection"""
    
    def __init__(self, contamination: float = 0.1):
        super().__init__()
        self.model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        self.is_trained = False
    
    async def calculate_anomaly_score(self, src_ip: str, events: List[Event]) -> float:
        """Calculate anomaly score for the given IP and events"""
        if not self.is_trained:
            return 0.0
        
        features = self._extract_features(src_ip, events)
        feature_vector = np.array([list(features.values())]).reshape(1, -1)
        
        # Scale features
        feature_vector_scaled = self.scaler.transform(feature_vector)
        
        # Get anomaly score (-1 to 1, where -1 is anomaly)
        score = self.model.decision_function(feature_vector_scaled)[0]
        
        # Convert to 0-1 probability (higher = more anomalous)
        # Isolation Forest returns negative scores for anomalies
        normalized_score = max(0, -score / 2 + 0.5)
        return min(normalized_score, 1.0)
    
    async def train_model(self, training_data: List[Dict[str, float]]) -> bool:
        """Train the Isolation Forest model"""
        try:
            if len(training_data) < 10:
                self.logger.warning("Insufficient training data for Isolation Forest")
                return False
            
            df = pd.DataFrame(training_data)
            X = df[self.feature_columns].fillna(0)
            
            # Fit scaler
            X_scaled = self.scaler.fit_transform(X)
            
            # Train model
            self.model.fit(X_scaled)
            self.is_trained = True
            
            # Save model
            model_path = self.model_dir / "isolation_forest.pkl"
            scaler_path = self.model_dir / "isolation_forest_scaler.pkl"
            
            joblib.dump(self.model, model_path)
            joblib.dump(self.scaler, scaler_path)
            
            self.logger.info(f"Isolation Forest trained with {len(training_data)} samples")
            return True
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            return False
    
    def load_model(self) -> bool:
        """Load pre-trained model"""
        try:
            model_path = self.model_dir / "isolation_forest.pkl"
            scaler_path = self.model_dir / "isolation_forest_scaler.pkl"
            
            if model_path.exists() and scaler_path.exists():
                self.model = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)
                self.is_trained = True
                self.logger.info("Isolation Forest model loaded")
                return True
        except Exception as e:
            self.logger.error(f"Model loading failed: {e}")
        
        return False


class LSTMDetector(BaseMLDetector):
    """LSTM Autoencoder for sequence anomaly detection"""
    
    def __init__(self, sequence_length: int = 10, hidden_size: int = 64):
        super().__init__()
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.criterion = nn.MSELoss()
        self.is_trained = False
        self.reconstruction_threshold = 0.05  # Will be set during training
    
    async def calculate_anomaly_score(self, src_ip: str, events: List[Event]) -> float:
        """Calculate anomaly score using LSTM reconstruction error"""
        if not self.is_trained or len(events) < self.sequence_length:
            return 0.0
        
        try:
            # Prepare sequence data
            sequence_data = self._prepare_sequence_data(events)
            if sequence_data.shape[0] == 0:
                return 0.0
            
            # Convert to tensor
            sequence_tensor = torch.tensor(
                sequence_data, dtype=torch.float32
            ).unsqueeze(0).to(self.device)
            
            # Get reconstruction
            self.model.eval()
            with torch.no_grad():
                reconstructed = self.model(sequence_tensor)
                mse = self.criterion(reconstructed, sequence_tensor).item()
            
            # Normalize to 0-1 score
            normalized_score = min(mse / self.reconstruction_threshold, 1.0)
            return normalized_score
            
        except Exception as e:
            self.logger.error(f"LSTM scoring error: {e}")
            return 0.0
    
    def _prepare_sequence_data(self, events: List[Event]) -> np.ndarray:
        """Prepare event sequence for LSTM input"""
        # Extract features for each event in time order
        event_features = []
        for event in events[-self.sequence_length:]:  # Take last N events
            features = self._extract_features(event.src_ip, [event])
            event_features.append(list(features.values()))
        
        # Pad if necessary
        while len(event_features) < self.sequence_length:
            event_features.insert(0, [0.0] * len(self.feature_columns))
        
        return np.array(event_features[-self.sequence_length:]).reshape(
            self.sequence_length, len(self.feature_columns)
        )
    
    async def train_model(self, training_data: List[Dict[str, float]]) -> bool:
        """Train the LSTM autoencoder"""
        try:
            if len(training_data) < 100:
                self.logger.warning("Insufficient training data for LSTM")
                return False
            
            # Prepare sequence datasets
            sequences = self._prepare_training_sequences(training_data)
            if len(sequences) == 0:
                return False
            
            # Initialize model
            input_size = len(self.feature_columns)
            self.model = LSTMAutoencoder(
                input_size=input_size,
                hidden_size=self.hidden_size,
                sequence_length=self.sequence_length
            ).to(self.device)
            
            # Prepare data loader
            dataset = TensorDataset(torch.tensor(sequences, dtype=torch.float32))
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
            
            # Training
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            
            self.model.train()
            reconstruction_errors = []
            
            for epoch in range(50):  # Training epochs
                epoch_loss = 0.0
                for batch in dataloader:
                    batch_data = batch[0].to(self.device)
                    
                    optimizer.zero_grad()
                    reconstructed = self.model(batch_data)
                    loss = self.criterion(reconstructed, batch_data)
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    reconstruction_errors.append(loss.item())
                
                if epoch % 10 == 0:
                    self.logger.info(f"LSTM Epoch {epoch}, Loss: {epoch_loss:.4f}")
            
            # Set reconstruction threshold based on training errors
            self.reconstruction_threshold = np.percentile(reconstruction_errors, 95)
            
            # Save model
            model_path = self.model_dir / "lstm_autoencoder.pth"
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'threshold': self.reconstruction_threshold,
                'input_size': input_size,
                'hidden_size': self.hidden_size,
                'sequence_length': self.sequence_length
            }, model_path)
            
            self.is_trained = True
            self.logger.info(f"LSTM autoencoder trained with threshold {self.reconstruction_threshold:.4f}")
            return True
            
        except Exception as e:
            self.logger.error(f"LSTM training failed: {e}")
            return False
    
    def _prepare_training_sequences(self, training_data: List[Dict[str, float]]) -> np.ndarray:
        """Prepare training sequences from feature data"""
        df = pd.DataFrame(training_data)
        sequences = []
        
        # Create sliding window sequences
        for i in range(len(df) - self.sequence_length + 1):
            sequence = df.iloc[i:i + self.sequence_length][self.feature_columns].fillna(0).values
            sequences.append(sequence)
        
        return np.array(sequences)
    
    def load_model(self) -> bool:
        """Load pre-trained LSTM model"""
        try:
            model_path = self.model_dir / "lstm_autoencoder.pth"
            
            if model_path.exists():
                checkpoint = torch.load(model_path, map_location=self.device)
                
                self.model = LSTMAutoencoder(
                    input_size=checkpoint['input_size'],
                    hidden_size=checkpoint['hidden_size'],
                    sequence_length=checkpoint['sequence_length']
                ).to(self.device)
                
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.reconstruction_threshold = checkpoint['threshold']
                self.is_trained = True
                
                self.logger.info("LSTM autoencoder model loaded")
                return True
        except Exception as e:
            self.logger.error(f"LSTM model loading failed: {e}")
        
        return False


class EnsembleMLDetector:
    """Ensemble of multiple ML detectors for robust anomaly detection"""
    
    def __init__(self):
        self.isolation_forest = IsolationForestDetector()
        self.lstm_detector = LSTMDetector()
        self.logger = logging.getLogger(__name__)
        
        # Model weights for ensemble
        self.weights = {
            'isolation_forest': 0.4,
            'lstm': 0.6
        }
    
    async def calculate_anomaly_score(self, src_ip: str, events: List[Event]) -> float:
        """Calculate ensemble anomaly score"""
        scores = {}
        
        # Get individual model scores
        iso_score = await self.isolation_forest.calculate_anomaly_score(src_ip, events)
        scores['isolation_forest'] = iso_score
        
        lstm_score = await self.lstm_detector.calculate_anomaly_score(src_ip, events)
        scores['lstm'] = lstm_score
        
        # Calculate weighted average
        total_weight = 0
        weighted_sum = 0
        
        for model_name, score in scores.items():
            if score > 0:  # Only include models that returned a score
                weight = self.weights.get(model_name, 0.1)
                weighted_sum += score * weight
                total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        ensemble_score = weighted_sum / total_weight
        return min(ensemble_score, 1.0)
    
    async def train_models(self, training_data: List[Dict[str, float]]) -> Dict[str, bool]:
        """Train all models in the ensemble"""
        results = {}
        
        # Train Isolation Forest
        results['isolation_forest'] = await self.isolation_forest.train_model(training_data)
        
        # Train LSTM
        results['lstm'] = await self.lstm_detector.train_model(training_data)
        
        self.logger.info(f"Ensemble training results: {results}")
        return results
    
    def load_models(self) -> Dict[str, bool]:
        """Load all pre-trained models"""
        results = {}
        
        results['isolation_forest'] = self.isolation_forest.load_model()
        results['lstm'] = self.lstm_detector.load_model()
        
        self.logger.info(f"Model loading results: {results}")
        return results
    
    def get_model_status(self) -> Dict[str, bool]:
        """Get training status of all models"""
        return {
            'isolation_forest': self.isolation_forest.is_trained,
            'lstm': self.lstm_detector.is_trained
        }


# Global ensemble detector instance
ml_detector = EnsembleMLDetector()


async def prepare_training_data_from_events(events: List[Event]) -> List[Dict[str, float]]:
    """Prepare training data from event history"""
    if not events:
        return []
    
    # Group events by source IP
    ip_groups = {}
    for event in events:
        if event.src_ip not in ip_groups:
            ip_groups[event.src_ip] = []
        ip_groups[event.src_ip].append(event)
    
    # Extract features for each IP
    training_data = []
    detector = BaseMLDetector()  # Use base detector for feature extraction
    
    for src_ip, ip_events in ip_groups.items():
        features = detector._extract_features(src_ip, ip_events)
        training_data.append(features)
    
    return training_data

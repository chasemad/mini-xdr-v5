"""
Deep Learning Model Integration for Mini-XDR
Integrates trained PyTorch models with the backend system
"""

import torch
import torch.nn as nn
import numpy as np
import joblib
import logging
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from .models import Event

logger = logging.getLogger(__name__)

class XDRThreatDetector(nn.Module):
    """
    Deep neural network for multi-class threat detection
    (Same architecture as training script)
    """
    def __init__(self, input_dim, hidden_dims=[512, 256, 128, 64], num_classes=2, dropout_rate=0.3):
        super(XDRThreatDetector, self).__init__()

        layers = []
        prev_dim = input_dim

        for i, dim in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class XDRAnomalyDetector(nn.Module):
    """
    Autoencoder for unsupervised anomaly detection
    (Same architecture as training script)
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

        # Self-attention
        attn_out, attention_weights = self.attention(lstm_out, lstm_out, lstm_out)

        # Attention pooling
        attention_scores = torch.softmax(self.attention_pooling(attn_out), dim=1)
        pooled = torch.sum(attn_out * attention_scores, dim=1)

        # Classification
        output = self.classifier(pooled)

        return output, attention_weights

class DeepLearningModelManager:
    """
    Manages all deep learning models for threat detection
    """
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Model instances
        self.threat_detector = None
        self.anomaly_detector = None
        self.lstm_detector = None

        # Preprocessing
        self.scaler = None
        self.label_encoder = None

        # Model metadata
        self.metadata = {}
        self.is_loaded = False

        # Feature extraction (consistent with training)
        self.feature_columns = [
            'event_count_1h', 'event_count_24h', 'unique_ports',
            'failed_login_count', 'session_duration_avg', 'password_diversity',
            'username_diversity', 'event_rate_per_minute', 'time_of_day',
            'is_weekend', 'unique_usernames', 'password_length_avg',
            'command_diversity', 'download_attempts', 'upload_attempts'
        ]

        logger.info(f"Deep Learning Model Manager initialized on {self.device}")

    def load_models(self, model_path: Optional[str] = None) -> Dict[str, bool]:
        """Load all available deep learning models"""
        results = {}

        try:
            # Try to load from SageMaker first, then local models
            if model_path:
                # SageMaker model path
                model_dir = Path(model_path)
            else:
                # Local model directory
                model_dir = self.model_dir

            logger.info(f"Loading models from: {model_dir}")

            # Load metadata first
            metadata_path = model_dir / "model_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                logger.info(f"Loaded metadata: {self.metadata.get('model_type', 'unknown')}")

            # Load preprocessing
            scaler_path = model_dir / "scaler.pkl"
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
                results['scaler'] = True
                logger.info("Scaler loaded successfully")

            label_encoder_path = model_dir / "label_encoder.pkl"
            if label_encoder_path.exists():
                self.label_encoder = joblib.load(label_encoder_path)
                results['label_encoder'] = True
                logger.info("Label encoder loaded successfully")

            # Load deep learning models
            results.update(self._load_deep_models(model_dir))

            # Check if any models loaded successfully
            self.is_loaded = any(results.values())

            if self.is_loaded:
                logger.info(f"Successfully loaded models: {[k for k, v in results.items() if v]}")
            else:
                logger.warning("No deep learning models loaded successfully")

            return results

        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            return {'error': str(e)}

    def _load_deep_models(self, model_dir: Path) -> Dict[str, bool]:
        """Load PyTorch models"""
        results = {}

        # Load threat detector
        threat_detector_path = model_dir / "threat_detector.pth"
        if threat_detector_path.exists():
            try:
                # Get model parameters from metadata
                features = self.metadata.get('features', 79)
                num_classes = self.metadata.get('num_classes', 2)

                # Use same architecture as training (small model for single GPU)
                self.threat_detector = XDRThreatDetector(
                    input_dim=features,
                    hidden_dims=[256, 128, 64],  # Small model architecture
                    num_classes=num_classes,
                    dropout_rate=0.2
                ).to(self.device)

                self.threat_detector.load_state_dict(torch.load(
                    threat_detector_path, map_location=self.device, weights_only=True
                ))
                self.threat_detector.eval()

                results['threat_detector'] = True
                logger.info("Deep threat detector loaded successfully")

            except Exception as e:
                logger.error(f"Failed to load threat detector: {e}")
                results['threat_detector'] = False

        # Load anomaly detector
        anomaly_detector_path = model_dir / "anomaly_detector.pth"
        if anomaly_detector_path.exists():
            try:
                features = self.metadata.get('features', 79)

                # Use same architecture as training (small model)
                self.anomaly_detector = XDRAnomalyDetector(
                    input_dim=features,
                    encoding_dims=[128, 64],  # Small model architecture
                    latent_dim=16
                ).to(self.device)

                self.anomaly_detector.load_state_dict(torch.load(
                    anomaly_detector_path, map_location=self.device, weights_only=True
                ))
                self.anomaly_detector.eval()

                results['anomaly_detector'] = True
                logger.info("Deep anomaly detector loaded successfully")

            except Exception as e:
                logger.error(f"Failed to load anomaly detector: {e}")
                results['anomaly_detector'] = False

        # Load LSTM detector (if available from sequential training)
        lstm_detector_path = model_dir / "lstm_attention.pth"
        if lstm_detector_path.exists():
            try:
                features = self.metadata.get('features', 79)
                num_classes = self.metadata.get('num_classes', 2)

                self.lstm_detector = LSTMAttentionDetector(
                    input_dim=features,
                    num_classes=num_classes
                ).to(self.device)

                self.lstm_detector.load_state_dict(torch.load(
                    lstm_detector_path, map_location=self.device, weights_only=True
                ))
                self.lstm_detector.eval()

                results['lstm_detector'] = True
                logger.info("LSTM attention detector loaded successfully")

            except Exception as e:
                logger.error(f"Failed to load LSTM detector: {e}")
                results['lstm_detector'] = False

        return results

    def _extract_features(self, src_ip: str, events: List[Event]) -> Dict[str, float]:
        """Extract features from events (consistent with training)"""
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
            features['time_of_day'] = avg_hour / 24.0
            features['is_weekend'] = float(any(e.ts.weekday() >= 5 for e in events))
        else:
            features['time_of_day'] = 0.5
            features['is_weekend'] = 0.0

        # Ensure all features are present
        for col in self.feature_columns:
            if col not in features:
                features[col] = 0.0

        return features

    async def calculate_threat_score(self, src_ip: str, events: List[Event]) -> Dict[str, float]:
        """Calculate threat scores using all available deep learning models"""
        if not self.is_loaded or not events:
            return {'ensemble_score': 0.0, 'confidence': 0.0}

        try:
            # Extract and prepare features
            features = self._extract_features(src_ip, events)
            feature_vector = np.array([list(features.values())]).astype(np.float32)

            # Scale features
            if self.scaler:
                feature_vector = self.scaler.transform(feature_vector)

            # Convert to tensor
            input_tensor = torch.tensor(feature_vector, dtype=torch.float32).to(self.device)

            scores = {}

            # Deep threat detection
            if self.threat_detector:
                with torch.no_grad():
                    logits = self.threat_detector(input_tensor)
                    probabilities = torch.softmax(logits, dim=1)
                    predicted_class = torch.argmax(probabilities, dim=1).item()

                    # For 7-class model: 0=Normal, 1-6=Various Attack Types
                    # Calculate overall threat probability (1 - normal_prob)
                    if probabilities.shape[1] == 7:  # Multi-class model
                        normal_prob = probabilities[0, 0].item()  # Class 0 = Normal
                        threat_prob = 1.0 - normal_prob  # Overall attack probability

                        # Store detailed attack classification
                        attack_classes = {
                            0: "Normal",
                            1: "DDoS/DoS Attack",
                            2: "Network Reconnaissance",
                            3: "Brute Force Attack",
                            4: "Web Application Attack",
                            5: "Malware/Botnet",
                            6: "Advanced Persistent Threat"
                        }
                        scores['attack_type'] = attack_classes.get(predicted_class, "Unknown")
                        scores['attack_confidence'] = probabilities[0, predicted_class].item()

                    else:  # Binary model fallback
                        threat_prob = probabilities[0, 1].item() if probabilities.shape[1] > 1 else probabilities[0, 0].item()
                        scores['attack_type'] = "Threat" if predicted_class == 1 else "Normal"
                        scores['attack_confidence'] = threat_prob

                    scores['threat_detector'] = threat_prob

            # Anomaly detection
            if self.anomaly_detector:
                with torch.no_grad():
                    reconstruction_error = self.anomaly_detector.get_reconstruction_error(input_tensor)
                    # Normalize reconstruction error to 0-1 scale
                    anomaly_score = torch.clamp(reconstruction_error / 10.0, 0, 1).item()
                    scores['anomaly_detector'] = anomaly_score

            # Sequential detection (if available and enough events)
            if self.lstm_detector and len(events) >= 10:
                # This would require sequence preparation - simplified for now
                scores['lstm_detector'] = 0.5  # Placeholder

            # Calculate ensemble score
            if scores:
                weights = {
                    'threat_detector': 0.4,
                    'anomaly_detector': 0.4,
                    'lstm_detector': 0.2
                }

                weighted_sum = 0.0
                total_weight = 0.0

                for model_name, score in scores.items():
                    weight = weights.get(model_name, 0.1)
                    weighted_sum += score * weight
                    total_weight += weight

                ensemble_score = weighted_sum / total_weight if total_weight > 0 else 0.0

                # Calculate confidence based on model agreement
                score_values = list(scores.values())
                confidence = 1.0 - np.var(score_values) if len(score_values) > 1 else 0.8

                return {
                    'ensemble_score': min(ensemble_score, 1.0),
                    'confidence': min(confidence, 1.0),
                    'individual_scores': scores,
                    'model_count': len(scores)
                }

            return {'ensemble_score': 0.0, 'confidence': 0.0}

        except Exception as e:
            logger.error(f"Deep learning threat scoring failed: {e}")
            return {'ensemble_score': 0.0, 'confidence': 0.0, 'error': str(e)}

    def get_model_status(self) -> Dict[str, Any]:
        """Get status of all loaded models"""
        return {
            'deep_learning_loaded': self.is_loaded,
            'device': str(self.device),
            'models_available': {
                'threat_detector': self.threat_detector is not None,
                'anomaly_detector': self.anomaly_detector is not None,
                'lstm_detector': self.lstm_detector is not None
            },
            'preprocessing_loaded': {
                'scaler': self.scaler is not None,
                'label_encoder': self.label_encoder is not None
            },
            'metadata': self.metadata,
            'gpu_available': torch.cuda.is_available(),
            'feature_count': len(self.feature_columns)
        }

# Global instance
deep_learning_manager = DeepLearningModelManager()
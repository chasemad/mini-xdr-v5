"""
Deep Learning Model Integration for Mini-XDR
Integrates trained PyTorch models with the backend system
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import torch
import torch.nn as nn

from .models import Event

logger = logging.getLogger(__name__)


class XDRThreatDetector(nn.Module):
    """
    Deep neural network for multi-class threat detection
    (Same architecture as training script)
    """

    def __init__(
        self,
        input_dim,
        hidden_dims=[512, 256, 128, 64],
        num_classes=2,
        dropout_rate=0.3,
    ):
        super(XDRThreatDetector, self).__init__()

        layers = []
        prev_dim = input_dim

        for i, dim in enumerate(hidden_dims):
            layers.extend(
                [
                    nn.Linear(prev_dim, dim),
                    nn.BatchNorm1d(dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                ]
            )
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
            encoder_layers.extend(
                [
                    nn.Linear(prev_dim, dim),
                    nn.BatchNorm1d(dim),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                ]
            )
            prev_dim = dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for dim in reversed(encoding_dims):
            decoder_layers.extend(
                [
                    nn.Linear(prev_dim, dim),
                    nn.BatchNorm1d(dim),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                ]
            )
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

    def __init__(
        self, input_dim, hidden_dim=256, num_layers=3, num_classes=2, dropout=0.3
    ):
        super(LSTMAttentionDetector, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            hidden_dim * 2, num_heads=8, dropout=dropout, batch_first=True
        )

        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
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
        # Fix path to models directory (relative to project root, not backend dir)
        if model_dir == "models":
            self.model_dir = Path(__file__).parent.parent.parent / "models"
        else:
            self.model_dir = Path(model_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Model instances
        self.threat_detector = None
        self.anomaly_detector = None
        self.lstm_detector = None
        self.lstm_autoencoder = None  # For sequential anomaly detection

        # Preprocessing
        self.scaler = None
        self.label_encoder = None

        # Model metadata
        self.metadata = {}
        self.is_loaded = False

        # Phase 2: Weighted Loss & Calibration
        self.temperature = 1.0  # Temperature for probability calibration
        self.per_class_thresholds = None  # Optimized thresholds per class
        self.class_weights = None  # Class weights for imbalanced data

        # Feature extraction (consistent with training)
        self.feature_columns = [
            "event_count_1h",
            "event_count_24h",
            "unique_ports",
            "failed_login_count",
            "session_duration_avg",
            "password_diversity",
            "username_diversity",
            "event_rate_per_minute",
            "time_of_day",
            "is_weekend",
            "unique_usernames",
            "password_length_avg",
            "command_diversity",
            "download_attempts",
            "upload_attempts",
        ]

        logger.info(f"Deep Learning Model Manager initialized on {self.device}")

    def set_temperature(self, temperature: float = 1.0):
        """
        Set temperature for probability calibration.

        Temperature scaling: p_i = softmax(z_i / T)
        - T = 1.0: No scaling (default)
        - T > 1.0: More uncertain probabilities
        - T < 1.0: More confident probabilities

        Args:
            temperature: Scaling factor
        """
        self.temperature = temperature
        logger.info(f"Temperature scaling set to {temperature:.4f}")

    def set_per_class_thresholds(self, thresholds: Dict[int, float]):
        """
        Set optimized decision thresholds per class.

        Instead of using 0.5 for all classes, use thresholds that maximize F1 score.

        Args:
            thresholds: Dict mapping class_id → threshold (0-1)
        """
        self.per_class_thresholds = thresholds
        logger.info(f"Per-class thresholds set: {thresholds}")

    def set_class_weights(self, weights: Dict[int, float]):
        """
        Set class weights for ensemble voting.

        Higher weights for minority/critical classes.

        Args:
            weights: Dict mapping class_id → weight
        """
        self.class_weights = weights
        logger.info(f"Class weights set: {weights}")

    def _apply_temperature_scaling(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply temperature scaling to logits."""
        import torch.nn.functional as F

        scaled_logits = logits / self.temperature
        return F.softmax(scaled_logits, dim=1)

    def _apply_per_class_thresholds(
        self, probabilities: torch.Tensor, predicted_class: int
    ) -> Tuple[int, float]:
        """
        Apply per-class thresholds to make final prediction.

        Args:
            probabilities: Class probabilities [batch, num_classes]
            predicted_class: Class with highest probability

        Returns:
            (final_class, confidence) tuple
        """
        if self.per_class_thresholds is None:
            # No custom thresholds, use standard argmax
            return predicted_class, probabilities[0, predicted_class].item()

        # Check if predicted class meets its threshold
        pred_prob = probabilities[0, predicted_class].item()
        threshold = self.per_class_thresholds.get(predicted_class, 0.5)

        if pred_prob >= threshold:
            return predicted_class, pred_prob
        else:
            # Predicted class doesn't meet threshold, check others
            for class_id in range(probabilities.shape[1]):
                class_prob = probabilities[0, class_id].item()
                class_threshold = self.per_class_thresholds.get(class_id, 0.5)
                if class_prob >= class_threshold:
                    return class_id, class_prob

            # No class meets threshold, return predicted with low confidence
            return predicted_class, pred_prob * 0.5

    def load_models(self, model_path: Optional[str] = None) -> Dict[str, bool]:
        """Load all available deep learning models"""
        results = {}

        try:
            # Load from specified path or default local models
            if model_path:
                # Custom model path
                model_dir = Path(model_path)
            else:
                # Local model directory
                model_dir = self.model_dir

            logger.info(f"Loading models from: {model_dir}")

            # Load metadata first
            metadata_path = model_dir / "model_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    self.metadata = json.load(f)
                logger.info(
                    f"Loaded metadata: {self.metadata.get('model_type', 'unknown')}"
                )

            # Load preprocessing
            scaler_path = model_dir / "scaler.pkl"
            if scaler_path.exists():
                loaded_scaler = joblib.load(scaler_path)
                # Validate scaler is not None and has required attributes
                if loaded_scaler is not None and hasattr(loaded_scaler, "transform"):
                    self.scaler = loaded_scaler
                    results["scaler"] = True
                    logger.info("Scaler loaded successfully")
                else:
                    logger.warning(
                        "Scaler file is corrupted or invalid - assuming pre-normalized data"
                    )
                    self.scaler = None
                    results["scaler"] = False
            else:
                logger.info("No scaler found - assuming pre-normalized data")
                self.scaler = None

            label_encoder_path = model_dir / "label_encoder.pkl"
            if label_encoder_path.exists():
                self.label_encoder = joblib.load(label_encoder_path)
                results["label_encoder"] = True
                logger.info("Label encoder loaded successfully")

            # Load deep learning models
            results.update(self._load_deep_models(model_dir))

            # Check if any models loaded successfully
            self.is_loaded = any(results.values())

            if self.is_loaded:
                logger.info(
                    f"Successfully loaded models: {[k for k, v in results.items() if v]}"
                )
            else:
                logger.warning("No deep learning models loaded successfully")

            return results

        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            return {"error": str(e)}

    def _load_deep_models(self, model_dir: Path) -> Dict[str, bool]:
        """Load PyTorch models"""
        results = {}

        # Load threat detector
        threat_detector_path = model_dir / "threat_detector.pth"
        if threat_detector_path.exists():
            try:
                # Get model parameters from metadata
                features = self.metadata.get("features", 79)
                num_classes = self.metadata.get("num_classes", 2)

                # Use same architecture as training (small model for single GPU)
                self.threat_detector = XDRThreatDetector(
                    input_dim=features,
                    hidden_dims=[256, 128, 64],  # Small model architecture
                    num_classes=num_classes,
                    dropout_rate=0.2,
                ).to(self.device)

                self.threat_detector.load_state_dict(
                    torch.load(
                        threat_detector_path,
                        map_location=self.device,
                        weights_only=True,
                    )
                )
                self.threat_detector.eval()

                results["threat_detector"] = True
                logger.info("Deep threat detector loaded successfully")

            except Exception as e:
                logger.error(f"Failed to load threat detector: {e}")
                results["threat_detector"] = False

        # Load anomaly detector
        anomaly_detector_path = model_dir / "anomaly_detector.pth"
        if anomaly_detector_path.exists():
            try:
                features = self.metadata.get("features", 79)

                # Use same architecture as training (small model)
                self.anomaly_detector = XDRAnomalyDetector(
                    input_dim=features,
                    encoding_dims=[128, 64],  # Small model architecture
                    latent_dim=16,
                ).to(self.device)

                self.anomaly_detector.load_state_dict(
                    torch.load(
                        anomaly_detector_path,
                        map_location=self.device,
                        weights_only=True,
                    )
                )
                self.anomaly_detector.eval()

                results["anomaly_detector"] = True
                logger.info("Deep anomaly detector loaded successfully")

            except Exception as e:
                logger.error(f"Failed to load anomaly detector: {e}")
                results["anomaly_detector"] = False

        # Load LSTM detector (if available from sequential training)
        lstm_detector_path = model_dir / "lstm_attention.pth"
        if lstm_detector_path.exists():
            try:
                features = self.metadata.get("features", 79)
                num_classes = self.metadata.get("num_classes", 2)

                self.lstm_detector = LSTMAttentionDetector(
                    input_dim=features, num_classes=num_classes
                ).to(self.device)

                self.lstm_detector.load_state_dict(
                    torch.load(
                        lstm_detector_path, map_location=self.device, weights_only=True
                    )
                )
                self.lstm_detector.eval()

                results["lstm_detector"] = True
                logger.info("LSTM attention detector loaded successfully")

            except Exception as e:
                logger.error(f"Failed to load LSTM detector: {e}")
                results["lstm_detector"] = False

        # Load LSTM autoencoder (for sequential anomaly detection)
        lstm_autoencoder_path = model_dir / "lstm_autoencoder.pth"
        if lstm_autoencoder_path.exists():
            try:
                # LSTM autoencoder uses 15 base features (from ml_engine.py)
                self.lstm_autoencoder = LSTMAutoencoder(
                    input_size=15, hidden_size=64, num_layers=2, dropout=0.2
                ).to(self.device)

                self.lstm_autoencoder.load_state_dict(
                    torch.load(
                        lstm_autoencoder_path,
                        map_location=self.device,
                        weights_only=True,
                    )
                )
                self.lstm_autoencoder.eval()

                results["lstm_autoencoder"] = True
                logger.info(
                    "✅ LSTM autoencoder loaded successfully for sequential anomaly detection"
                )

            except Exception as e:
                logger.error(f"Failed to load LSTM autoencoder: {e}")
                results["lstm_autoencoder"] = False

        return results

    def _extract_features(self, src_ip: str, events: List[Event]) -> Dict[str, float]:
        """Extract comprehensive features for deep learning models (79 features)"""
        if not events:
            # Return 79 zero features for model compatibility
            return {f"feature_{i}": 0.0 for i in range(79)}

        # Extract comprehensive 79-feature set
        features = {}

        # Time-based features (timezone-aware)
        now = datetime.now(timezone.utc)
        events_1h = [
            e
            for e in events
            if (
                now
                - (e.ts.replace(tzinfo=timezone.utc) if e.ts.tzinfo is None else e.ts)
            ).total_seconds()
            <= 3600
        ]
        events_24h = [
            e
            for e in events
            if (
                now
                - (e.ts.replace(tzinfo=timezone.utc) if e.ts.tzinfo is None else e.ts)
            ).total_seconds()
            <= 86400
        ]

        # Basic temporal features (0-9)
        features["event_count_1h"] = len(events_1h)
        features["event_count_24h"] = len(events_24h)
        features["event_rate_per_minute"] = len(events_1h) / 60.0
        features["time_span_hours"] = (
            (events[-1].ts - events[0].ts).total_seconds() / 3600.0
            if len(events) > 1
            else 0.0
        )
        features["time_of_day"] = now.hour / 24.0
        features["is_weekend"] = 1.0 if now.weekday() >= 5 else 0.0
        features["burst_intensity"] = (
            len(events)
            / max((events[-1].ts - events[0].ts).total_seconds() / 60.0, 1.0)
            if len(events) > 1
            else 0.0
        )
        features["session_persistence"] = len(
            set(e.raw.get("session", "") for e in events if e.raw)
        ) / max(len(events), 1)
        features["temporal_clustering"] = self._calculate_temporal_clustering(events)
        features["peak_activity_score"] = self._calculate_peak_activity(events_1h)

        # Port diversity
        unique_ports = len(set(e.dst_port for e in events if e.dst_port))
        features["unique_ports"] = unique_ports

        # Failed login analysis
        failed_logins = [e for e in events if e.eventid == "cowrie.login.failed"]
        features["failed_login_count"] = len(failed_logins)

        # Session analysis
        if events:
            time_span = (events[0].ts - events[-1].ts).total_seconds()
            features["session_duration_avg"] = time_span / max(len(events), 1)
            features["event_rate_per_minute"] = len(events) / max(time_span / 60, 1)
        else:
            features["session_duration_avg"] = 0
            features["event_rate_per_minute"] = 0

        # Credential analysis
        usernames = set()
        passwords = set()
        password_lengths = []

        for event in failed_logins:
            if hasattr(event, "raw") and event.raw:
                raw_data = event.raw if isinstance(event.raw, dict) else {}
                if "username" in raw_data:
                    usernames.add(raw_data["username"])
                if "password" in raw_data:
                    passwords.add(raw_data["password"])
                    password_lengths.append(len(str(raw_data["password"])))

        features["unique_usernames"] = len(usernames)
        features["password_diversity"] = len(passwords)
        features["username_diversity"] = len(usernames)
        features["password_length_avg"] = (
            np.mean(password_lengths) if password_lengths else 0
        )

        # Command analysis
        commands = set()
        download_count = 0
        upload_count = 0

        for event in events:
            if event.eventid == "cowrie.command.input":
                if hasattr(event, "raw") and event.raw:
                    raw_data = event.raw if isinstance(event.raw, dict) else {}
                    if "input" in raw_data:
                        commands.add(
                            raw_data["input"].split()[0] if raw_data["input"] else ""
                        )
            elif event.eventid in [
                "cowrie.session.file_download",
                "cowrie.session.file_upload",
            ]:
                if "download" in event.eventid:
                    download_count += 1
                else:
                    upload_count += 1

        features["command_diversity"] = len(commands)
        features["download_attempts"] = download_count
        features["upload_attempts"] = upload_count

        # Time-based features
        if events:
            avg_hour = np.mean([e.ts.hour for e in events])
            features["time_of_day"] = avg_hour / 24.0
            features["is_weekend"] = float(any(e.ts.weekday() >= 5 for e in events))
        else:
            features["time_of_day"] = 0.5
            features["is_weekend"] = 0.0

        # Ensure all features are present
        for col in self.feature_columns:
            if col not in features:
                features[col] = 0.0

        # Authentication features (10-19)
        failed_logins = len(
            [e for e in events if "failed" in e.eventid or "login.failed" in e.eventid]
        )
        features["failed_login_count"] = failed_logins
        features["failed_login_rate"] = failed_logins / max(len(events), 1)
        features["unique_usernames"] = len(
            set(
                e.raw.get("username", "")
                for e in events
                if e.raw and e.raw.get("username")
            )
        )
        features["unique_passwords"] = len(
            set(
                e.raw.get("password", "")
                for e in events
                if e.raw and e.raw.get("password")
            )
        )
        features["password_diversity"] = features["unique_passwords"] / max(
            failed_logins, 1
        )
        features["username_diversity"] = features["unique_usernames"] / max(
            failed_logins, 1
        )
        features["auth_success_rate"] = len(
            [e for e in events if "success" in e.eventid]
        ) / max(len(events), 1)
        features[
            "credential_stuffing_score"
        ] = self._calculate_credential_stuffing_score(events)
        features["password_spray_score"] = self._calculate_password_spray_score(events)
        features["brute_force_intensity"] = (
            failed_logins
            / max((events[-1].ts - events[0].ts).total_seconds() / 60.0, 1.0)
            if len(events) > 1
            else 0.0
        )

        # Protocol and network features (20-39)
        features["protocol_diversity"] = len(
            set(e.eventid.split(".")[0] for e in events if "." in e.eventid)
        )
        features["port_scanning_score"] = (
            len(set(e.dst_port for e in events if e.dst_port)) / 10.0
        )  # Normalize
        features["service_enumeration"] = self._calculate_service_enumeration(events)
        features["connection_persistence"] = len(
            set(e.raw.get("session", "") for e in events if e.raw)
        ) / max(len(events), 1)
        features["network_footprint"] = self._calculate_network_footprint(events)
        features["lateral_movement_score"] = self._calculate_lateral_movement(events)
        features["reconnaissance_score"] = self._calculate_reconnaissance_score(events)
        features["vulnerability_scanning"] = self._calculate_vuln_scanning(events)
        features["command_injection_score"] = self._calculate_command_injection(events)
        features["privilege_escalation_score"] = self._calculate_privilege_escalation(
            events
        )
        features["data_exfiltration_score"] = self._calculate_data_exfiltration(events)
        features["persistence_score"] = self._calculate_persistence_indicators(events)
        features["evasion_score"] = self._calculate_evasion_techniques(events)
        features["attack_sophistication"] = self._calculate_attack_sophistication(
            events
        )
        features["payload_complexity"] = self._calculate_payload_complexity(events)
        features["encryption_usage"] = self._calculate_encryption_indicators(events)
        features["anonymization_score"] = self._calculate_anonymization_score(events)
        features["multi_vector_score"] = self._calculate_multi_vector_attack(events)
        features["timing_attack_score"] = self._calculate_timing_attack(events)
        features["social_engineering_score"] = self._calculate_social_engineering(
            events
        )

        # Statistical features (40-59)
        event_intervals = []
        if len(events) > 1:
            for i in range(1, len(events)):
                interval = (events[i].ts - events[i - 1].ts).total_seconds()
                event_intervals.append(interval)

        if event_intervals:
            features["interval_mean"] = np.mean(event_intervals)
            features["interval_std"] = np.std(event_intervals)
            features["interval_median"] = np.median(event_intervals)
            features["interval_min"] = np.min(event_intervals)
            features["interval_max"] = np.max(event_intervals)
            features["interval_range"] = (
                features["interval_max"] - features["interval_min"]
            )
            features["interval_cv"] = features["interval_std"] / max(
                features["interval_mean"], 0.001
            )  # Coefficient of variation
        else:
            for key in [
                "interval_mean",
                "interval_std",
                "interval_median",
                "interval_min",
                "interval_max",
                "interval_range",
                "interval_cv",
            ]:
                features[key] = 0.0

        # Message and payload analysis (60-78)
        messages = [e.message for e in events if e.message]
        features["message_length_avg"] = (
            np.mean([len(m) for m in messages]) if messages else 0.0
        )
        features["message_length_std"] = (
            np.std([len(m) for m in messages]) if messages else 0.0
        )
        features["message_complexity"] = self._calculate_message_complexity(messages)
        features["payload_entropy"] = self._calculate_payload_entropy(events)
        features["command_diversity"] = self._calculate_command_diversity(events)
        features["sql_injection_score"] = self._calculate_sql_injection_score(events)
        features["xss_score"] = self._calculate_xss_score(events)
        features["path_traversal_score"] = self._calculate_path_traversal_score(events)
        features["remote_code_exec_score"] = self._calculate_rce_score(events)
        features["malware_indicators"] = self._calculate_malware_indicators(events)
        features["bot_behavior_score"] = self._calculate_bot_behavior(events)
        features["human_behavior_score"] = self._calculate_human_behavior(events)
        features["automation_score"] = self._calculate_automation_indicators(events)
        features["geolocation_anomaly"] = self._calculate_geolocation_anomaly(src_ip)
        features["reputation_score"] = self._calculate_reputation_score(src_ip)
        features["threat_intel_score"] = self._calculate_threat_intel_score(src_ip)
        features["historical_activity"] = self._calculate_historical_activity(src_ip)
        features["attack_campaign_score"] = self._calculate_campaign_indicators(events)
        features["attack_vector_diversity"] = self._calculate_attack_vector_diversity(
            events
        )

        # Ensure exactly 79 features
        feature_list = list(features.values())
        while len(feature_list) < 79:
            feature_list.append(0.0)
        feature_list = feature_list[:79]

        # Convert back to dict with standard names
        return {f"feature_{i}": feature_list[i] for i in range(79)}

    def _extract_lstm_features(
        self, src_ip: str, events: List[Event]
    ) -> List[List[float]]:
        """
        Extract 15 base features for LSTM autoencoder (matching ml_engine.py training).

        Returns sequence of shape (seq_len=10, features=15)
        """
        if not events or len(events) < 10:
            # Return zero sequence if not enough events
            return [[0.0] * 15 for _ in range(10)]

        # Take last 10 events for sequence
        sequence_events = events[-10:]
        sequence_features = []

        for i, event in enumerate(sequence_events):
            # Extract 15 base features (matching LSTM training)
            features = []

            # 1. Event count in window
            features.append(min(len(events) / 100.0, 1.0))

            # 2-3. Port features
            features.append((event.dst_port or 0) / 65535.0)
            features.append((event.src_port or 0) / 65535.0)

            # 4-6. Credential features
            username = (
                event.raw.get("username", "")
                if hasattr(event, "raw") and event.raw
                else ""
            )
            password = (
                event.raw.get("password", "")
                if hasattr(event, "raw") and event.raw
                else ""
            )
            features.append(len(username) / 50.0)  # Username length normalized
            features.append(len(password) / 50.0)  # Password length normalized
            features.append(1.0 if event.eventid == "cowrie.login.failed" else 0.0)

            # 7-8. Temporal features
            features.append(event.ts.hour / 24.0)  # Time of day
            features.append(1.0 if event.ts.weekday() >= 5 else 0.0)  # Weekend

            # 9-10. Command features
            command = (
                event.raw.get("input", "")
                if hasattr(event, "raw") and event.raw
                else ""
            )
            features.append(len(command) / 100.0)  # Command length
            features.append(
                1.0
                if any(cmd in command for cmd in ["wget", "curl", "download"])
                else 0.0
            )

            # 11-13. Session features
            features.append(1.0 if event.eventid == "cowrie.session.closed" else 0.0)
            features.append(1.0 if event.eventid == "cowrie.command.input" else 0.0)
            features.append(1.0 if event.eventid == "cowrie.login.success" else 0.0)

            # 14-15. Sequence position features
            features.append(i / 10.0)  # Position in sequence
            features.append(
                1.0 if i == len(sequence_events) - 1 else 0.0
            )  # Is last event

            sequence_features.append(features)

        return sequence_features

    def _calculate_temporal_clustering(self, events) -> float:
        """Calculate temporal clustering score"""
        if len(events) < 2:
            return 0.0
        # Simple clustering based on time gaps
        intervals = [
            (events[i].ts - events[i - 1].ts).total_seconds()
            for i in range(1, len(events))
        ]
        return (
            1.0 - (np.std(intervals) / max(np.mean(intervals), 1.0))
            if intervals
            else 0.0
        )

    def _calculate_peak_activity(self, events) -> float:
        """Calculate peak activity score"""
        return min(len(events) / 10.0, 1.0)  # Normalize to 0-1

    def _calculate_credential_stuffing_score(self, events) -> float:
        """Calculate credential stuffing indicators"""
        usernames = set()
        passwords = set()
        for e in events:
            if e.raw and isinstance(e.raw, dict):
                if "username" in e.raw:
                    usernames.add(e.raw["username"])
                if "password" in e.raw:
                    passwords.add(e.raw["password"])

        # High username diversity + low password diversity = credential stuffing
        username_diversity = len(usernames) / max(len(events), 1)
        password_diversity = len(passwords) / max(len(events), 1)

        if username_diversity > 0.5 and password_diversity < 0.3:
            return min(username_diversity * 2.0, 1.0)
        return 0.0

    def _calculate_password_spray_score(self, events) -> float:
        """Calculate password spray indicators"""
        usernames = set()
        passwords = set()
        for e in events:
            if e.raw and isinstance(e.raw, dict):
                if "username" in e.raw:
                    usernames.add(e.raw["username"])
                if "password" in e.raw:
                    passwords.add(e.raw["password"])

        # Low username diversity + high password diversity = password spray
        username_diversity = len(usernames) / max(len(events), 1)
        password_diversity = len(passwords) / max(len(events), 1)

        if username_diversity < 0.3 and password_diversity > 0.5:
            return min(password_diversity * 2.0, 1.0)
        return 0.0

    # Add placeholder methods for all the new features
    def _calculate_service_enumeration(self, events) -> float:
        """Service enumeration score"""
        return min(len(set(e.dst_port for e in events if e.dst_port)) / 20.0, 1.0)

    def _calculate_network_footprint(self, events) -> float:
        """Network footprint analysis"""
        return min(len(events) / 50.0, 1.0)

    def _calculate_lateral_movement(self, events) -> float:
        """Lateral movement indicators"""
        return 0.1  # Placeholder

    def _calculate_reconnaissance_score(self, events) -> float:
        """Reconnaissance activity score"""
        recon_events = [
            e
            for e in events
            if any(
                keyword in e.message.lower() for keyword in ["scan", "probe", "enum"]
            )
            if e.message
        ]
        return min(len(recon_events) / max(len(events), 1), 1.0)

    def _calculate_vuln_scanning(self, events) -> float:
        """Vulnerability scanning indicators"""
        return 0.1  # Placeholder

    def _calculate_command_injection(self, events) -> float:
        """Command injection indicators"""
        return 0.1  # Placeholder

    def _calculate_privilege_escalation(self, events) -> float:
        """Privilege escalation indicators"""
        return 0.1  # Placeholder

    def _calculate_data_exfiltration(self, events) -> float:
        """Data exfiltration indicators"""
        return 0.1  # Placeholder

    def _calculate_persistence_indicators(self, events) -> float:
        """Persistence mechanism indicators"""
        return 0.1  # Placeholder

    def _calculate_evasion_techniques(self, events) -> float:
        """Evasion technique indicators"""
        return 0.1  # Placeholder

    def _calculate_attack_sophistication(self, events) -> float:
        """Overall attack sophistication"""
        return min(len(events) / 20.0, 1.0)

    def _calculate_payload_complexity(self, events) -> float:
        """Payload complexity analysis"""
        return 0.1  # Placeholder

    def _calculate_encryption_indicators(self, events) -> float:
        """Encryption usage indicators"""
        return 0.1  # Placeholder

    def _calculate_anonymization_score(self, events) -> float:
        """Anonymization technique indicators"""
        return 0.1  # Placeholder

    def _calculate_multi_vector_attack(self, events) -> float:
        """Multi-vector attack indicators"""
        protocols = set(e.eventid.split(".")[0] for e in events if "." in e.eventid)
        return min(len(protocols) / 5.0, 1.0)

    def _calculate_timing_attack(self, events) -> float:
        """Timing attack indicators"""
        return 0.1  # Placeholder

    def _calculate_social_engineering(self, events) -> float:
        """Social engineering indicators"""
        return 0.1  # Placeholder

    def _calculate_message_complexity(self, messages) -> float:
        """Message complexity score"""
        if not messages:
            return 0.0
        import re

        complexity_indicators = 0
        for msg in messages:
            if re.search(r'[<>&"\']', msg):  # HTML/script chars
                complexity_indicators += 1
            if re.search(r"[;|&]", msg):  # Command separators
                complexity_indicators += 1
        return min(complexity_indicators / max(len(messages), 1), 1.0)

    def _calculate_payload_entropy(self, events) -> float:
        """Calculate payload entropy"""
        return 0.1  # Placeholder

    def _calculate_command_diversity(self, events) -> float:
        """Command diversity score"""
        commands = set()
        for e in events:
            if e.raw and isinstance(e.raw, dict):
                if "command" in e.raw:
                    commands.add(e.raw["command"])
        return min(len(commands) / 10.0, 1.0)

    def _calculate_sql_injection_score(self, events) -> float:
        """SQL injection indicators"""
        return 0.1  # Placeholder

    def _calculate_xss_score(self, events) -> float:
        """XSS indicators"""
        return 0.1  # Placeholder

    def _calculate_path_traversal_score(self, events) -> float:
        """Path traversal indicators"""
        return 0.1  # Placeholder

    def _calculate_rce_score(self, events) -> float:
        """Remote code execution indicators"""
        return 0.1  # Placeholder

    def _calculate_malware_indicators(self, events) -> float:
        """Malware behavior indicators"""
        return 0.1  # Placeholder

    def _calculate_bot_behavior(self, events) -> float:
        """Bot behavior indicators"""
        # Regular intervals, consistent patterns
        if len(events) < 3:
            return 0.0
        intervals = [
            (events[i].ts - events[i - 1].ts).total_seconds()
            for i in range(1, len(events))
        ]
        cv = np.std(intervals) / max(np.mean(intervals), 0.001)
        return max(0.0, 1.0 - cv)  # Low coefficient of variation = bot-like

    def _calculate_human_behavior(self, events) -> float:
        """Human behavior indicators"""
        return 1.0 - self._calculate_bot_behavior(events)

    def _calculate_automation_indicators(self, events) -> float:
        """Automation indicators"""
        return self._calculate_bot_behavior(events)

    def _calculate_geolocation_anomaly(self, src_ip) -> float:
        """Geolocation anomaly score"""
        return 0.1  # Placeholder

    def _calculate_reputation_score(self, src_ip) -> float:
        """IP reputation score"""
        return 0.1  # Placeholder

    def _calculate_threat_intel_score(self, src_ip) -> float:
        """Threat intelligence score"""
        return 0.1  # Placeholder

    def _calculate_historical_activity(self, src_ip) -> float:
        """Historical activity pattern"""
        return 0.1  # Placeholder

    def _calculate_campaign_indicators(self, events) -> float:
        """Attack campaign indicators"""
        return 0.1  # Placeholder

    def _calculate_attack_vector_diversity(self, events) -> float:
        """Attack vector diversity"""
        vectors = set(e.eventid for e in events)
        return min(len(vectors) / 10.0, 1.0)

        return features

    async def calculate_threat_score(
        self, src_ip: str, events: List[Event]
    ) -> Dict[str, float]:
        """Calculate threat scores using all available deep learning models"""
        if not self.is_loaded or not events:
            return {"ensemble_score": 0.0, "confidence": 0.0}

        try:
            # Extract and prepare features
            features = self._extract_features(src_ip, events)
            feature_vector = np.array([list(features.values())]).astype(np.float32)

            # Scale features
            if self.scaler:
                feature_vector = self.scaler.transform(feature_vector)

            # Convert to tensor
            input_tensor = torch.tensor(feature_vector, dtype=torch.float32).to(
                self.device
            )

            scores = {}

            # Deep threat detection
            if self.threat_detector:
                with torch.no_grad():
                    logits = self.threat_detector(input_tensor)

                    # Phase 2: Apply temperature scaling for calibration
                    if self.temperature != 1.0:
                        probabilities = self._apply_temperature_scaling(logits)
                    else:
                        probabilities = torch.softmax(logits, dim=1)

                    predicted_class = torch.argmax(probabilities, dim=1).item()

                    # Phase 2: Apply per-class thresholds if configured
                    if self.per_class_thresholds is not None:
                        (
                            predicted_class,
                            pred_confidence,
                        ) = self._apply_per_class_thresholds(
                            probabilities, predicted_class
                        )
                    else:
                        pred_confidence = probabilities[0, predicted_class].item()

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
                            6: "Advanced Persistent Threat",
                        }
                        scores["attack_type"] = attack_classes.get(
                            predicted_class, "Unknown"
                        )
                        scores[
                            "attack_confidence"
                        ] = pred_confidence  # Use calibrated confidence

                    else:  # Binary model fallback
                        threat_prob = (
                            probabilities[0, 1].item()
                            if probabilities.shape[1] > 1
                            else probabilities[0, 0].item()
                        )
                        scores["attack_type"] = (
                            "Threat" if predicted_class == 1 else "Normal"
                        )
                        scores[
                            "attack_confidence"
                        ] = pred_confidence  # Use calibrated confidence

                    scores["threat_detector"] = threat_prob

            # Anomaly detection
            if self.anomaly_detector:
                with torch.no_grad():
                    reconstruction_error = (
                        self.anomaly_detector.get_reconstruction_error(input_tensor)
                    )
                    # Normalize reconstruction error to 0-1 scale
                    anomaly_score = torch.clamp(
                        reconstruction_error / 10.0, 0, 1
                    ).item()
                    scores["anomaly_detector"] = anomaly_score

            # Sequential detection with LSTM attention (if available and enough events)
            if self.lstm_detector and len(events) >= 10:
                # This would require sequence preparation - simplified for now
                scores["lstm_detector"] = 0.5  # Placeholder

            # Sequential anomaly detection with LSTM autoencoder
            if self.lstm_autoencoder and len(events) >= 10:
                try:
                    # Extract 15 base features for LSTM (matching training)
                    lstm_features = self._extract_lstm_features(
                        src_ip, events[-10:]
                    )  # Last 10 events

                    # Create sequence: shape (batch=1, seq_len=10, features=15)
                    sequence = torch.tensor([lstm_features], dtype=torch.float32).to(
                        self.device
                    )

                    with torch.no_grad():
                        # Get reconstruction from autoencoder
                        reconstructed = self.lstm_autoencoder(sequence)

                        # Calculate reconstruction error (MSE)
                        mse = torch.mean((sequence - reconstructed) ** 2).item()

                        # Normalize to 0-1 scale (threshold from training: ~0.05)
                        # Higher reconstruction error = more anomalous
                        lstm_anomaly_score = min(
                            mse / 0.1, 1.0
                        )  # Scale by expected threshold

                        scores["lstm_autoencoder"] = lstm_anomaly_score

                except Exception as e:
                    logger.warning(f"LSTM autoencoder scoring failed: {e}")
                    # Don't add score if it fails

            # Calculate ensemble score
            if scores:
                weights = {
                    "threat_detector": 0.35,  # Deep classification
                    "anomaly_detector": 0.30,  # Autoencoder anomaly
                    "lstm_detector": 0.15,  # LSTM attention (if available)
                    "lstm_autoencoder": 0.20,  # Sequential anomaly
                }

                weighted_sum = 0.0
                total_weight = 0.0

                for model_name, score in scores.items():
                    weight = weights.get(model_name, 0.1)
                    weighted_sum += score * weight
                    total_weight += weight

                ensemble_score = (
                    weighted_sum / total_weight if total_weight > 0 else 0.0
                )

                # Calculate confidence based on model agreement
                score_values = list(scores.values())
                confidence = (
                    1.0 - np.var(score_values) if len(score_values) > 1 else 0.8
                )

                return {
                    "ensemble_score": min(ensemble_score, 1.0),
                    "confidence": min(confidence, 1.0),
                    "individual_scores": scores,
                    "model_count": len(scores),
                }

            return {"ensemble_score": 0.0, "confidence": 0.0}

        except Exception as e:
            logger.error(f"Deep learning threat scoring failed: {e}")
            return {"ensemble_score": 0.0, "confidence": 0.0, "error": str(e)}

    def get_model_status(self) -> Dict[str, Any]:
        """Get status of all loaded models"""
        return {
            "deep_learning_loaded": self.is_loaded,
            "device": str(self.device),
            "models_available": {
                "threat_detector": self.threat_detector is not None,
                "anomaly_detector": self.anomaly_detector is not None,
                "lstm_detector": self.lstm_detector is not None,
            },
            "preprocessing_loaded": {
                "scaler": self.scaler is not None,
                "label_encoder": self.label_encoder is not None,
            },
            "metadata": self.metadata,
            "gpu_available": torch.cuda.is_available(),
            "feature_count": len(self.feature_columns),
        }


# Global instance
deep_learning_manager = DeepLearningModelManager()

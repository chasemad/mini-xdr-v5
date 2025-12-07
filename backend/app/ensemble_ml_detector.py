"""
Ensemble ML Detector - Network Models + Windows Specialist
Combines existing 4M-trained network models with new Windows specialist
No need to retrain full dataset - modular approach!
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class EnsembleMLDetector:
    """
    Ensemble detector combining:
    1. Network attack models (DDoS, web, brute force, malware, APT)
    2. Windows specialist model (Kerberos, lateral movement, credential theft, etc.)

    Detection strategy:
    - Run event through both models
    - Use confidence-based voting
    - Windows specialist overrides for Windows-specific attacks
    """

    def __init__(
        self,
        network_model_dir="./models/local_trained_enhanced",
        windows_model_dir="./models/windows_specialist_13class",
        legacy_windows_model_dir="./models/windows_specialist",
    ):
        self.network_model_dir = Path(network_model_dir)
        self.windows_model_dir = Path(windows_model_dir)

        # Setup device
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Load models
        self.network_model = self._load_network_model()
        self.windows_specialist = self._load_windows_specialist()

        # Class mappings
        self.network_classes = {
            0: "normal",
            1: "ddos",
            2: "reconnaissance",
            3: "brute_force",
            4: "web_attack",
            5: "malware",
            6: "apt",
        }

        self.windows_classes = {
            0: "normal",
            1: "ddos",
            2: "reconnaissance",
            3: "brute_force",
            4: "web_attack",
            5: "malware",
            6: "apt",
            7: "kerberos_attack",
            8: "lateral_movement",
            9: "credential_theft",
            10: "privilege_escalation",
            11: "data_exfiltration",
            12: "insider_threat",
        }

        logger.info("✅ Ensemble detector initialized")
        logger.info(f"   Network models loaded: {self.network_model is not None}")
        logger.info(
            f"   Windows specialist loaded: {self.windows_specialist is not None}"
        )

    def _load_network_model(self):
        """Load existing network attack models"""
        try:
            general_model_path = (
                self.network_model_dir / "general" / "threat_detector.pth"
            )

            if not general_model_path.exists():
                logger.warning(f"Network model not found at {general_model_path}")
                return None

            # Load model checkpoint
            checkpoint = torch.load(general_model_path, map_location=self.device)

            # Recreate ThreatDetector architecture
            # Standard deep learning architecture
            model = self._create_threat_detector_model(
                input_dim=79,
                hidden_dims=[512, 256, 128, 64],
                num_classes=7,
                dropout_rate=0.3,
                use_attention=True,
            )

            # Handle both checkpoint formats: direct state_dict or wrapped
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                # Direct state_dict
                model.load_state_dict(checkpoint)
            model.to(self.device)
            model.eval()

            # Load scaler
            scaler_path = self.network_model_dir / "general" / "scaler.pkl"
            if scaler_path.exists():
                self.network_scaler = joblib.load(scaler_path)
            else:
                self.network_scaler = joblib.load(
                    "./models/scaler.pkl"
                )

            logger.info("✅ Loaded network model")
            return model

        except Exception as e:
            logger.error(f"Failed to load network model: {e}")
            return None

    def _create_threat_detector_model(
        self, input_dim, hidden_dims, num_classes, dropout_rate, use_attention
    ):
        """Create ThreatDetector model architecture (matches training script)"""
        import torch.nn as nn
        import torch.nn.functional as F

        class AttentionLayer(nn.Module):
            def __init__(self, input_dim, attention_dim):
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
                attention_weights = torch.softmax(
                    torch.matmul(q, k.transpose(-2, -1)) / (self.attention_dim**0.5),
                    dim=-1,
                )
                attended = torch.matmul(attention_weights, v).squeeze(1)
                output = self.output(attended)
                output = self.dropout(output)
                return output + x

        class UncertaintyBlock(nn.Module):
            def __init__(self, in_dim, out_dim, dropout_rate):
                super().__init__()
                self.linear = nn.Linear(in_dim, out_dim)
                self.dropout = nn.Dropout(dropout_rate)
                self.batch_norm = nn.BatchNorm1d(out_dim)

            def forward(self, x):
                x = self.linear(x)
                x = self.batch_norm(x)
                x = F.relu(x)
                x = self.dropout(x)
                return x

        class ThreatDetector(nn.Module):
            def __init__(
                self, input_dim, hidden_dims, num_classes, dropout_rate, use_attention
            ):
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

                self.skip_connections = nn.ModuleList(
                    [
                        nn.Linear(input_dim, hidden_dims[-1]),
                        nn.Linear(hidden_dims[0], hidden_dims[-1]),
                    ]
                )

                self.classifier = nn.Linear(prev_dim, num_classes)
                self.uncertainty_head = nn.Linear(prev_dim, 1)
                self.mc_dropout = nn.Dropout(dropout_rate)

            def forward(self, x, return_features=False):
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
                uncertainty = torch.sigmoid(
                    self.uncertainty_head(self.mc_dropout(features))
                )

                if return_features:
                    return logits, uncertainty, features

                return logits, uncertainty

        return ThreatDetector(
            input_dim, hidden_dims, num_classes, dropout_rate, use_attention
        )

    def _load_windows_specialist(self):
        """Load Windows 13-class specialist model"""
        try:
            model_path = self.windows_model_dir / "windows_13class_specialist.pth"
            scaler_path = self.windows_model_dir / "windows_13class_scaler.pkl"

            if not model_path.exists():
                logger.warning(f"Windows specialist not found at {model_path}")
                return None

            # Load scaler
            import pickle

            with open(scaler_path, "rb") as f:
                self.windows_scaler = pickle.load(f)

            # Recreate Windows 13-class specialist architecture
            # Windows 13-class specialist model
            model = Windows13ClassSpecialist(input_dim=79, num_classes=13)
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.to(self.device)
            model.eval()

            logger.info("✅ Loaded Windows 13-class specialist model")
            return model

        except Exception as e:
            logger.error(f"Failed to load Windows specialist: {e}")
            logger.warning("Attempting to load legacy 7-class model...")
            return self._load_legacy_windows_specialist()

    def _load_legacy_windows_specialist(self):
        """Load legacy 7-class Windows model as fallback"""
        try:
            legacy_path = Path(
                "./models/windows_specialist/windows_specialist.pth"
            )
            if not legacy_path.exists():
                return None

            checkpoint = torch.load(legacy_path, map_location=self.device)
            # Windows specialist model
            model = WindowsSpecialistModel(
                input_dim=79, hidden_dims=[256, 128, 64], num_classes=7
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(self.device)
            model.eval()

            self.windows_scaler = checkpoint["scaler"]
            logger.info("✅ Loaded legacy 7-class Windows specialist model")

            # Update class mapping for legacy model
            self.windows_classes = {
                0: "normal_windows",
                1: "kerberos_attack",
                2: "lateral_movement",
                3: "credential_theft",
                4: "privilege_escalation",
                5: "data_exfiltration",
                6: "insider_threat",
            }
            return model
        except Exception as e:
            logger.error(f"Failed to load legacy Windows specialist: {e}")
            return None

    async def detect_threat(self, event_features: np.ndarray) -> Dict:
        """
        Ensemble detection using both models

        Args:
            event_features: 79-dimensional feature vector

        Returns:
            {
                'threat_type': str,
                'confidence': float,
                'network_prediction': {...},
                'windows_prediction': {...},
                'ensemble_decision': str
            }
        """

        results = {
            "ensemble_decision": "normal",
            "confidence": 0.0,
            "threat_type": "normal",
            "network_prediction": None,
            "windows_prediction": None,
            "model_used": "ensemble",
        }

        # Prepare input
        features = np.array(event_features).reshape(1, -1).astype(np.float32)

        # Run through network model
        if self.network_model is not None:
            try:
                features_scaled = self.network_scaler.transform(features)
                features_tensor = torch.tensor(features_scaled, dtype=torch.float32).to(
                    self.device
                )

                with torch.no_grad():
                    network_output = self.network_model(features_tensor)
                    network_probs = F.softmax(network_output, dim=1)[0].cpu().numpy()
                    network_class = int(np.argmax(network_probs))
                    network_conf = float(network_probs[network_class])

                results["network_prediction"] = {
                    "class": network_class,
                    "threat_type": self.network_classes[network_class],
                    "confidence": network_conf,
                    "all_probabilities": network_probs.tolist(),
                }
            except Exception as e:
                logger.error(f"Network model error: {e}")

        # Run through Windows specialist
        if self.windows_specialist is not None:
            try:
                features_scaled = self.windows_scaler.transform(features)
                features_tensor = torch.tensor(features_scaled, dtype=torch.float32).to(
                    self.device
                )

                with torch.no_grad():
                    windows_output = self.windows_specialist(features_tensor)
                    windows_probs = F.softmax(windows_output, dim=1)[0].cpu().numpy()
                    windows_class = int(np.argmax(windows_probs))
                    windows_conf = float(windows_probs[windows_class])

                results["windows_prediction"] = {
                    "class": windows_class,
                    "threat_type": self.windows_classes[windows_class],
                    "confidence": windows_conf,
                    "all_probabilities": windows_probs.tolist(),
                }
            except Exception as e:
                logger.error(f"Windows specialist error: {e}")

        # Ensemble decision
        if results["network_prediction"] and results["windows_prediction"]:
            # Windows specialist takes priority for Windows attacks
            if (
                results["windows_prediction"]["class"] > 0
                and results["windows_prediction"]["confidence"] > 0.7
            ):
                results["ensemble_decision"] = "windows_attack"
                results["threat_type"] = results["windows_prediction"]["threat_type"]
                results["confidence"] = results["windows_prediction"]["confidence"]
                results["model_used"] = "windows_specialist"

            # Network model for network attacks
            elif (
                results["network_prediction"]["class"] > 0
                and results["network_prediction"]["confidence"] > 0.7
            ):
                results["ensemble_decision"] = "network_attack"
                results["threat_type"] = results["network_prediction"]["threat_type"]
                results["confidence"] = results["network_prediction"]["confidence"]
                results["model_used"] = "network_model"

            # Both detect as normal
            else:
                results["ensemble_decision"] = "normal"
                results["threat_type"] = "normal"
                results["confidence"] = min(
                    results["network_prediction"]["confidence"],
                    results["windows_prediction"]["confidence"],
                )
                results["model_used"] = "ensemble"

        elif results["network_prediction"]:
            # Only network model available
            if results["network_prediction"]["class"] > 0:
                results["ensemble_decision"] = "network_attack"
                results["threat_type"] = results["network_prediction"]["threat_type"]
                results["confidence"] = results["network_prediction"]["confidence"]
                results["model_used"] = "network_model"

        elif results["windows_prediction"]:
            # Only Windows model available
            if results["windows_prediction"]["class"] > 0:
                results["ensemble_decision"] = "windows_attack"
                results["threat_type"] = results["windows_prediction"]["threat_type"]
                results["confidence"] = results["windows_prediction"]["confidence"]
                results["model_used"] = "windows_specialist"

        return results

    def get_model_info(self) -> Dict:
        """Get information about loaded models"""
        return {
            "network_model": {
                "loaded": self.network_model is not None,
                "classes": list(self.network_classes.values()),
                "path": str(self.network_model_dir),
            },
            "windows_specialist": {
                "loaded": self.windows_specialist is not None,
                "classes": list(self.windows_classes.values()),
                "path": str(self.windows_model_dir),
            },
            "device": str(self.device),
            "ensemble_mode": "priority_voting",
        }


# For backward compatibility with existing code
class MLDetector(EnsembleMLDetector):
    """Alias for backward compatibility"""

    pass

"""
Ensemble Model for Revolutionary XDR Threat Detection

Combines multiple state-of-the-art models for robust predictions:
1. FT-Transformer (primary) - Best for tabular data
2. XGBoost - Gradient boosting baseline with SHAP explainability
3. Temporal LSTM - Sequence pattern detection

Uses uncertainty-weighted voting for final predictions.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    shap = None

from .ft_transformer import FTTransformer, FTTransformerConfig, FTTransformerDetector

logger = logging.getLogger(__name__)


@dataclass
class EnsembleConfig:
    """Configuration for ensemble model."""

    # Model weights for ensemble voting
    ft_transformer_weight: float = 0.5  # Primary model
    xgboost_weight: float = 0.3  # Gradient boosting
    lstm_weight: float = 0.2  # Temporal patterns

    # Use uncertainty weighting (dynamic weights based on model confidence)
    use_uncertainty_weighting: bool = True

    # Minimum confidence to include model in ensemble
    min_confidence_threshold: float = 0.3

    # XGBoost configuration
    xgb_n_estimators: int = 1000
    xgb_max_depth: int = 8
    xgb_learning_rate: float = 0.05
    xgb_n_jobs: int = -1

    # LSTM configuration
    lstm_hidden_dim: int = 128
    lstm_num_layers: int = 2
    lstm_sequence_length: int = 10

    # Paths
    models_dir: str = "/Users/chasemad/Desktop/mini-xdr/models"


class TemporalLSTM(nn.Module):
    """
    LSTM model for temporal pattern detection in security events.

    Processes sequences of events to detect attack patterns
    that unfold over time (multi-stage attacks, APT behavior).
    """

    def __init__(
        self,
        input_dim: int = 79,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 7,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        # Attention mechanism for sequence aggregation
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.attention = nn.Sequential(
            nn.Linear(lstm_output_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        # Uncertainty estimation
        self.uncertainty_head = nn.Sequential(
            nn.Linear(lstm_output_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through temporal LSTM.

        Args:
            x: (batch_size, seq_len, input_dim) sequence of events
            lengths: Optional sequence lengths for packing

        Returns:
            Dict with logits, probabilities, and uncertainty
        """
        batch_size, seq_len, _ = x.shape

        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)  # (B, S, H*2)

        # Attention-weighted aggregation
        attn_weights = self.attention(lstm_out)  # (B, S, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)

        # Weighted sum: (B, S, H*2) * (B, S, 1) -> (B, H*2)
        context = torch.sum(lstm_out * attn_weights, dim=1)

        # Classification
        logits = self.classifier(context)
        probs = torch.softmax(logits, dim=-1)

        # Uncertainty estimation
        uncertainty = self.uncertainty_head(context).squeeze(-1)

        return {
            "logits": logits,
            "probs": probs,
            "uncertainty": uncertainty,
            "attention_weights": attn_weights.squeeze(-1),
        }


class XGBoostWrapper:
    """
    XGBoost wrapper for threat detection with SHAP explainability.
    """

    CLASS_NAMES = {
        0: "Normal",
        1: "DDoS",
        2: "Reconnaissance",
        3: "Brute Force",
        4: "Web Attack",
        5: "Malware",
        6: "APT",
    }

    def __init__(self, config: EnsembleConfig):
        self.config = config
        self.model: Optional[xgb.XGBClassifier] = None
        self.explainer: Optional[shap.TreeExplainer] = None
        self.feature_names: Optional[List[str]] = None

        if not XGBOOST_AVAILABLE:
            logger.warning("XGBoost not available - this component will be disabled")

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        eval_set: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
    ):
        """Train XGBoost model."""
        if not XGBOOST_AVAILABLE:
            logger.error("XGBoost not available for training")
            return

        self.feature_names = feature_names or [
            f"feature_{i}" for i in range(X.shape[1])
        ]

        self.model = xgb.XGBClassifier(
            n_estimators=self.config.xgb_n_estimators,
            max_depth=self.config.xgb_max_depth,
            learning_rate=self.config.xgb_learning_rate,
            n_jobs=self.config.xgb_n_jobs,
            objective="multi:softprob",
            num_class=len(self.CLASS_NAMES),
            use_label_encoder=False,
            eval_metric="mlogloss",
            early_stopping_rounds=50,
            tree_method="hist",  # Fast histogram-based algorithm
        )

        self.model.fit(
            X,
            y,
            eval_set=eval_set or [(X, y)],
            verbose=True,
        )

        # Initialize SHAP explainer
        if SHAP_AVAILABLE:
            self.explainer = shap.TreeExplainer(self.model)
            logger.info("SHAP explainer initialized for XGBoost")

    def predict(self, X: np.ndarray) -> Dict[str, Any]:
        """Make predictions with confidence estimation."""
        if self.model is None:
            raise RuntimeError("Model not trained or loaded")

        probs = self.model.predict_proba(X)
        predictions = np.argmax(probs, axis=1)
        confidence = np.max(probs, axis=1)

        # Uncertainty as 1 - confidence (simple but effective for trees)
        uncertainty = 1.0 - confidence

        return {
            "predicted_class": predictions,
            "probs": probs,
            "confidence": confidence,
            "uncertainty": uncertainty,
        }

    def explain(self, X: np.ndarray) -> Dict[str, Any]:
        """Get SHAP explanations for predictions."""
        if self.explainer is None:
            return {"error": "SHAP explainer not available"}

        shap_values = self.explainer.shap_values(X)

        return {
            "shap_values": shap_values,
            "feature_names": self.feature_names,
            "expected_value": self.explainer.expected_value,
        }

    def save(self, path: str):
        """Save XGBoost model."""
        if self.model is not None:
            self.model.save_model(path)
            logger.info(f"XGBoost model saved to {path}")

    def load(self, path: str):
        """Load XGBoost model."""
        if XGBOOST_AVAILABLE:
            self.model = xgb.XGBClassifier()
            self.model.load_model(path)
            if SHAP_AVAILABLE:
                self.explainer = shap.TreeExplainer(self.model)
            logger.info(f"XGBoost model loaded from {path}")


class EnsembleDetector:
    """
    Ensemble threat detector combining multiple models.

    Uses uncertainty-weighted voting to combine:
    1. FT-Transformer - Best overall performance
    2. XGBoost - Strong baseline with explainability
    3. Temporal LSTM - Sequence pattern detection
    """

    CLASS_NAMES = {
        0: "Normal",
        1: "DDoS",
        2: "Reconnaissance",
        3: "Brute Force",
        4: "Web Attack",
        5: "Malware",
        6: "APT",
    }

    def __init__(
        self,
        config: Optional[EnsembleConfig] = None,
        device: Optional[str] = None,
    ):
        self.config = config or EnsembleConfig()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize models
        self.ft_transformer: Optional[FTTransformerDetector] = None
        self.xgboost: Optional[XGBoostWrapper] = None
        self.lstm: Optional[TemporalLSTM] = None

        # Model availability flags
        self.models_loaded = {
            "ft_transformer": False,
            "xgboost": False,
            "lstm": False,
        }

        # Scaler for preprocessing
        self.scaler = None

        logger.info(f"EnsembleDetector initialized on {self.device}")

    def load_models(
        self,
        ft_transformer_path: Optional[str] = None,
        xgboost_path: Optional[str] = None,
        lstm_path: Optional[str] = None,
    ):
        """Load all ensemble models."""
        models_dir = Path(self.config.models_dir)

        # Load FT-Transformer
        ft_path = (
            ft_transformer_path or models_dir / "revolutionary" / "ft_transformer.pth"
        )
        if Path(ft_path).exists():
            try:
                self.ft_transformer = FTTransformerDetector(model_path=str(ft_path))
                self.models_loaded["ft_transformer"] = True
                logger.info("FT-Transformer loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load FT-Transformer: {e}")
        else:
            # Initialize without weights (for training)
            self.ft_transformer = FTTransformerDetector()
            logger.info("FT-Transformer initialized without pre-trained weights")

        # Load XGBoost
        xgb_path = xgboost_path or models_dir / "revolutionary" / "xgboost.json"
        if XGBOOST_AVAILABLE:
            self.xgboost = XGBoostWrapper(self.config)
            if Path(xgb_path).exists():
                try:
                    self.xgboost.load(str(xgb_path))
                    self.models_loaded["xgboost"] = True
                    logger.info("XGBoost loaded successfully")
                except Exception as e:
                    logger.error(f"Failed to load XGBoost: {e}")

        # Load LSTM
        lstm_path_file = lstm_path or models_dir / "revolutionary" / "temporal_lstm.pth"
        self.lstm = TemporalLSTM(
            input_dim=79,
            hidden_dim=self.config.lstm_hidden_dim,
            num_layers=self.config.lstm_num_layers,
            num_classes=len(self.CLASS_NAMES),
        ).to(self.device)

        if Path(lstm_path_file).exists():
            try:
                checkpoint = torch.load(lstm_path_file, map_location=self.device)
                self.lstm.load_state_dict(checkpoint["model_state_dict"])
                self.models_loaded["lstm"] = True
                logger.info("Temporal LSTM loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load LSTM: {e}")

        loaded_count = sum(self.models_loaded.values())
        logger.info(f"Ensemble ready: {loaded_count}/3 models loaded")

    async def predict(
        self,
        features: np.ndarray,
        event_sequence: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Make ensemble prediction with uncertainty-weighted voting.

        Args:
            features: (batch_size, 79) feature array for single-event prediction
            event_sequence: Optional (batch_size, seq_len, 79) for temporal analysis

        Returns:
            Dict with ensemble prediction, confidence, uncertainty, and per-model results
        """
        model_results = {}
        weights = {}

        # FT-Transformer prediction
        if self.ft_transformer is not None:
            try:
                ft_result = await self.ft_transformer.predict(features)
                # Normalize key names for ensemble
                ft_result["probs"] = ft_result.get(
                    "class_probabilities",
                    ft_result.get("probs", np.zeros((len(features), 7))),
                )
                model_results["ft_transformer"] = ft_result

                # Calculate weight based on uncertainty
                if self.config.use_uncertainty_weighting:
                    # Lower uncertainty = higher weight
                    confidence = 1.0 - ft_result["uncertainty"].mean()
                    weights["ft_transformer"] = (
                        self.config.ft_transformer_weight * confidence
                    )
                else:
                    weights["ft_transformer"] = self.config.ft_transformer_weight
            except Exception as e:
                logger.error(f"FT-Transformer prediction failed: {e}")

        # XGBoost prediction
        if self.xgboost is not None and self.xgboost.model is not None:
            try:
                xgb_result = self.xgboost.predict(features)
                # Ensure probs key exists
                if "probs" not in xgb_result:
                    xgb_result["probs"] = xgb_result.get(
                        "class_probabilities", np.zeros((len(features), 7))
                    )
                model_results["xgboost"] = xgb_result

                if self.config.use_uncertainty_weighting:
                    confidence = xgb_result["confidence"].mean()
                    weights["xgboost"] = self.config.xgboost_weight * confidence
                else:
                    weights["xgboost"] = self.config.xgboost_weight
            except Exception as e:
                logger.error(f"XGBoost prediction failed: {e}")

        # LSTM prediction (if sequence data provided)
        if self.lstm is not None and event_sequence is not None:
            try:
                self.lstm.eval()
                with torch.no_grad():
                    seq_tensor = torch.tensor(
                        event_sequence, dtype=torch.float32, device=self.device
                    )
                    lstm_output = self.lstm(seq_tensor)

                lstm_result = {
                    "predicted_class": lstm_output["probs"]
                    .argmax(dim=-1)
                    .cpu()
                    .numpy(),
                    "probs": lstm_output["probs"].cpu().numpy(),
                    "confidence": lstm_output["probs"].max(dim=-1)[0].cpu().numpy(),
                    "uncertainty": lstm_output["uncertainty"].cpu().numpy(),
                }
                model_results["lstm"] = lstm_result

                if self.config.use_uncertainty_weighting:
                    confidence = lstm_result["confidence"].mean()
                    weights["lstm"] = self.config.lstm_weight * confidence
                else:
                    weights["lstm"] = self.config.lstm_weight
            except Exception as e:
                logger.error(f"LSTM prediction failed: {e}")

        # Weighted ensemble voting
        ensemble_result = self._weighted_ensemble(model_results, weights)

        # Add per-model results for analysis
        ensemble_result["model_results"] = model_results
        ensemble_result["model_weights"] = weights
        ensemble_result["models_used"] = list(model_results.keys())

        return ensemble_result

    def _weighted_ensemble(
        self,
        model_results: Dict[str, Dict[str, Any]],
        weights: Dict[str, float],
    ) -> Dict[str, Any]:
        """Combine model predictions using weighted voting."""
        if not model_results:
            return {
                "error": "No model predictions available",
                "predicted_class": np.array([-1]),
                "confidence": np.array([0.0]),
                "uncertainty": np.array([1.0]),
            }

        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight == 0:
            total_weight = 1.0

        normalized_weights = {k: v / total_weight for k, v in weights.items()}

        # Get batch size from first result
        first_result = next(iter(model_results.values()))
        batch_size = len(first_result["probs"])
        num_classes = first_result["probs"].shape[1]

        # Weighted probability averaging
        ensemble_probs = np.zeros((batch_size, num_classes))
        ensemble_uncertainty = np.zeros(batch_size)

        for model_name, result in model_results.items():
            weight = normalized_weights.get(model_name, 0)
            ensemble_probs += weight * result["probs"]
            ensemble_uncertainty += weight * result["uncertainty"]

        # Final predictions
        predicted_class = np.argmax(ensemble_probs, axis=1)
        confidence = np.max(ensemble_probs, axis=1)

        # Get threat type names
        threat_types = [
            self.CLASS_NAMES.get(c, f"Unknown_{c}") for c in predicted_class
        ]

        return {
            "predicted_class": predicted_class,
            "threat_type": threat_types,
            "class_probabilities": ensemble_probs,
            "confidence": confidence,
            "uncertainty": ensemble_uncertainty,
            "high_uncertainty": ensemble_uncertainty > 0.3,
        }

    def get_explanation(
        self,
        features: np.ndarray,
        use_shap: bool = True,
    ) -> Dict[str, Any]:
        """
        Get explanation for predictions.

        Combines FT-Transformer attention and XGBoost SHAP values.
        """
        explanations = {}

        # FT-Transformer attention-based importance
        if self.ft_transformer is not None:
            try:
                x = torch.tensor(features, dtype=torch.float32, device=self.device)
                importance = self.ft_transformer.model.get_feature_importance(x)
                explanations["ft_transformer_attention"] = importance.cpu().numpy()
            except Exception as e:
                logger.error(f"FT-Transformer explanation failed: {e}")

        # XGBoost SHAP explanation
        if use_shap and self.xgboost is not None and self.xgboost.explainer is not None:
            try:
                shap_result = self.xgboost.explain(features)
                explanations["xgboost_shap"] = shap_result
            except Exception as e:
                logger.error(f"XGBoost SHAP explanation failed: {e}")

        return explanations

    def save_all(self, output_dir: str):
        """Save all ensemble models."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save FT-Transformer
        if self.ft_transformer is not None:
            self.ft_transformer.save_model(str(output_path / "ft_transformer.pth"))

        # Save XGBoost
        if self.xgboost is not None and self.xgboost.model is not None:
            self.xgboost.save(str(output_path / "xgboost.json"))

        # Save LSTM
        if self.lstm is not None:
            torch.save(
                {"model_state_dict": self.lstm.state_dict()},
                output_path / "temporal_lstm.pth",
            )

        logger.info(f"Ensemble models saved to {output_dir}")

    def get_info(self) -> Dict[str, Any]:
        """Get ensemble configuration and status."""
        return {
            "ensemble_type": "uncertainty_weighted",
            "models": {
                "ft_transformer": {
                    "loaded": self.models_loaded.get("ft_transformer", False),
                    "weight": self.config.ft_transformer_weight,
                },
                "xgboost": {
                    "loaded": self.models_loaded.get("xgboost", False),
                    "weight": self.config.xgboost_weight,
                    "available": XGBOOST_AVAILABLE,
                },
                "lstm": {
                    "loaded": self.models_loaded.get("lstm", False),
                    "weight": self.config.lstm_weight,
                },
            },
            "config": {
                "use_uncertainty_weighting": self.config.use_uncertainty_weighting,
                "min_confidence_threshold": self.config.min_confidence_threshold,
            },
            "class_names": self.CLASS_NAMES,
        }


# Singleton instance
ensemble_detector: Optional[EnsembleDetector] = None


def get_ensemble_detector(
    config: Optional[EnsembleConfig] = None,
    load_models: bool = True,
) -> EnsembleDetector:
    """Get or create ensemble detector singleton."""
    global ensemble_detector

    if ensemble_detector is None:
        ensemble_detector = EnsembleDetector(config=config)
        if load_models:
            ensemble_detector.load_models()

    return ensemble_detector

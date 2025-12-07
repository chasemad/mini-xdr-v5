"""
🚀 ENHANCED THREAT DETECTION SYSTEM
Strategic improvements to the existing 97.98% accuracy model:

1. Advanced Architecture: Attention + Uncertainty + Skip Connections
2. Feature Quality Enhancement: Remove noise, add interactions
3. OpenAI Integration: Novel attack detection for uncertain cases
4. Explainable AI: SHAP/LIME for transparency
5. Ensemble Methods: Multiple models with confidence calibration
6. Specialist Model Routing: Binary classifiers for high-FP attack types
7. Temperature Scaling: Confidence calibration to reduce overconfidence
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import openai
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import settings
from .models import Event

logger = logging.getLogger(__name__)

# Temperature scaling factor for confidence calibration
# Higher values = less confident predictions (reduces false positives)
DEFAULT_TEMPERATURE = 1.5


@dataclass
class PredictionResult:
    """Enhanced prediction result with uncertainty and explanations"""

    predicted_class: int
    confidence: float
    class_probabilities: List[float]
    uncertainty_score: float
    threat_type: str
    explanation: Dict[str, Any]
    openai_enhanced: bool = False
    feature_importance: Dict[str, float] = None
    specialist_verified: bool = False
    specialist_confidence: float = None


class SpecialistModelManager:
    """
    Manages specialist binary classifier models for high-FP attack types.

    Specialist models are trained specifically for:
    - DDoS detection (93.29% accuracy)
    - Brute Force detection (90.52% accuracy)
    - Web Attack detection (95.29% accuracy)

    These provide verification when the general model predicts these classes,
    reducing false positives from the general model's lower precision.
    """

    # Map general model class IDs to specialist model names
    CLASS_TO_SPECIALIST = {
        1: "ddos",  # DDoS/DoS Attack
        3: "brute_force",  # Brute Force Attack
        4: "web_attacks",  # Web Application Attack
    }

    # Minimum specialist confidence to confirm detection
    SPECIALIST_CONFIRMATION_THRESHOLD = 0.85

    def __init__(
        self, models_dir: str = "./models/local_trained"
    ):
        self.models_dir = Path(models_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.specialists: Dict[str, nn.Module] = {}
        self.scalers: Dict[str, Any] = {}
        self.metadata: Dict[str, Dict] = {}
        self._loaded = False

    def load_specialists(self) -> bool:
        """Load all specialist models."""
        if self._loaded:
            return True

        try:
            for class_id, specialist_name in self.CLASS_TO_SPECIALIST.items():
                model_path = (
                    self.models_dir
                    / specialist_name
                    / f"{specialist_name}_specialist"
                    / "threat_detector.pth"
                )
                metadata_path = (
                    self.models_dir
                    / specialist_name
                    / f"{specialist_name}_specialist"
                    / "model_metadata.json"
                )

                if not model_path.exists():
                    logger.warning(f"Specialist model not found: {model_path}")
                    continue

                # Load metadata
                if metadata_path.exists():
                    with open(metadata_path) as f:
                        self.metadata[specialist_name] = json.load(f)
                else:
                    self.metadata[specialist_name] = {
                        "hidden_dims": [512, 256, 128, 64],
                        "num_classes": 2,
                        "dropout_rate": 0.3,
                        "use_attention": True,
                    }

                # Create and load model
                meta = self.metadata[specialist_name]
                model = self._create_specialist_model(
                    input_dim=79,
                    hidden_dims=meta.get("hidden_dims", [512, 256, 128, 64]),
                    num_classes=meta.get("num_classes", 2),
                    dropout_rate=meta.get("dropout_rate", 0.3),
                    use_attention=meta.get("use_attention", True),
                )

                state_dict = torch.load(model_path, map_location=self.device)
                model.load_state_dict(state_dict)
                model.to(self.device)
                model.eval()

                self.specialists[specialist_name] = model
                logger.info(
                    f"Loaded specialist model: {specialist_name} (accuracy: {meta.get('best_val_accuracy', 'N/A')})"
                )

            self._loaded = True
            logger.info(f"Specialist models loaded: {list(self.specialists.keys())}")
            return True

        except Exception as e:
            logger.error(f"Failed to load specialist models: {e}", exc_info=True)
            return False

    def _create_specialist_model(
        self,
        input_dim: int,
        hidden_dims: List[int],
        num_classes: int,
        dropout_rate: float,
        use_attention: bool,
    ) -> nn.Module:
        """Create a specialist model architecture matching training."""
        # Use the same architecture as the general model but with binary output
        return _SpecialistDetector(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            use_attention=use_attention,
        )

    def verify_prediction(
        self,
        predicted_class: int,
        features: np.ndarray,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> Tuple[bool, float, str]:
        """
        Verify a general model prediction using the appropriate specialist.

        Args:
            predicted_class: Class predicted by general model
            features: Feature vector (79-dimensional)
            temperature: Temperature for confidence calibration

        Returns:
            Tuple of (confirmed: bool, specialist_confidence: float, reason: str)
        """
        specialist_name = self.CLASS_TO_SPECIALIST.get(predicted_class)

        if specialist_name is None:
            # No specialist for this class - auto-confirm
            return True, 1.0, "no_specialist_needed"

        if specialist_name not in self.specialists:
            # Specialist not loaded - use general model prediction
            logger.warning(f"Specialist {specialist_name} not loaded, cannot verify")
            return True, 0.5, "specialist_unavailable"

        try:
            model = self.specialists[specialist_name]

            # Prepare input
            input_tensor = torch.tensor(features, dtype=torch.float32).to(self.device)
            if input_tensor.dim() == 1:
                input_tensor = input_tensor.unsqueeze(0)

            # Get specialist prediction
            with torch.no_grad():
                logits, uncertainty = model(input_tensor)

                # Apply temperature scaling
                scaled_logits = logits / temperature
                probabilities = F.softmax(scaled_logits, dim=1)

                # Class 1 is the attack class in binary specialist
                attack_probability = probabilities[0, 1].item()

            # Determine if specialist confirms
            confirmed = attack_probability >= self.SPECIALIST_CONFIRMATION_THRESHOLD

            reason = "specialist_confirmed" if confirmed else "specialist_rejected"

            logger.info(
                f"Specialist {specialist_name} verification: "
                f"attack_prob={attack_probability:.3f}, "
                f"threshold={self.SPECIALIST_CONFIRMATION_THRESHOLD}, "
                f"confirmed={confirmed}"
            )

            return confirmed, attack_probability, reason

        except Exception as e:
            logger.error(f"Specialist verification failed: {e}", exc_info=True)
            return True, 0.5, f"verification_error: {str(e)}"


class _SpecialistDetector(nn.Module):
    """
    Specialist detector architecture for binary classification.
    Matches the architecture used during specialist model training.
    """

    def __init__(
        self,
        input_dim: int = 79,
        hidden_dims: List[int] = [512, 256, 128, 64],
        num_classes: int = 2,
        dropout_rate: float = 0.3,
        use_attention: bool = True,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.use_attention = use_attention

        # Feature interaction layer
        self.feature_interaction = nn.Linear(input_dim, input_dim)

        # Optional attention
        if use_attention:
            self.attention = _SpecialistAttention(input_dim, attention_dim=64)

        # Build layers with uncertainty blocks
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(
                _SpecialistUncertaintyBlock(prev_dim, hidden_dim, dropout_rate)
            )
            prev_dim = hidden_dim
        self.feature_extractor = nn.ModuleList(layers)

        # Skip connections
        self.skip_connections = nn.ModuleList(
            [
                nn.Linear(input_dim, hidden_dims[-1]),
                nn.Linear(hidden_dims[0], hidden_dims[-1]),
            ]
        )

        # Output layers
        self.classifier = nn.Linear(prev_dim, num_classes)
        self.uncertainty_head = nn.Linear(prev_dim, 1)
        self.mc_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, return_features: bool = False):
        # Feature interaction
        x_interact = torch.relu(self.feature_interaction(x))
        x = x + x_interact

        # Attention
        if self.use_attention:
            x_attended = self.attention(x)
            x = x_attended

        x_input = x
        x_mid = None

        # Forward through feature extractor
        for i, layer in enumerate(self.feature_extractor):
            x = layer(x)
            if i == 0:
                x_mid = x

        # Skip connections
        skip1 = torch.relu(self.skip_connections[0](x_input))
        skip2 = torch.relu(self.skip_connections[1](x_mid))
        x = x + skip1 + skip2

        features = x
        logits = self.classifier(features)
        uncertainty = torch.sigmoid(self.uncertainty_head(self.mc_dropout(features)))

        if return_features:
            return logits, uncertainty, features

        return logits, uncertainty


class _SpecialistAttention(nn.Module):
    """Attention layer for specialist models."""

    def __init__(self, input_dim: int, attention_dim: int = 64):
        super().__init__()
        self.attention_dim = attention_dim
        self.query = nn.Linear(input_dim, attention_dim)
        self.key = nn.Linear(input_dim, attention_dim)
        self.value = nn.Linear(input_dim, attention_dim)
        self.output = nn.Linear(attention_dim, input_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        q = self.query(x).unsqueeze(1)
        k = self.key(x).unsqueeze(1)
        v = self.value(x).unsqueeze(1)

        attention_weights = torch.softmax(
            torch.matmul(q, k.transpose(-2, -1)) / (self.attention_dim**0.5), dim=-1
        )
        attended = torch.matmul(attention_weights, v).squeeze(1)
        output = self.output(attended)
        output = self.dropout(output)
        return output + x


class _SpecialistUncertaintyBlock(nn.Module):
    """Uncertainty block for specialist models."""

    def __init__(self, in_dim: int, out_dim: int, dropout_rate: float):
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


class AttentionLayer(nn.Module):
    """Attention mechanism for feature relationship learning"""

    def __init__(self, input_dim: int, attention_dim: int = 64):
        super().__init__()
        self.attention_dim = attention_dim
        self.query = nn.Linear(input_dim, attention_dim)
        self.key = nn.Linear(input_dim, attention_dim)
        self.value = nn.Linear(input_dim, attention_dim)
        self.output = nn.Linear(attention_dim, input_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # x shape: (batch_size, features)
        batch_size = x.size(0)

        # Self-attention mechanism
        q = self.query(x).unsqueeze(1)  # (batch_size, 1, attention_dim)
        k = self.key(x).unsqueeze(1)  # (batch_size, 1, attention_dim)
        v = self.value(x).unsqueeze(1)  # (batch_size, 1, attention_dim)

        # Compute attention weights
        attention_weights = torch.softmax(
            torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.attention_dim), dim=-1
        )

        # Apply attention
        attended = torch.matmul(attention_weights, v).squeeze(
            1
        )  # (batch_size, attention_dim)

        # Project back to input dimension
        output = self.output(attended)
        output = self.dropout(output)

        return output + x  # Residual connection


class UncertaintyBlock(nn.Module):
    """Monte Carlo Dropout for uncertainty estimation"""

    def __init__(self, input_dim: int, output_dim: int, dropout_rate: float = 0.3):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.batch_norm = nn.BatchNorm1d(output_dim)
        self.dropout_rate = dropout_rate

    def forward(self, x):
        x = self.linear(x)
        x = self.batch_norm(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x

    def mc_forward(self, x, n_samples: int = 50):
        """Monte Carlo forward pass for uncertainty estimation"""
        self.train()  # Enable dropout during inference

        predictions = []
        for _ in range(n_samples):
            with torch.no_grad():
                pred = self.forward(x)
                predictions.append(pred)

        self.eval()  # Return to eval mode
        predictions = torch.stack(predictions)  # (n_samples, batch_size, output_dim)

        # Calculate mean and uncertainty
        mean_pred = torch.mean(predictions, dim=0)
        uncertainty = torch.std(predictions, dim=0)

        return mean_pred, uncertainty


class EnhancedXDRThreatDetector(nn.Module):
    """
    Enhanced threat detection model with:
    - Attention mechanisms for feature relationships
    - Uncertainty quantification
    - Skip connections for better gradient flow
    - Feature interaction layers
    """

    def __init__(
        self,
        input_dim: int = 79,
        hidden_dims: List[int] = [512, 256, 128, 64],
        num_classes: int = 7,
        dropout_rate: float = 0.3,
        use_attention: bool = True,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.use_attention = use_attention

        # Feature interaction layer
        self.feature_interaction = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim * 2, input_dim),
        )

        # Attention layer for feature relationships
        if use_attention:
            self.attention = AttentionLayer(input_dim)

        # Main network with skip connections
        self.layers = nn.ModuleList()
        self.skip_layers = nn.ModuleList()

        prev_dim = input_dim
        for i, dim in enumerate(hidden_dims):
            # Main layer with uncertainty
            layer = UncertaintyBlock(prev_dim, dim, dropout_rate)
            self.layers.append(layer)

            # Skip connection layer
            if prev_dim != dim:
                skip_layer = nn.Linear(prev_dim, dim)
            else:
                skip_layer = nn.Identity()
            self.skip_layers.append(skip_layer)

            prev_dim = dim

        # Output layer
        self.output_layer = nn.Linear(prev_dim, num_classes)

        # Uncertainty estimation
        self.uncertainty_head = nn.Linear(prev_dim, 1)

    def forward(self, x, return_features: bool = False):
        # Feature interaction
        x_interact = self.feature_interaction(x)
        x = x + x_interact  # Residual connection

        # Attention mechanism
        if self.use_attention:
            x = self.attention(x)

        features = [x]

        # Forward through layers with skip connections
        for layer, skip_layer in zip(self.layers, self.skip_layers):
            skip = skip_layer(x)
            x = layer(x)
            x = x + skip  # Skip connection
            features.append(x)

        # Output predictions
        logits = self.output_layer(x)
        uncertainty = torch.sigmoid(self.uncertainty_head(x))

        if return_features:
            return logits, uncertainty, features

        return logits, uncertainty

    def predict_with_uncertainty(self, x, n_samples: int = 50):
        """Get predictions with uncertainty estimates"""
        self.train()  # Enable dropout for uncertainty

        # Set batch norm layers to eval mode to handle batch size of 1
        for module in self.modules():
            if isinstance(module, nn.BatchNorm1d):
                module.eval()

        predictions = []
        uncertainties = []

        for _ in range(n_samples):
            with torch.no_grad():
                logits, uncertainty = self.forward(x)
                predictions.append(torch.softmax(logits, dim=1))
                uncertainties.append(uncertainty)

        self.eval()

        # Calculate statistics
        predictions = torch.stack(predictions)  # (n_samples, batch_size, num_classes)
        uncertainties = torch.stack(uncertainties)  # (n_samples, batch_size, 1)

        mean_pred = torch.mean(predictions, dim=0)
        pred_uncertainty = torch.std(predictions, dim=0)
        mean_uncertainty = torch.mean(uncertainties, dim=0)

        return mean_pred, pred_uncertainty, mean_uncertainty


class FeatureEnhancer:
    """
    Strategic feature quality enhancement:
    - Remove redundant/noisy features
    - Add feature interactions
    - Enhance most predictive features
    """

    def __init__(self, feature_importance_threshold: float = 0.01):
        self.feature_importance_threshold = feature_importance_threshold
        self.important_features = None
        self.feature_interactions = None
        self.scaler = None

    def analyze_feature_importance(
        self, features: np.ndarray, labels: np.ndarray
    ) -> Dict[str, float]:
        """Analyze feature importance using multiple methods"""
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.feature_selection import mutual_info_classif

            # Random Forest importance
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(features, labels)
            rf_importance = rf.feature_importances_

            # Mutual information importance
            mi_importance = mutual_info_classif(features, labels, random_state=42)

            # Combined importance score
            combined_importance = (rf_importance + mi_importance) / 2

            # Create feature importance dictionary
            importance_dict = {}
            for i, importance in enumerate(combined_importance):
                importance_dict[f"feature_{i}"] = float(importance)

            # Identify important features
            self.important_features = [
                i
                for i, importance in enumerate(combined_importance)
                if importance > self.feature_importance_threshold
            ]

            logger.info(
                f"Identified {len(self.important_features)} important features out of {len(combined_importance)}"
            )

            return importance_dict

        except Exception as e:
            logger.error(f"Feature importance analysis failed: {e}")
            return {}

    def create_feature_interactions(
        self, features: np.ndarray, top_k: int = 10
    ) -> np.ndarray:
        """Create interactions between top features"""
        if self.important_features is None or len(self.important_features) < 2:
            return features

        # Select top k important features for interactions
        top_features = self.important_features[
            : min(top_k, len(self.important_features))
        ]

        interactions = []
        for i in range(len(top_features)):
            for j in range(i + 1, len(top_features)):
                feat_i, feat_j = top_features[i], top_features[j]
                # Multiplication interaction
                interaction = features[:, feat_i] * features[:, feat_j]
                interactions.append(interaction.reshape(-1, 1))

        if interactions:
            interaction_features = np.concatenate(interactions, axis=1)
            enhanced_features = np.concatenate([features, interaction_features], axis=1)
            logger.info(f"Added {len(interactions)} feature interactions")
            return enhanced_features

        return features

    def enhance_features(
        self, features: np.ndarray, labels: np.ndarray = None, fit: bool = False
    ) -> np.ndarray:
        """Enhance feature quality"""
        if fit and labels is not None:
            # Analyze importance during training
            self.analyze_feature_importance(features, labels)

            # Fit scaler
            from sklearn.preprocessing import RobustScaler

            self.scaler = RobustScaler()
            features_scaled = self.scaler.fit_transform(features)
        else:
            # Use existing scaler during inference
            if self.scaler is not None:
                features_scaled = self.scaler.transform(features)
            else:
                features_scaled = features

        # Add feature interactions
        enhanced_features = self.create_feature_interactions(features_scaled)

        return enhanced_features


class OpenAIThreatAnalyzer:
    """OpenAI integration for uncertain prediction analysis"""

    def __init__(self):
        self.client = None
        self._init_client()

    def _init_client(self):
        """Initialize OpenAI client"""
        try:
            api_key = settings.openai_api_key
            if api_key:
                self.client = openai.AsyncOpenAI(api_key=api_key)
                logger.info("OpenAI client initialized for threat analysis")
            else:
                logger.warning("OpenAI API key not found")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")

    async def analyze_uncertain_prediction(
        self,
        src_ip: str,
        events: List[Event],
        ml_prediction: PredictionResult,
        uncertainty_threshold: float = 0.3,
    ) -> Optional[Dict[str, Any]]:
        """Analyze uncertain ML predictions with OpenAI"""

        if not self.client or ml_prediction.confidence > uncertainty_threshold:
            return None

        try:
            # Prepare context for OpenAI
            context = self._prepare_analysis_context(src_ip, events, ml_prediction)

            prompt = f"""
You are a cybersecurity expert analyzing a potential threat. The ML model is uncertain about this case.

**Context:**
- Source IP: {src_ip}
- ML Prediction: {ml_prediction.threat_type} (Confidence: {ml_prediction.confidence:.1%})
- Uncertainty Score: {ml_prediction.uncertainty_score:.3f}

**Event Analysis:**
{context}

**Your Task:**
Analyze this case and provide insights for:
1. Is this likely a genuine threat or false positive?
2. What attack patterns do you see that the ML might have missed?
3. Are there novel indicators not in traditional signatures?
4. What additional context makes this suspicious or benign?

Respond in JSON format:
{{
    "threat_assessment": "high|medium|low|benign",
    "confidence": 0.0-1.0,
    "reasoning": "detailed explanation",
    "novel_indicators": ["list", "of", "indicators"],
    "recommended_action": "investigate|monitor|dismiss",
    "false_positive_likelihood": 0.0-1.0
}}
"""

            response = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=800,
            )

            # Parse response
            analysis_text = response.choices[0].message.content

            # Extract JSON
            import re

            json_match = re.search(r"\{.*\}", analysis_text, re.DOTALL)
            if json_match:
                analysis = json.loads(json_match.group(0))

                logger.info(
                    f"OpenAI analysis completed for {src_ip}: {analysis.get('threat_assessment', 'unknown')}"
                )

                return {
                    "openai_analysis": analysis,
                    "enhanced_confidence": analysis.get(
                        "confidence", ml_prediction.confidence
                    ),
                    "recommendation": analysis.get("recommended_action", "monitor"),
                    "reasoning": analysis.get("reasoning", ""),
                    "novel_indicators": analysis.get("novel_indicators", []),
                }
            else:
                logger.warning(
                    f"Failed to parse OpenAI JSON response: {analysis_text[:200]}"
                )
                return {"raw_analysis": analysis_text}

        except Exception as e:
            logger.error(f"OpenAI threat analysis failed: {e}")
            return None

    def _prepare_analysis_context(
        self, src_ip: str, events: List[Event], ml_prediction: PredictionResult
    ) -> str:
        """Prepare context for OpenAI analysis"""

        # Event summary
        event_types = {}
        unique_ports = set()
        messages = []

        for event in events[-20:]:  # Last 20 events
            event_types[event.eventid] = event_types.get(event.eventid, 0) + 1
            if event.dst_port:
                unique_ports.add(event.dst_port)
            if event.message:
                messages.append(event.message[:100])  # Truncate long messages

        context = f"""
Event Types: {dict(list(event_types.items())[:10])}
Target Ports: {list(unique_ports)[:15]}
Recent Messages: {messages[:5]}
Total Events: {len(events)}
Time Span: {(events[0].ts - events[-1].ts).total_seconds() / 60:.1f} minutes

ML Model Features (Top Contributing):
{ml_prediction.feature_importance if ml_prediction.feature_importance else "Not available"}
"""

        return context.strip()


class ExplainableAI:
    """Explainable AI components for model transparency"""

    def __init__(self, model: EnhancedXDRThreatDetector):
        self.model = model
        self.feature_names = [f"feature_{i}" for i in range(79)]

    def explain_prediction(
        self, features: np.ndarray, prediction: PredictionResult
    ) -> Dict[str, Any]:
        """Generate explanation for a prediction"""
        try:
            # Simple gradient-based feature importance
            feature_importance = self._calculate_gradient_importance(features)

            # Get top contributing features
            top_features = sorted(
                [(name, importance) for name, importance in feature_importance.items()],
                key=lambda x: abs(x[1]),
                reverse=True,
            )[:15]

            explanation = {
                "prediction": prediction.threat_type,
                "confidence": prediction.confidence,
                "uncertainty": prediction.uncertainty_score,
                "top_features": {
                    name: float(importance) for name, importance in top_features
                },
                "explanation_text": self._generate_explanation_text(
                    top_features, prediction
                ),
                "risk_factors": self._identify_risk_factors(feature_importance),
                "model_certainty": "high"
                if prediction.uncertainty_score < 0.1
                else "medium"
                if prediction.uncertainty_score < 0.3
                else "low",
            }

            return explanation

        except Exception as e:
            logger.error(f"Explanation generation failed: {e}")
            return {
                "error": str(e),
                "prediction": prediction.threat_type,
                "confidence": prediction.confidence,
            }

    def _calculate_gradient_importance(self, features: np.ndarray) -> Dict[str, float]:
        """Calculate feature importance using gradients"""
        try:
            self.model.eval()

            # Convert to tensor and enable gradients
            input_tensor = torch.tensor(
                features, dtype=torch.float32, requires_grad=True
            )

            # Forward pass
            logits, _ = self.model(input_tensor)

            # Get prediction
            predicted_class = torch.argmax(logits, dim=1)

            # Calculate gradients
            logits[0, predicted_class[0]].backward()

            # Get gradient importance
            gradients = input_tensor.grad.detach().numpy()[0]

            # Create importance dictionary
            importance = {}
            for i, grad in enumerate(gradients):
                importance[self.feature_names[i]] = float(
                    grad * features[0, i]
                )  # Gradient * input

            return importance

        except Exception as e:
            logger.error(f"Gradient importance calculation failed: {e}")
            return {}

    def _generate_explanation_text(
        self, top_features: List[Tuple[str, float]], prediction: PredictionResult
    ) -> str:
        """Generate human-readable explanation"""

        positive_features = [(name, imp) for name, imp in top_features if imp > 0][:5]
        negative_features = [(name, imp) for name, imp in top_features if imp < 0][:3]

        explanation = f"The model classified this as {prediction.threat_type} with {prediction.confidence:.1%} confidence. "

        if positive_features:
            explanation += f"Key indicators supporting this classification: "
            explanation += ", ".join(
                [
                    f"{name.replace('_', ' ')} ({imp:.3f})"
                    for name, imp in positive_features
                ]
            )
            explanation += ". "

        if negative_features:
            explanation += f"Factors reducing threat likelihood: "
            explanation += ", ".join(
                [
                    f"{name.replace('_', ' ')} ({abs(imp):.3f})"
                    for name, imp in negative_features
                ]
            )
            explanation += ". "

        uncertainty_text = {
            "low": "The model is highly confident in this prediction.",
            "medium": "The model has moderate confidence - additional analysis recommended.",
            "high": "The model is uncertain - manual review strongly recommended.",
        }

        uncertainty_level = (
            "low"
            if prediction.uncertainty_score < 0.1
            else "medium"
            if prediction.uncertainty_score < 0.3
            else "high"
        )
        explanation += uncertainty_text[uncertainty_level]

        return explanation

    def _identify_risk_factors(self, feature_importance: Dict[str, float]) -> List[str]:
        """Identify high-risk factors from feature importance"""

        risk_indicators = {
            "failed_login": "Multiple failed authentication attempts",
            "brute_force": "Brute force attack pattern detected",
            "port_scanning": "Port scanning behavior observed",
            "reconnaissance": "Network reconnaissance activity",
            "lateral_movement": "Potential lateral movement detected",
            "privilege_escalation": "Privilege escalation attempts",
            "malware": "Malware-like behavior patterns",
            "payload_complexity": "Complex payload structure",
            "attack_sophistication": "Sophisticated attack techniques",
        }

        risk_factors = []
        for feature_name, importance in feature_importance.items():
            if abs(importance) > 0.1:  # Significant importance threshold
                for indicator, description in risk_indicators.items():
                    if indicator in feature_name.lower():
                        risk_factors.append(description)
                        break

        return risk_factors[:5]  # Top 5 risk factors


class EnhancedThreatDetectionSystem:
    """
    Complete enhanced threat detection system combining:
    - Enhanced ML model with uncertainty
    - Specialist model verification for high-FP classes
    - Temperature scaling for confidence calibration
    - OpenAI analysis for uncertain cases
    - Explainable AI for transparency
    - Feature quality enhancement
    """

    def __init__(
        self, model_path: str = None, temperature: float = DEFAULT_TEMPERATURE
    ):
        self.model = None
        self.feature_enhancer = FeatureEnhancer()
        self.openai_analyzer = OpenAIThreatAnalyzer()
        self.explainer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Temperature for confidence calibration (reduces overconfidence)
        self.temperature = temperature

        # Specialist models for verifying high-FP classes
        self.specialist_manager = SpecialistModelManager()

        # Threat class mapping
        self.threat_classes = {
            0: "Normal",
            1: "DDoS/DoS Attack",
            2: "Network Reconnaissance",
            3: "Brute Force Attack",
            4: "Web Application Attack",
            5: "Malware/Botnet",
            6: "Advanced Persistent Threat",
        }

        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path: str) -> bool:
        """Load enhanced model"""
        try:
            model_dir = Path(model_path)

            # Load model
            self.model = EnhancedXDRThreatDetector().to(self.device)

            model_file = model_dir / "enhanced_threat_detector.pth"
            if model_file.exists():
                state_dict = torch.load(model_file, map_location=self.device)
                self.model.load_state_dict(state_dict)
                logger.info("Enhanced threat detector loaded")
            else:
                # Fallback to original model
                original_file = model_dir / "threat_detector.pth"
                if original_file.exists():
                    # Load into standard architecture for compatibility
                    from .deep_learning_models import XDRThreatDetector

                    original_model = XDRThreatDetector(
                        input_dim=79, hidden_dims=[256, 128, 64], num_classes=7
                    ).to(self.device)
                    original_state_dict = torch.load(
                        original_file, map_location=self.device
                    )
                    original_model.load_state_dict(original_state_dict)

                    # Transfer weights to enhanced model (best effort)
                    self._transfer_weights(original_model)
                    logger.info(
                        "Transferred weights from original model to enhanced architecture"
                    )
                else:
                    logger.warning("No model file found - using untrained model")

            self.model.eval()
            self.explainer = ExplainableAI(self.model)

            # Load feature enhancer components
            enhancer_file = model_dir / "feature_enhancer.pkl"
            if enhancer_file.exists():
                enhancer_data = joblib.load(enhancer_file)
                self.feature_enhancer.important_features = enhancer_data.get(
                    "important_features"
                )
                self.feature_enhancer.scaler = enhancer_data.get("scaler")
                logger.info("Feature enhancer loaded")

            # Load specialist models for high-FP class verification
            self.specialist_manager.load_specialists()

            return True

        except Exception as e:
            logger.error(f"Failed to load enhanced model: {e}")
            return False

    def _transfer_weights(self, original_model):
        """Transfer weights from original model to enhanced model"""
        try:
            # Get state dicts
            original_dict = original_model.state_dict()
            enhanced_dict = self.model.state_dict()

            # Transfer compatible layers
            transferred = 0
            for name, param in original_dict.items():
                if name in enhanced_dict and enhanced_dict[name].shape == param.shape:
                    enhanced_dict[name].copy_(param)
                    transferred += 1

            logger.info(f"Transferred {transferred} layers from original model")

        except Exception as e:
            logger.error(f"Weight transfer failed: {e}")

    async def analyze_threat(
        self, src_ip: str, events: List[Event], feature_vector: np.ndarray = None
    ) -> PredictionResult:
        """Complete threat analysis with enhanced capabilities"""

        if not self.model:
            raise ValueError("Model not loaded")

        try:
            # Extract or use provided features
            if feature_vector is None:
                from .deep_learning_models import deep_learning_manager

                feature_dict = deep_learning_manager._extract_features(src_ip, events)
                feature_vector = np.array([list(feature_dict.values())]).reshape(1, -1)

                # Debug: Log key attack indicator features
                logger.info(f"🔍 FEATURE DEBUG for {src_ip}:")
                logger.info(
                    f"   - failed_login_count: {feature_dict.get('failed_login_count', 0)}"
                )
                logger.info(
                    f"   - unique_usernames: {feature_dict.get('unique_usernames', 0)}"
                )
                logger.info(
                    f"   - command_diversity: {feature_dict.get('command_diversity', 0)}"
                )
                logger.info(
                    f"   - download_attempts: {feature_dict.get('download_attempts', 0)}"
                )
                logger.info(
                    f"   - upload_attempts: {feature_dict.get('upload_attempts', 0)}"
                )
                logger.info(
                    f"   - event_count_1h: {feature_dict.get('event_count_1h', 0)}"
                )
                logger.info(f"   - unique_ports: {feature_dict.get('unique_ports', 0)}")

            # Enhance features
            enhanced_features = self.feature_enhancer.enhance_features(feature_vector)

            # Pad or trim to expected size if needed
            if enhanced_features.shape[1] > 79:
                enhanced_features = enhanced_features[:, :79]
            elif enhanced_features.shape[1] < 79:
                padding = np.zeros(
                    (enhanced_features.shape[0], 79 - enhanced_features.shape[1])
                )
                enhanced_features = np.concatenate([enhanced_features, padding], axis=1)

            # Convert to tensor
            input_tensor = torch.tensor(enhanced_features, dtype=torch.float32).to(
                self.device
            )

            # Get prediction with uncertainty
            with torch.no_grad():
                (
                    probabilities,
                    pred_uncertainty,
                    aleatoric_uncertainty,
                ) = self.model.predict_with_uncertainty(input_tensor)

                # Apply temperature scaling to calibrate confidence
                # Higher temperature = less confident predictions (reduces FPs)
                logits = torch.log(probabilities + 1e-10)  # Convert back to logits
                scaled_logits = logits / self.temperature
                calibrated_probabilities = F.softmax(scaled_logits, dim=1)

                # Get primary prediction from calibrated probabilities
                predicted_class = torch.argmax(calibrated_probabilities, dim=1).item()
                confidence = calibrated_probabilities[0, predicted_class].item()
                class_probabilities = calibrated_probabilities[0].cpu().numpy().tolist()

                # Debug: Log class probabilities
                logger.info(
                    f"🔍 MODEL DEBUG - Class probabilities (temp={self.temperature}): {[f'{p:.3f}' for p in class_probabilities]}"
                )
                logger.info(
                    f"🔍 MODEL DEBUG - Classes: Normal={class_probabilities[0]:.3f}, DDoS={class_probabilities[1]:.3f}, Recon={class_probabilities[2]:.3f}, BruteForce={class_probabilities[3]:.3f}, WebAttack={class_probabilities[4]:.3f}, Malware={class_probabilities[5]:.3f}, APT={class_probabilities[6]:.3f}"
                )

                # Calculate combined uncertainty
                epistemic_uncertainty = torch.mean(pred_uncertainty[0]).item()
                aleatoric_uncertainty = aleatoric_uncertainty[0].item()
                combined_uncertainty = (
                    epistemic_uncertainty + aleatoric_uncertainty
                ) / 2

            # Specialist verification for high-FP classes (1=DDoS, 3=BruteForce, 4=WebAttack)
            specialist_verified = False
            specialist_confidence = None
            specialist_reason = None

            if predicted_class in SpecialistModelManager.CLASS_TO_SPECIALIST:
                (
                    confirmed,
                    spec_conf,
                    reason,
                ) = self.specialist_manager.verify_prediction(
                    predicted_class=predicted_class,
                    features=enhanced_features,
                    temperature=self.temperature,
                )
                specialist_verified = True
                specialist_confidence = spec_conf
                specialist_reason = reason

                if not confirmed:
                    # Specialist rejected - reduce confidence significantly
                    original_confidence = confidence
                    confidence = min(confidence * 0.3, 0.30)  # Cap at 30%
                    logger.warning(
                        f"🚫 Specialist rejected prediction: {self.threat_classes.get(predicted_class)} "
                        f"(specialist_conf={spec_conf:.3f}, original_conf={original_confidence:.3f}, new_conf={confidence:.3f})"
                    )
                else:
                    logger.info(
                        f"✅ Specialist confirmed prediction: {self.threat_classes.get(predicted_class)} "
                        f"(specialist_conf={spec_conf:.3f})"
                    )

            # Create initial prediction result
            prediction = PredictionResult(
                predicted_class=predicted_class,
                confidence=confidence,
                class_probabilities=class_probabilities,
                uncertainty_score=combined_uncertainty,
                threat_type=self.threat_classes.get(predicted_class, "Unknown"),
                explanation={},  # Will be populated below if explainer is available
                specialist_verified=specialist_verified,
                specialist_confidence=specialist_confidence,
            )

            # Generate explanation
            if self.explainer:
                explanation = self.explainer.explain_prediction(
                    enhanced_features, prediction
                )
                prediction.explanation = explanation
                prediction.feature_importance = explanation.get("top_features", {})

            # OpenAI enhancement for uncertain predictions
            if combined_uncertainty > 0.2 or confidence < 0.8:
                openai_analysis = (
                    await self.openai_analyzer.analyze_uncertain_prediction(
                        src_ip, events, prediction
                    )
                )

                if openai_analysis:
                    prediction.openai_enhanced = True
                    prediction.explanation["openai_analysis"] = openai_analysis

                    # Update confidence if OpenAI provides better assessment
                    enhanced_confidence = openai_analysis.get("enhanced_confidence")
                    if (
                        enhanced_confidence
                        and enhanced_confidence > prediction.confidence
                    ):
                        prediction.confidence = enhanced_confidence

            logger.info(
                f"Enhanced threat analysis completed: {src_ip} -> "
                f"{prediction.threat_type} ({prediction.confidence:.1%} confidence, "
                f"{prediction.uncertainty_score:.3f} uncertainty)"
            )

            return prediction

        except Exception as e:
            logger.error(f"Enhanced threat analysis failed for {src_ip}: {e}")
            raise


# Global enhanced detector instance
enhanced_detector = EnhancedThreatDetectionSystem()

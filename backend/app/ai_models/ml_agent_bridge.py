"""
ML-Agent Bridge for Revolutionary XDR

Integrates the FT-Transformer ensemble with the existing agent orchestrator:
- Routes high-uncertainty predictions to PredictiveHunter agent
- Feeds ML confidence directly to AdvancedCoordinationHub
- Uses existing conflict resolution for ML model disagreements
- Injects ML prediction details into agent prompts
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models" / "revolutionary"


@dataclass
class MLPredictionContext:
    """Context from ML prediction for agent routing and decisions."""

    # Primary prediction
    predicted_class: int
    threat_type: str
    confidence: float
    uncertainty: float

    # Class probabilities
    class_probabilities: Dict[int, float]

    # Model agreement
    models_agree: bool
    disagreeing_models: List[str]

    # Feature importance (top features)
    important_features: Dict[str, float]

    # Routing decision
    requires_agent_review: bool
    suggested_agents: List[str]

    # Timing
    prediction_time_ms: float
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for agent prompts."""
        return {
            "predicted_class": self.predicted_class,
            "threat_type": self.threat_type,
            "confidence": self.confidence,
            "uncertainty": self.uncertainty,
            "class_probabilities": self.class_probabilities,
            "models_agree": self.models_agree,
            "disagreeing_models": self.disagreeing_models,
            "important_features": self.important_features,
            "requires_agent_review": self.requires_agent_review,
            "suggested_agents": self.suggested_agents,
            "prediction_time_ms": self.prediction_time_ms,
            "timestamp": self.timestamp,
        }


class MLAgentBridge:
    """
    Bridge between ML models and AI agents.

    Responsibilities:
    1. Run ML ensemble predictions
    2. Analyze model agreement and uncertainty
    3. Route uncertain cases to appropriate agents
    4. Provide ML context to agent prompts
    5. Learn from agent decisions for model improvement
    """

    # Class names
    CLASS_NAMES = {
        0: "Normal",
        1: "DDoS",
        2: "Reconnaissance",
        3: "Brute Force",
        4: "Web Attack",
        5: "Malware",
        6: "APT",
    }

    # Agent routing thresholds
    UNCERTAINTY_THRESHOLD = 0.3  # Route to agents if uncertainty > this
    DISAGREEMENT_THRESHOLD = 0.2  # Route if model disagreement > this

    # Agent specializations by threat type
    THREAT_AGENT_MAPPING = {
        "DDoS": ["containment", "predictive_hunter"],
        "Reconnaissance": ["attribution", "deception"],
        "Brute Force": ["iam", "containment", "forensics"],
        "Web Attack": ["forensics", "containment"],
        "Malware": ["edr", "forensics", "containment"],
        "APT": ["attribution", "forensics", "predictive_hunter", "containment"],
        "Normal": [],
    }

    def __init__(self):
        self.ensemble = None
        self.scaler = None
        self._initialized = False

        # Performance tracking
        self.stats = {
            "predictions_made": 0,
            "agent_routing_count": 0,
            "model_disagreements": 0,
            "average_confidence": 0.0,
            "average_uncertainty": 0.0,
        }

    async def initialize(self):
        """Initialize the ML models."""
        if self._initialized:
            return

        try:
            # Try to load the revolutionary ensemble
            from .ensemble import EnsembleConfig, EnsembleDetector

            config = EnsembleConfig(
                use_uncertainty_weighting=True,
                min_confidence_threshold=0.3,
            )

            self.ensemble = EnsembleDetector(config=config)
            self.ensemble.load_models()

            # Load scaler if available
            scaler_path = MODELS_DIR / "scaler.pkl"
            if scaler_path.exists():
                import pickle

                with open(scaler_path, "rb") as f:
                    self.scaler = pickle.load(f)
                logger.info("Loaded feature scaler")

            self._initialized = True
            logger.info("MLAgentBridge initialized with revolutionary ensemble")

        except Exception as e:
            logger.warning(f"Failed to initialize revolutionary ensemble: {e}")
            logger.info("Will fall back to existing detection system")

    async def get_ml_context(
        self,
        features: np.ndarray,
        event_sequence: Optional[np.ndarray] = None,
    ) -> MLPredictionContext:
        """
        Get comprehensive ML prediction context for agents.

        Args:
            features: (batch_size, 79) feature array
            event_sequence: Optional (batch_size, seq_len, 79) for temporal analysis

        Returns:
            MLPredictionContext with full prediction details
        """
        start_time = datetime.now(timezone.utc)

        # Ensure initialized
        if not self._initialized:
            await self.initialize()

        # Scale features if scaler available
        if self.scaler is not None:
            features = self.scaler.transform(features)

        # Get ensemble prediction
        if self.ensemble is not None:
            result = await self.ensemble.predict(features, event_sequence)
        else:
            # Fallback to basic prediction
            result = await self._fallback_prediction(features)

        # Analyze model agreement
        models_agree, disagreeing_models = self._analyze_model_agreement(
            result.get("model_results", {})
        )

        # Calculate routing decision
        requires_review, suggested_agents = self._determine_agent_routing(
            result, models_agree
        )

        # Extract feature importance
        important_features = {}
        if self.ensemble is not None:
            try:
                explanation = self.ensemble.get_explanation(features)
                if "ft_transformer_attention" in explanation:
                    attention = explanation["ft_transformer_attention"]
                    # Get top 10 features by importance
                    indices = np.argsort(attention[0])[-10:][::-1]
                    for idx in indices:
                        important_features[f"feature_{idx}"] = float(attention[0][idx])
            except Exception as e:
                logger.debug(f"Failed to get feature importance: {e}")

        # Calculate timing
        prediction_time = (
            datetime.now(timezone.utc) - start_time
        ).total_seconds() * 1000

        # Update stats
        self._update_stats(result, requires_review, models_agree)

        # Get predicted values
        predicted_class = int(result["predicted_class"][0])
        threat_type = result.get(
            "threat_type", [self.CLASS_NAMES.get(predicted_class, "Unknown")]
        )[0]
        confidence = float(result["confidence"][0])
        uncertainty = float(result["uncertainty"][0])

        # Build class probabilities dict
        probs = result.get("class_probabilities", np.zeros((1, 7)))[0]
        class_probabilities = {i: float(p) for i, p in enumerate(probs)}

        return MLPredictionContext(
            predicted_class=predicted_class,
            threat_type=threat_type,
            confidence=confidence,
            uncertainty=uncertainty,
            class_probabilities=class_probabilities,
            models_agree=models_agree,
            disagreeing_models=disagreeing_models,
            important_features=important_features,
            requires_agent_review=requires_review,
            suggested_agents=suggested_agents,
            prediction_time_ms=prediction_time,
            timestamp=start_time.isoformat(),
        )

    async def _fallback_prediction(self, features: np.ndarray) -> Dict[str, Any]:
        """Fallback prediction when ensemble not available."""
        # Use existing enhanced threat detector
        try:
            from ..enhanced_threat_detector import enhanced_detector

            if enhanced_detector.model:
                import torch

                device = next(enhanced_detector.model.parameters()).device
                x = torch.tensor(features, dtype=torch.float32, device=device)

                with torch.no_grad():
                    output = enhanced_detector.model(x)

                probs = output["probs"].cpu().numpy()
                predicted_class = np.argmax(probs, axis=1)
                confidence = np.max(probs, axis=1)

                return {
                    "predicted_class": predicted_class,
                    "threat_type": [
                        self.CLASS_NAMES.get(c, "Unknown") for c in predicted_class
                    ],
                    "class_probabilities": probs,
                    "confidence": confidence,
                    "uncertainty": 1.0 - confidence,
                    "model_results": {},
                }
        except Exception as e:
            logger.warning(f"Fallback prediction failed: {e}")

        # Ultimate fallback - return normal
        return {
            "predicted_class": np.array([0]),
            "threat_type": ["Normal"],
            "class_probabilities": np.array([[1.0] + [0.0] * 6]),
            "confidence": np.array([0.5]),
            "uncertainty": np.array([0.5]),
            "model_results": {},
        }

    def _analyze_model_agreement(
        self,
        model_results: Dict[str, Dict[str, Any]],
    ) -> Tuple[bool, List[str]]:
        """Analyze agreement between ensemble models."""
        if not model_results or len(model_results) < 2:
            return True, []

        predictions = []
        for model_name, result in model_results.items():
            if "predicted_class" in result:
                predictions.append((model_name, result["predicted_class"][0]))

        if len(predictions) < 2:
            return True, []

        # Check if all models agree
        first_pred = predictions[0][1]
        disagreeing = [name for name, pred in predictions if pred != first_pred]

        models_agree = len(disagreeing) == 0

        if not models_agree:
            self.stats["model_disagreements"] += 1

        return models_agree, disagreeing

    def _determine_agent_routing(
        self,
        result: Dict[str, Any],
        models_agree: bool,
    ) -> Tuple[bool, List[str]]:
        """Determine if prediction should be routed to agents."""
        uncertainty = result.get("uncertainty", np.array([0.0]))[0]
        confidence = result.get("confidence", np.array([1.0]))[0]
        threat_type = result.get("threat_type", ["Normal"])[0]

        requires_review = False
        suggested_agents = []

        # Route if high uncertainty
        if uncertainty > self.UNCERTAINTY_THRESHOLD:
            requires_review = True
            suggested_agents.append("predictive_hunter")

        # Route if models disagree
        if not models_agree:
            requires_review = True
            suggested_agents.append("predictive_hunter")
            suggested_agents.append("attribution")

        # Route based on threat type
        if threat_type in self.THREAT_AGENT_MAPPING:
            type_agents = self.THREAT_AGENT_MAPPING[threat_type]

            # For critical threats, always suggest containment
            if threat_type in ["Malware", "APT"]:
                if "containment" not in suggested_agents:
                    suggested_agents.append("containment")
                requires_review = True

            # Add specialized agents
            for agent in type_agents:
                if agent not in suggested_agents:
                    suggested_agents.append(agent)

        # Limit to top 3 agents
        suggested_agents = suggested_agents[:3]

        if requires_review:
            self.stats["agent_routing_count"] += 1

        return requires_review, suggested_agents

    def _update_stats(
        self,
        result: Dict[str, Any],
        requires_review: bool,
        models_agree: bool,
    ):
        """Update bridge statistics."""
        self.stats["predictions_made"] += 1

        confidence = result.get("confidence", np.array([0.0]))[0]
        uncertainty = result.get("uncertainty", np.array([0.0]))[0]

        # Running average of confidence
        n = self.stats["predictions_made"]
        self.stats["average_confidence"] = (
            self.stats["average_confidence"] * (n - 1) + confidence
        ) / n
        self.stats["average_uncertainty"] = (
            self.stats["average_uncertainty"] * (n - 1) + uncertainty
        ) / n

    def get_coordination_input(
        self,
        ml_context: MLPredictionContext,
    ) -> Dict[str, Any]:
        """
        Format ML context for AdvancedCoordinationHub input.

        This is used by the coordination hub to make decisions
        about agent selection and execution strategy.
        """
        return {
            "ml_prediction": {
                "threat_type": ml_context.threat_type,
                "confidence": ml_context.confidence,
                "uncertainty": ml_context.uncertainty,
            },
            "model_agreement": {
                "agree": ml_context.models_agree,
                "disagreeing_models": ml_context.disagreeing_models,
            },
            "routing": {
                "requires_review": ml_context.requires_agent_review,
                "suggested_agents": ml_context.suggested_agents,
            },
            "confidence_score": ml_context.confidence,
            "risk_score": 1.0 - ml_context.confidence + ml_context.uncertainty,
        }

    def get_langchain_context(
        self,
        ml_context: MLPredictionContext,
    ) -> str:
        """
        Format ML context for LangChain agent prompt injection.

        Returns a formatted string to include in agent prompts.
        """
        # Build context string
        lines = [
            "## ML Analysis Results",
            f"- **Predicted Threat**: {ml_context.threat_type}",
            f"- **Confidence**: {ml_context.confidence:.1%}",
            f"- **Uncertainty**: {ml_context.uncertainty:.1%}",
        ]

        if not ml_context.models_agree:
            lines.append(
                f"- **Model Disagreement**: {', '.join(ml_context.disagreeing_models)}"
            )

        # Class probabilities
        lines.append("\n### Class Probabilities:")
        for cls, prob in sorted(
            ml_context.class_probabilities.items(), key=lambda x: -x[1]
        )[:3]:
            cls_name = self.CLASS_NAMES.get(cls, f"Class {cls}")
            lines.append(f"  - {cls_name}: {prob:.1%}")

        # Important features
        if ml_context.important_features:
            lines.append("\n### Key Indicators (Feature Importance):")
            for feat, imp in list(ml_context.important_features.items())[:5]:
                lines.append(f"  - {feat}: {imp:.3f}")

        # Routing recommendation
        if ml_context.requires_agent_review:
            lines.append(
                f"\n### Recommended Agents: {', '.join(ml_context.suggested_agents)}"
            )

        return "\n".join(lines)

    async def record_agent_decision(
        self,
        ml_context: MLPredictionContext,
        agent_decision: Dict[str, Any],
        final_verdict: str,
    ):
        """
        Record agent decision for future model improvement.

        This creates training samples for the continuous learning pipeline.
        """
        try:
            from ..learning.training_collector import training_collector

            # Create training sample
            sample = {
                "ml_prediction": ml_context.threat_type,
                "ml_confidence": ml_context.confidence,
                "ml_uncertainty": ml_context.uncertainty,
                "agent_verdict": final_verdict,
                "agent_reasoning": agent_decision.get("reasoning", ""),
                "timestamp": ml_context.timestamp,
                "requires_retraining": final_verdict != ml_context.threat_type,
            }

            # If agent disagreed with ML, this is valuable training data
            if final_verdict != ml_context.threat_type:
                logger.info(
                    f"Agent override recorded: ML said '{ml_context.threat_type}' "
                    f"but agent decided '{final_verdict}'"
                )

            await training_collector.record_decision_sample(sample)

        except Exception as e:
            logger.debug(f"Failed to record agent decision: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get bridge statistics."""
        return {
            **self.stats,
            "ensemble_available": self.ensemble is not None,
            "scaler_available": self.scaler is not None,
        }


# Singleton instance
ml_agent_bridge: Optional[MLAgentBridge] = None


async def get_ml_agent_bridge() -> MLAgentBridge:
    """Get or create ML-Agent bridge singleton."""
    global ml_agent_bridge

    if ml_agent_bridge is None:
        ml_agent_bridge = MLAgentBridge()
        await ml_agent_bridge.initialize()

    return ml_agent_bridge

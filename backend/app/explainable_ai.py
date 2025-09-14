"""
Explainable AI Integration with SHAP, LIME, and OpenAI
Provides comprehensive model explainability for threat detection systems
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import json
import base64
from concurrent.futures import ThreadPoolExecutor
import threading
from collections import defaultdict, deque

# Core ML libraries
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Explainable AI libraries (optional imports)
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    shap = None
    logging.warning("SHAP not available - install with: pip install shap")

try:
    import lime
    import lime.lime_tabular
    import lime.lime_text
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    lime = None
    logging.warning("LIME not available - install with: pip install lime")

# OpenAI integration
import openai
from .config import settings

# Local imports
from .models import Event, Incident
from .model_versioning import model_registry

logger = logging.getLogger(__name__)


class ExplanationScope(str, Enum):
    """Scope of explanation"""
    LOCAL = "local"        # Single prediction
    GLOBAL = "global"      # Overall model behavior
    COHORT = "cohort"      # Group of similar instances
    TEMPORAL = "temporal"   # Time-based patterns


class ExplanationType(str, Enum):
    """Type of explanation"""
    FEATURE_ATTRIBUTION = "feature_attribution"
    COUNTERFACTUAL = "counterfactual"
    EXAMPLE_BASED = "example_based"
    RULE_BASED = "rule_based"
    NARRATIVE = "narrative"


@dataclass
class ExplanationRequest:
    """Request for model explanation"""
    model_id: str
    model_version: str
    instance_data: Union[np.ndarray, Dict[str, Any]]
    feature_names: List[str]
    explanation_scope: ExplanationScope
    explanation_types: List[ExplanationType]
    user_context: Optional[Dict[str, Any]] = None
    confidence_threshold: float = 0.1
    max_features: int = 10


@dataclass
class FeatureAttribution:
    """Feature attribution result"""
    feature_name: str
    feature_value: Union[float, str]
    attribution_score: float
    confidence: float
    direction: str  # 'positive', 'negative', 'neutral'
    description: str


@dataclass
class CounterfactualExplanation:
    """Counterfactual explanation result"""
    original_prediction: Any
    counterfactual_prediction: Any
    feature_changes: Dict[str, Tuple[Any, Any]]  # feature -> (original, modified)
    change_summary: str
    feasibility_score: float
    distance_score: float


@dataclass 
class ExplanationResult:
    """Complete explanation result"""
    explanation_id: str
    model_id: str
    model_version: str
    timestamp: datetime
    prediction: Any
    confidence: float
    
    # Attribution explanations
    feature_attributions: List[FeatureAttribution]
    
    # Counterfactual explanations
    counterfactuals: List[CounterfactualExplanation]
    
    # Example-based explanations
    similar_examples: List[Dict[str, Any]]
    
    # Rule-based explanations
    decision_rules: List[str]
    
    # Narrative explanations
    narrative_summary: str
    technical_details: str
    
    # Metadata
    explanation_methods: List[str]
    processing_time: float
    explanation_quality: float


class SHAPExplainer:
    """SHAP-based explainer"""
    
    def __init__(self):
        self.explainers = {}  # model_id -> explainer
        self.background_data = {}  # model_id -> background dataset
        
    def initialize_explainer(self, model_id: str, model: BaseEstimator, 
                           background_data: np.ndarray, explainer_type: str = "auto"):
        """Initialize SHAP explainer for a model"""
        
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is not available")
        
        try:
            if explainer_type == "tree" or (explainer_type == "auto" and hasattr(model, 'estimators_')):
                # Tree-based explainer for ensemble models
                if hasattr(model, 'estimators_'):
                    self.explainers[model_id] = shap.TreeExplainer(model)
                else:
                    # Try TreeExplainer for single tree models
                    try:
                        self.explainers[model_id] = shap.TreeExplainer(model)
                    except:
                        # Fallback to KernelExplainer
                        self.explainers[model_id] = shap.KernelExplainer(
                            model.predict_proba if hasattr(model, 'predict_proba') else model.predict,
                            background_data[:min(100, len(background_data))]
                        )
            
            elif explainer_type == "linear" or (explainer_type == "auto" and hasattr(model, 'coef_')):
                # Linear explainer
                self.explainers[model_id] = shap.LinearExplainer(model, background_data)
            
            elif explainer_type == "deep" or (explainer_type == "auto" and hasattr(model, 'layers')):
                # Deep learning explainer (for neural networks)
                self.explainers[model_id] = shap.DeepExplainer(model, background_data)
            
            else:
                # Default to KernelExplainer (model-agnostic)
                self.explainers[model_id] = shap.KernelExplainer(
                    model.predict_proba if hasattr(model, 'predict_proba') else model.predict,
                    background_data[:min(100, len(background_data))]
                )
            
            self.background_data[model_id] = background_data
            logger.info(f"Initialized SHAP explainer for model {model_id}")
            
        except Exception as e:
            logger.error(f"Failed to initialize SHAP explainer for {model_id}: {e}")
            raise
    
    def explain_instance(self, model_id: str, instance: np.ndarray,
                        feature_names: List[str]) -> List[FeatureAttribution]:
        """Generate SHAP explanation for a single instance"""
        
        if model_id not in self.explainers:
            raise ValueError(f"No SHAP explainer found for model {model_id}")
        
        try:
            explainer = self.explainers[model_id]
            
            # Get SHAP values
            shap_values = explainer.shap_values(instance.reshape(1, -1))
            
            # Handle different SHAP value formats
            if isinstance(shap_values, list):
                # Multi-class: take positive class for binary classification
                if len(shap_values) == 2:
                    shap_values = shap_values[1]
                else:
                    # For multi-class, use the class with highest prediction
                    predicted_class = np.argmax(np.sum([sv for sv in shap_values], axis=0))
                    shap_values = shap_values[predicted_class]
            
            # Convert to 1D array
            if shap_values.ndim > 1:
                shap_values = shap_values.flatten()
            
            # Create feature attributions
            attributions = []
            
            for i, (feature_name, shap_value, feature_value) in enumerate(
                zip(feature_names, shap_values, instance)
            ):
                
                # Determine direction and confidence
                abs_value = abs(shap_value)
                direction = "positive" if shap_value > 0 else "negative" if shap_value < 0 else "neutral"
                confidence = min(abs_value / (np.max(np.abs(shap_values)) + 1e-8), 1.0)
                
                # Generate description
                if abs_value > 0.01:
                    effect = "increases" if shap_value > 0 else "decreases"
                    description = f"{feature_name} (value: {feature_value:.3f}) {effect} prediction by {abs_value:.3f}"
                else:
                    description = f"{feature_name} has minimal impact on prediction"
                
                attribution = FeatureAttribution(
                    feature_name=feature_name,
                    feature_value=feature_value,
                    attribution_score=shap_value,
                    confidence=confidence,
                    direction=direction,
                    description=description
                )
                
                attributions.append(attribution)
            
            # Sort by absolute attribution score
            attributions.sort(key=lambda x: abs(x.attribution_score), reverse=True)
            
            return attributions
            
        except Exception as e:
            logger.error(f"SHAP explanation failed for model {model_id}: {e}")
            raise
    
    def explain_global(self, model_id: str, sample_data: np.ndarray,
                      feature_names: List[str], max_samples: int = 100) -> Dict[str, Any]:
        """Generate global SHAP explanations"""
        
        if model_id not in self.explainers:
            raise ValueError(f"No SHAP explainer found for model {model_id}")
        
        try:
            explainer = self.explainers[model_id]
            
            # Limit sample size for efficiency
            if len(sample_data) > max_samples:
                indices = np.random.choice(len(sample_data), max_samples, replace=False)
                sample_data = sample_data[indices]
            
            # Get SHAP values for all samples
            shap_values = explainer.shap_values(sample_data)
            
            # Handle different formats
            if isinstance(shap_values, list):
                if len(shap_values) == 2:
                    shap_values = shap_values[1]  # Positive class for binary
                else:
                    shap_values = shap_values[0]  # First class for multi-class
            
            # Calculate global feature importance
            global_importance = np.mean(np.abs(shap_values), axis=0)
            
            # Calculate feature interactions (simplified)
            feature_interactions = {}
            for i in range(min(5, len(feature_names))):  # Top 5 features
                for j in range(i + 1, min(5, len(feature_names))):
                    interaction_strength = np.corrcoef(shap_values[:, i], shap_values[:, j])[0, 1]
                    if not np.isnan(interaction_strength):
                        feature_interactions[f"{feature_names[i]}_x_{feature_names[j]}"] = float(interaction_strength)
            
            # Create summary
            global_explanation = {
                'feature_importance': {
                    feature_names[i]: float(importance) 
                    for i, importance in enumerate(global_importance)
                },
                'feature_interactions': feature_interactions,
                'samples_analyzed': len(sample_data),
                'top_features': [
                    feature_names[i] for i in np.argsort(global_importance)[-10:][::-1]
                ]
            }
            
            return global_explanation
            
        except Exception as e:
            logger.error(f"Global SHAP explanation failed for model {model_id}: {e}")
            raise


class LIMEExplainer:
    """LIME-based explainer"""
    
    def __init__(self):
        self.explainers = {}  # model_id -> explainer
        self.scalers = {}     # model_id -> scaler
    
    def initialize_explainer(self, model_id: str, model: BaseEstimator,
                           training_data: np.ndarray, feature_names: List[str]):
        """Initialize LIME explainer for a model"""
        
        if not LIME_AVAILABLE:
            raise ImportError("LIME is not available")
        
        try:
            # Create LIME tabular explainer
            self.explainers[model_id] = lime.lime_tabular.LimeTabularExplainer(
                training_data,
                feature_names=feature_names,
                class_names=['Normal', 'Threat'],  # Assuming binary classification
                mode='classification',
                discretize_continuous=True
            )
            
            logger.info(f"Initialized LIME explainer for model {model_id}")
            
        except Exception as e:
            logger.error(f"Failed to initialize LIME explainer for {model_id}: {e}")
            raise
    
    def explain_instance(self, model_id: str, model: BaseEstimator, instance: np.ndarray,
                        feature_names: List[str], num_features: int = 10) -> List[FeatureAttribution]:
        """Generate LIME explanation for a single instance"""
        
        if model_id not in self.explainers:
            raise ValueError(f"No LIME explainer found for model {model_id}")
        
        try:
            explainer = self.explainers[model_id]
            
            # Generate LIME explanation
            explanation = explainer.explain_instance(
                instance,
                model.predict_proba if hasattr(model, 'predict_proba') else model.predict,
                num_features=min(num_features, len(feature_names))
            )
            
            # Extract feature attributions for positive class
            lime_values = explanation.as_map()[1] if hasattr(explanation, 'as_map') else []
            
            # Create feature attributions
            attributions = []
            
            # Initialize with zeros for all features
            attribution_dict = {i: 0.0 for i in range(len(feature_names))}
            
            # Fill in LIME values
            for feature_idx, importance in lime_values:
                if 0 <= feature_idx < len(feature_names):
                    attribution_dict[feature_idx] = importance
            
            # Create FeatureAttribution objects
            for i, (feature_name, lime_value) in enumerate(zip(feature_names, 
                                                              [attribution_dict[i] for i in range(len(feature_names))])):
                
                feature_value = instance[i] if i < len(instance) else 0
                abs_value = abs(lime_value)
                direction = "positive" if lime_value > 0 else "negative" if lime_value < 0 else "neutral"
                
                # Confidence based on relative importance
                max_importance = max([abs(v) for v in attribution_dict.values()]) + 1e-8
                confidence = min(abs_value / max_importance, 1.0)
                
                # Description
                if abs_value > 0.01:
                    effect = "supports" if lime_value > 0 else "opposes"
                    description = f"{feature_name} (value: {feature_value:.3f}) {effect} threat prediction (LIME score: {lime_value:.3f})"
                else:
                    description = f"{feature_name} has minimal influence on prediction"
                
                attribution = FeatureAttribution(
                    feature_name=feature_name,
                    feature_value=feature_value,
                    attribution_score=lime_value,
                    confidence=confidence,
                    direction=direction,
                    description=description
                )
                
                attributions.append(attribution)
            
            # Sort by absolute attribution score
            attributions.sort(key=lambda x: abs(x.attribution_score), reverse=True)
            
            return attributions
            
        except Exception as e:
            logger.error(f"LIME explanation failed for model {model_id}: {e}")
            raise


class CounterfactualGenerator:
    """Generate counterfactual explanations"""
    
    def __init__(self):
        self.feature_constraints = {}  # model_id -> constraints
    
    def generate_counterfactuals(self, model: BaseEstimator, instance: np.ndarray,
                               feature_names: List[str], target_class: Optional[int] = None,
                               max_changes: int = 3) -> List[CounterfactualExplanation]:
        """Generate counterfactual explanations"""
        
        try:
            original_prediction = model.predict(instance.reshape(1, -1))[0]
            
            if target_class is None:
                # Find opposite class for binary classification
                target_class = 1 - original_prediction if original_prediction in [0, 1] else 0
            
            counterfactuals = []
            
            # Simple counterfactual generation: perturb top features
            if hasattr(model, 'feature_importances_'):
                feature_importance = model.feature_importances_
            else:
                # Use random importance
                feature_importance = np.random.random(len(instance))
            
            # Get top features to modify
            top_features = np.argsort(feature_importance)[-max_changes:][::-1]
            
            for num_changes in range(1, min(max_changes + 1, len(top_features) + 1)):
                # Try different combinations of feature changes
                from itertools import combinations
                
                for feature_combo in combinations(top_features, num_changes):
                    modified_instance = instance.copy()
                    feature_changes = {}
                    
                    for feature_idx in feature_combo:
                        # Simple perturbation strategy
                        original_value = instance[feature_idx]
                        
                        # Try different modifications
                        if original_value == 0:
                            new_value = 1.0
                        elif original_value > 0:
                            new_value = 0.0
                        else:
                            new_value = -original_value
                        
                        modified_instance[feature_idx] = new_value
                        feature_changes[feature_names[feature_idx]] = (original_value, new_value)
                    
                    # Check if this counterfactual achieves target class
                    new_prediction = model.predict(modified_instance.reshape(1, -1))[0]
                    
                    if new_prediction == target_class:
                        # Calculate distance and feasibility
                        distance = np.linalg.norm(instance - modified_instance)
                        feasibility = 1.0 / (1.0 + distance)  # Higher for smaller changes
                        
                        change_summary = f"Changed {len(feature_changes)} features: {', '.join(feature_changes.keys())}"
                        
                        counterfactual = CounterfactualExplanation(
                            original_prediction=original_prediction,
                            counterfactual_prediction=new_prediction,
                            feature_changes=feature_changes,
                            change_summary=change_summary,
                            feasibility_score=feasibility,
                            distance_score=distance
                        )
                        
                        counterfactuals.append(counterfactual)
                        
                        # Limit number of counterfactuals
                        if len(counterfactuals) >= 3:
                            break
                
                if len(counterfactuals) >= 3:
                    break
            
            # Sort by feasibility
            counterfactuals.sort(key=lambda x: x.feasibility_score, reverse=True)
            
            return counterfactuals
            
        except Exception as e:
            logger.error(f"Counterfactual generation failed: {e}")
            return []


class NarrativeExplainer:
    """Generate narrative explanations using OpenAI"""
    
    def __init__(self):
        self.openai_client = None
        if hasattr(settings, 'OPENAI_API_KEY') and settings.OPENAI_API_KEY:
            openai.api_key = settings.OPENAI_API_KEY
            self.openai_client = openai
        
        self.explanation_cache = {}
        self.max_cache_size = 1000
    
    async def generate_narrative(self, prediction: Any, feature_attributions: List[FeatureAttribution],
                               counterfactuals: List[CounterfactualExplanation],
                               context: Dict[str, Any]) -> Tuple[str, str]:
        """Generate narrative explanation using AI"""
        
        if not self.openai_client:
            return self._generate_fallback_narrative(prediction, feature_attributions, counterfactuals)
        
        try:
            # Create cache key
            cache_key = hash(str(prediction) + str([attr.feature_name for attr in feature_attributions[:5]]))
            
            if cache_key in self.explanation_cache:
                return self.explanation_cache[cache_key]
            
            # Prepare explanation context
            top_features = feature_attributions[:5]
            feature_summary = []
            
            for attr in top_features:
                impact = "increases" if attr.direction == "positive" else "decreases"
                feature_summary.append(
                    f"- {attr.feature_name}: {impact} threat likelihood by {abs(attr.attribution_score):.3f} "
                    f"(current value: {attr.feature_value:.3f})"
                )
            
            counterfactual_summary = ""
            if counterfactuals:
                cf = counterfactuals[0]  # Use best counterfactual
                changes = [f"{k}: {v[0]:.3f} → {v[1]:.3f}" for k, v in list(cf.feature_changes.items())[:3]]
                counterfactual_summary = f"\nTo change the prediction from {cf.original_prediction} to {cf.counterfactual_prediction}, " \
                                       f"you would need to modify: {', '.join(changes)}"
            
            # Create prompts
            system_prompt = """You are an expert cybersecurity analyst explaining AI-driven threat detection decisions. 
            Provide clear, actionable explanations that help security analysts understand and trust the AI's reasoning.
            Focus on security implications and practical next steps."""
            
            user_prompt = f"""
            The AI model predicted: {prediction} (0=Normal, 1=Threat)
            
            Key factors influencing this decision:
            {chr(10).join(feature_summary)}
            
            {counterfactual_summary}
            
            Context: {context.get('incident_type', 'Unknown')} event from {context.get('source_ip', 'Unknown IP')}
            
            Please provide:
            1. Summary: A brief explanation of the decision (2-3 sentences)
            2. Technical Details: Detailed analysis including security implications and recommended actions (4-5 sentences)
            
            Format as:
            SUMMARY: [brief explanation]
            TECHNICAL: [detailed analysis]
            """
            
            # Get AI explanation
            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=400,
                temperature=0.3
            )
            
            ai_response = response.choices[0].message.content
            
            # Parse response
            if "TECHNICAL:" in ai_response:
                summary_part = ai_response.split("TECHNICAL:")[0].replace("SUMMARY:", "").strip()
                technical_part = ai_response.split("TECHNICAL:")[1].strip()
            else:
                # Fallback parsing
                parts = ai_response.split("\n\n")
                summary_part = parts[0] if parts else ai_response[:200]
                technical_part = "\n".join(parts[1:]) if len(parts) > 1 else ai_response[200:]
            
            # Cache result
            if len(self.explanation_cache) < self.max_cache_size:
                self.explanation_cache[cache_key] = (summary_part, technical_part)
            
            return summary_part, technical_part
            
        except Exception as e:
            logger.error(f"AI narrative generation failed: {e}")
            return self._generate_fallback_narrative(prediction, feature_attributions, counterfactuals)
    
    def _generate_fallback_narrative(self, prediction: Any, feature_attributions: List[FeatureAttribution],
                                   counterfactuals: List[CounterfactualExplanation]) -> Tuple[str, str]:
        """Generate fallback narrative without AI"""
        
        # Simple rule-based explanation
        threat_level = "High" if prediction == 1 else "Low"
        
        top_features = feature_attributions[:3]
        key_factors = [attr.feature_name for attr in top_features if abs(attr.attribution_score) > 0.1]
        
        summary = f"{threat_level} threat level detected. Key contributing factors: {', '.join(key_factors[:3])}."
        
        technical = f"The model analyzed {len(feature_attributions)} features and identified " \
                   f"{len(key_factors)} significant risk factors. "
        
        if top_features:
            strongest_factor = top_features[0]
            impact = "increases" if strongest_factor.direction == "positive" else "decreases"
            technical += f"The strongest factor is {strongest_factor.feature_name}, which {impact} " \
                        f"threat probability by {abs(strongest_factor.attribution_score):.3f}. "
        
        if counterfactuals:
            cf = counterfactuals[0]
            technical += f"To change the prediction, modify {len(cf.feature_changes)} key features: " \
                        f"{', '.join(list(cf.feature_changes.keys())[:2])}."
        
        return summary, technical


class ExplainableAIOrchestrator:
    """Main orchestrator for explainable AI functionality"""
    
    def __init__(self):
        self.shap_explainer = SHAPExplainer()
        self.lime_explainer = LIMEExplainer()
        self.counterfactual_generator = CounterfactualGenerator()
        self.narrative_explainer = NarrativeExplainer()
        
        self.initialized_models = set()
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="explainable_ai")
        
        logger.info("Explainable AI Orchestrator initialized")
    
    def initialize_for_model(self, model_id: str, model: BaseEstimator,
                           training_data: np.ndarray, feature_names: List[str]):
        """Initialize explainers for a specific model"""
        
        if model_id in self.initialized_models:
            logger.info(f"Model {model_id} already initialized for explainability")
            return
        
        try:
            # Initialize SHAP explainer
            if SHAP_AVAILABLE:
                self.shap_explainer.initialize_explainer(model_id, model, training_data)
            
            # Initialize LIME explainer
            if LIME_AVAILABLE:
                self.lime_explainer.initialize_explainer(model_id, model, training_data, feature_names)
            
            self.initialized_models.add(model_id)
            logger.info(f"Initialized explainable AI for model {model_id}")
            
        except Exception as e:
            logger.error(f"Failed to initialize explainable AI for model {model_id}: {e}")
            raise
    
    async def explain_prediction(self, request: ExplanationRequest) -> ExplanationResult:
        """Generate comprehensive explanation for a prediction"""
        
        start_time = datetime.now()
        
        try:
            # Load model
            model = model_registry.load_model(request.model_id, request.model_version)
            if not model:
                raise ValueError(f"Model {request.model_id} v{request.model_version} not found")
            
            # Prepare instance data
            if isinstance(request.instance_data, dict):
                # Convert dict to array based on feature names
                instance = np.array([request.instance_data.get(name, 0.0) for name in request.feature_names])
            else:
                instance = request.instance_data.flatten()
            
            # Make prediction
            prediction = model.predict(instance.reshape(1, -1))[0]
            confidence = 0.5  # Default confidence
            
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(instance.reshape(1, -1))[0]
                confidence = float(np.max(proba))
            
            # Initialize explainers if needed
            if request.model_id not in self.initialized_models:
                # Use a subset of training data (would need to be provided or stored)
                dummy_training_data = np.random.random((100, len(request.feature_names)))
                self.initialize_for_model(request.model_id, model, dummy_training_data, request.feature_names)
            
            # Generate feature attributions
            feature_attributions = []
            explanation_methods = []
            
            # SHAP explanations
            if (SHAP_AVAILABLE and ExplanationType.FEATURE_ATTRIBUTION in request.explanation_types and
                request.model_id in self.shap_explainer.explainers):
                try:
                    shap_attributions = self.shap_explainer.explain_instance(
                        request.model_id, instance, request.feature_names
                    )
                    feature_attributions.extend(shap_attributions[:request.max_features])
                    explanation_methods.append("SHAP")
                except Exception as e:
                    logger.warning(f"SHAP explanation failed: {e}")
            
            # LIME explanations
            if (LIME_AVAILABLE and ExplanationType.FEATURE_ATTRIBUTION in request.explanation_types and
                request.model_id in self.lime_explainer.explainers):
                try:
                    lime_attributions = self.lime_explainer.explain_instance(
                        request.model_id, model, instance, request.feature_names, request.max_features
                    )
                    # Combine with SHAP or use if SHAP unavailable
                    if not feature_attributions:
                        feature_attributions = lime_attributions[:request.max_features]
                        explanation_methods.append("LIME")
                except Exception as e:
                    logger.warning(f"LIME explanation failed: {e}")
            
            # Fallback to feature importance
            if not feature_attributions:
                feature_attributions = self._generate_feature_importance_attribution(
                    model, instance, request.feature_names
                )
                explanation_methods.append("Feature_Importance")
            
            # Generate counterfactuals
            counterfactuals = []
            if ExplanationType.COUNTERFACTUAL in request.explanation_types:
                try:
                    counterfactuals = self.counterfactual_generator.generate_counterfactuals(
                        model, instance, request.feature_names
                    )
                    if counterfactuals:
                        explanation_methods.append("Counterfactual")
                except Exception as e:
                    logger.warning(f"Counterfactual generation failed: {e}")
            
            # Generate narrative explanations
            narrative_summary = ""
            technical_details = ""
            
            if ExplanationType.NARRATIVE in request.explanation_types:
                try:
                    context = request.user_context or {}
                    narrative_summary, technical_details = await self.narrative_explainer.generate_narrative(
                        prediction, feature_attributions, counterfactuals, context
                    )
                    explanation_methods.append("AI_Narrative")
                except Exception as e:
                    logger.warning(f"Narrative generation failed: {e}")
                    narrative_summary = f"Prediction: {prediction} with {confidence:.2f} confidence"
                    technical_details = "Technical analysis not available"
            
            # Generate similar examples (simplified)
            similar_examples = self._find_similar_examples(instance, request.feature_names)
            
            # Generate decision rules (simplified)
            decision_rules = self._generate_decision_rules(model, instance, request.feature_names)
            
            # Calculate explanation quality
            explanation_quality = self._calculate_explanation_quality(
                feature_attributions, counterfactuals, confidence
            )
            
            # Create result
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = ExplanationResult(
                explanation_id=f"exp_{int(datetime.now().timestamp())}_{hash(str(instance))%10000}",
                model_id=request.model_id,
                model_version=request.model_version,
                timestamp=datetime.now(timezone.utc),
                prediction=prediction,
                confidence=confidence,
                feature_attributions=feature_attributions,
                counterfactuals=counterfactuals,
                similar_examples=similar_examples,
                decision_rules=decision_rules,
                narrative_summary=narrative_summary,
                technical_details=technical_details,
                explanation_methods=explanation_methods,
                processing_time=processing_time,
                explanation_quality=explanation_quality
            )
            
            logger.info(f"Generated explanation {result.explanation_id} for model {request.model_id}")
            return result
            
        except Exception as e:
            logger.error(f"Explanation generation failed: {e}")
            raise
    
    def _generate_feature_importance_attribution(self, model: BaseEstimator, instance: np.ndarray,
                                               feature_names: List[str]) -> List[FeatureAttribution]:
        """Generate attribution based on model feature importance"""
        
        attributions = []
        
        # Get feature importance
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_).flatten()
        else:
            # Random importance as fallback
            importance = np.random.random(len(feature_names))
        
        # Weight by instance values
        weighted_importance = importance * np.abs(instance)
        
        for i, (feature_name, weight, value) in enumerate(zip(feature_names, weighted_importance, instance)):
            
            direction = "positive" if weight > 0 else "neutral"
            confidence = min(weight / (np.max(weighted_importance) + 1e-8), 1.0)
            
            description = f"{feature_name} (value: {value:.3f}) contributes {weight:.3f} to prediction"
            
            attribution = FeatureAttribution(
                feature_name=feature_name,
                feature_value=value,
                attribution_score=weight,
                confidence=confidence,
                direction=direction,
                description=description
            )
            
            attributions.append(attribution)
        
        # Sort by importance
        attributions.sort(key=lambda x: abs(x.attribution_score), reverse=True)
        return attributions
    
    def _find_similar_examples(self, instance: np.ndarray, feature_names: List[str]) -> List[Dict[str, Any]]:
        """Find similar examples (simplified implementation)"""
        
        # This would typically use a similarity search in a database or vector store
        # For now, return empty list
        return []
    
    def _generate_decision_rules(self, model: BaseEstimator, instance: np.ndarray,
                               feature_names: List[str]) -> List[str]:
        """Generate decision rules (simplified)"""
        
        rules = []
        
        # Simple rule generation based on feature values
        for i, (feature_name, value) in enumerate(zip(feature_names, instance)):
            if value > 0.5:  # Arbitrary threshold
                rules.append(f"IF {feature_name} > 0.5 THEN threat_likelihood += 0.1")
        
        return rules[:5]  # Return top 5 rules
    
    def _calculate_explanation_quality(self, feature_attributions: List[FeatureAttribution],
                                     counterfactuals: List[CounterfactualExplanation],
                                     confidence: float) -> float:
        """Calculate quality score for explanation"""
        
        quality_factors = []
        
        # Factor 1: Number of significant features
        significant_features = len([attr for attr in feature_attributions if abs(attr.attribution_score) > 0.1])
        feature_quality = min(significant_features / 5.0, 1.0)
        quality_factors.append(feature_quality)
        
        # Factor 2: Counterfactual availability
        counterfactual_quality = min(len(counterfactuals) / 2.0, 1.0)
        quality_factors.append(counterfactual_quality)
        
        # Factor 3: Prediction confidence
        quality_factors.append(confidence)
        
        # Average quality
        return sum(quality_factors) / len(quality_factors)


# Global instance
explainable_ai = ExplainableAIOrchestrator()


async def explain_threat_prediction(model_id: str, model_version: str, 
                                  instance_data: Union[np.ndarray, Dict[str, Any]],
                                  feature_names: List[str],
                                  user_context: Optional[Dict[str, Any]] = None) -> ExplanationResult:
    """High-level function to explain threat predictions"""
    
    request = ExplanationRequest(
        model_id=model_id,
        model_version=model_version,
        instance_data=instance_data,
        feature_names=feature_names,
        explanation_scope=ExplanationScope.LOCAL,
        explanation_types=[
            ExplanationType.FEATURE_ATTRIBUTION,
            ExplanationType.COUNTERFACTUAL,
            ExplanationType.NARRATIVE
        ],
        user_context=user_context
    )
    
    return await explainable_ai.explain_prediction(request)


if __name__ == "__main__":
    # Test explainable AI system
    import asyncio
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    
    async def test_explainable_ai():
        print("Testing Explainable AI System...")
        
        # Generate test data
        X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
        feature_names = [f'feature_{i}' for i in range(10)]
        
        # Train model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        # Register model (simplified)
        model_registry.register_model(
            model=model,
            model_id="test_threat_model",
            version="v1.0",
            model_type="classifier",
            algorithm="random_forest",
            hyperparameters={},
            training_data_hash="test",
            description="Test model"
        )
        
        # Initialize explainable AI
        explainable_ai.initialize_for_model("test_threat_model", model, X[:100], feature_names)
        
        # Test prediction explanation
        test_instance = X[0]
        context = {
            'incident_type': 'suspicious_login',
            'source_ip': '192.168.1.100'
        }
        
        explanation = await explain_threat_prediction(
            model_id="test_threat_model",
            model_version="v1.0",
            instance_data=test_instance,
            feature_names=feature_names,
            user_context=context
        )
        
        print(f"Explanation ID: {explanation.explanation_id}")
        print(f"Prediction: {explanation.prediction}")
        print(f"Confidence: {explanation.confidence:.3f}")
        print(f"Explanation Quality: {explanation.explanation_quality:.3f}")
        print(f"Methods Used: {explanation.explanation_methods}")
        
        print("\nTop Feature Attributions:")
        for attr in explanation.feature_attributions[:5]:
            print(f"- {attr.feature_name}: {attr.attribution_score:.3f} ({attr.direction})")
        
        if explanation.narrative_summary:
            print(f"\nSummary: {explanation.narrative_summary}")
        
        if explanation.counterfactuals:
            print(f"\nCounterfactuals: {len(explanation.counterfactuals)} found")
            for cf in explanation.counterfactuals[:2]:
                print(f"- {cf.change_summary}")
        
        print("Explainable AI system test completed!")
    
    # Run test
    asyncio.run(test_explainable_ai())

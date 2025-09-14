"""
Advanced Ensemble Techniques with Explainable AI Integration
Implements stacking, boosting, meta-learning with SHAP/LIME explanations
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import json
import warnings
import pickle
from pathlib import Path

# Core ML libraries
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, 
    AdaBoostClassifier, BaggingClassifier, VotingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb

# Hyperparameter optimization
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    optuna = None
    logging.warning("Optuna not available - hyperparameter optimization disabled")

# Explainable AI
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    shap = None
    logging.warning("SHAP not available - SHAP explanations disabled")

try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    lime = None
    logging.warning("LIME not available - LIME explanations disabled")

# OpenAI for enhanced explanations
import openai
from .config import settings

# Local imports
from .models import Event, Incident
from .model_versioning import model_registry, ModelVersion

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class EnsembleMethod(str, Enum):
    """Available ensemble methods"""
    VOTING = "voting"
    BAGGING = "bagging"  
    BOOSTING = "boosting"
    STACKING = "stacking"
    BLENDING = "blending"
    DYNAMIC_SELECTION = "dynamic_selection"


class ExplanationMethod(str, Enum):
    """Explanation methods"""
    SHAP = "shap"
    LIME = "lime"
    FEATURE_IMPORTANCE = "feature_importance"
    PERMUTATION = "permutation"
    GRADIENT = "gradient"


@dataclass
class EnsembleConfig:
    """Ensemble configuration"""
    method: EnsembleMethod
    base_models: List[str]
    meta_model: Optional[str]
    cv_folds: int
    optimization_metric: str
    hyperparameter_tuning: bool
    explanation_methods: List[ExplanationMethod]
    dynamic_weights: bool
    performance_threshold: float


@dataclass 
class ModelExplanation:
    """Model explanation result"""
    method: ExplanationMethod
    model_id: str
    instance_id: str
    feature_names: List[str]
    feature_values: List[float]
    feature_importance: List[float]
    explanation_text: str
    confidence: float
    prediction: Any
    timestamp: datetime
    global_importance: Optional[Dict[str, float]] = None
    local_importance: Optional[Dict[str, float]] = None


@dataclass
class EnsemblePerformance:
    """Ensemble performance metrics"""
    ensemble_id: str
    timestamp: datetime
    cv_accuracy: float
    cv_std: float
    individual_scores: Dict[str, float]
    ensemble_score: float
    improvement_over_best: float
    training_time: float
    prediction_time: float
    model_weights: Dict[str, float]
    hyperparameters: Dict[str, Any]
    explanation_coverage: float


class MetaLearningOptimizer:
    """Meta-learning for ensemble optimization"""
    
    def __init__(self):
        self.meta_features_cache = {}
        self.algorithm_performance = {}
        self.dataset_characteristics = {}
        
    def extract_meta_features(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Extract meta-features from dataset"""
        
        cache_key = f"{X.shape}_{hash(str(X.tobytes()))}"
        if cache_key in self.meta_features_cache:
            return self.meta_features_cache[cache_key]
        
        meta_features = {}
        
        # Dataset characteristics
        meta_features['n_samples'] = X.shape[0]
        meta_features['n_features'] = X.shape[1]
        meta_features['n_classes'] = len(np.unique(y))
        
        # Feature statistics
        meta_features['mean_feature_mean'] = np.mean(np.mean(X, axis=0))
        meta_features['mean_feature_std'] = np.mean(np.std(X, axis=0))
        meta_features['feature_correlation'] = np.mean(np.abs(np.corrcoef(X.T)))
        
        # Class imbalance
        class_counts = np.bincount(y)
        meta_features['class_imbalance'] = np.max(class_counts) / np.min(class_counts)
        
        # Sparsity
        meta_features['sparsity'] = np.sum(X == 0) / X.size
        
        # Dimensionality
        meta_features['dimensionality_ratio'] = X.shape[1] / X.shape[0]
        
        self.meta_features_cache[cache_key] = meta_features
        return meta_features
    
    def recommend_ensemble_config(self, X: np.ndarray, y: np.ndarray) -> EnsembleConfig:
        """Recommend ensemble configuration based on dataset characteristics"""
        
        meta_features = self.extract_meta_features(X, y)
        
        # Rule-based recommendations based on meta-features
        n_samples = meta_features['n_samples']
        n_features = meta_features['n_features']
        class_imbalance = meta_features['class_imbalance']
        dimensionality_ratio = meta_features['dimensionality_ratio']
        
        # Select ensemble method
        if n_samples < 1000:
            method = EnsembleMethod.VOTING  # Simple for small datasets
        elif class_imbalance > 10:
            method = EnsembleMethod.BOOSTING  # Good for imbalanced data
        elif dimensionality_ratio > 0.5:
            method = EnsembleMethod.STACKING  # Good for high-dimensional data
        else:
            method = EnsembleMethod.DYNAMIC_SELECTION  # General purpose
        
        # Select base models
        base_models = ['random_forest', 'gradient_boosting']
        
        if n_features < 50:
            base_models.append('svm')
        
        if n_samples > 5000:
            base_models.append('xgboost')
        
        if class_imbalance < 5:
            base_models.append('naive_bayes')
        
        # Meta-model for stacking
        meta_model = 'logistic_regression' if method == EnsembleMethod.STACKING else None
        
        # Other parameters
        cv_folds = min(5, n_samples // 100) if n_samples > 500 else 3
        hyperparameter_tuning = n_samples > 1000 and OPTUNA_AVAILABLE
        
        explanation_methods = []
        if SHAP_AVAILABLE:
            explanation_methods.append(ExplanationMethod.SHAP)
        if LIME_AVAILABLE:
            explanation_methods.append(ExplanationMethod.LIME)
        explanation_methods.append(ExplanationMethod.FEATURE_IMPORTANCE)
        
        return EnsembleConfig(
            method=method,
            base_models=base_models,
            meta_model=meta_model,
            cv_folds=cv_folds,
            optimization_metric='f1_macro',
            hyperparameter_tuning=hyperparameter_tuning,
            explanation_methods=explanation_methods,
            dynamic_weights=True,
            performance_threshold=0.8
        )


class AdvancedEnsemble(BaseEstimator, ClassifierMixin):
    """Advanced ensemble classifier with multiple combination strategies"""
    
    def __init__(self, config: EnsembleConfig):
        self.config = config
        self.base_estimators = {}
        self.meta_estimator = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names = None
        self.model_weights = {}
        
        # Explanation components
        self.shap_explainer = None
        self.lime_explainer = None
        
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize base models and meta-model"""
        
        model_configs = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'xgboost': xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss'),
            'svm': SVC(probability=True, random_state=42),
            'logistic_regression': LogisticRegression(random_state=42),
            'naive_bayes': GaussianNB(),
            'decision_tree': DecisionTreeClassifier(random_state=42),
            'ada_boost': AdaBoostClassifier(random_state=42),
            'knn': KNeighborsClassifier()
        }
        
        # Initialize base estimators
        for model_name in self.config.base_models:
            if model_name in model_configs:
                self.base_estimators[model_name] = model_configs[model_name]
        
        # Initialize meta-estimator for stacking
        if self.config.meta_model:
            self.meta_estimator = model_configs.get(
                self.config.meta_model, 
                LogisticRegression(random_state=42)
            )
    
    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None):
        """Fit the ensemble model"""
        
        self.feature_names = feature_names or [f'feature_{i}' for i in range(X.shape[1])]
        
        # Fit scaler
        X_scaled = self.scaler.fit_transform(X)
        
        if self.config.method == EnsembleMethod.VOTING:
            self._fit_voting(X_scaled, y)
        elif self.config.method == EnsembleMethod.BAGGING:
            self._fit_bagging(X_scaled, y)
        elif self.config.method == EnsembleMethod.BOOSTING:
            self._fit_boosting(X_scaled, y)
        elif self.config.method == EnsembleMethod.STACKING:
            self._fit_stacking(X_scaled, y)
        elif self.config.method == EnsembleMethod.BLENDING:
            self._fit_blending(X_scaled, y)
        elif self.config.method == EnsembleMethod.DYNAMIC_SELECTION:
            self._fit_dynamic_selection(X_scaled, y)
        
        # Initialize explainers
        self._initialize_explainers(X_scaled)
        
        self.is_fitted = True
        return self
    
    def _fit_voting(self, X: np.ndarray, y: np.ndarray):
        """Fit voting ensemble"""
        
        estimators = [(name, model) for name, model in self.base_estimators.items()]
        
        if self.config.dynamic_weights:
            # Calculate weights based on cross-validation performance
            weights = []
            for name, model in estimators:
                cv_scores = cross_val_score(
                    model, X, y, cv=self.config.cv_folds, 
                    scoring=self.config.optimization_metric
                )
                weights.append(np.mean(cv_scores))
            
            # Normalize weights
            weights = np.array(weights)
            weights = weights / np.sum(weights)
            self.model_weights = {name: w for (name, _), w in zip(estimators, weights)}
        else:
            weights = None
            self.model_weights = {name: 1.0/len(estimators) for name, _ in estimators}
        
        self.ensemble_model = VotingClassifier(
            estimators=estimators,
            voting='soft',
            weights=weights
        )
        
        self.ensemble_model.fit(X, y)
    
    def _fit_bagging(self, X: np.ndarray, y: np.ndarray):
        """Fit bagging ensemble"""
        
        # Use the best performing base model for bagging
        best_model = self._select_best_base_model(X, y)
        
        self.ensemble_model = BaggingClassifier(
            estimator=best_model,
            n_estimators=len(self.base_estimators),
            random_state=42
        )
        
        self.ensemble_model.fit(X, y)
    
    def _fit_boosting(self, X: np.ndarray, y: np.ndarray):
        """Fit boosting ensemble"""
        
        # Use AdaBoost with the best base model
        best_model = self._select_best_base_model(X, y)
        
        self.ensemble_model = AdaBoostClassifier(
            estimator=best_model,
            n_estimators=50,
            random_state=42
        )
        
        self.ensemble_model.fit(X, y)
    
    def _fit_stacking(self, X: np.ndarray, y: np.ndarray):
        """Fit stacking ensemble"""
        
        from sklearn.model_selection import cross_val_predict
        
        # Generate out-of-fold predictions for each base model
        oof_predictions = np.zeros((X.shape[0], len(self.base_estimators)))
        
        for i, (name, model) in enumerate(self.base_estimators.items()):
            try:
                # Use cross-validation to generate out-of-fold predictions
                oof_pred = cross_val_predict(
                    model, X, y, cv=self.config.cv_folds, 
                    method='predict_proba'
                )
                if oof_pred.ndim == 2 and oof_pred.shape[1] == 2:
                    oof_predictions[:, i] = oof_pred[:, 1]  # Positive class probability
                else:
                    oof_predictions[:, i] = cross_val_predict(
                        model, X, y, cv=self.config.cv_folds, method='predict'
                    )
            except Exception as e:
                logger.warning(f"Failed to generate OOF predictions for {name}: {e}")
                # Fallback to simple predictions
                model.fit(X, y)
                oof_predictions[:, i] = model.predict(X)
        
        # Fit meta-estimator on out-of-fold predictions
        self.meta_estimator.fit(oof_predictions, y)
        
        # Fit base estimators on full data
        for name, model in self.base_estimators.items():
            model.fit(X, y)
        
        self.stacked_features = oof_predictions
    
    def _fit_blending(self, X: np.ndarray, y: np.ndarray):
        """Fit blending ensemble"""
        
        # Split data for blending
        split_idx = int(0.8 * len(X))
        X_blend, X_holdout = X[:split_idx], X[split_idx:]
        y_blend, y_holdout = y[:split_idx], y[split_idx:]
        
        # Train base models on blend set
        holdout_predictions = np.zeros((len(X_holdout), len(self.base_estimators)))
        
        for i, (name, model) in enumerate(self.base_estimators.items()):
            model.fit(X_blend, y_blend)
            if hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X_holdout)
                if pred.ndim == 2 and pred.shape[1] == 2:
                    holdout_predictions[:, i] = pred[:, 1]
                else:
                    holdout_predictions[:, i] = model.predict(X_holdout)
            else:
                holdout_predictions[:, i] = model.predict(X_holdout)
        
        # Train meta-estimator on holdout predictions
        self.meta_estimator.fit(holdout_predictions, y_holdout)
        
        # Retrain base models on full data
        for model in self.base_estimators.values():
            model.fit(X, y)
    
    def _fit_dynamic_selection(self, X: np.ndarray, y: np.ndarray):
        """Fit dynamic selection ensemble"""
        
        # Fit all base models
        for model in self.base_estimators.values():
            model.fit(X, y)
        
        # Calculate competence regions (simplified)
        self.competence_regions = {}
        
        for name, model in self.base_estimators.items():
            # Use cross-validation to assess local competence
            cv_scores = cross_val_score(
                model, X, y, cv=self.config.cv_folds, 
                scoring=self.config.optimization_metric
            )
            self.competence_regions[name] = np.mean(cv_scores)
    
    def _select_best_base_model(self, X: np.ndarray, y: np.ndarray) -> BaseEstimator:
        """Select the best performing base model"""
        
        best_score = -1
        best_model = None
        
        for name, model in self.base_estimators.items():
            cv_scores = cross_val_score(
                model, X, y, cv=self.config.cv_folds,
                scoring=self.config.optimization_metric
            )
            score = np.mean(cv_scores)
            
            if score > best_score:
                best_score = score
                best_model = clone(model)
        
        return best_model or list(self.base_estimators.values())[0]
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before making predictions")
        
        X_scaled = self.scaler.transform(X)
        
        if self.config.method == EnsembleMethod.STACKING:
            return self._predict_stacking(X_scaled)
        elif self.config.method == EnsembleMethod.BLENDING:
            return self._predict_blending(X_scaled)
        elif self.config.method == EnsembleMethod.DYNAMIC_SELECTION:
            return self._predict_dynamic_selection(X_scaled)
        else:
            return self.ensemble_model.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities"""
        
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before making predictions")
        
        X_scaled = self.scaler.transform(X)
        
        if self.config.method == EnsembleMethod.STACKING:
            return self._predict_proba_stacking(X_scaled)
        elif self.config.method == EnsembleMethod.BLENDING:
            return self._predict_proba_blending(X_scaled)
        elif self.config.method == EnsembleMethod.DYNAMIC_SELECTION:
            return self._predict_proba_dynamic_selection(X_scaled)
        else:
            return self.ensemble_model.predict_proba(X_scaled)
    
    def _predict_stacking(self, X: np.ndarray) -> np.ndarray:
        """Predict using stacking"""
        
        # Get predictions from base models
        base_predictions = np.zeros((X.shape[0], len(self.base_estimators)))
        
        for i, (name, model) in enumerate(self.base_estimators.items()):
            if hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X)
                if pred.ndim == 2 and pred.shape[1] == 2:
                    base_predictions[:, i] = pred[:, 1]
                else:
                    base_predictions[:, i] = model.predict(X)
            else:
                base_predictions[:, i] = model.predict(X)
        
        # Use meta-estimator for final prediction
        return self.meta_estimator.predict(base_predictions)
    
    def _predict_proba_stacking(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities using stacking"""
        
        base_predictions = np.zeros((X.shape[0], len(self.base_estimators)))
        
        for i, (name, model) in enumerate(self.base_estimators.items()):
            if hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X)
                if pred.ndim == 2 and pred.shape[1] == 2:
                    base_predictions[:, i] = pred[:, 1]
                else:
                    base_predictions[:, i] = model.predict(X)
            else:
                base_predictions[:, i] = model.predict(X)
        
        return self.meta_estimator.predict_proba(base_predictions)
    
    def _predict_blending(self, X: np.ndarray) -> np.ndarray:
        """Predict using blending"""
        return self._predict_stacking(X)  # Similar implementation
    
    def _predict_proba_blending(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities using blending"""
        return self._predict_proba_stacking(X)  # Similar implementation
    
    def _predict_dynamic_selection(self, X: np.ndarray) -> np.ndarray:
        """Predict using dynamic selection"""
        
        predictions = []
        
        for i in range(X.shape[0]):
            # Select best model for this instance (simplified)
            best_model_name = max(self.competence_regions.keys(), 
                                key=lambda k: self.competence_regions[k])
            best_model = self.base_estimators[best_model_name]
            
            pred = best_model.predict(X[i:i+1])
            predictions.append(pred[0])
        
        return np.array(predictions)
    
    def _predict_proba_dynamic_selection(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities using dynamic selection"""
        
        probabilities = []
        
        for i in range(X.shape[0]):
            best_model_name = max(self.competence_regions.keys(), 
                                key=lambda k: self.competence_regions[k])
            best_model = self.base_estimators[best_model_name]
            
            if hasattr(best_model, 'predict_proba'):
                prob = best_model.predict_proba(X[i:i+1])
                probabilities.append(prob[0])
            else:
                # Convert prediction to probability
                pred = best_model.predict(X[i:i+1])
                prob = np.zeros(2)  # Assuming binary classification
                prob[int(pred[0])] = 1.0
                probabilities.append(prob)
        
        return np.array(probabilities)
    
    def _initialize_explainers(self, X: np.ndarray):
        """Initialize explainable AI components"""
        
        try:
            if SHAP_AVAILABLE and ExplanationMethod.SHAP in self.config.explanation_methods:
                # Initialize SHAP explainer based on ensemble type
                if hasattr(self, 'ensemble_model') and hasattr(self.ensemble_model, 'estimators_'):
                    # Use tree-based explainer if possible
                    try:
                        self.shap_explainer = shap.TreeExplainer(self.ensemble_model.estimators_[0])
                    except:
                        # Fallback to kernel explainer
                        self.shap_explainer = shap.KernelExplainer(
                            self.predict_proba, X[:min(100, len(X))]
                        )
                else:
                    self.shap_explainer = shap.KernelExplainer(
                        self.predict_proba, X[:min(100, len(X))]
                    )
        except Exception as e:
            logger.warning(f"Failed to initialize SHAP explainer: {e}")
        
        try:
            if LIME_AVAILABLE and ExplanationMethod.LIME in self.config.explanation_methods:
                self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                    X,
                    feature_names=self.feature_names,
                    class_names=['Normal', 'Anomaly'],  # Assuming binary classification
                    mode='classification'
                )
        except Exception as e:
            logger.warning(f"Failed to initialize LIME explainer: {e}")


class ExplainableAIEngine:
    """Engine for generating model explanations using multiple methods"""
    
    def __init__(self):
        self.openai_client = None
        if hasattr(settings, 'OPENAI_API_KEY') and settings.OPENAI_API_KEY:
            openai.api_key = settings.OPENAI_API_KEY
            self.openai_client = openai
        
        self.explanation_cache = {}
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="explainable_ai")
    
    def explain_prediction(self, model: AdvancedEnsemble, instance: np.ndarray,
                          feature_names: List[str], methods: List[ExplanationMethod]) -> List[ModelExplanation]:
        """Generate explanations for a single prediction"""
        
        explanations = []
        prediction = model.predict(instance.reshape(1, -1))[0]
        
        for method in methods:
            try:
                if method == ExplanationMethod.SHAP and model.shap_explainer:
                    explanation = self._generate_shap_explanation(
                        model, instance, feature_names, prediction
                    )
                    explanations.append(explanation)
                
                elif method == ExplanationMethod.LIME and model.lime_explainer:
                    explanation = self._generate_lime_explanation(
                        model, instance, feature_names, prediction
                    )
                    explanations.append(explanation)
                
                elif method == ExplanationMethod.FEATURE_IMPORTANCE:
                    explanation = self._generate_feature_importance_explanation(
                        model, instance, feature_names, prediction
                    )
                    explanations.append(explanation)
                
            except Exception as e:
                logger.error(f"Failed to generate {method} explanation: {e}")
        
        # Enhance explanations with OpenAI if available
        if self.openai_client and explanations:
            enhanced_explanations = self._enhance_explanations_with_ai(explanations, instance)
            explanations.extend(enhanced_explanations)
        
        return explanations
    
    def _generate_shap_explanation(self, model: AdvancedEnsemble, instance: np.ndarray,
                                  feature_names: List[str], prediction: Any) -> ModelExplanation:
        """Generate SHAP-based explanation"""
        
        try:
            shap_values = model.shap_explainer.shap_values(instance.reshape(1, -1))
            
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Get positive class for binary classification
            
            shap_values = shap_values.flatten()
            
            # Create explanation
            feature_importance = shap_values.tolist()
            
            # Generate text explanation
            top_features = np.argsort(np.abs(shap_values))[-5:][::-1]
            explanation_text = "SHAP Analysis - Top contributing features:\n"
            
            for idx in top_features:
                contribution = "increases" if shap_values[idx] > 0 else "decreases"
                explanation_text += f"- {feature_names[idx]}: {contribution} prediction probability by {abs(shap_values[idx]):.3f}\n"
            
            return ModelExplanation(
                method=ExplanationMethod.SHAP,
                model_id="ensemble",
                instance_id=str(hash(instance.tobytes())),
                feature_names=feature_names,
                feature_values=instance.tolist(),
                feature_importance=feature_importance,
                explanation_text=explanation_text,
                confidence=0.9,
                prediction=prediction,
                timestamp=datetime.now(timezone.utc),
                local_importance={name: float(imp) for name, imp in zip(feature_names, feature_importance)}
            )
            
        except Exception as e:
            logger.error(f"SHAP explanation failed: {e}")
            raise
    
    def _generate_lime_explanation(self, model: AdvancedEnsemble, instance: np.ndarray,
                                  feature_names: List[str], prediction: Any) -> ModelExplanation:
        """Generate LIME-based explanation"""
        
        try:
            # Generate LIME explanation
            explanation = model.lime_explainer.explain_instance(
                instance, 
                model.predict_proba,
                num_features=min(10, len(feature_names))
            )
            
            # Extract feature importance
            lime_values = explanation.as_map()[1]  # Get positive class explanations
            feature_importance = [0.0] * len(feature_names)
            
            for feature_idx, importance in lime_values:
                if 0 <= feature_idx < len(feature_importance):
                    feature_importance[feature_idx] = importance
            
            # Generate text explanation
            explanation_text = "LIME Analysis - Local feature importance:\n"
            
            sorted_features = sorted(lime_values, key=lambda x: abs(x[1]), reverse=True)
            for feature_idx, importance in sorted_features[:5]:
                if 0 <= feature_idx < len(feature_names):
                    effect = "increases" if importance > 0 else "decreases"
                    explanation_text += f"- {feature_names[feature_idx]}: {effect} prediction probability by {abs(importance):.3f}\n"
            
            return ModelExplanation(
                method=ExplanationMethod.LIME,
                model_id="ensemble",
                instance_id=str(hash(instance.tobytes())),
                feature_names=feature_names,
                feature_values=instance.tolist(),
                feature_importance=feature_importance,
                explanation_text=explanation_text,
                confidence=0.85,
                prediction=prediction,
                timestamp=datetime.now(timezone.utc),
                local_importance={name: float(imp) for name, imp in zip(feature_names, feature_importance)}
            )
            
        except Exception as e:
            logger.error(f"LIME explanation failed: {e}")
            raise
    
    def _generate_feature_importance_explanation(self, model: AdvancedEnsemble, instance: np.ndarray,
                                               feature_names: List[str], prediction: Any) -> ModelExplanation:
        """Generate feature importance-based explanation"""
        
        try:
            # Get global feature importance from the model
            if hasattr(model, 'ensemble_model') and hasattr(model.ensemble_model, 'feature_importances_'):
                global_importance = model.ensemble_model.feature_importances_
            else:
                # Calculate average importance from base models
                importance_sum = np.zeros(len(feature_names))
                importance_count = 0
                
                for base_model in model.base_estimators.values():
                    if hasattr(base_model, 'feature_importances_'):
                        importance_sum += base_model.feature_importances_
                        importance_count += 1
                
                if importance_count > 0:
                    global_importance = importance_sum / importance_count
                else:
                    global_importance = np.ones(len(feature_names)) / len(feature_names)
            
            # Weight by instance values to get local importance
            local_importance = global_importance * np.abs(instance)
            local_importance = local_importance / np.sum(local_importance) if np.sum(local_importance) > 0 else local_importance
            
            # Generate text explanation
            top_indices = np.argsort(local_importance)[-5:][::-1]
            explanation_text = "Feature Importance Analysis - Most important features for this instance:\n"
            
            for idx in top_indices:
                explanation_text += f"- {feature_names[idx]}: {local_importance[idx]:.3f} importance (value: {instance[idx]:.3f})\n"
            
            return ModelExplanation(
                method=ExplanationMethod.FEATURE_IMPORTANCE,
                model_id="ensemble",
                instance_id=str(hash(instance.tobytes())),
                feature_names=feature_names,
                feature_values=instance.tolist(),
                feature_importance=local_importance.tolist(),
                explanation_text=explanation_text,
                confidence=0.75,
                prediction=prediction,
                timestamp=datetime.now(timezone.utc),
                global_importance={name: float(imp) for name, imp in zip(feature_names, global_importance)},
                local_importance={name: float(imp) for name, imp in zip(feature_names, local_importance)}
            )
            
        except Exception as e:
            logger.error(f"Feature importance explanation failed: {e}")
            raise
    
    def _enhance_explanations_with_ai(self, explanations: List[ModelExplanation], 
                                    instance: np.ndarray) -> List[ModelExplanation]:
        """Enhance explanations using OpenAI"""
        
        if not self.openai_client:
            return []
        
        try:
            # Combine explanation insights
            combined_insights = []
            for exp in explanations:
                combined_insights.append(f"{exp.method.value}: {exp.explanation_text}")
            
            # Create prompt for AI enhancement
            prompt = f"""
            Given the following machine learning model explanations for a cybersecurity threat detection:
            
            Instance features: {instance.tolist()[:10]}  # First 10 features for brevity
            Prediction: {explanations[0].prediction if explanations else 'Unknown'}
            
            Explanations:
            {chr(10).join(combined_insights)}
            
            Please provide:
            1. A concise summary of why this prediction was made
            2. The most critical factors influencing the decision
            3. Potential security implications
            4. Confidence assessment of the prediction
            
            Keep the response under 200 words and focus on actionable insights for cybersecurity analysts.
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert cybersecurity analyst with deep knowledge of machine learning threat detection systems."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=250,
                temperature=0.3
            )
            
            ai_explanation = response.choices[0].message.content
            
            # Create AI-enhanced explanation
            enhanced_explanation = ModelExplanation(
                method=ExplanationMethod.GRADIENT,  # Use gradient as a placeholder for AI-enhanced
                model_id="ensemble_ai",
                instance_id=str(hash(instance.tobytes())),
                feature_names=explanations[0].feature_names if explanations else [],
                feature_values=instance.tolist(),
                feature_importance=[],
                explanation_text=f"AI-Enhanced Analysis:\n{ai_explanation}",
                confidence=0.95,
                prediction=explanations[0].prediction if explanations else None,
                timestamp=datetime.now(timezone.utc)
            )
            
            return [enhanced_explanation]
            
        except Exception as e:
            logger.error(f"AI enhancement failed: {e}")
            return []
    
    def generate_global_explanations(self, model: AdvancedEnsemble, 
                                   X_sample: np.ndarray) -> Dict[str, Any]:
        """Generate global model explanations"""
        
        global_explanations = {}
        
        try:
            # Global feature importance
            if hasattr(model, 'ensemble_model') and hasattr(model.ensemble_model, 'feature_importances_'):
                importance = model.ensemble_model.feature_importances_
            else:
                # Average from base models
                importance_sum = np.zeros(X_sample.shape[1])
                count = 0
                
                for base_model in model.base_estimators.values():
                    if hasattr(base_model, 'feature_importances_'):
                        importance_sum += base_model.feature_importances_
                        count += 1
                
                importance = importance_sum / count if count > 0 else importance_sum
            
            global_explanations['feature_importance'] = {
                f'feature_{i}': float(imp) for i, imp in enumerate(importance)
            }
            
            # Model-specific insights
            global_explanations['model_weights'] = model.model_weights
            global_explanations['ensemble_method'] = model.config.method.value
            global_explanations['base_models'] = list(model.base_estimators.keys())
            
        except Exception as e:
            logger.error(f"Global explanation generation failed: {e}")
        
        return global_explanations


# Global instances
meta_learning_optimizer = MetaLearningOptimizer()
explainable_ai_engine = ExplainableAIEngine()


def create_optimized_ensemble(X: np.ndarray, y: np.ndarray, 
                            feature_names: Optional[List[str]] = None,
                            custom_config: Optional[EnsembleConfig] = None) -> AdvancedEnsemble:
    """Create an optimized ensemble model"""
    
    if custom_config:
        config = custom_config
    else:
        # Use meta-learning to recommend configuration
        config = meta_learning_optimizer.recommend_ensemble_config(X, y)
    
    # Create and fit ensemble
    ensemble = AdvancedEnsemble(config)
    ensemble.fit(X, y, feature_names)
    
    logger.info(f"Created {config.method.value} ensemble with {len(config.base_models)} base models")
    
    return ensemble


def explain_ensemble_prediction(ensemble: AdvancedEnsemble, instance: np.ndarray,
                              feature_names: List[str],
                              methods: List[ExplanationMethod] = None) -> List[ModelExplanation]:
    """Generate explanations for ensemble prediction"""
    
    if methods is None:
        methods = [ExplanationMethod.SHAP, ExplanationMethod.LIME, ExplanationMethod.FEATURE_IMPORTANCE]
    
    return explainable_ai_engine.explain_prediction(ensemble, instance, feature_names, methods)


if __name__ == "__main__":
    # Test the ensemble optimization system
    from sklearn.datasets import make_classification
    
    print("Testing Advanced Ensemble Optimization System...")
    
    # Generate test data
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=15, 
        n_redundant=5, n_classes=2, random_state=42
    )
    
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    
    # Create optimized ensemble
    ensemble = create_optimized_ensemble(X, y, feature_names)
    
    print(f"Ensemble method: {ensemble.config.method}")
    print(f"Base models: {ensemble.config.base_models}")
    
    # Make predictions
    predictions = ensemble.predict(X[:10])
    probabilities = ensemble.predict_proba(X[:10])
    
    print(f"Sample predictions: {predictions[:5]}")
    print(f"Sample probabilities: {probabilities[:3]}")
    
    # Generate explanations
    if SHAP_AVAILABLE or LIME_AVAILABLE:
        explanations = explain_ensemble_prediction(
            ensemble, X[0], feature_names, 
            [ExplanationMethod.FEATURE_IMPORTANCE]
        )
        
        for exp in explanations:
            print(f"\n{exp.method.value} Explanation:")
            print(exp.explanation_text[:200] + "...")
    
    print("Advanced ensemble optimization system test completed!")

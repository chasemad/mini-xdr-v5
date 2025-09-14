"""
Model Versioning and A/B Testing Framework
Manages ML model versions with performance tracking and automated rollback
"""

import asyncio
import logging
import json
import hashlib
import pickle
import base64
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import threading
from collections import defaultdict, deque

from sklearn.base import BaseEstimator, clone
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import joblib

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func, desc, text
from .models import Event, Incident, MLModel
from .db import AsyncSessionLocal
from .config import settings

logger = logging.getLogger(__name__)


class ModelStatus(str, Enum):
    """Model deployment status"""
    TRAINING = "training"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"
    FAILED = "failed"


class ComparisonResult(str, Enum):
    """A/B test comparison results"""
    A_WINS = "a_wins"
    B_WINS = "b_wins"
    TIE = "tie"
    INSUFFICIENT_DATA = "insufficient_data"


@dataclass
class ModelVersion:
    """Model version metadata"""
    model_id: str
    version: str
    status: ModelStatus
    created_at: datetime
    updated_at: datetime
    model_type: str
    algorithm: str
    hyperparameters: Dict[str, Any]
    training_data_hash: str
    model_checksum: str
    performance_metrics: Dict[str, float]
    deployment_config: Dict[str, Any]
    tags: List[str]
    description: str
    parent_version: Optional[str] = None


@dataclass
class PerformanceMetrics:
    """Model performance metrics"""
    timestamp: datetime
    model_id: str
    version: str
    dataset_size: int
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: Optional[float]
    confusion_matrix: List[List[int]]
    prediction_latency_ms: float
    throughput_qps: float
    memory_usage_mb: float
    cpu_usage_percent: float
    error_rate: float
    confidence_scores: List[float]
    feature_importance: Dict[str, float]


@dataclass
class ABTestConfig:
    """A/B test configuration"""
    test_id: str
    name: str
    description: str
    model_a_id: str
    model_a_version: str
    model_b_id: str
    model_b_version: str
    traffic_split: float  # 0.0 to 1.0 (fraction going to model B)
    success_metric: str   # 'accuracy', 'precision', 'f1', etc.
    min_sample_size: int
    max_duration_hours: int
    significance_threshold: float
    created_at: datetime
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    status: str = "created"


@dataclass
class ABTestResult:
    """A/B test results"""
    test_id: str
    winner: ComparisonResult
    confidence: float
    p_value: float
    effect_size: float
    samples_a: int
    samples_b: int
    metric_a: float
    metric_b: float
    statistical_significance: bool
    practical_significance: bool
    recommendation: str
    detailed_metrics: Dict[str, Any]


class ModelRegistry:
    """Registry for managing model versions"""
    
    def __init__(self, storage_path: str = "models"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # In-memory registry
        self.models = {}  # model_id -> {version -> ModelVersion}
        self.performance_history = {}  # model_id -> version -> [PerformanceMetrics]
        
        # Thread safety
        self.lock = threading.RLock()
        
        logger.info(f"Model registry initialized with storage path: {self.storage_path}")
    
    def register_model(self, model: BaseEstimator, model_id: str, version: str,
                      model_type: str, algorithm: str, hyperparameters: Dict[str, Any],
                      training_data_hash: str, description: str = "",
                      tags: List[str] = None, parent_version: str = None) -> ModelVersion:
        """Register a new model version"""
        
        with self.lock:
            if tags is None:
                tags = []
            
            # Calculate model checksum
            model_bytes = pickle.dumps(model)
            model_checksum = hashlib.sha256(model_bytes).hexdigest()
            
            # Save model to disk
            model_path = self.storage_path / f"{model_id}_{version}.pkl"
            joblib.dump(model, model_path)
            
            # Create model version
            model_version = ModelVersion(
                model_id=model_id,
                version=version,
                status=ModelStatus.TRAINING,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
                model_type=model_type,
                algorithm=algorithm,
                hyperparameters=hyperparameters,
                training_data_hash=training_data_hash,
                model_checksum=model_checksum,
                performance_metrics={},
                deployment_config={},
                tags=tags,
                description=description,
                parent_version=parent_version
            )
            
            # Add to registry
            if model_id not in self.models:
                self.models[model_id] = {}
            
            self.models[model_id][version] = model_version
            
            logger.info(f"Registered model {model_id} version {version}")
            return model_version
    
    def load_model(self, model_id: str, version: str) -> Optional[BaseEstimator]:
        """Load a model from storage"""
        
        model_path = self.storage_path / f"{model_id}_{version}.pkl"
        
        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            return None
        
        try:
            model = joblib.load(model_path)
            logger.debug(f"Loaded model {model_id} version {version}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model {model_id} version {version}: {e}")
            return None
    
    def update_model_status(self, model_id: str, version: str, status: ModelStatus):
        """Update model status"""
        
        with self.lock:
            if model_id in self.models and version in self.models[model_id]:
                self.models[model_id][version].status = status
                self.models[model_id][version].updated_at = datetime.now(timezone.utc)
                logger.info(f"Updated model {model_id} version {version} status to {status}")
    
    def record_performance(self, model_id: str, version: str, metrics: PerformanceMetrics):
        """Record performance metrics for a model version"""
        
        with self.lock:
            if model_id not in self.performance_history:
                self.performance_history[model_id] = {}
            
            if version not in self.performance_history[model_id]:
                self.performance_history[model_id][version] = deque(maxlen=1000)
            
            self.performance_history[model_id][version].append(metrics)
            
            # Update latest metrics in model version
            if model_id in self.models and version in self.models[model_id]:
                self.models[model_id][version].performance_metrics = {
                    'accuracy': metrics.accuracy,
                    'precision': metrics.precision,
                    'recall': metrics.recall,
                    'f1_score': metrics.f1_score,
                    'auc_roc': metrics.auc_roc,
                    'error_rate': metrics.error_rate
                }
                self.models[model_id][version].updated_at = datetime.now(timezone.utc)
    
    def get_model_versions(self, model_id: str) -> List[ModelVersion]:
        """Get all versions of a model"""
        
        if model_id not in self.models:
            return []
        
        return list(self.models[model_id].values())
    
    def get_production_models(self) -> List[ModelVersion]:
        """Get all models currently in production"""
        
        production_models = []
        
        for model_id, versions in self.models.items():
            for version, model_version in versions.items():
                if model_version.status == ModelStatus.PRODUCTION:
                    production_models.append(model_version)
        
        return production_models
    
    def get_performance_history(self, model_id: str, version: str, 
                               limit: int = 100) -> List[PerformanceMetrics]:
        """Get performance history for a model version"""
        
        if (model_id not in self.performance_history or 
            version not in self.performance_history[model_id]):
            return []
        
        history = list(self.performance_history[model_id][version])
        return history[-limit:]
    
    def compare_models(self, model_a_id: str, model_a_version: str,
                      model_b_id: str, model_b_version: str,
                      metric: str = 'f1_score') -> Dict[str, Any]:
        """Compare two model versions"""
        
        # Get recent performance data
        metrics_a = self.get_performance_history(model_a_id, model_a_version, limit=50)
        metrics_b = self.get_performance_history(model_b_id, model_b_version, limit=50)
        
        if not metrics_a or not metrics_b:
            return {
                'comparison': 'insufficient_data',
                'message': 'Insufficient performance data for comparison'
            }
        
        # Extract metric values
        values_a = [getattr(m, metric) for m in metrics_a if hasattr(m, metric)]
        values_b = [getattr(m, metric) for m in metrics_b if hasattr(m, metric)]
        
        if not values_a or not values_b:
            return {
                'comparison': 'metric_not_found',
                'message': f'Metric {metric} not found in performance data'
            }
        
        # Statistical comparison
        mean_a = np.mean(values_a)
        mean_b = np.mean(values_b)
        std_a = np.std(values_a)
        std_b = np.std(values_b)
        
        # Simple t-test equivalent
        pooled_std = np.sqrt((std_a**2 + std_b**2) / 2)
        t_stat = abs(mean_a - mean_b) / (pooled_std * np.sqrt(2 / min(len(values_a), len(values_b))))
        
        # Determine winner
        if t_stat > 1.96:  # 95% confidence
            winner = 'model_a' if mean_a > mean_b else 'model_b'
            confidence = min(0.95, t_stat / 1.96 * 0.8)
        else:
            winner = 'tie'
            confidence = 0.5
        
        return {
            'comparison': winner,
            'confidence': confidence,
            'model_a_mean': mean_a,
            'model_b_mean': mean_b,
            'model_a_std': std_a,
            'model_b_std': std_b,
            'samples_a': len(values_a),
            'samples_b': len(values_b),
            't_statistic': t_stat,
            'metric': metric
        }


class ABTestManager:
    """Manages A/B testing of model versions"""
    
    def __init__(self, model_registry: ModelRegistry):
        self.model_registry = model_registry
        self.active_tests = {}  # test_id -> ABTestConfig
        self.test_results = {}  # test_id -> ABTestResult
        self.test_data = defaultdict(lambda: {'a': [], 'b': []})  # test_id -> {a: [metrics], b: [metrics]}
        
        self.lock = threading.RLock()
        
        logger.info("A/B Test Manager initialized")
    
    def create_ab_test(self, name: str, description: str,
                      model_a_id: str, model_a_version: str,
                      model_b_id: str, model_b_version: str,
                      traffic_split: float = 0.5,
                      success_metric: str = 'f1_score',
                      min_sample_size: int = 100,
                      max_duration_hours: int = 168,  # 7 days
                      significance_threshold: float = 0.05) -> str:
        """Create a new A/B test"""
        
        test_id = f"test_{int(datetime.now().timestamp())}_{hash(name) % 10000}"
        
        test_config = ABTestConfig(
            test_id=test_id,
            name=name,
            description=description,
            model_a_id=model_a_id,
            model_a_version=model_a_version,
            model_b_id=model_b_id,
            model_b_version=model_b_version,
            traffic_split=traffic_split,
            success_metric=success_metric,
            min_sample_size=min_sample_size,
            max_duration_hours=max_duration_hours,
            significance_threshold=significance_threshold,
            created_at=datetime.now(timezone.utc)
        )
        
        with self.lock:
            self.active_tests[test_id] = test_config
        
        logger.info(f"Created A/B test {test_id}: {name}")
        return test_id
    
    def start_ab_test(self, test_id: str) -> bool:
        """Start an A/B test"""
        
        with self.lock:
            if test_id not in self.active_tests:
                logger.error(f"A/B test {test_id} not found")
                return False
            
            test_config = self.active_tests[test_id]
            
            if test_config.status != "created":
                logger.error(f"A/B test {test_id} already started or completed")
                return False
            
            # Verify models exist
            model_a = self.model_registry.load_model(test_config.model_a_id, test_config.model_a_version)
            model_b = self.model_registry.load_model(test_config.model_b_id, test_config.model_b_version)
            
            if not model_a or not model_b:
                logger.error(f"Failed to load models for A/B test {test_id}")
                return False
            
            # Start the test
            test_config.started_at = datetime.now(timezone.utc)
            test_config.status = "running"
            
            logger.info(f"Started A/B test {test_id}")
            return True
    
    def record_ab_test_result(self, test_id: str, model_used: str, 
                             prediction: Any, actual: Any, 
                             performance_metrics: Dict[str, float]):
        """Record a result for an A/B test"""
        
        if test_id not in self.active_tests:
            return
        
        test_config = self.active_tests[test_id]
        if test_config.status != "running":
            return
        
        with self.lock:
            # Determine which model was used
            if model_used == 'a':
                self.test_data[test_id]['a'].append({
                    'timestamp': datetime.now(timezone.utc),
                    'prediction': prediction,
                    'actual': actual,
                    'metrics': performance_metrics
                })
            elif model_used == 'b':
                self.test_data[test_id]['b'].append({
                    'timestamp': datetime.now(timezone.utc),
                    'prediction': prediction,
                    'actual': actual,
                    'metrics': performance_metrics
                })
        
        # Check if test should be concluded
        self._check_test_completion(test_id)
    
    def _check_test_completion(self, test_id: str):
        """Check if A/B test should be completed"""
        
        test_config = self.active_tests[test_id]
        test_data = self.test_data[test_id]
        
        # Check sample size
        samples_a = len(test_data['a'])
        samples_b = len(test_data['b'])
        
        min_samples_met = (samples_a >= test_config.min_sample_size and 
                          samples_b >= test_config.min_sample_size)
        
        # Check duration
        if test_config.started_at:
            duration = datetime.now(timezone.utc) - test_config.started_at
            max_duration_met = duration.total_seconds() / 3600 >= test_config.max_duration_hours
        else:
            max_duration_met = False
        
        # Complete test if conditions are met
        if min_samples_met or max_duration_met:
            self.complete_ab_test(test_id)
    
    def complete_ab_test(self, test_id: str, force: bool = False) -> Optional[ABTestResult]:
        """Complete an A/B test and analyze results"""
        
        if test_id not in self.active_tests:
            return None
        
        test_config = self.active_tests[test_id]
        test_data = self.test_data[test_id]
        
        if not force and test_config.status != "running":
            return None
        
        # Extract metric values
        metric_name = test_config.success_metric
        
        values_a = []
        values_b = []
        
        for result in test_data['a']:
            if metric_name in result['metrics']:
                values_a.append(result['metrics'][metric_name])
        
        for result in test_data['b']:
            if metric_name in result['metrics']:
                values_b.append(result['metrics'][metric_name])
        
        # Perform statistical analysis
        if len(values_a) < 10 or len(values_b) < 10:
            winner = ComparisonResult.INSUFFICIENT_DATA
            confidence = 0.0
            p_value = 1.0
            effect_size = 0.0
            statistical_significance = False
        else:
            mean_a = np.mean(values_a)
            mean_b = np.mean(values_b)
            std_a = np.std(values_a)
            std_b = np.std(values_b)
            
            # Cohen's d effect size
            pooled_std = np.sqrt((std_a**2 + std_b**2) / 2)
            effect_size = abs(mean_a - mean_b) / pooled_std if pooled_std > 0 else 0
            
            # Simple t-test approximation
            se_diff = np.sqrt(std_a**2/len(values_a) + std_b**2/len(values_b))
            t_stat = abs(mean_a - mean_b) / se_diff if se_diff > 0 else 0
            
            # Approximate p-value (simplified)
            if t_stat > 1.96:
                p_value = 0.05 / (t_stat / 1.96)
            else:
                p_value = 1.0 - (t_stat / 1.96 * 0.5)
            
            statistical_significance = p_value < test_config.significance_threshold
            confidence = min(0.95, 1 - p_value)
            
            # Determine winner
            if statistical_significance and effect_size > 0.2:  # Practical significance
                if mean_a > mean_b:
                    winner = ComparisonResult.A_WINS
                else:
                    winner = ComparisonResult.B_WINS
            else:
                winner = ComparisonResult.TIE
        
        # Create result
        ab_result = ABTestResult(
            test_id=test_id,
            winner=winner,
            confidence=confidence,
            p_value=p_value,
            effect_size=effect_size,
            samples_a=len(values_a),
            samples_b=len(values_b),
            metric_a=np.mean(values_a) if values_a else 0,
            metric_b=np.mean(values_b) if values_b else 0,
            statistical_significance=statistical_significance,
            practical_significance=effect_size > 0.2,
            recommendation=self._generate_recommendation(winner, effect_size, confidence),
            detailed_metrics={
                'metric_name': metric_name,
                'mean_a': np.mean(values_a) if values_a else 0,
                'mean_b': np.mean(values_b) if values_b else 0,
                'std_a': np.std(values_a) if values_a else 0,
                'std_b': np.std(values_b) if values_b else 0,
                't_statistic': t_stat if 't_stat' in locals() else 0
            }
        )
        
        # Update test status
        with self.lock:
            test_config.status = "completed"
            test_config.ended_at = datetime.now(timezone.utc)
            self.test_results[test_id] = ab_result
        
        logger.info(f"Completed A/B test {test_id}: {winner} (confidence: {confidence:.3f})")
        return ab_result
    
    def _generate_recommendation(self, winner: ComparisonResult, 
                               effect_size: float, confidence: float) -> str:
        """Generate recommendation based on A/B test results"""
        
        if winner == ComparisonResult.INSUFFICIENT_DATA:
            return "Collect more data before making a decision"
        
        elif winner == ComparisonResult.A_WINS:
            if effect_size > 0.5 and confidence > 0.8:
                return "Strong evidence for Model A - recommend deployment"
            elif effect_size > 0.2 and confidence > 0.7:
                return "Moderate evidence for Model A - consider deployment with monitoring"
            else:
                return "Weak evidence for Model A - continue testing or collect more data"
        
        elif winner == ComparisonResult.B_WINS:
            if effect_size > 0.5 and confidence > 0.8:
                return "Strong evidence for Model B - recommend deployment"
            elif effect_size > 0.2 and confidence > 0.7:
                return "Moderate evidence for Model B - consider deployment with monitoring"
            else:
                return "Weak evidence for Model B - continue testing or collect more data"
        
        else:  # TIE
            return "No significant difference detected - current model performance is adequate"
    
    def get_active_tests(self) -> List[ABTestConfig]:
        """Get all active A/B tests"""
        return [test for test in self.active_tests.values() if test.status == "running"]
    
    def get_test_results(self, test_id: str) -> Optional[ABTestResult]:
        """Get results for a specific test"""
        return self.test_results.get(test_id)
    
    def route_prediction_request(self, test_id: str) -> str:
        """Route prediction request to model A or B based on traffic split"""
        
        if test_id not in self.active_tests:
            return 'a'  # Default to model A
        
        test_config = self.active_tests[test_id]
        
        if test_config.status != "running":
            return 'a'
        
        # Route based on traffic split
        import random
        return 'b' if random.random() < test_config.traffic_split else 'a'


class ModelPerformanceMonitor:
    """Monitors model performance and triggers alerts"""
    
    def __init__(self, model_registry: ModelRegistry):
        self.model_registry = model_registry
        self.alerts = deque(maxlen=1000)
        
        # Performance thresholds
        self.thresholds = {
            'accuracy_drop': 0.05,     # 5% accuracy drop
            'error_rate_spike': 0.1,   # 10% error rate spike
            'latency_spike': 2.0,      # 2x latency increase
            'throughput_drop': 0.3     # 30% throughput drop
        }
        
        self.baseline_window = 24  # Hours for baseline calculation
        
        logger.info("Model Performance Monitor initialized")
    
    def evaluate_model_performance(self, model_id: str, version: str,
                                 test_data: np.ndarray, test_labels: np.ndarray) -> PerformanceMetrics:
        """Evaluate model performance on test data"""
        
        model = self.model_registry.load_model(model_id, version)
        if not model:
            raise ValueError(f"Model {model_id} version {version} not found")
        
        start_time = datetime.now()
        
        # Make predictions
        predictions = model.predict(test_data)
        prediction_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Calculate metrics
        accuracy = accuracy_score(test_labels, predictions)
        precision = precision_score(test_labels, predictions, average='weighted', zero_division=0)
        recall = recall_score(test_labels, predictions, average='weighted', zero_division=0)
        f1 = f1_score(test_labels, predictions, average='weighted', zero_division=0)
        
        # AUC-ROC (if binary classification)
        try:
            if len(np.unique(test_labels)) == 2:
                # Get prediction probabilities for AUC
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(test_data)
                    auc_roc = roc_auc_score(test_labels, proba[:, 1])
                elif hasattr(model, 'decision_function'):
                    scores = model.decision_function(test_data)
                    auc_roc = roc_auc_score(test_labels, scores)
                else:
                    auc_roc = None
            else:
                auc_roc = None
        except Exception:
            auc_roc = None
        
        # Confusion matrix
        cm = confusion_matrix(test_labels, predictions).tolist()
        
        # Error rate
        error_rate = 1 - accuracy
        
        # Confidence scores (if available)
        confidence_scores = []
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(test_data)
            confidence_scores = np.max(proba, axis=1).tolist()
        
        # Feature importance (if available)
        feature_importance = {}
        if hasattr(model, 'feature_importances_'):
            feature_importance = {
                f'feature_{i}': float(importance) 
                for i, importance in enumerate(model.feature_importances_)
            }
        
        # Create performance metrics
        metrics = PerformanceMetrics(
            timestamp=datetime.now(timezone.utc),
            model_id=model_id,
            version=version,
            dataset_size=len(test_data),
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            auc_roc=auc_roc,
            confusion_matrix=cm,
            prediction_latency_ms=prediction_time / len(test_data),
            throughput_qps=len(test_data) / (prediction_time / 1000),
            memory_usage_mb=0.0,  # Would need system monitoring
            cpu_usage_percent=0.0,  # Would need system monitoring
            error_rate=error_rate,
            confidence_scores=confidence_scores,
            feature_importance=feature_importance
        )
        
        # Record metrics
        self.model_registry.record_performance(model_id, version, metrics)
        
        # Check for alerts
        self._check_performance_alerts(metrics)
        
        return metrics
    
    def _check_performance_alerts(self, metrics: PerformanceMetrics):
        """Check if performance metrics trigger any alerts"""
        
        # Get baseline performance
        baseline_metrics = self._get_baseline_performance(metrics.model_id, metrics.version)
        
        if not baseline_metrics:
            return  # No baseline to compare against
        
        alerts = []
        
        # Check accuracy drop
        accuracy_drop = baseline_metrics['accuracy'] - metrics.accuracy
        if accuracy_drop > self.thresholds['accuracy_drop']:
            alerts.append({
                'type': 'accuracy_drop',
                'severity': 'high' if accuracy_drop > 0.1 else 'medium',
                'message': f"Accuracy dropped by {accuracy_drop:.3f} from baseline",
                'current': metrics.accuracy,
                'baseline': baseline_metrics['accuracy']
            })
        
        # Check error rate spike
        error_spike = metrics.error_rate - baseline_metrics['error_rate']
        if error_spike > self.thresholds['error_rate_spike']:
            alerts.append({
                'type': 'error_rate_spike',
                'severity': 'high',
                'message': f"Error rate increased by {error_spike:.3f}",
                'current': metrics.error_rate,
                'baseline': baseline_metrics['error_rate']
            })
        
        # Check latency spike
        if baseline_metrics['prediction_latency_ms'] > 0:
            latency_ratio = metrics.prediction_latency_ms / baseline_metrics['prediction_latency_ms']
            if latency_ratio > self.thresholds['latency_spike']:
                alerts.append({
                    'type': 'latency_spike',
                    'severity': 'medium',
                    'message': f"Prediction latency increased by {latency_ratio:.1f}x",
                    'current': metrics.prediction_latency_ms,
                    'baseline': baseline_metrics['prediction_latency_ms']
                })
        
        # Store alerts
        for alert in alerts:
            alert_record = {
                'timestamp': metrics.timestamp,
                'model_id': metrics.model_id,
                'version': metrics.version,
                **alert
            }
            self.alerts.append(alert_record)
            
            # Log alert
            logger.warning(f"Performance alert for {metrics.model_id} v{metrics.version}: {alert['message']}")
    
    def _get_baseline_performance(self, model_id: str, version: str) -> Optional[Dict[str, float]]:
        """Get baseline performance metrics for comparison"""
        
        # Get recent performance history
        history = self.model_registry.get_performance_history(model_id, version, limit=100)
        
        if len(history) < 10:
            return None  # Not enough data for baseline
        
        # Use metrics from last 24 hours as baseline
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=self.baseline_window)
        baseline_metrics = [m for m in history if m.timestamp >= cutoff_time]
        
        if len(baseline_metrics) < 5:
            # Fallback to older data if recent data is insufficient
            baseline_metrics = history[-20:]
        
        # Calculate baseline averages
        return {
            'accuracy': np.mean([m.accuracy for m in baseline_metrics]),
            'precision': np.mean([m.precision for m in baseline_metrics]),
            'recall': np.mean([m.recall for m in baseline_metrics]),
            'f1_score': np.mean([m.f1_score for m in baseline_metrics]),
            'error_rate': np.mean([m.error_rate for m in baseline_metrics]),
            'prediction_latency_ms': np.mean([m.prediction_latency_ms for m in baseline_metrics]),
            'throughput_qps': np.mean([m.throughput_qps for m in baseline_metrics])
        }
    
    def get_recent_alerts(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent performance alerts"""
        
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        recent_alerts = [
            alert for alert in self.alerts 
            if alert['timestamp'] >= cutoff_time
        ]
        
        return recent_alerts
    
    def get_model_health_summary(self, model_id: str, version: str) -> Dict[str, Any]:
        """Get health summary for a model"""
        
        # Get recent metrics
        recent_metrics = self.model_registry.get_performance_history(model_id, version, limit=10)
        
        if not recent_metrics:
            return {'status': 'no_data', 'message': 'No performance data available'}
        
        # Calculate recent averages
        recent_accuracy = np.mean([m.accuracy for m in recent_metrics])
        recent_error_rate = np.mean([m.error_rate for m in recent_metrics])
        recent_latency = np.mean([m.prediction_latency_ms for m in recent_metrics])
        
        # Get recent alerts
        recent_alerts = [
            alert for alert in self.alerts 
            if (alert['model_id'] == model_id and alert['version'] == version and
                alert['timestamp'] >= datetime.now(timezone.utc) - timedelta(hours=1))
        ]
        
        # Determine health status
        if len(recent_alerts) > 0:
            high_severity_alerts = [a for a in recent_alerts if a['severity'] == 'high']
            if high_severity_alerts:
                status = 'unhealthy'
            else:
                status = 'degraded'
        elif recent_accuracy > 0.8 and recent_error_rate < 0.1:
            status = 'healthy'
        elif recent_accuracy > 0.7:
            status = 'degraded'
        else:
            status = 'unhealthy'
        
        return {
            'status': status,
            'accuracy': recent_accuracy,
            'error_rate': recent_error_rate,
            'latency_ms': recent_latency,
            'recent_alerts': len(recent_alerts),
            'data_points': len(recent_metrics),
            'last_evaluation': recent_metrics[-1].timestamp.isoformat() if recent_metrics else None
        }


# Global instances
model_registry = ModelRegistry()
ab_test_manager = ABTestManager(model_registry)
performance_monitor = ModelPerformanceMonitor(model_registry)


async def initialize_model_versioning():
    """Initialize model versioning system"""
    logger.info("Model versioning system initialized")
    return True


if __name__ == "__main__":
    # Test the model versioning system
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    
    print("Testing Model Versioning System...")
    
    # Create test data
    X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
    
    # Create and train a model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X[:800], y[:800])
    
    # Register the model
    model_version = model_registry.register_model(
        model=model,
        model_id="test_model",
        version="v1.0",
        model_type="classifier",
        algorithm="random_forest",
        hyperparameters={"n_estimators": 10},
        training_data_hash="test_hash",
        description="Test model for demonstration"
    )
    
    print(f"Registered model: {model_version.model_id} v{model_version.version}")
    
    # Evaluate performance
    metrics = performance_monitor.evaluate_model_performance(
        "test_model", "v1.0", X[800:], y[800:]
    )
    
    print(f"Model performance - Accuracy: {metrics.accuracy:.3f}, F1: {metrics.f1_score:.3f}")
    
    # Create A/B test
    # First create a second model version
    model2 = RandomForestClassifier(n_estimators=20, random_state=42)
    model2.fit(X[:800], y[:800])
    
    model_version2 = model_registry.register_model(
        model=model2,
        model_id="test_model",
        version="v2.0",
        model_type="classifier",
        algorithm="random_forest",
        hyperparameters={"n_estimators": 20},
        training_data_hash="test_hash_2",
        description="Improved test model",
        parent_version="v1.0"
    )
    
    # Create A/B test
    test_id = ab_test_manager.create_ab_test(
        name="Random Forest Comparison",
        description="Compare v1.0 vs v2.0",
        model_a_id="test_model",
        model_a_version="v1.0",
        model_b_id="test_model",
        model_b_version="v2.0"
    )
    
    print(f"Created A/B test: {test_id}")
    
    # Start test
    ab_test_manager.start_ab_test(test_id)
    print("A/B test started")
    
    # Simulate some test results
    for i in range(50):
        model_used = ab_test_manager.route_prediction_request(test_id)
        # Simulate metrics (model B is slightly better)
        if model_used == 'a':
            simulated_metrics = {'f1_score': 0.85 + np.random.normal(0, 0.05)}
        else:
            simulated_metrics = {'f1_score': 0.87 + np.random.normal(0, 0.05)}
        
        ab_test_manager.record_ab_test_result(
            test_id, model_used, 
            prediction=1, actual=1,  # Dummy values
            performance_metrics=simulated_metrics
        )
    
    # Complete test
    result = ab_test_manager.complete_ab_test(test_id, force=True)
    if result:
        print(f"A/B test result: {result.winner} (confidence: {result.confidence:.3f})")
        print(f"Recommendation: {result.recommendation}")
    
    print("Model versioning system test completed!")

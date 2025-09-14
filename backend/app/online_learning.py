"""
Real-Time ML Model Adaptation System
Enables models to adapt to new threats in real-time with concept drift detection
"""

import asyncio
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import json
import joblib
from pathlib import Path
import warnings
from collections import deque
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier
from sklearn.utils import shuffle

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func, desc
from .models import Event, Incident, MLModel
from .db import AsyncSessionLocal
from .config import settings

logger = logging.getLogger(__name__)


class AdaptationStrategy(str, Enum):
    """Online learning adaptation strategies"""
    INCREMENTAL = "incremental"  # Pure online learning
    MINI_BATCH = "mini_batch"    # Mini-batch updates
    SLIDING_WINDOW = "sliding_window"  # Fixed window retraining
    ENSEMBLE_VOTING = "ensemble_voting"  # Multiple model voting


class ConceptDriftType(str, Enum):
    """Types of concept drift detection"""
    GRADUAL = "gradual"     # Slow change over time
    SUDDEN = "sudden"       # Abrupt change
    RECURRING = "recurring"  # Cyclical patterns
    UNKNOWN = "unknown"     # Unclassified drift


@dataclass
class DriftDetectionResult:
    """Result of concept drift detection"""
    drift_detected: bool
    drift_type: ConceptDriftType
    drift_magnitude: float  # 0.0 to 1.0
    confidence: float       # Detection confidence
    detection_method: str
    timestamp: datetime
    affected_features: List[str]
    recommendation: str


@dataclass
class AdaptationMetrics:
    """Metrics for tracking adaptation performance"""
    timestamp: datetime
    accuracy_before: float
    accuracy_after: float
    adaptation_time: float  # seconds
    samples_processed: int
    drift_magnitude: float
    strategy_used: AdaptationStrategy
    model_version: str
    success: bool
    error_message: Optional[str] = None


class ConceptDriftDetector:
    """Detects concept drift using multiple statistical methods"""
    
    def __init__(self, window_size: int = 1000, sensitivity: float = 0.1):
        self.window_size = window_size
        self.sensitivity = sensitivity
        
        # Detection methods
        self.detection_methods = {
            'adwin': self._adwin_detection,
            'ddm': self._ddm_detection,
            'eddm': self._eddm_detection,
            'statistical': self._statistical_detection
        }
        
        # Historical data buffers
        self.prediction_buffer = deque(maxlen=window_size)
        self.performance_buffer = deque(maxlen=window_size)
        self.feature_buffers = {}
        
        # ADWIN parameters
        self.adwin_delta = 0.002
        self.adwin_min_window = 50
        
        # DDM parameters
        self.ddm_alpha_warning = 2.0
        self.ddm_alpha_drift = 3.0
        self.ddm_min_instances = 30
        
        # Statistical tracking
        self.baseline_stats = None
        self.last_drift_time = None
    
    def update(self, predictions: np.ndarray, ground_truth: np.ndarray, 
               features: Optional[np.ndarray] = None) -> DriftDetectionResult:
        """Update detector with new predictions and detect drift"""
        
        # Calculate prediction accuracy
        accuracy = accuracy_score(ground_truth, predictions)
        self.prediction_buffer.append(accuracy)
        
        # Update feature statistics if available
        if features is not None:
            for i, feature_val in enumerate(features.mean(axis=0)):
                if i not in self.feature_buffers:
                    self.feature_buffers[i] = deque(maxlen=self.window_size)
                self.feature_buffers[i].append(feature_val)
        
        # Run drift detection methods
        detection_results = {}
        for method_name, method_func in self.detection_methods.items():
            try:
                detection_results[method_name] = method_func()
            except Exception as e:
                logger.warning(f"Drift detection method {method_name} failed: {e}")
                detection_results[method_name] = False
        
        # Combine results using ensemble approach
        drift_detected = self._ensemble_drift_decision(detection_results)
        
        # Determine drift characteristics
        if drift_detected:
            drift_type = self._classify_drift_type()
            drift_magnitude = self._calculate_drift_magnitude()
            confidence = self._calculate_detection_confidence(detection_results)
            affected_features = self._identify_affected_features()
            recommendation = self._generate_adaptation_recommendation(drift_type, drift_magnitude)
            
            self.last_drift_time = datetime.now(timezone.utc)
            
            logger.warning(f"Concept drift detected! Type: {drift_type}, Magnitude: {drift_magnitude:.3f}")
        else:
            drift_type = ConceptDriftType.UNKNOWN
            drift_magnitude = 0.0
            confidence = 0.0
            affected_features = []
            recommendation = "Continue monitoring"
        
        return DriftDetectionResult(
            drift_detected=drift_detected,
            drift_type=drift_type,
            drift_magnitude=drift_magnitude,
            confidence=confidence,
            detection_method="ensemble",
            timestamp=datetime.now(timezone.utc),
            affected_features=affected_features,
            recommendation=recommendation
        )
    
    def _adwin_detection(self) -> bool:
        """ADWIN (Adaptive Windowing) drift detection"""
        if len(self.prediction_buffer) < self.adwin_min_window:
            return False
        
        # Simple ADWIN implementation
        # Split window and compare means
        buffer_array = np.array(self.prediction_buffer)
        mid_point = len(buffer_array) // 2
        
        mean1 = np.mean(buffer_array[:mid_point])
        mean2 = np.mean(buffer_array[mid_point:])
        
        # Calculate statistical difference
        variance = np.var(buffer_array)
        n1, n2 = mid_point, len(buffer_array) - mid_point
        
        if variance > 0:
            threshold = np.sqrt((2 * variance / min(n1, n2)) * np.log(2 / self.adwin_delta))
            return abs(mean1 - mean2) > threshold
        
        return False
    
    def _ddm_detection(self) -> bool:
        """Drift Detection Method (DDM)"""
        if len(self.prediction_buffer) < self.ddm_min_instances:
            return False
        
        # Calculate error rate and standard deviation
        errors = [1 - acc for acc in self.prediction_buffer]
        error_rate = np.mean(errors)
        error_std = np.std(errors)
        
        # Check for drift
        if error_std > 0:
            z_score = (error_rate - min(errors)) / error_std
            return z_score > self.ddm_alpha_drift
        
        return False
    
    def _eddm_detection(self) -> bool:
        """Early Drift Detection Method (EDDM)"""
        if len(self.prediction_buffer) < 30:
            return False
        
        # Calculate distances between errors
        errors = [1 - acc for acc in self.prediction_buffer]
        error_positions = [i for i, e in enumerate(errors) if e > 0.5]
        
        if len(error_positions) < 2:
            return False
        
        # Calculate average distance between errors
        distances = np.diff(error_positions)
        avg_distance = np.mean(distances[-10:]) if len(distances) >= 10 else np.mean(distances)
        historical_avg = np.mean(distances[:-5]) if len(distances) > 5 else avg_distance
        
        # Detect significant change in error pattern
        if historical_avg > 0:
            change_ratio = abs(avg_distance - historical_avg) / historical_avg
            return change_ratio > self.sensitivity * 2
        
        return False
    
    def _statistical_detection(self) -> bool:
        """Statistical significance test for drift"""
        if len(self.prediction_buffer) < 50:
            return False
        
        # Compare recent window with historical baseline
        recent_window = list(self.prediction_buffer)[-25:]
        historical_window = list(self.prediction_buffer)[:-25]
        
        if len(historical_window) < 25:
            return False
        
        # Perform two-sample t-test equivalent
        mean_recent = np.mean(recent_window)
        mean_historical = np.mean(historical_window)
        
        std_recent = np.std(recent_window)
        std_historical = np.std(historical_window)
        
        if std_recent > 0 and std_historical > 0:
            # Simplified statistical test
            pooled_std = np.sqrt((std_recent**2 + std_historical**2) / 2)
            t_stat = abs(mean_recent - mean_historical) / (pooled_std * np.sqrt(2/25))
            return t_stat > 2.0  # Approximate 95% confidence threshold
        
        return False
    
    def _ensemble_drift_decision(self, detection_results: Dict[str, bool]) -> bool:
        """Make ensemble decision on drift detection"""
        # Require majority vote for drift detection
        positive_votes = sum(detection_results.values())
        total_votes = len(detection_results)
        
        return positive_votes >= (total_votes // 2 + 1)
    
    def _classify_drift_type(self) -> ConceptDriftType:
        """Classify the type of drift based on patterns"""
        if len(self.prediction_buffer) < 100:
            return ConceptDriftType.UNKNOWN
        
        # Analyze trend in recent performance
        recent_data = np.array(list(self.prediction_buffer)[-50:])
        older_data = np.array(list(self.prediction_buffer)[-100:-50])
        
        # Check for gradual vs sudden change
        recent_trend = np.polyfit(range(len(recent_data)), recent_data, 1)[0]
        older_trend = np.polyfit(range(len(older_data)), older_data, 1)[0]
        
        trend_change = abs(recent_trend - older_trend)
        
        if trend_change > 0.01:  # Significant trend change
            return ConceptDriftType.SUDDEN
        elif abs(recent_trend) > 0.005:  # Gradual trend
            return ConceptDriftType.GRADUAL
        else:
            return ConceptDriftType.RECURRING
    
    def _calculate_drift_magnitude(self) -> float:
        """Calculate the magnitude of detected drift"""
        if len(self.prediction_buffer) < 50:
            return 0.0
        
        recent_performance = np.mean(list(self.prediction_buffer)[-25:])
        baseline_performance = np.mean(list(self.prediction_buffer)[:-25])
        
        magnitude = abs(recent_performance - baseline_performance)
        return min(magnitude, 1.0)  # Cap at 1.0
    
    def _calculate_detection_confidence(self, detection_results: Dict[str, bool]) -> float:
        """Calculate confidence in drift detection"""
        positive_votes = sum(detection_results.values())
        total_votes = len(detection_results)
        
        return positive_votes / total_votes
    
    def _identify_affected_features(self) -> List[str]:
        """Identify which features are most affected by drift"""
        affected_features = []
        
        for feature_idx, buffer in self.feature_buffers.items():
            if len(buffer) < 50:
                continue
            
            # Compare recent vs historical feature values
            recent_values = list(buffer)[-25:]
            historical_values = list(buffer)[:-25]
            
            if len(historical_values) < 25:
                continue
            
            # Calculate statistical difference
            mean_diff = abs(np.mean(recent_values) - np.mean(historical_values))
            std_pooled = np.sqrt((np.var(recent_values) + np.var(historical_values)) / 2)
            
            if std_pooled > 0 and mean_diff / std_pooled > 1.5:
                affected_features.append(f"feature_{feature_idx}")
        
        return affected_features[:5]  # Return top 5 affected features
    
    def _generate_adaptation_recommendation(self, drift_type: ConceptDriftType, 
                                          drift_magnitude: float) -> str:
        """Generate recommendation for adaptation strategy"""
        
        if drift_magnitude > 0.7:
            return "Full model retraining recommended - significant concept drift detected"
        elif drift_magnitude > 0.4:
            if drift_type == ConceptDriftType.SUDDEN:
                return "Immediate adaptation with ensemble update required"
            else:
                return "Incremental adaptation with increased learning rate"
        elif drift_magnitude > 0.2:
            return "Monitor closely and apply gradual adaptation"
        else:
            return "Continue monitoring - minor drift detected"


class OnlineLearningEngine:
    """Main engine for real-time model adaptation"""
    
    def __init__(self, adaptation_strategy: AdaptationStrategy = AdaptationStrategy.INCREMENTAL):
        self.adaptation_strategy = adaptation_strategy
        self.drift_detector = ConceptDriftDetector()
        
        # Model storage
        self.current_models = {}
        self.model_versions = {}
        self.adaptation_history = []
        
        # Online learners
        self.online_classifiers = {
            'sgd': SGDClassifier(loss='modified_huber', random_state=42),
            'passive_aggressive': PassiveAggressiveClassifier(random_state=42)
        }
        
        # Configuration
        self.min_adaptation_samples = 50
        self.max_adaptation_frequency = timedelta(minutes=5)  # Max once per 5 minutes
        self.last_adaptation_time = None
        
        # Performance tracking
        self.performance_metrics = deque(maxlen=1000)
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="online_learning")
        
        logger.info(f"Online Learning Engine initialized with strategy: {adaptation_strategy}")
    
    async def adapt_model(self, model_name: str, new_data: np.ndarray, 
                         labels: np.ndarray, features: Optional[np.ndarray] = None) -> bool:
        """Adapt model with new data"""
        
        try:
            # Check adaptation frequency limit
            if (self.last_adaptation_time and 
                datetime.now(timezone.utc) - self.last_adaptation_time < self.max_adaptation_frequency):
                logger.debug("Skipping adaptation - too frequent")
                return False
            
            # Validate input data
            if len(new_data) < self.min_adaptation_samples:
                logger.debug(f"Insufficient samples for adaptation: {len(new_data)}")
                return False
            
            start_time = time.time()
            
            # Get current model performance baseline
            accuracy_before = await self._evaluate_current_model(model_name, new_data, labels)
            
            # Detect concept drift
            predictions = await self._predict_with_current_model(model_name, new_data)
            drift_result = self.drift_detector.update(predictions, labels, features)
            
            # Decide on adaptation strategy
            should_adapt = drift_result.drift_detected or self.adaptation_strategy != AdaptationStrategy.INCREMENTAL
            
            if should_adapt:
                # Perform adaptation
                success = await self._perform_adaptation(
                    model_name, new_data, labels, drift_result
                )
                
                if success:
                    # Evaluate adapted model
                    accuracy_after = await self._evaluate_current_model(model_name, new_data, labels)
                    adaptation_time = time.time() - start_time
                    
                    # Record metrics
                    metrics = AdaptationMetrics(
                        timestamp=datetime.now(timezone.utc),
                        accuracy_before=accuracy_before,
                        accuracy_after=accuracy_after,
                        adaptation_time=adaptation_time,
                        samples_processed=len(new_data),
                        drift_magnitude=drift_result.drift_magnitude,
                        strategy_used=self.adaptation_strategy,
                        model_version=self._get_model_version(model_name),
                        success=True
                    )
                    
                    self.adaptation_history.append(metrics)
                    self.last_adaptation_time = datetime.now(timezone.utc)
                    
                    logger.info(f"Model {model_name} adapted successfully. "
                              f"Accuracy: {accuracy_before:.3f} -> {accuracy_after:.3f}")
                    
                    return True
                else:
                    logger.error(f"Failed to adapt model {model_name}")
                    return False
            else:
                logger.debug("No adaptation needed - no significant drift detected")
                return False
                
        except Exception as e:
            logger.error(f"Error during model adaptation: {e}")
            
            # Record failed adaptation
            metrics = AdaptationMetrics(
                timestamp=datetime.now(timezone.utc),
                accuracy_before=0.0,
                accuracy_after=0.0,
                adaptation_time=0.0,
                samples_processed=len(new_data),
                drift_magnitude=0.0,
                strategy_used=self.adaptation_strategy,
                model_version=self._get_model_version(model_name),
                success=False,
                error_message=str(e)
            )
            self.adaptation_history.append(metrics)
            
            return False
    
    async def _perform_adaptation(self, model_name: str, new_data: np.ndarray, 
                                 labels: np.ndarray, drift_result: DriftDetectionResult) -> bool:
        """Perform the actual model adaptation"""
        
        try:
            if self.adaptation_strategy == AdaptationStrategy.INCREMENTAL:
                return await self._incremental_adaptation(model_name, new_data, labels)
            
            elif self.adaptation_strategy == AdaptationStrategy.MINI_BATCH:
                return await self._mini_batch_adaptation(model_name, new_data, labels)
            
            elif self.adaptation_strategy == AdaptationStrategy.SLIDING_WINDOW:
                return await self._sliding_window_adaptation(model_name, new_data, labels)
            
            elif self.adaptation_strategy == AdaptationStrategy.ENSEMBLE_VOTING:
                return await self._ensemble_adaptation(model_name, new_data, labels, drift_result)
            
            else:
                logger.error(f"Unknown adaptation strategy: {self.adaptation_strategy}")
                return False
                
        except Exception as e:
            logger.error(f"Adaptation strategy failed: {e}")
            return False
    
    async def _incremental_adaptation(self, model_name: str, new_data: np.ndarray, 
                                    labels: np.ndarray) -> bool:
        """Perform incremental online learning"""
        
        # Use online classifiers for incremental learning
        for classifier_name, classifier in self.online_classifiers.items():
            try:
                # Fit incrementally
                classifier.partial_fit(new_data, labels)
                logger.debug(f"Incremental fit completed for {classifier_name}")
            except Exception as e:
                logger.warning(f"Incremental fit failed for {classifier_name}: {e}")
        
        # Update model version
        self._increment_model_version(model_name)
        return True
    
    async def _mini_batch_adaptation(self, model_name: str, new_data: np.ndarray, 
                                   labels: np.ndarray) -> bool:
        """Perform mini-batch adaptation"""
        
        batch_size = min(100, len(new_data) // 2)
        
        # Process in mini-batches
        for i in range(0, len(new_data), batch_size):
            batch_data = new_data[i:i + batch_size]
            batch_labels = labels[i:i + batch_size]
            
            # Update each online classifier
            for classifier_name, classifier in self.online_classifiers.items():
                try:
                    classifier.partial_fit(batch_data, batch_labels)
                except Exception as e:
                    logger.warning(f"Mini-batch adaptation failed for {classifier_name}: {e}")
        
        self._increment_model_version(model_name)
        return True
    
    async def _sliding_window_adaptation(self, model_name: str, new_data: np.ndarray, 
                                       labels: np.ndarray) -> bool:
        """Perform sliding window retraining"""
        
        # This would require storing historical data
        # For now, implement as mini-batch adaptation
        logger.info("Sliding window adaptation - using mini-batch approach")
        return await self._mini_batch_adaptation(model_name, new_data, labels)
    
    async def _ensemble_adaptation(self, model_name: str, new_data: np.ndarray, 
                                 labels: np.ndarray, drift_result: DriftDetectionResult) -> bool:
        """Perform ensemble-based adaptation"""
        
        # Train multiple models and combine
        ensemble_results = {}
        
        for classifier_name, classifier in self.online_classifiers.items():
            try:
                # Clone and train new version
                new_classifier = clone(classifier)
                new_classifier.fit(new_data, labels)
                
                # Evaluate performance
                predictions = new_classifier.predict(new_data)
                accuracy = accuracy_score(labels, predictions)
                
                ensemble_results[classifier_name] = {
                    'model': new_classifier,
                    'accuracy': accuracy
                }
                
            except Exception as e:
                logger.warning(f"Ensemble training failed for {classifier_name}: {e}")
        
        # Select best performing model or combine
        if ensemble_results:
            best_model = max(ensemble_results.values(), key=lambda x: x['accuracy'])
            # Update the classifier with best performing one
            best_name = [k for k, v in ensemble_results.items() if v == best_model][0]
            self.online_classifiers[best_name] = best_model['model']
            
            self._increment_model_version(model_name)
            return True
        
        return False
    
    async def _evaluate_current_model(self, model_name: str, data: np.ndarray, 
                                    labels: np.ndarray) -> float:
        """Evaluate current model performance"""
        
        try:
            predictions = await self._predict_with_current_model(model_name, data)
            return accuracy_score(labels, predictions)
        except Exception as e:
            logger.warning(f"Model evaluation failed: {e}")
            return 0.0
    
    async def _predict_with_current_model(self, model_name: str, data: np.ndarray) -> np.ndarray:
        """Make predictions with current model"""
        
        # Use ensemble of online classifiers
        predictions_list = []
        
        for classifier_name, classifier in self.online_classifiers.items():
            try:
                pred = classifier.predict(data)
                predictions_list.append(pred)
            except Exception as e:
                logger.warning(f"Prediction failed for {classifier_name}: {e}")
        
        if predictions_list:
            # Simple majority voting
            stacked_predictions = np.stack(predictions_list)
            return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=stacked_predictions)
        else:
            # Fallback: return random predictions
            return np.random.randint(0, 2, size=len(data))
    
    def _get_model_version(self, model_name: str) -> str:
        """Get current model version"""
        return self.model_versions.get(model_name, "v1.0")
    
    def _increment_model_version(self, model_name: str):
        """Increment model version"""
        current_version = self.model_versions.get(model_name, "v1.0")
        
        # Extract version number and increment
        try:
            version_num = float(current_version[1:])  # Remove 'v' prefix
            new_version = f"v{version_num + 0.1:.1f}"
            self.model_versions[model_name] = new_version
        except:
            self.model_versions[model_name] = "v1.1"
    
    def get_adaptation_metrics(self) -> List[Dict[str, Any]]:
        """Get adaptation performance metrics"""
        return [asdict(metric) for metric in self.adaptation_history[-50:]]  # Last 50 adaptations
    
    def get_drift_status(self) -> Dict[str, Any]:
        """Get current drift detection status"""
        return {
            'buffer_size': len(self.drift_detector.prediction_buffer),
            'last_drift_time': self.drift_detector.last_drift_time,
            'detection_sensitivity': self.drift_detector.sensitivity,
            'window_size': self.drift_detector.window_size
        }
    
    async def shutdown(self):
        """Shutdown the online learning engine"""
        self.executor.shutdown(wait=True)
        logger.info("Online Learning Engine shutdown complete")


# Global instance
online_learning_engine = OnlineLearningEngine()


async def initialize_online_learning():
    """Initialize online learning system"""
    try:
        logger.info("Initializing Online Learning System")
        # Additional initialization logic can go here
        return True
    except Exception as e:
        logger.error(f"Failed to initialize online learning: {e}")
        return False


async def adapt_models_with_new_data(events: List[Event]) -> Dict[str, Any]:
    """Adapt models using new event data"""
    
    if len(events) < online_learning_engine.min_adaptation_samples:
        return {'success': False, 'reason': 'Insufficient samples'}
    
    try:
        # Convert events to feature vectors
        # This is a simplified feature extraction - should be enhanced
        features = []
        labels = []
        
        for event in events:
            # Extract basic features
            feature_vector = [
                hash(event.src_ip) % 1000,  # IP hash feature
                event.dst_port or 0,        # Port feature
                len(event.message or ""),   # Message length
                1 if event.anomaly_score and event.anomaly_score > 0.5 else 0  # Anomaly flag
            ]
            
            features.append(feature_vector)
            
            # Label based on anomaly score (this should be improved with ground truth)
            label = 1 if event.anomaly_score and event.anomaly_score > 0.7 else 0
            labels.append(label)
        
        # Convert to numpy arrays
        feature_array = np.array(features)
        label_array = np.array(labels)
        
        # Perform adaptation
        success = await online_learning_engine.adapt_model(
            model_name="threat_detection",
            new_data=feature_array,
            labels=label_array,
            features=feature_array
        )
        
        return {
            'success': success,
            'samples_processed': len(events),
            'adaptation_strategy': online_learning_engine.adaptation_strategy,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to adapt models with new data: {e}")
        return {'success': False, 'error': str(e)}


if __name__ == "__main__":
    # Test the online learning system
    async def test_online_learning():
        print("Testing Online Learning System...")
        
        # Initialize
        await initialize_online_learning()
        
        # Generate test data
        test_data = np.random.rand(100, 4)
        test_labels = np.random.randint(0, 2, 100)
        
        # Test adaptation
        result = await online_learning_engine.adapt_model(
            model_name="test_model",
            new_data=test_data,
            labels=test_labels,
            features=test_data
        )
        
        print(f"Adaptation result: {result}")
        
        # Get metrics
        metrics = online_learning_engine.get_adaptation_metrics()
        print(f"Adaptation metrics: {len(metrics)} records")
        
        # Get drift status
        drift_status = online_learning_engine.get_drift_status()
        print(f"Drift status: {drift_status}")
        
        # Shutdown
        await online_learning_engine.shutdown()
    
    # Run test
    asyncio.run(test_online_learning())

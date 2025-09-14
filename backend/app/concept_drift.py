"""
Advanced Concept Drift Detection for Real-Time ML Adaptation
Implements multiple drift detection algorithms with ensemble decision making
"""

import numpy as np
import pandas as pd
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    stats = None
    logging.warning("SciPy not available - using fallback statistical functions")

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timezone
from collections import deque
import logging
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class DriftAlgorithm(str, Enum):
    """Available drift detection algorithms"""
    ADWIN = "adwin"
    DDM = "ddm" 
    EDDM = "eddm"
    HDDM_A = "hddm_a"
    HDDM_W = "hddm_w"
    KSWIN = "kswin"
    PAGE_HINKLEY = "page_hinkley"
    STATISTICAL = "statistical"


@dataclass
class DriftAlert:
    """Drift detection alert"""
    algorithm: DriftAlgorithm
    drift_detected: bool
    warning_detected: bool
    drift_magnitude: float
    confidence: float
    timestamp: datetime
    data_points_processed: int
    affected_dimensions: List[int]
    alert_message: str


class ADWINDetector:
    """Adaptive Windowing (ADWIN) drift detector"""
    
    def __init__(self, delta: float = 0.002):
        self.delta = delta
        self.window = deque()
        self.total = 0.0
        self.variance = 0.0
        self.width = 0
        self.drift_detected = False
        
    def add_element(self, value: float) -> bool:
        """Add new element and check for drift"""
        self.window.append(value)
        self.width += 1
        
        if self.width == 1:
            self.total = value
            self.variance = 0.0
        else:
            self.total += value
            mean = self.total / self.width
            
            # Update variance incrementally
            diff = value - mean
            self.variance += diff * diff
        
        # Check for drift using ADWIN algorithm
        self.drift_detected = self._detect_change()
        
        return self.drift_detected
    
    def _detect_change(self) -> bool:
        """Detect concept drift using ADWIN method"""
        if self.width < 10:  # Minimum window size
            return False
        
        # Try different cut points
        for i in range(5, self.width - 5):
            # Split window at point i
            left_sum = sum(list(self.window)[:i])
            right_sum = sum(list(self.window)[i:])
            
            left_mean = left_sum / i
            right_mean = right_sum / (self.width - i)
            
            # Calculate bound for significant difference
            m = 1.0 / (1.0/i + 1.0/(self.width - i))
            delta_prime = np.sqrt((2.0 * self.variance / self.width) * (1.0/m) * np.log(2.0/self.delta))
            
            if abs(left_mean - right_mean) >= delta_prime:
                # Drift detected, remove old data
                for _ in range(i):
                    removed = self.window.popleft()
                    self.total -= removed
                    self.width -= 1
                return True
        
        return False


class DDMDetector:
    """Drift Detection Method (DDM)"""
    
    def __init__(self, alpha_warning: float = 2.0, alpha_drift: float = 3.0):
        self.alpha_warning = alpha_warning
        self.alpha_drift = alpha_drift
        
        self.n_min = 30  # Minimum number of examples
        self.n = 0
        self.p = 1.0  # Error rate
        self.s = 0.0  # Standard deviation
        
        # Historical values
        self.p_min = float('inf')
        self.s_min = float('inf')
        
        self.in_warning = False
        self.drift_detected = False
    
    def add_element(self, prediction: int, true_value: int) -> bool:
        """Add new prediction and true value"""
        self.n += 1
        
        # Calculate error (1 if wrong prediction, 0 if correct)
        error = 1 if prediction != true_value else 0
        
        # Update error rate
        self.p += (error - self.p) / self.n
        
        # Update standard deviation
        self.s = np.sqrt(self.p * (1 - self.p) / self.n)
        
        # Check if we have enough samples
        if self.n < self.n_min:
            return False
        
        # Update minimum values
        if self.p + self.s < self.p_min + self.s_min:
            self.p_min = self.p
            self.s_min = self.s
        
        # Check for drift
        if self.p + self.s > self.p_min + self.alpha_drift * self.s_min:
            self.drift_detected = True
            self._reset()
            return True
        
        # Check for warning
        if self.p + self.s > self.p_min + self.alpha_warning * self.s_min:
            self.in_warning = True
        else:
            self.in_warning = False
        
        return False
    
    def _reset(self):
        """Reset detector after drift detection"""
        self.n = 0
        self.p = 1.0
        self.s = 0.0
        self.p_min = float('inf')
        self.s_min = float('inf')
        self.in_warning = False
        self.drift_detected = False


class PageHinkleyDetector:
    """Page-Hinkley Test for drift detection"""
    
    def __init__(self, min_instances: int = 30, delta: float = 0.005, threshold: float = 50, alpha: float = 0.9999):
        self.min_instances = min_instances
        self.delta = delta
        self.threshold = threshold
        self.alpha = alpha
        
        # Internal state
        self.n = 0
        self.x_sum = 0.0
        self.x_mean = 0.0
        self.sum_positive = 0.0
        self.sum_negative = 0.0
        
        self.drift_detected = False
    
    def add_element(self, value: float) -> bool:
        """Add new value and check for drift"""
        self.n += 1
        self.x_sum += value
        
        if self.n >= self.min_instances:
            self.x_mean = self.x_sum / self.n
            
            # Calculate cumulative sums
            diff = value - self.x_mean - self.delta
            self.sum_positive = max(0, self.alpha * self.sum_positive + diff)
            
            diff = self.x_mean - value - self.delta
            self.sum_negative = max(0, self.alpha * self.sum_negative + diff)
            
            # Check for drift
            if self.sum_positive > self.threshold or self.sum_negative > self.threshold:
                self.drift_detected = True
                self._reset()
                return True
        
        return False
    
    def _reset(self):
        """Reset detector"""
        self.n = 0
        self.x_sum = 0.0
        self.x_mean = 0.0
        self.sum_positive = 0.0
        self.sum_negative = 0.0
        self.drift_detected = False


class KSWINDetector:
    """Kolmogorov-Smirnov Windowing (KSWIN) detector"""
    
    def __init__(self, alpha: float = 0.005, window_size: int = 100, stat_size: int = 30):
        self.alpha = alpha
        self.window_size = window_size
        self.stat_size = stat_size
        
        self.window = deque(maxlen=window_size)
        self.drift_detected = False
    
    def add_element(self, value: float) -> bool:
        """Add element and check for drift"""
        self.window.append(value)
        
        if len(self.window) >= 2 * self.stat_size:
            # Get two samples for comparison
            recent_sample = list(self.window)[-self.stat_size:]
            older_sample = list(self.window)[-2*self.stat_size:-self.stat_size]
            
            # Perform Kolmogorov-Smirnov test
            try:
                if SCIPY_AVAILABLE:
                    ks_statistic, p_value = stats.ks_2samp(older_sample, recent_sample)
                    
                    if p_value < self.alpha:
                        self.drift_detected = True
                        return True
                else:
                    # Fallback: simple statistical comparison without KS test
                    mean_diff = abs(np.mean(recent_sample) - np.mean(older_sample))
                    std_pooled = np.sqrt((np.var(recent_sample) + np.var(older_sample)) / 2)
                    
                    if std_pooled > 0:
                        z_score = mean_diff / std_pooled
                        # Simplified threshold (equivalent to p < 0.05 for normal distribution)
                        if z_score > 1.96:
                            self.drift_detected = True
                            return True
                    
            except Exception as e:
                logger.warning(f"Statistical test failed: {e}")
        
        return False


class EnsembleDriftDetector:
    """Ensemble drift detector combining multiple algorithms"""
    
    def __init__(self, algorithms: Optional[List[DriftAlgorithm]] = None, 
                 voting_threshold: float = 0.5):
        
        # Default algorithms to use
        if algorithms is None:
            algorithms = [
                DriftAlgorithm.ADWIN,
                DriftAlgorithm.DDM,
                DriftAlgorithm.PAGE_HINKLEY,
                DriftAlgorithm.KSWIN
            ]
        
        self.algorithms = algorithms
        self.voting_threshold = voting_threshold
        
        # Initialize detectors
        self.detectors = {}
        self._initialize_detectors()
        
        # State tracking
        self.data_points = 0
        self.alerts_history = deque(maxlen=1000)
        
    def _initialize_detectors(self):
        """Initialize all drift detection algorithms"""
        
        for algorithm in self.algorithms:
            if algorithm == DriftAlgorithm.ADWIN:
                self.detectors[algorithm] = ADWINDetector()
            
            elif algorithm == DriftAlgorithm.DDM:
                self.detectors[algorithm] = DDMDetector()
            
            elif algorithm == DriftAlgorithm.PAGE_HINKLEY:
                self.detectors[algorithm] = PageHinkleyDetector()
            
            elif algorithm == DriftAlgorithm.KSWIN:
                self.detectors[algorithm] = KSWINDetector()
            
            # Add more detectors as needed
            
        logger.info(f"Initialized {len(self.detectors)} drift detectors")
    
    def add_element(self, value: float, prediction: Optional[int] = None, 
                   true_value: Optional[int] = None) -> Dict[str, DriftAlert]:
        """Add new data point and check all detectors"""
        
        self.data_points += 1
        alerts = {}
        
        for algorithm, detector in self.detectors.items():
            drift_detected = False
            
            try:
                # Call appropriate method based on detector type
                if algorithm == DriftAlgorithm.DDM and prediction is not None and true_value is not None:
                    drift_detected = detector.add_element(prediction, true_value)
                else:
                    drift_detected = detector.add_element(value)
                
                # Create alert
                alert = DriftAlert(
                    algorithm=algorithm,
                    drift_detected=drift_detected,
                    warning_detected=getattr(detector, 'in_warning', False),
                    drift_magnitude=self._calculate_drift_magnitude(algorithm, detector),
                    confidence=0.8 if drift_detected else 0.2,
                    timestamp=datetime.now(timezone.utc),
                    data_points_processed=self.data_points,
                    affected_dimensions=[0],  # Single dimension for now
                    alert_message=f"Drift {'detected' if drift_detected else 'not detected'} by {algorithm}"
                )
                
                alerts[algorithm.value] = alert
                
                if drift_detected:
                    logger.warning(f"Drift detected by {algorithm}")
                
            except Exception as e:
                logger.error(f"Error in detector {algorithm}: {e}")
        
        # Store alerts
        self.alerts_history.extend(alerts.values())
        
        return alerts
    
    def _calculate_drift_magnitude(self, algorithm: DriftAlgorithm, detector) -> float:
        """Calculate drift magnitude based on detector state"""
        
        if algorithm == DriftAlgorithm.ADWIN:
            return min(detector.variance, 1.0)
        
        elif algorithm == DriftAlgorithm.DDM:
            if hasattr(detector, 'p') and hasattr(detector, 'p_min'):
                return min(abs(detector.p - detector.p_min), 1.0)
        
        elif algorithm == DriftAlgorithm.PAGE_HINKLEY:
            if hasattr(detector, 'sum_positive') and hasattr(detector, 'sum_negative'):
                return min((detector.sum_positive + detector.sum_negative) / 100.0, 1.0)
        
        return 0.5  # Default magnitude
    
    def get_ensemble_decision(self, recent_alerts: Optional[Dict[str, DriftAlert]] = None) -> DriftAlert:
        """Get ensemble decision on drift detection"""
        
        if recent_alerts is None:
            # Get most recent alert for each algorithm
            recent_alerts = {}
            for algorithm in self.algorithms:
                for alert in reversed(self.alerts_history):
                    if alert.algorithm == algorithm:
                        recent_alerts[algorithm.value] = alert
                        break
        
        # Count positive detections
        drift_votes = sum(1 for alert in recent_alerts.values() if alert.drift_detected)
        total_votes = len(recent_alerts)
        
        # Ensemble decision
        ensemble_drift_detected = (drift_votes / total_votes) >= self.voting_threshold
        
        # Calculate ensemble metrics
        avg_magnitude = np.mean([alert.drift_magnitude for alert in recent_alerts.values()])
        avg_confidence = np.mean([alert.confidence for alert in recent_alerts.values()])
        
        # Affected algorithms
        detecting_algorithms = [alert.algorithm.value for alert in recent_alerts.values() if alert.drift_detected]
        
        return DriftAlert(
            algorithm=DriftAlgorithm.STATISTICAL,  # Ensemble uses statistical approach
            drift_detected=ensemble_drift_detected,
            warning_detected=drift_votes > 0,
            drift_magnitude=avg_magnitude,
            confidence=avg_confidence,
            timestamp=datetime.now(timezone.utc),
            data_points_processed=self.data_points,
            affected_dimensions=[0],
            alert_message=f"Ensemble decision: {drift_votes}/{total_votes} algorithms detected drift ({detecting_algorithms})"
        )
    
    def add_batch(self, values: List[float], predictions: Optional[List[int]] = None,
                 true_values: Optional[List[int]] = None) -> List[Dict[str, DriftAlert]]:
        """Process batch of values"""
        
        results = []
        
        for i, value in enumerate(values):
            pred = predictions[i] if predictions else None
            true_val = true_values[i] if true_values else None
            
            alerts = self.add_element(value, pred, true_val)
            results.append(alerts)
        
        return results
    
    def get_detector_status(self) -> Dict[str, Any]:
        """Get status of all detectors"""
        
        status = {
            'total_data_points': self.data_points,
            'active_detectors': len(self.detectors),
            'recent_alerts': len([a for a in self.alerts_history if a.drift_detected]),
            'algorithms': list(self.algorithms),
            'voting_threshold': self.voting_threshold
        }
        
        # Add detector-specific status
        for algorithm, detector in self.detectors.items():
            algo_status = {'type': algorithm.value}
            
            if hasattr(detector, 'window'):
                algo_status['window_size'] = len(detector.window)
            
            if hasattr(detector, 'n'):
                algo_status['samples_processed'] = detector.n
            
            if hasattr(detector, 'in_warning'):
                algo_status['warning_state'] = detector.in_warning
            
            status[f'detector_{algorithm.value}'] = algo_status
        
        return status
    
    def reset_all_detectors(self):
        """Reset all detectors to initial state"""
        
        self._initialize_detectors()
        self.data_points = 0
        self.alerts_history.clear()
        
        logger.info("All drift detectors reset")
    
    def get_drift_timeline(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get drift detection timeline for the last N hours"""
        
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        timeline = []
        for alert in self.alerts_history:
            if alert.timestamp >= cutoff_time:
                timeline.append({
                    'timestamp': alert.timestamp.isoformat(),
                    'algorithm': alert.algorithm.value,
                    'drift_detected': alert.drift_detected,
                    'magnitude': alert.drift_magnitude,
                    'confidence': alert.confidence
                })
        
        return sorted(timeline, key=lambda x: x['timestamp'])


# Utility functions for integration

def create_drift_detector(algorithms: Optional[List[str]] = None, 
                         voting_threshold: float = 0.5) -> EnsembleDriftDetector:
    """Create a drift detector with specified algorithms"""
    
    if algorithms:
        algorithm_enums = [DriftAlgorithm(algo) for algo in algorithms]
    else:
        algorithm_enums = None
    
    return EnsembleDriftDetector(algorithm_enums, voting_threshold)


def analyze_drift_patterns(alerts: List[DriftAlert]) -> Dict[str, Any]:
    """Analyze patterns in drift detection alerts"""
    
    if not alerts:
        return {'no_data': True}
    
    # Count detections by algorithm
    by_algorithm = {}
    for alert in alerts:
        algo = alert.algorithm.value
        if algo not in by_algorithm:
            by_algorithm[algo] = {'total': 0, 'drift': 0, 'warnings': 0}
        
        by_algorithm[algo]['total'] += 1
        if alert.drift_detected:
            by_algorithm[algo]['drift'] += 1
        if alert.warning_detected:
            by_algorithm[algo]['warnings'] += 1
    
    # Time-based analysis
    timestamps = [alert.timestamp for alert in alerts]
    if len(timestamps) > 1:
        time_span = max(timestamps) - min(timestamps)
        drift_frequency = sum(1 for a in alerts if a.drift_detected) / max(time_span.total_seconds() / 3600, 1)  # Per hour
    else:
        drift_frequency = 0
    
    # Magnitude analysis
    magnitudes = [alert.drift_magnitude for alert in alerts if alert.drift_detected]
    avg_magnitude = np.mean(magnitudes) if magnitudes else 0
    max_magnitude = max(magnitudes) if magnitudes else 0
    
    return {
        'total_alerts': len(alerts),
        'drift_detections': sum(1 for a in alerts if a.drift_detected),
        'warning_alerts': sum(1 for a in alerts if a.warning_detected),
        'by_algorithm': by_algorithm,
        'drift_frequency_per_hour': drift_frequency,
        'average_magnitude': avg_magnitude,
        'maximum_magnitude': max_magnitude,
        'time_span_hours': time_span.total_seconds() / 3600 if 'time_span' in locals() else 0
    }


if __name__ == "__main__":
    # Test the drift detection system
    try:
        import matplotlib.pyplot as plt
        MATPLOTLIB_AVAILABLE = True
    except ImportError:
        MATPLOTLIB_AVAILABLE = False
        plt = None
    
    print("Testing Concept Drift Detection System...")
    
    # Create detector
    detector = create_drift_detector()
    
    # Generate test data with concept drift
    np.random.seed(42)
    
    # Phase 1: Normal data
    normal_data = np.random.normal(0, 1, 200)
    
    # Phase 2: Shifted data (concept drift)
    shifted_data = np.random.normal(2, 1.5, 200)
    
    # Phase 3: Return to normal
    return_data = np.random.normal(0.5, 1, 200)
    
    # Combine data
    test_data = np.concatenate([normal_data, shifted_data, return_data])
    
    # Process data and collect results
    all_alerts = []
    ensemble_decisions = []
    
    print("Processing test data...")
    for i, value in enumerate(test_data):
        alerts = detector.add_element(value)
        all_alerts.extend(alerts.values())
        
        # Get ensemble decision every 10 points
        if i % 10 == 0:
            ensemble_decision = detector.get_ensemble_decision()
            ensemble_decisions.append((i, ensemble_decision.drift_detected, ensemble_decision.drift_magnitude))
    
    # Analyze results
    print(f"\nProcessed {len(test_data)} data points")
    print(f"Total alerts: {len(all_alerts)}")
    print(f"Drift detections: {sum(1 for a in all_alerts if a.drift_detected)}")
    
    # Get status
    status = detector.get_detector_status()
    print(f"Detector status: {status}")
    
    # Analyze patterns
    patterns = analyze_drift_patterns(all_alerts)
    print(f"Drift patterns: {patterns}")
    
    if MATPLOTLIB_AVAILABLE:
        print("Concept drift detection test completed with visualization!")
    else:
        print("Concept drift detection test completed (visualization unavailable)!")

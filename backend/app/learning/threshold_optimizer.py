"""
Per-Class Threshold Optimization

Instead of using a fixed 0.5 threshold for all classes, this module finds optimal
decision thresholds that maximize F1 score (or other metrics) for each class.

This is crucial for imbalanced datasets where different classes have different
optimal operating points.

Example:
- Normal traffic: High threshold (0.9) to avoid false positives
- Critical attacks: Lower threshold (0.3) to catch more threats
- Rare attacks: Very low threshold (0.2) to avoid missing them

Usage:
```python
from app.learning.threshold_optimizer import ThresholdOptimizer

# On validation set after training
optimizer = ThresholdOptimizer(metric='f1')
thresholds = optimizer.optimize(val_probs, val_labels)

# Apply to model
model_manager.set_per_class_thresholds(thresholds)
```
"""

import logging
from collections import defaultdict
from typing import Callable, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Check if scikit-optimize is available
try:
    from skopt import gp_minimize
    from skopt.space import Real
    from skopt.utils import use_named_args

    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False
    logger.warning(
        "scikit-optimize not installed. Install with: " "pip install scikit-optimize"
    )


class ThresholdOptimizer:
    """
    Optimize decision thresholds per class to maximize a given metric.

    Supports:
    - F1 score (default, balances precision and recall)
    - Precision (minimize false positives)
    - Recall (minimize false negatives)
    - Matthews Correlation Coefficient (MCC, handles imbalance well)
    """

    def __init__(
        self, metric: str = "f1", search_method: str = "grid", n_thresholds: int = 100
    ):
        """
        Initialize threshold optimizer.

        Args:
            metric: Metric to optimize ('f1', 'precision', 'recall', 'mcc')
            search_method: 'grid' (fast, good) or 'bayesian' (slower, better)
            n_thresholds: Number of thresholds to try (for grid search)
        """
        self.metric = metric
        self.search_method = search_method
        self.n_thresholds = n_thresholds

        self.metric_functions = {
            "f1": self._calculate_f1,
            "precision": self._calculate_precision,
            "recall": self._calculate_recall,
            "mcc": self._calculate_mcc,
        }

        if metric not in self.metric_functions:
            raise ValueError(
                f"Unknown metric: {metric}. Choose from {list(self.metric_functions.keys())}"
            )

        logger.info(
            f"ThresholdOptimizer initialized: metric={metric}, "
            f"method={search_method}, n_thresholds={n_thresholds}"
        )

    def optimize(
        self, probabilities: np.ndarray, labels: np.ndarray, verbose: bool = True
    ) -> Dict[int, float]:
        """
        Find optimal thresholds for each class.

        Args:
            probabilities: Predicted probabilities [n_samples, n_classes]
            labels: True labels [n_samples]
            verbose: Print progress

        Returns:
            Dict mapping class_id → optimal_threshold
        """
        n_classes = probabilities.shape[1]
        optimal_thresholds = {}

        for class_id in range(n_classes):
            # Binary problem: class_id vs rest
            binary_probs = probabilities[:, class_id]
            binary_labels = (labels == class_id).astype(int)

            # Find optimal threshold for this class
            if self.search_method == "grid":
                threshold, score = self._grid_search(binary_probs, binary_labels)
            elif self.search_method == "bayesian" and SKOPT_AVAILABLE:
                threshold, score = self._bayesian_search(binary_probs, binary_labels)
            else:
                # Fallback to grid search
                threshold, score = self._grid_search(binary_probs, binary_labels)

            optimal_thresholds[class_id] = threshold

            if verbose:
                # Count samples for this class
                n_positive = np.sum(binary_labels)
                logger.info(
                    f"Class {class_id}: threshold={threshold:.4f}, "
                    f"{self.metric}={score:.4f}, samples={n_positive}"
                )

        return optimal_thresholds

    def _grid_search(
        self, probabilities: np.ndarray, labels: np.ndarray
    ) -> Tuple[float, float]:
        """
        Grid search over threshold values.

        Args:
            probabilities: Probabilities for one class [n_samples]
            labels: Binary labels [n_samples]

        Returns:
            (best_threshold, best_score) tuple
        """
        thresholds = np.linspace(0.01, 0.99, self.n_thresholds)
        best_threshold = 0.5
        best_score = -np.inf

        metric_func = self.metric_functions[self.metric]

        for threshold in thresholds:
            predictions = (probabilities >= threshold).astype(int)
            score = metric_func(labels, predictions)

            if score > best_score:
                best_score = score
                best_threshold = threshold

        return best_threshold, best_score

    def _bayesian_search(
        self, probabilities: np.ndarray, labels: np.ndarray, n_calls: int = 50
    ) -> Tuple[float, float]:
        """
        Bayesian optimization of threshold (uses Gaussian processes).

        Args:
            probabilities: Probabilities for one class [n_samples]
            labels: Binary labels [n_samples]
            n_calls: Number of optimization iterations

        Returns:
            (best_threshold, best_score) tuple
        """
        if not SKOPT_AVAILABLE:
            logger.warning(
                "Bayesian search requires scikit-optimize, using grid search"
            )
            return self._grid_search(probabilities, labels)

        metric_func = self.metric_functions[self.metric]

        # Define search space
        space = [Real(0.01, 0.99, name="threshold")]

        @use_named_args(space)
        def objective(threshold):
            predictions = (probabilities >= threshold).astype(int)
            score = metric_func(labels, predictions)
            return -score  # Minimize negative score = maximize score

        # Run optimization
        result = gp_minimize(objective, space, n_calls=n_calls, random_state=42)

        best_threshold = result.x[0]
        best_score = -result.fun

        return best_threshold, best_score

    def _calculate_f1(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate F1 score (harmonic mean of precision and recall)."""
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if precision + recall == 0:
            return 0.0

        f1 = 2 * (precision * recall) / (precision + recall)
        return f1

    def _calculate_precision(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate precision (minimize false positives)."""
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))

        return tp / (tp + fp) if (tp + fp) > 0 else 0.0

    def _calculate_recall(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate recall (minimize false negatives)."""
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

    def _calculate_mcc(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Matthews Correlation Coefficient.

        MCC is a balanced measure that works well for imbalanced datasets.
        Range: [-1, 1] where 1 is perfect, 0 is random, -1 is inverse.
        """
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        numerator = (tp * tn) - (fp * fn)
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

        if denominator == 0:
            return 0.0

        mcc = numerator / denominator
        return mcc

    def evaluate_thresholds(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray,
        thresholds: Dict[int, float],
    ) -> Dict[str, float]:
        """
        Evaluate performance with given thresholds.

        Args:
            probabilities: Model probabilities [n_samples, n_classes]
            labels: True labels [n_samples]
            thresholds: Per-class thresholds

        Returns:
            Dict with per-class and overall metrics
        """
        n_classes = probabilities.shape[1]
        n_samples = len(labels)

        # Make predictions using per-class thresholds
        predictions = np.zeros(n_samples, dtype=int)

        for i in range(n_samples):
            # Find class with highest probability that exceeds its threshold
            max_class = -1
            max_prob = -1

            for class_id in range(n_classes):
                prob = probabilities[i, class_id]
                threshold = thresholds.get(class_id, 0.5)

                if prob >= threshold and prob > max_prob:
                    max_prob = prob
                    max_class = class_id

            # If no class meets threshold, use argmax
            if max_class == -1:
                max_class = np.argmax(probabilities[i])

            predictions[i] = max_class

        # Calculate metrics
        metrics = {}

        # Overall accuracy
        metrics["accuracy"] = np.mean(predictions == labels)

        # Per-class metrics
        for class_id in range(n_classes):
            binary_labels = (labels == class_id).astype(int)
            binary_preds = (predictions == class_id).astype(int)

            f1 = self._calculate_f1(binary_labels, binary_preds)
            precision = self._calculate_precision(binary_labels, binary_preds)
            recall = self._calculate_recall(binary_labels, binary_preds)

            metrics[f"class_{class_id}_f1"] = f1
            metrics[f"class_{class_id}_precision"] = precision
            metrics[f"class_{class_id}_recall"] = recall

        # Macro-averaged metrics (simple average across classes)
        metrics["macro_f1"] = np.mean(
            [metrics[f"class_{i}_f1"] for i in range(n_classes)]
        )
        metrics["macro_precision"] = np.mean(
            [metrics[f"class_{i}_precision"] for i in range(n_classes)]
        )
        metrics["macro_recall"] = np.mean(
            [metrics[f"class_{i}_recall"] for i in range(n_classes)]
        )

        return metrics


def optimize_thresholds_simple(
    probabilities: np.ndarray, labels: np.ndarray, metric: str = "f1"
) -> Dict[int, float]:
    """
    Simple function to optimize thresholds.

    Args:
        probabilities: Model probabilities [n_samples, n_classes]
        labels: True labels [n_samples]
        metric: Metric to optimize

    Returns:
        Per-class thresholds
    """
    optimizer = ThresholdOptimizer(metric=metric, search_method="grid")
    return optimizer.optimize(probabilities, labels)


__all__ = [
    "ThresholdOptimizer",
    "optimize_thresholds_simple",
    "SKOPT_AVAILABLE",
]

"""
Weighted Loss Functions for Class-Imbalanced Training

Implements focal loss and class-weighted cross-entropy to address the 79.6% class imbalance
in the CICIDS2017 dataset.

Key Features:
- Focal Loss: Focus on hard-to-classify examples
- Class Weights: Inversely proportional to class frequency
- Label Smoothing: Prevent overconfidence
- Temperature Scaling: Calibrate probability outputs

Usage in Training:
```python
from app.learning.weighted_loss import FocalLoss, calculate_class_weights

# Calculate weights from training data
class_weights = calculate_class_weights(y_train)

# Use focal loss instead of CrossEntropyLoss
criterion = FocalLoss(alpha=class_weights, gamma=2.0)

# In training loop
loss = criterion(outputs, targets)
```
"""

import logging
from collections import Counter
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.

    Paper: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    https://arxiv.org/abs/1708.02002

    Focal Loss = -α_t * (1 - p_t)^γ * log(p_t)

    Where:
    - α_t: Class weight (higher for minority classes)
    - γ: Focusing parameter (typically 2.0)
    - p_t: Predicted probability for the true class

    The (1 - p_t)^γ term down-weights easy examples and focuses on hard ones.
    """

    def __init__(
        self,
        alpha: Optional[Union[float, torch.Tensor]] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ):
        """
        Initialize Focal Loss.

        Args:
            alpha: Class weights (Tensor of shape [num_classes])
            gamma: Focusing parameter (0 = standard CE, higher = more focus on hard)
            reduction: 'mean', 'sum', or 'none'
            label_smoothing: Label smoothing factor (0.0-0.1)
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

        logger.info(
            f"FocalLoss initialized: gamma={gamma}, "
            f"label_smoothing={label_smoothing}, "
            f"alpha={'provided' if alpha is not None else 'none'}"
        )

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            inputs: Model outputs (logits) of shape [batch_size, num_classes]
            targets: Ground truth labels of shape [batch_size]

        Returns:
            Focal loss value
        """
        # Apply label smoothing if specified
        if self.label_smoothing > 0:
            num_classes = inputs.size(1)
            targets_one_hot = F.one_hot(targets, num_classes).float()
            # Smooth labels: (1 - ε) * y_true + ε / num_classes
            targets_one_hot = (
                1 - self.label_smoothing
            ) * targets_one_hot + self.label_smoothing / num_classes
            ce_loss = F.cross_entropy(inputs, targets_one_hot, reduction="none")
        else:
            # Standard cross-entropy
            ce_loss = F.cross_entropy(inputs, targets, reduction="none")

        # Get probabilities
        pt = torch.exp(-ce_loss)  # p_t = exp(-CE)

        # Apply focal term: (1 - p_t)^γ
        focal_weight = (1 - pt) ** self.gamma

        # Apply class weights (alpha)
        if self.alpha is not None:
            if isinstance(self.alpha, (list, np.ndarray)):
                alpha_t = torch.tensor(self.alpha, device=inputs.device)[targets]
            elif isinstance(self.alpha, torch.Tensor):
                alpha_t = self.alpha.to(inputs.device)[targets]
            else:
                alpha_t = self.alpha
            focal_loss = alpha_t * focal_weight * ce_loss
        else:
            focal_loss = focal_weight * ce_loss

        # Apply reduction
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class WeightedCrossEntropyLoss(nn.Module):
    """
    Cross-Entropy Loss with class weights and label smoothing.

    Simpler alternative to Focal Loss when class imbalance is moderate.
    """

    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ):
        """
        Initialize weighted CE loss.

        Args:
            weight: Class weights (Tensor of shape [num_classes])
            label_smoothing: Label smoothing factor
            reduction: 'mean', 'sum', or 'none'
        """
        super(WeightedCrossEntropyLoss, self).__init__()
        self.weight = weight
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute weighted cross-entropy loss."""
        return F.cross_entropy(
            inputs,
            targets,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
            reduction=self.reduction,
        )


def calculate_class_weights(
    y: Union[np.ndarray, list],
    method: str = "inverse_frequency",
    normalize: bool = True,
) -> torch.Tensor:
    """
    Calculate class weights to address imbalance.

    Methods:
    - 'inverse_frequency': weight = total_samples / (num_classes * class_count)
    - 'effective_samples': weight = (1 - β) / (1 - β^n_i) where β = 0.999
    - 'balanced': weight = total_samples / (num_classes * class_count)

    Args:
        y: Labels (n_samples,)
        method: Weighting method
        normalize: Normalize weights to sum to num_classes

    Returns:
        Class weights as torch.Tensor
    """
    class_counts = Counter(y)
    num_classes = len(class_counts)
    total_samples = len(y)

    if method == "inverse_frequency":
        # Standard sklearn method
        weights = {
            class_id: total_samples / (num_classes * count)
            for class_id, count in class_counts.items()
        }

    elif method == "effective_samples":
        # Class-Balanced Loss (Cui et al., 2019)
        # More aggressive for highly imbalanced datasets
        beta = 0.999
        weights = {
            class_id: (1 - beta) / (1 - beta**count)
            for class_id, count in class_counts.items()
        }

    elif method == "balanced":
        # Equal weight to all classes
        weights = {class_id: 1.0 for class_id in class_counts.keys()}

    else:
        raise ValueError(f"Unknown method: {method}")

    # Convert to sorted array (by class ID)
    weight_array = np.array([weights[i] for i in sorted(weights.keys())])

    # Normalize if requested
    if normalize:
        weight_array = weight_array * num_classes / weight_array.sum()

    logger.info(f"Calculated class weights ({method}):")
    for class_id in sorted(weights.keys()):
        logger.info(f"  Class {class_id}: {weights[class_id]:.4f}")

    return torch.tensor(weight_array, dtype=torch.float32)


def apply_temperature_scaling(
    logits: torch.Tensor, temperature: float = 1.0
) -> torch.Tensor:
    """
    Apply temperature scaling to calibrate probabilities.

    Temperature scaling: p_i = exp(z_i / T) / Σ exp(z_j / T)

    Args:
        logits: Model outputs (before softmax)
        temperature: Scaling factor (>1 = more uncertain, <1 = more confident)

    Returns:
        Calibrated probabilities
    """
    scaled_logits = logits / temperature
    return F.softmax(scaled_logits, dim=1)


class TemperatureScaling(nn.Module):
    """
    Learn optimal temperature for probability calibration.

    Usage:
    ```python
    # After training, calibrate on validation set
    temp_scaler = TemperatureScaling()
    temp_scaler.fit(val_logits, val_labels)

    # Use for inference
    calibrated_probs = temp_scaler(test_logits)
    ```
    """

    def __init__(self):
        super(TemperatureScaling, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply learned temperature scaling."""
        return apply_temperature_scaling(logits, self.temperature.item())

    def fit(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 50,
    ):
        """
        Learn optimal temperature by minimizing NLL on validation set.

        Args:
            logits: Validation logits
            labels: Validation labels
            lr: Learning rate
            max_iter: Maximum optimization iterations
        """
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def eval():
            optimizer.zero_grad()
            loss = F.cross_entropy(self.forward(logits), labels)
            loss.backward()
            return loss

        optimizer.step(eval)
        logger.info(f"Optimal temperature: {self.temperature.item():.4f}")


def get_recommended_loss(
    y_train: np.ndarray, imbalance_ratio: Optional[float] = None
) -> nn.Module:
    """
    Recommend loss function based on dataset characteristics.

    Args:
        y_train: Training labels
        imbalance_ratio: Majority/minority ratio (computed if not provided)

    Returns:
        Configured loss function
    """
    if imbalance_ratio is None:
        class_counts = Counter(y_train)
        majority_count = max(class_counts.values())
        minority_count = min(class_counts.values())
        imbalance_ratio = majority_count / minority_count

    logger.info(f"Dataset imbalance ratio: {imbalance_ratio:.2f}")

    if imbalance_ratio > 100:
        # Severe imbalance: Use focal loss with effective samples
        logger.info("Recommendation: Focal Loss with effective sample weights")
        class_weights = calculate_class_weights(y_train, method="effective_samples")
        return FocalLoss(alpha=class_weights, gamma=2.5, label_smoothing=0.1)

    elif imbalance_ratio > 20:
        # High imbalance: Use focal loss with inverse frequency
        logger.info("Recommendation: Focal Loss with inverse frequency weights")
        class_weights = calculate_class_weights(y_train, method="inverse_frequency")
        return FocalLoss(alpha=class_weights, gamma=2.0, label_smoothing=0.05)

    elif imbalance_ratio > 5:
        # Moderate imbalance: Use weighted CE
        logger.info("Recommendation: Weighted Cross-Entropy")
        class_weights = calculate_class_weights(y_train, method="inverse_frequency")
        return WeightedCrossEntropyLoss(weight=class_weights, label_smoothing=0.05)

    else:
        # Balanced dataset: Standard CE
        logger.info("Recommendation: Standard Cross-Entropy")
        return nn.CrossEntropyLoss(label_smoothing=0.05)


__all__ = [
    "FocalLoss",
    "WeightedCrossEntropyLoss",
    "calculate_class_weights",
    "apply_temperature_scaling",
    "TemperatureScaling",
    "get_recommended_loss",
]

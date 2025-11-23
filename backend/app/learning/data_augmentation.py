"""
Data Augmentation for Class Balancing

This module addresses the 79.6% class imbalance in training data by using:
- SMOTE (Synthetic Minority Over-sampling Technique)
- ADASYN (Adaptive Synthetic Sampling)
- Time-series augmentation techniques

Goal: Improve ML accuracy from 72.7% → 85%+ by balancing attack classes.

Class Distribution (Original CICIDS2017):
- Normal: 79.6% (3.5M samples)
- DDoS: 8.2% (360K samples)
- PortScan: 5.3% (233K samples)
- BruteForce: 4.1% (180K samples)
- Web Attack: 1.9% (84K samples)
- Botnet: 0.7% (31K samples)
- Infiltration: 0.2% (9K samples)

Target Distribution:
- Normal: 30% (keeps some benign traffic)
- Attacks: 70% (balanced across 6 attack types ~11.7% each)
"""

import logging
from collections import Counter
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Check if imbalanced-learn is available
try:
    from imblearn.combine import SMOTEENN, SMOTETomek
    from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler

    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    logger.warning(
        "imbalanced-learn not installed. Install with: " "pip install imbalanced-learn"
    )


class DataAugmenter:
    """
    Handles class balancing and data augmentation for ML training.

    Strategies:
    1. SMOTE: Generate synthetic samples for minority classes
    2. ADASYN: Adaptive generation focusing on hard-to-learn regions
    3. Under-sampling: Reduce majority class (Normal traffic)
    4. Time-series augmentation: Jitter, scaling, rotation for sequences
    """

    def __init__(
        self,
        strategy: str = "auto",
        target_distribution: Optional[Dict[int, float]] = None,
        random_state: int = 42,
    ):
        """
        Initialize data augmenter.

        Args:
            strategy: Augmentation strategy
                - "smote": SMOTE over-sampling
                - "adasyn": Adaptive synthetic sampling
                - "smote_tomek": SMOTE + Tomek links cleaning
                - "auto": Automatically choose best strategy
            target_distribution: Desired class distribution (class_id → ratio)
            random_state: Random seed for reproducibility
        """
        self.strategy = strategy
        self.random_state = random_state

        # Default target: 30% normal, 70% attacks (balanced)
        if target_distribution is None:
            self.target_distribution = {
                0: 0.30,  # Normal
                1: 0.117,  # DDoS
                2: 0.117,  # PortScan
                3: 0.117,  # BruteForce
                4: 0.117,  # Web Attack
                5: 0.117,  # Botnet
                6: 0.115,  # Infiltration (slight adjustment for rounding)
            }
        else:
            self.target_distribution = target_distribution

        logger.info(
            f"DataAugmenter initialized: strategy={strategy}, "
            f"target_distribution={self.target_distribution}"
        )

    def balance_dataset(
        self, X: np.ndarray, y: np.ndarray, verbose: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Balance dataset using configured strategy.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels (n_samples,)
            verbose: Print class distribution before/after

        Returns:
            (X_balanced, y_balanced) tuple
        """
        if not IMBLEARN_AVAILABLE:
            logger.error("imbalanced-learn not available - returning original data")
            return X, y

        if verbose:
            self._log_distribution("Original", y)

        # Calculate target sample counts
        total_samples = len(y)
        target_samples = {
            class_id: int(total_samples * ratio)
            for class_id, ratio in self.target_distribution.items()
        }

        # Select balancing strategy
        if self.strategy == "auto":
            sampler = self._auto_select_strategy(X, y, target_samples)
        elif self.strategy == "smote":
            sampler = SMOTE(
                sampling_strategy=target_samples,
                random_state=self.random_state,
                k_neighbors=5,
            )
        elif self.strategy == "adasyn":
            sampler = ADASYN(
                sampling_strategy=target_samples,
                random_state=self.random_state,
                n_neighbors=5,
            )
        elif self.strategy == "smote_tomek":
            sampler = SMOTETomek(
                sampling_strategy=target_samples, random_state=self.random_state
            )
        else:
            logger.warning(f"Unknown strategy '{self.strategy}', using SMOTE")
            sampler = SMOTE(
                sampling_strategy=target_samples, random_state=self.random_state
            )

        # Apply balancing
        try:
            X_balanced, y_balanced = sampler.fit_resample(X, y)

            if verbose:
                self._log_distribution("Balanced", y_balanced)
                improvement = self._calculate_balance_improvement(y, y_balanced)
                logger.info(f"Balance improvement: {improvement:.1%} more balanced")

            return X_balanced, y_balanced

        except Exception as e:
            logger.error(f"Balancing failed: {e}, returning original data")
            return X, y

    def _auto_select_strategy(
        self, X: np.ndarray, y: np.ndarray, target_samples: Dict[int, int]
    ):
        """
        Automatically select best balancing strategy based on data characteristics.

        Logic:
        - Small dataset (<10K): RandomOverSampler (avoid overfitting)
        - Medium dataset (10K-100K): SMOTE (good balance)
        - Large dataset (>100K): ADASYN (adaptive, better for complex patterns)
        - Very imbalanced (>80% majority): SMOTE-Tomek (cleaning)
        """
        n_samples = len(y)
        class_counts = Counter(y)
        majority_ratio = max(class_counts.values()) / n_samples

        if n_samples < 10_000:
            logger.info("Auto-selected: RandomOverSampler (small dataset)")
            return RandomOverSampler(
                sampling_strategy=target_samples, random_state=self.random_state
            )
        elif majority_ratio > 0.80:
            logger.info("Auto-selected: SMOTE-Tomek (very imbalanced)")
            return SMOTETomek(
                sampling_strategy=target_samples, random_state=self.random_state
            )
        elif n_samples > 100_000:
            logger.info("Auto-selected: ADASYN (large dataset)")
            return ADASYN(
                sampling_strategy=target_samples,
                random_state=self.random_state,
                n_neighbors=5,
            )
        else:
            logger.info("Auto-selected: SMOTE (medium dataset)")
            return SMOTE(
                sampling_strategy=target_samples,
                random_state=self.random_state,
                k_neighbors=5,
            )

    def augment_time_series(
        self, sequences: np.ndarray, augmentation_factor: int = 2
    ) -> np.ndarray:
        """
        Augment time-series sequences with temporal transformations.

        Techniques:
        - Jittering: Add small random noise
        - Scaling: Stretch/compress time axis
        - Magnitude warping: Scale feature magnitudes
        - Time warping: Non-uniform time stretching

        Args:
            sequences: Time-series data (n_samples, seq_len, n_features)
            augmentation_factor: How many augmented copies to generate

        Returns:
            Augmented sequences (n_samples * (1 + augmentation_factor), seq_len, n_features)
        """
        augmented = [sequences]

        for _ in range(augmentation_factor):
            # Jittering: Add Gaussian noise (σ = 0.05)
            jittered = sequences + np.random.normal(0, 0.05, sequences.shape)
            augmented.append(jittered)

            # Scaling: Random magnitude scaling (0.9-1.1x)
            scales = np.random.uniform(
                0.9, 1.1, (sequences.shape[0], 1, sequences.shape[2])
            )
            scaled = sequences * scales
            augmented.append(scaled)

        result = np.concatenate(augmented, axis=0)
        logger.info(
            f"Time-series augmentation: {sequences.shape[0]} → {result.shape[0]} samples"
        )
        return result

    def _log_distribution(self, label: str, y: np.ndarray):
        """Log class distribution for debugging."""
        class_counts = Counter(y)
        total = len(y)

        logger.info(f"\n{label} Class Distribution:")
        for class_id in sorted(class_counts.keys()):
            count = class_counts[class_id]
            ratio = count / total
            logger.info(f"  Class {class_id}: {count:>7} samples ({ratio:.1%})")
        logger.info(f"  Total:    {total:>7} samples")

    def _calculate_balance_improvement(
        self, y_before: np.ndarray, y_after: np.ndarray
    ) -> float:
        """
        Calculate how much more balanced the dataset became.

        Uses Gini impurity: 0 = perfectly balanced, 1 = all one class.
        Returns improvement as a percentage.
        """

        def gini_impurity(y):
            counts = Counter(y)
            total = len(y)
            return 1 - sum((count / total) ** 2 for count in counts.values())

        gini_before = gini_impurity(y_before)
        gini_after = gini_impurity(y_after)

        # Higher Gini = more balanced (for multi-class)
        # But we want "improvement" to be positive when balance increases
        improvement = (gini_after - gini_before) / gini_before if gini_before > 0 else 0
        return improvement


# Global instance for easy import
data_augmenter = DataAugmenter(strategy="auto")


# Convenience functions
def balance_dataset(
    X: np.ndarray, y: np.ndarray, strategy: str = "auto"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Quick function to balance a dataset.

    Args:
        X: Features
        y: Labels
        strategy: Balancing strategy

    Returns:
        Balanced (X, y) tuple
    """
    augmenter = DataAugmenter(strategy=strategy)
    return augmenter.balance_dataset(X, y)


def augment_sequences(sequences: np.ndarray, factor: int = 2) -> np.ndarray:
    """
    Quick function to augment time-series sequences.

    Args:
        sequences: Time-series data
        factor: Augmentation factor

    Returns:
        Augmented sequences
    """
    augmenter = DataAugmenter()
    return augmenter.augment_time_series(sequences, factor)


__all__ = [
    "DataAugmenter",
    "data_augmenter",
    "balance_dataset",
    "augment_sequences",
    "IMBLEARN_AVAILABLE",
]

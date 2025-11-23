"""
Learning and Feedback Systems - Phase 2 Enhanced

This package implements continuous learning mechanisms:
- Vector memory for learned false positives (Phase 1)
- Data augmentation for class balancing (Phase 2)
- Automated retraining pipelines (Phase 2)
- Concept drift detection
- Feedback loops

Phase 2 Improvements:
- SMOTE/ADASYN for 79.6% → 40% class imbalance
- Automated retraining using Council corrections
- 72.7% → 85%+ ML accuracy target
"""

from .data_augmentation import (
    IMBLEARN_AVAILABLE,
    DataAugmenter,
    augment_sequences,
    balance_dataset,
    data_augmenter,
)
from .model_retrainer import ModelRetrainer, model_retrainer
from .retrain_scheduler import (
    RetrainScheduler,
    get_scheduler_status,
    retrain_scheduler,
    start_retrain_scheduler,
    stop_retrain_scheduler,
    trigger_manual_retrain,
)
from .threshold_optimizer import (
    SKOPT_AVAILABLE,
    ThresholdOptimizer,
    optimize_thresholds_simple,
)
from .training_collector import (
    TrainingCollector,
    collect_council_correction,
    training_collector,
)
from .vector_memory import (
    VectorMemory,
    search_similar_incidents,
    store_council_correction,
)
from .weighted_loss import (
    FocalLoss,
    TemperatureScaling,
    WeightedCrossEntropyLoss,
    apply_temperature_scaling,
    calculate_class_weights,
    get_recommended_loss,
)

__all__ = [
    "VectorMemory",
    "search_similar_incidents",
    "store_council_correction",
    "DataAugmenter",
    "data_augmenter",
    "balance_dataset",
    "augment_sequences",
    "IMBLEARN_AVAILABLE",
    "FocalLoss",
    "WeightedCrossEntropyLoss",
    "calculate_class_weights",
    "apply_temperature_scaling",
    "TemperatureScaling",
    "get_recommended_loss",
    "ThresholdOptimizer",
    "optimize_thresholds_simple",
    "SKOPT_AVAILABLE",
    "TrainingCollector",
    "training_collector",
    "collect_council_correction",
    "RetrainScheduler",
    "retrain_scheduler",
    "start_retrain_scheduler",
    "stop_retrain_scheduler",
    "get_scheduler_status",
    "trigger_manual_retrain",
    "ModelRetrainer",
    "model_retrainer",
]

"""
Feature Engineering Package - Phase 2

This package implements advanced feature engineering and caching:
- Feature store: Redis-backed pre-computation and caching
- Feature pipeline: Parallel extraction for performance
- Feature versions: Schema versioning for model compatibility

Goal: 30% faster inference (100ms → 70ms)
"""

from .advanced_features import AdvancedFeatureExtractor, advanced_feature_extractor
from .feature_pipeline import (
    FeaturePipeline,
    extract_ip_features,
    extract_ip_features_batch,
    feature_pipeline,
)
from .feature_store import (
    FeatureStore,
    feature_store,
    get_ip_features,
    store_ip_features,
)
from .integration_adapter import (
    enhanced_extract_features,
    enhanced_extract_features_batch,
    migrate_to_feature_store,
)

__all__ = [
    "FeatureStore",
    "feature_store",
    "store_ip_features",
    "get_ip_features",
    "FeaturePipeline",
    "feature_pipeline",
    "extract_ip_features",
    "extract_ip_features_batch",
    "enhanced_extract_features",
    "enhanced_extract_features_batch",
    "migrate_to_feature_store",
    "AdvancedFeatureExtractor",
    "advanced_feature_extractor",
]

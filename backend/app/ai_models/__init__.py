"""
Revolutionary ML Models for Mini-XDR

This package contains state-of-the-art deep learning models for
cybersecurity threat detection:

- FT-Transformer: Feature Tokenizer Transformer for tabular data
- EnsembleDetector: Uncertainty-weighted ensemble combining multiple models
- EvidentialHead: Dirichlet-based uncertainty quantification
"""

from .ensemble import (
    EnsembleConfig,
    EnsembleDetector,
    TemporalLSTM,
    XGBoostWrapper,
    get_ensemble_detector,
)
from .ft_transformer import (
    EvidentialHead,
    FTTransformer,
    FTTransformerConfig,
    FTTransformerDetector,
    get_ft_transformer_detector,
)

# Note: Database models are imported directly from the main models.py file
# This avoids circular import issues

__all__ = [
    # FT-Transformer
    "FTTransformer",
    "FTTransformerConfig",
    "FTTransformerDetector",
    "EvidentialHead",
    "get_ft_transformer_detector",
    # Ensemble
    "EnsembleDetector",
    "EnsembleConfig",
    "TemporalLSTM",
    "XGBoostWrapper",
    "get_ensemble_detector",
]

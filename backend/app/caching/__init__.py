"""
Caching Module - Redis-based Feature Caching

This module provides caching infrastructure for ML feature extraction.

Key Components:
- FeatureCache: Redis-backed cache for 79-dimensional feature vectors
- Cache key format: features:{src_ip}:{event_hash}
- Default TTL: 300 seconds (5 minutes)

Performance Impact:
- Without cache: ~50ms feature extraction
- With cache: ~5ms cache lookup
- 10x speed improvement on cache hits

Usage:
    from app.caching import get_feature_cache

    cache = get_feature_cache()
    features = await cache.get_cached_features(src_ip, events)
    if features is None:
        features = extract_features(src_ip, events)
        await cache.set_cached_features(src_ip, features, events)
"""

from .feature_cache import FeatureCache, get_feature_cache

__all__ = ["FeatureCache", "get_feature_cache"]

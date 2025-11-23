"""
Council of Models - Prometheus Metrics

Tracks Council performance, API usage, costs, and effectiveness.
"""

import logging

from prometheus_client import Counter, Gauge, Histogram

logger = logging.getLogger(__name__)

# ===== Routing Decisions =====
council_routing_decisions = Counter(
    "council_routing_decisions_total",
    "Total number of routing decisions made by the Council",
    ["route_type"],  # autonomous_response, gemini_judge, vector_memory, full_forensics
)

# ===== API Calls =====
council_api_calls = Counter(
    "council_api_calls_total",
    "Total number of API calls to Council agents",
    ["agent"],  # gemini, grok, openai
)

council_api_costs = Counter(
    "council_api_costs_dollars_total", "Estimated API costs in dollars", ["agent"]
)

# ===== Processing Time =====
council_processing_time = Histogram(
    "council_processing_time_seconds",
    "Time spent processing incidents through Council",
    ["route"],  # Which route was taken
)

# ===== Confidence Scores =====
council_confidence_scores = Histogram(
    "council_confidence_scores",
    "Distribution of final confidence scores",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

ml_confidence_scores = Histogram(
    "ml_confidence_scores",
    "Distribution of ML model confidence scores",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

# ===== Council Overrides =====
council_overrides = Counter(
    "council_overrides_total",
    "Number of times Council overrode ML predictions",
    ["ml_prediction", "council_verdict"],  # Track what was overridden
)

# ===== Vector Memory =====
vector_cache_hits = Counter(
    "vector_cache_hits_total",
    "Number of times vector memory found similar past incident",
)

vector_cache_misses = Counter(
    "vector_cache_misses_total", "Number of times vector memory had no similar incident"
)

vector_cache_hit_rate = Gauge(
    "vector_cache_hit_rate", "Current cache hit rate (rolling average)"
)

# ===== Verdicts =====
council_verdicts = Counter(
    "council_verdicts_total",
    "Final verdicts from Council",
    ["verdict"],  # THREAT, FALSE_POSITIVE, UNCERTAIN, INVESTIGATE
)

# ===== Active Incidents =====
council_active_incidents = Gauge(
    "council_active_incidents",
    "Number of incidents currently being processed by Council",
)


# ===== Helper Functions =====


def record_routing_decision(route_type: str):
    """Record a routing decision"""
    council_routing_decisions.labels(route_type=route_type).inc()
    logger.debug(f"Recorded routing decision: {route_type}")


def record_api_call(agent: str, cost_estimate: float = 0.0):
    """Record an API call and its estimated cost"""
    council_api_calls.labels(agent=agent).inc()

    if cost_estimate > 0:
        council_api_costs.labels(agent=agent).inc(cost_estimate)

    logger.debug(f"Recorded API call: {agent}, cost: ${cost_estimate:.4f}")


def record_processing_time(route: str, duration_seconds: float):
    """Record how long it took to process an incident"""
    council_processing_time.labels(route=route).observe(duration_seconds)
    logger.debug(f"Recorded processing time: {route}, {duration_seconds:.3f}s")


def record_confidence_scores(ml_confidence: float, council_confidence: float):
    """Record confidence scores from ML and Council"""
    ml_confidence_scores.observe(ml_confidence)
    council_confidence_scores.observe(council_confidence)


def record_override(ml_prediction: str, council_verdict: str):
    """Record when Council overrides ML prediction"""
    council_overrides.labels(
        ml_prediction=ml_prediction, council_verdict=council_verdict
    ).inc()
    logger.info(f"Council override: ML={ml_prediction} → Council={council_verdict}")


def record_cache_hit():
    """Record vector memory cache hit"""
    vector_cache_hits.inc()
    _update_cache_hit_rate()


def record_cache_miss():
    """Record vector memory cache miss"""
    vector_cache_misses.inc()
    _update_cache_hit_rate()


def _update_cache_hit_rate():
    """Calculate and update cache hit rate"""
    try:
        hits = vector_cache_hits._value.get()
        misses = vector_cache_misses._value.get()
        total = hits + misses

        if total > 0:
            hit_rate = hits / total
            vector_cache_hit_rate.set(hit_rate)
    except Exception as e:
        logger.warning(f"Failed to update cache hit rate: {e}")


def record_verdict(verdict: str):
    """Record final Council verdict"""
    council_verdicts.labels(verdict=verdict).inc()


def increment_active_incidents():
    """Increment active incident counter"""
    council_active_incidents.inc()


def decrement_active_incidents():
    """Decrement active incident counter"""
    council_active_incidents.dec()


# ===== Cost Estimation =====

# API cost estimates (as of 2025)
API_COSTS = {
    "gemini": 0.20,  # Per request (approximate for Gemini Pro)
    "grok": 0.10,  # Placeholder (API pricing TBD)
    "openai": 0.15,  # Per request (GPT-4o approximate)
}


def estimate_api_cost(agent: str) -> float:
    """Get estimated cost for an API call"""
    return API_COSTS.get(agent, 0.0)


# ===== Metrics Summary =====


def get_metrics_summary() -> dict:
    """
    Get a summary of Council metrics for dashboard display.

    Returns:
        dict: Summary statistics
    """
    try:
        # Get counter values
        routing_stats = {}
        for route_type in [
            "autonomous_response",
            "gemini_judge",
            "vector_memory",
            "full_forensics",
        ]:
            routing_stats[route_type] = council_routing_decisions.labels(
                route_type=route_type
            )._value.get()

        api_call_stats = {}
        api_cost_stats = {}
        for agent in ["gemini", "grok", "openai"]:
            api_call_stats[agent] = council_api_calls.labels(agent=agent)._value.get()
            api_cost_stats[agent] = council_api_costs.labels(agent=agent)._value.get()

        cache_hits = vector_cache_hits._value.get()
        cache_misses = vector_cache_misses._value.get()
        cache_total = cache_hits + cache_misses
        cache_hit_rate_val = (cache_hits / cache_total * 100) if cache_total > 0 else 0

        verdict_stats = {}
        for verdict in ["THREAT", "FALSE_POSITIVE", "UNCERTAIN", "INVESTIGATE"]:
            verdict_stats[verdict] = council_verdicts.labels(
                verdict=verdict
            )._value.get()

        return {
            "routing": routing_stats,
            "api_calls": api_call_stats,
            "api_costs": {
                "total_dollars": sum(api_cost_stats.values()),
                "by_agent": api_cost_stats,
            },
            "vector_cache": {
                "hits": cache_hits,
                "misses": cache_misses,
                "hit_rate_percent": round(cache_hit_rate_val, 2),
                "savings_estimate_dollars": cache_hits
                * API_COSTS["gemini"],  # Saved Gemini calls
            },
            "verdicts": verdict_stats,
            "active_incidents": int(council_active_incidents._value.get()),
            "performance": {
                "total_incidents_processed": sum(routing_stats.values()),
                "override_rate_percent": 0,  # TODO: Calculate from overrides counter
            },
        }

    except Exception as e:
        logger.error(f"Failed to get metrics summary: {e}")
        return {"error": str(e)}


# Export
__all__ = [
    "record_routing_decision",
    "record_api_call",
    "record_processing_time",
    "record_confidence_scores",
    "record_override",
    "record_cache_hit",
    "record_cache_miss",
    "record_verdict",
    "increment_active_incidents",
    "decrement_active_incidents",
    "estimate_api_cost",
    "get_metrics_summary",
]

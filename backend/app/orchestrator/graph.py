"""
Council of Models Orchestrator - LangGraph State Machine

This module implements the two-layer intelligence system:
- Layer 1 (Fast): Traditional ML models (<50ms)
- Layer 2 (Deep): GenAI Council (Gemini 3, Grok, OpenAI) for verification

The orchestrator routes incidents based on ML confidence scores.
"""

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, TypedDict


class XDRState(TypedDict):
    """
    State object that flows through the Council orchestration graph.

    This unified state contains data from both the Fast ML layer and
    the Deep GenAI Council layer, enabling seamless handoff between nodes.
    """

    # ===== Incident Identification =====
    incident_id: Optional[int]  # Database incident ID (if created)
    flow_id: str  # Unique identifier for this detection flow
    src_ip: str  # Source IP address being analyzed
    timestamp: str  # ISO format timestamp

    # ===== Fast ML Layer Data (79-dimensional pipeline) =====
    raw_features: List[float]  # The 79 extracted feature dimensions
    ml_prediction: Dict[str, Any]  # {
    #     "class": str,           # e.g., "BruteForce", "DDoS", "Normal"
    #     "confidence": float,    # 0.0-1.0
    #     "model": str,           # "general" or "specialist"
    #     "threat_score": float,  # 0.0-1.0
    #     "attack_type": str,     # Human-readable attack type
    #     "model_type": str       # e.g., "ensemble", "deep_learning"
    # }

    # ===== Raw Event Data =====
    events: List[Dict[str, Any]]  # Raw events that triggered detection
    event_count: int
    time_window: str  # e.g., "last_1h", "last_24h"

    # ===== Council Layer Data (Added by GenAI agents) =====
    gemini_analysis: Optional[str]  # Deep reasoning from Gemini 3
    gemini_verdict: Optional[str]  # "CONFIRM", "OVERRIDE", "UNCERTAIN"
    gemini_confidence: Optional[float]  # Gemini's confidence in its verdict
    gemini_reasoning: Optional[str]  # Explanation of Gemini's decision

    grok_intel: Optional[str]  # External threat intelligence from Grok (X.com)
    grok_threat_score: Optional[float]  # 0-100 threat score from Grok
    grok_references: Optional[List[str]]  # URLs to X posts/research

    openai_remediation: Optional[str]  # Generated remediation script
    openai_action_plan: Optional[List[str]]  # Step-by-step action items

    # ===== Private Model Layer =====
    sanitized_data: Optional[str]  # PII-scrubbed data for cloud analysis
    local_llm_analysis: Optional[str]  # Analysis from on-prem Llama-3

    # ===== Decision & Routing =====
    final_verdict: Literal["THREAT", "FALSE_POSITIVE", "UNCERTAIN", "INVESTIGATE"]
    confidence_score: float  # Final confidence (0.0-1.0)
    routing_path: List[str]  # Track which nodes processed this incident

    # ===== Action & Response =====
    action_plan: List[str]  # Actions to take (e.g., ["block_ip", "isolate_host"])
    actions_taken: List[Dict[str, Any]]  # Record of executed actions
    requires_human_review: bool

    # ===== Learning & Feedback =====
    council_override: bool  # Did Council override ML prediction?
    override_reason: Optional[str]  # Why Council overrode
    analyst_feedback: Optional[str]  # Human analyst confirmation/correction
    embedding_stored: bool  # Was this incident stored in vector DB?

    # ===== Metadata =====
    processing_time_ms: float  # Total processing time
    api_calls_made: List[str]  # Track API usage (for cost monitoring)
    error: Optional[str]  # Any errors encountered


# Default initial state factory
def create_initial_state(
    src_ip: str,
    events: List[Dict[str, Any]],
    ml_prediction: Dict[str, Any],
    raw_features: List[float],
) -> XDRState:
    """
    Create initial state for a new incident entering the orchestrator.

    Args:
        src_ip: Source IP address
        events: List of raw events
        ml_prediction: Prediction from the Fast ML layer
        raw_features: 79-dimensional feature vector

    Returns:
        XDRState with initial values
    """
    return XDRState(
        # Identification
        incident_id=None,
        flow_id=f"{src_ip}_{datetime.utcnow().isoformat()}",
        src_ip=src_ip,
        timestamp=datetime.utcnow().isoformat(),
        # Fast ML Layer
        raw_features=raw_features,
        ml_prediction=ml_prediction,
        events=events,
        event_count=len(events),
        time_window="last_1h",
        # Council Layer (initially empty)
        gemini_analysis=None,
        gemini_verdict=None,
        gemini_confidence=None,
        gemini_reasoning=None,
        grok_intel=None,
        grok_threat_score=None,
        grok_references=None,
        openai_remediation=None,
        openai_action_plan=None,
        # Private Model Layer
        sanitized_data=None,
        local_llm_analysis=None,
        # Decision
        final_verdict="UNCERTAIN",
        confidence_score=ml_prediction.get("confidence", 0.0),
        routing_path=[],
        # Action
        action_plan=[],
        actions_taken=[],
        requires_human_review=False,
        # Learning
        council_override=False,
        override_reason=None,
        analyst_feedback=None,
        embedding_stored=False,
        # Metadata
        processing_time_ms=0.0,
        api_calls_made=[],
        error=None,
    )

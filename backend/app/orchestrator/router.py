"""
Confidence-Based Router for Council of Models

This router implements the core intelligence that decides:
- When to trust the Fast ML models (high confidence)
- When to engage the Deep GenAI Council (uncertain)
- When to trigger full forensic investigation (low confidence)
"""

import logging
from typing import Literal

from .graph import XDRState

logger = logging.getLogger(__name__)

# Router decision types
RouterDecision = Literal[
    "autonomous_response",  # High confidence: trust ML, act immediately
    "gemini_judge",  # Medium confidence: ask Council for verification
    "full_forensics",  # Low confidence: deep investigation needed
    "vector_lookup",  # Check if we've seen this pattern before
]


def confidence_router(state: XDRState) -> RouterDecision:
    """
    Primary routing function that decides where to send the incident.

    Decision Logic:
    1. Very High Confidence (>0.90) + Specialist Model → Trust and act
    2. High Confidence (0.70-0.90) + General Model → Verify with Gemini
    3. Medium Confidence (0.50-0.70) → Check vector memory, then Gemini
    4. Low Confidence (<0.50) → Full forensics + Gemini

    Args:
        state: Current XDRState with ML prediction

    Returns:
        RouterDecision indicating next node
    """
    confidence = state["ml_prediction"].get("confidence", 0.0)
    model_type = state["ml_prediction"].get("model", "general")
    attack_type = state["ml_prediction"].get("class", "Unknown")

    logger.info(
        f"Routing decision for {state['src_ip']}: "
        f"confidence={confidence:.2f}, model={model_type}, attack={attack_type}"
    )

    # ===== HIGH CONFIDENCE PATH: Trust the ML model =====
    if confidence > 0.90:
        # Specialist models (93%+ accuracy) are highly trustworthy
        if model_type == "specialist":
            logger.info(
                f"HIGH CONFIDENCE + SPECIALIST: Trusting {model_type} model "
                f"({confidence:.2%}) for {attack_type}"
            )
            state["routing_path"].append("autonomous_response")
            return "autonomous_response"

        # General model at very high confidence can also be trusted
        elif confidence > 0.95:
            logger.info(
                f"VERY HIGH CONFIDENCE: Trusting general model "
                f"({confidence:.2%}) for {attack_type}"
            )
            state["routing_path"].append("autonomous_response")
            return "autonomous_response"

        # High confidence but general model → verify with Gemini
        else:
            logger.info(
                f"HIGH CONFIDENCE but general model: Verifying with Gemini "
                f"({confidence:.2%}) for {attack_type}"
            )
            state["routing_path"].append("gemini_judge")
            return "gemini_judge"

    # ===== MEDIUM-HIGH CONFIDENCE: Verify with Council =====
    elif 0.70 <= confidence <= 0.90:
        logger.info(
            f"MEDIUM-HIGH CONFIDENCE: Asking Gemini to verify "
            f"({confidence:.2%}) for {attack_type}"
        )
        state["routing_path"].append("gemini_judge")
        return "gemini_judge"

    # ===== MEDIUM CONFIDENCE: Check vector memory first =====
    elif 0.50 <= confidence < 0.70:
        # Check if we've seen a similar pattern that Gemini already analyzed
        logger.info(
            f"MEDIUM CONFIDENCE: Checking vector memory before Gemini "
            f"({confidence:.2%}) for {attack_type}"
        )
        state["routing_path"].append("vector_lookup")
        return "vector_lookup"

    # ===== LOW CONFIDENCE: Full investigation =====
    else:
        logger.info(
            f"LOW CONFIDENCE: Triggering full forensics "
            f"({confidence:.2%}) for {attack_type}"
        )
        state["routing_path"].append("full_forensics")
        return "full_forensics"


def route_decision(state: XDRState) -> str:
    """
    Wrapper for confidence_router that returns the next node name.

    This is used by LangGraph's conditional routing system.

    Args:
        state: Current XDRState

    Returns:
        String name of the next node to execute
    """
    decision = confidence_router(state)

    # Map router decisions to LangGraph node names
    node_mapping = {
        "autonomous_response": "response_agent",
        "gemini_judge": "gemini_judge",
        "full_forensics": "forensics_agent",
        "vector_lookup": "vector_memory",
    }

    next_node = node_mapping.get(decision, "gemini_judge")

    logger.info(f"Routing {state['src_ip']} to node: {next_node}")

    return next_node


def should_engage_grok(state: XDRState) -> bool:
    """
    Decide if we should query Grok for external threat intelligence.

    Grok is engaged when:
    - Unknown file hashes (not in local threat DB)
    - Recently registered domains (< 30 days old)
    - Unknown destination IPs
    - IOCs that need real-time validation
    - Any THREAT verdict that could benefit from intel

    Args:
        state: Current XDRState

    Returns:
        True if Grok should be queried
    """
    # Check if we have IOCs that need external validation
    events = state.get("events", [])

    # Look for unknown hashes, new domains, external IPs, etc.
    has_unknown_iocs = any(
        event.get("file_hash") or event.get("domain") for event in events
    )

    # Also engage for external IPs or when we have a confirmed threat
    has_external_ip = state.get("src_ip", "").split(".")[0] not in [
        "10",
        "172",
        "192",
        "127",
    ]
    is_threat = (
        state.get("final_verdict") == "THREAT"
        or state.get("gemini_verdict") == "CONFIRM"
    )

    # Only engage Grok if ML confidence is not very high OR we have a threat
    confidence = state["ml_prediction"].get("confidence", 0.0)

    # Engage Grok more liberally - for any threat or IOC presence
    should_engage = (
        (has_unknown_iocs and confidence < 0.85)
        or (is_threat and has_external_ip)
        or is_threat
    )

    if should_engage:
        logger.info(
            f"Engaging Grok for external intel: "
            f"IOCs={has_unknown_iocs}, external_ip={has_external_ip}, "
            f"is_threat={is_threat}, confidence={confidence:.2%}"
        )

    return should_engage


def should_engage_openai(state: XDRState) -> bool:
    """
    Decide if we should engage OpenAI for remediation script generation.

    OpenAI is engaged when:
    - Final verdict is THREAT
    - Any detection that requires response actions
    - Complex remediation needed (firewall rules, scripts)

    Args:
        state: Current XDRState

    Returns:
        True if OpenAI should generate remediation
    """
    verdict = state.get("final_verdict")
    gemini_verdict = state.get("gemini_verdict")

    # Engage for threats or when investigation is needed
    requires_action = (
        verdict == "THREAT" or verdict == "INVESTIGATE" or gemini_verdict == "CONFIRM"
    )

    # Only generate remediation if we have any confidence
    confidence = state.get("confidence_score", 0.0)
    ml_confidence = state["ml_prediction"].get("confidence", 0.0)

    # Use the higher of the two confidence values
    effective_confidence = max(confidence, ml_confidence)

    # Lower threshold - engage OpenAI for any actionable detection
    should_engage = requires_action and effective_confidence > 0.40

    if should_engage:
        logger.info(
            f"Engaging OpenAI for remediation: "
            f"verdict={verdict}, gemini={gemini_verdict}, confidence={effective_confidence:.2%}"
        )

    return should_engage


# Export for easy imports
__all__ = [
    "confidence_router",
    "route_decision",
    "should_engage_grok",
    "should_engage_openai",
    "RouterDecision",
]

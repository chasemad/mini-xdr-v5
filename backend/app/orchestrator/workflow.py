"""
Council Orchestrator Workflow - The Main State Machine

This module builds the LangGraph workflow that coordinates:
1. Fast ML detection
2. Confidence-based routing
3. Council agents (Gemini, Grok, OpenAI)
4. Vector memory lookups
5. Response execution

Flow:
    ML Detection
        ↓
    Router (confidence-based)
        ├─> High Confidence → Autonomous Response
        ├─> Medium → Vector Lookup → [Found: Reuse] or [Not Found: Gemini]
        └─> Low → Gemini + Full Forensics

"""

import asyncio
import logging
import time
from typing import Any, Dict, Literal

try:
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, StateGraph

    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    END = "END"  # Define END as string when LangGraph not available
    logging.warning("LangGraph not available - orchestrator will not function")

from app.council.gemini_judge import gemini_judge_node
from app.council.grok_intel import grok_intel_node
from app.council.openai_remediation import openai_remediation_node
from app.learning.vector_memory import (
    search_similar_incidents,
    store_council_correction,
)

from .graph import XDRState
from .metrics import (
    decrement_active_incidents,
    estimate_api_cost,
    increment_active_incidents,
    record_api_call,
    record_cache_hit,
    record_cache_miss,
    record_confidence_scores,
    record_override,
    record_processing_time,
    record_routing_decision,
    record_verdict,
)
from .router import route_decision, should_engage_grok, should_engage_openai

logger = logging.getLogger(__name__)


# ===== Node Definitions =====


async def vector_lookup_node(state: XDRState) -> Dict[str, Any]:
    """
    Check vector memory for similar past incidents.

    If we find a highly similar incident (score > 0.95), reuse Gemini's
    past reasoning instead of making a new API call.

    Returns:
        Updated state, possibly with cached Gemini analysis
    """
    logger.info(f"Checking vector memory for {state['src_ip']}")

    similar = await search_similar_incidents(state)

    if similar and similar[0]["score"] > 0.95:
        # Highly similar incident found - reuse Council's past decision
        past = similar[0]["payload"]

        logger.info(
            f"CACHE HIT: Reusing past Council decision (similarity: {similar[0]['score']:.3f})"
        )

        # Record cache hit
        record_cache_hit()

        state["gemini_verdict"] = past.get("gemini_verdict", "UNCERTAIN")
        state["gemini_confidence"] = past.get("gemini_confidence", 0.5)
        state[
            "gemini_reasoning"
        ] = f"[CACHED from similar incident] {past.get('gemini_reasoning', '')}"
        state["final_verdict"] = past.get("final_verdict", "INVESTIGATE")
        state["confidence_score"] = past.get("confidence_score", 0.5)

        # Mark that we saved an API call
        state["api_calls_made"].append("vector_cache_hit")

        return state

    else:
        # No similar incident - need to ask Gemini
        logger.info("CACHE MISS: No similar incident found, routing to Gemini")

        # Record cache miss
        record_cache_miss()

        state["routing_path"].append("gemini_judge")

        # Continue to Gemini node
        return await gemini_judge_node(state)


async def response_agent_node(state: XDRState) -> Dict[str, Any]:
    """
    Execute autonomous response for high-confidence detections.

    This node handles incidents where we trust the ML model (>90% confidence).
    """
    logger.info(
        f"Executing autonomous response for {state['src_ip']}: "
        f"{state['ml_prediction']['class']}"
    )

    # Set final verdict
    state["final_verdict"] = "THREAT"
    state["confidence_score"] = state["ml_prediction"]["confidence"]

    # Generate action plan (template-based for now)
    attack_type = state["ml_prediction"]["class"]
    src_ip = state["src_ip"]

    basic_actions = [
        f"Block IP {src_ip} at firewall",
        f"Create incident ticket for {attack_type}",
        "Alert SOC team",
        "Monitor for 24 hours",
    ]

    state["action_plan"].extend(basic_actions)

    logger.info(f"Autonomous response planned: {len(state['action_plan'])} actions")

    return state


async def forensics_agent_node(state: XDRState) -> Dict[str, Any]:
    """
    Trigger full forensic investigation for low-confidence detections.

    This node is engaged when ML confidence is below 50% (very uncertain).
    """
    logger.info(f"Initiating full forensics for {state['src_ip']}")

    # Mark for human review
    state["requires_human_review"] = True

    # Add forensic collection tasks
    forensic_actions = [
        "Capture full packet trace (tcpdump)",
        "Collect system logs from source host",
        "Memory dump if malware suspected",
        "Review authentication logs",
        "Check threat intelligence feeds",
    ]

    state["action_plan"].extend(forensic_actions)

    # Also ask Gemini for deep analysis
    state["routing_path"].append("gemini_judge")

    return await gemini_judge_node(state)


async def decision_finalizer_node(state: XDRState) -> Dict[str, Any]:
    """
    Finalize the decision and prepare for action execution.

    This node runs after Council analysis to determine next steps.
    """
    logger.info(
        f"Finalizing decision for {state['src_ip']}: "
        f"verdict={state['final_verdict']}, confidence={state['confidence_score']:.2%}"
    )

    # Record verdict
    record_verdict(state["final_verdict"])

    # Record if Council overrode ML
    if state.get("council_override"):
        ml_class = state["ml_prediction"]["class"]
        council_verdict = state.get("gemini_verdict", "UNKNOWN")
        record_override(ml_class, council_verdict)

    # Record confidence scores
    ml_conf = state["ml_prediction"].get("confidence", 0.0)
    council_conf = state.get("confidence_score", 0.0)
    record_confidence_scores(ml_conf, council_conf)

    # Store Council correction in vector DB
    if state.get("gemini_verdict") and not state.get("embedding_stored"):
        await store_council_correction(state)

    # Check if we should engage Grok for threat intel
    if should_engage_grok(state):
        state["routing_path"].append("grok_intel")
        await grok_intel_node(state)

    # Check if we should engage OpenAI for remediation
    if should_engage_openai(state):
        state["routing_path"].append("openai_remediation")
        await openai_remediation_node(state)

    return state


def route_after_gemini(state: XDRState) -> Literal["decision_finalizer", END]:
    """Route after Gemini analysis."""
    return "decision_finalizer"


def route_after_vector(
    state: XDRState,
) -> Literal["gemini_judge", "decision_finalizer"]:
    """Route after vector lookup."""
    # If we got a cache hit, go to finalizer
    if "vector_cache_hit" in state.get("api_calls_made", []):
        return "decision_finalizer"
    else:
        # Vector lookup already called Gemini, go to finalizer
        return "decision_finalizer"


# ===== Workflow Builder =====


def build_council_workflow() -> StateGraph:
    """
    Build the complete Council of Models workflow.

    Returns:
        Compiled LangGraph workflow
    """
    if not LANGGRAPH_AVAILABLE:
        logger.error("LangGraph not available - cannot build workflow")
        return None

    # Create workflow
    workflow = StateGraph(XDRState)

    # Add nodes
    workflow.add_node("vector_lookup", vector_lookup_node)
    workflow.add_node("gemini_judge", gemini_judge_node)
    workflow.add_node("grok_intel", grok_intel_node)
    workflow.add_node("openai_remediation", openai_remediation_node)
    workflow.add_node("response_agent", response_agent_node)
    workflow.add_node("forensics_agent", forensics_agent_node)
    workflow.add_node("decision_finalizer", decision_finalizer_node)

    # Set entry point with conditional routing
    workflow.set_conditional_entry_point(
        route_decision,
        {
            "vector_memory": "vector_lookup",
            "gemini_judge": "gemini_judge",
            "response_agent": "response_agent",
            "forensics_agent": "forensics_agent",
        },
    )

    # Define edges
    workflow.add_conditional_edges(
        "vector_lookup",
        route_after_vector,
        {"gemini_judge": "gemini_judge", "decision_finalizer": "decision_finalizer"},
    )

    workflow.add_conditional_edges(
        "gemini_judge",
        route_after_gemini,
        {"decision_finalizer": "decision_finalizer", END: END},
    )

    workflow.add_edge("response_agent", "decision_finalizer")
    workflow.add_edge("forensics_agent", "decision_finalizer")

    workflow.add_edge("decision_finalizer", END)

    # Add memory checkpointing (optional, for debugging)
    memory = MemorySaver()

    # Compile workflow
    app = workflow.compile(checkpointer=memory)

    logger.info("Council workflow built successfully")

    return app


# ===== Main Orchestrator Function =====


async def orchestrate_incident(state: XDRState) -> XDRState:
    """
    Main entry point for Council orchestration.

    Args:
        state: Initial XDRState with ML prediction

    Returns:
        Final XDRState with Council decisions and action plan
    """
    logger.info(
        f"🎯 Orchestrating incident: {state['src_ip']} - "
        f"{state['ml_prediction']['class']} ({state['ml_prediction']['confidence']:.2%})"
    )

    # Track active incident
    increment_active_incidents()
    start_time = time.time()

    try:
        # Build workflow
        workflow = build_council_workflow()

        if workflow is None:
            logger.error("Failed to build workflow - using fallback logic")
            # Fallback: just use ML prediction
            state["final_verdict"] = (
                "THREAT"
                if state["ml_prediction"]["confidence"] > 0.7
                else "INVESTIGATE"
            )
            return state

        # Execute workflow
        config = {"configurable": {"thread_id": state["flow_id"]}}

        final_state = await workflow.ainvoke(state, config)

        # Record routing decision and processing time
        if final_state.get("routing_path"):
            primary_route = final_state["routing_path"][0]
            record_routing_decision(primary_route)

            duration = time.time() - start_time
            record_processing_time(primary_route, duration)

        logger.info(
            f"✅ Orchestration complete: {final_state['src_ip']} - "
            f"verdict={final_state['final_verdict']}, "
            f"confidence={final_state['confidence_score']:.2%}, "
            f"path={' → '.join(final_state['routing_path'])}"
        )

        return final_state

    except Exception as e:
        logger.error(f"Orchestration error: {e}", exc_info=True)
        state["error"] = f"Orchestration failed: {str(e)}"
        state["final_verdict"] = "INVESTIGATE"
        state["requires_human_review"] = True
        return state

    finally:
        # Always decrement counter
        decrement_active_incidents()


# Export
__all__ = [
    "build_council_workflow",
    "orchestrate_incident",
    "vector_lookup_node",
    "response_agent_node",
    "forensics_agent_node",
    "decision_finalizer_node",
]

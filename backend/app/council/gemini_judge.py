"""
Gemini Judge Node - The Council's Deep Reasoning Engine

Gemini 3 acts as a "Second Opinion Judge" that verifies uncertain ML predictions.
With 1M+ token context window, it can analyze extensive logs and provide
human-level reasoning about complex attack patterns.
"""

import json
import logging
import os
import time
from typing import Any, Dict

try:
    from langchain_google_genai import ChatGoogleGenerativeAI

    GEMINI_AVAILABLE = True
except ImportError:
    try:
        from langchain_google_vertexai import ChatVertexAI

        GEMINI_AVAILABLE = True
    except ImportError:
        GEMINI_AVAILABLE = False
        logging.warning("Gemini SDK not available - judge node will use fallback logic")

from app.orchestrator.graph import XDRState

logger = logging.getLogger(__name__)


# Initialize Gemini client (singleton)
_gemini_client = None


def get_gemini_client():
    """Get or create Gemini client instance."""
    global _gemini_client

    if _gemini_client is None and GEMINI_AVAILABLE:
        try:
            # Try Google AI Studio first (simpler, API key based)
            google_api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

            if google_api_key:
                # Use Google AI Studio (recommended)
                try:
                    _gemini_client = ChatGoogleGenerativeAI(
                        model="gemini-1.5-pro",
                        google_api_key=google_api_key,
                        temperature=0.1,  # Low temperature for consistent security decisions
                        max_output_tokens=8192,
                    )
                    logger.info("✅ Gemini Judge client initialized (Google AI Studio)")
                    return _gemini_client
                except NameError:
                    # ChatGoogleGenerativeAI not available, try Vertex AI
                    pass

            # Fall back to Vertex AI (requires GCP project setup)
            project_id = os.getenv("GCP_PROJECT_ID")
            location = os.getenv("GCP_LOCATION", "us-central1")

            if project_id:
                _gemini_client = ChatVertexAI(
                    model="gemini-1.5-pro",
                    project=project_id,
                    location=location,
                    temperature=0.1,
                    max_tokens=8192,
                )
                logger.info("✅ Gemini Judge client initialized (Vertex AI)")
            else:
                logger.warning(
                    "No Gemini credentials found. Set GOOGLE_API_KEY or GCP_PROJECT_ID"
                )
                return None

        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
            _gemini_client = None

    return _gemini_client


async def gemini_judge_node(state: XDRState) -> Dict[str, Any]:
    """
    Gemini Judge Node - Provides deep reasoning on uncertain ML predictions.

    This node is called when:
    - ML confidence is between 0.50 and 0.90 (uncertain range)
    - General model detected something (72.7% accuracy needs verification)
    - Complex attack patterns that need human-level reasoning

    Gemini's Role:
    1. Analyze the raw features and event timeline
    2. Look for legitimate automation (backup scripts, monitoring tools)
    3. Detect subtle patterns ML models miss
    4. Provide explainable reasoning for analysts

    Args:
        state: Current XDRState with ML prediction

    Returns:
        Updated state with Gemini's verdict and reasoning
    """
    start_time = time.time()

    logger.info(
        f"Gemini Judge evaluating {state['src_ip']}: "
        f"ML predicted {state['ml_prediction']['class']} "
        f"with {state['ml_prediction']['confidence']:.2%} confidence"
    )

    # Track API call
    state["api_calls_made"].append("gemini_judge")

    # Record API call with cost estimate
    from app.orchestrator.metrics import estimate_api_cost, record_api_call

    gemini_cost = estimate_api_cost("gemini")
    record_api_call("gemini", gemini_cost)

    # Get Gemini client
    client = get_gemini_client()

    if client is None:
        # Fallback: Use rule-based logic if Gemini unavailable
        return _fallback_judge(state)

    try:
        # Construct the prompt for Gemini
        prompt = _build_gemini_prompt(state)

        # Call Gemini for analysis
        response = await client.ainvoke(prompt)

        # Parse Gemini's response
        verdict_data = _parse_gemini_response(response.content)

        # Update state with Gemini's analysis (format for frontend)
        state["gemini_analysis"] = {
            "reasoning": verdict_data.get("reasoning", response.content),
            "confidence": verdict_data.get("confidence", 0.5),
            "verdict": verdict_data.get("verdict", "UNCERTAIN"),
            "suggested_actions": verdict_data.get("suggested_actions", []),
            "requires_human_review": verdict_data.get("requires_human_review", False),
            "raw_response": response.content,
        }
        state["gemini_verdict"] = verdict_data.get("verdict", "UNCERTAIN")
        state["gemini_confidence"] = verdict_data.get("confidence", 0.5)
        state["gemini_reasoning"] = verdict_data.get("reasoning", "")

        # Determine if Gemini overrode the ML prediction
        ml_class = state["ml_prediction"]["class"]
        gemini_verdict = verdict_data.get("verdict")

        if gemini_verdict == "OVERRIDE":
            state["council_override"] = True
            state["override_reason"] = verdict_data.get("reasoning")
            state["final_verdict"] = "FALSE_POSITIVE"
            logger.warning(
                f"Gemini OVERRODE ML prediction: "
                f"ML said '{ml_class}', Gemini says FALSE POSITIVE"
            )
        elif gemini_verdict == "CONFIRM":
            state["final_verdict"] = "THREAT"
            logger.info(
                f"Gemini CONFIRMED ML prediction: '{ml_class}' is a genuine threat"
            )
        else:  # UNCERTAIN
            state["final_verdict"] = "INVESTIGATE"
            state["requires_human_review"] = True
            logger.info(
                f"Gemini UNCERTAIN about ML prediction: '{ml_class}' needs human review"
            )

        # Update confidence score (weighted average of ML + Gemini)
        ml_confidence = state["ml_prediction"]["confidence"]
        gemini_confidence = verdict_data.get("confidence", 0.5)

        # If Gemini overrides, trust Gemini more (70/30 split)
        if state["council_override"]:
            state["confidence_score"] = gemini_confidence * 0.7 + ml_confidence * 0.3
        else:
            # If Gemini confirms, boost confidence (50/50 split)
            state["confidence_score"] = (gemini_confidence + ml_confidence) / 2

        elapsed_ms = (time.time() - start_time) * 1000
        state["processing_time_ms"] += elapsed_ms

        logger.info(
            f"Gemini Judge completed in {elapsed_ms:.0f}ms: "
            f"verdict={state['gemini_verdict']}, "
            f"confidence={state['confidence_score']:.2%}"
        )

    except Exception as e:
        logger.error(f"Gemini Judge error: {e}", exc_info=True)
        state["error"] = f"Gemini Judge failed: {str(e)}"
        # Fall back to ML prediction
        return _fallback_judge(state)

    return state


def _build_gemini_prompt(state: XDRState) -> str:
    """
    Construct a detailed prompt for Gemini to analyze the incident.

    The prompt provides:
    - ML model's prediction and confidence
    - Raw 79-dimensional features
    - Event timeline
    - Task instructions
    """
    ml_pred = state["ml_prediction"]
    features = state["raw_features"]
    events = state["events"]

    # Format events for readability
    events_summary = "\n".join(
        [
            f"  - {event.get('timestamp', 'N/A')}: {event.get('event_type', 'Unknown')} "
            f"from {event.get('src_ip', 'N/A')} to {event.get('dst_ip', 'N/A')}"
            for event in events[:20]  # Limit to first 20 events
        ]
    )

    if len(events) > 20:
        events_summary += f"\n  ... and {len(events) - 20} more events"

    prompt = f"""
You are a cybersecurity expert analyzing a potential security incident detected by ML models.

CONTEXT:
- Source IP: {state['src_ip']}
- Event Count: {state['event_count']}
- Time Window: {state['time_window']}

ML MODEL PREDICTION:
- Classification: {ml_pred['class']}
- Confidence: {ml_pred['confidence']:.2%}
- Model Type: {ml_pred['model']}
- Threat Score: {ml_pred.get('threat_score', 'N/A')}

CONCERN:
Our ML model has {ml_pred['confidence']:.2%} confidence, which is in the uncertain range.
We need your expert analysis to confirm or override this prediction.

RAW FEATURES (79-dimensional vector):
{_format_features(features)}

EVENT TIMELINE:
{events_summary}

YOUR TASK:
1. **Analyze Temporal Patterns**: Look at inter-arrival times, burst patterns, and regularity.
   - Are the events too regular (scripted behavior)?
   - Are they bursty (manual or attack behavior)?

2. **Check for Legitimate Automation**:
   - Could this be a backup script? (regular intervals, same commands)
   - Could this be monitoring? (periodic checks, same user agent)
   - Could this be DevOps automation? (scheduled tasks, known IPs)

3. **Evaluate Attack Indicators**:
   - Brute force: Failed login patterns, username/password diversity
   - DDoS: High volume, distributed sources, connection failures
   - Reconnaissance: Port scanning, service enumeration
   - Web attack: SQL injection, XSS, path traversal patterns

4. **Make a Decision**:
   - CONFIRM: The ML model is correct, this is a genuine {ml_pred['class']} attack
   - OVERRIDE: The ML model is wrong, this is legitimate/benign activity
   - UNCERTAIN: Need more data or human analyst review

IMPORTANT:
- Be conservative: False positives are costly (block legitimate users)
- Explain your reasoning clearly for the analyst
- If unsure, say UNCERTAIN and explain what additional data would help

OUTPUT FORMAT (JSON):
{{
  "verdict": "CONFIRM" | "OVERRIDE" | "UNCERTAIN",
  "confidence": 0.0-1.0,
  "reasoning": "Detailed explanation of your decision...",
  "suggested_actions": ["action1", "action2"],
  "requires_human_review": true/false
}}

Provide your analysis:
"""

    return prompt


def _format_features(features: list) -> str:
    """Format the 79-dimensional feature vector for readability."""
    if not features or len(features) < 79:
        return "Feature vector incomplete or missing"

    # Feature names (matching ml_feature_extractor.py)
    feature_names = [
        "hour",
        "day_of_week",
        "is_weekend",
        "is_business_hours",
        "time_since_first_event",
        "event_count_1min",
        "event_count_5min",
        "event_count_1h",
        "event_count_24h",
        "events_per_minute",
        "dst_port_normalized",
        "src_port_normalized",
        "unique_dst_ports",
        "unique_src_ports",
        "port_diversity",
        # ... (abbreviated for brevity)
    ]

    formatted = []
    for i, value in enumerate(features[:20]):  # Show first 20 features
        name = feature_names[i] if i < len(feature_names) else f"feature_{i}"
        formatted.append(f"  {name}: {value:.3f}")

    if len(features) > 20:
        formatted.append(f"  ... and {len(features) - 20} more features")

    return "\n".join(formatted)


def _parse_gemini_response(response_text: str) -> Dict[str, Any]:
    """
    Parse Gemini's response text into structured data.

    Attempts to extract JSON from the response, with fallback to text parsing.
    """
    try:
        # Try to find JSON in the response
        json_start = response_text.find("{")
        json_end = response_text.rfind("}") + 1

        if json_start != -1 and json_end > json_start:
            json_str = response_text[json_start:json_end]
            data = json.loads(json_str)
            return data
        else:
            # Fallback: Try to parse as plain JSON
            return json.loads(response_text)

    except json.JSONDecodeError:
        # Fallback: Extract verdict from text
        logger.warning("Failed to parse JSON from Gemini response, using text analysis")

        response_lower = response_text.lower()

        if "confirm" in response_lower:
            verdict = "CONFIRM"
            confidence = 0.8
        elif "override" in response_lower or "false positive" in response_lower:
            verdict = "OVERRIDE"
            confidence = 0.7
        else:
            verdict = "UNCERTAIN"
            confidence = 0.5

        return {
            "verdict": verdict,
            "confidence": confidence,
            "reasoning": response_text,
            "suggested_actions": [],
            "requires_human_review": verdict == "UNCERTAIN",
        }


def _fallback_judge(state: XDRState) -> Dict[str, Any]:
    """
    Fallback logic when Gemini is unavailable.

    Uses simple rule-based heuristics to make a decision.
    """
    logger.info("Using fallback judge logic (Gemini unavailable)")

    ml_confidence = state["ml_prediction"]["confidence"]
    ml_class = state["ml_prediction"]["class"]

    # Conservative fallback: Trust specialist models, verify general model
    if state["ml_prediction"].get("model") == "specialist" and ml_confidence > 0.85:
        state["final_verdict"] = "THREAT"
        state["confidence_score"] = ml_confidence
        state["gemini_verdict"] = "CONFIRM"
        reasoning = f"Fallback analysis: Specialist model detected {ml_class} with high confidence ({ml_confidence:.1%}). Threat confirmed."
        state["gemini_reasoning"] = reasoning
    elif ml_confidence < 0.60:
        state["final_verdict"] = "INVESTIGATE"
        state["requires_human_review"] = True
        state["gemini_verdict"] = "UNCERTAIN"
        reasoning = f"Fallback analysis: Low confidence detection ({ml_confidence:.1%}) for {ml_class}. Requires human review for verification."
        state["gemini_reasoning"] = reasoning
    else:
        state["final_verdict"] = "THREAT"
        state["confidence_score"] = ml_confidence * 0.8  # Reduce confidence slightly
        state["gemini_verdict"] = "CONFIRM"
        reasoning = f"Fallback analysis: ML model detected {ml_class} with moderate confidence ({ml_confidence:.1%}). Accepting prediction with caution."
        state["gemini_reasoning"] = reasoning

    # Format for frontend
    state["gemini_analysis"] = {
        "reasoning": state["gemini_reasoning"],
        "confidence": state.get("confidence_score", ml_confidence),
        "verdict": state["gemini_verdict"],
        "suggested_actions": [
            f"Block {state['src_ip']}",
            "Review related events",
            "Monitor for 24 hours",
        ],
        "requires_human_review": state.get("requires_human_review", False),
        "fallback_used": True,
    }

    return state


# Export for testing
__all__ = [
    "gemini_judge_node",
    "get_gemini_client",
]

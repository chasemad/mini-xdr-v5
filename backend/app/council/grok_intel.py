"""
Grok Intel Node - External Threat Intelligence Scout

Grok searches X (Twitter) for real-time threat intelligence about:
- File hashes being discussed by security researchers
- Recently registered domains
- IP addresses associated with campaigns
- Zero-day vulnerabilities and exploits

This adds "Feature #80" - Internet-aware threat detection.
"""

import logging
import os
import time
from typing import Any, Dict, Optional

from app.orchestrator.graph import XDRState

logger = logging.getLogger(__name__)

# Grok API configuration (when available)
GROK_API_KEY = os.getenv("GROK_API_KEY")
GROK_API_URL = os.getenv("GROK_API_URL", "https://api.x.ai/v1")


async def grok_intel_node(state: XDRState) -> Dict[str, Any]:
    """
    Query Grok for external threat intelligence from X (Twitter).

    This node is engaged when:
    - Unknown file hashes detected
    - Recently registered domains visited
    - IOCs that need real-time validation

    Args:
        state: Current XDRState

    Returns:
        Updated state with Grok intelligence
    """
    start_time = time.time()

    logger.info(f"Grok Intel scouting for {state['src_ip']}")

    # Track API call
    state["api_calls_made"].append("grok_intel")

    # Record API call (if actually calling Grok)
    from app.orchestrator.metrics import estimate_api_cost, record_api_call

    # Extract IOCs from events
    iocs = _extract_iocs(state["events"])

    if not iocs:
        logger.info("No IOCs to query - skipping Grok")
        state["grok_intel"] = "No IOCs detected"
        state["grok_threat_score"] = 0.0
        return state

    try:
        # Query Grok for each IOC
        intel_results = []

        for ioc_type, ioc_value in iocs:
            logger.info(f"Querying Grok about {ioc_type}: {ioc_value}")

            # Check if Grok API is configured
            if not GROK_API_KEY:
                logger.warning("GROK_API_KEY not configured - using placeholder")
                intel_results.append(_placeholder_grok_response(ioc_type, ioc_value))
            else:
                # Call Grok API (implementation when API is available)
                result = await _query_grok_api(ioc_type, ioc_value)
                intel_results.append(result)

        # Aggregate results
        threat_score = _calculate_grok_threat_score(intel_results)
        intel_summary = _format_grok_intel(intel_results)

        state["grok_intel"] = intel_summary
        state["grok_threat_score"] = threat_score
        state["grok_references"] = [r.get("url") for r in intel_results if r.get("url")]

        # Add Grok score as "Feature #80"
        if len(state["raw_features"]) == 79:
            state["raw_features"].append(threat_score / 100.0)  # Normalize to 0-1
            logger.info(f"Added Feature #80 (Grok Threat Score): {threat_score}/100")

        elapsed_ms = (time.time() - start_time) * 1000
        state["processing_time_ms"] += elapsed_ms

        logger.info(
            f"Grok Intel completed in {elapsed_ms:.0f}ms: "
            f"threat_score={threat_score}/100"
        )

    except Exception as e:
        logger.error(f"Grok Intel error: {e}", exc_info=True)
        state["grok_intel"] = f"Error querying Grok: {str(e)}"
        state["grok_threat_score"] = 0.0

    return state


def _extract_iocs(events: list) -> list:
    """
    Extract Indicators of Compromise from events.

    Returns list of (ioc_type, ioc_value) tuples.
    """
    iocs = []

    for event in events:
        # File hashes
        if file_hash := event.get("file_hash"):
            iocs.append(("hash", file_hash))

        # Domains
        if domain := event.get("domain"):
            iocs.append(("domain", domain))

        # Destination IPs (external only)
        if dst_ip := event.get("dst_ip"):
            if not _is_private_ip(dst_ip):
                iocs.append(("ip", dst_ip))

    # Deduplicate
    return list(set(iocs))


def _is_private_ip(ip: str) -> bool:
    """Check if IP is in private ranges."""
    parts = ip.split(".")
    if len(parts) != 4:
        return False

    first_octet = int(parts[0])
    second_octet = int(parts[1])

    # Private ranges: 10.x.x.x, 172.16-31.x.x, 192.168.x.x
    return (
        first_octet == 10
        or (first_octet == 172 and 16 <= second_octet <= 31)
        or (first_octet == 192 and second_octet == 168)
    )


async def _query_grok_api(ioc_type: str, ioc_value: str) -> Dict[str, Any]:
    """
    Query Grok API for threat intelligence (placeholder for when API is available).

    Args:
        ioc_type: Type of IOC (hash, domain, ip)
        ioc_value: The IOC value

    Returns:
        Dict with threat intelligence
    """
    # TODO: Implement actual Grok API call when xAI releases the API

    # For now, return placeholder
    return _placeholder_grok_response(ioc_type, ioc_value)


def _placeholder_grok_response(ioc_type: str, ioc_value: str) -> Dict[str, Any]:
    """
    Placeholder response until Grok API is available.

    In production, this would query X for security researcher discussions.
    """
    return {
        "ioc_type": ioc_type,
        "ioc_value": ioc_value,
        "threat_score": 0,  # 0-100
        "mentions": 0,
        "sentiment": "neutral",
        "related_campaigns": [],
        "researcher_notes": [],
        "url": None,
        "last_updated": "N/A",
        "status": "grok_api_not_configured",
    }


def _calculate_grok_threat_score(intel_results: list) -> float:
    """
    Calculate aggregate threat score from Grok intelligence.

    Scoring:
    - High threat score (80-100): Multiple security researcher mentions, known malicious
    - Medium threat score (40-80): Some mentions, suspicious
    - Low threat score (0-40): No mentions or benign
    """
    if not intel_results:
        return 0.0

    # Average threat scores
    scores = [r.get("threat_score", 0) for r in intel_results]
    avg_score = sum(scores) / len(scores)

    # Boost if multiple IOCs are flagged
    flagged_count = sum(1 for s in scores if s > 50)
    if flagged_count > 1:
        avg_score = min(100, avg_score * 1.2)  # 20% boost

    return round(avg_score, 1)


def _format_grok_intel(intel_results: list) -> str:
    """Format Grok intelligence for human readability."""
    if not intel_results:
        return "No external intelligence found"

    lines = ["External Threat Intelligence (from Grok):"]

    for result in intel_results:
        ioc = f"{result['ioc_type'].upper()}: {result['ioc_value']}"
        score = result.get("threat_score", 0)

        if score > 70:
            status = "HIGH RISK"
        elif score > 40:
            status = "SUSPICIOUS"
        else:
            status = "Unknown/Clean"

        lines.append(f"  - {ioc}: {status} (score: {score}/100)")

        if mentions := result.get("mentions"):
            lines.append(f"    Mentioned by {mentions} security researchers")

        if campaigns := result.get("related_campaigns"):
            lines.append(f"    Campaigns: {', '.join(campaigns)}")

    return "\n".join(lines)


# Export
__all__ = ["grok_intel_node"]

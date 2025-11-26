"""
OpenAI Remediation Node - The Tactical Engineer

OpenAI GPT-4o generates precise remediation scripts and action plans:
- Firewall rules (Palo Alto, Cisco, iptables)
- PowerShell scripts for endpoint response
- Network isolation commands
- Forensic collection scripts

This ensures automated responses are accurate and safe.
"""

import json
import logging
import os
import time
from typing import Any, Dict, List

try:
    from langchain_openai import ChatOpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logging.warning("OpenAI SDK not available")

from app.orchestrator.graph import XDRState

logger = logging.getLogger(__name__)

# Initialize OpenAI client (singleton)
_openai_client = None


def get_openai_client():
    """Get or create OpenAI client instance."""
    global _openai_client

    if _openai_client is None and OPENAI_AVAILABLE:
        try:
            # Use settings to get API key (loaded from .env file)
            from ..config import settings

            api_key = settings.openai_api_key or os.getenv("OPENAI_API_KEY")

            if not api_key:
                logger.warning(
                    "OPENAI_API_KEY not set - remediation will use templates"
                )
                return None

            _openai_client = ChatOpenAI(
                model="gpt-4o",  # Precise, fast, cost-effective
                api_key=api_key,
                temperature=0.0,  # Deterministic for security actions
            )

            logger.info("OpenAI Remediation client initialized")

        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            _openai_client = None

    return _openai_client


async def openai_remediation_node(state: XDRState) -> Dict[str, Any]:
    """
    Generate precise remediation scripts using OpenAI GPT-4o.

    This node is engaged when:
    - Final verdict is THREAT
    - Confidence score > 0.70 (high enough to act)
    - Automated response is approved

    Args:
        state: Current XDRState with threat classification

    Returns:
        Updated state with remediation plan
    """
    start_time = time.time()

    logger.info(
        f"OpenAI generating remediation for {state['src_ip']}: "
        f"{state['ml_prediction']['class']}"
    )

    # Track API call
    state["api_calls_made"].append("openai_remediation")

    client = get_openai_client()

    if client is None:
        # Fallback: Use template-based remediation
        return _template_remediation(state)

    try:
        # Build remediation prompt
        prompt = _build_remediation_prompt(state)

        # Call OpenAI
        response = await client.ainvoke(prompt)

        # Parse response
        remediation_data = _parse_remediation_response(response.content)

        # Format for frontend - expects recommended_actions as list of strings
        action_descriptions = []
        for action in remediation_data.get("actions", []):
            if isinstance(action, dict):
                action_descriptions.append(action.get("description", str(action)))
            else:
                action_descriptions.append(str(action))

        state["openai_remediation"] = {
            "recommended_actions": action_descriptions,
            "success_criteria": remediation_data.get("success_criteria", ""),
            "monitoring_plan": remediation_data.get("monitoring_plan", ""),
            "raw_response": response.content,
            "actions_detailed": remediation_data.get("actions", []),
        }
        state["openai_action_plan"] = remediation_data.get("actions", [])

        # Update action plan in state
        state["action_plan"].extend(action_descriptions)

        elapsed_ms = (time.time() - start_time) * 1000
        state["processing_time_ms"] += elapsed_ms

        logger.info(
            f"OpenAI Remediation completed in {elapsed_ms:.0f}ms: "
            f"{len(state['action_plan'])} actions planned"
        )

    except Exception as e:
        logger.error(f"OpenAI Remediation error: {e}", exc_info=True)
        state["error"] = f"OpenAI Remediation failed: {str(e)}"
        return _template_remediation(state)

    return state


def _build_remediation_prompt(state: XDRState) -> str:
    """Build detailed prompt for OpenAI to generate remediation."""
    ml_pred = state["ml_prediction"]
    src_ip = state["src_ip"]
    attack_type = ml_pred["class"]

    # Get Gemini's reasoning if available
    gemini_context = ""
    if state.get("gemini_reasoning"):
        gemini_context = f"\n\nSecurity Analysis:\n{state['gemini_reasoning']}"

    prompt = f"""
You are a cybersecurity automation engineer generating precise remediation actions.

INCIDENT DETAILS:
- Source IP: {src_ip}
- Attack Type: {attack_type}
- Confidence: {state['confidence_score']:.2%}
- Severity: HIGH

{gemini_context}

YOUR TASK:
Generate a safe, precise action plan to contain and remediate this threat.

REQUIREMENTS:
1. **Be Specific**: Include exact commands, not just descriptions
2. **Be Safe**: Verify commands won't break legitimate services
3. **Be Reversible**: Include rollback instructions
4. **Prioritize**: List actions in order of importance

AVAILABLE ACTIONS:
- Block IP address (firewall, iptables, cloud security groups)
- Isolate host (network segmentation)
- Kill process (EDR integration)
- Quarantine file (move to secure location)
- Disable user account (Active Directory)
- Capture forensics (memory dump, network capture)
- Alert SOC team (send notification)

CONTEXT-SPECIFIC GUIDANCE:
- Brute Force Attack: Block IP, disable compromised accounts, enable MFA
- DDoS Attack: Rate limit, block IP ranges, enable DDoS mitigation
- Web Attack: Block IP, patch vulnerable endpoint, review logs
- Malware: Isolate host, quarantine files, scan network for spread

OUTPUT FORMAT (JSON):
{{
  "actions": [
    {{
      "priority": 1,
      "action_type": "block_ip",
      "description": "Block source IP at firewall",
      "command": "sudo iptables -A INPUT -s {src_ip} -j DROP",
      "rollback_command": "sudo iptables -D INPUT -s {src_ip} -j DROP",
      "requires_approval": false,
      "estimated_impact": "Low - single IP blocked"
    }},
    {{
      "priority": 2,
      "action_type": "isolate_host",
      "description": "Isolate affected host from network",
      "command": "...",
      "rollback_command": "...",
      "requires_approval": true,
      "estimated_impact": "High - host becomes unavailable"
    }}
  ],
  "success_criteria": "IP blocked, no further malicious activity observed",
  "monitoring_plan": "Monitor for 24 hours, check for evasion attempts"
}}

Generate the remediation plan:
"""

    return prompt


def _parse_remediation_response(response_text: str) -> Dict[str, Any]:
    """Parse OpenAI's remediation response into structured data."""
    try:
        # Try to extract JSON
        json_start = response_text.find("{")
        json_end = response_text.rfind("}") + 1

        if json_start != -1 and json_end > json_start:
            json_str = response_text[json_start:json_end]
            data = json.loads(json_str)
            return data
        else:
            return json.loads(response_text)

    except json.JSONDecodeError:
        logger.warning("Failed to parse JSON from OpenAI, using text parsing")

        # Fallback: Extract action items from text
        actions = []
        lines = response_text.split("\n")

        for line in lines:
            if "block" in line.lower() or "isolate" in line.lower():
                actions.append(
                    {
                        "priority": len(actions) + 1,
                        "action_type": "manual",
                        "description": line.strip(),
                        "requires_approval": True,
                    }
                )

        return {"actions": actions}


def _template_remediation(state: XDRState) -> Dict[str, Any]:
    """
    Fallback template-based remediation when OpenAI unavailable.

    Uses predefined playbooks for common attack types.
    """
    logger.info("Using template-based remediation (OpenAI unavailable)")

    attack_type = state["ml_prediction"]["class"]
    src_ip = state["src_ip"]

    # Template playbooks
    templates = {
        "BruteForce": [
            f"Block IP {src_ip} at firewall for 24 hours",
            "Review authentication logs for compromised accounts",
            "Enable MFA for targeted accounts",
            "Update password policies",
        ],
        "DDoS": [
            f"Rate limit traffic from {src_ip}",
            "Enable DDoS mitigation (CloudFlare, AWS Shield)",
            "Block IP range if part of botnet",
            "Scale infrastructure if needed",
        ],
        "WebAttack": [
            f"Block IP {src_ip} at WAF",
            "Review and patch vulnerable endpoints",
            "Enable stricter input validation",
            "Scan for web shells",
        ],
        "Malware": [
            "Isolate affected hosts immediately",
            f"Block C2 IP {src_ip}",
            "Quarantine malicious files",
            "Initiate EDR scan on network",
        ],
    }

    actions = templates.get(
        attack_type,
        [f"Block IP {src_ip}", "Investigate manually", "Review logs for IOCs"],
    )

    state["openai_action_plan"] = actions
    state["action_plan"].extend(actions)
    state["openai_remediation"] = {
        "recommended_actions": actions,
        "success_criteria": f"Threat from {src_ip} contained",
        "monitoring_plan": "Monitor for 24 hours for any follow-up activity",
        "template_used": attack_type,
    }

    return state


# Export
__all__ = ["openai_remediation_node", "get_openai_client"]

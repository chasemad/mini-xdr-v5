import os
import json
import logging
from typing import Any, Dict, List
from .config import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a Tier-1 SOC analyst. Respond ONLY as compact JSON with keys: "
    "summary (1-2 sentences), severity (low|medium|high), "
    "recommendation (contain_now|watch|ignore), rationale (array of 3 short bullets)."
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "block_ip",
            "parameters": {
                "type": "object",
                "properties": {
                    "incident_id": {"type": "integer"}
                },
                "required": ["incident_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "schedule_unblock",
            "parameters": {
                "type": "object",
                "properties": {
                    "incident_id": {"type": "integer"},
                    "minutes": {"type": "integer", "minimum": 1}
                },
                "required": ["incident_id", "minutes"]
            }
        }
    },
]


def _openai_triage(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Triage using OpenAI API"""
    try:
        from openai import OpenAI
        
        client = OpenAI(api_key=settings.openai_api_key)
        
        response = client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(payload)}
            ],
            tools=TOOLS,
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content
        return json.loads(content)
        
    except Exception as e:
        logger.error(f"OpenAI triage error: {e}")
        return {
            "summary": f"LLM parsing error: {str(e)[:100]}",
            "severity": "low",
            "recommendation": "watch",
            "rationale": ["Failed to parse OpenAI response", "Manual review required", "Check API configuration"]
        }


def _xai_triage(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Triage using xAI (Grok) API"""
    try:
        import requests
        
        headers = {
            "Authorization": f"Bearer {settings.xai_api_key}",
            "Content-Type": "application/json"
        }
        
        body = {
            "model": settings.xai_model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(payload)}
            ],
            "tools": TOOLS,
            "temperature": 0
        }
        
        response = requests.post(
            "https://api.x.ai/v1/chat/completions",
            headers=headers,
            json=body,
            timeout=30
        )
        response.raise_for_status()
        
        data = response.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "{}")
        
        return json.loads(content)
        
    except Exception as e:
        logger.error(f"xAI triage error: {e}")
        return {
            "summary": f"LLM parsing error (xAI): {str(e)[:100]}",
            "severity": "low",
            "recommendation": "watch",
            "rationale": ["Failed to parse xAI response", "Manual review required", "Check API configuration"]
        }


def run_triage(incident: Dict[str, Any], recent_events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Run triage analysis on an incident
    
    Args:
        incident: Incident data dictionary
        recent_events: List of recent event dictionaries
        
    Returns:
        Triage note dictionary with summary, severity, recommendation, rationale
    """
    # Prepare payload for LLM
    payload = {
        "incident": incident,
        "recent_events": recent_events[-20:]  # Limit to last 20 events
    }
    
    provider = settings.llm_provider.lower()
    
    logger.info(f"Running triage for incident {incident.get('id')} using provider: {provider}")
    
    if provider == "xai":
        return _xai_triage(payload)
    elif provider == "ollama":
        from .local_triager import ollama_triage
        return ollama_triage(payload)
    else:
        return _openai_triage(payload)


def generate_default_triage(incident: Dict[str, Any], event_count: int = 0) -> Dict[str, Any]:
    """Generate a default triage note when LLM is unavailable"""
    severity = "medium" if event_count >= 10 else "low"
    recommendation = "contain_now" if event_count >= 15 else "watch"
    
    return {
        "summary": f"SSH brute-force detected from {incident['src_ip']} with {event_count} attempts.",
        "severity": severity,
        "recommendation": recommendation,
        "rationale": [
            f"Detected {event_count} failed SSH attempts",
            "Source IP exhibits brute-force behavior pattern",
            "Automatic threshold-based detection triggered"
        ]
    }

"""
Local LLM triage using Ollama
Install Ollama and run: ollama pull llama3.2:3b
"""
import json
import requests
from typing import Dict, Any, List

def ollama_triage(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Triage using local Ollama model"""
    try:
        prompt = f"""
You are a cybersecurity analyst. Analyze this SSH brute-force incident and respond with JSON:

Incident: {json.dumps(payload['incident'])}
Recent Events: {len(payload['recent_events'])} events

Respond with JSON only:
{{
  "summary": "Brief 1-2 sentence summary",
  "severity": "low|medium|high", 
  "recommendation": "contain_now|watch|ignore",
  "rationale": ["reason1", "reason2", "reason3"]
}}
"""
        
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.2:3b",
                "prompt": prompt,
                "stream": False,
                "format": "json"
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return json.loads(result['response'])
            
    except Exception as e:
        print(f"Ollama error: {e}")
    
    # Fallback
    return {
        "summary": "Local analysis unavailable",
        "severity": "medium",
        "recommendation": "watch", 
        "rationale": ["Local LLM failed", "Manual review needed"]
    }

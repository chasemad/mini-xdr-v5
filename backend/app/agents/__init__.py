"""
AI Agents for Mini-XDR

This package contains specialized AI agents for autonomous security operations:
- Attribution Agent: Threat actor identification
- Containment Agent: Incident containment orchestration
- Forensics Agent: Evidence collection
- Deception Agent: Honeypot management
- IAM Agent: Active Directory monitoring
- EDR Agent: Endpoint detection and response
- DLP Agent: Data loss prevention
- Threat Hunter: Proactive threat hunting
- Rollback Agent: False positive remediation
"""

from .attribution_agent import attribution_tracker
from .containment_agent import containment_orchestrator
from .deception_agent import deception_manager
from .dlp_agent import dlp_agent
from .forensics_agent import forensics_investigator

__all__ = [
    "attribution_tracker",
    "containment_orchestrator",
    "forensics_investigator",
    "deception_manager",
    "dlp_agent",
]

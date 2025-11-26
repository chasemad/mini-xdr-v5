"""
AI Agents for Mini-XDR

This package contains specialized AI agents for autonomous security operations:
- Attribution Agent: Threat actor identification, campaign analysis, IP reputation
- Containment Agent: Incident containment orchestration, IP blocking, emergency isolation
- Forensics Agent: Evidence collection, timeline reconstruction, case management
- Deception Agent: Honeypot management, attacker profiling, adaptive lures
- IAM Agent: Active Directory monitoring, credential protection, identity management
- EDR Agent: Endpoint detection and response, process control, file quarantine
- DLP Agent: Data loss prevention, sensitive data scanning, upload blocking
- Predictive Hunter: Proactive threat hunting, behavioral analysis, hypothesis generation
- NLP Analyzer: Natural language processing for security events
- Ingestion Agent: Data ingestion and normalization
"""

# Core agents
from .attribution_agent import AttributionAgent, attribution_tracker
from .containment_agent import ContainmentAgent, containment_orchestrator

# Coordination
from .coordination_hub import (
    AdvancedCoordinationHub,
    AgentCapability,
    ConflictResolutionStrategy,
    CoordinationContext,
    CoordinationStrategy,
)
from .deception_agent import DeceptionAgent, deception_manager
from .dlp_agent import DLPAgent, dlp_agent

# Extended agents
from .edr_agent import EDRAgent
from .forensics_agent import ForensicsAgent, forensics_investigator
from .iam_agent import IAMAgent
from .ingestion_agent import IngestionAgent
from .nlp_analyzer import NaturalLanguageThreatAnalyzer as NLPAnalyzer
from .predictive_hunter import PredictiveThreatHunter as PredictiveHuntingAgent

__all__ = [
    # Core agent classes
    "AttributionAgent",
    "ContainmentAgent",
    "ForensicsAgent",
    "DeceptionAgent",
    "EDRAgent",
    "IAMAgent",
    "DLPAgent",
    "PredictiveHuntingAgent",
    "NLPAnalyzer",
    "IngestionAgent",
    # Singleton instances (for backward compatibility)
    "attribution_tracker",
    "containment_orchestrator",
    "forensics_investigator",
    "deception_manager",
    "dlp_agent",
    # Coordination components
    "AdvancedCoordinationHub",
    "AgentCapability",
    "CoordinationContext",
    "CoordinationStrategy",
    "ConflictResolutionStrategy",
]

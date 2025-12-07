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
- LangChain Orchestrator: ReAct-style agent orchestration using GPT-4
- LangChain Tools: Standardized tool wrappers for agent capabilities
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

# LangChain integration (Phase 3)
try:
    from .langchain_orchestrator import (
        LangChainOrchestrator,
        OrchestrationResult,
        langchain_orchestrator,
        orchestrate_with_langchain,
    )
    from .tools import create_xdr_tools, get_tool_descriptions

    LANGCHAIN_AGENTS_AVAILABLE = True
except ImportError:
    LANGCHAIN_AGENTS_AVAILABLE = False
    langchain_orchestrator = None
    orchestrate_with_langchain = None

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
    # LangChain integration
    "LangChainOrchestrator",
    "OrchestrationResult",
    "langchain_orchestrator",
    "orchestrate_with_langchain",
    "create_xdr_tools",
    "get_tool_descriptions",
    "LANGCHAIN_AGENTS_AVAILABLE",
]

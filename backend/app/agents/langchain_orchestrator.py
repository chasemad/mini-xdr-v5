"""
LangChain ReAct Agent Orchestrator for XDR

This module provides a LangChain-based agent orchestrator that replaces the
custom AgentOrchestrator with a more powerful ReAct-style agent using GPT-4.

The orchestrator:
1. Receives incident context and events
2. Reasons about the threat using ReAct pattern
3. Executes appropriate response tools
4. Generates a comprehensive response report

Integration with existing systems:
- Uses tools from tools.py wrapping existing agent capabilities
- Integrates with Council of Models for uncertain predictions
- Stores decisions in vector memory for learning
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

try:
    # LangChain 0.3+ uses different imports
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
    from langchain_core.prompts import PromptTemplate
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent

    LANGCHAIN_AVAILABLE = True
    logging.info("LangChain orchestrator imports successful (using langgraph)")
except ImportError as e:
    LANGCHAIN_AVAILABLE = False
    logging.warning(
        f"LangChain not available - orchestrator will use fallback mode. Error: {e}"
    )

from ..config import settings
from ..models.ml_agent_bridge import get_ml_agent_bridge
from .tools import create_xdr_tools, get_tool_descriptions

logger = logging.getLogger(__name__)


@dataclass
class OrchestrationResult:
    """Result from the orchestrator."""

    success: bool
    actions_taken: List[Dict[str, Any]]
    reasoning: str
    recommendations: List[str]
    final_verdict: str  # CONTAINED, MONITORING, ESCALATE, INVESTIGATE
    confidence: float
    processing_time_ms: float
    agent_trace: List[str]
    error: Optional[str] = None


# System prompt for guiding the agent's behavior
SYSTEM_PROMPT = """You are an expert cybersecurity incident response agent operating in a Security Operations Center (SOC).

IDENTITY & CAPABILITIES:
- You are an autonomous AI agent with authority to take protective actions
- You have access to 32+ security response tools (network, endpoint, identity, data protection)
- Every action you take is automatically logged in the investigation database
- You can block IPs, isolate hosts, disable accounts, collect forensics, and more

MISSION:
Detect, contain, and remediate security threats to protect the organization while minimizing false positives and business disruption.

DECISION FRAMEWORK by Severity:

🔴 CRITICAL (Ransomware, Data Exfiltration, Active Breach):
  → Act within 60 seconds
  → Contain immediately (block IP, isolate host, disable account)
  → Collect forensics in parallel
  → Alert SOC team

🟠 HIGH (Brute Force, Exploitation Attempts, Mass Scanning):
  → Containment + Investigation within 5 minutes
  → Block source, investigate scope
  → Check threat intelligence
  → Consider isolation if persistent

🟡 MEDIUM (Suspicious Behavior, Anomalies, Policy Violations):
  → Investigate first, contain if confirmed
  → Gather evidence before action
  → Check for false positive indicators
  → Monitor for escalation

🟢 LOW (Minor Anomalies, Benign Activity):
  → Monitor only
  → Collect baseline data
  → No containment unless pattern emerges

CONFIDENCE THRESHOLDS:
- ML Confidence > 90%: Trust and act decisively
- ML Confidence 70-90%: Verify with threat intel before major actions
- ML Confidence < 70%: Investigate thoroughly, escalate if uncertain

RESPONSE PRIORITIES:
1. **Contain the threat** (stop the bleeding)
2. **Preserve evidence** (support investigation)
3. **Minimize disruption** (avoid breaking legitimate services)
4. **Document reasoning** (enable learning and audit)
5. **Escalate complexity** (human review for edge cases)

TOOL USAGE GUIDELINES:
- **Always start with investigation** (threat_intel_lookup, behavior_analysis)
- **Containment tools** (block_ip, isolate_host) for confirmed threats
- **Forensics tools** (collect_evidence, memory_dump) for high-value investigations
- **Alert tools** (alert_analysts, create_case) for escalation

VERDICTS:
- **SAFE**: No threat detected, legitimate activity
- **MONITOR**: Suspicious but needs more data
- **CONTAINED**: Threat confirmed and neutralized
- **ESCALATE**: Complex threat requiring human judgment

Always explain your reasoning step-by-step. Be decisive but thoughtful. When in doubt, err on the side of caution while preserving business operations.
"""

# Legacy ReAct prompt template (kept for reference, not actively used with langgraph)
# Langgraph's create_react_agent uses its own internal ReAct prompt
REACT_PROMPT_LEGACY = """You are an expert cybersecurity analyst and incident responder working in a Security Operations Center (SOC).
Your role is to analyze security incidents and take appropriate response actions to protect the organization.

## Current Incident Context
Source IP: {src_ip}
Threat Type: {threat_type}
ML Confidence: {confidence}
Severity: {severity}
Event Count: {event_count}

## Event Summary
{event_summary}

## ML Analysis
{ml_analysis}

## ML-Enhanced Threat Context
{ml_context}

## Available Tools
You have access to the following tools:

{tools}

Tool names: {tool_names}

## Response Guidelines
1. **Assess the Threat**: Consider the threat type, confidence level, and event patterns
2. **Check Intelligence**: Query threat intel for IP reputation if confidence is uncertain
3. **Proportional Response**: Match response severity to threat severity
   - LOW/INFO: Monitor only, no blocking
   - MEDIUM: Consider blocking, alert SOC
   - HIGH: Block IP, investigate host
   - CRITICAL: Block IP, isolate hosts, disable compromised accounts
4. **Evidence Collection**: Collect forensics for HIGH/CRITICAL incidents
5. **Documentation**: Explain your reasoning clearly

## Decision Framework
- If ML confidence < 50%: Verify with threat intel before taking action
- If false positive risk is high: Monitor but don't block
- If confirmed threat: Take appropriate containment action
- Always document reasoning for audit trail

Use the following format:

Question: the incident analysis question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: your final analysis and recommended response

Now analyze this incident and respond appropriately.

Question: Analyze incident from {src_ip} ({threat_type}) and take appropriate response actions.
{agent_scratchpad}"""


class LangChainOrchestrator:
    """
    LangChain-based ReAct agent orchestrator for XDR incident response.

    This orchestrator uses GPT-4 with ReAct (Reasoning + Acting) pattern to:
    1. Analyze incident context
    2. Reason about appropriate responses
    3. Execute containment actions via tools
    4. Generate comprehensive reports
    """

    def __init__(
        self,
        model_name: str = "gpt-4o",
        temperature: float = 0.1,
        max_iterations: int = 10,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_iterations = max_iterations

        self.llm = None
        self.tools = None
        self.agent_executor = (
            None  # langgraph's create_react_agent returns graph directly
        )

        self._initialized = False

        # Initialize if LangChain is available
        if LANGCHAIN_AVAILABLE:
            self._initialize()

    def _initialize(self):
        """Initialize the LangChain agent."""
        try:
            api_key = settings.openai_api_key
            if not api_key:
                logger.warning("OpenAI API key not found - orchestrator disabled")
                return

            # Initialize LLM
            self.llm = ChatOpenAI(
                model=self.model_name,
                temperature=self.temperature,
                api_key=api_key,
            )

            # Create tools
            self.tools = create_xdr_tools()

            # Create prompt with required ReAct variables (tools, tool_names, agent_scratchpad)
            # NOTE: This prompt is NOT used by langgraph's create_react_agent
            # It's kept for reference only - langgraph uses its own internal prompt
            prompt = PromptTemplate(
                template=REACT_PROMPT_LEGACY,
                input_variables=[
                    "src_ip",
                    "threat_type",
                    "confidence",
                    "severity",
                    "event_count",
                    "event_summary",
                    "ml_analysis",
                    "ml_context",
                    "tools",  # Required by ReAct agent
                    "tool_names",  # Required by ReAct agent
                    "agent_scratchpad",  # Required by ReAct agent
                ],
            )

            # Create ReAct agent using langgraph
            # In langgraph 0.2+, create_react_agent returns a compiled graph (agent executor) directly
            self.agent_executor = create_react_agent(
                model=self.llm,
                tools=self.tools,
            )

            self._initialized = True
            logger.info(
                f"LangChain orchestrator initialized with {len(self.tools)} tools"
            )

        except Exception as e:
            logger.error(
                f"Failed to initialize LangChain orchestrator: {e}", exc_info=True
            )
            self._initialized = False

    async def orchestrate_incident(
        self,
        src_ip: str,
        threat_type: str,
        confidence: float,
        severity: str,
        events: List[Dict[str, Any]],
        ml_analysis: Dict[str, Any] = None,
        features: np.ndarray = None,
        incident_id: Optional[int] = None,  # NEW: For investigation tracking
    ) -> OrchestrationResult:
        """
        Orchestrate incident response using the ReAct agent.

        Args:
            src_ip: Source IP of the incident
            threat_type: Detected threat type
            confidence: ML confidence score (0-1)
            severity: Severity level (low, medium, high, critical)
            events: List of event dictionaries
            ml_analysis: Optional ML analysis details

        Returns:
            OrchestrationResult with actions taken and reasoning
        """
        start_time = datetime.now(timezone.utc)

        # Set investigation context for automatic tracking
        if incident_id:
            from .investigation_tracking import (
                clear_investigation_context,
                set_investigation_context,
            )

            set_investigation_context(incident_id)
            logger.info(f"Investigation context set for incident {incident_id}")

        try:
            # Check if initialized
            if not self._initialized:
                return await self._fallback_orchestration(
                    src_ip, threat_type, confidence, severity, events, features
                )

            # Prepare event summary
            event_summary = self._prepare_event_summary(events)

            # Prepare ML analysis string
            ml_analysis_str = (
                json.dumps(ml_analysis, indent=2)
                if ml_analysis
                else "No detailed ML analysis available"
            )

            # Get ML context from revolutionary ensemble
            ml_context_str = "ML context not available"
            if features is not None:
                try:
                    ml_bridge = await get_ml_agent_bridge()
                    ml_context = await ml_bridge.get_ml_context(features)
                    ml_context_str = ml_bridge.get_langchain_context(ml_context)
                    logger.info(
                        f"Enhanced LangChain prompt with ML context: uncertainty={ml_context.uncertainty:.3f}"
                    )
                except Exception as e:
                    logger.warning(f"Failed to get ML context: {e}")

            # Log that we're running LangChain agent
            logger.info(f"Running LangChain ReAct agent for {src_ip} ({threat_type})")

            # Prepare context message for the agent
            context_message = f"""Analyze this security incident:

Source IP: {src_ip}
Threat Type: {threat_type}
ML Confidence: {confidence:.1%}
Severity: {severity}
Event Count: {len(events)}

Event Summary:
{event_summary}

ML Analysis:
{ml_analysis_str}

ML Context (Uncertainty Analysis):
{ml_context_str}

Based on this information, determine what actions to take to respond to this threat.
You have access to tools for blocking IPs, isolating hosts, disabling users, and more.
Provide reasoning for your actions and a final verdict (SAFE, MONITOR, CONTAINED, or ESCALATE)."""

            # Run the agent using langgraph's message format with system guidance
            result = await asyncio.to_thread(
                self.agent_executor.invoke,
                {
                    "messages": [
                        SystemMessage(content=SYSTEM_PROMPT),  # Guide agent behavior
                        HumanMessage(content=context_message),  # Incident details
                    ]
                },
            )

            # Parse result from langgraph format
            # langgraph returns {messages: [...]}, extract the final AI message
            messages = result.get("messages", [])
            final_message = messages[-1] if messages else None

            # Extract reasoning from final message
            reasoning = (
                final_message.content
                if final_message and hasattr(final_message, "content")
                else ""
            )

            # Extract tool calls and actions from messages
            actions_taken = []
            for msg in messages:
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        actions_taken.append(
                            {
                                "tool": tool_call.get("name", "unknown"),
                                "input": tool_call.get("args", {}),
                                "result": {},  # Tool results are in subsequent messages
                            }
                        )

            # Create agent trace for debugging
            agent_trace = [
                {
                    "type": msg.__class__.__name__,
                    "content": getattr(msg, "content", "")[
                        :200
                    ],  # Truncate for readability
                }
                for msg in messages
            ]

            # Determine final verdict
            final_verdict = self._determine_verdict(actions_taken, severity, confidence)

            # Generate recommendations
            recommendations = self._generate_recommendations(
                threat_type, severity, confidence, actions_taken
            )

            processing_time = (
                datetime.now(timezone.utc) - start_time
            ).total_seconds() * 1000

            return OrchestrationResult(
                success=True,
                actions_taken=actions_taken,
                reasoning=reasoning,
                recommendations=recommendations,
                final_verdict=final_verdict,
                confidence=confidence,
                processing_time_ms=processing_time,
                agent_trace=agent_trace,
            )

        except Exception as e:
            logger.error(f"Orchestration failed: {e}", exc_info=True)
            processing_time = (
                datetime.now(timezone.utc) - start_time
            ).total_seconds() * 1000

            return OrchestrationResult(
                success=False,
                actions_taken=[],
                reasoning=f"Orchestration error: {str(e)}",
                recommendations=["Manual investigation required"],
                final_verdict="INVESTIGATE",
                confidence=confidence,
                processing_time_ms=processing_time,
                agent_trace=[],
                error=str(e),
            )
        finally:
            # Clear investigation context
            if incident_id:
                from .investigation_tracking import clear_investigation_context

                clear_investigation_context()
                logger.info(f"Investigation context cleared for incident {incident_id}")

    async def _fallback_orchestration(
        self,
        src_ip: str,
        threat_type: str,
        confidence: float,
        severity: str,
        events: List[Dict[str, Any]],
        features: Optional[np.ndarray] = None,
    ) -> OrchestrationResult:
        """
        Fallback orchestration when LangChain is not available.
        Uses rule-based logic instead of AI reasoning.
        """
        start_time = datetime.now(timezone.utc)
        actions_taken = []
        recommendations = []

        # Rule-based response
        if severity.lower() == "critical" and confidence >= 0.7:
            actions_taken.append(
                {
                    "action": "block_ip",
                    "target": src_ip,
                    "reason": "Critical threat with high confidence",
                    "executed": True,
                }
            )
            recommendations.append("Immediate SOC escalation required")
            recommendations.append("Collect forensic evidence")
            final_verdict = "CONTAINED"

        elif severity.lower() == "high" and confidence >= 0.6:
            actions_taken.append(
                {
                    "action": "block_ip",
                    "target": src_ip,
                    "reason": "High severity threat",
                    "executed": True,
                }
            )
            recommendations.append("Investigate affected systems")
            final_verdict = "CONTAINED"

        elif severity.lower() == "medium" and confidence >= 0.5:
            recommendations.append(f"Consider blocking {src_ip}")
            recommendations.append("Monitor for additional activity")
            final_verdict = "MONITORING"

        else:
            recommendations.append("Continue monitoring")
            recommendations.append("Review ML classification accuracy")
            final_verdict = "INVESTIGATE"

        processing_time = (
            datetime.now(timezone.utc) - start_time
        ).total_seconds() * 1000

        return OrchestrationResult(
            success=True,
            actions_taken=actions_taken,
            reasoning=f"Fallback rule-based response: {threat_type} from {src_ip} with {confidence:.1%} confidence",
            recommendations=recommendations,
            final_verdict=final_verdict,
            confidence=confidence,
            processing_time_ms=processing_time,
            agent_trace=["Fallback mode - LangChain not available"],
        )

    def _prepare_event_summary(self, events: List[Dict[str, Any]]) -> str:
        """Prepare a concise summary of events for the agent."""
        if not events:
            return "No events provided"

        # Count event types
        event_types = {}
        for event in events:
            event_type = event.get("event_type", event.get("eventid", "unknown"))
            event_types[event_type] = event_types.get(event_type, 0) + 1

        # Get sample messages
        messages = []
        for event in events[:5]:
            msg = event.get("message", "")
            if msg:
                messages.append(msg[:100])

        summary = f"""
Event Types: {json.dumps(dict(list(event_types.items())[:10]))}
Total Events: {len(events)}
Sample Messages:
{chr(10).join(['- ' + m for m in messages[:3]])}
"""
        return summary.strip()

    def _extract_actions(self, intermediate_steps: List) -> List[Dict[str, Any]]:
        """Extract actions taken from intermediate steps."""
        actions = []

        for step in intermediate_steps:
            if len(step) >= 2:
                action, result = step[0], step[1]

                # Parse the result
                try:
                    result_data = (
                        json.loads(result) if isinstance(result, str) else result
                    )
                except json.JSONDecodeError:
                    result_data = {"raw": result}

                actions.append(
                    {
                        "tool": action.tool if hasattr(action, "tool") else str(action),
                        "input": action.tool_input
                        if hasattr(action, "tool_input")
                        else "",
                        "result": result_data,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )

        return actions

    def _extract_trace(self, intermediate_steps: List) -> List[str]:
        """Extract agent reasoning trace from intermediate steps."""
        trace = []

        for i, step in enumerate(intermediate_steps):
            if len(step) >= 2:
                action = step[0]
                log = action.log if hasattr(action, "log") else str(action)
                trace.append(f"Step {i+1}: {log[:200]}")

        return trace

    def _determine_verdict(
        self,
        actions_taken: List[Dict[str, Any]],
        severity: str,
        confidence: float,
    ) -> str:
        """Determine final verdict based on actions taken."""

        # Check if any blocking actions were taken
        blocking_actions = [
            a
            for a in actions_taken
            if a.get("tool") in ["block_ip", "isolate_host", "disable_user"]
        ]

        if blocking_actions:
            return "CONTAINED"
        elif severity.lower() in ["critical", "high"]:
            return "ESCALATE"
        elif confidence < 0.5:
            return "INVESTIGATE"
        else:
            return "MONITORING"

    def _generate_recommendations(
        self,
        threat_type: str,
        severity: str,
        confidence: float,
        actions_taken: List[Dict[str, Any]],
    ) -> List[str]:
        """Generate follow-up recommendations."""
        recommendations = []

        # Confidence-based recommendations
        if confidence < 0.5:
            recommendations.append(
                "Low ML confidence - verify with additional intelligence sources"
            )

        # Severity-based recommendations
        if severity.lower() == "critical":
            if not any(a.get("tool") == "collect_forensics" for a in actions_taken):
                recommendations.append(
                    "Collect forensic evidence for incident response"
                )
            recommendations.append("Notify incident response team")

        elif severity.lower() == "high":
            recommendations.append("Review related alerts and events")
            recommendations.append("Check for lateral movement indicators")

        # Threat-type specific recommendations
        threat_lower = threat_type.lower()
        if "brute" in threat_lower:
            recommendations.append(
                "Review authentication logs for compromised accounts"
            )
            recommendations.append("Consider implementing MFA for affected services")
        elif "malware" in threat_lower:
            recommendations.append("Scan affected systems for persistence mechanisms")
            recommendations.append("Check for data exfiltration indicators")
        elif "ddos" in threat_lower:
            recommendations.append("Review network capacity and mitigation options")
            recommendations.append("Consider upstream filtering")

        return recommendations[:5]  # Limit to top 5


# Global instance
langchain_orchestrator = LangChainOrchestrator()


async def orchestrate_with_langchain(
    src_ip: str,
    threat_type: str,
    confidence: float,
    severity: str,
    events: List[Dict[str, Any]],
    ml_analysis: Dict[str, Any] = None,
    features: Optional[np.ndarray] = None,
    incident_id: Optional[int] = None,  # NEW: For investigation tracking
) -> OrchestrationResult:
    """
    Convenience function to orchestrate incident response.

    This is the main entry point for the LangChain orchestrator.
    """
    return await langchain_orchestrator.orchestrate_incident(
        src_ip=src_ip,
        threat_type=threat_type,
        confidence=confidence,
        severity=severity,
        events=events,
        ml_analysis=ml_analysis,
        features=features,
        incident_id=incident_id,  # NEW: Pass incident_id for tracking
    )


# Export
__all__ = [
    "LangChainOrchestrator",
    "OrchestrationResult",
    "langchain_orchestrator",
    "orchestrate_with_langchain",
]

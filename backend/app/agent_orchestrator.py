"""
Agent Orchestration Framework for Mini-XDR
Coordinates communication and decision fusion between AI agents
"""
import asyncio
import ipaddress
import json
import logging
import uuid
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from sqlalchemy import select

from .agents.attribution_agent import AttributionAgent
from .agents.containment_agent import ContainmentAgent
from .agents.coordination_hub import (
    AdvancedCoordinationHub,
    AgentCapability,
    ConflictResolutionStrategy,
    CoordinationContext,
    CoordinationStrategy,
)
from .agents.deception_agent import DeceptionAgent
from .agents.dlp_agent import DLPAgent
from .agents.edr_agent import EDRAgent
from .agents.forensics_agent import ForensicsAgent
from .agents.iam_agent import IAMAgent
from .agents.predictive_hunter import PredictiveThreatHunter
from .config import settings
from .models import Action, Event, Incident

# Define logger BEFORE imports that might use it in except blocks
logger = logging.getLogger(__name__)

# LangChain orchestrator integration (Phase 3)
try:
    from .agents.langchain_orchestrator import (
        OrchestrationResult,
        langchain_orchestrator,
        orchestrate_with_langchain,
    )

    LANGCHAIN_ORCHESTRATOR_AVAILABLE = (
        langchain_orchestrator._initialized if langchain_orchestrator else False
    )
except ImportError:
    LANGCHAIN_ORCHESTRATOR_AVAILABLE = False
    langchain_orchestrator = None
    orchestrate_with_langchain = None
    logger.info("LangChain orchestrator not available - using standard orchestration")


class AgentRole(Enum):
    """Roles that agents can play in the orchestration"""

    ATTRIBUTION = "attribution"
    CONTAINMENT = "containment"
    FORENSICS = "forensics"
    DECEPTION = "deception"
    EDR = "edr"
    IAM = "iam"
    DLP = "dlp"
    PREDICTIVE_HUNTER = "predictive_hunter"
    COORDINATOR = "coordinator"


class MessageType(Enum):
    """Types of messages agents can exchange"""

    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    COORDINATION = "coordination"
    DECISION = "decision"
    ERROR = "error"


class WorkflowStatus(Enum):
    """Status of orchestration workflows"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class AgentMessage:
    """Message format for inter-agent communication"""

    message_id: str
    sender: str
    recipient: str
    message_type: MessageType
    payload: Dict[str, Any]
    timestamp: datetime
    correlation_id: str
    priority: int = 1  # 1-5, higher is more important
    ttl: int = 300  # Time to live in seconds


@dataclass
class WorkflowContext:
    """Context for orchestration workflows"""

    workflow_id: str
    incident_id: int
    status: WorkflowStatus
    start_time: datetime
    end_time: Optional[datetime]
    agents_involved: List[str]
    current_step: str
    results: Dict[str, Any]
    decisions: List[Dict[str, Any]]
    errors: List[str]


@dataclass
class AgentDecision:
    """Decision made by an agent"""

    agent_id: str
    decision_type: str
    confidence: float
    reasoning: str
    actions: List[Dict[str, Any]]
    timestamp: datetime
    evidence: Dict[str, Any]


class SharedAgentMemory:
    """Shared memory system for agents to store and retrieve information"""

    def __init__(self):
        self.memory: Dict[str, Any] = {}
        self.ttl_cache: Dict[str, datetime] = {}

    async def store(self, key: str, value: Any, ttl_seconds: int = 3600):
        """Store information in shared memory"""
        self.memory[key] = value
        self.ttl_cache[key] = datetime.utcnow() + timedelta(seconds=ttl_seconds)

    async def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve information from shared memory"""
        if key in self.ttl_cache and datetime.utcnow() > self.ttl_cache[key]:
            # Key has expired
            await self.delete(key)
            return None

        return self.memory.get(key)

    async def delete(self, key: str):
        """Delete information from shared memory"""
        self.memory.pop(key, None)
        self.ttl_cache.pop(key, None)

    async def cleanup_expired(self):
        """Clean up expired entries"""
        now = datetime.utcnow()
        expired_keys = [key for key, expiry in self.ttl_cache.items() if now > expiry]

        for key in expired_keys:
            await self.delete(key)

        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired memory entries")


class AgentOrchestrator:
    """
    Central orchestrator for coordinating AI agents in the Mini-XDR system
    """

    def __init__(self):
        self.agent_id = "orchestrator_v2"
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Initialize all agents
        self.agents = {
            AgentRole.ATTRIBUTION: AttributionAgent(),
            AgentRole.CONTAINMENT: ContainmentAgent(),
            AgentRole.FORENSICS: ForensicsAgent(),
            AgentRole.DECEPTION: DeceptionAgent(),
            AgentRole.EDR: EDRAgent(),
            AgentRole.IAM: IAMAgent(),
            AgentRole.DLP: DLPAgent(),
            AgentRole.PREDICTIVE_HUNTER: PredictiveThreatHunter(),
        }

        # Advanced coordination system
        self.coordination_hub = AdvancedCoordinationHub()

        # Shared systems
        self.shared_memory = SharedAgentMemory()
        self.message_queue: List[AgentMessage] = []
        self.active_workflows: Dict[str, WorkflowContext] = {}

        # Configuration
        self.max_concurrent_workflows = 10
        self.message_timeout = 30  # seconds
        self.decision_timeout = 60  # seconds

        # Statistics
        self.stats = {
            "workflows_completed": 0,
            "workflows_failed": 0,
            "messages_processed": 0,
            "decisions_made": 0,
            "coordinations_executed": 0,
            "conflicts_resolved": 0,
            "agent_invocations": 0,
            "start_time": datetime.utcnow(),
        }

    async def initialize(self):
        """Initialize the orchestrator and all agents"""
        try:
            self.logger.info("Initializing Enhanced Agent Orchestrator...")

            # Initialize shared memory cleanup task
            asyncio.create_task(self._periodic_cleanup())

            # Register agents with coordination hub
            await self._register_agents_with_coordination_hub()

            # Test agent connectivity
            await self._test_agent_connectivity()

            self.logger.info("Enhanced Agent Orchestrator initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize orchestrator: {e}")
            raise

    async def _test_agent_connectivity(self):
        """Test connectivity to all agents"""
        for role, agent in self.agents.items():
            try:
                # Simple connectivity test
                if hasattr(agent, "agent_id"):
                    self.logger.info(
                        f"Agent {role.value} ({agent.agent_id}) is responsive"
                    )
                else:
                    self.logger.warning(
                        f"Agent {role.value} may not be properly initialized"
                    )
            except Exception as e:
                self.logger.error(f"Connectivity test failed for {role.value}: {e}")

    async def _periodic_cleanup(self):
        """Periodic cleanup of expired data"""
        while True:
            try:
                await self.shared_memory.cleanup_expired()

                # Clean up old workflows
                cutoff_time = datetime.utcnow() - timedelta(hours=24)
                old_workflows = [
                    wf_id
                    for wf_id, wf in self.active_workflows.items()
                    if wf.status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED]
                    and wf.end_time
                    and wf.end_time < cutoff_time
                ]

                for wf_id in old_workflows:
                    del self.active_workflows[wf_id]

                if old_workflows:
                    self.logger.info(f"Cleaned up {len(old_workflows)} old workflows")

            except Exception as e:
                self.logger.error(f"Periodic cleanup failed: {e}")

            await asyncio.sleep(300)  # Run every 5 minutes

    async def _register_agents_with_coordination_hub(self):
        """Register all agents with the coordination hub"""

        # Define agent capabilities
        agent_capabilities = {
            AgentRole.ATTRIBUTION: AgentCapability(
                name="attribution_agent",
                domain_expertise=[
                    "threat_attribution",
                    "campaign_analysis",
                    "actor_profiling",
                    "ip_reputation",
                ],
                confidence_threshold=0.7,
                execution_time_estimate=15.0,
                resource_requirements={"cpu": 0.2, "memory": 0.3},
                dependencies=[],
                success_rate=0.85,
            ),
            AgentRole.CONTAINMENT: AgentCapability(
                name="containment_agent",
                domain_expertise=[
                    "incident_containment",
                    "ip_blocking",
                    "threat_response",
                    "emergency_isolation",
                ],
                confidence_threshold=0.8,
                execution_time_estimate=5.0,
                resource_requirements={"cpu": 0.1, "memory": 0.1},
                dependencies=[],
                success_rate=0.92,
            ),
            AgentRole.FORENSICS: AgentCapability(
                name="forensics_agent",
                domain_expertise=[
                    "evidence_collection",
                    "forensic_analysis",
                    "case_management",
                    "timeline_reconstruction",
                ],
                confidence_threshold=0.75,
                execution_time_estimate=30.0,
                resource_requirements={"cpu": 0.3, "memory": 0.4},
                dependencies=[],
                success_rate=0.88,
            ),
            AgentRole.DECEPTION: AgentCapability(
                name="deception_agent",
                domain_expertise=[
                    "deception_deployment",
                    "attacker_profiling",
                    "honeypot_management",
                    "adaptive_lures",
                ],
                confidence_threshold=0.6,
                execution_time_estimate=20.0,
                resource_requirements={"cpu": 0.2, "memory": 0.2},
                dependencies=[],
                success_rate=0.78,
            ),
            AgentRole.EDR: AgentCapability(
                name="edr_agent",
                domain_expertise=[
                    "endpoint_detection",
                    "process_monitoring",
                    "file_quarantine",
                    "host_isolation",
                    "registry_analysis",
                ],
                confidence_threshold=0.75,
                execution_time_estimate=10.0,
                resource_requirements={"cpu": 0.2, "memory": 0.2},
                dependencies=[],
                success_rate=0.90,
            ),
            AgentRole.IAM: AgentCapability(
                name="iam_agent",
                domain_expertise=[
                    "identity_management",
                    "credential_monitoring",
                    "privilege_escalation_detection",
                    "service_account_audit",
                ],
                confidence_threshold=0.8,
                execution_time_estimate=8.0,
                resource_requirements={"cpu": 0.1, "memory": 0.2},
                dependencies=[],
                success_rate=0.88,
            ),
            AgentRole.DLP: AgentCapability(
                name="dlp_agent",
                domain_expertise=[
                    "data_classification",
                    "exfiltration_detection",
                    "sensitive_data_scanning",
                    "policy_enforcement",
                ],
                confidence_threshold=0.7,
                execution_time_estimate=12.0,
                resource_requirements={"cpu": 0.3, "memory": 0.3},
                dependencies=[],
                success_rate=0.85,
            ),
            AgentRole.PREDICTIVE_HUNTER: AgentCapability(
                name="predictive_hunter_agent",
                domain_expertise=[
                    "threat_prediction",
                    "behavioral_analysis",
                    "hypothesis_generation",
                    "anomaly_detection",
                    "proactive_hunting",
                ],
                confidence_threshold=0.65,
                execution_time_estimate=45.0,
                resource_requirements={"cpu": 0.4, "memory": 0.5},
                dependencies=[],
                success_rate=0.80,
            ),
        }

        # Register each agent
        for role, capabilities in agent_capabilities.items():
            self.coordination_hub.register_agent(role.value, capabilities)
            self.logger.info(f"Registered {role.value} with coordination hub")

    def _normalize_agent_role(self, agent_type: str) -> AgentRole:
        if not agent_type:
            return AgentRole.ATTRIBUTION
        agent_lower = agent_type.lower()
        for role in AgentRole:
            if role.value == agent_lower:
                return role
        alias_map = {
            "threat_intel": AgentRole.ATTRIBUTION,
            "intel": AgentRole.ATTRIBUTION,
            "investigation": AgentRole.FORENSICS,
            "response": AgentRole.CONTAINMENT,
            "deceive": AgentRole.DECEPTION,
            "endpoint": AgentRole.EDR,
            "endpoint_detection": AgentRole.EDR,
            "identity": AgentRole.IAM,
            "active_directory": AgentRole.IAM,
            "ad": AgentRole.IAM,
            "data_loss_prevention": AgentRole.DLP,
            "data_protection": AgentRole.DLP,
            "hunter": AgentRole.PREDICTIVE_HUNTER,
            "threat_hunter": AgentRole.PREDICTIVE_HUNTER,
            "hunting": AgentRole.PREDICTIVE_HUNTER,
            "prediction": AgentRole.PREDICTIVE_HUNTER,
        }
        if agent_lower in alias_map:
            return alias_map[agent_lower]
        raise ValueError(f"Unknown agent type: {agent_type}")

    def _normalize_context_input(self, context: Optional[Any]) -> Dict[str, Any]:
        if context is None:
            return {}
        if isinstance(context, dict):
            return context
        if isinstance(context, str):
            return {"topic": context}
        return {"value": context}

    def _looks_like_ip(self, value: Optional[str]) -> bool:
        if not value:
            return False
        try:
            ipaddress.ip_address(value)
            return True
        except ValueError:
            return False

    async def _gather_incident_snapshot(
        self, incident_id: Optional[int], db_session
    ) -> Dict[str, Any]:
        if not incident_id or not db_session:
            return {}

        try:
            incident = await db_session.get(Incident, incident_id)
            if not incident:
                return {}

            events_result = await db_session.execute(
                select(Event)
                .where(Event.src_ip == incident.src_ip)
                .order_by(Event.ts.desc())
                .limit(50)
            )
            events = events_result.scalars().all()

            actions_result = await db_session.execute(
                select(Action)
                .where(Action.incident_id == incident_id)
                .order_by(Action.created_at.desc())
                .limit(20)
            )
            actions = actions_result.scalars().all()

            return {"incident": incident, "events": events, "actions": actions}
        except Exception as snapshot_error:
            self.logger.debug(f"Failed to gather incident snapshot: {snapshot_error}")
            return {}

    def _build_attribution_summary(
        self, snapshot: Dict[str, Any], task: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        incident: Optional[Incident] = snapshot.get("incident")
        events: List[Event] = snapshot.get("events", [])
        ip = None
        if incident:
            ip = incident.src_ip
        ip = ip or context.get("ip") or context.get("source_ip")

        event_counts = Counter(e.eventid for e in events if getattr(e, "eventid", None))
        username_counts = Counter()
        password_counts = Counter()

        for event in events:
            raw = event.raw
            if not isinstance(raw, dict):
                try:
                    raw = json.loads(raw) if raw else {}
                except Exception:
                    raw = {}

            username = raw.get("username") or raw.get("user") or raw.get("login")
            if username:
                username_counts[username] += 1

            password = raw.get("password") or raw.get("pass")
            if password:
                password_counts[password] += 1

        summary_fragments = []
        if ip:
            summary_fragments.append(f"IP {ip}")
        summary_fragments.append(f"{len(events)} events analyzed")
        if event_counts:
            top_event, top_count = event_counts.most_common(1)[0]
            summary_fragments.append(f"dominant event {top_event} ({top_count})")

        detail_body = "; ".join(summary_fragments)
        analysis = {
            "task": task,
            "source_ip": ip,
            "event_counts": dict(event_counts),
            "top_usernames": username_counts.most_common(5),
            "top_passwords": password_counts.most_common(5),
            "context": context,
        }

        return {
            "detail": f"Threat attribution analysis completed: {detail_body}",
            "response": detail_body,
            "analysis": analysis,
        }

    def _build_containment_summary(
        self, snapshot: Dict[str, Any], task: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        incident: Optional[Incident] = snapshot.get("incident")
        actions: List[Action] = snapshot.get("actions", [])
        ip = None
        if incident:
            ip = incident.src_ip
        ip = ip or context.get("ip") or context.get("source_ip")

        containment_actions = [
            action
            for action in actions
            if action.action
            and any(
                keyword in action.action
                for keyword in ["block", "isolate", "waf", "contain"]
            )
        ]
        success_actions = [
            action
            for action in containment_actions
            if (action.result or "").lower() in {"success", "completed"}
        ]
        failure_actions = [
            action
            for action in containment_actions
            if (action.result or "").lower() in {"failed", "error"}
        ]

        last_detail = containment_actions[0].detail if containment_actions else None
        detail = f"Containment summary for {ip or 'target'}: {len(success_actions)} successful, {len(failure_actions)} failed actions"
        if last_detail:
            detail += f"; last action detail: {last_detail[:120]}"

        analysis = {
            "task": task,
            "source_ip": ip,
            "actions_reviewed": len(containment_actions),
            "recent_actions": [
                {
                    "action": action.action,
                    "result": action.result,
                    "detail": action.detail,
                    "created_at": action.created_at.isoformat()
                    if action.created_at
                    else None,
                }
                for action in containment_actions[:5]
            ],
            "context": context,
        }

        return {"detail": detail, "response": detail, "analysis": analysis}

    def _build_forensics_summary(
        self, snapshot: Dict[str, Any], task: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        incident: Optional[Incident] = snapshot.get("incident")
        events: List[Event] = snapshot.get("events", [])
        ip = None
        if incident:
            ip = incident.src_ip
        ip = ip or context.get("ip") or context.get("source_ip")

        command_events = [
            event
            for event in events
            if event.eventid and "command" in event.eventid.lower()
        ]
        file_events = [
            event
            for event in events
            if event.eventid
            and any(
                token in event.eventid.lower()
                for token in ["file", "download", "upload"]
            )
        ]

        detail = (
            f"Forensic review for {ip or 'target'}: {len(events)} events, "
            f"{len(command_events)} command interactions, {len(file_events)} file activities"
        )

        latest_commands = []
        for event in command_events[:5]:
            raw = event.raw
            if not isinstance(raw, dict):
                try:
                    raw = json.loads(raw) if raw else {}
                except Exception:
                    raw = {}
            latest_commands.append(
                raw.get("input") or raw.get("command") or event.message
            )

        analysis = {
            "task": task,
            "source_ip": ip,
            "events_analyzed": len(events),
            "command_events": len(command_events),
            "file_events": len(file_events),
            "sample_commands": [cmd for cmd in latest_commands if cmd],
            "context": context,
        }

        return {"detail": detail, "response": detail, "analysis": analysis}

    def _build_deception_summary(
        self, snapshot: Dict[str, Any], task: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        events: List[Event] = snapshot.get("events", [])
        unique_sources = {event.src_ip for event in events if event.src_ip}
        detail = (
            f"Deception telemetry: {len(unique_sources)} unique sources interacting with honeypot "
            f"across {len(events)} recent events"
        )

        analysis = {
            "task": task,
            "unique_sources": sorted(unique_sources),
            "events_analyzed": len(events),
            "context": context,
        }

        return {"detail": detail, "response": detail, "analysis": analysis}

    async def orchestrate_agent_task(
        self,
        agent_type: str,
        task: str,
        query: Optional[str] = None,
        context: Optional[Any] = None,
        incident_id: Optional[int] = None,
        db_session=None,
    ) -> Dict[str, Any]:
        """
        Orchestrate a task by invoking the appropriate AI agent.

        This method now actually invokes the agent's analysis methods rather than
        just building summaries from database data.
        """
        role = self._normalize_agent_role(agent_type)
        context_data = self._normalize_context_input(context)
        invocation_start = datetime.utcnow()

        # Provide IP context from query if possible
        if not context_data.get("ip") and query and self._looks_like_ip(query.strip()):
            context_data["ip"] = query.strip()

        # Gather incident snapshot for context
        snapshot = await self._gather_incident_snapshot(incident_id, db_session)
        incident = snapshot.get("incident")
        events = snapshot.get("events", [])

        self.logger.info(f"🤖 Invoking {role.value} agent for task: {task}")
        self.stats["agent_invocations"] += 1

        try:
            # Get the agent instance
            agent = self.agents.get(role)
            if not agent:
                raise ValueError(f"Agent {role.value} not found in orchestrator")

            # Execute agent-specific tasks
            result = await self._execute_agent_task(
                role, agent, task, query, context_data, incident, events, db_session
            )

            execution_time = (datetime.utcnow() - invocation_start).total_seconds()

            self.logger.info(
                f"✅ {role.value} agent completed task '{task}' in {execution_time:.2f}s"
            )

            return {
                "success": True,
                "detail": result.get("detail", f"{role.value} analysis completed"),
                "response": result.get("response", result.get("detail")),
                "analysis": result.get("analysis", result),
                "agent": role.value,
                "task": task,
                "context": context_data,
                "execution_time": execution_time,
                "agent_invoked": True,
            }

        except Exception as e:
            self.logger.error(f"❌ {role.value} agent task '{task}' failed: {e}")

            # Fallback to summary builders for basic response
            fallback_result = await self._fallback_to_summary(
                role, snapshot, task, context_data
            )

            return {
                "success": False,
                "error": str(e),
                "fallback_used": True,
                "detail": fallback_result.get("detail"),
                "response": fallback_result.get("response"),
                "analysis": fallback_result.get("analysis"),
                "agent": role.value,
                "task": task,
                "context": context_data,
            }

    async def _execute_agent_task(
        self,
        role: AgentRole,
        agent: Any,
        task: str,
        query: Optional[str],
        context: Dict[str, Any],
        incident: Optional[Incident],
        events: List[Event],
        db_session,
    ) -> Dict[str, Any]:
        """Execute the appropriate method based on agent role and task"""

        ip_address = context.get("ip") or (incident.src_ip if incident else None)

        if role == AgentRole.ATTRIBUTION:
            return await self._execute_attribution_task(
                agent, task, incident, events, ip_address, db_session
            )

        elif role == AgentRole.CONTAINMENT:
            return await self._execute_containment_task(
                agent, task, incident, events, context, db_session
            )

        elif role == AgentRole.FORENSICS:
            return await self._execute_forensics_task(
                agent, task, incident, events, db_session
            )

        elif role == AgentRole.DECEPTION:
            return await self._execute_deception_task(
                agent, task, incident, events, context
            )

        elif role == AgentRole.EDR:
            return await self._execute_edr_task(agent, task, context, incident)

        elif role == AgentRole.IAM:
            return await self._execute_iam_task(agent, task, context, events, incident)

        elif role == AgentRole.DLP:
            return await self._execute_dlp_task(agent, task, context, incident)

        elif role == AgentRole.PREDICTIVE_HUNTER:
            return await self._execute_hunter_task(agent, task, incident, events)

        else:
            raise ValueError(f"No execution handler for agent role: {role.value}")

    async def _execute_attribution_task(
        self,
        agent,
        task: str,
        incident: Optional[Incident],
        events: List[Event],
        ip_address: Optional[str],
        db_session,
    ) -> Dict[str, Any]:
        """Execute Attribution Agent tasks"""

        # Threat actor analysis tasks
        if task in [
            "analyze_threat_actor",
            "analyze_attribution",
            "campaign_analysis",
            "profile_threat_actor",
            "identify_botnet_campaign",
        ]:
            incidents = [incident] if incident else []
            result = await agent.analyze_attribution(incidents, events, db_session)
            actor_info = result.get("actor_attribution", {})
            return {
                "detail": f"Attribution analysis completed: {result.get('confidence_score', 0):.2f} confidence, actor: {actor_info.get('primary_actor', 'unknown')}",
                "analysis": result,
            }

        # IP reputation and threat intel
        elif task in ["ip_reputation", "analyze_ip", "threat_intel"] and ip_address:
            result = await agent.analyze_ip_reputation(ip_address)
            reputation_score = result.get("reputation_score", 0)
            return {
                "detail": f"IP {ip_address} reputation score: {reputation_score}/100",
                "analysis": result,
            }

        # Scanner/reconnaissance profiling
        elif task in ["profile_scanner", "analyze_reconnaissance"]:
            incidents = [incident] if incident else []
            result = await agent.analyze_attribution(incidents, events, db_session)
            infra = result.get("infrastructure_analysis", {})
            return {
                "detail": f"Scanner profiled: {infra.get('unique_sources', 0)} unique sources, {infra.get('geographic_distribution', {})}",
                "analysis": result,
            }

        # TTP analysis
        elif task == "ttp_analysis":
            if hasattr(agent, "_analyze_ttps"):
                result = await agent._analyze_ttps(events)
                return {
                    "detail": f"TTP analysis identified {len(result.get('identified_ttps', []))} techniques",
                    "analysis": result,
                }

        # Default attribution analysis for any unhandled task
        incidents = [incident] if incident else []
        result = await agent.analyze_attribution(incidents, events, db_session)
        return {
            "detail": f"Attribution analysis ({task}): confidence {result.get('confidence_score', 0):.2f}",
            "analysis": result,
        }

    async def _execute_containment_task(
        self,
        agent,
        task: str,
        incident: Optional[Incident],
        events: List[Event],
        context: Dict[str, Any],
        db_session,
    ) -> Dict[str, Any]:
        """Execute Containment Agent tasks"""

        # Emergency containment tasks
        if task in [
            "orchestrate_response",
            "contain",
            "auto_contain",
            "emergency_isolation",
            "isolate_and_terminate",
        ]:
            if incident:
                result = await agent.orchestrate_response(incident, events, db_session)
                actions = result.get("actions", [])
                return {
                    "detail": f"Emergency containment: {len(actions)} actions executed - {result.get('reason', 'completed')}",
                    "response": result.get("reason"),
                    "analysis": result,
                }

        # IP blocking
        elif task in ["block_ip", "ip_block"]:
            ip_address = context.get("ip") or (incident.src_ip if incident else None)
            if ip_address and incident:
                result = await agent.orchestrate_response(incident, events, db_session)
                return {
                    "detail": f"IP blocking action for {ip_address}",
                    "analysis": result,
                }

        # Full isolation
        elif task == "full_isolation":
            if incident:
                result = await agent.orchestrate_response(incident, events, db_session)
                return {
                    "detail": f"Full isolation containment for {incident.src_ip}",
                    "analysis": result,
                }

        # Rate limiting (DDoS mitigation)
        elif task in ["enable_rate_limiting", "rate_limit", "ddos_mitigation"]:
            if incident:
                result = await agent.orchestrate_response(incident, events, db_session)
                return {
                    "detail": f"Rate limiting enabled for {incident.src_ip} - DDoS mitigation active",
                    "analysis": result,
                    "mitigation_type": "rate_limiting",
                }

        # Host correlation (lateral movement)
        elif task in ["correlate_hosts", "lateral_movement_analysis"]:
            if incident:
                result = await agent.orchestrate_response(incident, events, db_session)
                # Extract related hosts from events
                related_hosts = set()
                for event in events:
                    if hasattr(event, "dst_ip") and event.dst_ip:
                        related_hosts.add(event.dst_ip)
                return {
                    "detail": f"Host correlation: {len(related_hosts)} related hosts identified for {incident.src_ip}",
                    "analysis": result,
                    "related_hosts": list(related_hosts)[:10],
                }

        # Default to orchestrate_response
        if incident:
            result = await agent.orchestrate_response(incident, events, db_session)
            return {
                "detail": f"Containment task '{task}' executed for {incident.src_ip}",
                "analysis": result,
            }

        return {
            "detail": f"Containment task '{task}' - no incident provided",
            "analysis": {},
        }

    async def _execute_forensics_task(
        self,
        agent,
        task: str,
        incident: Optional[Incident],
        events: List[Event],
        db_session,
    ) -> Dict[str, Any]:
        """Execute Forensics Agent tasks"""

        # Case initiation
        if task in ["initiate_case", "start_investigation", "create_case"]:
            if incident:
                case_id = await agent.initiate_forensic_case(
                    incident=incident,
                    investigator="orchestrator",
                    evidence_types=["event_logs", "network_artifacts"],
                )
                return {
                    "detail": f"Forensic case initiated: {case_id}",
                    "analysis": {"case_id": case_id},
                }

        # Evidence collection
        elif task in [
            "collect_evidence",
            "evidence_collection",
            "capture_session_details",
        ]:
            if incident:
                case_id = await agent.initiate_forensic_case(
                    incident=incident,
                    investigator="orchestrator",
                    evidence_types=[
                        "event_logs",
                        "network_artifacts",
                        "command_logs",
                        "session_data",
                    ],
                )
                await agent.collect_evidence(
                    case_id=case_id,
                    incident=incident,
                    evidence_types=[
                        "event_logs",
                        "network_artifacts",
                        "command_logs",
                        "session_data",
                    ],
                    db_session=db_session,
                )
                return {
                    "detail": f"Evidence collected for case {case_id} ({len(events)} events captured)",
                    "analysis": {"case_id": case_id, "events_captured": len(events)},
                }

        # General attack pattern analysis
        elif task in ["analyze_evidence", "analyze_attack_pattern", "full_analysis"]:
            if incident:
                case_id = await agent.initiate_forensic_case(
                    incident=incident,
                    investigator="orchestrator",
                    evidence_types=["event_logs", "network_artifacts"],
                )
                await agent.collect_evidence(
                    case_id=case_id,
                    incident=incident,
                    evidence_types=["event_logs", "network_artifacts"],
                    db_session=db_session,
                )
                analysis_result = await agent.analyze_evidence(
                    case_id=case_id, evidence_ids=None
                )
                return {
                    "detail": f"Attack pattern analysis complete: {analysis_result.get('risk_assessment', {}).get('overall_risk_score', 0):.2f} risk",
                    "analysis": {"case_id": case_id, "analysis": analysis_result},
                }

        # Web attack analysis (SQL injection, XSS, etc.)
        elif task in [
            "analyze_web_attack",
            "analyze_injection_payload",
            "analyze_xss_payload",
            "analyze_database_attack",
        ]:
            if incident:
                case_id = await agent.initiate_forensic_case(
                    incident=incident,
                    investigator="orchestrator",
                    evidence_types=["event_logs", "http_requests", "payload_samples"],
                )
                await agent.collect_evidence(
                    case_id=case_id,
                    incident=incident,
                    evidence_types=["event_logs", "http_requests"],
                    db_session=db_session,
                )
                analysis_result = await agent.analyze_evidence(
                    case_id=case_id, evidence_ids=None
                )
                return {
                    "detail": f"Web attack forensics complete for {incident.src_ip}: {task}",
                    "analysis": {
                        "case_id": case_id,
                        "attack_type": task,
                        "analysis": analysis_result,
                    },
                }

        # Network pattern analysis
        elif task in ["analyze_network_pattern", "analyze_data_transfer"]:
            if incident:
                case_id = await agent.initiate_forensic_case(
                    incident=incident,
                    investigator="orchestrator",
                    evidence_types=[
                        "network_artifacts",
                        "traffic_captures",
                        "data_flows",
                    ],
                )
                await agent.collect_evidence(
                    case_id=case_id,
                    incident=incident,
                    evidence_types=["network_artifacts", "event_logs"],
                    db_session=db_session,
                )
                analysis_result = await agent.analyze_evidence(
                    case_id=case_id, evidence_ids=None
                )
                return {
                    "detail": f"Network pattern analysis: {len(events)} network events analyzed",
                    "analysis": {"case_id": case_id, "analysis": analysis_result},
                }

        # Command chain / shell activity analysis
        elif task in ["analyze_command_chain", "analyze_exploit_pattern"]:
            if incident:
                # Filter for command events
                command_events = [
                    e for e in events if e.eventid and "command" in e.eventid.lower()
                ]
                case_id = await agent.initiate_forensic_case(
                    incident=incident,
                    investigator="orchestrator",
                    evidence_types=[
                        "command_logs",
                        "shell_history",
                        "process_execution",
                    ],
                )
                await agent.collect_evidence(
                    case_id=case_id,
                    incident=incident,
                    evidence_types=["command_logs", "event_logs"],
                    db_session=db_session,
                )
                analysis_result = await agent.analyze_evidence(
                    case_id=case_id, evidence_ids=None
                )
                return {
                    "detail": f"Command chain analysis: {len(command_events)} command executions analyzed",
                    "analysis": {
                        "case_id": case_id,
                        "command_events": len(command_events),
                        "analysis": analysis_result,
                    },
                }

        # Privilege escalation analysis
        elif task in ["analyze_privilege_escalation"]:
            if incident:
                case_id = await agent.initiate_forensic_case(
                    incident=incident,
                    investigator="orchestrator",
                    evidence_types=["auth_logs", "privilege_changes", "sudo_logs"],
                )
                await agent.collect_evidence(
                    case_id=case_id,
                    incident=incident,
                    evidence_types=["event_logs", "auth_logs"],
                    db_session=db_session,
                )
                analysis_result = await agent.analyze_evidence(
                    case_id=case_id, evidence_ids=None
                )
                return {
                    "detail": f"Privilege escalation investigation for {incident.src_ip}",
                    "analysis": {"case_id": case_id, "analysis": analysis_result},
                }

        # Account activity analysis
        elif task in ["analyze_account_activity"]:
            if incident:
                case_id = await agent.initiate_forensic_case(
                    incident=incident,
                    investigator="orchestrator",
                    evidence_types=["auth_logs", "session_logs", "credential_usage"],
                )
                await agent.collect_evidence(
                    case_id=case_id,
                    incident=incident,
                    evidence_types=["event_logs", "auth_logs"],
                    db_session=db_session,
                )
                analysis_result = await agent.analyze_evidence(
                    case_id=case_id, evidence_ids=None
                )
                # Count unique usernames from events
                usernames = set()
                for event in events:
                    if hasattr(event, "raw") and isinstance(event.raw, dict):
                        if event.raw.get("username"):
                            usernames.add(event.raw.get("username"))
                return {
                    "detail": f"Account activity analysis: {len(usernames)} unique accounts identified",
                    "analysis": {
                        "case_id": case_id,
                        "accounts_found": list(usernames)[:10],
                        "analysis": analysis_result,
                    },
                }

        # Timeline reconstruction
        elif task == "timeline_reconstruction":
            if incident and hasattr(agent, "_build_attack_timeline"):
                timeline = await agent._build_attack_timeline(events)
                return {
                    "detail": f"Timeline reconstructed with {len(timeline.get('timeline_events', []))} events",
                    "analysis": timeline,
                }

        # Default forensic analysis for any unhandled task
        if incident:
            case_id = await agent.initiate_forensic_case(
                incident=incident,
                investigator="orchestrator",
                evidence_types=["event_logs"],
            )
            await agent.collect_evidence(
                case_id=case_id,
                incident=incident,
                evidence_types=["event_logs"],
                db_session=db_session,
            )
            analysis = await agent.analyze_evidence(case_id=case_id, evidence_ids=None)
            return {
                "detail": f"Forensic task '{task}' completed for {incident.src_ip}",
                "analysis": {"case_id": case_id, "task": task, "analysis": analysis},
            }

        return {
            "detail": f"Forensic task '{task}' - no incident provided",
            "analysis": {},
        }

    async def _execute_deception_task(
        self,
        agent,
        task: str,
        incident: Optional[Incident],
        events: List[Event],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute Deception Agent tasks"""

        if task in ["analyze_behavior", "attacker_profiling", "profile_attacker"]:
            profiles = await agent.analyze_attacker_behavior(
                events=events, timeframe_hours=24
            )
            return {
                "detail": f"Analyzed {len(profiles)} attacker profiles",
                "analysis": {"attacker_profiles": profiles},
            }

        elif task in ["generate_strategy", "deception_strategy", "adaptive_deception"]:
            threat_intel = context.get("threat_intel", {})
            current_attacks = context.get(
                "attack_types", ["brute_force", "reconnaissance"]
            )
            org_profile = context.get(
                "org_profile", {"industry": "technology", "size": "enterprise"}
            )

            strategy = await agent.ai_powered_deception_strategy(
                threat_intelligence=threat_intel,
                current_attacks=current_attacks,
                organizational_profile=org_profile,
            )
            return {
                "detail": "AI-powered deception strategy generated",
                "analysis": {"strategy": strategy},
            }

        elif task in ["deploy_honeypot", "deploy_lure"]:
            if hasattr(agent, "deploy_adaptive_lure"):
                lure_type = context.get("lure_type", "ssh")
                ip = context.get("ip") or (incident.src_ip if incident else None)
                result = await agent.deploy_adaptive_lure(ip, lure_type, {})
                return {"detail": f"Deployed {lure_type} lure", "analysis": result}

        # Default to behavior analysis
        profiles = await agent.analyze_attacker_behavior(
            events=events, timeframe_hours=24
        )
        return {
            "detail": f"Deception task '{task}' completed",
            "analysis": {"attacker_profiles": profiles},
        }

    async def _execute_edr_task(
        self, agent, task: str, context: Dict[str, Any], incident: Optional[Incident]
    ) -> Dict[str, Any]:
        """Execute EDR Agent tasks"""

        hostname = context.get("hostname", "localhost")
        incident_id = incident.id if incident else None

        if task in ["kill_process", "terminate_process"]:
            process_name = context.get("process_name")
            pid = context.get("pid")
            result = await agent.execute_action(
                action_name="kill_process",
                params={"hostname": hostname, "process_name": process_name, "pid": pid},
                incident_id=incident_id,
            )
            return {
                "detail": f"Process termination: {result.get('message')}",
                "analysis": result,
            }

        elif task in ["quarantine_file", "isolate_file"]:
            file_path = context.get("file_path", "")
            result = await agent.execute_action(
                action_name="quarantine_file",
                params={"hostname": hostname, "file_path": file_path},
                incident_id=incident_id,
            )
            return {
                "detail": f"File quarantine: {result.get('message')}",
                "analysis": result,
            }

        elif task in ["isolate_host", "host_isolation"]:
            level = context.get("isolation_level", "strict")
            result = await agent.execute_action(
                action_name="isolate_host",
                params={"hostname": hostname, "level": level},
                incident_id=incident_id,
            )
            return {
                "detail": f"Host isolation: {result.get('message')}",
                "analysis": result,
            }

        elif task in ["collect_memory", "memory_dump"]:
            result = await agent.execute_action(
                action_name="collect_memory_dump",
                params={"hostname": hostname},
                incident_id=incident_id,
            )
            return {
                "detail": f"Memory collection: {result.get('message')}",
                "analysis": result,
            }

        return {
            "detail": f"EDR task '{task}' - specify action parameters",
            "analysis": {},
        }

    async def _execute_iam_task(
        self,
        agent,
        task: str,
        context: Dict[str, Any],
        events: List[Event],
        incident: Optional[Incident],
    ) -> Dict[str, Any]:
        """Execute IAM Agent tasks"""

        username = context.get("username")
        incident_id = incident.id if incident else None

        if task in ["disable_account", "disable_user"]:
            if username:
                result = await agent.execute_action(
                    action_name="disable_user_account",
                    params={"username": username},
                    incident_id=incident_id,
                )
                return {
                    "detail": f"Account disabled: {result.get('message')}",
                    "analysis": result,
                }

        elif task in ["reset_password", "force_password_reset"]:
            if username:
                result = await agent.execute_action(
                    action_name="reset_password",
                    params={"username": username},
                    incident_id=incident_id,
                )
                return {
                    "detail": f"Password reset: {result.get('message')}",
                    "analysis": result,
                }

        elif task in ["quarantine_user", "restrict_access"]:
            if username:
                result = await agent.execute_action(
                    action_name="quarantine_user",
                    params={
                        "username": username,
                        "restrictions": context.get("restrictions", ["logon_denied"]),
                    },
                    incident_id=incident_id,
                )
                return {
                    "detail": f"User quarantined: {result.get('message')}",
                    "analysis": result,
                }

        elif task in ["analyze_auth", "authentication_analysis"]:
            if events and hasattr(agent, "analyze_authentication_event"):
                analysis_results = []
                for event in events[:10]:  # Analyze up to 10 events
                    result = await agent.analyze_authentication_event(event)
                    if result:
                        analysis_results.append(result)
                return {
                    "detail": f"Analyzed {len(analysis_results)} authentication events",
                    "analysis": {
                        "events_analyzed": len(analysis_results),
                        "findings": analysis_results,
                    },
                }

        return {
            "detail": f"IAM task '{task}' - provide username context",
            "analysis": {},
        }

    async def _execute_dlp_task(
        self, agent, task: str, context: Dict[str, Any], incident: Optional[Incident]
    ) -> Dict[str, Any]:
        """Execute DLP Agent tasks"""

        incident_id = incident.id if incident else None

        if task in ["scan_file", "file_scan"]:
            file_path = context.get("file_path", "")
            result = await agent.execute_action(
                action_name="scan_file",
                params={"file_path": file_path},
                incident_id=incident_id,
            )
            return {"detail": f"File scan: {result.get('message')}", "analysis": result}

        elif task in ["block_upload", "prevent_upload"]:
            hostname = context.get("hostname", "localhost")
            process_name = context.get("process_name", "")
            destination = context.get("destination", "")
            result = await agent.execute_action(
                action_name="block_upload",
                params={
                    "hostname": hostname,
                    "process_name": process_name,
                    "destination": destination,
                },
                incident_id=incident_id,
            )
            return {
                "detail": f"Upload blocked: {result.get('message')}",
                "analysis": result,
            }

        elif task in ["quarantine_sensitive", "protect_sensitive"]:
            hostname = context.get("hostname", "localhost")
            file_path = context.get("file_path", "")
            result = await agent.execute_action(
                action_name="quarantine_sensitive_file",
                params={"hostname": hostname, "file_path": file_path},
                incident_id=incident_id,
            )
            return {
                "detail": f"Sensitive file quarantined: {result.get('message')}",
                "analysis": result,
            }

        return {"detail": f"DLP task '{task}' - specify file path", "analysis": {}}

    async def _execute_hunter_task(
        self, agent, task: str, incident: Optional[Incident], events: List[Event]
    ) -> Dict[str, Any]:
        """Execute Predictive Hunter Agent tasks"""

        incidents = [incident] if incident else []

        if task in ["hunt", "threat_hunt", "proactive_hunt"]:
            result = await agent.execute_predictive_hunt(
                incidents=incidents, events=events, time_window=timedelta(hours=24)
            )
            return {
                "detail": f"Hunt completed: {result.get('analysis_summary', {}).get('hypotheses_generated', 0)} hypotheses generated",
                "analysis": result,
            }

        elif task in ["predict", "threat_prediction"]:
            result = await agent.execute_predictive_hunt(
                incidents=incidents, events=events, time_window=timedelta(hours=24)
            )
            predictions = result.get("threat_predictions", [])
            return {
                "detail": f"Generated {len(predictions)} threat predictions",
                "analysis": {
                    "predictions": predictions,
                    "risk_assessment": result.get("risk_assessment"),
                },
            }

        elif task in ["behavioral_analysis", "anomaly_detection"]:
            result = await agent.execute_predictive_hunt(
                incidents=incidents, events=events, time_window=timedelta(hours=24)
            )
            return {
                "detail": f"Behavioral analysis: {result.get('analysis_summary', {}).get('anomalies_detected', 0)} anomalies detected",
                "analysis": result.get("behavioral_analysis", {}),
            }

        elif task in ["generate_hypotheses", "hypothesis_generation"]:
            result = await agent.execute_predictive_hunt(
                incidents=incidents, events=events, time_window=timedelta(hours=24)
            )
            hypotheses = result.get("hunting_hypotheses", [])
            return {
                "detail": f"Generated {len(hypotheses)} hunting hypotheses",
                "analysis": {"hypotheses": hypotheses},
            }

        # Default to full hunt
        result = await agent.execute_predictive_hunt(
            incidents=incidents, events=events, time_window=timedelta(hours=24)
        )
        return {
            "detail": f"Predictive hunt task '{task}' completed",
            "analysis": result,
        }

    async def _fallback_to_summary(
        self,
        role: AgentRole,
        snapshot: Dict[str, Any],
        task: str,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Fallback to summary builders when agent invocation fails"""

        builders = {
            AgentRole.ATTRIBUTION: self._build_attribution_summary,
            AgentRole.CONTAINMENT: self._build_containment_summary,
            AgentRole.FORENSICS: self._build_forensics_summary,
            AgentRole.DECEPTION: self._build_deception_summary,
            AgentRole.EDR: self._build_edr_summary,
            AgentRole.IAM: self._build_iam_summary,
            AgentRole.DLP: self._build_dlp_summary,
            AgentRole.PREDICTIVE_HUNTER: self._build_hunter_summary,
        }

        builder = builders.get(role, self._build_generic_summary)
        return builder(snapshot, task, context)

    def _build_edr_summary(
        self, snapshot: Dict[str, Any], task: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build EDR summary when agent invocation fails"""
        hostname = context.get("hostname", "unknown")
        return {
            "detail": f"EDR task '{task}' for {hostname} (fallback mode)",
            "response": f"EDR analysis requested for {hostname}",
            "analysis": {
                "task": task,
                "hostname": hostname,
                "context": context,
                "fallback": True,
            },
        }

    def _build_iam_summary(
        self, snapshot: Dict[str, Any], task: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build IAM summary when agent invocation fails"""
        username = context.get("username", "unknown")
        return {
            "detail": f"IAM task '{task}' for {username} (fallback mode)",
            "response": f"IAM analysis requested for {username}",
            "analysis": {
                "task": task,
                "username": username,
                "context": context,
                "fallback": True,
            },
        }

    def _build_dlp_summary(
        self, snapshot: Dict[str, Any], task: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build DLP summary when agent invocation fails"""
        file_path = context.get("file_path", "unknown")
        return {
            "detail": f"DLP task '{task}' for {file_path} (fallback mode)",
            "response": f"DLP analysis requested",
            "analysis": {
                "task": task,
                "file_path": file_path,
                "context": context,
                "fallback": True,
            },
        }

    def _build_hunter_summary(
        self, snapshot: Dict[str, Any], task: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build Predictive Hunter summary when agent invocation fails"""
        events = snapshot.get("events", [])
        return {
            "detail": f"Hunter task '{task}' on {len(events)} events (fallback mode)",
            "response": f"Threat hunting analysis requested",
            "analysis": {
                "task": task,
                "events_count": len(events),
                "context": context,
                "fallback": True,
            },
        }

    def _build_generic_summary(
        self, snapshot: Dict[str, Any], task: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generic fallback summary"""
        return {
            "detail": f"Task '{task}' processed (fallback mode)",
            "response": f"Analysis requested",
            "analysis": {"task": task, "context": context, "fallback": True},
        }

    async def orchestrate_incident_response(
        self,
        incident: Incident,
        recent_events: List[Event],
        db_session=None,
        workflow_type: str = "comprehensive",
        use_langchain: bool = True,
    ) -> Dict[str, Any]:
        """
        Orchestrate a comprehensive incident response using all available agents.

        When LangChain is available and use_langchain=True, uses the ReAct-style
        GPT-4 agent for intelligent reasoning. Falls back to rule-based workflow
        when LangChain is unavailable.

        Args:
            incident: The incident to respond to
            recent_events: Recent events related to the incident
            db_session: Database session for persistence
            workflow_type: Type of orchestration workflow
            use_langchain: Whether to try LangChain orchestrator first

        Returns:
            Comprehensive orchestration results
        """

        # Try LangChain orchestrator first if available and enabled
        if (
            use_langchain
            and LANGCHAIN_ORCHESTRATOR_AVAILABLE
            and langchain_orchestrator
        ):
            try:
                langchain_result = await self._orchestrate_with_langchain(
                    incident, recent_events, db_session
                )
                if langchain_result.get("success"):
                    return langchain_result
                # If LangChain fails, fall through to regular workflow
                self.logger.warning(
                    "LangChain orchestration failed, using standard workflow"
                )
            except Exception as e:
                self.logger.warning(
                    f"LangChain orchestration error: {e}, using standard workflow"
                )

        workflow_id = f"wf_{incident.id}_{int(datetime.utcnow().timestamp())}"

        try:
            # Create workflow context
            workflow = WorkflowContext(
                workflow_id=workflow_id,
                incident_id=incident.id,
                status=WorkflowStatus.RUNNING,
                start_time=datetime.utcnow(),
                end_time=None,
                agents_involved=[],
                current_step="initialization",
                results={},
                decisions=[],
                errors=[],
            )

            self.active_workflows[workflow_id] = workflow

            self.logger.info(
                f"Starting orchestrated response for incident {incident.id}"
            )

            # Execute workflow based on type
            if workflow_type == "comprehensive":
                results = await self._execute_comprehensive_workflow(
                    incident, recent_events, workflow, db_session
                )
            elif workflow_type == "rapid":
                results = await self._execute_rapid_workflow(
                    incident, recent_events, workflow, db_session
                )
            else:
                results = await self._execute_basic_workflow(
                    incident, recent_events, workflow, db_session
                )

            # Complete workflow
            workflow.status = WorkflowStatus.COMPLETED
            workflow.end_time = datetime.utcnow()
            workflow.results = results

            self.stats["workflows_completed"] += 1

            self.logger.info(
                f"Completed orchestrated response for incident {incident.id}"
            )

            return {
                "success": True,
                "workflow_id": workflow_id,
                "results": results,
                "execution_time": (
                    workflow.end_time - workflow.start_time
                ).total_seconds(),
                "agents_involved": workflow.agents_involved,
            }

        except Exception as e:
            self.logger.error(
                f"Orchestrated response failed for incident {incident.id}: {e}"
            )

            # Mark workflow as failed
            if workflow_id in self.active_workflows:
                self.active_workflows[workflow_id].status = WorkflowStatus.FAILED
                self.active_workflows[workflow_id].errors.append(str(e))
                self.active_workflows[workflow_id].end_time = datetime.utcnow()

            self.stats["workflows_failed"] += 1

            return {
                "success": False,
                "workflow_id": workflow_id,
                "error": str(e),
                "partial_results": getattr(workflow, "results", {})
                if "workflow" in locals()
                else {},
            }

    async def _orchestrate_with_langchain(
        self,
        incident: Incident,
        recent_events: List[Event],
        db_session=None,
    ) -> Dict[str, Any]:
        """
        Orchestrate incident response using LangChain ReAct agent.

        This provides intelligent reasoning and action selection using GPT-4.

        Args:
            incident: The incident to respond to
            recent_events: Recent events related to the incident
            db_session: Database session for persistence

        Returns:
            Orchestration results compatible with standard workflow output
        """
        if not LANGCHAIN_ORCHESTRATOR_AVAILABLE or not orchestrate_with_langchain:
            return {"success": False, "error": "LangChain not available"}

        try:
            # Convert events to dict format for LangChain
            events_data = [
                {
                    "event_type": e.eventid,
                    "message": e.message,
                    "timestamp": str(e.ts) if e.ts else None,
                    "dst_port": e.dst_port,
                    "src_port": getattr(e, "src_port", None),
                }
                for e in recent_events[:20]  # Limit to recent 20
            ]

            # Get threat info from incident
            threat_type = incident.threat_category or incident.reason or "Unknown"
            confidence = incident.ml_confidence or 0.5
            severity = incident.escalation_level or "medium"

            # Get ML analysis from triage note if available
            ml_analysis = None
            if incident.triage_note:
                if isinstance(incident.triage_note, dict):
                    ml_analysis = incident.triage_note
                else:
                    try:
                        ml_analysis = json.loads(incident.triage_note)
                    except:
                        pass

            # Extract features for ML-Agent bridge
            features = None
            try:
                from .ml_feature_extractor import ml_feature_extractor

                features = ml_feature_extractor.extract_features(
                    incident.src_ip, recent_events
                )
            except Exception as e:
                logger.debug(f"Feature extraction for LangChain failed: {e}")

            # Call LangChain orchestrator
            result = await orchestrate_with_langchain(
                src_ip=incident.src_ip,
                threat_type=threat_type,
                confidence=confidence,
                severity=severity,
                events=events_data,
                ml_analysis=ml_analysis,
                features=features,
                incident_id=incident.id,  # NEW: Pass incident_id for investigation tracking
            )

            # Convert OrchestrationResult to standard workflow output format
            return {
                "success": result.success,
                "workflow_id": f"langchain_{incident.id}_{int(datetime.utcnow().timestamp())}",
                "orchestration_method": "langchain_react",
                "results": {
                    "final_decision": {
                        "verdict": result.final_verdict,
                        "should_contain": result.final_verdict
                        in ["CONTAINED", "ESCALATE"],
                        "priority_level": severity,
                        "reasoning": result.reasoning,
                        "confidence": result.confidence,
                    },
                    "actions_taken": result.actions_taken,
                    "recommendations": result.recommendations,
                    "agent_trace": result.agent_trace,
                },
                "execution_time": result.processing_time_ms / 1000,
                "agents_involved": ["langchain_orchestrator"],
                "langchain_verdict": result.final_verdict,
                "langchain_reasoning": result.reasoning,
            }

        except Exception as e:
            self.logger.error(f"LangChain orchestration failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "orchestration_method": "langchain_react",
            }

    async def enhanced_orchestrate_incident_response(
        self,
        incident: Incident,
        recent_events: List[Event],
        db_session=None,
        coordination_strategy: str = "adaptive",
        max_agents: int = 4,
        automation_level: str = "high",
    ) -> Dict[str, Any]:
        """
        Enhanced incident response using advanced multi-agent coordination

        Args:
            incident: The incident to respond to
            recent_events: Recent events related to the incident
            db_session: Database session for persistence
            coordination_strategy: Strategy for coordination ("adaptive", "parallel", "sequential", "hierarchical")
            max_agents: Maximum number of agents to coordinate
            automation_level: Level of automation ("low", "medium", "high")

        Returns:
            Enhanced orchestration results with advanced coordination analytics
        """

        orchestration_start = datetime.utcnow()
        orchestration_id = (
            f"enhanced_orch_{incident.id}_{int(orchestration_start.timestamp())}"
        )

        try:
            self.logger.info(
                f"Starting enhanced orchestration {orchestration_id} for incident {incident.id}"
            )

            # Create coordination context
            context = CoordinationContext(
                incident_severity=getattr(incident, "escalation_level", "medium"),
                time_constraints=timedelta(minutes=30)
                if incident.status == "new"
                else None,
                resource_availability={"cpu": 0.8, "memory": 0.7, "network": 0.9},
                stakeholder_priority="high"
                if getattr(incident, "escalation_level", "medium") == "critical"
                else "medium",
                automation_level=automation_level,
                risk_tolerance=0.3
                if getattr(incident, "escalation_level", "medium") == "critical"
                else 0.6,
                compliance_requirements=["data_protection", "incident_logging"],
            )

            # Determine required capabilities based on incident type
            required_capabilities = self._determine_required_capabilities(
                incident, recent_events
            )

            # Use coordination hub for advanced agent coordination
            coordination_result = await self.coordination_hub.coordinate_agents(
                incident=incident,
                required_capabilities=required_capabilities,
                context=context,
                max_agents=max_agents,
            )

            if coordination_result["success"]:
                # Process coordination results
                orchestration_results = {
                    "success": True,
                    "orchestration_id": orchestration_id,
                    "coordination_strategy": coordination_result["strategy"],
                    "agents_coordinated": coordination_result["agents_involved"],
                    "execution_time": coordination_result["execution_time"],
                    "coordination_results": coordination_result["results"],
                    "performance_summary": coordination_result["performance_summary"],
                    "decision_analytics": self._generate_decision_analytics(
                        coordination_result
                    ),
                    "agent_performance_metrics": self._extract_agent_metrics(
                        coordination_result
                    ),
                    "recommendations": self._generate_orchestration_recommendations(
                        coordination_result, context
                    ),
                }

                # Update orchestrator statistics
                self.stats["coordinations_executed"] += 1
                if coordination_result.get("conflicts_resolved", 0) > 0:
                    self.stats["conflicts_resolved"] += coordination_result[
                        "conflicts_resolved"
                    ]

                self.logger.info(
                    f"Enhanced orchestration {orchestration_id} completed successfully"
                )

                return orchestration_results

            else:
                # Fallback to legacy orchestration
                self.logger.warning(
                    f"Coordination failed, falling back to legacy orchestration: {coordination_result.get('error')}"
                )

                fallback_result = await self.orchestrate_incident_response(
                    incident, recent_events, db_session, "comprehensive"
                )

                # Add coordination attempt info
                fallback_result["coordination_attempted"] = True
                fallback_result["coordination_error"] = coordination_result.get("error")
                fallback_result["fallback_used"] = True

                return fallback_result

        except Exception as e:
            self.logger.error(f"Enhanced orchestration {orchestration_id} failed: {e}")

            # Fallback to legacy orchestration
            try:
                fallback_result = await self.orchestrate_incident_response(
                    incident, recent_events, db_session, "basic"
                )
                fallback_result["coordination_attempted"] = True
                fallback_result["coordination_error"] = str(e)
                fallback_result["fallback_used"] = True
                return fallback_result

            except Exception as fallback_error:
                return {
                    "success": False,
                    "orchestration_id": orchestration_id,
                    "error": f"Both enhanced and fallback orchestration failed: {e}, {fallback_error}",
                    "coordination_attempted": True,
                }

    def _determine_required_capabilities(
        self, incident: Incident, recent_events: List[Event]
    ) -> List[str]:
        """Determine what agent capabilities are required for this incident"""

        capabilities = []

        # Always include containment for active incidents
        if incident.status in ["new", "open"]:
            capabilities.append("incident_containment")

        # Include attribution for significant incidents
        if len(recent_events) > 10 or getattr(
            incident, "escalation_level", "medium"
        ) in ["high", "critical"]:
            capabilities.append("threat_attribution")

        # Include forensics for complex incidents
        if len(recent_events) > 20 or incident.reason in [
            "malware",
            "data_exfiltration",
        ]:
            capabilities.append("forensic_analysis")

        # Include deception for ongoing attacks
        if incident.status == "open" and len(recent_events) > 5:
            capabilities.append("deception_deployment")

        # Default to basic capabilities if none determined
        if not capabilities:
            capabilities = ["incident_containment", "threat_attribution"]

        return capabilities

    def _generate_decision_analytics(
        self, coordination_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate analytics about the decision-making process"""

        analytics = {
            "coordination_efficiency": 0.0,
            "decision_quality_score": 0.0,
            "agent_collaboration_score": 0.0,
            "conflict_resolution_effectiveness": 0.0,
            "resource_utilization": 0.0,
        }

        try:
            # Coordination efficiency (based on execution time vs estimated)
            actual_time = coordination_result.get("execution_time", 0)
            if actual_time > 0:
                # Assume baseline of 60 seconds for comparison
                efficiency = max(0, min(1, (60 - actual_time) / 60))
                analytics["coordination_efficiency"] = efficiency

            # Decision quality (based on agent confidence and performance)
            performance_summary = coordination_result.get("performance_summary", {})
            successful_agents = performance_summary.get("successful_agents", 0)
            total_agents = performance_summary.get("agents_participated", 1)

            analytics["decision_quality_score"] = successful_agents / total_agents

            # Agent collaboration (based on conflicts and resolution)
            conflicts_detected = performance_summary.get("conflicts_detected", False)
            if conflicts_detected:
                # Lower score for conflicts, but credit for resolution
                analytics["agent_collaboration_score"] = 0.6
                analytics["conflict_resolution_effectiveness"] = 0.8
            else:
                analytics["agent_collaboration_score"] = 0.9
                analytics["conflict_resolution_effectiveness"] = 1.0

            # Resource utilization (mock calculation)
            analytics["resource_utilization"] = min(0.8, total_agents / 5.0)

        except Exception as e:
            self.logger.error(f"Failed to generate decision analytics: {e}")

        return analytics

    def _extract_agent_metrics(
        self, coordination_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract performance metrics for individual agents"""

        metrics = {}

        try:
            results = coordination_result.get("results", {})
            agent_results = results.get("agent_results", {})

            for agent_id, result in agent_results.items():
                metrics[agent_id] = {
                    "success": result.get("success", False),
                    "execution_time": result.get("execution_time", 0.0),
                    "confidence_score": getattr(
                        result.get("decision"), "confidence", 0.0
                    )
                    if result.get("decision")
                    else 0.0,
                    "findings_count": len(result.get("findings", [])),
                    "error_occurred": "error" in result,
                }

        except Exception as e:
            self.logger.error(f"Failed to extract agent metrics: {e}")

        return metrics

    def _generate_orchestration_recommendations(
        self, coordination_result: Dict[str, Any], context: CoordinationContext
    ) -> List[str]:
        """Generate recommendations based on orchestration results"""

        recommendations = []

        try:
            performance_summary = coordination_result.get("performance_summary", {})

            # Performance-based recommendations
            success_rate = performance_summary.get("successful_agents", 0) / max(
                performance_summary.get("agents_participated", 1), 1
            )

            if success_rate < 0.8:
                recommendations.append(
                    "Consider reviewing agent configurations - lower than expected success rate"
                )

            # Time-based recommendations
            execution_time = coordination_result.get("execution_time", 0)
            if execution_time > 45:
                recommendations.append(
                    "Consider parallel coordination strategy for faster response times"
                )

            # Conflict-based recommendations
            if performance_summary.get("conflicts_detected", False):
                recommendations.append(
                    "Review agent decision criteria to reduce conflicts"
                )

            # Context-based recommendations
            if context.incident_severity == "critical" and execution_time > 30:
                recommendations.append(
                    "Implement fast-track coordination for critical incidents"
                )

            # Resource optimization recommendations
            agents_used = performance_summary.get("agents_participated", 0)
            if agents_used < 2:
                recommendations.append(
                    "Consider involving additional agents for more comprehensive analysis"
                )
            elif agents_used > 4:
                recommendations.append(
                    "Evaluate if all agents are necessary - consider optimizing agent selection"
                )

            # Generic recommendations
            if not recommendations:
                recommendations.append(
                    "Orchestration completed successfully - no immediate optimizations needed"
                )

        except Exception as e:
            self.logger.error(f"Failed to generate orchestration recommendations: {e}")
            recommendations.append(
                "Unable to generate recommendations due to analysis error"
            )

        return recommendations

    async def _execute_comprehensive_workflow(
        self,
        incident: Incident,
        recent_events: List[Event],
        workflow: WorkflowContext,
        db_session=None,
    ) -> Dict[str, Any]:
        """Execute comprehensive multi-agent workflow"""

        results = {
            "attribution": {},
            "forensics": {},
            "containment": {},
            "deception": {},
            "coordination": {},
            "final_decision": {},
        }

        # Step 1: Attribution Analysis
        workflow.current_step = "attribution_analysis"
        try:
            attr_agent = self.agents[AgentRole.ATTRIBUTION]
            attr_results = await attr_agent.analyze_attribution(
                incidents=[incident], events=recent_events, db_session=db_session
            )

            results["attribution"] = attr_results
            workflow.agents_involved.append("attribution")
            workflow.decisions.append(
                {
                    "agent": "attribution",
                    "decision_type": "attribution_analysis",
                    "confidence": attr_results.get("confidence_score", 0.0),
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )

        except Exception as e:
            self.logger.error(f"Attribution analysis failed: {e}")
            workflow.errors.append(f"Attribution: {e}")

        # Step 2: Forensic Investigation
        workflow.current_step = "forensic_investigation"
        try:
            forensic_agent = self.agents[AgentRole.FORENSICS]
            case_id = await forensic_agent.initiate_forensic_case(
                incident=incident,
                investigator="orchestrator",
                evidence_types=["event_logs", "network_artifacts"],
            )

            # Collect evidence
            await forensic_agent.collect_evidence(
                case_id=case_id,
                incident=incident,
                evidence_types=["event_logs", "network_artifacts"],
                db_session=db_session,
            )

            # Analyze evidence
            analysis_results = await forensic_agent.analyze_evidence(
                case_id=case_id, evidence_ids=None
            )

            results["forensics"] = {"case_id": case_id, "analysis": analysis_results}
            workflow.agents_involved.append("forensics")

        except Exception as e:
            self.logger.error(f"Forensic investigation failed: {e}")
            workflow.errors.append(f"Forensics: {e}")

        # Step 3: Threat Intelligence Enhancement
        workflow.current_step = "threat_intelligence"
        try:
            attr_agent = self.agents[AgentRole.ATTRIBUTION]
            intel_results = await attr_agent.analyze_ip_reputation(incident.src_ip)
            results["threat_intelligence"] = intel_results

        except Exception as e:
            self.logger.error(f"Threat intelligence lookup failed: {e}")
            workflow.errors.append(f"Threat Intel: {e}")

        # Step 4: Containment Decision
        workflow.current_step = "containment_decision"
        try:
            containment_agent = self.agents[AgentRole.CONTAINMENT]

            # Create containment decision based on previous analysis
            decision_input = {
                "incident": incident,
                "attribution": results.get("attribution", {}),
                "forensics": results.get("forensics", {}),
                "threat_intel": results.get("threat_intelligence", {}),
                "recent_events": recent_events,
            }

            containment_results = await containment_agent.orchestrate_response(
                incident=incident, recent_events=recent_events, db_session=db_session
            )

            results["containment"] = containment_results
            workflow.agents_involved.append("containment")
            workflow.decisions.append(
                {
                    "agent": "containment",
                    "decision_type": "containment_actions",
                    "confidence": containment_results.get("confidence", 0.0),
                    "actions": containment_results.get("actions", []),
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )

        except Exception as e:
            self.logger.error(f"Containment decision failed: {e}")
            workflow.errors.append(f"Containment: {e}")

        # Step 5: Deception Strategy (if needed)
        workflow.current_step = "deception_strategy"
        try:
            # Only run deception if we have high-confidence malicious activity
            risk_score = results.get("attribution", {}).get("confidence_score", 0)
            if risk_score > 0.7:
                deception_agent = self.agents[AgentRole.DECEPTION]

                # Analyze attacker behavior
                attacker_profiles = await deception_agent.analyze_attacker_behavior(
                    events=recent_events, timeframe_hours=24
                )

                # Generate deception strategy
                strategy = await deception_agent.ai_powered_deception_strategy(
                    threat_intelligence=results.get("threat_intelligence", {}),
                    current_attacks=["brute_force", "reconnaissance"],
                    organizational_profile={
                        "industry": "technology",
                        "size": "enterprise",
                    },
                )

                results["deception"] = {
                    "attacker_profiles": attacker_profiles,
                    "strategy": strategy,
                }
                workflow.agents_involved.append("deception")

        except Exception as e:
            self.logger.error(f"Deception strategy failed: {e}")
            workflow.errors.append(f"Deception: {e}")

        # Step 6: Final Coordination and Decision Fusion
        workflow.current_step = "coordination"
        try:
            coordination_results = await self._coordinate_final_decision(
                results, incident, workflow
            )
            results["coordination"] = coordination_results
            results["final_decision"] = coordination_results.get("final_decision", {})

        except Exception as e:
            self.logger.error(f"Final coordination failed: {e}")
            workflow.errors.append(f"Coordination: {e}")

        return results

    async def _execute_rapid_workflow(
        self,
        incident: Incident,
        recent_events: List[Event],
        workflow: WorkflowContext,
        db_session=None,
    ) -> Dict[str, Any]:
        """Execute rapid response workflow for high-priority incidents"""

        # Focus on containment and forensics, skip attribution
        results = {"containment": {}, "forensics": {}, "rapid_response": True}

        # Parallel execution of critical agents
        tasks = []

        # Containment task
        containment_task = asyncio.create_task(
            self.agents[AgentRole.CONTAINMENT].orchestrate_response(
                incident, recent_events, db_session
            )
        )
        tasks.append(("containment", containment_task))

        # Forensic task
        forensic_task = asyncio.create_task(
            self._rapid_forensic_analysis(incident, recent_events, db_session)
        )
        tasks.append(("forensics", forensic_task))

        # Execute parallel tasks
        for agent_name, task in tasks:
            try:
                result = await task
                results[agent_name] = result
                workflow.agents_involved.append(agent_name)
            except Exception as e:
                self.logger.error(f"Rapid workflow {agent_name} failed: {e}")
                workflow.errors.append(f"{agent_name}: {e}")

        return results

    async def _execute_basic_workflow(
        self,
        incident: Incident,
        recent_events: List[Event],
        workflow: WorkflowContext,
        db_session=None,
    ) -> Dict[str, Any]:
        """Execute basic workflow for low-priority incidents"""

        # Simple containment-focused workflow
        containment_agent = self.agents[AgentRole.CONTAINMENT]
        results = await containment_agent.orchestrate_response(
            incident, recent_events, db_session
        )

        workflow.agents_involved.append("containment")

        return {"containment": results, "basic_workflow": True}

    async def _rapid_forensic_analysis(
        self, incident: Incident, recent_events: List[Event], db_session=None
    ) -> Dict[str, Any]:
        """Rapid forensic analysis for high-priority incidents"""

        forensic_agent = self.agents[AgentRole.FORENSICS]

        # Quick evidence collection
        case_id = await forensic_agent.initiate_forensic_case(
            incident=incident,
            investigator="orchestrator_rapid",
            evidence_types=["event_logs"],
        )

        # Rapid analysis
        analysis = await forensic_agent.analyze_evidence(
            case_id=case_id, evidence_ids=None
        )

        return {
            "case_id": case_id,
            "rapid_analysis": analysis,
            "evidence_collected": ["event_logs"],
        }

    async def _coordinate_final_decision(
        self,
        agent_results: Dict[str, Any],
        incident: Incident,
        workflow: WorkflowContext,
    ) -> Dict[str, Any]:
        """Coordinate final decision from all agent inputs"""

        coordination = {
            "decision_factors": {},
            "confidence_levels": {},
            "recommended_actions": [],
            "risk_assessment": {},
            "final_decision": {},
        }

        # Extract key decision factors
        attribution_confidence = agent_results.get("attribution", {}).get(
            "confidence_score", 0
        )
        containment_confidence = agent_results.get("containment", {}).get(
            "confidence", 0
        )
        forensic_risk = (
            agent_results.get("forensics", {})
            .get("analysis", {})
            .get("risk_assessment", {})
            .get("overall_risk_score", 0)
        )

        coordination["decision_factors"] = {
            "attribution_confidence": attribution_confidence,
            "containment_confidence": containment_confidence,
            "forensic_risk_score": forensic_risk,
            "threat_intel_risk": agent_results.get("threat_intelligence", {}).get(
                "reputation_score", 0
            ),
        }

        # Calculate overall confidence
        weights = {"attribution": 0.3, "containment": 0.3, "forensic": 0.4}
        overall_confidence = (
            attribution_confidence * weights["attribution"]
            + containment_confidence * weights["containment"]
            + (1 - forensic_risk) * weights["forensic"]  # Invert forensic risk
        )

        coordination["confidence_levels"] = {
            "overall": overall_confidence,
            "attribution": attribution_confidence,
            "containment": containment_confidence,
            "forensic": 1 - forensic_risk,
        }

        # Determine final actions
        actions = []

        # Always include containment actions if any were recommended
        containment_actions = agent_results.get("containment", {}).get("actions", [])
        if containment_actions:
            actions.extend(containment_actions)

        # Add forensic recommendations
        forensic_recs = (
            agent_results.get("forensics", {})
            .get("analysis", {})
            .get("recommendations", [])
        )
        if forensic_recs:
            actions.extend(
                [
                    {"action": "forensic_recommendation", "details": rec}
                    for rec in forensic_recs
                ]
            )

        # Add deception actions if high risk
        if overall_confidence > 0.7:
            deception_actions = (
                agent_results.get("deception", {})
                .get("strategy", {})
                .get("adaptive_responses", [])
            )
            if deception_actions:
                actions.extend(deception_actions)

        coordination["recommended_actions"] = actions

        # Final risk assessment
        if overall_confidence > 0.8:
            risk_level = "critical"
        elif overall_confidence > 0.6:
            risk_level = "high"
        elif overall_confidence > 0.4:
            risk_level = "medium"
        else:
            risk_level = "low"

        coordination["risk_assessment"] = {
            "level": risk_level,
            "score": overall_confidence,
            "escalation_recommended": risk_level in ["high", "critical"],
        }

        # Final decision
        coordination["final_decision"] = {
            "should_contain": overall_confidence > 0.5,
            "should_investigate": overall_confidence > 0.3,
            "should_escalate": risk_level in ["high", "critical"],
            "priority_level": "high"
            if risk_level in ["high", "critical"]
            else "medium",
            "automated_response": overall_confidence > 0.6,
            "human_review_required": overall_confidence < 0.8,
        }

        return coordination

    async def send_agent_message(
        self,
        recipient: AgentRole,
        message_type: MessageType,
        payload: Dict[str, Any],
        priority: int = 1,
    ) -> str:
        """Send a message to another agent"""

        message = AgentMessage(
            message_id=str(uuid.uuid4()),
            sender=self.agent_id,
            recipient=recipient.value,
            message_type=message_type,
            payload=payload,
            timestamp=datetime.utcnow(),
            correlation_id=str(uuid.uuid4()),
            priority=priority,
        )

        # Add to message queue
        self.message_queue.append(message)

        # Process message immediately for critical communications
        if priority >= 4:
            await self._process_message(message)

        self.logger.info(f"Sent message {message.message_id} to {recipient.value}")

        return message.message_id

    async def _process_message(self, message: AgentMessage):
        """Process an agent message"""

        try:
            # Route message to appropriate agent
            if message.recipient in [role.value for role in AgentRole]:
                recipient_role = AgentRole(message.recipient)
                agent = self.agents.get(recipient_role)

                if agent:
                    # Handle message based on type
                    if message.message_type == MessageType.REQUEST:
                        response = await self._handle_agent_request(agent, message)
                        if response:
                            # Send response back
                            await self.send_agent_message(
                                recipient=AgentRole(message.sender),
                                message_type=MessageType.RESPONSE,
                                payload=response,
                                priority=message.priority,
                            )

                    elif message.message_type == MessageType.NOTIFICATION:
                        await self._handle_agent_notification(agent, message)

                    self.stats["messages_processed"] += 1

        except Exception as e:
            self.logger.error(f"Failed to process message {message.message_id}: {e}")

    async def _handle_agent_request(
        self, agent, message: AgentMessage
    ) -> Optional[Dict[str, Any]]:
        """Handle a request message to an agent"""

        # This would contain logic to translate messages into agent method calls
        # For now, return a basic acknowledgment
        return {
            "message_id": message.message_id,
            "status": "received",
            "timestamp": datetime.utcnow().isoformat(),
        }

    async def _handle_agent_notification(self, agent, message: AgentMessage):
        """Handle a notification message to an agent"""

        # Store notification in shared memory for agent to retrieve
        key = f"notification_{message.message_id}"
        await self.shared_memory.store(key, message.payload, ttl_seconds=3600)

    async def get_orchestrator_status(self) -> Dict[str, Any]:
        """Get comprehensive status of the orchestrator"""

        status = {
            "orchestrator_id": self.agent_id,
            "uptime_seconds": (
                datetime.utcnow() - self.stats["start_time"]
            ).total_seconds(),
            "agents": {},
            "active_workflows": len(self.active_workflows),
            "message_queue_length": len(self.message_queue),
            "statistics": self.stats.copy(),
        }

        # Agent status
        for role, agent in self.agents.items():
            agent_status = {
                "status": "active",
                "agent_id": getattr(agent, "agent_id", "unknown"),
                "last_activity": getattr(agent, "last_activity", None),
            }

            # Check if agent is responsive
            try:
                if hasattr(agent, "agent_id"):
                    agent_status["responsive"] = True
                else:
                    agent_status["responsive"] = False
                    agent_status["status"] = "error"
            except:
                agent_status["responsive"] = False
                agent_status["status"] = "error"

            status["agents"][role.value] = agent_status

        return status

    async def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel an active workflow"""

        if workflow_id in self.active_workflows:
            workflow = self.active_workflows[workflow_id]
            if workflow.status == WorkflowStatus.RUNNING:
                workflow.status = WorkflowStatus.CANCELLED
                workflow.end_time = datetime.utcnow()
                self.logger.info(f"Cancelled workflow {workflow_id}")
                return True

        return False

    async def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific workflow"""

        workflow = self.active_workflows.get(workflow_id)
        if workflow:
            return {
                "workflow_id": workflow.workflow_id,
                "incident_id": workflow.incident_id,
                "status": workflow.status.value,
                "start_time": workflow.start_time.isoformat(),
                "end_time": workflow.end_time.isoformat()
                if workflow.end_time
                else None,
                "current_step": workflow.current_step,
                "agents_involved": workflow.agents_involved,
                "errors": workflow.errors,
                "execution_time": (
                    (workflow.end_time or datetime.utcnow()) - workflow.start_time
                ).total_seconds(),
            }

        return None


# Global orchestrator instance
orchestrator = AgentOrchestrator()


async def get_orchestrator() -> AgentOrchestrator:
    """Get the global orchestrator instance"""
    if not hasattr(orchestrator, "_initialized"):
        await orchestrator.initialize()
        orchestrator._initialized = True
    return orchestrator

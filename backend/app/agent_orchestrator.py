"""
Agent Orchestration Framework for Mini-XDR
Coordinates communication and decision fusion between AI agents
"""
import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

from .agents.attribution_agent import AttributionAgent
from .agents.containment_agent import ContainmentAgent
from .agents.forensics_agent import ForensicsAgent
from .agents.deception_agent import DeceptionAgent
from .agents.coordination_hub import (
    AdvancedCoordinationHub, AgentCapability, CoordinationContext,
    CoordinationStrategy, ConflictResolutionStrategy
)
from .models import Incident, Event, Action
from .config import settings


logger = logging.getLogger(__name__)


class AgentRole(Enum):
    """Roles that agents can play in the orchestration"""
    ATTRIBUTION = "attribution"
    CONTAINMENT = "containment"
    FORENSICS = "forensics"
    DECEPTION = "deception"
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
        expired_keys = [
            key for key, expiry in self.ttl_cache.items()
            if now > expiry
        ]

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

        # Initialize agents
        self.agents = {
            AgentRole.ATTRIBUTION: AttributionAgent(),
            AgentRole.CONTAINMENT: ContainmentAgent(),
            AgentRole.FORENSICS: ForensicsAgent(),
            AgentRole.DECEPTION: DeceptionAgent()
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
            "start_time": datetime.utcnow()
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
                if hasattr(agent, 'agent_id'):
                    self.logger.info(f"Agent {role.value} ({agent.agent_id}) is responsive")
                else:
                    self.logger.warning(f"Agent {role.value} may not be properly initialized")
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
                    wf_id for wf_id, wf in self.active_workflows.items()
                    if wf.status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED]
                    and wf.end_time and wf.end_time < cutoff_time
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
                domain_expertise=["threat_attribution", "campaign_analysis", "actor_profiling"],
                confidence_threshold=0.7,
                execution_time_estimate=15.0,
                resource_requirements={"cpu": 0.2, "memory": 0.3},
                dependencies=[],
                success_rate=0.85
            ),
            AgentRole.CONTAINMENT: AgentCapability(
                name="containment_agent", 
                domain_expertise=["incident_containment", "ip_blocking", "threat_response"],
                confidence_threshold=0.8,
                execution_time_estimate=5.0,
                resource_requirements={"cpu": 0.1, "memory": 0.1},
                dependencies=[],
                success_rate=0.92
            ),
            AgentRole.FORENSICS: AgentCapability(
                name="forensics_agent",
                domain_expertise=["evidence_collection", "forensic_analysis", "case_management"],
                confidence_threshold=0.75,
                execution_time_estimate=30.0,
                resource_requirements={"cpu": 0.3, "memory": 0.4},
                dependencies=[],
                success_rate=0.88
            ),
            AgentRole.DECEPTION: AgentCapability(
                name="deception_agent",
                domain_expertise=["deception_deployment", "attacker_profiling", "honeypot_management"],
                confidence_threshold=0.6,
                execution_time_estimate=20.0,
                resource_requirements={"cpu": 0.2, "memory": 0.2},
                dependencies=[],
                success_rate=0.78
            )
        }
        
        # Register each agent
        for role, capabilities in agent_capabilities.items():
            self.coordination_hub.register_agent(role.value, capabilities)
            self.logger.info(f"Registered {role.value} with coordination hub")

    async def orchestrate_incident_response(
        self,
        incident: Incident,
        recent_events: List[Event],
        db_session=None,
        workflow_type: str = "comprehensive"
    ) -> Dict[str, Any]:
        """
        Orchestrate a comprehensive incident response using all available agents

        Args:
            incident: The incident to respond to
            recent_events: Recent events related to the incident
            db_session: Database session for persistence
            workflow_type: Type of orchestration workflow

        Returns:
            Comprehensive orchestration results
        """

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
                errors=[]
            )

            self.active_workflows[workflow_id] = workflow

            self.logger.info(f"Starting orchestrated response for incident {incident.id}")

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

            self.logger.info(f"Completed orchestrated response for incident {incident.id}")

            return {
                "success": True,
                "workflow_id": workflow_id,
                "results": results,
                "execution_time": (workflow.end_time - workflow.start_time).total_seconds(),
                "agents_involved": workflow.agents_involved
            }

        except Exception as e:
            self.logger.error(f"Orchestrated response failed for incident {incident.id}: {e}")

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
                "partial_results": getattr(workflow, 'results', {}) if 'workflow' in locals() else {}
            }

    async def enhanced_orchestrate_incident_response(
        self,
        incident: Incident,
        recent_events: List[Event],
        db_session=None,
        coordination_strategy: str = "adaptive",
        max_agents: int = 4,
        automation_level: str = "high"
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
        orchestration_id = f"enhanced_orch_{incident.id}_{int(orchestration_start.timestamp())}"
        
        try:
            self.logger.info(f"Starting enhanced orchestration {orchestration_id} for incident {incident.id}")
            
            # Create coordination context
            context = CoordinationContext(
                incident_severity=getattr(incident, 'escalation_level', 'medium'),
                time_constraints=timedelta(minutes=30) if incident.status == 'new' else None,
                resource_availability={"cpu": 0.8, "memory": 0.7, "network": 0.9},
                stakeholder_priority='high' if getattr(incident, 'escalation_level', 'medium') == 'critical' else 'medium',
                automation_level=automation_level,
                risk_tolerance=0.3 if getattr(incident, 'escalation_level', 'medium') == 'critical' else 0.6,
                compliance_requirements=["data_protection", "incident_logging"]
            )
            
            # Determine required capabilities based on incident type
            required_capabilities = self._determine_required_capabilities(incident, recent_events)
            
            # Use coordination hub for advanced agent coordination
            coordination_result = await self.coordination_hub.coordinate_agents(
                incident=incident,
                required_capabilities=required_capabilities,
                context=context,
                max_agents=max_agents
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
                    "decision_analytics": self._generate_decision_analytics(coordination_result),
                    "agent_performance_metrics": self._extract_agent_metrics(coordination_result),
                    "recommendations": self._generate_orchestration_recommendations(coordination_result, context)
                }
                
                # Update orchestrator statistics
                self.stats["coordinations_executed"] += 1
                if coordination_result.get("conflicts_resolved", 0) > 0:
                    self.stats["conflicts_resolved"] += coordination_result["conflicts_resolved"]
                
                self.logger.info(f"Enhanced orchestration {orchestration_id} completed successfully")
                
                return orchestration_results
                
            else:
                # Fallback to legacy orchestration
                self.logger.warning(f"Coordination failed, falling back to legacy orchestration: {coordination_result.get('error')}")
                
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
                    "coordination_attempted": True
                }
    
    def _determine_required_capabilities(self, incident: Incident, recent_events: List[Event]) -> List[str]:
        """Determine what agent capabilities are required for this incident"""
        
        capabilities = []
        
        # Always include containment for active incidents
        if incident.status in ['new', 'open']:
            capabilities.append("incident_containment")
        
        # Include attribution for significant incidents
        if len(recent_events) > 10 or getattr(incident, 'escalation_level', 'medium') in ['high', 'critical']:
            capabilities.append("threat_attribution")
        
        # Include forensics for complex incidents
        if len(recent_events) > 20 or incident.reason in ['malware', 'data_exfiltration']:
            capabilities.append("forensic_analysis")
        
        # Include deception for ongoing attacks
        if incident.status == 'open' and len(recent_events) > 5:
            capabilities.append("deception_deployment")
        
        # Default to basic capabilities if none determined
        if not capabilities:
            capabilities = ["incident_containment", "threat_attribution"]
        
        return capabilities
    
    def _generate_decision_analytics(self, coordination_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate analytics about the decision-making process"""
        
        analytics = {
            "coordination_efficiency": 0.0,
            "decision_quality_score": 0.0,
            "agent_collaboration_score": 0.0,
            "conflict_resolution_effectiveness": 0.0,
            "resource_utilization": 0.0
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
    
    def _extract_agent_metrics(self, coordination_result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract performance metrics for individual agents"""
        
        metrics = {}
        
        try:
            results = coordination_result.get("results", {})
            agent_results = results.get("agent_results", {})
            
            for agent_id, result in agent_results.items():
                metrics[agent_id] = {
                    "success": result.get("success", False),
                    "execution_time": result.get("execution_time", 0.0),
                    "confidence_score": getattr(result.get("decision"), "confidence", 0.0) if result.get("decision") else 0.0,
                    "findings_count": len(result.get("findings", [])),
                    "error_occurred": "error" in result
                }
        
        except Exception as e:
            self.logger.error(f"Failed to extract agent metrics: {e}")
        
        return metrics
    
    def _generate_orchestration_recommendations(
        self, 
        coordination_result: Dict[str, Any], 
        context: CoordinationContext
    ) -> List[str]:
        """Generate recommendations based on orchestration results"""
        
        recommendations = []
        
        try:
            performance_summary = coordination_result.get("performance_summary", {})
            
            # Performance-based recommendations
            success_rate = performance_summary.get("successful_agents", 0) / max(performance_summary.get("agents_participated", 1), 1)
            
            if success_rate < 0.8:
                recommendations.append("Consider reviewing agent configurations - lower than expected success rate")
            
            # Time-based recommendations
            execution_time = coordination_result.get("execution_time", 0)
            if execution_time > 45:
                recommendations.append("Consider parallel coordination strategy for faster response times")
            
            # Conflict-based recommendations
            if performance_summary.get("conflicts_detected", False):
                recommendations.append("Review agent decision criteria to reduce conflicts")
            
            # Context-based recommendations
            if context.incident_severity == 'critical' and execution_time > 30:
                recommendations.append("Implement fast-track coordination for critical incidents")
            
            # Resource optimization recommendations
            agents_used = performance_summary.get("agents_participated", 0)
            if agents_used < 2:
                recommendations.append("Consider involving additional agents for more comprehensive analysis")
            elif agents_used > 4:
                recommendations.append("Evaluate if all agents are necessary - consider optimizing agent selection")
            
            # Generic recommendations
            if not recommendations:
                recommendations.append("Orchestration completed successfully - no immediate optimizations needed")
            
        except Exception as e:
            self.logger.error(f"Failed to generate orchestration recommendations: {e}")
            recommendations.append("Unable to generate recommendations due to analysis error")
        
        return recommendations

    async def _execute_comprehensive_workflow(
        self,
        incident: Incident,
        recent_events: List[Event],
        workflow: WorkflowContext,
        db_session=None
    ) -> Dict[str, Any]:
        """Execute comprehensive multi-agent workflow"""

        results = {
            "attribution": {},
            "forensics": {},
            "containment": {},
            "deception": {},
            "coordination": {},
            "final_decision": {}
        }

        # Step 1: Attribution Analysis
        workflow.current_step = "attribution_analysis"
        try:
            attr_agent = self.agents[AgentRole.ATTRIBUTION]
            attr_results = await attr_agent.analyze_attribution(
                incidents=[incident],
                events=recent_events,
                db_session=db_session
            )

            results["attribution"] = attr_results
            workflow.agents_involved.append("attribution")
            workflow.decisions.append({
                "agent": "attribution",
                "decision_type": "attribution_analysis",
                "confidence": attr_results.get("confidence_score", 0.0),
                "timestamp": datetime.utcnow().isoformat()
            })

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
                evidence_types=["event_logs", "network_artifacts"]
            )

            # Collect evidence
            await forensic_agent.collect_evidence(
                case_id=case_id,
                incident=incident,
                evidence_types=["event_logs", "network_artifacts"],
                db_session=db_session
            )

            # Analyze evidence
            analysis_results = await forensic_agent.analyze_evidence(
                case_id=case_id,
                evidence_ids=None
            )

            results["forensics"] = {
                "case_id": case_id,
                "analysis": analysis_results
            }
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
                "recent_events": recent_events
            }

            containment_results = await containment_agent.orchestrate_response(
                incident=incident,
                recent_events=recent_events,
                db_session=db_session
            )

            results["containment"] = containment_results
            workflow.agents_involved.append("containment")
            workflow.decisions.append({
                "agent": "containment",
                "decision_type": "containment_actions",
                "confidence": containment_results.get("confidence", 0.0),
                "actions": containment_results.get("actions", []),
                "timestamp": datetime.utcnow().isoformat()
            })

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
                    events=recent_events,
                    timeframe_hours=24
                )

                # Generate deception strategy
                strategy = await deception_agent.ai_powered_deception_strategy(
                    threat_intelligence=results.get("threat_intelligence", {}),
                    current_attacks=["brute_force", "reconnaissance"],
                    organizational_profile={"industry": "technology", "size": "enterprise"}
                )

                results["deception"] = {
                    "attacker_profiles": attacker_profiles,
                    "strategy": strategy
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
        db_session=None
    ) -> Dict[str, Any]:
        """Execute rapid response workflow for high-priority incidents"""

        # Focus on containment and forensics, skip attribution
        results = {
            "containment": {},
            "forensics": {},
            "rapid_response": True
        }

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
        db_session=None
    ) -> Dict[str, Any]:
        """Execute basic workflow for low-priority incidents"""

        # Simple containment-focused workflow
        containment_agent = self.agents[AgentRole.CONTAINMENT]
        results = await containment_agent.orchestrate_response(
            incident, recent_events, db_session
        )

        workflow.agents_involved.append("containment")

        return {
            "containment": results,
            "basic_workflow": True
        }

    async def _rapid_forensic_analysis(
        self,
        incident: Incident,
        recent_events: List[Event],
        db_session=None
    ) -> Dict[str, Any]:
        """Rapid forensic analysis for high-priority incidents"""

        forensic_agent = self.agents[AgentRole.FORENSICS]

        # Quick evidence collection
        case_id = await forensic_agent.initiate_forensic_case(
            incident=incident,
            investigator="orchestrator_rapid",
            evidence_types=["event_logs"]
        )

        # Rapid analysis
        analysis = await forensic_agent.analyze_evidence(
            case_id=case_id,
            evidence_ids=None
        )

        return {
            "case_id": case_id,
            "rapid_analysis": analysis,
            "evidence_collected": ["event_logs"]
        }

    async def _coordinate_final_decision(
        self,
        agent_results: Dict[str, Any],
        incident: Incident,
        workflow: WorkflowContext
    ) -> Dict[str, Any]:
        """Coordinate final decision from all agent inputs"""

        coordination = {
            "decision_factors": {},
            "confidence_levels": {},
            "recommended_actions": [],
            "risk_assessment": {},
            "final_decision": {}
        }

        # Extract key decision factors
        attribution_confidence = agent_results.get("attribution", {}).get("confidence_score", 0)
        containment_confidence = agent_results.get("containment", {}).get("confidence", 0)
        forensic_risk = agent_results.get("forensics", {}).get("analysis", {}).get("risk_assessment", {}).get("overall_risk_score", 0)

        coordination["decision_factors"] = {
            "attribution_confidence": attribution_confidence,
            "containment_confidence": containment_confidence,
            "forensic_risk_score": forensic_risk,
            "threat_intel_risk": agent_results.get("threat_intelligence", {}).get("reputation_score", 0)
        }

        # Calculate overall confidence
        weights = {"attribution": 0.3, "containment": 0.3, "forensic": 0.4}
        overall_confidence = (
            attribution_confidence * weights["attribution"] +
            containment_confidence * weights["containment"] +
            (1 - forensic_risk) * weights["forensic"]  # Invert forensic risk
        )

        coordination["confidence_levels"] = {
            "overall": overall_confidence,
            "attribution": attribution_confidence,
            "containment": containment_confidence,
            "forensic": 1 - forensic_risk
        }

        # Determine final actions
        actions = []

        # Always include containment actions if any were recommended
        containment_actions = agent_results.get("containment", {}).get("actions", [])
        if containment_actions:
            actions.extend(containment_actions)

        # Add forensic recommendations
        forensic_recs = agent_results.get("forensics", {}).get("analysis", {}).get("recommendations", [])
        if forensic_recs:
            actions.extend([{"action": "forensic_recommendation", "details": rec} for rec in forensic_recs])

        # Add deception actions if high risk
        if overall_confidence > 0.7:
            deception_actions = agent_results.get("deception", {}).get("strategy", {}).get("adaptive_responses", [])
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
            "escalation_recommended": risk_level in ["high", "critical"]
        }

        # Final decision
        coordination["final_decision"] = {
            "should_contain": overall_confidence > 0.5,
            "should_investigate": overall_confidence > 0.3,
            "should_escalate": risk_level in ["high", "critical"],
            "priority_level": "high" if risk_level in ["high", "critical"] else "medium",
            "automated_response": overall_confidence > 0.6,
            "human_review_required": overall_confidence < 0.8
        }

        return coordination

    async def send_agent_message(
        self,
        recipient: AgentRole,
        message_type: MessageType,
        payload: Dict[str, Any],
        priority: int = 1
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
            priority=priority
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
                                priority=message.priority
                            )

                    elif message.message_type == MessageType.NOTIFICATION:
                        await self._handle_agent_notification(agent, message)

                    self.stats["messages_processed"] += 1

        except Exception as e:
            self.logger.error(f"Failed to process message {message.message_id}: {e}")

    async def _handle_agent_request(self, agent, message: AgentMessage) -> Optional[Dict[str, Any]]:
        """Handle a request message to an agent"""

        # This would contain logic to translate messages into agent method calls
        # For now, return a basic acknowledgment
        return {
            "message_id": message.message_id,
            "status": "received",
            "timestamp": datetime.utcnow().isoformat()
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
            "uptime_seconds": (datetime.utcnow() - self.stats["start_time"]).total_seconds(),
            "agents": {},
            "active_workflows": len(self.active_workflows),
            "message_queue_length": len(self.message_queue),
            "statistics": self.stats.copy()
        }

        # Agent status
        for role, agent in self.agents.items():
            agent_status = {
                "status": "active",
                "agent_id": getattr(agent, 'agent_id', 'unknown'),
                "last_activity": getattr(agent, 'last_activity', None)
            }

            # Check if agent is responsive
            try:
                if hasattr(agent, 'agent_id'):
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
                "end_time": workflow.end_time.isoformat() if workflow.end_time else None,
                "current_step": workflow.current_step,
                "agents_involved": workflow.agents_involved,
                "errors": workflow.errors,
                "execution_time": (
                    (workflow.end_time or datetime.utcnow()) - workflow.start_time
                ).total_seconds()
            }

        return None


# Global orchestrator instance
orchestrator = AgentOrchestrator()


async def get_orchestrator() -> AgentOrchestrator:
    """Get the global orchestrator instance"""
    if not hasattr(orchestrator, '_initialized'):
        await orchestrator.initialize()
        orchestrator._initialized = True
    return orchestrator

"""
SOAR-Style Playbook Engine for Enhanced Mini-XDR
Automated workflow orchestration for incident response
"""
import asyncio
import json
import yaml
import logging
import time
from typing import Dict, List, Any, Optional, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
from enum import Enum
import jinja2
import re

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None

from .models import Incident, Event, Action
from .agents.containment_agent import ContainmentAgent, ThreatHuntingAgent, RollbackAgent
from .agents.attribution_agent import AttributionAgent
from .agents.forensics_agent import ForensicsAgent
from .external_intel import ThreatIntelligence
from .ml_engine import EnsembleMLDetector
from .responder import block_ip, unblock_ip
from .config import settings

logger = logging.getLogger(__name__)


class PlaybookStatus(Enum):
    """Playbook execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    CANCELLED = "cancelled"


class StepStatus(Enum):
    """Individual step status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class PlaybookStep:
    """Individual playbook step"""
    step_id: str
    name: str
    action: str
    parameters: Dict[str, Any]
    conditions: Dict[str, Any] = field(default_factory=dict)
    timeout_seconds: int = 300
    retry_count: int = 0
    max_retries: int = 3
    depends_on: List[str] = field(default_factory=list)
    status: StepStatus = StepStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@dataclass
class PlaybookExecution:
    """Playbook execution instance"""
    execution_id: str
    playbook_id: str
    incident_id: int
    status: PlaybookStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    steps: List[PlaybookStep] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    results: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class Playbook:
    """Playbook definition"""
    playbook_id: str
    name: str
    description: str
    version: str
    trigger_conditions: Dict[str, Any]
    steps: List[PlaybookStep]
    variables: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


class PlaybookEngine:
    """SOAR-style playbook execution engine"""
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client or self._init_llm_client()
        self.logger = logging.getLogger(__name__)
        
        # Initialize agents
        self.containment_agent = ContainmentAgent()
        self.threat_hunter = ThreatHuntingAgent()
        self.rollback_agent = RollbackAgent()
        self.attribution_agent = AttributionAgent()
        self.forensics_agent = ForensicsAgent()
        self.threat_intel = ThreatIntelligence()
        self.ml_detector = EnsembleMLDetector()
        
        # Playbook storage
        self.playbooks: Dict[str, Playbook] = {}
        self.executions: Dict[str, PlaybookExecution] = {}
        
        # Action registry
        self.action_registry = {
            # Containment actions
            "block_ip": self._action_block_ip,
            "unblock_ip": self._action_unblock_ip,
            "isolate_host": self._action_isolate_host,
            
            # Investigation actions
            "collect_evidence": self._action_collect_evidence,
            "analyze_logs": self._action_analyze_logs,
            "threat_hunt": self._action_threat_hunt,
            "attribution_analysis": self._action_attribution_analysis,
            
            # Intelligence actions
            "lookup_ip": self._action_lookup_ip,
            "query_threat_intel": self._action_query_threat_intel,
            "ml_analysis": self._action_ml_analysis,
            
            # Communication actions
            "notify_analyst": self._action_notify_analyst,
            "send_alert": self._action_send_alert,
            "update_ticket": self._action_update_ticket,
            
            # Orchestration actions
            "wait": self._action_wait,
            "condition_check": self._action_condition_check,
            "loop": self._action_loop,
            "parallel_execution": self._action_parallel_execution,
            
            # AI-powered actions
            "ai_decision": self._action_ai_decision,
            "ai_analysis": self._action_ai_analysis,
            "generate_report": self._action_generate_report
        }
        
        # Template engine for dynamic content
        self.template_env = jinja2.Environment(
            loader=jinja2.BaseLoader(),
            autoescape=True
        )
        
        # Load built-in playbooks
        asyncio.create_task(self._load_builtin_playbooks())
    
    def _init_llm_client(self):
        """Initialize LLM client for AI-powered actions"""
        try:
            if settings.llm_provider.lower() == "openai" and settings.openai_api_key:
                if ChatOpenAI:
                    return ChatOpenAI(
                        openai_api_key=settings.openai_api_key,
                        model_name=settings.openai_model,
                        temperature=0.2
                    )
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM client: {e}")
        return None
    
    async def _load_builtin_playbooks(self):
        """Load built-in playbooks"""
        
        builtin_playbooks = [
            self._create_ssh_brute_force_playbook(),
            self._create_malware_detection_playbook(),
            self._create_lateral_movement_playbook(),
            self._create_data_exfiltration_playbook(),
            self._create_comprehensive_investigation_playbook()
        ]
        
        for playbook in builtin_playbooks:
            self.playbooks[playbook.playbook_id] = playbook
            self.logger.info(f"Loaded built-in playbook: {playbook.name}")
    
    def _create_ssh_brute_force_playbook(self) -> Playbook:
        """Create SSH brute force response playbook"""
        
        steps = [
            PlaybookStep(
                step_id="initial_analysis",
                name="Initial Analysis",
                action="ml_analysis",
                parameters={
                    "analysis_type": "brute_force_detection",
                    "src_ip": "{{ incident.src_ip }}",
                    "lookback_hours": 2
                }
            ),
            PlaybookStep(
                step_id="threat_intel_lookup",
                name="Threat Intelligence Lookup",
                action="lookup_ip",
                parameters={
                    "ip": "{{ incident.src_ip }}"
                }
            ),
            PlaybookStep(
                step_id="risk_assessment",
                name="Risk Assessment",
                action="ai_decision",
                parameters={
                    "decision_type": "containment_urgency",
                    "context": {
                        "ml_result": "{{ steps.initial_analysis.result }}",
                        "threat_intel": "{{ steps.threat_intel_lookup.result }}"
                    }
                },
                depends_on=["initial_analysis", "threat_intel_lookup"]
            ),
            PlaybookStep(
                step_id="immediate_containment",
                name="Immediate Containment",
                action="block_ip",
                parameters={
                    "ip": "{{ incident.src_ip }}",
                    "duration": 3600
                },
                conditions={
                    "if": "{{ steps.risk_assessment.result.should_contain == true }}"
                },
                depends_on=["risk_assessment"]
            ),
            PlaybookStep(
                step_id="evidence_collection",
                name="Evidence Collection",
                action="collect_evidence",
                parameters={
                    "case_type": "ssh_brute_force",
                    "evidence_types": ["event_logs", "network_artifacts"],
                    "incident_id": "{{ incident.id }}"
                },
                depends_on=["immediate_containment"]
            ),
            PlaybookStep(
                step_id="threat_hunting",
                name="Proactive Threat Hunting",
                action="threat_hunt",
                parameters={
                    "hunt_type": "related_campaigns",
                    "src_ip": "{{ incident.src_ip }}",
                    "lookback_hours": 24
                },
                depends_on=["evidence_collection"]
            ),
            PlaybookStep(
                step_id="analyst_notification",
                name="Analyst Notification",
                action="notify_analyst",
                parameters={
                    "priority": "{{ steps.risk_assessment.result.priority }}",
                    "summary": "SSH brute force detected from {{ incident.src_ip }}",
                    "actions_taken": "{{ steps.immediate_containment.result }}"
                },
                depends_on=["threat_hunting"]
            ),
            PlaybookStep(
                step_id="generate_report",
                name="Generate Incident Report",
                action="generate_report",
                parameters={
                    "report_type": "ssh_brute_force",
                    "incident_data": "{{ incident }}",
                    "investigation_results": {
                        "ml_analysis": "{{ steps.initial_analysis.result }}",
                        "threat_intel": "{{ steps.threat_intel_lookup.result }}",
                        "evidence": "{{ steps.evidence_collection.result }}",
                        "hunt_results": "{{ steps.threat_hunting.result }}"
                    }
                },
                depends_on=["analyst_notification"]
            )
        ]
        
        return Playbook(
            playbook_id="ssh_brute_force_response",
            name="SSH Brute Force Response",
            description="Automated response to SSH brute force attacks",
            version="1.0",
            trigger_conditions={
                "incident_reason": {"contains": ["brute", "ssh", "login"]},
                "event_count": {"gt": 10},
                "escalation_level": {"in": ["medium", "high", "critical"]}
            },
            steps=steps,
            tags=["brute_force", "ssh", "containment", "automated"]
        )
    
    def _create_malware_detection_playbook(self) -> Playbook:
        """Create malware detection response playbook"""
        
        steps = [
            PlaybookStep(
                step_id="malware_analysis",
                name="Malware Analysis",
                action="ai_analysis",
                parameters={
                    "analysis_type": "malware_classification",
                    "incident_data": "{{ incident }}",
                    "context": "file_download_detected"
                }
            ),
            PlaybookStep(
                step_id="threat_intel_correlation",
                name="Threat Intelligence Correlation",
                action="query_threat_intel",
                parameters={
                    "indicators": ["{{ incident.src_ip }}"],
                    "correlation_type": "malware_campaign"
                }
            ),
            PlaybookStep(
                step_id="forensic_collection",
                name="Forensic Evidence Collection",
                action="collect_evidence",
                parameters={
                    "case_type": "malware_incident",
                    "evidence_types": ["file_artifacts", "memory_dump", "network_artifacts"],
                    "incident_id": "{{ incident.id }}",
                    "preservation_priority": "high"
                },
                depends_on=["malware_analysis"]
            ),
            PlaybookStep(
                step_id="attribution_analysis",
                name="Attribution Analysis",
                action="attribution_analysis",
                parameters={
                    "analysis_scope": "campaign_correlation",
                    "incident_data": "{{ incident }}",
                    "evidence_data": "{{ steps.forensic_collection.result }}"
                },
                depends_on=["forensic_collection", "threat_intel_correlation"]
            ),
            PlaybookStep(
                step_id="containment_decision",
                name="Containment Decision",
                action="ai_decision",
                parameters={
                    "decision_type": "malware_containment",
                    "context": {
                        "malware_analysis": "{{ steps.malware_analysis.result }}",
                        "attribution": "{{ steps.attribution_analysis.result }}",
                        "threat_intel": "{{ steps.threat_intel_correlation.result }}"
                    }
                },
                depends_on=["attribution_analysis"]
            ),
            PlaybookStep(
                step_id="execute_containment",
                name="Execute Containment",
                action="block_ip",
                parameters={
                    "ip": "{{ incident.src_ip }}",
                    "duration": "{{ steps.containment_decision.result.duration }}",
                    "reason": "malware_detection"
                },
                conditions={
                    "if": "{{ steps.containment_decision.result.should_contain == true }}"
                },
                depends_on=["containment_decision"]
            ),
            PlaybookStep(
                step_id="escalate_to_analyst",
                name="Escalate to Analyst",
                action="send_alert",
                parameters={
                    "alert_type": "malware_detection",
                    "priority": "high",
                    "details": {
                        "incident_id": "{{ incident.id }}",
                        "malware_family": "{{ steps.malware_analysis.result.family }}",
                        "attribution": "{{ steps.attribution_analysis.result.summary }}",
                        "containment_status": "{{ steps.execute_containment.result.status }}"
                    }
                },
                depends_on=["execute_containment"]
            )
        ]
        
        return Playbook(
            playbook_id="malware_detection_response",
            name="Malware Detection Response",
            description="Comprehensive response to malware detection incidents",
            version="1.0",
            trigger_conditions={
                "incident_reason": {"contains": ["malware", "download", "file"]},
                "eventid": {"contains": ["file_download", "command.input"]},
                "escalation_level": {"in": ["high", "critical"]}
            },
            steps=steps,
            tags=["malware", "forensics", "attribution", "automated"]
        )
    
    def _create_lateral_movement_playbook(self) -> Playbook:
        """Create lateral movement detection playbook"""
        
        steps = [
            PlaybookStep(
                step_id="movement_analysis",
                name="Lateral Movement Analysis",
                action="threat_hunt",
                parameters={
                    "hunt_type": "lateral_movement",
                    "src_ip": "{{ incident.src_ip }}",
                    "lookback_hours": 24,
                    "scope": "network_wide"
                }
            ),
            PlaybookStep(
                step_id="network_mapping",
                name="Network Infrastructure Mapping",
                action="ai_analysis",
                parameters={
                    "analysis_type": "network_topology",
                    "target_ip": "{{ incident.src_ip }}",
                    "hunt_results": "{{ steps.movement_analysis.result }}"
                },
                depends_on=["movement_analysis"]
            ),
            PlaybookStep(
                step_id="impact_assessment",
                name="Impact Assessment",
                action="ai_decision",
                parameters={
                    "decision_type": "lateral_movement_impact",
                    "context": {
                        "movement_pattern": "{{ steps.movement_analysis.result }}",
                        "network_topology": "{{ steps.network_mapping.result }}"
                    }
                },
                depends_on=["network_mapping"]
            ),
            PlaybookStep(
                step_id="coordinated_containment",
                name="Coordinated Containment",
                action="parallel_execution",
                parameters={
                    "actions": [
                        {
                            "action": "block_ip",
                            "parameters": {"ip": "{{ incident.src_ip }}", "duration": 7200}
                        },
                        {
                            "action": "isolate_host", 
                            "parameters": {"targets": "{{ steps.impact_assessment.result.affected_hosts }}"}
                        }
                    ]
                },
                conditions={
                    "if": "{{ steps.impact_assessment.result.contains_immediately == true }}"
                },
                depends_on=["impact_assessment"]
            ),
            PlaybookStep(
                step_id="forensic_preservation",
                name="Forensic Evidence Preservation",
                action="collect_evidence",
                parameters={
                    "case_type": "lateral_movement",
                    "evidence_types": ["event_logs", "memory_dump", "network_artifacts", "system_state"],
                    "incident_id": "{{ incident.id }}",
                    "scope": "multi_host"
                },
                depends_on=["coordinated_containment"]
            ),
            PlaybookStep(
                step_id="campaign_attribution",
                name="Campaign Attribution",
                action="attribution_analysis",
                parameters={
                    "analysis_scope": "advanced_persistent_threat",
                    "evidence_data": "{{ steps.forensic_preservation.result }}",
                    "movement_pattern": "{{ steps.movement_analysis.result }}"
                },
                depends_on=["forensic_preservation"]
            ),
            PlaybookStep(
                step_id="incident_escalation",
                name="Critical Incident Escalation",
                action="send_alert",
                parameters={
                    "alert_type": "critical_security_incident",
                    "priority": "critical",
                    "escalation_chain": ["security_team", "incident_commander", "ciso"],
                    "details": {
                        "incident_type": "lateral_movement",
                        "affected_systems": "{{ steps.impact_assessment.result.affected_hosts }}",
                        "attribution": "{{ steps.campaign_attribution.result }}",
                        "containment_status": "{{ steps.coordinated_containment.result }}"
                    }
                },
                depends_on=["campaign_attribution"]
            )
        ]
        
        return Playbook(
            playbook_id="lateral_movement_response",
            name="Lateral Movement Response",
            description="Advanced response to lateral movement detection",
            version="1.0",
            trigger_conditions={
                "threat_category": {"equals": "lateral_movement"},
                "escalation_level": {"in": ["high", "critical"]},
                "event_count": {"gt": 5}
            },
            steps=steps,
            tags=["lateral_movement", "apt", "coordinated_response", "critical"]
        )
    
    def _create_data_exfiltration_playbook(self) -> Playbook:
        """Create data exfiltration response playbook"""
        
        steps = [
            PlaybookStep(
                step_id="exfiltration_analysis",
                name="Data Exfiltration Analysis",
                action="ai_analysis",
                parameters={
                    "analysis_type": "data_exfiltration",
                    "incident_data": "{{ incident }}",
                    "scope": "data_flow_analysis"
                }
            ),
            PlaybookStep(
                step_id="immediate_blocking",
                name="Immediate Traffic Blocking",
                action="block_ip",
                parameters={
                    "ip": "{{ incident.src_ip }}",
                    "duration": 86400,  # 24 hours
                    "reason": "data_exfiltration_prevention"
                }
            ),
            PlaybookStep(
                step_id="data_impact_assessment",
                name="Data Impact Assessment",
                action="ai_decision",
                parameters={
                    "decision_type": "data_breach_severity",
                    "context": {
                        "exfiltration_analysis": "{{ steps.exfiltration_analysis.result }}",
                        "incident_details": "{{ incident }}"
                    }
                },
                depends_on=["exfiltration_analysis", "immediate_blocking"]
            ),
            PlaybookStep(
                step_id="comprehensive_evidence_collection",
                name="Comprehensive Evidence Collection",
                action="collect_evidence",
                parameters={
                    "case_type": "data_exfiltration",
                    "evidence_types": ["event_logs", "file_artifacts", "network_artifacts", "memory_dump", "system_state"],
                    "incident_id": "{{ incident.id }}",
                    "preservation_priority": "critical"
                },
                depends_on=["data_impact_assessment"]
            ),
            PlaybookStep(
                step_id="legal_notification_check",
                name="Legal Notification Requirements Check",
                action="ai_decision",
                parameters={
                    "decision_type": "legal_notification_required",
                    "context": {
                        "data_assessment": "{{ steps.data_impact_assessment.result }}",
                        "evidence_summary": "{{ steps.comprehensive_evidence_collection.result }}"
                    }
                },
                depends_on=["comprehensive_evidence_collection"]
            ),
            PlaybookStep(
                step_id="executive_notification",
                name="Executive Team Notification",
                action="send_alert",
                parameters={
                    "alert_type": "data_breach_notification",
                    "priority": "critical",
                    "escalation_chain": ["ciso", "ceo", "legal_counsel"],
                    "details": {
                        "breach_severity": "{{ steps.data_impact_assessment.result.severity }}",
                        "legal_requirements": "{{ steps.legal_notification_check.result }}",
                        "evidence_status": "{{ steps.comprehensive_evidence_collection.result.status }}"
                    }
                },
                depends_on=["legal_notification_check"]
            ),
            PlaybookStep(
                step_id="forensic_investigation",
                name="Full Forensic Investigation",
                action="generate_report",
                parameters={
                    "report_type": "data_breach_forensic",
                    "comprehensive": true,
                    "evidence_data": "{{ steps.comprehensive_evidence_collection.result }}",
                    "legal_format": true
                },
                depends_on=["executive_notification"]
            )
        ]
        
        return Playbook(
            playbook_id="data_exfiltration_response",
            name="Data Exfiltration Response",
            description="Critical response to data exfiltration incidents",
            version="1.0",
            trigger_conditions={
                "incident_reason": {"contains": ["exfiltration", "data_breach", "upload"]},
                "eventid": {"contains": ["file_upload", "large_transfer"]},
                "escalation_level": {"equals": "critical"}
            },
            steps=steps,
            tags=["data_breach", "exfiltration", "legal", "executive", "critical"]
        )
    
    def _create_comprehensive_investigation_playbook(self) -> Playbook:
        """Create comprehensive investigation playbook"""
        
        steps = [
            PlaybookStep(
                step_id="initial_triage",
                name="Initial Incident Triage",
                action="ai_analysis",
                parameters={
                    "analysis_type": "incident_classification",
                    "incident_data": "{{ incident }}",
                    "comprehensive": true
                }
            ),
            PlaybookStep(
                step_id="multi_source_analysis",
                name="Multi-Source Intelligence Analysis",
                action="parallel_execution",
                parameters={
                    "actions": [
                        {
                            "action": "ml_analysis",
                            "parameters": {"src_ip": "{{ incident.src_ip }}", "lookback_hours": 24}
                        },
                        {
                            "action": "threat_hunt",
                            "parameters": {"hunt_type": "comprehensive", "src_ip": "{{ incident.src_ip }}"}
                        },
                        {
                            "action": "lookup_ip",
                            "parameters": {"ip": "{{ incident.src_ip }}"}
                        }
                    ]
                },
                depends_on=["initial_triage"]
            ),
            PlaybookStep(
                step_id="evidence_orchestration",
                name="Evidence Collection Orchestration",
                action="collect_evidence",
                parameters={
                    "case_type": "comprehensive_investigation",
                    "evidence_types": ["event_logs", "file_artifacts", "network_artifacts", "memory_dump", "system_state"],
                    "incident_id": "{{ incident.id }}",
                    "chain_of_custody": true
                },
                depends_on=["multi_source_analysis"]
            ),
            PlaybookStep(
                step_id="attribution_and_campaign_analysis",
                name="Attribution and Campaign Analysis",
                action="attribution_analysis",
                parameters={
                    "analysis_scope": "full_campaign_analysis",
                    "incident_data": "{{ incident }}",
                    "intelligence_data": "{{ steps.multi_source_analysis.result }}",
                    "evidence_data": "{{ steps.evidence_orchestration.result }}"
                },
                depends_on=["evidence_orchestration"]
            ),
            PlaybookStep(
                step_id="risk_and_impact_assessment",
                name="Risk and Impact Assessment",
                action="ai_decision",
                parameters={
                    "decision_type": "comprehensive_risk_assessment",
                    "context": {
                        "triage": "{{ steps.initial_triage.result }}",
                        "intelligence": "{{ steps.multi_source_analysis.result }}",
                        "attribution": "{{ steps.attribution_and_campaign_analysis.result }}"
                    }
                },
                depends_on=["attribution_and_campaign_analysis"]
            ),
            PlaybookStep(
                step_id="response_orchestration",
                name="Response Action Orchestration",
                action="ai_decision",
                parameters={
                    "decision_type": "response_strategy",
                    "context": {
                        "risk_assessment": "{{ steps.risk_and_impact_assessment.result }}",
                        "available_actions": ["containment", "monitoring", "deception", "legal_action"]
                    }
                },
                depends_on=["risk_and_impact_assessment"]
            ),
            PlaybookStep(
                step_id="execute_response_plan",
                name="Execute Response Plan",
                action="parallel_execution",
                parameters={
                    "actions": "{{ steps.response_orchestration.result.action_plan }}"
                },
                depends_on=["response_orchestration"]
            ),
            PlaybookStep(
                step_id="comprehensive_reporting",
                name="Comprehensive Investigation Report",
                action="generate_report",
                parameters={
                    "report_type": "comprehensive_investigation",
                    "include_all_phases": true,
                    "executive_summary": true,
                    "technical_details": true,
                    "evidence_appendix": true,
                    "investigation_data": {
                        "triage": "{{ steps.initial_triage.result }}",
                        "intelligence": "{{ steps.multi_source_analysis.result }}",
                        "evidence": "{{ steps.evidence_orchestration.result }}",
                        "attribution": "{{ steps.attribution_and_campaign_analysis.result }}",
                        "response": "{{ steps.execute_response_plan.result }}"
                    }
                },
                depends_on=["execute_response_plan"]
            )
        ]
        
        return Playbook(
            playbook_id="comprehensive_investigation",
            name="Comprehensive Security Investigation",
            description="Full-scale investigation playbook for complex incidents",
            version="1.0",
            trigger_conditions={
                "escalation_level": {"equals": "critical"},
                "manual_trigger": true,
                "investigation_type": {"equals": "comprehensive"}
            },
            steps=steps,
            tags=["comprehensive", "investigation", "advanced", "manual"]
        )
    
    async def execute_playbook(
        self, 
        playbook_id: str, 
        incident: Incident, 
        context: Dict[str, Any] = None,
        db_session=None
    ) -> str:
        """
        Execute a playbook for an incident
        
        Args:
            playbook_id: ID of the playbook to execute
            incident: The incident to respond to
            context: Additional context variables
            db_session: Database session
            
        Returns:
            Execution ID
        """
        try:
            if playbook_id not in self.playbooks:
                raise ValueError(f"Playbook {playbook_id} not found")
            
            playbook = self.playbooks[playbook_id]
            execution_id = f"exec_{incident.id}_{playbook_id}_{int(time.time())}"
            
            # Create execution instance
            execution = PlaybookExecution(
                execution_id=execution_id,
                playbook_id=playbook_id,
                incident_id=incident.id,
                status=PlaybookStatus.PENDING,
                started_at=datetime.utcnow(),
                steps=[step for step in playbook.steps],  # Copy steps
                context=context or {},
                results={}
            )
            
            # Add incident data to context
            execution.context.update({
                "incident": {
                    "id": incident.id,
                    "src_ip": incident.src_ip,
                    "reason": incident.reason,
                    "escalation_level": incident.escalation_level,
                    "risk_score": incident.risk_score,
                    "created_at": incident.created_at.isoformat()
                }
            })
            
            self.executions[execution_id] = execution
            
            # Start execution asynchronously
            asyncio.create_task(self._execute_playbook_async(execution_id, db_session))
            
            self.logger.info(f"Started playbook execution {execution_id} for incident {incident.id}")
            
            return execution_id
            
        except Exception as e:
            self.logger.error(f"Failed to start playbook execution: {e}")
            raise
    
    async def _execute_playbook_async(self, execution_id: str, db_session=None):
        """Execute playbook asynchronously"""
        
        execution = self.executions[execution_id]
        execution.status = PlaybookStatus.RUNNING
        
        try:
            # Build dependency graph
            dependency_graph = self._build_dependency_graph(execution.steps)
            
            # Execute steps according to dependencies
            await self._execute_steps_with_dependencies(execution, dependency_graph, db_session)
            
            execution.status = PlaybookStatus.COMPLETED
            execution.completed_at = datetime.utcnow()
            
            self.logger.info(f"Playbook execution {execution_id} completed successfully")
            
        except Exception as e:
            execution.status = PlaybookStatus.FAILED
            execution.error = str(e)
            execution.completed_at = datetime.utcnow()
            
            self.logger.error(f"Playbook execution {execution_id} failed: {e}")
    
    def _build_dependency_graph(self, steps: List[PlaybookStep]) -> Dict[str, List[str]]:
        """Build step dependency graph"""
        
        graph = {}
        
        for step in steps:
            graph[step.step_id] = step.depends_on.copy()
        
        return graph
    
    async def _execute_steps_with_dependencies(
        self, 
        execution: PlaybookExecution, 
        dependency_graph: Dict[str, List[str]],
        db_session=None
    ):
        """Execute steps respecting dependencies"""
        
        completed_steps = set()
        
        while len(completed_steps) < len(execution.steps):
            # Find steps ready to execute
            ready_steps = []
            
            for step in execution.steps:
                if (step.step_id not in completed_steps and 
                    step.status == StepStatus.PENDING and
                    all(dep in completed_steps for dep in step.depends_on)):
                    ready_steps.append(step)
            
            if not ready_steps:
                # Check for circular dependencies or errors
                pending_steps = [s for s in execution.steps if s.step_id not in completed_steps]
                if pending_steps:
                    raise RuntimeError(f"Circular dependency or error in steps: {[s.step_id for s in pending_steps]}")
                break
            
            # Execute ready steps in parallel
            tasks = []
            for step in ready_steps:
                task = asyncio.create_task(
                    self._execute_step(step, execution, db_session),
                    name=f"step_{step.step_id}"
                )
                tasks.append((step.step_id, task))
            
            # Wait for all ready steps to complete
            for step_id, task in tasks:
                try:
                    await task
                    completed_steps.add(step_id)
                except Exception as e:
                    self.logger.error(f"Step {step_id} failed: {e}")
                    # Continue with other steps unless it's a critical failure
    
    async def _execute_step(
        self, 
        step: PlaybookStep, 
        execution: PlaybookExecution,
        db_session=None
    ):
        """Execute a single playbook step"""
        
        step.status = StepStatus.RUNNING
        step.started_at = datetime.utcnow()
        
        try:
            # Check conditions
            if step.conditions and not await self._evaluate_conditions(step.conditions, execution):
                step.status = StepStatus.SKIPPED
                step.completed_at = datetime.utcnow()
                step.result = {"skipped": True, "reason": "conditions_not_met"}
                return
            
            # Render parameters with context
            rendered_params = await self._render_parameters(step.parameters, execution)
            
            # Execute action with retry logic
            for attempt in range(step.max_retries + 1):
                try:
                    result = await self._execute_action(
                        step.action, 
                        rendered_params, 
                        execution,
                        db_session,
                        timeout=step.timeout_seconds
                    )
                    
                    step.result = result
                    step.status = StepStatus.COMPLETED
                    step.completed_at = datetime.utcnow()
                    
                    # Update execution context with step result
                    execution.context[f"steps.{step.step_id}.result"] = result
                    
                    self.logger.info(f"Step {step.step_id} completed successfully")
                    break
                    
                except Exception as e:
                    step.retry_count = attempt
                    
                    if attempt < step.max_retries:
                        self.logger.warning(f"Step {step.step_id} attempt {attempt + 1} failed, retrying: {e}")
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    else:
                        raise e
        
        except Exception as e:
            step.status = StepStatus.FAILED
            step.error = str(e)
            step.completed_at = datetime.utcnow()
            
            self.logger.error(f"Step {step.step_id} failed: {e}")
    
    async def _evaluate_conditions(self, conditions: Dict[str, Any], execution: PlaybookExecution) -> bool:
        """Evaluate step conditions"""
        
        try:
            # Simple condition evaluation
            if "if" in conditions:
                condition_expr = conditions["if"]
                # Render condition with context
                rendered_condition = await self._render_template(condition_expr, execution.context)
                
                # Basic evaluation (in production, use a safer expression evaluator)
                if rendered_condition.lower() in ["true", "1", "yes"]:
                    return True
                elif rendered_condition.lower() in ["false", "0", "no"]:
                    return False
                else:
                    # Try to evaluate as Python expression (unsafe - use with caution)
                    try:
                        return bool(eval(rendered_condition))
                    except:
                        return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Condition evaluation failed: {e}")
            return False
    
    async def _render_parameters(self, parameters: Dict[str, Any], execution: PlaybookExecution) -> Dict[str, Any]:
        """Render parameter templates with execution context"""
        
        rendered = {}
        
        for key, value in parameters.items():
            if isinstance(value, str):
                rendered[key] = await self._render_template(value, execution.context)
            elif isinstance(value, dict):
                rendered[key] = await self._render_parameters(value, execution)
            elif isinstance(value, list):
                rendered[key] = [
                    await self._render_template(item, execution.context) if isinstance(item, str) else item
                    for item in value
                ]
            else:
                rendered[key] = value
        
        return rendered
    
    async def _render_template(self, template: str, context: Dict[str, Any]) -> str:
        """Render Jinja2 template with context"""
        
        try:
            template_obj = self.template_env.from_string(template)
            return template_obj.render(**context)
        except Exception as e:
            self.logger.warning(f"Template rendering failed for '{template}': {e}")
            return template
    
    async def _execute_action(
        self, 
        action: str, 
        parameters: Dict[str, Any], 
        execution: PlaybookExecution,
        db_session=None,
        timeout: int = 300
    ) -> Dict[str, Any]:
        """Execute a playbook action"""
        
        if action not in self.action_registry:
            raise ValueError(f"Unknown action: {action}")
        
        action_func = self.action_registry[action]
        
        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                action_func(parameters, execution, db_session),
                timeout=timeout
            )
            
            return result
            
        except asyncio.TimeoutError:
            raise RuntimeError(f"Action {action} timed out after {timeout} seconds")
        except Exception as e:
            raise RuntimeError(f"Action {action} failed: {e}")
    
    # Action implementations
    async def _action_block_ip(self, params: Dict[str, Any], execution: PlaybookExecution, db_session=None) -> Dict[str, Any]:
        """Block IP address action"""
        
        ip = params["ip"]
        duration = params.get("duration", 3600)
        reason = params.get("reason", "playbook_execution")
        
        status, detail = await block_ip(ip, duration)
        
        return {
            "action": "block_ip",
            "ip": ip,
            "duration": duration,
            "status": status,
            "detail": detail,
            "reason": reason
        }
    
    async def _action_unblock_ip(self, params: Dict[str, Any], execution: PlaybookExecution, db_session=None) -> Dict[str, Any]:
        """Unblock IP address action"""
        
        ip = params["ip"]
        
        status, detail = await unblock_ip(ip)
        
        return {
            "action": "unblock_ip",
            "ip": ip,
            "status": status,
            "detail": detail
        }
    
    async def _action_isolate_host(self, params: Dict[str, Any], execution: PlaybookExecution, db_session=None) -> Dict[str, Any]:
        """Isolate host action (placeholder)"""
        
        targets = params.get("targets", [])
        isolation_level = params.get("level", "network")
        
        # Placeholder implementation
        self.logger.info(f"Would isolate hosts {targets} at level {isolation_level}")
        
        return {
            "action": "isolate_host",
            "targets": targets,
            "isolation_level": isolation_level,
            "status": "simulated",
            "detail": f"Host isolation simulated for {len(targets)} targets"
        }
    
    async def _action_collect_evidence(self, params: Dict[str, Any], execution: PlaybookExecution, db_session=None) -> Dict[str, Any]:
        """Collect forensic evidence action"""
        
        case_type = params.get("case_type", "incident_response")
        evidence_types = params.get("evidence_types", ["event_logs"])
        incident_id = params.get("incident_id")
        
        if not incident_id:
            raise ValueError("incident_id required for evidence collection")
        
        # Create mock incident for forensics agent
        mock_incident = type('Incident', (), {
            'id': incident_id,
            'src_ip': execution.context.get("incident", {}).get("src_ip", "unknown"),
            'created_at': datetime.utcnow()
        })()
        
        # Initiate forensic case
        case_id = await self.forensics_agent.initiate_forensic_case(
            mock_incident, 
            investigator="playbook_engine"
        )
        
        # Collect evidence
        evidence_ids = await self.forensics_agent.collect_evidence(
            case_id, 
            mock_incident, 
            evidence_types,
            db_session
        )
        
        return {
            "action": "collect_evidence",
            "case_id": case_id,
            "evidence_types": evidence_types,
            "evidence_ids": evidence_ids,
            "evidence_count": len(evidence_ids)
        }
    
    async def _action_analyze_logs(self, params: Dict[str, Any], execution: PlaybookExecution, db_session=None) -> Dict[str, Any]:
        """Analyze logs action"""
        
        analysis_type = params.get("analysis_type", "general")
        src_ip = params.get("src_ip")
        lookback_hours = params.get("lookback_hours", 24)
        
        # Use ML detector for log analysis
        if src_ip and db_session:
            # Mock events for analysis
            mock_events = []
            score = await self.ml_detector.calculate_anomaly_score(src_ip, mock_events)
            
            return {
                "action": "analyze_logs",
                "analysis_type": analysis_type,
                "src_ip": src_ip,
                "anomaly_score": score,
                "findings": f"Log analysis completed for {src_ip}"
            }
        
        return {
            "action": "analyze_logs",
            "status": "no_data",
            "detail": "Insufficient data for log analysis"
        }
    
    async def _action_threat_hunt(self, params: Dict[str, Any], execution: PlaybookExecution, db_session=None) -> Dict[str, Any]:
        """Threat hunting action"""
        
        hunt_type = params.get("hunt_type", "general")
        src_ip = params.get("src_ip")
        lookback_hours = params.get("lookback_hours", 24)
        
        # Execute threat hunt
        hunt_results = await self.threat_hunter.hunt_for_threats(db_session, lookback_hours)
        
        return {
            "action": "threat_hunt",
            "hunt_type": hunt_type,
            "src_ip": src_ip,
            "findings": hunt_results,
            "findings_count": len(hunt_results)
        }
    
    async def _action_attribution_analysis(self, params: Dict[str, Any], execution: PlaybookExecution, db_session=None) -> Dict[str, Any]:
        """Attribution analysis action"""
        
        analysis_scope = params.get("analysis_scope", "incident")
        incident_data = params.get("incident_data", {})
        
        # Mock attribution analysis
        attribution_result = {
            "actor_type": "unknown",
            "confidence": 0.5,
            "indicators": [],
            "campaign_correlation": False
        }
        
        return {
            "action": "attribution_analysis",
            "analysis_scope": analysis_scope,
            "result": attribution_result,
            "summary": "Attribution analysis completed"
        }
    
    async def _action_lookup_ip(self, params: Dict[str, Any], execution: PlaybookExecution, db_session=None) -> Dict[str, Any]:
        """IP lookup action"""
        
        ip = params["ip"]
        
        # Use threat intelligence service
        intel_result = await self.threat_intel.lookup_ip(ip)
        
        return {
            "action": "lookup_ip",
            "ip": ip,
            "result": {
                "is_malicious": intel_result.is_malicious if intel_result else False,
                "risk_score": intel_result.risk_score if intel_result else 0.0,
                "category": intel_result.category if intel_result else "unknown"
            }
        }
    
    async def _action_query_threat_intel(self, params: Dict[str, Any], execution: PlaybookExecution, db_session=None) -> Dict[str, Any]:
        """Query threat intelligence action"""
        
        indicators = params.get("indicators", [])
        correlation_type = params.get("correlation_type", "general")
        
        # Query threat intelligence for each indicator
        results = {}
        for indicator in indicators:
            intel_result = await self.threat_intel.lookup_ip(indicator)
            results[indicator] = {
                "is_malicious": intel_result.is_malicious if intel_result else False,
                "risk_score": intel_result.risk_score if intel_result else 0.0
            }
        
        return {
            "action": "query_threat_intel",
            "indicators": indicators,
            "correlation_type": correlation_type,
            "results": results
        }
    
    async def _action_ml_analysis(self, params: Dict[str, Any], execution: PlaybookExecution, db_session=None) -> Dict[str, Any]:
        """ML analysis action"""
        
        analysis_type = params.get("analysis_type", "anomaly_detection")
        src_ip = params.get("src_ip")
        lookback_hours = params.get("lookback_hours", 2)
        
        # Use ML detector
        if src_ip:
            mock_events = []  # In production, query actual events
            score = await self.ml_detector.calculate_anomaly_score(src_ip, mock_events)
            
            return {
                "action": "ml_analysis",
                "analysis_type": analysis_type,
                "src_ip": src_ip,
                "anomaly_score": score,
                "model_status": self.ml_detector.get_model_status()
            }
        
        return {
            "action": "ml_analysis",
            "status": "no_data",
            "detail": "Insufficient data for ML analysis"
        }
    
    async def _action_notify_analyst(self, params: Dict[str, Any], execution: PlaybookExecution, db_session=None) -> Dict[str, Any]:
        """Notify analyst action"""
        
        priority = params.get("priority", "medium")
        summary = params.get("summary", "Playbook notification")
        actions_taken = params.get("actions_taken", "")
        
        # Placeholder notification
        self.logger.info(f"ANALYST NOTIFICATION [{priority}]: {summary} - Actions: {actions_taken}")
        
        return {
            "action": "notify_analyst",
            "priority": priority,
            "summary": summary,
            "status": "sent",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _action_send_alert(self, params: Dict[str, Any], execution: PlaybookExecution, db_session=None) -> Dict[str, Any]:
        """Send alert action"""
        
        alert_type = params.get("alert_type", "general")
        priority = params.get("priority", "medium")
        escalation_chain = params.get("escalation_chain", ["security_team"])
        details = params.get("details", {})
        
        # Placeholder alert sending
        self.logger.warning(f"ALERT [{priority}] {alert_type}: {details} -> {escalation_chain}")
        
        return {
            "action": "send_alert",
            "alert_type": alert_type,
            "priority": priority,
            "escalation_chain": escalation_chain,
            "status": "sent",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _action_update_ticket(self, params: Dict[str, Any], execution: PlaybookExecution, db_session=None) -> Dict[str, Any]:
        """Update ticket action"""
        
        ticket_id = params.get("ticket_id")
        update_type = params.get("update_type", "comment")
        content = params.get("content", "")
        
        # Placeholder ticket update
        self.logger.info(f"TICKET UPDATE {ticket_id} [{update_type}]: {content}")
        
        return {
            "action": "update_ticket",
            "ticket_id": ticket_id,
            "update_type": update_type,
            "status": "updated"
        }
    
    async def _action_wait(self, params: Dict[str, Any], execution: PlaybookExecution, db_session=None) -> Dict[str, Any]:
        """Wait action"""
        
        duration = params.get("duration", 60)
        reason = params.get("reason", "workflow_delay")
        
        await asyncio.sleep(duration)
        
        return {
            "action": "wait",
            "duration": duration,
            "reason": reason,
            "status": "completed"
        }
    
    async def _action_condition_check(self, params: Dict[str, Any], execution: PlaybookExecution, db_session=None) -> Dict[str, Any]:
        """Condition check action"""
        
        condition = params.get("condition", "true")
        
        # Evaluate condition
        result = await self._evaluate_conditions({"if": condition}, execution)
        
        return {
            "action": "condition_check",
            "condition": condition,
            "result": result,
            "status": "evaluated"
        }
    
    async def _action_loop(self, params: Dict[str, Any], execution: PlaybookExecution, db_session=None) -> Dict[str, Any]:
        """Loop action"""
        
        iterations = params.get("iterations", 1)
        actions = params.get("actions", [])
        
        results = []
        
        for i in range(iterations):
            iteration_results = []
            for action_def in actions:
                action_name = action_def.get("action")
                action_params = action_def.get("parameters", {})
                
                if action_name in self.action_registry:
                    result = await self.action_registry[action_name](action_params, execution, db_session)
                    iteration_results.append(result)
            
            results.append({
                "iteration": i + 1,
                "results": iteration_results
            })
        
        return {
            "action": "loop",
            "iterations": iterations,
            "results": results,
            "status": "completed"
        }
    
    async def _action_parallel_execution(self, params: Dict[str, Any], execution: PlaybookExecution, db_session=None) -> Dict[str, Any]:
        """Parallel execution action"""
        
        actions = params.get("actions", [])
        
        # Execute actions in parallel
        tasks = []
        for action_def in actions:
            action_name = action_def.get("action")
            action_params = action_def.get("parameters", {})
            
            if action_name in self.action_registry:
                task = asyncio.create_task(
                    self.action_registry[action_name](action_params, execution, db_session)
                )
                tasks.append((action_name, task))
        
        # Wait for all tasks to complete
        results = []
        for action_name, task in tasks:
            try:
                result = await task
                results.append({
                    "action": action_name,
                    "result": result,
                    "status": "success"
                })
            except Exception as e:
                results.append({
                    "action": action_name,
                    "error": str(e),
                    "status": "failed"
                })
        
        return {
            "action": "parallel_execution",
            "results": results,
            "total_actions": len(actions),
            "successful_actions": len([r for r in results if r["status"] == "success"])
        }
    
    async def _action_ai_decision(self, params: Dict[str, Any], execution: PlaybookExecution, db_session=None) -> Dict[str, Any]:
        """AI-powered decision action"""
        
        decision_type = params.get("decision_type", "general")
        context = params.get("context", {})
        
        if not self.llm_client:
            return {
                "action": "ai_decision",
                "decision_type": decision_type,
                "result": {"decision": "no_ai", "confidence": 0.0},
                "status": "no_ai_available"
            }
        
        # Prepare AI prompt based on decision type
        prompts = {
            "containment_urgency": """
                Based on the incident analysis, determine if immediate containment is required.
                
                Context: {context}
                
                Respond with JSON:
                {{
                    "should_contain": true/false,
                    "urgency": "low|medium|high|critical",
                    "reasoning": "explanation",
                    "confidence": 0.85
                }}
            """,
            "response_strategy": """
                Analyze the incident and recommend a response strategy.
                
                Context: {context}
                
                Respond with JSON:
                {{
                    "strategy": "contain|monitor|investigate|legal_action",
                    "action_plan": [action objects],
                    "priority": "low|medium|high|critical",
                    "reasoning": "explanation"
                }}
            """
        }
        
        prompt_template = prompts.get(decision_type, """
            Make a decision based on the provided context.
            
            Context: {context}
            
            Respond with JSON format containing your decision and reasoning.
        """)
        
        prompt = prompt_template.format(context=json.dumps(context, indent=2))
        
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.llm_client.invoke(prompt)
            )
            
            # Parse AI response
            import re
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                ai_result = json.loads(json_match.group())
                
                return {
                    "action": "ai_decision",
                    "decision_type": decision_type,
                    "result": ai_result,
                    "status": "completed"
                }
        
        except Exception as e:
            self.logger.error(f"AI decision failed: {e}")
        
        # Fallback decision
        return {
            "action": "ai_decision",
            "decision_type": decision_type,
            "result": {"decision": "fallback", "confidence": 0.5},
            "status": "fallback"
        }
    
    async def _action_ai_analysis(self, params: Dict[str, Any], execution: PlaybookExecution, db_session=None) -> Dict[str, Any]:
        """AI-powered analysis action"""
        
        analysis_type = params.get("analysis_type", "general")
        incident_data = params.get("incident_data", {})
        context = params.get("context", {})
        
        if not self.llm_client:
            return {
                "action": "ai_analysis",
                "analysis_type": analysis_type,
                "result": {"analysis": "no_ai", "confidence": 0.0},
                "status": "no_ai_available"
            }
        
        prompt = f"""
        Perform {analysis_type} analysis on the following incident data:
        
        Incident: {json.dumps(incident_data, indent=2)}
        Context: {json.dumps(context, indent=2)}
        
        Provide detailed analysis in JSON format with findings, risk assessment, and recommendations.
        """
        
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.llm_client.invoke(prompt)
            )
            
            return {
                "action": "ai_analysis",
                "analysis_type": analysis_type,
                "result": {"analysis": response.content, "confidence": 0.8},
                "status": "completed"
            }
        
        except Exception as e:
            self.logger.error(f"AI analysis failed: {e}")
            return {
                "action": "ai_analysis",
                "analysis_type": analysis_type,
                "result": {"analysis": "analysis_failed", "error": str(e)},
                "status": "failed"
            }
    
    async def _action_generate_report(self, params: Dict[str, Any], execution: PlaybookExecution, db_session=None) -> Dict[str, Any]:
        """Generate report action"""
        
        report_type = params.get("report_type", "incident")
        incident_data = params.get("incident_data", {})
        investigation_results = params.get("investigation_results", {})
        
        # Generate comprehensive report
        report = {
            "report_id": f"report_{execution.execution_id}_{int(time.time())}",
            "report_type": report_type,
            "generated_at": datetime.utcnow().isoformat(),
            "incident_summary": incident_data,
            "investigation_findings": investigation_results,
            "playbook_execution": {
                "playbook_id": execution.playbook_id,
                "execution_id": execution.execution_id,
                "steps_completed": len([s for s in execution.steps if s.status == StepStatus.COMPLETED])
            }
        }
        
        return {
            "action": "generate_report",
            "report_type": report_type,
            "report": report,
            "status": "generated"
        }
    
    async def check_playbook_triggers(self, incident: Incident, events: List[Event] = None) -> List[str]:
        """
        Check which playbooks should be triggered for an incident
        
        Args:
            incident: The incident to check
            events: Related events
            
        Returns:
            List of matching playbook IDs
        """
        matching_playbooks = []
        
        for playbook_id, playbook in self.playbooks.items():
            if await self._evaluate_trigger_conditions(playbook.trigger_conditions, incident, events):
                matching_playbooks.append(playbook_id)
        
        return matching_playbooks
    
    async def _evaluate_trigger_conditions(
        self, 
        conditions: Dict[str, Any], 
        incident: Incident, 
        events: List[Event] = None
    ) -> bool:
        """Evaluate playbook trigger conditions"""
        
        try:
            for condition_key, condition_value in conditions.items():
                if condition_key == "incident_reason":
                    if isinstance(condition_value, dict):
                        if "contains" in condition_value:
                            if not any(term.lower() in incident.reason.lower() for term in condition_value["contains"]):
                                return False
                        elif "equals" in condition_value:
                            if incident.reason.lower() != condition_value["equals"].lower():
                                return False
                
                elif condition_key == "escalation_level":
                    if isinstance(condition_value, dict):
                        if "in" in condition_value:
                            if incident.escalation_level not in condition_value["in"]:
                                return False
                        elif "equals" in condition_value:
                            if incident.escalation_level != condition_value["equals"]:
                                return False
                
                elif condition_key == "event_count":
                    if events and isinstance(condition_value, dict):
                        if "gt" in condition_value:
                            if len(events) <= condition_value["gt"]:
                                return False
                        elif "lt" in condition_value:
                            if len(events) >= condition_value["lt"]:
                                return False
                
                elif condition_key == "manual_trigger":
                    # Manual triggers are not automatically evaluated
                    if condition_value:
                        return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Trigger condition evaluation failed: {e}")
            return False
    
    async def get_execution_status(self, execution_id: str) -> Dict[str, Any]:
        """Get status of a playbook execution"""
        
        if execution_id not in self.executions:
            return {"error": f"Execution {execution_id} not found"}
        
        execution = self.executions[execution_id]
        
        status = {
            "execution_id": execution_id,
            "playbook_id": execution.playbook_id,
            "incident_id": execution.incident_id,
            "status": execution.status.value,
            "started_at": execution.started_at.isoformat(),
            "completed_at": execution.completed_at.isoformat() if execution.completed_at else None,
            "steps": [
                {
                    "step_id": step.step_id,
                    "name": step.name,
                    "status": step.status.value,
                    "started_at": step.started_at.isoformat() if step.started_at else None,
                    "completed_at": step.completed_at.isoformat() if step.completed_at else None,
                    "error": step.error
                }
                for step in execution.steps
            ],
            "error": execution.error
        }
        
        return status
    
    async def list_available_playbooks(self) -> List[Dict[str, Any]]:
        """List all available playbooks"""
        
        playbooks = []
        
        for playbook_id, playbook in self.playbooks.items():
            playbooks.append({
                "playbook_id": playbook_id,
                "name": playbook.name,
                "description": playbook.description,
                "version": playbook.version,
                "step_count": len(playbook.steps),
                "tags": playbook.tags,
                "created_at": playbook.created_at.isoformat(),
                "updated_at": playbook.updated_at.isoformat()
            })
        
        return playbooks

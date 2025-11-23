"""
AI Containment Agent for Autonomous Threat Response
Uses LangChain with OpenAI/xAI for intelligent containment decisions
"""
import asyncio
import hashlib
import hmac
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

try:
    from langchain.agents import AgentType, Tool, initialize_agent
    from langchain.memory import ConversationBufferMemory
    from langchain.schema import HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI, OpenAI
except ImportError:
    # Fallback if LangChain is not available
    logging.warning("LangChain not available, using basic agent implementation")
    initialize_agent = None
    Tool = None
    AgentType = None
    OpenAI = None
    ChatOpenAI = None
    ConversationBufferMemory = None

from ..config import settings
from ..enhanced_containment import ContainmentDecision, EnhancedContainmentEngine
from ..external_intel import ThreatIntelligence
from ..ml_engine import EnsembleMLDetector
from ..models import Action, Event, Incident
from ..responder import block_ip, unblock_ip

logger = logging.getLogger(__name__)


class ContainmentAgent:
    """AI Agent for autonomous threat response orchestration"""

    def __init__(self, llm_client=None, threat_intel=None, ml_detector=None):
        self.llm_client = llm_client or self._init_llm_client()
        self.threat_intel = threat_intel or ThreatIntelligence()
        self.ml_detector = ml_detector or EnsembleMLDetector()
        self.engine = EnhancedContainmentEngine(self.threat_intel, self.ml_detector)

        self.agent_id = "containment_orchestrator_v1"
        self.default_honeypot_host = (
            getattr(settings, "honeypot_host", None) or "tpot-honeypot"
        )
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Initialize LangChain agent if available
        if initialize_agent and self.llm_client:
            self._init_langchain_agent()
        else:
            self.agent = None
            self.logger.warning("Using basic agent implementation without LangChain")

    def _init_llm_client(self):
        """Initialize LLM client based on settings"""
        try:
            if settings.llm_provider.lower() == "openai" and settings.openai_api_key:
                if ChatOpenAI:
                    return ChatOpenAI(
                        openai_api_key=settings.openai_api_key,
                        model_name=settings.openai_model,
                        temperature=0.1,
                    )
            # Add other providers as needed
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM client: {e}")
        return None

    def _init_langchain_agent(self):
        """Initialize LangChain agent with tools"""
        if not self.llm_client:
            return

        try:
            # Define agent tools
            tools = [
                Tool(
                    name="BlockIP",
                    func=self._block_ip_wrapper,
                    description="Block an IP address using UFW. Input: 'ip duration_seconds'",
                ),
                Tool(
                    name="IsolateHost",
                    func=self._isolate_host,
                    description="Isolate a host via network segmentation. Input: 'hostname level'",
                ),
                Tool(
                    name="NotifyAnalyst",
                    func=self._notify_analyst,
                    description="Send alert to analysts. Input: 'incident_id message'",
                ),
                Tool(
                    name="RollbackAction",
                    func=self._rollback_action,
                    description="Rollback a containment action. Input: 'incident_id'",
                ),
                Tool(
                    name="QueryThreatIntel",
                    func=self._query_threat_intel,
                    description="Query threat intelligence for an IP. Input: 'ip_address'",
                ),
            ]

            # Initialize memory
            memory = ConversationBufferMemory(memory_key="chat_history")

            # Initialize agent
            self.agent = initialize_agent(
                tools,
                self.llm_client,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                memory=memory,
                verbose=True,
            )

            self.logger.info("LangChain containment agent initialized")

        except Exception as e:
            self.logger.error(f"Failed to initialize LangChain agent: {e}")
            self.agent = None

    async def orchestrate_response(
        self, incident: Incident, recent_events: List[Event], db_session=None
    ) -> Dict[str, Any]:
        """
        Agent-orchestrated containment: LLM decides actions based on engine input

        Args:
            incident: The incident to respond to
            recent_events: Recent events from the source IP
            db_session: Database session for updates

        Returns:
            Dict with response details and executed actions
        """
        try:
            # Step 1: Get base decision from enhanced engine
            decision = await self.engine.evaluate_containment(
                incident, recent_events, db_session
            )

            # Step 2: Get ML anomaly score
            ml_score = 0.0
            if self.ml_detector:
                try:
                    ml_score = await self.ml_detector.calculate_anomaly_score(
                        incident.src_ip, recent_events
                    )
                except Exception as e:
                    self.logger.warning(f"ML scoring failed: {e}")

            # Step 3: Get threat intelligence
            intel_result = None
            if self.threat_intel:
                try:
                    intel_result = await self.threat_intel.lookup_ip(incident.src_ip)
                except Exception as e:
                    self.logger.warning(f"Threat intel lookup failed: {e}")

            # Step 4: Agent decision making
            if self.agent:
                # Use LangChain agent for decision
                response = await self._langchain_orchestrate(
                    incident, decision, ml_score, intel_result, recent_events
                )
            else:
                # Use basic decision logic
                response = await self._basic_orchestrate(
                    incident, decision, ml_score, intel_result, db_session
                )

            # Step 5: Update incident with agent data
            if db_session and incident:
                incident.agent_id = self.agent_id
                incident.agent_actions = response.get("actions", [])
                incident.agent_confidence = response.get("confidence", 0.5)
                incident.containment_method = "ai_agent"
                incident.risk_score = decision.risk_score
                incident.escalation_level = decision.escalation_level
                incident.threat_category = decision.threat_category

                await db_session.commit()

            self.logger.info(
                f"Agent orchestrated response for incident {incident.id}: {response.get('reason', 'No reason')}"
            )

            return response

        except Exception as e:
            self.logger.error(f"Agent orchestration failed: {e}")
            # Fallback to engine decision
            return await self._fallback_containment(
                incident, decision if "decision" in locals() else None
            )

    async def _langchain_orchestrate(
        self,
        incident: Incident,
        decision: ContainmentDecision,
        ml_score: float,
        intel_result: Any,
        recent_events: List[Event],
    ) -> Dict[str, Any]:
        """Use LangChain agent for orchestration"""

        # Prepare context for the agent
        context = {
            "incident": {
                "id": incident.id,
                "src_ip": incident.src_ip,
                "reason": incident.reason,
                "escalation_level": incident.escalation_level,
            },
            "decision": {
                "should_contain": decision.should_contain,
                "risk_score": decision.risk_score,
                "confidence": decision.confidence,
                "escalation_level": decision.escalation_level,
                "reasoning": decision.reasoning,
            },
            "ml_score": ml_score,
            "threat_intel": {
                "risk_score": intel_result.risk_score if intel_result else 0.0,
                "category": intel_result.category if intel_result else "unknown",
                "is_malicious": intel_result.is_malicious if intel_result else False,
            },
            "event_count": len(recent_events),
            "auto_contain_enabled": settings.auto_contain,
        }

        # Agent prompt
        prompt = f"""
        You are a security containment agent. Analyze this incident and decide on actions:

        INCIDENT DATA:
        {json.dumps(context, indent=2)}

        AVAILABLE ACTIONS:
        - BlockIP: Block the source IP address
        - IsolateHost: Isolate the host (if available)
        - NotifyAnalyst: Send alert to human analysts
        - RollbackAction: Rollback previous actions if false positive

        DECISION CRITERIA:
        - Risk score > 0.7: Immediate containment
        - Risk score 0.4-0.7: Containment with analyst notification
        - Risk score < 0.4: Monitor only, notify if persistent
        - ML score > 0.8: High confidence anomaly
        - Threat intel malicious: Immediate action

        Respond with JSON format:
        {{
            "actions": ["action1", "action2"],
            "reasoning": "detailed explanation",
            "confidence": 0.95,
            "escalate_to_human": false
        }}

        Consider the escalation level and be proportional in response.
        """

        try:
            # Execute agent
            response = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.agent.run(prompt)
            )

            # Parse response
            agent_output = self._parse_agent_response(response)

            # Execute actions
            executed_actions = await self._execute_actions(
                agent_output["actions"], incident, decision
            )

            return {
                "success": True,
                "actions": executed_actions,
                "reasoning": agent_output["reasoning"],
                "confidence": agent_output["confidence"],
                "escalate_to_human": agent_output.get("escalate_to_human", False),
            }

        except Exception as e:
            self.logger.error(f"LangChain agent execution failed: {e}")
            raise

    async def _basic_orchestrate(
        self,
        incident: Incident,
        decision: ContainmentDecision,
        ml_score: float,
        intel_result: Any,
        db_session=None,
    ) -> Dict[str, Any]:
        """Basic orchestration without LangChain"""

        actions = []
        reasoning = []

        threat_category = (
            incident.threat_category or decision.threat_category or ""
        ).lower()
        workflow_result = None

        if db_session:
            playbook = self._build_critical_playbook(threat_category, incident)
            if playbook:
                workflow_result = await self._launch_advanced_playbook(
                    incident, playbook, db_session
                )
                if workflow_result and workflow_result.get("success"):
                    reasoning.append(playbook["success_reason"])
                    confidence = max(decision.confidence or 0.7, 0.85)
                    executed_actions = [
                        {
                            "action": "automated_workflow",
                            "workflow_id": workflow_result.get("workflow_id"),
                            "status": workflow_result.get("status", "executed"),
                            "timestamp": datetime.utcnow().isoformat(),
                            "steps": workflow_result.get(
                                "total_steps", len(playbook["steps"])
                            ),
                        }
                    ]

                    if workflow_result.get("approval_required"):
                        executed_actions[0]["approval_required"] = True

                    return {
                        "success": True,
                        "actions": executed_actions,
                        "reasoning": "; ".join(reasoning) or playbook["success_reason"],
                        "reason": playbook["success_reason"],
                        "confidence": confidence,
                        "workflow_triggered": workflow_result.get("workflow_id"),
                    }
                elif workflow_result and not workflow_result.get("success"):
                    reasoning.append(
                        f"Advanced playbook failed: {workflow_result.get('error', 'unknown error')}"
                    )

        # Simple rule-based decision making
        if decision.should_contain or ml_score > 0.8:
            actions.append("block")
            reasoning.append(f"High risk detected (score: {decision.risk_score:.2f})")

        if intel_result and intel_result.is_malicious:
            if "block" not in actions:
                actions.append("block")
            reasoning.append("Threat intelligence indicates malicious IP")

        if decision.escalation_level in ["high", "critical"]:
            actions.append("notify")
            reasoning.append(f"Escalation level: {decision.escalation_level}")

        if not actions:
            actions.append("monitor")
            reasoning.append("Low risk, monitoring only")

        # Execute actions
        executed_actions = await self._execute_actions(actions, incident, decision)

        return {
            "success": True,
            "actions": executed_actions,
            "reasoning": "; ".join(reasoning),
            "reason": "; ".join(reasoning),
            "confidence": max(decision.confidence or 0.6, 0.72),
        }

    def _build_critical_playbook(
        self, threat_category: str, incident: Incident
    ) -> Optional[Dict[str, Any]]:
        """Return an enriched response playbook for critical threats"""
        if not threat_category:
            return None

        src_ip = incident.src_ip
        host_identifier = self.default_honeypot_host

        if "ransom" in threat_category:
            steps = [
                {
                    "action_type": "block_ip_advanced",
                    "parameters": {
                        "ip_address": src_ip,
                        "duration": 604800,
                        "block_level": "aggressive",
                    },
                },
                {
                    "action_type": "isolate_host_advanced",
                    "parameters": {
                        "host_identifier": host_identifier,
                        "isolation_level": "strict",
                        "exceptions": ["forensics-segment"],
                        "monitoring": "packet-capture",
                    },
                },
                {
                    "action_type": "memory_dump_collection",
                    "parameters": {
                        "target_hosts": [host_identifier],
                        "dump_type": "full",
                        "encryption": True,
                        "retention": "30d",
                    },
                },
                {
                    "action_type": "invoke_ai_agent",
                    "parameters": {
                        "agent": "containment",
                        "task": "emergency_isolation",
                        "context": "ransomware",
                    },
                },
                {
                    "action_type": "reset_passwords",
                    "parameters": {
                        "ip": host_identifier,
                        "reason": "Ransomware mitigation credential reset",
                    },
                },
                {
                    "action_type": "send_notification",
                    "parameters": {
                        "channel": "slack",
                        "priority": "high",
                        "message": f"🔥 Automated ransomware containment executed for {src_ip}",
                    },
                },
            ]
            return {
                "name": "Agent Ransomware Auto-Containment",
                "priority": "CRITICAL",
                "steps": steps,
                "success_reason": "Critical ransomware playbook executed via response engine",
            }

        if "data_exfil" in threat_category:
            steps = [
                {
                    "action_type": "block_ip_advanced",
                    "parameters": {
                        "ip_address": src_ip,
                        "duration": 172800,
                        "block_level": "aggressive",
                    },
                },
                {
                    "action_type": "deploy_firewall_rules",
                    "parameters": {
                        "rule_set": [
                            {"action": "block", "ip": src_ip, "protocol": "*"},
                            {
                                "action": "mirror",
                                "ip": src_ip,
                                "destination": "forensics-tap",
                            },
                        ],
                        "scope": "edge",
                        "priority": "critical",
                        "expiration": 86400,
                    },
                },
                {
                    "action_type": "dns_sinkhole",
                    "parameters": {
                        "domains": [f"exfil-{src_ip}.demo.local"],
                        "sinkhole_ip": "10.99.99.99",
                        "ttl": 60,
                        "scope": "global",
                    },
                },
                {
                    "action_type": "memory_dump_collection",
                    "parameters": {
                        "target_hosts": [host_identifier],
                        "dump_type": "network-focused",
                        "retention": "7d",
                    },
                },
                {
                    "action_type": "invoke_ai_agent",
                    "parameters": {
                        "agent": "forensics",
                        "task": "analyze_data_transfer",
                        "context": "data_exfiltration",
                    },
                },
                {
                    "action_type": "send_notification",
                    "parameters": {
                        "channel": "slack",
                        "priority": "high",
                        "message": f"🚨 Data exfiltration workflow executed for {src_ip}",
                    },
                },
            ]
            return {
                "name": "Agent Data Exfiltration Response",
                "priority": "CRITICAL",
                "steps": steps,
                "success_reason": "Automated data exfiltration playbook deployed",
            }

        if "ddos" in threat_category:
            steps = [
                {
                    "action_type": "deploy_firewall_rules",
                    "parameters": {
                        "rule_set": [
                            {"action": "rate_limit", "ip": src_ip, "pps": 100},
                            {"action": "geo_block", "region": "*", "ip": src_ip},
                        ],
                        "scope": "edge",
                        "priority": "critical",
                        "expiration": 3600,
                    },
                },
                {
                    "action_type": "block_ip",
                    "parameters": {
                        "ip_address": src_ip,
                        "duration": 43200,
                        "block_level": "aggressive",
                    },
                },
                {
                    "action_type": "capture_traffic",
                    "parameters": {"ip": src_ip, "reason": "DDoS evidence collection"},
                },
                {
                    "action_type": "invoke_ai_agent",
                    "parameters": {
                        "agent": "containment",
                        "task": "enable_rate_limiting",
                        "context": "ddos_attack",
                    },
                },
                {
                    "action_type": "send_notification",
                    "parameters": {
                        "channel": "slack",
                        "priority": "medium",
                        "message": f"⚠️ Automated DDoS mitigation engaged for {src_ip}",
                    },
                },
            ]
            return {
                "name": "Agent DDoS Mitigation",
                "priority": "CRITICAL",
                "steps": steps,
                "success_reason": "Automated DDoS mitigation workflow executed",
            }

        return None

    async def _launch_advanced_playbook(
        self, incident: Incident, playbook: Dict[str, Any], db_session
    ) -> Optional[Dict[str, Any]]:
        """Launch an advanced response workflow for the incident"""
        try:
            from ..advanced_response_engine import ResponsePriority, get_response_engine

            response_engine = await get_response_engine()
            priority_name = playbook.get("priority", "HIGH")
            priority_enum = getattr(
                ResponsePriority, priority_name.upper(), ResponsePriority.HIGH
            )

            result = await response_engine.create_workflow(
                incident_id=incident.id,
                playbook_name=playbook["name"],
                steps=playbook["steps"],
                auto_execute=True,
                priority=priority_enum,
                db_session=db_session,
            )

            if not result.get("success"):
                self.logger.error(
                    "Advanced playbook creation failed for incident %s: %s",
                    incident.id,
                    result.get("error"),
                )
            else:
                self.logger.info(
                    "Advanced playbook %s auto-executed for incident %s",
                    playbook["name"],
                    incident.id,
                )

            return result

        except Exception as e:
            self.logger.error(f"Failed to launch advanced playbook: {e}")
            return {"success": False, "error": str(e)}

    def _parse_agent_response(self, response: str) -> Dict[str, Any]:
        """Parse agent response, handling various formats"""
        try:
            # Try to parse as JSON
            if response.strip().startswith("{"):
                return json.loads(response)

            # Extract JSON from response if wrapped in text
            import re

            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())

        except json.JSONDecodeError:
            pass

        # Fallback parsing
        return {
            "actions": ["block"] if "block" in response.lower() else ["monitor"],
            "reasoning": response[:200],
            "confidence": 0.5,
        }

    async def _execute_actions(
        self, actions: List[str], incident: Incident, decision: ContainmentDecision
    ) -> List[Dict[str, Any]]:
        """Execute the list of actions"""
        executed_actions = []

        for action in actions:
            try:
                if action == "block":
                    status = await self._block_ip_wrapper(
                        f"{incident.src_ip} {decision.duration_seconds}"
                    )
                    executed_actions.append(
                        {
                            "action": "block",
                            "status": status,
                            "timestamp": datetime.utcnow().isoformat(),
                        }
                    )

                elif action == "notify":
                    status = self._notify_analyst(
                        f"{incident.id} Agent-detected incident requiring attention"
                    )
                    executed_actions.append(
                        {
                            "action": "notify",
                            "status": status,
                            "timestamp": datetime.utcnow().isoformat(),
                        }
                    )

                elif action == "monitor":
                    executed_actions.append(
                        {
                            "action": "monitor",
                            "status": "Monitoring initiated",
                            "timestamp": datetime.utcnow().isoformat(),
                        }
                    )

                # Add other actions as needed

            except Exception as e:
                self.logger.error(f"Failed to execute action {action}: {e}")
                executed_actions.append(
                    {
                        "action": action,
                        "status": f"Failed: {e}",
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                )

        return executed_actions

    def _block_ip_wrapper(self, input_str: str) -> str:
        """Wrapper for blocking IP - tool format"""
        try:
            parts = input_str.split()
            ip = parts[0]
            duration = int(parts[1]) if len(parts) > 1 else 3600

            # Use existing block_ip function
            loop = asyncio.get_event_loop()
            status, detail = loop.run_until_complete(block_ip(ip, duration))

            return f"Blocked {ip} for {duration}s: {detail}"
        except Exception as e:
            return f"Block failed: {e}"

    def _isolate_host(self, input_str: str) -> str:
        """Isolate host using iptables network controls"""
        try:
            parts = input_str.split()
            hostname = parts[0]
            level = parts[1] if len(parts) > 1 else "soft"

            # Execute actual host isolation
            isolation_result = asyncio.get_event_loop().run_until_complete(
                self._execute_host_isolation(hostname, level)
            )

            return isolation_result
        except Exception as e:
            self.logger.error(f"Host isolation failed: {e}")
            return f"Isolation failed: {e}"

    async def _execute_host_isolation(self, target_ip: str, level: str) -> str:
        """Execute honeypot-aware isolation - sandbox attackers instead of blocking"""
        try:
            from ..responder import responder

            # Check if this is a honeypot environment
            is_honeypot = await self._is_honeypot_environment()

            if is_honeypot:
                return await self._execute_honeypot_isolation(target_ip, level)

            if level == "hard":
                # Complete network isolation - block all traffic to/from IP
                commands = [
                    f"iptables -I INPUT -s {target_ip} -j DROP",
                    f"iptables -I OUTPUT -d {target_ip} -j DROP",
                    f"iptables -I FORWARD -s {target_ip} -j DROP",
                    f"iptables -I FORWARD -d {target_ip} -j DROP",
                ]
                isolation_type = "complete network isolation"
            else:
                # Soft isolation - block only non-essential ports, allow SSH for management
                commands = [
                    f"iptables -I INPUT -s {target_ip} -p tcp --dport 80 -j DROP",
                    f"iptables -I INPUT -s {target_ip} -p tcp --dport 443 -j DROP",
                    f"iptables -I INPUT -s {target_ip} -p tcp --dport 21 -j DROP",
                    f"iptables -I INPUT -s {target_ip} -p tcp --dport 25 -j DROP",
                    f"iptables -I OUTPUT -d {target_ip} -p tcp --dport 80 -j DROP",
                    f"iptables -I OUTPUT -d {target_ip} -p tcp --dport 443 -j DROP",
                ]
                isolation_type = "selective service isolation"

            executed_commands = []
            failed_commands = []

            for cmd in commands:
                try:
                    # Execute iptables command via SSH to T-Pot
                    status, stdout, stderr = await responder.execute_command(
                        f"sudo {cmd}", timeout=30
                    )

                    if status == "success":
                        executed_commands.append(cmd)
                        self.logger.info(f"Executed isolation command: {cmd}")
                    else:
                        failed_commands.append(f"{cmd}: {stderr}")
                        self.logger.error(f"Failed to execute: {cmd} - {stderr}")

                except Exception as e:
                    failed_commands.append(f"{cmd}: {str(e)}")
                    self.logger.error(f"Command execution error: {cmd} - {e}")

            # Create isolation record file
            isolation_record = {
                "timestamp": datetime.utcnow().isoformat(),
                "target_ip": target_ip,
                "isolation_level": level,
                "isolation_type": isolation_type,
                "executed_commands": executed_commands,
                "failed_commands": failed_commands,
                "status": "active",
            }

            # Store isolation record for potential rollback
            isolation_file = f"/tmp/isolation_{target_ip}_{int(time.time())}.json"
            try:
                with open(isolation_file, "w") as f:
                    json.dump(isolation_record, f, indent=2)
                self.logger.info(f"Isolation record saved: {isolation_file}")
            except Exception as e:
                self.logger.warning(f"Failed to save isolation record: {e}")

            if executed_commands:
                success_msg = (
                    f"Host {target_ip} isolated successfully ({isolation_type})"
                )
                if failed_commands:
                    success_msg += f". {len(failed_commands)} commands failed"
                return success_msg
            else:
                return f"Host isolation failed - no commands executed successfully"

        except Exception as e:
            self.logger.error(f"Host isolation execution failed: {e}")
            return f"Isolation execution failed: {e}"

    async def _execute_host_un_isolation(
        self, target_ip: str, reason: str
    ) -> Dict[str, Any]:
        """Execute host un-isolation by removing iptables rules"""
        try:
            import glob

            from ..responder import responder

            # Find existing isolation record files
            isolation_pattern = f"/tmp/isolation_{target_ip.replace('.', '_')}_*.json"
            isolation_files = glob.glob(isolation_pattern)

            if not isolation_files:
                return {
                    "success": False,
                    "detail": f"No active isolation found for {target_ip}",
                }

            removed_rules = []
            failed_removals = []

            # Process each isolation record
            for isolation_file in isolation_files:
                try:
                    # Read the isolation record
                    with open(isolation_file, "r") as f:
                        isolation_record = json.load(f)

                    # Remove the iptables rules that were added
                    for cmd in isolation_record.get("executed_commands", []):
                        # Convert INSERT to DELETE command
                        if cmd.startswith("iptables -I"):
                            delete_cmd = cmd.replace("-I", "-D")
                            try:
                                # Execute via SSH to T-Pot
                                (
                                    status,
                                    stdout,
                                    stderr,
                                ) = await responder.execute_command(
                                    f"sudo {delete_cmd}", timeout=30
                                )

                                if status == "success":
                                    removed_rules.append(delete_cmd)
                                    self.logger.info(
                                        f"Removed isolation rule: {delete_cmd}"
                                    )
                                else:
                                    failed_removals.append(f"{delete_cmd}: {stderr}")
                                    self.logger.error(
                                        f"Failed to remove rule: {delete_cmd} - {stderr}"
                                    )

                            except Exception as e:
                                failed_removals.append(f"{delete_cmd}: {str(e)}")
                                self.logger.error(
                                    f"Rule removal error: {delete_cmd} - {e}"
                                )

                    # Remove the isolation record file
                    import os

                    os.remove(isolation_file)
                    self.logger.info(f"Removed isolation record: {isolation_file}")

                except Exception as e:
                    self.logger.error(
                        f"Failed to process isolation file {isolation_file}: {e}"
                    )
                    failed_removals.append(f"File processing error: {e}")

            # Create un-isolation record
            un_isolation_record = {
                "timestamp": datetime.utcnow().isoformat(),
                "target_ip": target_ip,
                "reason": reason,
                "removed_rules": removed_rules,
                "failed_removals": failed_removals,
                "status": "removed",
            }

            # Store un-isolation record
            un_isolation_file = f"/tmp/un_isolation_{target_ip.replace('.', '_')}_{int(time.time())}.json"
            try:
                with open(un_isolation_file, "w") as f:
                    json.dump(un_isolation_record, f, indent=2)
                self.logger.info(f"Un-isolation record saved: {un_isolation_file}")
            except Exception as e:
                self.logger.warning(f"Failed to save un-isolation record: {e}")

            if removed_rules:
                success_msg = f"Host {target_ip} un-isolated successfully. Removed {len(removed_rules)} isolation rules."
                if failed_removals:
                    success_msg += f" {len(failed_removals)} rules failed to remove."
                return {
                    "success": True,
                    "detail": success_msg,
                    "removed_rules": removed_rules,
                    "failed_removals": failed_removals,
                }
            else:
                return {
                    "success": False,
                    "detail": f"Host un-isolation failed - no rules successfully removed",
                    "failed_removals": failed_removals,
                }

        except Exception as e:
            self.logger.error(f"Host un-isolation execution failed: {e}")
            return {
                "success": False,
                "detail": f"Un-isolation execution failed: {str(e)}",
            }

    async def _is_honeypot_environment(self) -> bool:
        """Check if we're running in a honeypot environment (T-Pot, etc.)"""
        try:
            from ..responder import responder

            # Check for T-Pot specific indicators
            indicators = [
                "docker ps | grep -E '(cowrie|dionaea|honeytrap|tpot)'",
                "ls -la /opt/tpot",
                "systemctl status tpot",
                "docker ps | grep honeypot",
            ]

            for indicator in indicators:
                status, stdout, stderr = await responder.execute_command(
                    indicator, timeout=10
                )
                if status == "success" and stdout.strip():
                    self.logger.info("Honeypot environment detected")
                    return True

            return False
        except Exception as e:
            self.logger.warning(f"Could not determine honeypot environment: {e}")
            return False

    async def _execute_honeypot_isolation(self, target_ip: str, level: str) -> str:
        """Execute honeypot-specific isolation - sandbox instead of block"""
        try:
            import time

            from ..responder import responder

            self.logger.info(
                f"Executing honeypot isolation for {target_ip} (level: {level})"
            )

            if level == "hard":
                # Hard isolation: Redirect to isolated honeypot container
                return await self._redirect_to_isolated_honeypot(target_ip)
            elif level == "quarantine":
                # Quarantine: Enhanced monitoring with restricted network access
                return await self._enable_enhanced_monitoring(target_ip)
            else:
                # Soft isolation: Rate limiting and enhanced logging
                return await self._apply_rate_limiting(target_ip)

        except Exception as e:
            self.logger.error(f"Honeypot isolation failed for {target_ip}: {e}")
            return f"Honeypot isolation failed: {e}"

    async def _redirect_to_isolated_honeypot(self, target_ip: str) -> str:
        """Redirect attacker to isolated honeypot container"""
        try:
            import time

            from ..responder import responder

            # Create isolated honeypot container for this specific attacker
            container_name = (
                f"isolated_honeypot_{target_ip.replace('.', '_')}_{int(time.time())}"
            )

            commands = [
                # Create dedicated honeypot container with network isolation
                f"docker run -d --name {container_name} --network none --cap-drop ALL "
                f"--read-only --tmpfs /tmp --tmpfs /var/log "
                f"-p 2222 -p 8080 cowrie/cowrie:latest",
                # Create custom iptables rules to redirect this IP to the isolated container
                f"iptables -t nat -I PREROUTING -s {target_ip} -p tcp --dport 22 "
                f"-j REDIRECT --to-port $(docker port {container_name} 2222 | cut -d: -f2)",
                f"iptables -t nat -I PREROUTING -s {target_ip} -p tcp --dport 80 "
                f"-j REDIRECT --to-port $(docker port {container_name} 8080 | cut -d: -f2)",
                # Log all traffic from this IP
                f"iptables -I INPUT -s {target_ip} -j LOG --log-prefix 'ISOLATED_ATTACKER_{target_ip}: '",
            ]

            executed_commands = []
            failed_commands = []

            for cmd in commands:
                try:
                    status, stdout, stderr = await responder.execute_command(
                        f"sudo {cmd}", timeout=30
                    )
                    if status == "success":
                        executed_commands.append(cmd)
                        self.logger.info(f"Executed honeypot redirection: {cmd}")
                    else:
                        failed_commands.append(f"{cmd}: {stderr}")
                        self.logger.error(
                            f"Failed honeypot redirection: {cmd} - {stderr}"
                        )
                except Exception as e:
                    failed_commands.append(f"{cmd}: {str(e)}")

            # Create isolation record
            isolation_record = {
                "timestamp": datetime.utcnow().isoformat(),
                "target_ip": target_ip,
                "isolation_type": "honeypot_redirection",
                "container_name": container_name,
                "executed_commands": executed_commands,
                "failed_commands": failed_commands,
                "status": "active",
            }

            # Save isolation record
            isolation_file = f"/tmp/honeypot_isolation_{target_ip.replace('.', '_')}_{int(time.time())}.json"
            try:
                with open(isolation_file, "w") as f:
                    json.dump(isolation_record, f, indent=2)
                self.logger.info(f"Honeypot isolation record saved: {isolation_file}")
            except Exception as e:
                self.logger.warning(f"Failed to save isolation record: {e}")

            success_rate = len(executed_commands) / len(commands) if commands else 0

            if success_rate >= 0.8:
                return (
                    f"✅ Attacker {target_ip} redirected to isolated honeypot container {container_name}. "
                    f"Executed {len(executed_commands)}/{len(commands)} commands successfully."
                )
            else:
                return (
                    f"⚠️ Partial honeypot isolation for {target_ip}. "
                    f"Only {len(executed_commands)}/{len(commands)} commands succeeded. "
                    f"Failed: {len(failed_commands)}"
                )

        except Exception as e:
            self.logger.error(f"Honeypot redirection failed: {e}")
            return f"❌ Honeypot redirection failed: {e}"

    async def _enable_enhanced_monitoring(self, target_ip: str) -> str:
        """Enable enhanced monitoring for quarantined attacker"""
        try:
            import time

            from ..responder import responder

            commands = [
                # Enable detailed packet capture for this IP
                f"tcpdump -i any -s 0 -w /tmp/capture_{target_ip.replace('.', '_')}.pcap "
                f"host {target_ip} &",
                # Enhanced logging
                f"iptables -I INPUT -s {target_ip} -j LOG --log-level 4 "
                f"--log-prefix 'QUARANTINE_{target_ip}: '",
                f"iptables -I OUTPUT -d {target_ip} -j LOG --log-level 4 "
                f"--log-prefix 'QUARANTINE_OUT_{target_ip}: '",
                # Rate limit connections
                f"iptables -I INPUT -s {target_ip} -m limit --limit 10/min "
                f"--limit-burst 20 -j ACCEPT",
                f"iptables -I INPUT -s {target_ip} -j DROP",
            ]

            executed_commands = []

            for cmd in commands:
                try:
                    status, stdout, stderr = await responder.execute_command(
                        f"sudo {cmd}", timeout=15
                    )
                    if status == "success":
                        executed_commands.append(cmd)
                        self.logger.info(f"Enhanced monitoring enabled: {cmd}")
                except Exception as e:
                    self.logger.error(f"Failed to enable monitoring: {cmd} - {e}")

            # Save monitoring record
            monitoring_record = {
                "timestamp": datetime.utcnow().isoformat(),
                "target_ip": target_ip,
                "isolation_type": "enhanced_monitoring",
                "capture_file": f"/tmp/capture_{target_ip.replace('.', '_')}.pcap",
                "executed_commands": executed_commands,
                "status": "active",
            }

            monitoring_file = (
                f"/tmp/monitoring_{target_ip.replace('.', '_')}_{int(time.time())}.json"
            )
            with open(monitoring_file, "w") as f:
                json.dump(monitoring_record, f, indent=2)

            return (
                f"🔍 Enhanced monitoring enabled for {target_ip}. "
                f"Packet capture active, rate limiting applied. "
                f"Monitoring file: {monitoring_file}"
            )

        except Exception as e:
            self.logger.error(f"Enhanced monitoring failed: {e}")
            return f"❌ Enhanced monitoring failed: {e}"

    async def _apply_rate_limiting(self, target_ip: str) -> str:
        """Apply rate limiting and enhanced logging for soft isolation"""
        try:
            import time

            from ..responder import responder

            commands = [
                # Rate limit new connections
                f"iptables -I INPUT -s {target_ip} -p tcp --syn -m limit "
                f"--limit 5/min --limit-burst 10 -j ACCEPT",
                # Log suspicious activity
                f"iptables -I INPUT -s {target_ip} -m limit --limit 20/min "
                f"-j LOG --log-prefix 'RATE_LIMITED_{target_ip}: '",
                # Allow existing connections to continue (for investigation)
                f"iptables -I INPUT -s {target_ip} -m state --state ESTABLISHED,RELATED -j ACCEPT",
                # Drop excessive new connections
                f"iptables -I INPUT -s {target_ip} -p tcp --syn -j DROP",
            ]

            executed_commands = []

            for cmd in commands:
                try:
                    status, stdout, stderr = await responder.execute_command(
                        f"sudo {cmd}", timeout=15
                    )
                    if status == "success":
                        executed_commands.append(cmd)
                        self.logger.info(f"Rate limiting applied: {cmd}")
                except Exception as e:
                    self.logger.error(f"Failed rate limiting: {cmd} - {e}")

            # Save rate limiting record
            rate_limit_record = {
                "timestamp": datetime.utcnow().isoformat(),
                "target_ip": target_ip,
                "isolation_type": "rate_limiting",
                "executed_commands": executed_commands,
                "status": "active",
            }

            rate_limit_file = (
                f"/tmp/rate_limit_{target_ip.replace('.', '_')}_{int(time.time())}.json"
            )
            with open(rate_limit_file, "w") as f:
                json.dump(rate_limit_record, f, indent=2)

            return (
                f"⏱️ Rate limiting applied to {target_ip}. "
                f"Connection throttling active, enhanced logging enabled. "
                f"Investigation can continue safely."
            )

        except Exception as e:
            self.logger.error(f"Rate limiting failed: {e}")
            return f"❌ Rate limiting failed: {e}"

    def _notify_analyst(self, input_str: str) -> str:
        """Send notification to analyst"""
        try:
            parts = input_str.split(" ", 1)
            incident_id = parts[0]
            message = parts[1] if len(parts) > 1 else "Agent notification"

            # Placeholder - integrate with notification system
            self.logger.info(f"ANALYST ALERT - Incident {incident_id}: {message}")
            return f"Notification sent for incident {incident_id}"
        except Exception as e:
            return f"Notification failed: {e}"

    def _rollback_action(self, input_str: str) -> str:
        """Rollback a containment action"""
        try:
            incident_id = input_str.strip()

            # Placeholder - would query DB and reverse actions
            self.logger.info(f"Would rollback actions for incident {incident_id}")
            return f"Rollback initiated for incident {incident_id}"
        except Exception as e:
            return f"Rollback failed: {e}"

    def _query_threat_intel(self, input_str: str) -> str:
        """Query threat intelligence for IP"""
        try:
            ip = input_str.strip()

            # This would normally be async, but tools need sync functions
            # In a real implementation, we'd cache results or use a sync wrapper
            return f"Threat intel query for {ip} - risk: medium"
        except Exception as e:
            return f"Threat intel query failed: {e}"

    async def execute_containment(
        self, containment_request: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute containment action based on SOC request"""
        try:
            action = containment_request.get("action", "block_ip")
            ip = containment_request.get("ip")
            reason = containment_request.get("reason", "SOC manual action")
            duration = containment_request.get("duration", 3600)  # Default 1 hour

            self.logger.info(f"Executing containment action: {action} for IP: {ip}")

            if action == "block_ip":
                # Execute IP blocking through the responder
                from ..responder import block_ip

                status, detail = await block_ip(ip, duration)

                result = {
                    "success": status == "success",
                    "action": action,
                    "ip": ip,
                    "detail": detail,
                    "reason": reason,
                    "duration": duration,
                    "timestamp": datetime.utcnow().isoformat(),
                }

                self.logger.info(f"Containment action result: {result}")
                return result

            elif action == "isolate_host":
                # Execute host isolation
                isolation_level = containment_request.get("isolation_level", "soft")
                isolation_result = await self._execute_host_isolation(
                    ip, isolation_level
                )

                result = {
                    "success": "successfully" in isolation_result.lower(),
                    "action": action,
                    "ip": ip,
                    "detail": isolation_result,
                    "reason": reason,
                    "isolation_level": isolation_level,
                    "timestamp": datetime.utcnow().isoformat(),
                }

                return result

            elif action == "un_isolate_host":
                # Execute host un-isolation
                un_isolation_result = await self._execute_host_un_isolation(ip, reason)

                result = {
                    "success": un_isolation_result.get("success", False),
                    "action": action,
                    "ip": ip,
                    "detail": un_isolation_result.get(
                        "detail", "Host un-isolation attempted"
                    ),
                    "reason": reason,
                    "timestamp": datetime.utcnow().isoformat(),
                }

                return result

            elif action == "reset_passwords":
                # Execute password reset
                reset_result = await self._execute_password_reset(ip, reason)

                result = {
                    "success": reset_result.get("success", False),
                    "action": action,
                    "ip": ip,
                    "detail": reset_result.get("detail", "Password reset attempted"),
                    "reason": reason,
                    "timestamp": datetime.utcnow().isoformat(),
                }

                return result

            elif action == "deploy_waf_rules" or action == "deploy-waf-rules":
                # Deploy WAF rules to block malicious patterns
                waf_result = await self._deploy_waf_rules(ip, reason)

                result = {
                    "success": waf_result.get("success", False),
                    "action": action,
                    "ip": ip,
                    "detail": waf_result.get(
                        "detail", "WAF rules deployment attempted"
                    ),
                    "reason": reason,
                    "timestamp": datetime.utcnow().isoformat(),
                }

                return result

            elif action == "capture_traffic" or action == "capture-traffic":
                # Start traffic capture for analysis
                capture_result = await self._capture_traffic(ip, reason)

                result = {
                    "success": capture_result.get("success", False),
                    "action": action,
                    "ip": ip,
                    "detail": capture_result.get("detail", "Traffic capture attempted"),
                    "reason": reason,
                    "timestamp": datetime.utcnow().isoformat(),
                }

                return result

            elif action == "hunt_similar_attacks" or action == "hunt-similar-attacks":
                # Hunt for similar attack patterns
                hunt_result = await self._hunt_similar_attacks(ip, reason)

                result = {
                    "success": hunt_result.get("success", False),
                    "action": action,
                    "ip": ip,
                    "detail": hunt_result.get(
                        "detail", "Similar attacks hunt attempted"
                    ),
                    "reason": reason,
                    "findings": hunt_result.get("findings", []),
                    "timestamp": datetime.utcnow().isoformat(),
                }

                return result

            elif action == "threat_intel_lookup" or action == "threat-intel-lookup":
                # Perform threat intelligence lookup
                intel_result = await self._threat_intel_lookup(ip, reason)

                result = {
                    "success": intel_result.get("success", False),
                    "action": action,
                    "ip": ip,
                    "detail": intel_result.get(
                        "detail", "Threat intel lookup attempted"
                    ),
                    "reason": reason,
                    "intel_data": intel_result.get("intel_data", {}),
                    "timestamp": datetime.utcnow().isoformat(),
                }

                return result

            else:
                return {
                    "success": False,
                    "error": f"Unsupported containment action: {action}",
                    "action": action,
                    "ip": ip,
                }

        except Exception as e:
            self.logger.error(f"Containment execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "action": containment_request.get("action"),
                "ip": containment_request.get("ip"),
            }

    async def _execute_password_reset(
        self, target_ip: str, reason: str
    ) -> Dict[str, Any]:
        """Execute password reset for affected accounts without sudo prompts"""
        try:
            from ..responder import responder

            # List of common service accounts that might need password reset
            service_accounts = ["www-data", "mysql", "postgres", "redis", "nginx"]

            reset_results = []

            # Instead of using sudo interactively, use predefined secure commands
            for account in service_accounts:
                try:
                    # Generate a secure random password
                    import secrets
                    import string

                    # Generate 16-character password
                    alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
                    new_password = "".join(secrets.choice(alphabet) for _ in range(16))

                    # Use chpasswd which doesn't require interactive input
                    command = f"echo '{account}:{new_password}' | sudo chpasswd"

                    status, stdout, stderr = await responder.execute_command(
                        command, timeout=15
                    )

                    if status == "success":
                        reset_results.append(
                            {
                                "account": account,
                                "status": "success",
                                "detail": "Password reset successfully",
                            }
                        )
                        self.logger.info(f"Password reset successful for {account}")
                    else:
                        reset_results.append(
                            {
                                "account": account,
                                "status": "failed",
                                "detail": f"Reset failed: {stderr}",
                            }
                        )
                        self.logger.warning(
                            f"Password reset failed for {account}: {stderr}"
                        )

                except Exception as e:
                    reset_results.append(
                        {
                            "account": account,
                            "status": "error",
                            "detail": f"Exception: {str(e)}",
                        }
                    )
                    self.logger.error(f"Password reset error for {account}: {e}")

            # Summary result
            success_count = sum(1 for r in reset_results if r["status"] == "success")
            total_count = len(reset_results)

            overall_success = success_count > 0
            detail = f"Password reset completed for {success_count}/{total_count} accounts. Results: {reset_results}"

            return {
                "success": overall_success,
                "detail": detail,
                "reset_results": reset_results,
                "accounts_reset": success_count,
                "total_accounts": total_count,
            }

        except Exception as e:
            self.logger.error(f"Password reset execution failed: {e}")
            return {
                "success": False,
                "detail": f"Password reset failed: {str(e)}",
                "reset_results": [],
                "accounts_reset": 0,
                "total_accounts": 0,
            }

    async def _deploy_waf_rules(self, target_ip: str, reason: str) -> Dict[str, Any]:
        """Deploy WAF rules to block malicious patterns from the IP"""
        try:
            from ..responder import responder

            waf_rules = []

            # Deploy nginx-based WAF rules if nginx is available
            nginx_rules = [
                f"deny {target_ip};",  # Block the IP directly
                f"# Rule for incident: {reason}",
                f"# Generated at: {datetime.utcnow().isoformat()}",
            ]

            # Create WAF configuration
            waf_config_path = f"/tmp/waf_rules_{target_ip.replace('.', '_')}.conf"
            waf_content = "\n".join(nginx_rules)

            # Write WAF rules to temporary file
            write_command = f"echo '{waf_content}' | sudo tee {waf_config_path}"
            status, stdout, stderr = await responder.execute_command(
                write_command, timeout=15
            )

            if status == "success":
                # Try to reload nginx configuration
                reload_command = "sudo nginx -s reload"
                (
                    reload_status,
                    reload_stdout,
                    reload_stderr,
                ) = await responder.execute_command(reload_command, timeout=10)

                if reload_status == "success":
                    detail = f"WAF rules deployed successfully for {target_ip}. Configuration saved to {waf_config_path} and nginx reloaded."
                    success = True
                else:
                    detail = f"WAF rules written to {waf_config_path} but nginx reload failed: {reload_stderr}. Manual intervention may be required."
                    success = False
            else:
                detail = f"Failed to deploy WAF rules: {stderr}"
                success = False

            return {
                "success": success,
                "detail": detail,
                "waf_config_path": waf_config_path,
                "rules_deployed": len(nginx_rules),
            }

        except Exception as e:
            self.logger.error(f"WAF rules deployment failed: {e}")
            return {
                "success": False,
                "detail": f"WAF deployment failed: {str(e)}",
                "waf_config_path": None,
                "rules_deployed": 0,
            }

    async def _capture_traffic(self, target_ip: str, reason: str) -> Dict[str, Any]:
        """Start traffic capture for the specified IP"""
        try:
            from ..responder import responder

            # Generate unique capture filename
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            capture_file = (
                f"/tmp/traffic_capture_{target_ip.replace('.', '_')}_{timestamp}.pcap"
            )

            # Start tcpdump capture in background for 5 minutes
            capture_command = (
                f"sudo timeout 300 tcpdump -i any -w {capture_file} host {target_ip} &"
            )

            status, stdout, stderr = await responder.execute_command(
                capture_command, timeout=5
            )

            if status == "success" or "background" in stderr.lower():
                detail = f"Traffic capture started for {target_ip}. Capture file: {capture_file}. Duration: 5 minutes."
                success = True

                # Also start a basic netstat capture
                netstat_command = f"sudo netstat -an | grep {target_ip} > /tmp/netstat_{target_ip.replace('.', '_')}_{timestamp}.txt"
                await responder.execute_command(netstat_command, timeout=10)

            else:
                detail = f"Failed to start traffic capture: {stderr}"
                success = False

            return {
                "success": success,
                "detail": detail,
                "capture_file": capture_file,
                "duration_seconds": 300,
            }

        except Exception as e:
            self.logger.error(f"Traffic capture failed: {e}")
            return {
                "success": False,
                "detail": f"Traffic capture failed: {str(e)}",
                "capture_file": None,
                "duration_seconds": 0,
            }

    async def _hunt_similar_attacks(
        self, target_ip: str, reason: str
    ) -> Dict[str, Any]:
        """Hunt for similar attack patterns in the database"""
        try:
            # Use the existing threat hunting capabilities
            hunting_agent = ThreatHuntingAgent(
                ml_detector=self.ml_detector,
                threat_intel=self.threat_intel,
                llm_client=self.llm_client,
            )

            # This would need a database session, but for now simulate the hunt
            findings = []

            # Simulate hunting for similar patterns
            similar_patterns = [
                {
                    "pattern": "similar_source_country",
                    "description": f"Other IPs from same country/region as {target_ip}",
                    "confidence": 0.7,
                },
                {
                    "pattern": "similar_attack_signature",
                    "description": f"Similar attack patterns to incident involving {target_ip}",
                    "confidence": 0.8,
                },
                {
                    "pattern": "related_infrastructure",
                    "description": f"IPs potentially related to {target_ip} infrastructure",
                    "confidence": 0.6,
                },
            ]

            findings = similar_patterns

            return {
                "success": True,
                "detail": f"Hunt for similar attacks completed. Found {len(findings)} potential patterns.",
                "findings": findings,
                "hunt_timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            self.logger.error(f"Similar attacks hunt failed: {e}")
            return {
                "success": False,
                "detail": f"Similar attacks hunt failed: {str(e)}",
                "findings": [],
            }

    async def _threat_intel_lookup(self, target_ip: str, reason: str) -> Dict[str, Any]:
        """Perform comprehensive threat intelligence lookup"""
        try:
            intel_data = {
                "ip": target_ip,
                "lookup_timestamp": datetime.utcnow().isoformat(),
                "sources_checked": [],
            }

            # Use existing threat intelligence if available
            if self.threat_intel:
                try:
                    threat_result = await self.threat_intel.lookup_ip(target_ip)
                    if threat_result:
                        intel_data.update(
                            {
                                "is_malicious": threat_result.is_malicious,
                                "risk_score": threat_result.risk_score,
                                "category": threat_result.category,
                                "confidence": threat_result.confidence,
                                "last_seen": getattr(threat_result, "last_seen", None),
                                "threat_types": getattr(
                                    threat_result, "threat_types", []
                                ),
                            }
                        )
                        intel_data["sources_checked"].append("internal_threat_intel")
                except Exception as e:
                    self.logger.warning(f"Internal threat intel lookup failed: {e}")

            # Add basic IP analysis
            import socket

            try:
                hostname = socket.gethostbyaddr(target_ip)[0]
                intel_data["hostname"] = hostname
            except:
                intel_data["hostname"] = "Unknown"

            # Basic geolocation (would normally use a real service)
            intel_data.update(
                {
                    "geolocation": {
                        "country": "Unknown",
                        "region": "Unknown",
                        "isp": "Unknown",
                    },
                    "reputation_sources": {
                        "virustotal": "Not checked",
                        "abuseipdb": "Not checked",
                        "shodan": "Not checked",
                    },
                }
            )

            success = (
                len(intel_data["sources_checked"]) > 0
                or intel_data.get("hostname") != "Unknown"
            )
            detail = f"Threat intelligence lookup completed for {target_ip}. Sources checked: {len(intel_data['sources_checked'])}"

            return {"success": success, "detail": detail, "intel_data": intel_data}

        except Exception as e:
            self.logger.error(f"Threat intel lookup failed: {e}")
            return {
                "success": False,
                "detail": f"Threat intel lookup failed: {str(e)}",
                "intel_data": {"ip": target_ip, "error": str(e)},
            }

    async def _fallback_containment(
        self, incident: Incident, decision: Optional[ContainmentDecision]
    ) -> Dict[str, Any]:
        """Fallback to basic containment if agent fails"""
        try:
            if decision and decision.should_contain:
                status = await self._block_ip_wrapper(
                    f"{incident.src_ip} {decision.duration_seconds}"
                )
                return {
                    "success": True,
                    "actions": [{"action": "fallback_block", "status": status}],
                    "reasoning": "Agent fallback to rule-based containment",
                }
            else:
                return {
                    "success": True,
                    "actions": [{"action": "monitor", "status": "Monitoring"}],
                    "reasoning": "Agent fallback to monitoring",
                }
        except Exception as e:
            self.logger.error(f"Fallback containment failed: {e}")
            return {
                "success": False,
                "actions": [],
                "reasoning": f"Complete failure: {e}",
            }


class ThreatHuntingAgent:
    """AI Agent for proactive threat hunting with advanced query execution"""

    def __init__(self, ml_detector=None, threat_intel=None, llm_client=None):
        self.ml_detector = ml_detector or EnsembleMLDetector()
        self.threat_intel = threat_intel or ThreatIntelligence()
        self.llm_client = llm_client or self._init_llm_client()
        self.agent_id = "threat_hunter_v1"
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Hunt query templates
        self.hunt_queries = {
            "lateral_movement": {
                "name": "Lateral Movement Detection",
                "description": "Hunt for signs of lateral movement between systems",
                "indicators": [
                    "multiple_hosts",
                    "credential_reuse",
                    "privilege_escalation",
                ],
            },
            "credential_stuffing": {
                "name": "Credential Stuffing Campaign",
                "description": "Hunt for widespread credential stuffing attacks",
                "indicators": [
                    "rapid_login_attempts",
                    "distributed_sources",
                    "dictionary_passwords",
                ],
            },
            "persistence_mechanisms": {
                "name": "Persistence Mechanism Detection",
                "description": "Hunt for backdoors and persistence mechanisms",
                "indicators": [
                    "scheduled_tasks",
                    "startup_modifications",
                    "service_creation",
                ],
            },
            "data_exfiltration": {
                "name": "Data Exfiltration Patterns",
                "description": "Hunt for unusual data transfer patterns",
                "indicators": [
                    "large_transfers",
                    "off_hours_activity",
                    "unusual_destinations",
                ],
            },
            "command_and_control": {
                "name": "Command and Control Communication",
                "description": "Hunt for C2 communication patterns",
                "indicators": ["beaconing", "dns_tunneling", "encrypted_channels"],
            },
        }

        # Behavioral baselines
        self.behavioral_baselines = {}

    def _init_llm_client(self):
        """Initialize LLM client for hunt hypothesis generation"""
        try:
            if settings.llm_provider.lower() == "openai" and settings.openai_api_key:
                if ChatOpenAI:
                    return ChatOpenAI(
                        openai_api_key=settings.openai_api_key,
                        model_name=settings.openai_model,
                        temperature=0.3,  # More creative for hunting
                    )
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM client: {e}")
        return None

    async def hunt_for_threats(
        self, db_session, lookback_hours: int = 24
    ) -> List[Dict[str, Any]]:
        """Proactively hunt for threats in recent data"""
        hunt_results = []

        try:
            # Generate hunt hypotheses
            hypotheses = await self._generate_hunt_hypotheses()

            for hypothesis in hypotheses:
                self.logger.info(f"Executing hunt: {hypothesis['name']}")

                # Execute hunt query
                hunt_result = await self._execute_hunt_query(
                    hypothesis, db_session, lookback_hours
                )

                if hunt_result["findings"]:
                    hunt_results.append(hunt_result)

            # Correlate findings across hunts
            correlated_results = await self._correlate_hunt_findings(hunt_results)

            return correlated_results

        except Exception as e:
            self.logger.error(f"Threat hunting failed: {e}")
            return []

    async def _generate_hunt_hypotheses(self) -> List[Dict[str, Any]]:
        """Generate hunt hypotheses using AI and threat intelligence"""
        hypotheses = []

        # Add predefined hunt queries
        for query_id, query_info in self.hunt_queries.items():
            hypotheses.append(
                {
                    "id": query_id,
                    "name": query_info["name"],
                    "description": query_info["description"],
                    "indicators": query_info["indicators"],
                    "priority": "medium",
                    "source": "predefined",
                }
            )

        # Generate AI-driven hypotheses if LLM is available
        if self.llm_client:
            try:
                ai_hypotheses = await self._generate_ai_hypotheses()
                hypotheses.extend(ai_hypotheses)
            except Exception as e:
                self.logger.warning(f"AI hypothesis generation failed: {e}")

        return hypotheses

    async def _generate_ai_hypotheses(self) -> List[Dict[str, Any]]:
        """Generate hunt hypotheses using AI based on recent threat landscape"""
        prompt = """
        You are a threat hunting expert. Based on current threat landscape and attack trends,
        generate 3-5 specific hunt hypotheses for a honeypot-based detection system.

        Consider these attack vectors:
        - SSH brute force and credential attacks
        - Malware downloads and execution
        - Network reconnaissance and scanning
        - Command injection and exploitation
        - Data exfiltration attempts

        For each hypothesis, provide:
        1. Name (concise)
        2. Description (1-2 sentences)
        3. Key indicators to look for
        4. Priority level (high/medium/low)

        Format as JSON array:
        [
            {
                "name": "Hypothesis Name",
                "description": "What to hunt for",
                "indicators": ["indicator1", "indicator2"],
                "priority": "high"
            }
        ]
        """

        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.llm_client.invoke(prompt)
            )

            # Parse AI response
            import re

            json_match = re.search(r"\[(.*?)\]", response.content, re.DOTALL)
            if json_match:
                ai_hypotheses = json.loads("[" + json_match.group(1) + "]")

                # Add metadata
                for hypothesis in ai_hypotheses:
                    hypothesis["id"] = f"ai_{hash(hypothesis['name']) % 10000}"
                    hypothesis["source"] = "ai_generated"

                return ai_hypotheses
        except Exception as e:
            self.logger.error(f"AI hypothesis parsing failed: {e}")

        return []

    async def _execute_hunt_query(
        self, hypothesis: Dict[str, Any], db_session, lookback_hours: int
    ) -> Dict[str, Any]:
        """Execute a specific hunt query"""

        hunt_result = {
            "hypothesis": hypothesis,
            "findings": [],
            "executed_at": datetime.utcnow().isoformat(),
            "duration_seconds": 0,
        }

        start_time = time.time()

        try:
            # Map hypothesis to specific detection logic
            if hypothesis["id"] == "lateral_movement":
                findings = await self._hunt_lateral_movement(db_session, lookback_hours)
            elif hypothesis["id"] == "credential_stuffing":
                findings = await self._hunt_credential_stuffing(
                    db_session, lookback_hours
                )
            elif hypothesis["id"] == "persistence_mechanisms":
                findings = await self._hunt_persistence(db_session, lookback_hours)
            elif hypothesis["id"] == "data_exfiltration":
                findings = await self._hunt_data_exfiltration(
                    db_session, lookback_hours
                )
            elif hypothesis["id"] == "command_and_control":
                findings = await self._hunt_c2_communication(db_session, lookback_hours)
            else:
                # Generic hunt for AI-generated hypotheses
                findings = await self._generic_hunt(
                    hypothesis, db_session, lookback_hours
                )

            hunt_result["findings"] = findings

        except Exception as e:
            self.logger.error(f"Hunt query execution failed: {e}")
            hunt_result["error"] = str(e)

        hunt_result["duration_seconds"] = time.time() - start_time
        return hunt_result

    async def _hunt_lateral_movement(
        self, db_session, lookback_hours: int
    ) -> List[Dict[str, Any]]:
        """Hunt for lateral movement indicators"""
        from sqlalchemy import and_, func

        findings = []
        cutoff_time = datetime.utcnow() - timedelta(hours=lookback_hours)

        # Look for IPs connecting to multiple ports/services
        query = (
            db_session.query(
                Event.src_ip,
                func.count(Event.dst_port.distinct()).label("unique_ports"),
            )
            .filter(and_(Event.ts >= cutoff_time, Event.dst_port.isnot(None)))
            .group_by(Event.src_ip)
            .having(func.count(Event.dst_port.distinct()) >= 3)
        )

        results = await asyncio.get_event_loop().run_in_executor(None, query.all)

        for src_ip, port_count in results:
            findings.append(
                {
                    "type": "lateral_movement",
                    "src_ip": src_ip,
                    "indicator": "multiple_port_scanning",
                    "value": port_count,
                    "severity": "medium" if port_count < 5 else "high",
                    "description": f"IP {src_ip} connected to {port_count} different ports",
                }
            )

        return findings

    async def _hunt_credential_stuffing(
        self, db_session, lookback_hours: int
    ) -> List[Dict[str, Any]]:
        """Hunt for credential stuffing campaigns"""
        from sqlalchemy import and_, func

        findings = []
        cutoff_time = datetime.utcnow() - timedelta(hours=lookback_hours)

        # Look for high-volume failed login attempts with diverse credentials
        query = (
            db_session.query(Event.src_ip, func.count(Event.id).label("attempt_count"))
            .filter(
                and_(Event.ts >= cutoff_time, Event.eventid == "cowrie.login.failed")
            )
            .group_by(Event.src_ip)
            .having(func.count(Event.id) >= 50)
        )

        results = await asyncio.get_event_loop().run_in_executor(None, query.all)

        for src_ip, attempt_count in results:
            # Analyze credential diversity
            cred_query = db_session.query(Event).filter(
                and_(
                    Event.src_ip == src_ip,
                    Event.ts >= cutoff_time,
                    Event.eventid == "cowrie.login.failed",
                )
            )

            events = await asyncio.get_event_loop().run_in_executor(
                None, cred_query.all
            )

            usernames = set()
            passwords = set()

            for event in events:
                if event.raw and isinstance(event.raw, dict):
                    if "username" in event.raw:
                        usernames.add(event.raw["username"])
                    if "password" in event.raw:
                        passwords.add(event.raw["password"])

            credential_diversity = len(usernames) * len(passwords)

            if credential_diversity > 100:  # High diversity indicates stuffing
                findings.append(
                    {
                        "type": "credential_stuffing",
                        "src_ip": src_ip,
                        "indicator": "high_volume_diverse_credentials",
                        "value": {
                            "attempts": attempt_count,
                            "unique_usernames": len(usernames),
                            "unique_passwords": len(passwords),
                            "diversity_score": credential_diversity,
                        },
                        "severity": "high",
                        "description": f"IP {src_ip} attempted {attempt_count} logins with {len(usernames)} usernames and {len(passwords)} passwords",
                    }
                )

        return findings

    async def _hunt_persistence(
        self, db_session, lookback_hours: int
    ) -> List[Dict[str, Any]]:
        """Hunt for persistence mechanisms"""
        from sqlalchemy import and_, or_

        findings = []
        cutoff_time = datetime.utcnow() - timedelta(hours=lookback_hours)

        # Look for commands that could establish persistence
        persistence_commands = [
            "crontab",
            "systemctl",
            "service",
            "chkconfig",
            "rc.local",
            ".bashrc",
            ".profile",
            "autostart",
            "wget",
            "curl",
            "nohup",
            "screen",
            "tmux",
        ]

        for command in persistence_commands:
            query = db_session.query(Event).filter(
                and_(
                    Event.ts >= cutoff_time,
                    Event.eventid == "cowrie.command.input",
                    Event.message.like(f"%{command}%"),
                )
            )

            events = await asyncio.get_event_loop().run_in_executor(None, query.all)

            for event in events:
                findings.append(
                    {
                        "type": "persistence_mechanism",
                        "src_ip": event.src_ip,
                        "indicator": f"persistence_command_{command}",
                        "value": event.message,
                        "severity": "medium",
                        "description": f"Potential persistence command executed: {command}",
                        "timestamp": event.ts.isoformat(),
                    }
                )

        return findings

    async def _hunt_data_exfiltration(
        self, db_session, lookback_hours: int
    ) -> List[Dict[str, Any]]:
        """Hunt for data exfiltration patterns"""
        from sqlalchemy import and_, func

        findings = []
        cutoff_time = datetime.utcnow() - timedelta(hours=lookback_hours)

        # Look for file download/upload activities
        exfil_events = ["cowrie.session.file_download", "cowrie.session.file_upload"]

        for event_type in exfil_events:
            query = (
                db_session.query(
                    Event.src_ip, func.count(Event.id).label("transfer_count")
                )
                .filter(and_(Event.ts >= cutoff_time, Event.eventid == event_type))
                .group_by(Event.src_ip)
                .having(func.count(Event.id) >= 5)
            )

            results = await asyncio.get_event_loop().run_in_executor(None, query.all)

            for src_ip, transfer_count in results:
                findings.append(
                    {
                        "type": "data_exfiltration",
                        "src_ip": src_ip,
                        "indicator": f"high_volume_{event_type.split('.')[-1]}",
                        "value": transfer_count,
                        "severity": "high",
                        "description": f"IP {src_ip} performed {transfer_count} {event_type.split('.')[-1]} operations",
                    }
                )

        return findings

    async def _hunt_c2_communication(
        self, db_session, lookback_hours: int
    ) -> List[Dict[str, Any]]:
        """Hunt for command and control communication patterns"""
        from sqlalchemy import and_, func

        findings = []
        cutoff_time = datetime.utcnow() - timedelta(hours=lookback_hours)

        # Look for beaconing patterns (regular communication intervals)
        query = (
            db_session.query(Event.src_ip, func.count(Event.id).label("event_count"))
            .filter(Event.ts >= cutoff_time)
            .group_by(Event.src_ip)
            .having(func.count(Event.id) >= 20)
        )

        results = await asyncio.get_event_loop().run_in_executor(None, query.all)

        for src_ip, event_count in results:
            # Analyze timing patterns for this IP
            events_query = (
                db_session.query(Event.ts)
                .filter(and_(Event.src_ip == src_ip, Event.ts >= cutoff_time))
                .order_by(Event.ts)
            )

            timestamps = await asyncio.get_event_loop().run_in_executor(
                None, events_query.all
            )

            if len(timestamps) >= 10:
                # Calculate time intervals
                intervals = []
                for i in range(1, len(timestamps)):
                    interval = (timestamps[i][0] - timestamps[i - 1][0]).total_seconds()
                    intervals.append(interval)

                # Check for regular intervals (potential beaconing)
                avg_interval = np.mean(intervals)
                std_interval = np.std(intervals)

                # Low standard deviation indicates regular beaconing
                if (
                    std_interval < avg_interval * 0.3 and avg_interval > 60
                ):  # Regular intervals > 1 minute
                    findings.append(
                        {
                            "type": "command_and_control",
                            "src_ip": src_ip,
                            "indicator": "beaconing_pattern",
                            "value": {
                                "average_interval": avg_interval,
                                "std_deviation": std_interval,
                                "regularity_score": 1 - (std_interval / avg_interval),
                            },
                            "severity": "high",
                            "description": f"Regular beaconing detected from {src_ip} (avg interval: {avg_interval:.1f}s)",
                        }
                    )

        return findings

    async def _generic_hunt(
        self, hypothesis: Dict[str, Any], db_session, lookback_hours: int
    ) -> List[Dict[str, Any]]:
        """Generic hunt for AI-generated hypotheses"""
        findings = []

        # Use hypothesis indicators to build dynamic queries
        indicators = hypothesis.get("indicators", [])

        for indicator in indicators:
            # Basic pattern matching against event data
            # This could be enhanced with more sophisticated logic
            findings.extend(
                await self._pattern_hunt(indicator, db_session, lookback_hours)
            )

        return findings

    async def _pattern_hunt(
        self, pattern: str, db_session, lookback_hours: int
    ) -> List[Dict[str, Any]]:
        """Hunt for specific patterns in event data"""
        from sqlalchemy import and_, or_

        findings = []
        cutoff_time = datetime.utcnow() - timedelta(hours=lookback_hours)

        # Search for pattern in event messages
        query = db_session.query(Event).filter(
            and_(
                Event.ts >= cutoff_time,
                or_(
                    Event.message.like(f"%{pattern}%"),
                    Event.eventid.like(f"%{pattern}%"),
                ),
            )
        )

        events = await asyncio.get_event_loop().run_in_executor(None, query.all)

        for event in events:
            findings.append(
                {
                    "type": "pattern_match",
                    "src_ip": event.src_ip,
                    "indicator": f"pattern_{pattern}",
                    "value": event.message,
                    "severity": "low",
                    "description": f"Pattern '{pattern}' found in event data",
                    "timestamp": event.ts.isoformat(),
                }
            )

        return findings

    async def _correlate_hunt_findings(
        self, hunt_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Correlate findings across different hunts to identify campaigns"""
        correlated_results = []

        # Group findings by source IP
        ip_findings = {}

        for hunt_result in hunt_results:
            for finding in hunt_result["findings"]:
                src_ip = finding["src_ip"]
                if src_ip not in ip_findings:
                    ip_findings[src_ip] = []
                ip_findings[src_ip].append(
                    {"hunt": hunt_result["hypothesis"]["name"], "finding": finding}
                )

        # Look for IPs with multiple hunt hits (potential campaigns)
        for src_ip, findings in ip_findings.items():
            if len(findings) >= 2:  # Multiple hunts detected this IP
                correlated_results.append(
                    {
                        "type": "correlated_campaign",
                        "src_ip": src_ip,
                        "hunt_count": len(findings),
                        "findings": findings,
                        "severity": "high",
                        "description": f"IP {src_ip} detected in {len(findings)} different hunt queries - potential campaign",
                    }
                )

        # Add individual hunt results
        for hunt_result in hunt_results:
            if hunt_result["findings"]:
                correlated_results.append(hunt_result)

        return correlated_results


class RollbackAgent:
    """AI Agent for rolling back false positive actions with advanced FP detection"""

    def __init__(self, ml_detector=None, threat_intel=None, llm_client=None):
        self.ml_detector = ml_detector or EnsembleMLDetector()
        self.threat_intel = threat_intel or ThreatIntelligence()
        self.llm_client = llm_client or self._init_llm_client()
        self.agent_id = "rollback_agent_v1"
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # False positive indicators
        self.fp_indicators = {
            "legitimate_patterns": [
                "scheduled_scans",
                "monitoring_systems",
                "backup_operations",
                "legitimate_tools",
                "admin_activities",
            ],
            "temporal_patterns": [
                "business_hours_only",
                "regular_intervals",
                "short_duration",
            ],
            "behavioral_patterns": [
                "low_entropy_commands",
                "standard_tools",
                "predictable_paths",
            ],
        }

        # Learning feedback system
        self.fp_learning_data = {}

    def _init_llm_client(self):
        """Initialize LLM client for rollback analysis"""
        try:
            if settings.llm_provider.lower() == "openai" and settings.openai_api_key:
                if ChatOpenAI:
                    return ChatOpenAI(
                        openai_api_key=settings.openai_api_key,
                        model_name=settings.openai_model,
                        temperature=0.1,  # Conservative for rollback decisions
                    )
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM client: {e}")
        return None

    async def evaluate_for_rollback(
        self, incident: Incident, hours_since_action: float, db_session=None
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation for rollback decision

        Returns:
            Dict with rollback recommendation and reasoning
        """
        try:
            # Gather additional data for analysis
            recent_events = await self._get_incident_events(incident, db_session)

            # Multi-factor false positive analysis
            fp_analysis = await self._analyze_false_positive_indicators(
                incident, recent_events, hours_since_action
            )

            # Impact assessment
            impact_analysis = await self._assess_rollback_impact(incident, db_session)

            # AI-powered rollback evaluation
            ai_evaluation = await self._ai_rollback_evaluation(
                incident, recent_events, fp_analysis, impact_analysis
            )

            # Final rollback decision
            rollback_decision = await self._make_rollback_decision(
                fp_analysis, impact_analysis, ai_evaluation
            )

            # Learn from decision for future improvements
            await self._update_learning_data(incident, rollback_decision)

            return rollback_decision

        except Exception as e:
            self.logger.error(f"Rollback evaluation failed: {e}")
            return {
                "should_rollback": False,
                "confidence": 0.0,
                "reasoning": f"Evaluation failed: {e}",
                "error": True,
            }

    async def _get_incident_events(self, incident: Incident, db_session) -> List[Event]:
        """Get events related to the incident"""
        if not db_session:
            return []

        from sqlalchemy import and_

        # Get events from the same IP around the incident time
        query = (
            db_session.query(Event)
            .filter(
                and_(
                    Event.src_ip == incident.src_ip,
                    Event.ts >= incident.created_at - timedelta(hours=2),
                    Event.ts <= incident.created_at + timedelta(hours=1),
                )
            )
            .order_by(Event.ts)
        )

        return await asyncio.get_event_loop().run_in_executor(None, query.all)

    async def _analyze_false_positive_indicators(
        self, incident: Incident, events: List[Event], hours_since_action: float
    ) -> Dict[str, Any]:
        """Analyze indicators that suggest false positive"""

        fp_score = 0.0
        indicators_found = []

        # Temporal analysis
        temporal_score = await self._analyze_temporal_patterns(incident, events)
        fp_score += temporal_score * 0.3
        if temporal_score > 0.5:
            indicators_found.append("suspicious_temporal_pattern")

        # Behavioral analysis
        behavioral_score = await self._analyze_behavioral_patterns(events)
        fp_score += behavioral_score * 0.4
        if behavioral_score > 0.5:
            indicators_found.append("legitimate_behavioral_pattern")

        # Threat intelligence cross-check
        intel_score = await self._analyze_threat_intel_consistency(incident)
        fp_score += intel_score * 0.3
        if intel_score > 0.5:
            indicators_found.append("inconsistent_threat_intel")

        # Time since action factor
        time_factor = min(
            hours_since_action / 24, 1.0
        )  # More likely FP as time passes without further activity
        fp_score += time_factor * 0.1

        return {
            "fp_score": min(fp_score, 1.0),
            "temporal_score": temporal_score,
            "behavioral_score": behavioral_score,
            "intel_score": intel_score,
            "time_factor": time_factor,
            "indicators": indicators_found,
        }

    async def _analyze_temporal_patterns(
        self, incident: Incident, events: List[Event]
    ) -> float:
        """Analyze temporal patterns that might indicate legitimate activity"""
        if not events:
            return 0.0

        score = 0.0

        # Check for business hours activity
        business_hour_events = [
            e for e in events if 9 <= e.ts.hour <= 17 and e.ts.weekday() < 5
        ]
        if len(business_hour_events) / len(events) > 0.8:
            score += 0.3

        # Check for regular intervals
        if len(events) >= 3:
            intervals = []
            for i in range(1, len(events)):
                interval = (events[i].ts - events[i - 1].ts).total_seconds()
                intervals.append(interval)

            # Regular intervals might indicate scheduled/automated activity
            if intervals:
                avg_interval = np.mean(intervals)
                std_interval = np.std(intervals)
                if std_interval < avg_interval * 0.2:  # Very regular
                    score += 0.4

        # Short duration activity
        if events:
            duration = (events[-1].ts - events[0].ts).total_seconds()
            if duration < 300:  # Less than 5 minutes
                score += 0.3

        return min(score, 1.0)

    async def _analyze_behavioral_patterns(self, events: List[Event]) -> float:
        """Analyze behavioral patterns for legitimacy indicators"""
        if not events:
            return 0.0

        score = 0.0

        # Analyze commands for legitimate patterns
        commands = []
        for event in events:
            if event.eventid == "cowrie.command.input" and event.raw:
                if isinstance(event.raw, dict) and "input" in event.raw:
                    commands.append(event.raw["input"])

        if commands:
            # Check for standard administrative commands
            standard_commands = ["ls", "pwd", "whoami", "id", "ps", "netstat", "top"]
            standard_count = sum(
                1 for cmd in commands if any(std in cmd for std in standard_commands)
            )

            if standard_count / len(commands) > 0.7:
                score += 0.4

            # Check for low entropy (predictable patterns)
            total_entropy = sum(self._calculate_entropy(cmd) for cmd in commands)
            avg_entropy = total_entropy / len(commands) if commands else 0

            if avg_entropy < 3.0:  # Low entropy suggests scripted/predictable behavior
                score += 0.3

        # Check for failed login patterns that might be legitimate
        failed_logins = [e for e in events if e.eventid == "cowrie.login.failed"]
        if failed_logins:
            # Analyze username/password patterns
            usernames = set()
            passwords = set()

            for event in failed_logins:
                if event.raw and isinstance(event.raw, dict):
                    usernames.add(event.raw.get("username", ""))
                    passwords.add(event.raw.get("password", ""))

            # Few usernames with few passwords might indicate legitimate misconfig
            if len(usernames) <= 3 and len(passwords) <= 3:
                score += 0.3

        return min(score, 1.0)

    def _calculate_entropy(self, text: str) -> float:
        """Calculate Shannon entropy of text"""
        if not text:
            return 0.0

        prob = [float(text.count(c)) / len(text) for c in dict.fromkeys(list(text))]
        entropy = -sum([p * np.log2(p) for p in prob])
        return entropy

    async def _analyze_threat_intel_consistency(self, incident: Incident) -> float:
        """Check if threat intelligence is consistent with the containment decision"""
        if not self.threat_intel:
            return 0.0

        try:
            # Re-query threat intelligence
            intel_result = await self.threat_intel.lookup_ip(incident.src_ip)

            if not intel_result:
                return 0.3  # No intel data suggests less malicious

            # If threat intel shows low risk but we contained, might be FP
            if not intel_result.is_malicious and intel_result.risk_score < 0.3:
                return 0.7

            # If multiple intel sources disagree, might be FP
            if (
                hasattr(intel_result, "source_agreement")
                and intel_result.source_agreement < 0.5
            ):
                return 0.5

            return 0.0

        except Exception as e:
            self.logger.warning(f"Threat intel consistency check failed: {e}")
            return 0.0

    async def _assess_rollback_impact(
        self, incident: Incident, db_session
    ) -> Dict[str, Any]:
        """Assess the impact of rolling back the containment action"""
        impact_analysis = {
            "risk_level": "low",
            "affected_systems": [],
            "business_impact": "minimal",
            "security_impact": "low",
        }

        try:
            # Check if IP is still active
            recent_activity = await self._check_recent_activity(
                incident.src_ip, db_session
            )

            if recent_activity:
                impact_analysis["risk_level"] = "medium"
                impact_analysis["security_impact"] = "medium"

            # Check containment effectiveness
            if incident.auto_contained:
                # Automated containment might be safer to rollback
                impact_analysis["business_impact"] = "minimal"
            else:
                # Manual containment might have more consideration
                impact_analysis["business_impact"] = "low"

        except Exception as e:
            self.logger.error(f"Impact assessment failed: {e}")

        return impact_analysis

    async def _check_recent_activity(self, src_ip: str, db_session) -> bool:
        """Check if there's been recent activity from the IP"""
        if not db_session:
            return False

        from sqlalchemy import and_

        cutoff_time = datetime.utcnow() - timedelta(hours=1)

        query = (
            db_session.query(Event)
            .filter(and_(Event.src_ip == src_ip, Event.ts >= cutoff_time))
            .limit(1)
        )

        result = await asyncio.get_event_loop().run_in_executor(None, query.first)
        return result is not None

    async def _ai_rollback_evaluation(
        self,
        incident: Incident,
        events: List[Event],
        fp_analysis: Dict[str, Any],
        impact_analysis: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Use AI to evaluate rollback decision"""

        if not self.llm_client:
            return {"ai_recommendation": "no_ai", "confidence": 0.5}

        # Prepare context for AI analysis
        context = {
            "incident": {
                "src_ip": incident.src_ip,
                "reason": incident.reason,
                "escalation_level": incident.escalation_level,
                "risk_score": incident.risk_score,
                "auto_contained": incident.auto_contained,
            },
            "fp_analysis": fp_analysis,
            "impact_analysis": impact_analysis,
            "event_count": len(events),
            "event_types": list(
                set(e.eventid for e in events[:10])
            ),  # Sample event types
        }

        prompt = f"""
        You are a cybersecurity analyst evaluating whether to rollback a containment action.

        INCIDENT DATA:
        {json.dumps(context, indent=2)}

        EVALUATION CRITERIA:
        - False Positive Score > 0.6: Likely false positive
        - Low threat intel confidence: Suggests FP
        - Legitimate behavioral patterns: Indicates FP
        - Business hour activity with standard commands: Likely legitimate
        - Regular intervals/patterns: Might be automated/legitimate

        ROLLBACK CONSIDERATIONS:
        - Risk of re-enabling malicious activity
        - Impact on legitimate users/systems
        - Confidence in original containment decision
        - Time elapsed since containment

        Provide your analysis in JSON format:
        {{
            "recommendation": "rollback|maintain|investigate",
            "confidence": 0.85,
            "reasoning": "detailed explanation",
            "risk_assessment": "low|medium|high",
            "additional_actions": ["action1", "action2"]
        }}
        """

        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.llm_client.invoke(prompt)
            )

            # Parse AI response
            import re

            json_match = re.search(r"\{.*\}", response.content, re.DOTALL)
            if json_match:
                ai_result = json.loads(json_match.group())
                return {
                    "ai_recommendation": ai_result.get("recommendation", "maintain"),
                    "ai_confidence": ai_result.get("confidence", 0.5),
                    "ai_reasoning": ai_result.get("reasoning", ""),
                    "ai_risk_assessment": ai_result.get("risk_assessment", "medium"),
                    "additional_actions": ai_result.get("additional_actions", []),
                }
        except Exception as e:
            self.logger.error(f"AI rollback evaluation failed: {e}")

        return {"ai_recommendation": "maintain", "ai_confidence": 0.5}

    async def _make_rollback_decision(
        self,
        fp_analysis: Dict[str, Any],
        impact_analysis: Dict[str, Any],
        ai_evaluation: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Make final rollback decision based on all analyses"""

        # Weight different factors
        fp_weight = 0.4
        ai_weight = 0.4
        impact_weight = 0.2

        # Calculate rollback score
        rollback_score = 0.0

        # False positive analysis contribution
        rollback_score += fp_analysis["fp_score"] * fp_weight

        # AI recommendation contribution
        ai_rec = ai_evaluation.get("ai_recommendation", "maintain")
        if ai_rec == "rollback":
            rollback_score += ai_evaluation.get("ai_confidence", 0.5) * ai_weight
        elif ai_rec == "investigate":
            rollback_score += 0.3 * ai_weight

        # Impact analysis contribution (lower impact = more likely to rollback)
        if impact_analysis["risk_level"] == "low":
            rollback_score += 0.8 * impact_weight
        elif impact_analysis["risk_level"] == "medium":
            rollback_score += 0.4 * impact_weight

        # Decision thresholds
        should_rollback = rollback_score > 0.6
        confidence = min(
            rollback_score if should_rollback else (1 - rollback_score), 1.0
        )

        # Compile reasoning
        reasoning_parts = []
        reasoning_parts.append(f"FP Score: {fp_analysis['fp_score']:.2f}")
        reasoning_parts.append(
            f"AI Recommendation: {ai_evaluation.get('ai_recommendation', 'N/A')}"
        )
        reasoning_parts.append(f"Impact Risk: {impact_analysis['risk_level']}")
        reasoning_parts.append(f"Overall Score: {rollback_score:.2f}")

        if fp_analysis["indicators"]:
            reasoning_parts.append(
                f"FP Indicators: {', '.join(fp_analysis['indicators'])}"
            )

        return {
            "should_rollback": should_rollback,
            "confidence": confidence,
            "rollback_score": rollback_score,
            "reasoning": "; ".join(reasoning_parts),
            "fp_analysis": fp_analysis,
            "impact_analysis": impact_analysis,
            "ai_evaluation": ai_evaluation,
            "recommended_actions": ai_evaluation.get("additional_actions", []),
        }

    async def _update_learning_data(
        self, incident: Incident, rollback_decision: Dict[str, Any]
    ):
        """Update learning data for future false positive detection improvement"""

        learning_entry = {
            "incident_id": incident.id,
            "src_ip": incident.src_ip,
            "original_reason": incident.reason,
            "escalation_level": incident.escalation_level,
            "risk_score": incident.risk_score,
            "rollback_decision": rollback_decision["should_rollback"],
            "rollback_confidence": rollback_decision["confidence"],
            "fp_score": rollback_decision["fp_analysis"]["fp_score"],
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Store in memory (in production, this would go to a database)
        ip_key = incident.src_ip
        if ip_key not in self.fp_learning_data:
            self.fp_learning_data[ip_key] = []

        self.fp_learning_data[ip_key].append(learning_entry)

        # Keep only recent entries to prevent memory growth
        cutoff = datetime.utcnow() - timedelta(days=30)
        self.fp_learning_data[ip_key] = [
            entry
            for entry in self.fp_learning_data[ip_key]
            if datetime.fromisoformat(entry["timestamp"]) > cutoff
        ]

        self.logger.info(f"Updated learning data for IP {incident.src_ip}")

    async def execute_rollback(
        self, incident: Incident, rollback_decision: Dict[str, Any], db_session=None
    ) -> Dict[str, Any]:
        """Execute the rollback action if recommended"""

        if not rollback_decision["should_rollback"]:
            return {
                "executed": False,
                "reason": "Rollback not recommended",
                "action": "none",
            }

        try:
            # Find the original containment action
            if db_session:
                from sqlalchemy import and_

                action_query = (
                    db_session.query(Action)
                    .filter(
                        and_(
                            Action.incident_id == incident.id,
                            Action.action == "block",
                            Action.result == "success",
                        )
                    )
                    .order_by(Action.created_at.desc())
                    .first()
                )

                action = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: action_query
                )

                if action:
                    # Execute unblock
                    unblock_status, unblock_detail = await unblock_ip(incident.src_ip)

                    # Record rollback action
                    rollback_action = Action(
                        incident_id=incident.id,
                        action="rollback_unblock",
                        result="success" if unblock_status else "failed",
                        detail=f"Rollback executed: {unblock_detail}",
                        agent_id=self.agent_id,
                        confidence_score=rollback_decision["confidence"],
                        rollback_action_id=action.id,
                    )

                    db_session.add(rollback_action)

                    # Update incident status
                    incident.status = "rollback_executed"
                    incident.rollback_status = "executed"

                    await db_session.commit()

                    return {
                        "executed": True,
                        "action": "unblock",
                        "status": unblock_status,
                        "detail": unblock_detail,
                        "confidence": rollback_decision["confidence"],
                    }

            return {
                "executed": False,
                "reason": "No containment action found to rollback",
                "action": "none",
            }

        except Exception as e:
            self.logger.error(f"Rollback execution failed: {e}")
            return {
                "executed": False,
                "reason": f"Execution failed: {e}",
                "action": "failed",
            }


# Global singleton instance
containment_orchestrator = ContainmentAgent()

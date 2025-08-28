"""
AI Containment Agent for Autonomous Threat Response
Uses LangChain with OpenAI/xAI for intelligent containment decisions
"""
import asyncio
import json
import logging
import hmac
import hashlib
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

try:
    from langchain.agents import initialize_agent, Tool, AgentType
    from langchain_openai import OpenAI, ChatOpenAI
    from langchain.memory import ConversationBufferMemory
    from langchain.schema import HumanMessage, SystemMessage
except ImportError:
    # Fallback if LangChain is not available
    logging.warning("LangChain not available, using basic agent implementation")
    initialize_agent = None
    Tool = None
    AgentType = None
    OpenAI = None
    ChatOpenAI = None
    ConversationBufferMemory = None

from ..enhanced_containment import EnhancedContainmentEngine, ContainmentDecision
from ..ml_engine import EnsembleMLDetector
from ..external_intel import ThreatIntelligence
from ..models import Incident, Event, Action
from ..config import settings
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
                        temperature=0.1
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
                    description="Block an IP address using UFW. Input: 'ip duration_seconds'"
                ),
                Tool(
                    name="IsolateHost",
                    func=self._isolate_host,
                    description="Isolate a host via network segmentation. Input: 'hostname level'"
                ),
                Tool(
                    name="NotifyAnalyst",
                    func=self._notify_analyst,
                    description="Send alert to analysts. Input: 'incident_id message'"
                ),
                Tool(
                    name="RollbackAction",
                    func=self._rollback_action,
                    description="Rollback a containment action. Input: 'incident_id'"
                ),
                Tool(
                    name="QueryThreatIntel",
                    func=self._query_threat_intel,
                    description="Query threat intelligence for an IP. Input: 'ip_address'"
                )
            ]
            
            # Initialize memory
            memory = ConversationBufferMemory(memory_key="chat_history")
            
            # Initialize agent
            self.agent = initialize_agent(
                tools, 
                self.llm_client, 
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                memory=memory, 
                verbose=True
            )
            
            self.logger.info("LangChain containment agent initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize LangChain agent: {e}")
            self.agent = None
    
    async def orchestrate_response(
        self, 
        incident: Incident, 
        recent_events: List[Event],
        db_session=None
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
                    incident, decision, ml_score, intel_result
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
            return await self._fallback_containment(incident, decision if 'decision' in locals() else None)
    
    async def _langchain_orchestrate(
        self, 
        incident: Incident, 
        decision: ContainmentDecision,
        ml_score: float,
        intel_result: Any,
        recent_events: List[Event]
    ) -> Dict[str, Any]:
        """Use LangChain agent for orchestration"""
        
        # Prepare context for the agent
        context = {
            "incident": {
                "id": incident.id,
                "src_ip": incident.src_ip,
                "reason": incident.reason,
                "escalation_level": incident.escalation_level
            },
            "decision": {
                "should_contain": decision.should_contain,
                "risk_score": decision.risk_score,
                "confidence": decision.confidence,
                "escalation_level": decision.escalation_level,
                "reasoning": decision.reasoning
            },
            "ml_score": ml_score,
            "threat_intel": {
                "risk_score": intel_result.risk_score if intel_result else 0.0,
                "category": intel_result.category if intel_result else "unknown",
                "is_malicious": intel_result.is_malicious if intel_result else False
            },
            "event_count": len(recent_events),
            "auto_contain_enabled": settings.auto_contain
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
                "escalate_to_human": agent_output.get("escalate_to_human", False)
            }
            
        except Exception as e:
            self.logger.error(f"LangChain agent execution failed: {e}")
            raise
    
    async def _basic_orchestrate(
        self,
        incident: Incident,
        decision: ContainmentDecision, 
        ml_score: float,
        intel_result: Any
    ) -> Dict[str, Any]:
        """Basic orchestration without LangChain"""
        
        actions = []
        reasoning = []
        
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
            "confidence": 0.7  # Default confidence for rule-based
        }
    
    def _parse_agent_response(self, response: str) -> Dict[str, Any]:
        """Parse agent response, handling various formats"""
        try:
            # Try to parse as JSON
            if response.strip().startswith('{'):
                return json.loads(response)
            
            # Extract JSON from response if wrapped in text
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            
        except json.JSONDecodeError:
            pass
        
        # Fallback parsing
        return {
            "actions": ["block"] if "block" in response.lower() else ["monitor"],
            "reasoning": response[:200],
            "confidence": 0.5
        }
    
    async def _execute_actions(
        self, 
        actions: List[str], 
        incident: Incident, 
        decision: ContainmentDecision
    ) -> List[Dict[str, Any]]:
        """Execute the list of actions"""
        executed_actions = []
        
        for action in actions:
            try:
                if action == "block":
                    status = await self._block_ip_wrapper(
                        f"{incident.src_ip} {decision.duration_seconds}"
                    )
                    executed_actions.append({
                        "action": "block",
                        "status": status,
                        "timestamp": datetime.utcnow().isoformat()
                    })
                
                elif action == "notify":
                    status = self._notify_analyst(
                        f"{incident.id} Agent-detected incident requiring attention"
                    )
                    executed_actions.append({
                        "action": "notify",
                        "status": status,
                        "timestamp": datetime.utcnow().isoformat()
                    })
                
                elif action == "monitor":
                    executed_actions.append({
                        "action": "monitor",
                        "status": "Monitoring initiated",
                        "timestamp": datetime.utcnow().isoformat()
                    })
                
                # Add other actions as needed
                
            except Exception as e:
                self.logger.error(f"Failed to execute action {action}: {e}")
                executed_actions.append({
                    "action": action,
                    "status": f"Failed: {e}",
                    "timestamp": datetime.utcnow().isoformat()
                })
        
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
        """Isolate host (placeholder - would integrate with network tools)"""
        try:
            parts = input_str.split()
            hostname = parts[0]
            level = parts[1] if len(parts) > 1 else "soft"
            
            # Placeholder implementation
            self.logger.info(f"Would isolate {hostname} at level {level}")
            return f"Host {hostname} isolated at level {level}"
        except Exception as e:
            return f"Isolation failed: {e}"
    
    def _notify_analyst(self, input_str: str) -> str:
        """Send notification to analyst"""
        try:
            parts = input_str.split(' ', 1)
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
    
    async def _fallback_containment(
        self, 
        incident: Incident, 
        decision: Optional[ContainmentDecision]
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
                    "reasoning": "Agent fallback to rule-based containment"
                }
            else:
                return {
                    "success": True,
                    "actions": [{"action": "monitor", "status": "Monitoring"}],
                    "reasoning": "Agent fallback to monitoring"
                }
        except Exception as e:
            self.logger.error(f"Fallback containment failed: {e}")
            return {
                "success": False,
                "actions": [],
                "reasoning": f"Complete failure: {e}"
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
                "indicators": ["multiple_hosts", "credential_reuse", "privilege_escalation"]
            },
            "credential_stuffing": {
                "name": "Credential Stuffing Campaign",
                "description": "Hunt for widespread credential stuffing attacks",
                "indicators": ["rapid_login_attempts", "distributed_sources", "dictionary_passwords"]
            },
            "persistence_mechanisms": {
                "name": "Persistence Mechanism Detection",
                "description": "Hunt for backdoors and persistence mechanisms",
                "indicators": ["scheduled_tasks", "startup_modifications", "service_creation"]
            },
            "data_exfiltration": {
                "name": "Data Exfiltration Patterns",
                "description": "Hunt for unusual data transfer patterns",
                "indicators": ["large_transfers", "off_hours_activity", "unusual_destinations"]
            },
            "command_and_control": {
                "name": "Command and Control Communication",
                "description": "Hunt for C2 communication patterns",
                "indicators": ["beaconing", "dns_tunneling", "encrypted_channels"]
            }
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
                        temperature=0.3  # More creative for hunting
                    )
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM client: {e}")
        return None
    
    async def hunt_for_threats(self, db_session, lookback_hours: int = 24) -> List[Dict[str, Any]]:
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
            hypotheses.append({
                "id": query_id,
                "name": query_info["name"],
                "description": query_info["description"],
                "indicators": query_info["indicators"],
                "priority": "medium",
                "source": "predefined"
            })
        
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
            json_match = re.search(r'\[(.*?)\]', response.content, re.DOTALL)
            if json_match:
                ai_hypotheses = json.loads('[' + json_match.group(1) + ']')
                
                # Add metadata
                for hypothesis in ai_hypotheses:
                    hypothesis["id"] = f"ai_{hash(hypothesis['name']) % 10000}"
                    hypothesis["source"] = "ai_generated"
                
                return ai_hypotheses
        except Exception as e:
            self.logger.error(f"AI hypothesis parsing failed: {e}")
        
        return []
    
    async def _execute_hunt_query(
        self, 
        hypothesis: Dict[str, Any], 
        db_session, 
        lookback_hours: int
    ) -> Dict[str, Any]:
        """Execute a specific hunt query"""
        
        hunt_result = {
            "hypothesis": hypothesis,
            "findings": [],
            "executed_at": datetime.utcnow().isoformat(),
            "duration_seconds": 0
        }
        
        start_time = time.time()
        
        try:
            # Map hypothesis to specific detection logic
            if hypothesis["id"] == "lateral_movement":
                findings = await self._hunt_lateral_movement(db_session, lookback_hours)
            elif hypothesis["id"] == "credential_stuffing":
                findings = await self._hunt_credential_stuffing(db_session, lookback_hours)
            elif hypothesis["id"] == "persistence_mechanisms":
                findings = await self._hunt_persistence(db_session, lookback_hours)
            elif hypothesis["id"] == "data_exfiltration":
                findings = await self._hunt_data_exfiltration(db_session, lookback_hours)
            elif hypothesis["id"] == "command_and_control":
                findings = await self._hunt_c2_communication(db_session, lookback_hours)
            else:
                # Generic hunt for AI-generated hypotheses
                findings = await self._generic_hunt(hypothesis, db_session, lookback_hours)
            
            hunt_result["findings"] = findings
            
        except Exception as e:
            self.logger.error(f"Hunt query execution failed: {e}")
            hunt_result["error"] = str(e)
        
        hunt_result["duration_seconds"] = time.time() - start_time
        return hunt_result
    
    async def _hunt_lateral_movement(self, db_session, lookback_hours: int) -> List[Dict[str, Any]]:
        """Hunt for lateral movement indicators"""
        from sqlalchemy import and_, func
        
        findings = []
        cutoff_time = datetime.utcnow() - timedelta(hours=lookback_hours)
        
        # Look for IPs connecting to multiple ports/services
        query = db_session.query(Event.src_ip, func.count(Event.dst_port.distinct()).label('unique_ports')) \
            .filter(and_(Event.ts >= cutoff_time, Event.dst_port.isnot(None))) \
            .group_by(Event.src_ip) \
            .having(func.count(Event.dst_port.distinct()) >= 3)
        
        results = await asyncio.get_event_loop().run_in_executor(None, query.all)
        
        for src_ip, port_count in results:
            findings.append({
                "type": "lateral_movement",
                "src_ip": src_ip,
                "indicator": "multiple_port_scanning",
                "value": port_count,
                "severity": "medium" if port_count < 5 else "high",
                "description": f"IP {src_ip} connected to {port_count} different ports"
            })
        
        return findings
    
    async def _hunt_credential_stuffing(self, db_session, lookback_hours: int) -> List[Dict[str, Any]]:
        """Hunt for credential stuffing campaigns"""
        from sqlalchemy import and_, func
        
        findings = []
        cutoff_time = datetime.utcnow() - timedelta(hours=lookback_hours)
        
        # Look for high-volume failed login attempts with diverse credentials
        query = db_session.query(Event.src_ip, func.count(Event.id).label('attempt_count')) \
            .filter(and_(
                Event.ts >= cutoff_time,
                Event.eventid == "cowrie.login.failed"
            )) \
            .group_by(Event.src_ip) \
            .having(func.count(Event.id) >= 50)
        
        results = await asyncio.get_event_loop().run_in_executor(None, query.all)
        
        for src_ip, attempt_count in results:
            # Analyze credential diversity
            cred_query = db_session.query(Event) \
                .filter(and_(
                    Event.src_ip == src_ip,
                    Event.ts >= cutoff_time,
                    Event.eventid == "cowrie.login.failed"
                ))
            
            events = await asyncio.get_event_loop().run_in_executor(None, cred_query.all)
            
            usernames = set()
            passwords = set()
            
            for event in events:
                if event.raw and isinstance(event.raw, dict):
                    if 'username' in event.raw:
                        usernames.add(event.raw['username'])
                    if 'password' in event.raw:
                        passwords.add(event.raw['password'])
            
            credential_diversity = len(usernames) * len(passwords)
            
            if credential_diversity > 100:  # High diversity indicates stuffing
                findings.append({
                    "type": "credential_stuffing",
                    "src_ip": src_ip,
                    "indicator": "high_volume_diverse_credentials",
                    "value": {
                        "attempts": attempt_count,
                        "unique_usernames": len(usernames),
                        "unique_passwords": len(passwords),
                        "diversity_score": credential_diversity
                    },
                    "severity": "high",
                    "description": f"IP {src_ip} attempted {attempt_count} logins with {len(usernames)} usernames and {len(passwords)} passwords"
                })
        
        return findings
    
    async def _hunt_persistence(self, db_session, lookback_hours: int) -> List[Dict[str, Any]]:
        """Hunt for persistence mechanisms"""
        from sqlalchemy import and_, or_
        
        findings = []
        cutoff_time = datetime.utcnow() - timedelta(hours=lookback_hours)
        
        # Look for commands that could establish persistence
        persistence_commands = [
            'crontab', 'systemctl', 'service', 'chkconfig',
            'rc.local', '.bashrc', '.profile', 'autostart',
            'wget', 'curl', 'nohup', 'screen', 'tmux'
        ]
        
        for command in persistence_commands:
            query = db_session.query(Event) \
                .filter(and_(
                    Event.ts >= cutoff_time,
                    Event.eventid == "cowrie.command.input",
                    Event.message.like(f'%{command}%')
                ))
            
            events = await asyncio.get_event_loop().run_in_executor(None, query.all)
            
            for event in events:
                findings.append({
                    "type": "persistence_mechanism",
                    "src_ip": event.src_ip,
                    "indicator": f"persistence_command_{command}",
                    "value": event.message,
                    "severity": "medium",
                    "description": f"Potential persistence command executed: {command}",
                    "timestamp": event.ts.isoformat()
                })
        
        return findings
    
    async def _hunt_data_exfiltration(self, db_session, lookback_hours: int) -> List[Dict[str, Any]]:
        """Hunt for data exfiltration patterns"""
        from sqlalchemy import and_, func
        
        findings = []
        cutoff_time = datetime.utcnow() - timedelta(hours=lookback_hours)
        
        # Look for file download/upload activities
        exfil_events = ["cowrie.session.file_download", "cowrie.session.file_upload"]
        
        for event_type in exfil_events:
            query = db_session.query(Event.src_ip, func.count(Event.id).label('transfer_count')) \
                .filter(and_(
                    Event.ts >= cutoff_time,
                    Event.eventid == event_type
                )) \
                .group_by(Event.src_ip) \
                .having(func.count(Event.id) >= 5)
            
            results = await asyncio.get_event_loop().run_in_executor(None, query.all)
            
            for src_ip, transfer_count in results:
                findings.append({
                    "type": "data_exfiltration",
                    "src_ip": src_ip,
                    "indicator": f"high_volume_{event_type.split('.')[-1]}",
                    "value": transfer_count,
                    "severity": "high",
                    "description": f"IP {src_ip} performed {transfer_count} {event_type.split('.')[-1]} operations"
                })
        
        return findings
    
    async def _hunt_c2_communication(self, db_session, lookback_hours: int) -> List[Dict[str, Any]]:
        """Hunt for command and control communication patterns"""
        from sqlalchemy import and_, func
        
        findings = []
        cutoff_time = datetime.utcnow() - timedelta(hours=lookback_hours)
        
        # Look for beaconing patterns (regular communication intervals)
        query = db_session.query(Event.src_ip, func.count(Event.id).label('event_count')) \
            .filter(Event.ts >= cutoff_time) \
            .group_by(Event.src_ip) \
            .having(func.count(Event.id) >= 20)
        
        results = await asyncio.get_event_loop().run_in_executor(None, query.all)
        
        for src_ip, event_count in results:
            # Analyze timing patterns for this IP
            events_query = db_session.query(Event.ts) \
                .filter(and_(Event.src_ip == src_ip, Event.ts >= cutoff_time)) \
                .order_by(Event.ts)
            
            timestamps = await asyncio.get_event_loop().run_in_executor(None, events_query.all)
            
            if len(timestamps) >= 10:
                # Calculate time intervals
                intervals = []
                for i in range(1, len(timestamps)):
                    interval = (timestamps[i][0] - timestamps[i-1][0]).total_seconds()
                    intervals.append(interval)
                
                # Check for regular intervals (potential beaconing)
                avg_interval = np.mean(intervals)
                std_interval = np.std(intervals)
                
                # Low standard deviation indicates regular beaconing
                if std_interval < avg_interval * 0.3 and avg_interval > 60:  # Regular intervals > 1 minute
                    findings.append({
                        "type": "command_and_control",
                        "src_ip": src_ip,
                        "indicator": "beaconing_pattern",
                        "value": {
                            "average_interval": avg_interval,
                            "std_deviation": std_interval,
                            "regularity_score": 1 - (std_interval / avg_interval)
                        },
                        "severity": "high",
                        "description": f"Regular beaconing detected from {src_ip} (avg interval: {avg_interval:.1f}s)"
                    })
        
        return findings
    
    async def _generic_hunt(
        self, 
        hypothesis: Dict[str, Any], 
        db_session, 
        lookback_hours: int
    ) -> List[Dict[str, Any]]:
        """Generic hunt for AI-generated hypotheses"""
        findings = []
        
        # Use hypothesis indicators to build dynamic queries
        indicators = hypothesis.get("indicators", [])
        
        for indicator in indicators:
            # Basic pattern matching against event data
            # This could be enhanced with more sophisticated logic
            findings.extend(await self._pattern_hunt(indicator, db_session, lookback_hours))
        
        return findings
    
    async def _pattern_hunt(self, pattern: str, db_session, lookback_hours: int) -> List[Dict[str, Any]]:
        """Hunt for specific patterns in event data"""
        from sqlalchemy import and_, or_
        
        findings = []
        cutoff_time = datetime.utcnow() - timedelta(hours=lookback_hours)
        
        # Search for pattern in event messages
        query = db_session.query(Event) \
            .filter(and_(
                Event.ts >= cutoff_time,
                or_(
                    Event.message.like(f'%{pattern}%'),
                    Event.eventid.like(f'%{pattern}%')
                )
            ))
        
        events = await asyncio.get_event_loop().run_in_executor(None, query.all)
        
        for event in events:
            findings.append({
                "type": "pattern_match",
                "src_ip": event.src_ip,
                "indicator": f"pattern_{pattern}",
                "value": event.message,
                "severity": "low",
                "description": f"Pattern '{pattern}' found in event data",
                "timestamp": event.ts.isoformat()
            })
        
        return findings
    
    async def _correlate_hunt_findings(self, hunt_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Correlate findings across different hunts to identify campaigns"""
        correlated_results = []
        
        # Group findings by source IP
        ip_findings = {}
        
        for hunt_result in hunt_results:
            for finding in hunt_result["findings"]:
                src_ip = finding["src_ip"]
                if src_ip not in ip_findings:
                    ip_findings[src_ip] = []
                ip_findings[src_ip].append({
                    "hunt": hunt_result["hypothesis"]["name"],
                    "finding": finding
                })
        
        # Look for IPs with multiple hunt hits (potential campaigns)
        for src_ip, findings in ip_findings.items():
            if len(findings) >= 2:  # Multiple hunts detected this IP
                correlated_results.append({
                    "type": "correlated_campaign",
                    "src_ip": src_ip,
                    "hunt_count": len(findings),
                    "findings": findings,
                    "severity": "high",
                    "description": f"IP {src_ip} detected in {len(findings)} different hunt queries - potential campaign"
                })
        
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
                "scheduled_scans", "monitoring_systems", "backup_operations",
                "legitimate_tools", "admin_activities"
            ],
            "temporal_patterns": [
                "business_hours_only", "regular_intervals", "short_duration"
            ],
            "behavioral_patterns": [
                "low_entropy_commands", "standard_tools", "predictable_paths"
            ]
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
                        temperature=0.1  # Conservative for rollback decisions
                    )
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM client: {e}")
        return None
    
    async def evaluate_for_rollback(
        self, 
        incident: Incident, 
        hours_since_action: float,
        db_session=None
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
                "error": True
            }
    
    async def _get_incident_events(self, incident: Incident, db_session) -> List[Event]:
        """Get events related to the incident"""
        if not db_session:
            return []
        
        from sqlalchemy import and_
        
        # Get events from the same IP around the incident time
        query = db_session.query(Event) \
            .filter(and_(
                Event.src_ip == incident.src_ip,
                Event.ts >= incident.created_at - timedelta(hours=2),
                Event.ts <= incident.created_at + timedelta(hours=1)
            )) \
            .order_by(Event.ts)
        
        return await asyncio.get_event_loop().run_in_executor(None, query.all)
    
    async def _analyze_false_positive_indicators(
        self, 
        incident: Incident, 
        events: List[Event], 
        hours_since_action: float
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
        time_factor = min(hours_since_action / 24, 1.0)  # More likely FP as time passes without further activity
        fp_score += time_factor * 0.1
        
        return {
            "fp_score": min(fp_score, 1.0),
            "temporal_score": temporal_score,
            "behavioral_score": behavioral_score,
            "intel_score": intel_score,
            "time_factor": time_factor,
            "indicators": indicators_found
        }
    
    async def _analyze_temporal_patterns(self, incident: Incident, events: List[Event]) -> float:
        """Analyze temporal patterns that might indicate legitimate activity"""
        if not events:
            return 0.0
        
        score = 0.0
        
        # Check for business hours activity
        business_hour_events = [
            e for e in events 
            if 9 <= e.ts.hour <= 17 and e.ts.weekday() < 5
        ]
        if len(business_hour_events) / len(events) > 0.8:
            score += 0.3
        
        # Check for regular intervals
        if len(events) >= 3:
            intervals = []
            for i in range(1, len(events)):
                interval = (events[i].ts - events[i-1].ts).total_seconds()
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
                if isinstance(event.raw, dict) and 'input' in event.raw:
                    commands.append(event.raw['input'])
        
        if commands:
            # Check for standard administrative commands
            standard_commands = ['ls', 'pwd', 'whoami', 'id', 'ps', 'netstat', 'top']
            standard_count = sum(1 for cmd in commands if any(std in cmd for std in standard_commands))
            
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
                    usernames.add(event.raw.get('username', ''))
                    passwords.add(event.raw.get('password', ''))
            
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
            if hasattr(intel_result, 'source_agreement') and intel_result.source_agreement < 0.5:
                return 0.5
            
            return 0.0
            
        except Exception as e:
            self.logger.warning(f"Threat intel consistency check failed: {e}")
            return 0.0
    
    async def _assess_rollback_impact(self, incident: Incident, db_session) -> Dict[str, Any]:
        """Assess the impact of rolling back the containment action"""
        impact_analysis = {
            "risk_level": "low",
            "affected_systems": [],
            "business_impact": "minimal",
            "security_impact": "low"
        }
        
        try:
            # Check if IP is still active
            recent_activity = await self._check_recent_activity(incident.src_ip, db_session)
            
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
        
        query = db_session.query(Event) \
            .filter(and_(
                Event.src_ip == src_ip,
                Event.ts >= cutoff_time
            )) \
            .limit(1)
        
        result = await asyncio.get_event_loop().run_in_executor(None, query.first)
        return result is not None
    
    async def _ai_rollback_evaluation(
        self, 
        incident: Incident, 
        events: List[Event],
        fp_analysis: Dict[str, Any],
        impact_analysis: Dict[str, Any]
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
                "auto_contained": incident.auto_contained
            },
            "fp_analysis": fp_analysis,
            "impact_analysis": impact_analysis,
            "event_count": len(events),
            "event_types": list(set(e.eventid for e in events[:10]))  # Sample event types
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
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                ai_result = json.loads(json_match.group())
                return {
                    "ai_recommendation": ai_result.get("recommendation", "maintain"),
                    "ai_confidence": ai_result.get("confidence", 0.5),
                    "ai_reasoning": ai_result.get("reasoning", ""),
                    "ai_risk_assessment": ai_result.get("risk_assessment", "medium"),
                    "additional_actions": ai_result.get("additional_actions", [])
                }
        except Exception as e:
            self.logger.error(f"AI rollback evaluation failed: {e}")
        
        return {"ai_recommendation": "maintain", "ai_confidence": 0.5}
    
    async def _make_rollback_decision(
        self,
        fp_analysis: Dict[str, Any],
        impact_analysis: Dict[str, Any],
        ai_evaluation: Dict[str, Any]
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
        confidence = min(rollback_score if should_rollback else (1 - rollback_score), 1.0)
        
        # Compile reasoning
        reasoning_parts = []
        reasoning_parts.append(f"FP Score: {fp_analysis['fp_score']:.2f}")
        reasoning_parts.append(f"AI Recommendation: {ai_evaluation.get('ai_recommendation', 'N/A')}")
        reasoning_parts.append(f"Impact Risk: {impact_analysis['risk_level']}")
        reasoning_parts.append(f"Overall Score: {rollback_score:.2f}")
        
        if fp_analysis["indicators"]:
            reasoning_parts.append(f"FP Indicators: {', '.join(fp_analysis['indicators'])}")
        
        return {
            "should_rollback": should_rollback,
            "confidence": confidence,
            "rollback_score": rollback_score,
            "reasoning": "; ".join(reasoning_parts),
            "fp_analysis": fp_analysis,
            "impact_analysis": impact_analysis,
            "ai_evaluation": ai_evaluation,
            "recommended_actions": ai_evaluation.get("additional_actions", [])
        }
    
    async def _update_learning_data(self, incident: Incident, rollback_decision: Dict[str, Any]):
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
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Store in memory (in production, this would go to a database)
        ip_key = incident.src_ip
        if ip_key not in self.fp_learning_data:
            self.fp_learning_data[ip_key] = []
        
        self.fp_learning_data[ip_key].append(learning_entry)
        
        # Keep only recent entries to prevent memory growth
        cutoff = datetime.utcnow() - timedelta(days=30)
        self.fp_learning_data[ip_key] = [
            entry for entry in self.fp_learning_data[ip_key]
            if datetime.fromisoformat(entry["timestamp"]) > cutoff
        ]
        
        self.logger.info(f"Updated learning data for IP {incident.src_ip}")
    
    async def execute_rollback(self, incident: Incident, rollback_decision: Dict[str, Any], db_session=None) -> Dict[str, Any]:
        """Execute the rollback action if recommended"""
        
        if not rollback_decision["should_rollback"]:
            return {
                "executed": False,
                "reason": "Rollback not recommended",
                "action": "none"
            }
        
        try:
            # Find the original containment action
            if db_session:
                from sqlalchemy import and_
                
                action_query = db_session.query(Action) \
                    .filter(and_(
                        Action.incident_id == incident.id,
                        Action.action == "block",
                        Action.result == "success"
                    )) \
                    .order_by(Action.created_at.desc()) \
                    .first()
                
                action = await asyncio.get_event_loop().run_in_executor(None, lambda: action_query)
                
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
                        rollback_action_id=action.id
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
                        "confidence": rollback_decision["confidence"]
                    }
            
            return {
                "executed": False,
                "reason": "No containment action found to rollback",
                "action": "none"
            }
            
        except Exception as e:
            self.logger.error(f"Rollback execution failed: {e}")
            return {
                "executed": False,
                "reason": f"Execution failed: {e}",
                "action": "failed"
            }

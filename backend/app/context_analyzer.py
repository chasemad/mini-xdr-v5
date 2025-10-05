"""
Context Analyzer for Mini-XDR
Advanced incident context analysis with multi-dimensional threat assessment.
"""

import logging
import asyncio
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from sqlalchemy import select, and_, func, desc, or_
from sqlalchemy.ext.asyncio import AsyncSession

from .models import Incident, Event, ResponseWorkflow, AdvancedResponseAction
from .config import settings
from .secrets_manager import get_secure_env
from .ml_engine import ml_detector
from .external_intel import threat_intel

logger = logging.getLogger(__name__)


@dataclass
class ThreatContext:
    """Comprehensive threat context data structure"""
    severity_score: float
    attack_vector: str
    threat_category: str
    confidence: float
    indicators: List[str]
    timeline: List[Dict[str, Any]]
    attribution: Dict[str, Any]
    impact_assessment: Dict[str, Any]


@dataclass
class ResponseContext:
    """Response-specific context data"""
    urgency_level: str
    compliance_requirements: List[str]
    business_impact: Dict[str, Any]
    technical_constraints: Dict[str, Any]
    resource_availability: Dict[str, Any]


class AttackPhase(str, Enum):
    RECONNAISSANCE = "reconnaissance"
    INITIAL_ACCESS = "initial_access"
    EXECUTION = "execution"
    PERSISTENCE = "persistence"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DEFENSE_EVASION = "defense_evasion"
    CREDENTIAL_ACCESS = "credential_access"
    DISCOVERY = "discovery"
    LATERAL_MOVEMENT = "lateral_movement"
    COLLECTION = "collection"
    COMMAND_CONTROL = "command_control"
    EXFILTRATION = "exfiltration"
    IMPACT = "impact"


class ContextAnalyzer:
    """
    Advanced context analyzer providing multi-dimensional incident analysis
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.initialized = False
        
        # Analysis models and configurations
        self.severity_models = self._initialize_severity_models()
        self.attack_patterns = self._initialize_attack_patterns()
        self.threat_taxonomy = self._initialize_threat_taxonomy()
        
        # Geolocation and threat intelligence caches
        self.geo_cache = {}
        self.intel_cache = {}
        
    async def initialize(self):
        """Initialize context analyzer"""
        try:
            self.initialized = True
            self.logger.info("Context Analyzer initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Context Analyzer: {e}")
    
    def _initialize_severity_models(self) -> Dict[str, Any]:
        """Initialize severity assessment models"""
        return {
            "event_frequency": {
                "thresholds": {"low": 5, "medium": 20, "high": 100, "critical": 500},
                "weight": 0.25
            },
            "attack_complexity": {
                "indicators": {
                    "simple": ["single_event", "basic_pattern"],
                    "moderate": ["multi_stage", "persistence_attempts"],
                    "complex": ["multi_vector", "evasion_techniques", "lateral_movement"],
                    "advanced": ["zero_day", "custom_tools", "attribution_evasion"]
                },
                "weight": 0.30
            },
            "impact_potential": {
                "factors": ["data_exposure", "system_compromise", "service_disruption", "compliance_violation"],
                "weight": 0.25
            },
            "threat_intelligence": {
                "reputation_threshold": 0.7,
                "known_campaigns": True,
                "weight": 0.20
            }
        }
    
    def _initialize_attack_patterns(self) -> Dict[str, Any]:
        """Initialize attack pattern recognition models"""
        return {
            "brute_force": {
                "signatures": ["failed_login", "credential_testing", "dictionary_attack"],
                "event_patterns": ["rapid_attempts", "multiple_accounts"],
                "mitre_tactics": ["TA0006"],  # Credential Access
                "typical_duration": {"min": 300, "max": 3600},
                "sophistication": "low"
            },
            "ddos": {
                "signatures": ["high_volume", "syn_flood", "udp_flood", "http_flood"],
                "event_patterns": ["traffic_spike", "service_degradation"],
                "mitre_tactics": ["TA0040"],  # Impact
                "typical_duration": {"min": 60, "max": 7200},
                "sophistication": "medium"
            },
            "malware": {
                "signatures": ["file_execution", "process_injection", "network_callback"],
                "event_patterns": ["persistence_mechanism", "lateral_movement"],
                "mitre_tactics": ["TA0002", "TA0003", "TA0008"],  # Execution, Persistence, Lateral Movement
                "typical_duration": {"min": 600, "max": 86400},
                "sophistication": "high"
            },
            "apt": {
                "signatures": ["living_off_land", "custom_tools", "encrypted_communication"],
                "event_patterns": ["slow_and_low", "multi_stage", "attribution_evasion"],
                "mitre_tactics": ["TA0001", "TA0005", "TA0008", "TA0010"],  # Initial Access, Defense Evasion, Lateral Movement, Exfiltration
                "typical_duration": {"min": 86400, "max": 2592000},  # Days to months
                "sophistication": "very_high"
            },
            "insider_threat": {
                "signatures": ["off_hours_access", "unusual_data_volume", "privilege_abuse"],
                "event_patterns": ["gradual_escalation", "data_staging"],
                "mitre_tactics": ["TA0009", "TA0010"],  # Collection, Exfiltration
                "typical_duration": {"min": 3600, "max": 604800},  # Hours to weeks
                "sophistication": "medium"
            }
        }
    
    def _initialize_threat_taxonomy(self) -> Dict[str, Any]:
        """Initialize threat classification taxonomy"""
        return {
            "attack_vectors": {
                "network": ["tcp", "udp", "icmp", "http", "https", "dns"],
                "endpoint": ["file_system", "registry", "process", "service"],
                "email": ["phishing", "spear_phishing", "malicious_attachment"],
                "web": ["sql_injection", "xss", "csrf", "directory_traversal"],
                "social": ["social_engineering", "pretexting", "baiting"]
            },
            "threat_actors": {
                "automated": ["botnet", "scanner", "crawler"],
                "opportunistic": ["script_kiddie", "commodity_malware"],
                "targeted": ["apt_group", "insider", "nation_state"],
                "organized": ["cybercriminal_group", "ransomware_operator"]
            },
            "attack_goals": {
                "disruption": ["ddos", "defacement", "destruction"],
                "theft": ["data_exfiltration", "credential_harvesting", "financial_fraud"],
                "access": ["persistence", "lateral_movement", "privilege_escalation"],
                "espionage": ["reconnaissance", "surveillance", "intelligence_gathering"]
            }
        }
    
    async def analyze_comprehensive_context(
        self,
        incident_id: int,
        db_session: AsyncSession,
        include_predictions: bool = True
    ) -> Dict[str, Any]:
        """
        Perform comprehensive context analysis for an incident
        """
        try:
            # Get incident and related data
            incident = await db_session.get(Incident, incident_id)
            if not incident:
                return {"success": False, "error": "Incident not found"}
            
            # Get related events (expanded timeframe for better context)
            events_result = await db_session.execute(
                select(Event).where(
                    and_(
                        Event.src_ip == incident.src_ip,
                        Event.ts >= incident.created_at - timedelta(hours=24),
                        Event.ts <= incident.created_at + timedelta(hours=2)
                    )
                ).order_by(Event.ts.desc()).limit(1000)
            )
            events = events_result.scalars().all()
            
            # Get similar incidents for pattern analysis
            similar_incidents = await self._get_similar_incidents(incident, db_session)
            
            # Perform multi-dimensional analysis
            analysis_tasks = [
                self._analyze_threat_context(incident, events),
                self._analyze_response_context(incident, events, db_session),
                self._analyze_temporal_patterns(events),
                self._analyze_behavioral_indicators(incident, events),
                self._analyze_infrastructure_context(incident, events),
                self._analyze_attribution_indicators(incident, events)
            ]
            
            # Run analysis in parallel for efficiency
            results = await asyncio.gather(*analysis_tasks, return_exceptions=True)
            
            # Combine results
            context_analysis = {
                "incident_id": incident_id,
                "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
                "threat_context": results[0] if not isinstance(results[0], Exception) else {},
                "response_context": results[1] if not isinstance(results[1], Exception) else {},
                "temporal_analysis": results[2] if not isinstance(results[2], Exception) else {},
                "behavioral_analysis": results[3] if not isinstance(results[3], Exception) else {},
                "infrastructure_analysis": results[4] if not isinstance(results[4], Exception) else {},
                "attribution_analysis": results[5] if not isinstance(results[5], Exception) else {},
                "similar_incidents": similar_incidents,
                "analysis_quality": self._assess_analysis_quality(results)
            }
            
            # Add predictive analysis if requested
            if include_predictions:
                context_analysis["predictive_analysis"] = await self._generate_predictive_analysis(
                    incident, context_analysis, db_session
                )
            
            return {
                "success": True,
                "context_analysis": context_analysis
            }
            
        except Exception as e:
            self.logger.error(f"Failed to analyze context: {e}")
            return {"success": False, "error": str(e)}
    
    async def _analyze_threat_context(self, incident: Incident, events: List[Event]) -> ThreatContext:
        """Analyze threat-specific context"""
        
        # Calculate severity score
        severity_score = await self._calculate_threat_severity(incident, events)
        
        # Identify attack vector
        attack_vector = self._identify_attack_vector(events)
        
        # Classify threat category
        threat_category = self._classify_threat_category(incident, events)
        
        # Extract threat indicators
        indicators = self._extract_threat_indicators(incident, events)
        
        # Build attack timeline
        timeline = self._build_attack_timeline(events)
        
        # Assess attribution
        attribution = await self._assess_attribution(incident, events)
        
        # Calculate impact assessment
        impact_assessment = self._assess_threat_impact(incident, events)
        
        # Calculate overall confidence
        confidence = self._calculate_threat_confidence(
            severity_score, attack_vector, threat_category, indicators
        )
        
        return ThreatContext(
            severity_score=severity_score,
            attack_vector=attack_vector,
            threat_category=threat_category,
            confidence=confidence,
            indicators=indicators,
            timeline=timeline,
            attribution=attribution,
            impact_assessment=impact_assessment
        )
    
    async def _analyze_response_context(
        self, 
        incident: Incident, 
        events: List[Event], 
        db_session: AsyncSession
    ) -> ResponseContext:
        """Analyze response-specific context"""
        
        # Determine urgency level
        urgency_level = self._determine_urgency_level(incident, events)
        
        # Assess compliance requirements
        compliance_requirements = self._assess_compliance_requirements(incident)
        
        # Calculate business impact
        business_impact = self._assess_business_impact(incident, events)
        
        # Identify technical constraints
        technical_constraints = await self._identify_technical_constraints(incident, db_session)
        
        # Assess resource availability
        resource_availability = await self._assess_resource_availability(db_session)
        
        return ResponseContext(
            urgency_level=urgency_level,
            compliance_requirements=compliance_requirements,
            business_impact=business_impact,
            technical_constraints=technical_constraints,
            resource_availability=resource_availability
        )
    
    async def _calculate_threat_severity(self, incident: Incident, events: List[Event]) -> float:
        """Calculate comprehensive threat severity score"""
        
        severity_score = 0.0
        
        # Event frequency analysis
        event_count = len(events)
        frequency_model = self.severity_models["event_frequency"]
        for level, threshold in frequency_model["thresholds"].items():
            if event_count >= threshold:
                if level == "critical":
                    severity_score += 0.9 * frequency_model["weight"]
                elif level == "high":
                    severity_score += 0.7 * frequency_model["weight"]
                elif level == "medium":
                    severity_score += 0.5 * frequency_model["weight"]
                else:
                    severity_score += 0.3 * frequency_model["weight"]
                break
        
        # Attack complexity assessment
        complexity_score = self._assess_attack_complexity(incident, events)
        severity_score += complexity_score * self.severity_models["attack_complexity"]["weight"]
        
        # Impact potential assessment
        impact_score = self._assess_impact_potential(incident, events)
        severity_score += impact_score * self.severity_models["impact_potential"]["weight"]
        
        # Threat intelligence integration
        intel_score = await self._get_threat_intel_severity(incident.src_ip)
        severity_score += intel_score * self.severity_models["threat_intelligence"]["weight"]
        
        return min(severity_score, 1.0)
    
    def _identify_attack_vector(self, events: List[Event]) -> str:
        """Identify the primary attack vector"""
        
        # Analyze destination ports
        port_analysis = {}
        for event in events:
            if event.dst_port:
                port_analysis[event.dst_port] = port_analysis.get(event.dst_port, 0) + 1
        
        # Common port mappings
        port_vectors = {
            22: "ssh",
            23: "telnet", 
            25: "smtp",
            53: "dns",
            80: "http",
            443: "https",
            993: "imaps",
            995: "pop3s",
            3389: "rdp",
            5432: "postgresql",
            3306: "mysql"
        }
        
        if port_analysis:
            most_common_port = max(port_analysis, key=port_analysis.get)
            return port_vectors.get(most_common_port, f"port_{most_common_port}")
        
        return "unknown"
    
    def _classify_threat_category(self, incident: Incident, events: List[Event]) -> str:
        """Classify the threat into a specific category"""
        
        # Use existing incident classification if available
        if hasattr(incident, 'threat_category') and incident.threat_category:
            return incident.threat_category
        
        # Analyze based on incident reason and events
        reason_lower = incident.reason.lower()
        
        # Pattern matching for threat classification
        if "brute" in reason_lower or "password" in reason_lower:
            return "credential_access"
        elif "ddos" in reason_lower or "flood" in reason_lower:
            return "denial_of_service"
        elif "malware" in reason_lower or "trojan" in reason_lower:
            return "malware_infection"
        elif "phish" in reason_lower or "email" in reason_lower:
            return "phishing_campaign"
        elif "scan" in reason_lower or "probe" in reason_lower:
            return "reconnaissance"
        elif "insider" in reason_lower or "internal" in reason_lower:
            return "insider_threat"
        
        # Event-based classification
        if len(events) > 100:
            return "denial_of_service"
        elif len(set(e.dst_port for e in events if e.dst_port)) > 10:
            return "reconnaissance"
        
        return "unknown_threat"
    
    def _extract_threat_indicators(self, incident: Incident, events: List[Event]) -> List[str]:
        """Extract specific threat indicators"""
        
        indicators = []
        
        # Event-based indicators
        if len(events) > 100:
            indicators.append("high_volume_activity")
        
        if len(set(e.dst_port for e in events if e.dst_port)) > 5:
            indicators.append("port_scanning")
        
        if len(set(e.src_ip for e in events if e.src_ip)) > 1:
            indicators.append("distributed_attack")
        
        # Time-based indicators
        if events:
            timestamps = [e.ts for e in events if e.ts]
            if timestamps:
                duration = (max(timestamps) - min(timestamps)).total_seconds()
                if duration > 3600:
                    indicators.append("persistent_attack")
                elif duration < 60:
                    indicators.append("burst_attack")
        
        # Pattern-based indicators from incident reason
        reason_indicators = {
            "failed login": ["credential_testing"],
            "connection refused": ["service_probing"],
            "timeout": ["network_scanning"],
            "authentication": ["auth_bypass_attempt"]
        }
        
        for pattern, pattern_indicators in reason_indicators.items():
            if pattern in incident.reason.lower():
                indicators.extend(pattern_indicators)
        
        return list(set(indicators))  # Remove duplicates
    
    def _build_attack_timeline(self, events: List[Event]) -> List[Dict[str, Any]]:
        """Build a detailed attack timeline"""
        
        timeline = []
        
        if not events:
            return timeline
        
        # Sort events by timestamp
        sorted_events = sorted(events, key=lambda e: e.ts or datetime.min)
        
        # Group events by time windows (5-minute intervals)
        time_windows = {}
        for event in sorted_events:
            if event.ts:
                window = event.ts.replace(minute=(event.ts.minute // 5) * 5, second=0, microsecond=0)
                if window not in time_windows:
                    time_windows[window] = []
                time_windows[window].append(event)
        
        # Build timeline entries
        for window, window_events in sorted(time_windows.items()):
            timeline.append({
                "timestamp": window.isoformat(),
                "event_count": len(window_events),
                "unique_ports": len(set(e.dst_port for e in window_events if e.dst_port)),
                "event_types": list(set(e.eventid for e in window_events if e.eventid)),
                "phase": self._determine_attack_phase(window_events),
                "intensity": "high" if len(window_events) > 10 else "medium" if len(window_events) > 3 else "low"
            })
        
        return timeline
    
    async def _assess_attribution(self, incident: Incident, events: List[Event]) -> Dict[str, Any]:
        """Assess potential threat attribution"""
        
        attribution = {
            "confidence": 0.0,
            "indicators": [],
            "probable_actor_type": "unknown",
            "sophistication_level": "unknown",
            "geographical_indicators": {},
            "infrastructure_analysis": {}
        }
        
        try:
            # Get geographical information
            if threat_intel:
                geo_info = await threat_intel.get_ip_geolocation(incident.src_ip)
                attribution["geographical_indicators"] = geo_info
                
                # Assess based on geography
                if geo_info.get("country_code") in ["CN", "RU", "KP", "IR"]:
                    attribution["confidence"] += 0.3
                    attribution["indicators"].append("high_risk_geography")
                    attribution["probable_actor_type"] = "nation_state_affiliated"
            
            # Assess sophistication based on attack patterns
            complexity_score = self._assess_attack_complexity(incident, events)
            if complexity_score > 0.8:
                attribution["sophistication_level"] = "advanced"
                attribution["probable_actor_type"] = "apt_group"
            elif complexity_score > 0.6:
                attribution["sophistication_level"] = "intermediate"
                attribution["probable_actor_type"] = "organized_crime"
            else:
                attribution["sophistication_level"] = "basic"
                attribution["probable_actor_type"] = "opportunistic"
            
            # Infrastructure analysis
            attribution["infrastructure_analysis"] = {
                "hosting_provider": "unknown",
                "infrastructure_type": "unknown",
                "bulletproof_hosting": False,
                "tor_exit_node": False
            }
            
        except Exception as e:
            self.logger.warning(f"Attribution analysis failed: {e}")
        
        return attribution
    
    def _assess_threat_impact(self, incident: Incident, events: List[Event]) -> Dict[str, Any]:
        """Assess the potential impact of the threat"""
        
        impact = {
            "scope": "local",
            "affected_systems": 1,
            "data_at_risk": "low",
            "service_impact": "minimal",
            "financial_impact": "low",
            "reputation_impact": "minimal",
            "compliance_impact": "none"
        }
        
        # Assess scope based on event patterns
        unique_targets = len(set(e.dst_ip for e in events if e.dst_ip))
        if unique_targets > 10:
            impact["scope"] = "enterprise"
            impact["affected_systems"] = unique_targets
        elif unique_targets > 3:
            impact["scope"] = "departmental"
            impact["affected_systems"] = unique_targets
        
        # Assess data risk based on attack vector
        if any(port in [22, 3389, 445] for port in [e.dst_port for e in events if e.dst_port]):
            impact["data_at_risk"] = "high"
            impact["service_impact"] = "significant"
        
        # Assess financial impact based on attack type
        if "brute" in incident.reason.lower():
            impact["financial_impact"] = "medium"
        elif "ddos" in incident.reason.lower():
            impact["financial_impact"] = "high"
            impact["service_impact"] = "critical"
        
        return impact
    
    async def _analyze_temporal_patterns(self, events: List[Event]) -> Dict[str, Any]:
        """Analyze temporal attack patterns"""
        
        if not events:
            return {"pattern": "no_data", "confidence": 0.0}
        
        # Sort events by timestamp
        sorted_events = [e for e in events if e.ts]
        sorted_events.sort(key=lambda e: e.ts)
        
        if len(sorted_events) < 2:
            return {"pattern": "single_event", "confidence": 1.0}
        
        # Calculate time intervals
        intervals = []
        for i in range(1, len(sorted_events)):
            interval = (sorted_events[i].ts - sorted_events[i-1].ts).total_seconds()
            intervals.append(interval)
        
        # Analyze patterns
        avg_interval = sum(intervals) / len(intervals)
        interval_variance = sum((x - avg_interval) ** 2 for x in intervals) / len(intervals)
        
        # Classify temporal pattern
        if avg_interval < 1:
            pattern = "burst_attack"
            confidence = 0.9
        elif avg_interval < 60 and interval_variance < 100:
            pattern = "regular_intervals"
            confidence = 0.8
        elif interval_variance < avg_interval * 0.1:
            pattern = "steady_rate"
            confidence = 0.7
        else:
            pattern = "irregular_pattern"
            confidence = 0.5
        
        return {
            "pattern": pattern,
            "confidence": confidence,
            "average_interval_seconds": avg_interval,
            "total_duration_seconds": (sorted_events[-1].ts - sorted_events[0].ts).total_seconds(),
            "event_rate_per_minute": len(events) / max(1, (sorted_events[-1].ts - sorted_events[0].ts).total_seconds() / 60),
            "intervals_analysis": {
                "min": min(intervals),
                "max": max(intervals),
                "variance": interval_variance
            }
        }
    
    async def _analyze_behavioral_indicators(self, incident: Incident, events: List[Event]) -> Dict[str, Any]:
        """Analyze behavioral indicators of the attack"""
        
        behaviors = {
            "persistence": self._analyze_persistence_behavior(events),
            "evasion": self._analyze_evasion_behavior(events),
            "reconnaissance": self._analyze_reconnaissance_behavior(events),
            "exploitation": self._analyze_exploitation_behavior(incident, events),
            "impact": self._analyze_impact_behavior(events)
        }
        
        # Calculate overall behavioral score
        behavioral_score = sum(behaviors.values()) / len(behaviors)
        
        # Determine behavioral category
        behavioral_category = max(behaviors, key=behaviors.get)
        
        return {
            "overall_score": behavioral_score,
            "primary_behavior": behavioral_category,
            "behavior_scores": behaviors,
            "sophistication_indicators": self._extract_sophistication_indicators(behaviors),
            "attacker_intent": self._infer_attacker_intent(behaviors),
            "attack_progression": self._analyze_attack_progression(events)
        }
    
    async def _analyze_infrastructure_context(self, incident: Incident, events: List[Event]) -> Dict[str, Any]:
        """Analyze infrastructure and network context"""
        
        # Analyze source infrastructure
        source_analysis = {
            "ip_address": incident.src_ip,
            "ip_type": self._classify_ip_type(incident.src_ip),
            "hosting_analysis": await self._analyze_hosting_infrastructure(incident.src_ip),
            "network_behavior": self._analyze_network_behavior(events),
            "protocol_analysis": self._analyze_protocol_usage(events)
        }
        
        # Analyze target infrastructure
        target_analysis = {
            "targeted_services": self._identify_targeted_services(events),
            "attack_surface": self._assess_attack_surface(events),
            "vulnerability_indicators": self._identify_vulnerability_indicators(events)
        }
        
        return {
            "source_analysis": source_analysis,
            "target_analysis": target_analysis,
            "network_topology": self._infer_network_topology(events),
            "communication_patterns": self._analyze_communication_patterns(events)
        }
    
    async def _analyze_attribution_indicators(self, incident: Incident, events: List[Event]) -> Dict[str, Any]:
        """Analyze attribution indicators"""
        
        attribution_indicators = {
            "timing_analysis": self._analyze_attack_timing(events),
            "tool_signatures": self._identify_tool_signatures(events),
            "campaign_indicators": await self._identify_campaign_indicators(incident, events),
            "infrastructure_reuse": await self._analyze_infrastructure_reuse(incident.src_ip),
            "behavioral_fingerprint": self._generate_behavioral_fingerprint(events)
        }
        
        # Calculate attribution confidence
        attribution_confidence = self._calculate_attribution_confidence(attribution_indicators)
        
        return {
            "confidence": attribution_confidence,
            "indicators": attribution_indicators,
            "probable_actor": self._infer_probable_actor(attribution_indicators),
            "campaign_likelihood": self._assess_campaign_likelihood(attribution_indicators)
        }
    
    async def _get_similar_incidents(self, incident: Incident, db_session: AsyncSession) -> List[Dict[str, Any]]:
        """Find similar incidents for pattern analysis"""
        
        try:
            # Look for incidents with similar characteristics
            similar_result = await db_session.execute(
                select(Incident).where(
                    and_(
                        Incident.id != incident.id,
                        or_(
                            Incident.src_ip == incident.src_ip,
                            Incident.reason.like(f"%{incident.reason[:20]}%"),
                            and_(
                                hasattr(Incident, 'threat_category'),
                                getattr(Incident, 'threat_category', None) == getattr(incident, 'threat_category', None)
                            )
                        ),
                        Incident.created_at >= incident.created_at - timedelta(days=30)
                    )
                ).order_by(Incident.created_at.desc()).limit(10)
            )
            
            similar_incidents = similar_result.scalars().all()
            
            return [
                {
                    "incident_id": sim.id,
                    "src_ip": sim.src_ip,
                    "reason": sim.reason,
                    "created_at": sim.created_at.isoformat() if sim.created_at else None,
                    "status": sim.status,
                    "similarity_score": self._calculate_similarity_score(incident, sim)
                }
                for sim in similar_incidents
            ]
            
        except Exception as e:
            self.logger.warning(f"Failed to find similar incidents: {e}")
            return []
    
    async def _generate_predictive_analysis(
        self,
        incident: Incident,
        context_analysis: Dict[str, Any],
        db_session: AsyncSession
    ) -> Dict[str, Any]:
        """Generate predictive analysis for incident progression"""
        
        try:
            threat_context = context_analysis.get("threat_context", {})
            temporal_analysis = context_analysis.get("temporal_analysis", {})
            
            # Predict escalation probability
            escalation_probability = await self._predict_escalation_probability(
                incident, threat_context, temporal_analysis
            )
            
            # Predict lateral movement risk
            lateral_movement_risk = self._predict_lateral_movement_risk(
                threat_context, context_analysis.get("infrastructure_analysis", {})
            )
            
            # Predict attack duration
            predicted_duration = self._predict_attack_duration(
                temporal_analysis, threat_context
            )
            
            # Predict next likely targets
            next_targets = await self._predict_next_targets(incident, context_analysis, db_session)
            
            return {
                "escalation_probability": escalation_probability,
                "lateral_movement_risk": lateral_movement_risk,
                "predicted_duration_hours": predicted_duration,
                "next_likely_targets": next_targets,
                "recommended_monitoring": self._recommend_monitoring_enhancements(context_analysis),
                "early_warning_indicators": self._identify_early_warning_indicators(context_analysis)
            }
            
        except Exception as e:
            self.logger.error(f"Predictive analysis failed: {e}")
            return {"error": str(e)}
    
    # Helper methods (simplified implementations)
    def _assess_attack_complexity(self, incident: Incident, events: List[Event]) -> float:
        """Assess attack complexity score (0.0-1.0)"""
        complexity_score = 0.0
        
        # Multiple target ports indicate complexity
        unique_ports = len(set(e.dst_port for e in events if e.dst_port))
        if unique_ports > 10:
            complexity_score += 0.4
        elif unique_ports > 5:
            complexity_score += 0.2
        
        # Long duration indicates persistence/sophistication
        if events:
            timestamps = [e.ts for e in events if e.ts]
            if timestamps:
                duration_hours = (max(timestamps) - min(timestamps)).total_seconds() / 3600
                if duration_hours > 24:
                    complexity_score += 0.3
                elif duration_hours > 2:
                    complexity_score += 0.2
        
        # Event variety indicates sophisticated tools
        unique_event_types = len(set(e.eventid for e in events if e.eventid))
        if unique_event_types > 5:
            complexity_score += 0.3
        
        return min(complexity_score, 1.0)
    
    def _assess_impact_potential(self, incident: Incident, events: List[Event]) -> float:
        """Assess potential impact score (0.0-1.0)"""
        impact_score = 0.0
        
        # High-value target ports
        critical_ports = [22, 3389, 445, 993, 995, 5432, 3306]
        if any(e.dst_port in critical_ports for e in events if e.dst_port):
            impact_score += 0.4
        
        # Volume indicates potential for disruption
        if len(events) > 1000:
            impact_score += 0.4
        elif len(events) > 100:
            impact_score += 0.2
        
        # Persistence indicates potential for damage
        if events:
            timestamps = [e.ts for e in events if e.ts]
            if timestamps and (max(timestamps) - min(timestamps)).total_seconds() > 3600:
                impact_score += 0.2
        
        return min(impact_score, 1.0)
    
    async def _get_threat_intel_severity(self, ip_address: str) -> float:
        """Get threat intelligence severity score"""
        try:
            if threat_intel:
                intel_data = await threat_intel.check_ip_reputation(ip_address)
                return intel_data.get("reputation_score", 0.5)
        except Exception as e:
            self.logger.warning(f"Threat intel lookup failed: {e}")
        
        return 0.5  # Default neutral score
    
    def _calculate_threat_confidence(
        self, 
        severity_score: float, 
        attack_vector: str, 
        threat_category: str, 
        indicators: List[str]
    ) -> float:
        """Calculate overall threat confidence"""
        
        confidence = 0.0
        
        # Severity contributes to confidence
        confidence += severity_score * 0.4
        
        # Known attack vector increases confidence
        if attack_vector != "unknown":
            confidence += 0.2
        
        # Classified threat category increases confidence
        if threat_category != "unknown_threat":
            confidence += 0.2
        
        # Multiple indicators increase confidence
        indicator_score = min(len(indicators) / 5.0, 1.0) * 0.2
        confidence += indicator_score
        
        return min(confidence, 1.0)
    
    # Additional helper methods for analysis completeness
    def _assess_analysis_quality(self, analysis_results: List[Any]) -> Dict[str, Any]:
        """Assess the quality of the context analysis"""
        
        successful_analyses = sum(1 for result in analysis_results if not isinstance(result, Exception))
        total_analyses = len(analysis_results)
        
        quality_score = successful_analyses / total_analyses if total_analyses > 0 else 0.0
        
        return {
            "score": quality_score,
            "successful_analyses": successful_analyses,
            "total_analyses": total_analyses,
            "completeness": "high" if quality_score > 0.8 else "medium" if quality_score > 0.5 else "low"
        }
    
    def _determine_urgency_level(self, incident: Incident, events: List[Event]) -> str:
        """Determine response urgency level"""
        
        # Base urgency on event volume and attack patterns
        if len(events) > 1000:
            return "critical"
        elif len(events) > 100:
            return "high"
        elif len(events) > 20:
            return "medium"
        else:
            return "low"
    
    def _assess_business_impact(self, incident: Incident, events: List[Event]) -> Dict[str, Any]:
        """Assess business impact of the incident"""
        
        return {
            "revenue_impact": "low",
            "operational_impact": "minimal",
            "customer_impact": "none",
            "reputation_impact": "minimal",
            "estimated_cost": 1000,  # USD
            "recovery_time_estimate": 120  # minutes
        }
    
    async def _identify_technical_constraints(self, incident: Incident, db_session: AsyncSession) -> Dict[str, Any]:
        """Identify technical constraints for response"""
        
        return {
            "network_limitations": [],
            "system_dependencies": [],
            "maintenance_windows": [],
            "resource_constraints": []
        }
    
    async def _assess_resource_availability(self, db_session: AsyncSession) -> Dict[str, Any]:
        """Assess available resources for response"""
        
        return {
            "analyst_availability": "high",
            "technical_resources": "available",
            "automation_capacity": "normal",
            "external_support": "available"
        }
    
    # Simplified implementations of complex analysis methods
    def _determine_attack_phase(self, events: List[Event]) -> str:
        """Determine MITRE ATT&CK phase"""
        return AttackPhase.INITIAL_ACCESS.value  # Simplified
    
    def _classify_ip_type(self, ip_address: str) -> str:
        """Classify IP address type"""
        return "public"  # Simplified
    
    async def _analyze_hosting_infrastructure(self, ip_address: str) -> Dict[str, Any]:
        """Analyze hosting infrastructure"""
        return {"provider": "unknown", "type": "unknown"}  # Simplified
    
    def _analyze_network_behavior(self, events: List[Event]) -> Dict[str, Any]:
        """Analyze network behavior patterns"""
        return {"pattern": "standard", "anomalies": []}  # Simplified
    
    def _analyze_protocol_usage(self, events: List[Event]) -> Dict[str, Any]:
        """Analyze protocol usage patterns"""
        return {"primary_protocol": "tcp", "unusual_protocols": []}  # Simplified
    
    def _identify_targeted_services(self, events: List[Event]) -> List[str]:
        """Identify targeted services"""
        return ["ssh", "web"]  # Simplified
    
    def _assess_attack_surface(self, events: List[Event]) -> Dict[str, Any]:
        """Assess attack surface"""
        return {"exposed_services": 3, "risk_level": "medium"}  # Simplified
    
    def _identify_vulnerability_indicators(self, events: List[Event]) -> List[str]:
        """Identify vulnerability indicators"""
        return ["weak_authentication", "unpatched_services"]  # Simplified
    
    def _infer_network_topology(self, events: List[Event]) -> Dict[str, Any]:
        """Infer network topology"""
        return {"type": "standard", "complexity": "medium"}  # Simplified
    
    def _analyze_communication_patterns(self, events: List[Event]) -> Dict[str, Any]:
        """Analyze communication patterns"""
        return {"pattern": "client_server", "encryption": "unknown"}  # Simplified
    
    def _analyze_attack_timing(self, events: List[Event]) -> Dict[str, Any]:
        """Analyze attack timing patterns"""
        return {"timezone": "UTC", "business_hours": False}  # Simplified
    
    def _identify_tool_signatures(self, events: List[Event]) -> List[str]:
        """Identify tool signatures"""
        return ["generic_scanner"]  # Simplified
    
    async def _identify_campaign_indicators(self, incident: Incident, events: List[Event]) -> Dict[str, Any]:
        """Identify campaign indicators"""
        return {"campaign_probability": 0.3, "indicators": []}  # Simplified
    
    async def _analyze_infrastructure_reuse(self, ip_address: str) -> Dict[str, Any]:
        """Analyze infrastructure reuse patterns"""
        return {"reuse_probability": 0.5, "related_ips": []}  # Simplified
    
    def _generate_behavioral_fingerprint(self, events: List[Event]) -> Dict[str, Any]:
        """Generate behavioral fingerprint"""
        return {"fingerprint": "basic_scanner", "confidence": 0.6}  # Simplified
    
    def _calculate_attribution_confidence(self, indicators: Dict[str, Any]) -> float:
        """Calculate attribution confidence"""
        return 0.5  # Simplified
    
    def _infer_probable_actor(self, indicators: Dict[str, Any]) -> str:
        """Infer probable threat actor"""
        return "unknown"  # Simplified
    
    def _assess_campaign_likelihood(self, indicators: Dict[str, Any]) -> float:
        """Assess likelihood this is part of a larger campaign"""
        return 0.3  # Simplified
    
    def _calculate_similarity_score(self, incident1: Incident, incident2: Incident) -> float:
        """Calculate similarity score between incidents"""
        score = 0.0
        
        if incident1.src_ip == incident2.src_ip:
            score += 0.5
        
        if incident1.reason.lower() in incident2.reason.lower() or incident2.reason.lower() in incident1.reason.lower():
            score += 0.3
        
        return score
    
    # Additional behavioral analysis methods
    def _analyze_persistence_behavior(self, events: List[Event]) -> float:
        """Analyze persistence indicators"""
        if not events:
            return 0.0
        
        timestamps = [e.ts for e in events if e.ts]
        if timestamps:
            duration = (max(timestamps) - min(timestamps)).total_seconds()
            return min(duration / 86400, 1.0)  # Normalize to 24 hours max
        
        return 0.0
    
    def _analyze_evasion_behavior(self, events: List[Event]) -> float:
        """Analyze evasion indicators"""
        # Look for indicators like port variations, timing randomization
        unique_ports = len(set(e.dst_port for e in events if e.dst_port))
        return min(unique_ports / 20.0, 1.0)  # Normalize to 20 ports max
    
    def _analyze_reconnaissance_behavior(self, events: List[Event]) -> float:
        """Analyze reconnaissance indicators"""
        # Port scanning, service enumeration
        unique_ports = len(set(e.dst_port for e in events if e.dst_port))
        if unique_ports > 10:
            return 0.8
        elif unique_ports > 5:
            return 0.5
        else:
            return 0.2
    
    def _analyze_exploitation_behavior(self, incident: Incident, events: List[Event]) -> float:
        """Analyze exploitation indicators"""
        # Look for exploitation attempts in incident reason and events
        exploitation_keywords = ["exploit", "vulnerability", "buffer", "injection", "overflow"]
        if any(keyword in incident.reason.lower() for keyword in exploitation_keywords):
            return 0.8
        return 0.3
    
    def _analyze_impact_behavior(self, events: List[Event]) -> float:
        """Analyze impact-related behavior"""
        # High volume suggests impact intent
        return min(len(events) / 1000.0, 1.0)
    
    def _extract_sophistication_indicators(self, behaviors: Dict[str, float]) -> List[str]:
        """Extract sophistication indicators from behaviors"""
        indicators = []
        
        if behaviors.get("evasion", 0) > 0.7:
            indicators.append("advanced_evasion")
        if behaviors.get("persistence", 0) > 0.8:
            indicators.append("sophisticated_persistence")
        if behaviors.get("reconnaissance", 0) > 0.6:
            indicators.append("thorough_reconnaissance")
        
        return indicators
    
    def _infer_attacker_intent(self, behaviors: Dict[str, float]) -> str:
        """Infer attacker intent from behaviors"""
        if behaviors.get("impact", 0) > 0.7:
            return "disruption"
        elif behaviors.get("reconnaissance", 0) > 0.6:
            return "intelligence_gathering"
        elif behaviors.get("persistence", 0) > 0.7:
            return "long_term_access"
        else:
            return "opportunistic"
    
    def _analyze_attack_progression(self, events: List[Event]) -> List[str]:
        """Analyze attack progression phases"""
        phases = []
        
        if len(events) > 0:
            phases.append("initial_contact")
        if len(events) > 20:
            phases.append("active_exploitation")
        if len(events) > 100:
            phases.append("impact_phase")
        
        return phases
    
    async def _predict_escalation_probability(
        self, 
        incident: Incident, 
        threat_context: Dict[str, Any], 
        temporal_analysis: Dict[str, Any]
    ) -> float:
        """Predict probability of incident escalation"""
        
        escalation_score = 0.0
        
        # High severity increases escalation probability
        if threat_context.get("severity_score", 0) > 0.7:
            escalation_score += 0.4
        
        # Persistent attacks more likely to escalate
        if temporal_analysis.get("pattern") == "persistent_attack":
            escalation_score += 0.3
        
        # Complex attacks more likely to escalate
        if threat_context.get("confidence", 0) > 0.8:
            escalation_score += 0.3
        
        return min(escalation_score, 1.0)
    
    def _predict_lateral_movement_risk(
        self, 
        threat_context: Dict[str, Any], 
        infrastructure_analysis: Dict[str, Any]
    ) -> float:
        """Predict lateral movement risk"""
        
        risk_score = 0.0
        
        # Credential access attacks have high lateral movement risk
        if threat_context.get("threat_category") == "credential_access":
            risk_score += 0.6
        
        # Administrative service targeting increases risk
        if "ssh" in str(infrastructure_analysis) or "rdp" in str(infrastructure_analysis):
            risk_score += 0.4
        
        return min(risk_score, 1.0)
    
    def _predict_attack_duration(
        self, 
        temporal_analysis: Dict[str, Any], 
        threat_context: Dict[str, Any]
    ) -> float:
        """Predict remaining attack duration in hours"""
        
        current_duration = temporal_analysis.get("total_duration_seconds", 0) / 3600
        pattern = temporal_analysis.get("pattern", "unknown")
        
        if pattern == "burst_attack":
            return max(0.5 - current_duration, 0)  # Short duration
        elif pattern == "persistent_attack":
            return max(24 - current_duration, 0)  # Up to 24 hours
        else:
            return max(2 - current_duration, 0)  # Default 2 hours
    
    async def _predict_next_targets(
        self, 
        incident: Incident, 
        context_analysis: Dict[str, Any], 
        db_session: AsyncSession
    ) -> List[Dict[str, Any]]:
        """Predict next likely targets"""
        
        # Simplified prediction based on attack patterns
        return [
            {"target_type": "similar_services", "probability": 0.7},
            {"target_type": "same_network", "probability": 0.5},
            {"target_type": "administrative_systems", "probability": 0.3}
        ]
    
    def _recommend_monitoring_enhancements(self, context_analysis: Dict[str, Any]) -> List[str]:
        """Recommend monitoring enhancements"""
        
        recommendations = ["enhanced_logging", "network_monitoring"]
        
        threat_context = context_analysis.get("threat_context", {})
        if threat_context.get("threat_category") == "credential_access":
            recommendations.append("authentication_monitoring")
        
        return recommendations
    
    def _identify_early_warning_indicators(self, context_analysis: Dict[str, Any]) -> List[str]:
        """Identify early warning indicators for similar attacks"""
        
        return [
            "unusual_login_patterns",
            "service_enumeration_attempts", 
            "network_scanning_activity",
            "failed_authentication_spikes"
        ]


# Global instance
context_analyzer = ContextAnalyzer()


async def get_context_analyzer() -> ContextAnalyzer:
    """Get the global context analyzer instance"""
    if not context_analyzer.initialized:
        await context_analyzer.initialize()
    return context_analyzer








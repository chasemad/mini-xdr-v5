"""
AI Response Advisor for Mini-XDR
Provides intelligent response recommendations with confidence scoring and contextual analysis.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import and_, desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from .config import settings
from .external_intel import threat_intel
from .ml_engine import ml_detector
from .models import AdvancedResponseAction, Event, Incident, ResponseWorkflow
from .secrets_manager import get_secure_env

logger = logging.getLogger(__name__)


class RecommendationConfidence(str, Enum):
    VERY_LOW = "very_low"  # 0.0-0.3
    LOW = "low"  # 0.3-0.5
    MEDIUM = "medium"  # 0.5-0.7
    HIGH = "high"  # 0.7-0.9
    VERY_HIGH = "very_high"  # 0.9-1.0


class ResponseStrategy(str, Enum):
    IMMEDIATE = "immediate"  # High confidence, immediate action
    CAUTIOUS = "cautious"  # Medium confidence, requires approval
    INVESTIGATIVE = "investigative"  # Low confidence, gather more data
    COLLABORATIVE = "collaborative"  # Requires human input


class AIResponseAdvisor:
    """
    AI-powered response advisor that analyzes incidents and recommends optimal response strategies.
    """

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.llm_provider = settings.llm_provider
        self.openai_client = None
        self.initialized = False

        # Response knowledge base
        self.response_patterns = self._initialize_response_patterns()

        # Context analysis weights
        self.context_weights = {
            "threat_severity": 0.25,
            "attack_pattern": 0.20,
            "impact_scope": 0.15,
            "historical_success": 0.15,
            "compliance_requirements": 0.10,
            "business_context": 0.10,
            "technical_feasibility": 0.05,
        }

    async def initialize(self):
        """Initialize AI components"""
        try:
            if self.llm_provider == "openai":
                import openai

                api_key = get_secure_env("OPENAI_API_KEY", "mini-xdr/openai-api-key")
                if api_key:
                    self.openai_client = openai.AsyncOpenAI(api_key=api_key)
                    self.initialized = True
                    self.logger.info(
                        "OpenAI client initialized for AI Response Advisor"
                    )
                else:
                    self.logger.warning(
                        "OpenAI API key not found, using fallback recommendations"
                    )

        except Exception as e:
            self.logger.error(f"Failed to initialize AI Response Advisor: {e}")

    def _initialize_response_patterns(self) -> Dict[str, Any]:
        """Initialize response pattern knowledge base"""
        return {
            # Malware response patterns
            "malware": {
                "indicators": [
                    "suspicious_process",
                    "file_modification",
                    "network_callback",
                ],
                "recommended_actions": [
                    {
                        "action": "isolate_host_advanced",
                        "priority": 1,
                        "confidence": 0.9,
                    },
                    {
                        "action": "memory_dump_collection",
                        "priority": 2,
                        "confidence": 0.8,
                    },
                    {"action": "block_ip_advanced", "priority": 3, "confidence": 0.7},
                ],
                "escalation_threshold": 0.8,
                "typical_duration": 1800,  # 30 minutes
            },
            # DDoS response patterns
            "ddos": {
                "indicators": [
                    "high_connection_rate",
                    "bandwidth_spike",
                    "service_degradation",
                ],
                "recommended_actions": [
                    {"action": "traffic_redirection", "priority": 1, "confidence": 0.9},
                    {
                        "action": "deploy_firewall_rules",
                        "priority": 2,
                        "confidence": 0.8,
                    },
                    {"action": "block_ip_advanced", "priority": 3, "confidence": 0.7},
                ],
                "escalation_threshold": 0.7,
                "typical_duration": 600,  # 10 minutes
            },
            # Brute force response patterns
            "brute_force": {
                "indicators": [
                    "failed_login_attempts",
                    "multiple_source_ips",
                    "credential_testing",
                ],
                "recommended_actions": [
                    {"action": "block_ip_advanced", "priority": 1, "confidence": 0.8},
                    {"action": "account_disable", "priority": 2, "confidence": 0.7},
                    {"action": "password_reset_bulk", "priority": 3, "confidence": 0.6},
                ],
                "escalation_threshold": 0.6,
                "typical_duration": 300,  # 5 minutes
            },
            # Insider threat response patterns
            "insider_threat": {
                "indicators": [
                    "unusual_data_access",
                    "off_hours_activity",
                    "privilege_escalation",
                ],
                "recommended_actions": [
                    {
                        "action": "memory_dump_collection",
                        "priority": 1,
                        "confidence": 0.9,
                    },
                    {"action": "account_disable", "priority": 2, "confidence": 0.8},
                    {
                        "action": "iam_policy_restriction",
                        "priority": 3,
                        "confidence": 0.7,
                    },
                ],
                "escalation_threshold": 0.9,
                "typical_duration": 2400,  # 40 minutes
            },
            # Phishing response patterns
            "phishing": {
                "indicators": [
                    "suspicious_email",
                    "credential_harvesting",
                    "domain_spoofing",
                ],
                "recommended_actions": [
                    {"action": "email_recall", "priority": 1, "confidence": 0.8},
                    {"action": "dns_sinkhole", "priority": 2, "confidence": 0.7},
                    {"action": "password_reset_bulk", "priority": 3, "confidence": 0.6},
                ],
                "escalation_threshold": 0.5,
                "typical_duration": 900,  # 15 minutes
            },
        }

    async def get_response_recommendations(
        self,
        incident_id: int,
        db_session: AsyncSession,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Get AI-powered response recommendations for an incident
        """
        try:
            # Get incident and related events
            incident = await db_session.get(Incident, incident_id)
            if not incident:
                return {"success": False, "error": "Incident not found"}

            # Get related events
            events_result = await db_session.execute(
                select(Event)
                .where(Event.src_ip == incident.src_ip)
                .order_by(Event.ts.desc())
                .limit(50)
            )
            events = events_result.scalars().all()

            # Get existing response history
            workflows_result = await db_session.execute(
                select(ResponseWorkflow)
                .where(ResponseWorkflow.incident_id == incident_id)
                .order_by(ResponseWorkflow.created_at.desc())
            )
            existing_workflows = workflows_result.scalars().all()

            # Analyze incident context
            context_analysis = await self._analyze_incident_context(
                incident, events, existing_workflows, context
            )

            # Generate recommendations
            recommendations = await self._generate_recommendations(
                incident, context_analysis, db_session
            )

            # Get confidence scoring
            confidence_analysis = await self._calculate_confidence_scores(
                recommendations, context_analysis
            )

            # Generate natural language explanations
            explanations = await self._generate_explanations(
                recommendations, context_analysis, confidence_analysis
            )

            return {
                "success": True,
                "incident_id": incident_id,
                "context_analysis": context_analysis,
                "recommendations": recommendations,
                "confidence_analysis": confidence_analysis,
                "explanations": explanations,
                "strategy": self._determine_response_strategy(confidence_analysis),
                "estimated_duration": self._estimate_response_duration(recommendations),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            self.logger.error(f"Failed to generate response recommendations: {e}")
            return {"success": False, "error": str(e)}

    async def _analyze_incident_context(
        self,
        incident: Incident,
        events: List[Event],
        existing_workflows: List[ResponseWorkflow],
        additional_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Analyze incident context for AI recommendations"""

        context = {
            "incident_severity": self._assess_incident_severity(incident, events),
            "attack_pattern": self._identify_attack_pattern(incident, events),
            "impact_scope": self._assess_impact_scope(incident, events),
            "threat_intelligence": await self._get_threat_intelligence(incident.src_ip),
            "historical_patterns": self._analyze_historical_patterns(incident, events),
            "compliance_context": self._assess_compliance_requirements(incident),
            "business_context": additional_context or {},
            "previous_responses": self._analyze_previous_responses(existing_workflows),
            "ml_analysis": await self._get_ml_analysis(incident, events),
        }

        return context

    def _assess_incident_severity(
        self, incident: Incident, events: List[Event]
    ) -> Dict[str, Any]:
        """Assess the severity of the incident"""

        # Calculate severity based on various factors
        severity_score = 0.0
        factors = []

        # Event frequency
        event_count = len(events)
        if event_count > 100:
            severity_score += 0.3
            factors.append(f"High event volume ({event_count} events)")
        elif event_count > 20:
            severity_score += 0.2
            factors.append(f"Moderate event volume ({event_count} events)")

        # Escalation level from incident
        if hasattr(incident, "escalation_level"):
            level_scores = {"low": 0.1, "medium": 0.3, "high": 0.6, "critical": 0.9}
            level_score = level_scores.get(incident.escalation_level, 0.3)
            severity_score += level_score
            factors.append(f"Escalation level: {incident.escalation_level}")

        # Risk score from incident
        if hasattr(incident, "risk_score") and incident.risk_score:
            severity_score += incident.risk_score * 0.4
            factors.append(f"Risk score: {incident.risk_score:.2f}")

        return {
            "score": min(severity_score, 1.0),
            "level": self._score_to_severity_level(severity_score),
            "factors": factors,
        }

    def _identify_attack_pattern(
        self, incident: Incident, events: List[Event]
    ) -> Dict[str, Any]:
        """Identify the attack pattern from incident and events"""

        patterns = {
            "brute_force": 0.0,
            "ddos": 0.0,
            "malware": 0.0,
            "insider_threat": 0.0,
            "phishing": 0.0,
            "apt": 0.0,
        }

        # Analyze event patterns
        if len(events) > 50:
            patterns["ddos"] += 0.4

        # Analyze incident reason
        reason_lower = incident.reason.lower()
        if "brute" in reason_lower or "password" in reason_lower:
            patterns["brute_force"] += 0.6
        elif "ddos" in reason_lower or "flood" in reason_lower:
            patterns["ddos"] += 0.7
        elif "malware" in reason_lower or "trojan" in reason_lower:
            patterns["malware"] += 0.6

        # Get the most likely pattern
        likely_pattern = max(patterns, key=patterns.get)
        confidence = patterns[likely_pattern]

        return {
            "pattern": likely_pattern,
            "confidence": confidence,
            "all_scores": patterns,
            "indicators": self._extract_pattern_indicators(events, likely_pattern),
        }

    def _assess_impact_scope(
        self, incident: Incident, events: List[Event]
    ) -> Dict[str, Any]:
        """Assess the potential impact scope"""

        # Analyze source IPs
        unique_sources = len(set(event.src_ip for event in events if event.src_ip))

        # Analyze target ports
        unique_ports = len(set(event.dst_port for event in events if event.dst_port))

        # Calculate scope score
        scope_score = 0.0
        if unique_sources > 10:
            scope_score += 0.4
        if unique_ports > 5:
            scope_score += 0.3
        if len(events) > 100:
            scope_score += 0.3

        return {
            "score": min(scope_score, 1.0),
            "unique_sources": unique_sources,
            "unique_ports": unique_ports,
            "total_events": len(events),
            "scope_level": self._score_to_scope_level(scope_score),
        }

    async def _get_threat_intelligence(self, ip_address: str) -> Dict[str, Any]:
        """Get threat intelligence for the IP address"""
        try:
            if threat_intel:
                intel_data = await threat_intel.check_ip_reputation(ip_address)
                return {
                    "reputation_score": intel_data.get("reputation_score", 0.5),
                    "threat_categories": intel_data.get("categories", []),
                    "last_seen": intel_data.get("last_seen"),
                    "geographical_info": intel_data.get("geo_info", {}),
                    "confidence": intel_data.get("confidence", 0.5),
                }
        except Exception as e:
            self.logger.warning(f"Failed to get threat intelligence: {e}")

        return {"reputation_score": 0.5, "threat_categories": [], "confidence": 0.0}

    def _analyze_historical_patterns(
        self, incident: Incident, events: List[Event]
    ) -> Dict[str, Any]:
        """Analyze historical attack patterns"""

        # Group events by time windows
        hourly_counts = {}
        for event in events:
            if event.ts:
                hour = event.ts.replace(minute=0, second=0, microsecond=0)
                hourly_counts[hour] = hourly_counts.get(hour, 0) + 1

        # Detect patterns
        peak_hour = max(hourly_counts, key=hourly_counts.get) if hourly_counts else None
        peak_count = hourly_counts.get(peak_hour, 0) if peak_hour else 0

        return {
            "peak_activity": {
                "hour": peak_hour.isoformat() if peak_hour else None,
                "event_count": peak_count,
            },
            "attack_duration": self._calculate_attack_duration(events),
            "intensity_pattern": self._analyze_intensity_pattern(events),
            "persistence_score": self._calculate_persistence_score(events),
        }

    def _assess_compliance_requirements(self, incident: Incident) -> Dict[str, Any]:
        """Assess compliance requirements for the incident"""

        compliance_requirements = []
        urgency_multiplier = 1.0

        # Default compliance frameworks
        compliance_requirements.extend(["SOC2", "ISO27001"])

        # Check for PCI DSS requirements (financial data)
        if hasattr(incident, "threat_category"):
            if incident.threat_category in ["credential_stuffing", "payment_fraud"]:
                compliance_requirements.append("PCI-DSS")
                urgency_multiplier *= 1.3

        # Check for HIPAA requirements (healthcare)
        if "health" in incident.reason.lower() or "medical" in incident.reason.lower():
            compliance_requirements.append("HIPAA")
            urgency_multiplier *= 1.5

        # Check for GDPR requirements (personal data)
        if (
            "personal" in incident.reason.lower()
            or "privacy" in incident.reason.lower()
        ):
            compliance_requirements.append("GDPR")
            urgency_multiplier *= 1.4

        return {
            "frameworks": compliance_requirements,
            "urgency_multiplier": urgency_multiplier,
            "reporting_required": len(compliance_requirements) > 2,
            "max_response_time": 3600 // urgency_multiplier,  # 1 hour baseline
        }

    def _analyze_previous_responses(
        self, workflows: List[ResponseWorkflow]
    ) -> Dict[str, Any]:
        """Analyze effectiveness of previous responses"""

        if not workflows:
            return {"success_rate": 0.0, "average_duration": 0, "recommendations": []}

        # Calculate success metrics
        successful_workflows = [
            w for w in workflows if w.success_rate and w.success_rate > 0.7
        ]
        overall_success_rate = sum(w.success_rate or 0 for w in workflows) / len(
            workflows
        )

        # Analyze effective actions
        effective_actions = []
        for workflow in successful_workflows:
            if workflow.steps:
                for step in workflow.steps:
                    effective_actions.append(step.get("action_type"))

        return {
            "success_rate": overall_success_rate,
            "total_workflows": len(workflows),
            "successful_workflows": len(successful_workflows),
            "average_duration": sum(w.execution_time_ms or 0 for w in workflows)
            / len(workflows),
            "effective_actions": list(set(effective_actions)),
            "recommendations": self._extract_historical_recommendations(
                successful_workflows
            ),
        }

    async def _get_ml_analysis(
        self, incident: Incident, events: List[Event]
    ) -> Dict[str, Any]:
        """Get ML analysis for the incident"""
        try:
            if ml_detector and events:
                # Use the first event as representative
                event = events[0]
                ml_scores = await ml_detector.predict_event_threat(event)

                return {
                    "anomaly_score": ml_scores.get("anomaly_score", 0.5),
                    "threat_probability": ml_scores.get("threat_probability", 0.5),
                    "model_confidence": ml_scores.get("model_confidence", 0.5),
                    "feature_importance": ml_scores.get("feature_importance", {}),
                    "ensemble_scores": ml_scores.get("ensemble_scores", {}),
                }
        except Exception as e:
            self.logger.warning(f"Failed to get ML analysis: {e}")

        return {
            "anomaly_score": 0.5,
            "threat_probability": 0.5,
            "model_confidence": 0.0,
        }

    async def _generate_recommendations(
        self,
        incident: Incident,
        context_analysis: Dict[str, Any],
        db_session: AsyncSession,
    ) -> List[Dict[str, Any]]:
        """Generate response action recommendations"""

        recommendations = []

        # Get attack pattern and base recommendations
        attack_pattern = context_analysis["attack_pattern"]["pattern"]
        pattern_data = self.response_patterns.get(attack_pattern, {})
        base_actions = pattern_data.get("recommended_actions", [])

        # Enhance recommendations with context
        for action_data in base_actions:
            action_type = action_data["action"]
            base_confidence = action_data["confidence"]

            # Adjust confidence based on context
            adjusted_confidence = await self._adjust_confidence_with_context(
                base_confidence, context_analysis, action_type
            )

            # Generate parameters based on context
            parameters = await self._generate_action_parameters(
                action_type, incident, context_analysis
            )

            recommendation = {
                "action_type": action_type,
                "priority": action_data["priority"],
                "confidence": adjusted_confidence,
                "confidence_level": self._confidence_to_level(adjusted_confidence),
                "parameters": parameters,
                "estimated_duration": self._estimate_action_duration(
                    action_type, context_analysis
                ),
                "safety_considerations": self._get_safety_considerations(action_type),
                "rollback_plan": self._generate_rollback_plan(action_type, parameters),
                "approval_required": adjusted_confidence < 0.8
                or action_type
                in ["deploy_firewall_rules", "resource_isolation", "account_disable"],
            }

            recommendations.append(recommendation)

        # Sort by priority and confidence
        recommendations.sort(key=lambda x: (x["priority"], -x["confidence"]))

        # Add AI-generated recommendations if OpenAI is available
        if self.initialized and self.openai_client:
            ai_recommendations = await self._get_ai_enhanced_recommendations(
                incident, context_analysis, recommendations
            )
            recommendations.extend(ai_recommendations)

        return recommendations[:10]  # Limit to top 10 recommendations

    async def _adjust_confidence_with_context(
        self, base_confidence: float, context_analysis: Dict[str, Any], action_type: str
    ) -> float:
        """Adjust confidence based on context analysis"""

        adjusted_confidence = base_confidence

        # Adjust based on threat intelligence
        threat_intel = context_analysis.get("threat_intelligence", {})
        if threat_intel.get("reputation_score", 0.5) > 0.8:
            adjusted_confidence += 0.1
        elif threat_intel.get("reputation_score", 0.5) < 0.3:
            adjusted_confidence -= 0.1

        # Adjust based on ML analysis
        ml_analysis = context_analysis.get("ml_analysis", {})
        if ml_analysis.get("threat_probability", 0.5) > 0.8:
            adjusted_confidence += 0.15
        elif ml_analysis.get("threat_probability", 0.5) < 0.3:
            adjusted_confidence -= 0.1

        # Adjust based on previous response success
        previous_responses = context_analysis.get("previous_responses", {})
        if action_type in previous_responses.get("effective_actions", []):
            adjusted_confidence += 0.1

        # Adjust based on incident severity
        severity = context_analysis.get("incident_severity", {})
        if severity.get("score", 0.5) > 0.8:
            adjusted_confidence += 0.1

        return max(0.0, min(1.0, adjusted_confidence))

    async def _generate_action_parameters(
        self, action_type: str, incident: Incident, context_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate context-aware parameters for actions"""

        base_params = {
            "target": incident.src_ip,
            "incident_id": incident.id,
            "reason": f"AI-recommended response for {action_type}",
        }

        # Action-specific parameter generation
        if action_type == "block_ip_advanced":
            severity = context_analysis.get("incident_severity", {}).get("score", 0.5)
            duration = (
                3600 if severity > 0.7 else 1800
            )  # 1 hour for high severity, 30 min for lower

            base_params.update(
                {
                    "ip_address": incident.src_ip,
                    "duration": duration,
                    "block_level": "comprehensive" if severity > 0.8 else "standard",
                    "geo_restrictions": severity > 0.6,
                }
            )

        elif action_type == "isolate_host_advanced":
            severity = context_analysis.get("incident_severity", {}).get("score", 0.5)
            base_params.update(
                {
                    "host_identifier": incident.src_ip,
                    "isolation_level": "strict" if severity > 0.7 else "soft",
                    "monitoring": "enhanced" if severity > 0.8 else "standard",
                    "exceptions": [] if severity > 0.9 else ["dns", "ntp"],
                }
            )

        elif action_type == "memory_dump_collection":
            base_params.update(
                {
                    "target_hosts": [incident.src_ip],
                    "dump_type": "full",
                    "encryption": True,
                    "retention": "evidence_grade",
                }
            )

        return base_params

    async def _calculate_confidence_scores(
        self, recommendations: List[Dict[str, Any]], context_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate detailed confidence scores for recommendations"""

        overall_confidence = (
            sum(r["confidence"] for r in recommendations) / len(recommendations)
            if recommendations
            else 0.0
        )

        # Factor in context quality
        context_quality = self._assess_context_quality(context_analysis)
        adjusted_overall_confidence = overall_confidence * context_quality

        return {
            "overall_confidence": adjusted_overall_confidence,
            "context_quality": context_quality,
            "recommendation_count": len(recommendations),
            "high_confidence_actions": len(
                [r for r in recommendations if r["confidence"] > 0.8]
            ),
            "requires_approval": len(
                [r for r in recommendations if r.get("approval_required", False)]
            ),
            "confidence_distribution": self._calculate_confidence_distribution(
                recommendations
            ),
        }

    async def _generate_explanations(
        self,
        recommendations: List[Dict[str, Any]],
        context_analysis: Dict[str, Any],
        confidence_analysis: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate natural language explanations for recommendations"""

        explanations = {
            "summary": self._generate_summary_explanation(
                context_analysis, confidence_analysis
            ),
            "action_explanations": {},
            "strategy_rationale": self._generate_strategy_rationale(context_analysis),
            "risk_assessment": self._generate_risk_assessment(context_analysis),
        }

        # Generate explanation for each recommended action
        for rec in recommendations:
            action_type = rec["action_type"]
            explanations["action_explanations"][action_type] = {
                "rationale": self._explain_action_choice(
                    action_type, rec, context_analysis
                ),
                "expected_outcome": self._explain_expected_outcome(action_type, rec),
                "risks": self._explain_action_risks(action_type, rec),
                "alternatives": self._suggest_action_alternatives(
                    action_type, recommendations
                ),
            }

        # Generate AI-enhanced explanations if available
        if self.initialized and self.openai_client:
            try:
                ai_explanation = await self._get_ai_explanation(
                    context_analysis, recommendations, confidence_analysis
                )
                explanations["ai_insight"] = ai_explanation
            except Exception as e:
                self.logger.warning(f"Failed to get AI explanation: {e}")

        return explanations

    async def _get_ai_enhanced_recommendations(
        self,
        incident: Incident,
        context_analysis: Dict[str, Any],
        base_recommendations: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Get AI-enhanced recommendations using OpenAI"""

        if not self.openai_client:
            return []

        try:
            # Prepare context for AI
            ai_context = {
                "incident": {
                    "id": incident.id,
                    "src_ip": incident.src_ip,
                    "reason": incident.reason,
                    "created_at": incident.created_at.isoformat()
                    if incident.created_at
                    else None,
                },
                "context_summary": {
                    "severity": context_analysis.get("incident_severity", {}),
                    "attack_pattern": context_analysis.get("attack_pattern", {}),
                    "threat_intel": context_analysis.get("threat_intelligence", {}),
                    "ml_analysis": context_analysis.get("ml_analysis", {}),
                },
                "existing_recommendations": [
                    {
                        "action": rec["action_type"],
                        "confidence": rec["confidence"],
                        "priority": rec["priority"],
                    }
                    for rec in base_recommendations
                ],
            }

            # Generate AI recommendations
            response = await self.openai_client.chat.completions.create(
                model=settings.openai_model,
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert cybersecurity analyst specializing in incident response.
                        Analyze the incident context and provide additional response recommendations that complement
                        the existing ones. Focus on innovative or nuanced approaches that might be overlooked.

                        Return your response as a JSON array of recommendations with this structure:
                        {
                          "action_type": "action_name",
                          "confidence": 0.8,
                          "priority": 4,
                          "rationale": "Why this action is recommended",
                          "parameters": {"key": "value"}
                        }""",
                    },
                    {
                        "role": "user",
                        "content": f"Analyze this incident and provide additional recommendations:\n\n{json.dumps(ai_context, indent=2)}",
                    },
                ],
                temperature=0.3,
                max_tokens=1000,
            )

            # Parse AI response
            ai_content = response.choices[0].message.content
            if ai_content:
                try:
                    ai_recommendations = json.loads(ai_content)
                    if isinstance(ai_recommendations, list):
                        return ai_recommendations[:3]  # Limit to 3 AI recommendations
                except json.JSONDecodeError:
                    self.logger.warning("Failed to parse AI recommendations as JSON")

        except Exception as e:
            self.logger.error(f"Failed to get AI-enhanced recommendations: {e}")

        return []

    async def _get_ai_explanation(
        self,
        context_analysis: Dict[str, Any],
        recommendations: List[Dict[str, Any]],
        confidence_analysis: Dict[str, Any],
    ) -> str:
        """Get AI-generated explanation for the recommendations"""

        if not self.openai_client:
            return "AI explanations unavailable"

        try:
            response = await self.openai_client.chat.completions.create(
                model=settings.openai_model,
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert cybersecurity analyst. Explain the response recommendations
                        in clear, professional language. Focus on the reasoning behind the recommendations and
                        what the analyst should expect from implementing them.""",
                    },
                    {
                        "role": "user",
                        "content": f"""Explain these incident response recommendations:

Context: {json.dumps(context_analysis, indent=2)}
Recommendations: {json.dumps(recommendations, indent=2)}
Confidence: {json.dumps(confidence_analysis, indent=2)}

Provide a clear explanation of:
1. Why these actions were recommended
2. What outcomes to expect
3. Any risks or considerations
4. Priority and sequencing rationale""",
                    },
                ],
                temperature=0.5,
                max_tokens=800,
            )

            return response.choices[0].message.content or "No explanation generated"

        except Exception as e:
            self.logger.error(f"Failed to get AI explanation: {e}")
            return f"AI explanation unavailable: {str(e)}"

    # Helper methods
    def _score_to_severity_level(self, score: float) -> str:
        if score >= 0.8:
            return "critical"
        elif score >= 0.6:
            return "high"
        elif score >= 0.4:
            return "medium"
        else:
            return "low"

    def _score_to_scope_level(self, score: float) -> str:
        if score >= 0.7:
            return "enterprise"
        elif score >= 0.5:
            return "department"
        elif score >= 0.3:
            return "team"
        else:
            return "individual"

    def _confidence_to_level(self, confidence: float) -> RecommendationConfidence:
        if confidence >= 0.9:
            return RecommendationConfidence.VERY_HIGH
        elif confidence >= 0.7:
            return RecommendationConfidence.HIGH
        elif confidence >= 0.5:
            return RecommendationConfidence.MEDIUM
        elif confidence >= 0.3:
            return RecommendationConfidence.LOW
        else:
            return RecommendationConfidence.VERY_LOW

    def _determine_response_strategy(
        self, confidence_analysis: Dict[str, Any]
    ) -> ResponseStrategy:
        overall_confidence = confidence_analysis.get("overall_confidence", 0.0)
        high_confidence_count = confidence_analysis.get("high_confidence_actions", 0)

        if overall_confidence > 0.8 and high_confidence_count > 2:
            return ResponseStrategy.IMMEDIATE
        elif overall_confidence > 0.6:
            return ResponseStrategy.CAUTIOUS
        elif overall_confidence > 0.4:
            return ResponseStrategy.INVESTIGATIVE
        else:
            return ResponseStrategy.COLLABORATIVE

    def _estimate_response_duration(self, recommendations: List[Dict[str, Any]]) -> int:
        return sum(rec.get("estimated_duration", 300) for rec in recommendations)

    def _assess_context_quality(self, context_analysis: Dict[str, Any]) -> float:
        quality_score = 0.0
        max_score = 0.0

        # Check each context component
        components = [
            ("incident_severity", 0.2),
            ("attack_pattern", 0.2),
            ("threat_intelligence", 0.15),
            ("ml_analysis", 0.15),
            ("historical_patterns", 0.15),
            ("compliance_context", 0.1),
            ("previous_responses", 0.05),
        ]

        for component, weight in components:
            max_score += weight
            if component in context_analysis:
                data = context_analysis[component]
                if isinstance(data, dict) and data.get("confidence", 0) > 0:
                    quality_score += weight * data.get("confidence", 0.5)
                else:
                    quality_score += weight * 0.5

        return quality_score / max_score if max_score > 0 else 0.5

    def _generate_summary_explanation(
        self, context_analysis: Dict[str, Any], confidence_analysis: Dict[str, Any]
    ) -> str:
        """Generate a summary explanation of the recommendations"""

        severity = context_analysis.get("incident_severity", {})
        pattern = context_analysis.get("attack_pattern", {})
        confidence = confidence_analysis.get("overall_confidence", 0.0)

        return f"""Based on the analysis, this appears to be a {severity.get('level', 'medium')} severity
        {pattern.get('pattern', 'unknown')} attack with {self._confidence_to_level(confidence).value} confidence
        in our recommendations. The AI system suggests {confidence_analysis.get('recommendation_count', 0)}
        response actions with an overall confidence of {confidence:.1%}."""

    def _generate_strategy_rationale(self, context_analysis: Dict[str, Any]) -> str:
        """Generate rationale for the response strategy"""

        pattern = context_analysis.get("attack_pattern", {})
        severity = context_analysis.get("incident_severity", {})

        return f"""The recommended strategy is based on the identified {pattern.get('pattern', 'unknown')}
        attack pattern with {pattern.get('confidence', 0.0):.1%} confidence. Given the {severity.get('level', 'medium')}
        severity level, immediate containment actions are prioritized while preserving evidence for forensic analysis."""

    def _generate_risk_assessment(self, context_analysis: Dict[str, Any]) -> str:
        """Generate risk assessment for the recommendations"""

        impact = context_analysis.get("impact_scope", {})
        compliance = context_analysis.get("compliance_context", {})

        return f"""Risk assessment indicates {impact.get('scope_level', 'moderate')} scope impact affecting
        {impact.get('unique_sources', 1)} source(s). Compliance requirements include {', '.join(compliance.get('frameworks', ['SOC2']))}
        with maximum response time of {compliance.get('max_response_time', 3600)//60} minutes."""

    # Additional helper methods for completeness
    def _extract_pattern_indicators(
        self, events: List[Event], pattern: str
    ) -> List[str]:
        """Extract indicators supporting the identified pattern"""
        indicators = []

        if pattern == "brute_force":
            indicators.extend(["multiple_failed_attempts", "sequential_login_tries"])
        elif pattern == "ddos":
            indicators.extend(["high_connection_rate", "bandwidth_spike"])

        return indicators

    def _calculate_attack_duration(self, events: List[Event]) -> Dict[str, Any]:
        """Calculate the duration of the attack"""
        if not events:
            return {"duration_seconds": 0, "start_time": None, "end_time": None}

        timestamps = [e.ts for e in events if e.ts]
        if not timestamps:
            return {"duration_seconds": 0, "start_time": None, "end_time": None}

        start_time = min(timestamps)
        end_time = max(timestamps)
        duration = (end_time - start_time).total_seconds()

        return {
            "duration_seconds": duration,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
        }

    def _analyze_intensity_pattern(self, events: List[Event]) -> str:
        """Analyze the intensity pattern of the attack"""
        if len(events) > 100:
            return "high_intensity"
        elif len(events) > 20:
            return "medium_intensity"
        else:
            return "low_intensity"

    def _calculate_persistence_score(self, events: List[Event]) -> float:
        """Calculate how persistent the attack is"""
        if not events:
            return 0.0

        duration_data = self._calculate_attack_duration(events)
        duration_hours = duration_data["duration_seconds"] / 3600

        if duration_hours > 24:
            return 0.9
        elif duration_hours > 8:
            return 0.7
        elif duration_hours > 2:
            return 0.5
        else:
            return 0.3

    def _extract_historical_recommendations(
        self, workflows: List[ResponseWorkflow]
    ) -> List[str]:
        """Extract recommendations from historical successful workflows"""
        recommendations = []

        for workflow in workflows:
            if workflow.success_rate and workflow.success_rate > 0.8:
                recommendations.append(
                    f"Consider using {workflow.playbook_name} pattern"
                )

        return recommendations

    def _estimate_action_duration(
        self, action_type: str, context_analysis: Dict[str, Any]
    ) -> int:
        """Estimate duration for specific action"""
        base_durations = {
            "block_ip_advanced": 120,
            "isolate_host_advanced": 300,
            "memory_dump_collection": 600,
            "deploy_firewall_rules": 180,
            "dns_sinkhole": 90,
        }

        return base_durations.get(action_type, 300)

    def _get_safety_considerations(self, action_type: str) -> List[str]:
        """Get safety considerations for action type"""
        safety_db = {
            "isolate_host_advanced": [
                "May disrupt legitimate user access",
                "Requires rollback plan",
            ],
            "deploy_firewall_rules": [
                "Could block legitimate traffic",
                "Test rules in staging first",
            ],
            "account_disable": [
                "May impact business operations",
                "Notify affected users",
            ],
        }

        return safety_db.get(action_type, ["Standard safety protocols apply"])

    def _generate_rollback_plan(
        self, action_type: str, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate rollback plan for action"""
        rollback_plans = {
            "block_ip_advanced": {
                "action": "unblock_ip",
                "parameters": {"ip_address": parameters.get("ip_address")},
                "estimated_time": 60,
            },
            "isolate_host_advanced": {
                "action": "un_isolate_host",
                "parameters": {"host_identifier": parameters.get("host_identifier")},
                "estimated_time": 120,
            },
        }

        return rollback_plans.get(
            action_type, {"action": "manual_rollback", "estimated_time": 300}
        )

    def _calculate_confidence_distribution(
        self, recommendations: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        """Calculate distribution of confidence levels"""
        distribution = {"very_high": 0, "high": 0, "medium": 0, "low": 0, "very_low": 0}

        for rec in recommendations:
            level = self._confidence_to_level(rec["confidence"]).value
            distribution[level] += 1

        return distribution

    def _explain_action_choice(
        self,
        action_type: str,
        recommendation: Dict[str, Any],
        context_analysis: Dict[str, Any],
    ) -> str:
        """Explain why this action was chosen"""

        explanations = {
            "block_ip_advanced": f"IP blocking recommended due to {context_analysis.get('attack_pattern', {}).get('pattern', 'suspicious')} pattern with {recommendation['confidence']:.1%} confidence",
            "isolate_host_advanced": f"Host isolation recommended to prevent lateral movement with {recommendation['confidence']:.1%} confidence",
            "memory_dump_collection": "Memory dump collection recommended for forensic analysis and malware detection",
        }

        return explanations.get(
            action_type,
            f"Action recommended based on pattern analysis with {recommendation['confidence']:.1%} confidence",
        )

    def _explain_expected_outcome(
        self, action_type: str, recommendation: Dict[str, Any]
    ) -> str:
        """Explain expected outcome of the action"""

        outcomes = {
            "block_ip_advanced": "Will prevent further attacks from this IP address and related networks",
            "isolate_host_advanced": "Will prevent malware spread while maintaining system for analysis",
            "memory_dump_collection": "Will provide forensic evidence for threat analysis and attribution",
        }

        return outcomes.get(action_type, "Will help contain and investigate the threat")

    def _explain_action_risks(
        self, action_type: str, recommendation: Dict[str, Any]
    ) -> List[str]:
        """Explain risks associated with the action"""

        risks = {
            "block_ip_advanced": [
                "May block legitimate users",
                "Could trigger attacker to change tactics",
            ],
            "isolate_host_advanced": [
                "May disrupt business operations",
                "Could alert sophisticated attackers",
            ],
            "deploy_firewall_rules": [
                "Risk of blocking legitimate traffic",
                "Potential performance impact",
            ],
        }

        return risks.get(action_type, ["Standard operational risks apply"])

    def _suggest_action_alternatives(
        self, action_type: str, all_recommendations: List[Dict[str, Any]]
    ) -> List[str]:
        """Suggest alternative actions"""

        alternatives = {
            "block_ip_advanced": ["traffic_redirection", "dns_sinkhole"],
            "isolate_host_advanced": ["memory_dump_collection", "process_termination"],
            "deploy_firewall_rules": ["traffic_redirection", "block_ip_advanced"],
        }

        # Filter out alternatives that are already recommended
        recommended_actions = {rec["action_type"] for rec in all_recommendations}
        available_alternatives = [
            alt
            for alt in alternatives.get(action_type, [])
            if alt not in recommended_actions
        ]

        return available_alternatives


# Global instance
ai_response_advisor = AIResponseAdvisor()


async def get_ai_advisor() -> AIResponseAdvisor:
    """Get the global AI response advisor instance"""
    if not ai_response_advisor.initialized:
        await ai_response_advisor.initialize()
    return ai_response_advisor

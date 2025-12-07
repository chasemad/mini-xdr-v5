"""
Enhanced Containment Engine with AI Agent Integration
Provides intelligent containment decisions using ML scores and threat intelligence
"""
import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

# Import models directly from models.py to avoid circular import issues
from . import models as db_models
from .config import settings

logger = logging.getLogger(__name__)


@dataclass
class ContainmentDecision:
    """Represents a containment decision with justification"""

    should_contain: bool
    confidence: float  # 0.0 to 1.0
    risk_score: float  # 0.0 to 1.0
    duration_seconds: int
    method: str  # "rule_based", "ml_driven", "ai_agent"
    reasoning: List[str]
    escalation_level: str  # "low", "medium", "high", "critical"
    policy_id: Optional[str] = None
    threat_category: Optional[str] = None


class EnhancedContainmentEngine:
    """Advanced containment engine with ML and threat intelligence integration"""

    def __init__(self, threat_intel=None, ml_detector=None):
        self.threat_intel = threat_intel
        self.ml_detector = ml_detector
        self.base_thresholds = {
            "low": 0.2,
            "medium": 0.5,
            "high": 0.8,
            "critical": 0.95,
        }

        # Containment durations (seconds)
        self.default_durations = {
            "low": 300,  # 5 minutes
            "medium": 1800,  # 30 minutes
            "high": 3600,  # 1 hour
            "critical": 7200,  # 2 hours
        }

    async def evaluate_containment(
        self, incident, recent_events: List, db: AsyncSession = None
    ) -> ContainmentDecision:
        """
        Comprehensive containment evaluation using multiple factors

        Args:
            incident: The incident to evaluate
            recent_events: Recent events from the source IP
            db: Database session for policy queries

        Returns:
            ContainmentDecision with recommendation and justification
        """
        reasoning = []
        risk_factors = {}

        # 1. Traditional threshold-based evaluation
        threshold_risk = await self._evaluate_threshold_risk(incident, recent_events)
        risk_factors["threshold"] = threshold_risk
        reasoning.append(f"Threshold analysis: {threshold_risk:.2f} risk")

        # 2. ML-based anomaly detection (if available)
        ml_risk = 0.0
        if self.ml_detector:
            try:
                ml_risk = await self.ml_detector.calculate_anomaly_score(
                    incident.src_ip, recent_events
                )
                risk_factors["ml_anomaly"] = ml_risk
                reasoning.append(f"ML anomaly score: {ml_risk:.2f}")
            except Exception as e:
                logger.warning(f"ML evaluation failed: {e}")
                reasoning.append("ML evaluation unavailable")

        # 3. Threat intelligence lookup (if available)
        threat_intel_risk = 0.0
        threat_category = None
        if self.threat_intel:
            try:
                intel_result = await self.threat_intel.lookup_ip(incident.src_ip)
                # ThreatIntelResult is a dataclass, use attribute access
                threat_intel_risk = (
                    getattr(intel_result, "risk_score", 0.0) if intel_result else 0.0
                )
                threat_category = (
                    getattr(intel_result, "category", "unknown")
                    if intel_result
                    else "unknown"
                )
                risk_factors["threat_intel"] = threat_intel_risk
                reasoning.append(
                    f"Threat intel risk: {threat_intel_risk:.2f} ({threat_category})"
                )
            except Exception as e:
                logger.warning(f"Threat intel lookup failed: {e}")
                reasoning.append("Threat intelligence unavailable")

        # 4. Behavioral pattern analysis
        behavioral_risk = await self._analyze_behavioral_patterns(recent_events)
        risk_factors["behavioral"] = behavioral_risk
        reasoning.append(f"Behavioral analysis: {behavioral_risk:.2f}")

        # 5. Check policy overrides (if database available)
        policy_decision = None
        if db:
            policy_decision = await self._check_policy_overrides(
                incident, risk_factors, db
            )

        # 6. Calculate composite risk score
        composite_risk = self._calculate_composite_risk(risk_factors)

        # 7. Determine escalation level and containment decision (pass incident for enhanced logic)
        escalation_level = self._determine_escalation_level(composite_risk, incident)
        should_contain = composite_risk >= self.base_thresholds["medium"]

        # Policy override takes precedence
        if policy_decision:
            should_contain = policy_decision["should_contain"]
            reasoning.append(f"Policy override: {policy_decision['reasoning']}")

        # 8. Determine confidence based on available data
        confidence = self._calculate_confidence(risk_factors, len(recent_events))

        # 9. Select duration based on risk level
        duration = self.default_durations[escalation_level]

        return ContainmentDecision(
            should_contain=should_contain,
            confidence=confidence,
            risk_score=composite_risk,
            duration_seconds=duration,
            method="enhanced_engine",
            reasoning=reasoning,
            escalation_level=escalation_level,
            policy_id=policy_decision.get("policy_id") if policy_decision else None,
            threat_category=threat_category,
        )

    async def _evaluate_threshold_risk(self, incident, recent_events: List) -> float:
        """Evaluate risk based on traditional thresholds"""

        # Count failed login attempts
        failed_logins = sum(
            1 for event in recent_events if event.eventid == "cowrie.login.failed"
        )

        # Calculate time span
        if len(recent_events) > 1:
            time_span = (recent_events[0].ts - recent_events[-1].ts).total_seconds()
            rate = failed_logins / max(time_span / 60, 1)  # attempts per minute
        else:
            rate = 1

        # Normalize to 0-1 risk score
        # High rate (>10/min) = high risk
        risk_score = min(rate / 10.0, 1.0)

        # Boost for high volume
        if failed_logins > 50:
            risk_score = min(risk_score + 0.3, 1.0)

        return risk_score

    async def _analyze_behavioral_patterns(self, recent_events: List) -> float:
        """Analyze behavioral patterns for risk assessment"""
        if not recent_events:
            return 0.0

        risk_indicators = []

        # 1. Check for password spraying patterns
        usernames = set()
        passwords = set()

        for event in recent_events:
            if hasattr(event, "raw") and event.raw:
                raw_data = event.raw if isinstance(event.raw, dict) else {}
                if "username" in raw_data:
                    usernames.add(raw_data["username"])
                if "password" in raw_data:
                    passwords.add(raw_data["password"])

        if len(passwords) > 20 and len(usernames) < 5:
            risk_indicators.append(0.8)  # Strong password spray indicator

        # 2. Check for rapid-fire attempts
        if len(recent_events) > 100:
            risk_indicators.append(0.7)  # High volume

        # 3. Check for persistence (long attack duration)
        if len(recent_events) > 1:
            duration_hours = (
                recent_events[0].ts - recent_events[-1].ts
            ).total_seconds() / 3600
            if duration_hours > 2:
                risk_indicators.append(0.6)  # Persistent attack

        # 4. Check for unusual ports or protocols
        unique_ports = set(event.dst_port for event in recent_events if event.dst_port)
        if len(unique_ports) > 5:
            risk_indicators.append(0.4)  # Port scanning behavior

        # Return maximum risk indicator or 0.1 baseline
        return max(risk_indicators) if risk_indicators else 0.1

    async def _check_policy_overrides(
        self, incident, risk_factors: Dict[str, float], db: AsyncSession
    ) -> Optional[Dict[str, Any]]:
        """Check for policy-based containment overrides"""
        try:
            # Query active policies ordered by priority
            query = (
                select(db_models.ContainmentPolicy)
                .where(db_models.ContainmentPolicy.status == "active")
                .order_by(db_models.ContainmentPolicy.priority)
            )

            result = await db.execute(query)
            policies = result.scalars().all()

            for policy in policies:
                if await self._policy_matches_incident(policy, incident, risk_factors):
                    # Extract actions from policy
                    actions = policy.actions or {}
                    should_contain = any(
                        action in ["block_ip", "isolate_host"]
                        for action in actions.keys()
                    )

                    return {
                        "should_contain": should_contain,
                        "policy_id": policy.name,
                        "reasoning": f"Policy '{policy.name}' triggered",
                    }

        except Exception as e:
            logger.error(f"Policy evaluation error: {e}")

        return None

    async def _policy_matches_incident(
        self,
        policy,
        incident,
        risk_factors: Dict[str, float],
    ) -> bool:
        """Check if a policy matches the current incident"""
        conditions = policy.conditions or {}

        # Check risk score conditions
        if "risk_score" in conditions:
            composite_risk = self._calculate_composite_risk(risk_factors)
            if not self._evaluate_condition(composite_risk, conditions["risk_score"]):
                return False

        # Check threat category conditions
        if "threat_category" in conditions:
            if incident.threat_category != conditions["threat_category"]:
                return False

        # Check escalation level conditions
        if "escalation_level" in conditions:
            if incident.escalation_level not in conditions["escalation_level"]:
                return False

        return True

    def _evaluate_condition(self, value: float, condition: Any) -> bool:
        """Evaluate a single policy condition"""
        if isinstance(condition, (int, float)):
            return value >= condition
        elif isinstance(condition, dict):
            if "min" in condition and value < condition["min"]:
                return False
            if "max" in condition and value > condition["max"]:
                return False
            return True
        elif isinstance(condition, str):
            # String conditions like "> 0.5"
            try:
                if condition.startswith(">"):
                    return value > float(condition[1:].strip())
                elif condition.startswith("<"):
                    return value < float(condition[1:].strip())
                elif condition.startswith(">="):
                    return value >= float(condition[2:].strip())
                elif condition.startswith("<="):
                    return value <= float(condition[2:].strip())
            except ValueError:
                pass

        return True

    def _calculate_composite_risk(self, risk_factors: Dict[str, float]) -> float:
        """Calculate weighted composite risk score"""
        if not risk_factors:
            return 0.0

        # Weights for different risk factors
        weights = {
            "threshold": 0.3,
            "ml_anomaly": 0.4,
            "threat_intel": 0.2,
            "behavioral": 0.1,
        }

        weighted_sum = 0.0
        total_weight = 0.0

        for factor, risk in risk_factors.items():
            weight = weights.get(factor, 0.1)
            weighted_sum += risk * weight
            total_weight += weight

        return min(weighted_sum / max(total_weight, 0.1), 1.0)

    def _determine_escalation_level(self, risk_score: float, incident=None) -> str:
        """Determine escalation level based on risk score and threat indicators"""
        base_level = "low"

        if risk_score >= self.base_thresholds["critical"]:
            base_level = "critical"
        elif risk_score >= self.base_thresholds["high"]:
            base_level = "high"
        elif risk_score >= self.base_thresholds["medium"]:
            base_level = "medium"

        # Check incident for critical threat categories that should elevate priority
        if incident and incident.threat_category:
            critical_threats = [
                "malware",
                "ransomware",
                "data_exfiltration",
                "privilege_escalation",
                "lateral_movement",
                "backdoor",
                "trojan",
                "cryptominer",
            ]
            threat_cat = incident.threat_category.lower()

            if any(ct in threat_cat for ct in critical_threats):
                # Ensure at least HIGH priority for critical threats
                if base_level not in ["critical", "high"]:
                    logger.info(
                        f"Elevating escalation to HIGH due to critical threat category: {threat_cat}"
                    )
                    return "high"

        return base_level

    def _calculate_confidence(
        self, risk_factors: Dict[str, float], event_count: int
    ) -> float:
        """Calculate confidence in the containment decision"""
        base_confidence = 0.5

        # More data sources = higher confidence
        data_source_count = len(risk_factors)
        confidence_boost = data_source_count * 0.1

        # More events = higher confidence (up to a point)
        event_confidence = min(event_count / 50.0, 0.3)

        # ML and threat intel provide higher confidence
        if "ml_anomaly" in risk_factors:
            confidence_boost += 0.2
        if "threat_intel" in risk_factors:
            confidence_boost += 0.15

        return min(base_confidence + confidence_boost + event_confidence, 1.0)


# Example policies for seeding the database
DEFAULT_POLICIES = [
    {
        "name": "high_risk_ssh",
        "description": "High-risk SSH brute force attacks",
        "priority": 10,
        "conditions": {
            "risk_score": {"min": 0.8},
            "threat_category": ["brute_force", "password_spray"],
        },
        "actions": {
            "block_ip": {"duration": 3600},
            "isolate_host": {"level": "hard"},
            "notify_analyst": {"message": "Critical SSH attack detected"},
        },
        "agent_override": True,
        "escalation_threshold": 0.9,
    },
    {
        "name": "low_risk_probe",
        "description": "Low-risk reconnaissance activities",
        "priority": 100,
        "conditions": {"risk_score": {"max": 0.3}},
        "actions": {"monitor_only": True},
        "agent_override": False,
        "escalation_threshold": 0.5,
    },
]

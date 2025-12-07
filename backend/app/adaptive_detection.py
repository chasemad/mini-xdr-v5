"""
Intelligent Adaptive Attack Detection Engine
Analyzes behavioral patterns to identify suspicious activity without hardcoded rules
"""
import asyncio
import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# from scipy import stats  # Disabled for macOS compatibility
from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession

# Import models lazily to avoid circular import issues
try:
    from .models import Event, Incident
except ImportError:
    # Handle circular import during module loading
    Event = None
    Incident = None
from .config import settings

logger = logging.getLogger(__name__)


@dataclass
class BehavioralPattern:
    """Represents a detected behavioral pattern"""

    pattern_type: str
    src_ip: str
    severity: str
    confidence: float
    indicators: Dict[str, Any]
    features: Dict[str, float]
    description: str
    threat_score: float


class RapidEnumerationDetector:
    """Detects systematic scanning/enumeration patterns"""

    def __init__(self):
        self.min_requests = 5
        self.max_time_span = 300  # 5 minutes
        self.diversity_threshold = 0.7

    async def evaluate(self, features: Dict[str, float]) -> float:
        """Evaluate rapid enumeration patterns"""
        try:
            # Look for rapid, diverse requests
            request_rate = features.get("request_rate_per_minute", 0)
            path_diversity = features.get("path_diversity_ratio", 0)
            error_rate = features.get("error_rate", 0)
            unique_params = features.get("unique_parameters", 0)

            # Score based on enumeration indicators
            score = 0.0

            # High request rate
            if request_rate > 10:
                score += 0.3
            elif request_rate > 5:
                score += 0.2

            # High path diversity (scanning multiple endpoints)
            if path_diversity > 0.8:
                score += 0.3
            elif path_diversity > 0.5:
                score += 0.2

            # High error rate (probing non-existent resources)
            if error_rate > 0.7:
                score += 0.3
            elif error_rate > 0.4:
                score += 0.2

            # Parameter enumeration
            if unique_params > 20:
                score += 0.2
            elif unique_params > 10:
                score += 0.1

            return min(score, 1.0)

        except Exception as e:
            logger.error(f"Rapid enumeration evaluation error: {e}")
            return 0.0


class ErrorSeekingDetector:
    """Detects attempts to trigger application errors for information disclosure"""

    def __init__(self):
        self.error_threshold = 0.3

    async def evaluate(self, features: Dict[str, float]) -> float:
        """Evaluate error-seeking behavior"""
        try:
            error_rate = features.get("error_rate", 0)
            sql_injection_attempts = features.get("sql_injection_attempts", 0)
            path_traversal_attempts = features.get("path_traversal_attempts", 0)
            malformed_requests = features.get("malformed_requests", 0)

            score = 0.0

            # High error rate with specific attack patterns
            if error_rate > 0.6 and sql_injection_attempts > 0:
                score += 0.4
            elif error_rate > 0.3:
                score += 0.2

            # Specific attack patterns
            if sql_injection_attempts > 3:
                score += 0.3
            if path_traversal_attempts > 2:
                score += 0.3
            if malformed_requests > 5:
                score += 0.2

            return min(score, 1.0)

        except Exception as e:
            logger.error(f"Error seeking evaluation error: {e}")
            return 0.0


class ProgressiveComplexityDetector:
    """Detects escalating attack sophistication over time"""

    def __init__(self):
        self.complexity_threshold = 0.5

    async def evaluate(self, features: Dict[str, float]) -> float:
        """Evaluate progressive complexity patterns"""
        try:
            parameter_complexity = features.get("parameter_complexity_trend", 0)
            attack_sophistication = features.get("attack_sophistication", 0)
            payload_size_trend = features.get("payload_size_trend", 0)
            encoding_attempts = features.get("encoding_attempts", 0)

            score = 0.0

            # Increasing complexity over time
            if parameter_complexity > 0.7:
                score += 0.3
            if attack_sophistication > 0.6:
                score += 0.3
            if payload_size_trend > 0.5:
                score += 0.2
            if encoding_attempts > 2:
                score += 0.2

            return min(score, 1.0)

        except Exception as e:
            logger.error(f"Progressive complexity evaluation error: {e}")
            return 0.0


class TimingAnomalyDetector:
    """Identifies non-human timing patterns"""

    def __init__(self):
        self.human_variance_threshold = 0.3

    async def evaluate(self, features: Dict[str, float]) -> float:
        """Evaluate timing anomalies"""
        try:
            request_interval_variance = features.get("request_interval_variance", 0)
            sub_second_requests = features.get("sub_second_requests", 0)
            perfectly_timed_requests = features.get("perfectly_timed_requests", 0)
            burst_patterns = features.get("burst_patterns", 0)

            score = 0.0

            # Low variance suggests automated behavior
            if request_interval_variance < 0.1:
                score += 0.3
            elif request_interval_variance < 0.2:
                score += 0.2

            # High frequency automated requests
            if sub_second_requests > 5:
                score += 0.3
            elif sub_second_requests > 2:
                score += 0.2

            # Perfectly timed requests (scripted)
            if perfectly_timed_requests > 3:
                score += 0.3

            # Burst patterns
            if burst_patterns > 2:
                score += 0.2

            return min(score, 1.0)

        except Exception as e:
            logger.error(f"Timing anomaly evaluation error: {e}")
            return 0.0


class ParameterFuzzingDetector:
    """Detects parameter manipulation and fuzzing attempts"""

    def __init__(self):
        self.fuzzing_threshold = 0.4

    async def evaluate(self, features: Dict[str, float]) -> float:
        """Evaluate parameter fuzzing patterns"""
        try:
            unique_parameters = features.get("unique_parameters", 0)
            parameter_mutations = features.get("parameter_mutations", 0)
            boundary_value_tests = features.get("boundary_value_tests", 0)
            injection_patterns = features.get("injection_patterns", 0)

            score = 0.0

            # High parameter diversity
            if unique_parameters > 30:
                score += 0.3
            elif unique_parameters > 15:
                score += 0.2

            # Parameter mutations
            if parameter_mutations > 10:
                score += 0.3
            elif parameter_mutations > 5:
                score += 0.2

            # Boundary value testing
            if boundary_value_tests > 3:
                score += 0.2

            # Injection patterns
            if injection_patterns > 5:
                score += 0.3

            return min(score, 1.0)

        except Exception as e:
            logger.error(f"Parameter fuzzing evaluation error: {e}")
            return 0.0


class BehaviorAnalyzer:
    """Analyzes request patterns to identify suspicious behavior without hardcoded rules"""

    def __init__(self):
        # Behavioral pattern detectors
        self.pattern_detectors = {
            "rapid_enumeration": RapidEnumerationDetector(),
            "error_seeking": ErrorSeekingDetector(),
            "progressive_complexity": ProgressiveComplexityDetector(),
            "timing_anomalies": TimingAnomalyDetector(),
            "parameter_fuzzing": ParameterFuzzingDetector(),
        }

        # Adaptive threshold (can be learned over time)
        self.adaptive_threshold = 0.6
        self.min_events_for_analysis = 3

    async def analyze_ip_behavior(
        self, db: AsyncSession, src_ip: str, window_minutes: int = 30
    ) -> Optional[BehavioralPattern]:
        """Analyze behavioral patterns for suspicious activity"""

        # Get recent events for this IP
        recent_events = await self._get_recent_events(db, src_ip, window_minutes)

        if len(recent_events) < self.min_events_for_analysis:
            return None

        # Extract behavioral features
        features = await self._extract_behavioral_features(recent_events)

        # Evaluate with each detector
        anomaly_scores = {}
        for detector_name, detector in self.pattern_detectors.items():
            score = await detector.evaluate(features)
            anomaly_scores[detector_name] = score

        # Calculate composite threat score
        threat_score = self._calculate_composite_score(anomaly_scores)

        if threat_score > self.adaptive_threshold:
            return self._create_behavioral_incident(
                src_ip, anomaly_scores, features, threat_score
            )

        return None

    async def _get_recent_events(
        self, db: AsyncSession, src_ip: str, window_minutes: int
    ) -> List:
        """Get recent events for behavioral analysis"""
        window_start = datetime.now(timezone.utc) - timedelta(minutes=window_minutes)

        query = (
            select(Event)
            .where(and_(Event.src_ip == src_ip, Event.ts >= window_start))
            .order_by(Event.ts.desc())
        )

        result = await db.execute(query)
        return result.scalars().all()

    async def _extract_behavioral_features(self, events: List) -> Dict[str, float]:
        """Extract behavioral features from events"""
        if not events:
            return {}

        features = {}

        # Time-based features
        time_span = (
            (events[0].ts - events[-1].ts).total_seconds() if len(events) > 1 else 1
        )
        features["request_rate_per_minute"] = len(events) / max(time_span / 60, 1)

        # Request timing analysis
        if len(events) > 1:
            intervals = []
            for i in range(len(events) - 1):
                interval = (events[i].ts - events[i + 1].ts).total_seconds()
                intervals.append(interval)

            features["request_interval_variance"] = (
                np.var(intervals) if intervals else 0
            )
            features["sub_second_requests"] = sum(1 for i in intervals if i < 1)
            features["perfectly_timed_requests"] = sum(
                1 for i in intervals if abs(i - round(i)) < 0.1
            )

            # Detect burst patterns
            burst_count = 0
            current_burst = 0
            for interval in intervals:
                if interval < 2:  # Less than 2 seconds
                    current_burst += 1
                else:
                    if current_burst >= 3:
                        burst_count += 1
                    current_burst = 0
            features["burst_patterns"] = burst_count

        # Web-specific features
        web_events = [
            e
            for e in events
            if "http" in str(e.raw).lower() or e.dst_port in [80, 443, 8080]
        ]
        if web_events:
            features.update(await self._extract_web_features(web_events))

        # SSH-specific features
        ssh_events = [e for e in events if e.eventid.startswith("cowrie")]
        if ssh_events:
            features.update(await self._extract_ssh_features(ssh_events))

        return features

    async def _extract_web_features(self, web_events: List) -> Dict[str, float]:
        """Extract web-specific behavioral features"""
        features = {}

        # Parse request details
        paths = []
        parameters = []
        user_agents = []
        status_codes = []
        attack_indicators = []

        for event in web_events:
            try:
                raw_data = (
                    event.raw
                    if isinstance(event.raw, dict)
                    else json.loads(event.raw)
                    if event.raw
                    else {}
                )

                # Extract request components
                if "path" in raw_data:
                    paths.append(raw_data["path"])
                if "parameters" in raw_data:
                    parameters.extend(raw_data["parameters"])
                if "user_agent" in raw_data:
                    user_agents.append(raw_data["user_agent"])
                if "status_code" in raw_data:
                    status_codes.append(raw_data["status_code"])
                if "attack_indicators" in raw_data:
                    attack_indicators.extend(raw_data["attack_indicators"])

            except (json.JSONDecodeError, TypeError):
                continue

        # Path diversity
        unique_paths = len(set(paths))
        total_requests = len(web_events)
        features["path_diversity_ratio"] = unique_paths / max(total_requests, 1)

        # Error rate
        error_responses = sum(1 for code in status_codes if code >= 400)
        features["error_rate"] = error_responses / max(total_requests, 1)

        # Parameter analysis
        features["unique_parameters"] = len(set(parameters))
        features["parameter_mutations"] = self._count_parameter_mutations(parameters)

        # Attack pattern analysis
        features["sql_injection_attempts"] = sum(
            1 for indicator in attack_indicators if "sql" in indicator.lower()
        )
        features["path_traversal_attempts"] = sum(
            1 for indicator in attack_indicators if "traversal" in indicator.lower()
        )
        features["injection_patterns"] = len(
            [
                i
                for i in attack_indicators
                if any(
                    pattern in i.lower()
                    for pattern in ["injection", "script", "payload"]
                )
            ]
        )

        # Sophistication analysis
        features["attack_sophistication"] = self._calculate_attack_sophistication(
            attack_indicators
        )
        features["encoding_attempts"] = sum(
            1 for indicator in attack_indicators if "encoding" in indicator.lower()
        )

        # Malformed requests
        features["malformed_requests"] = sum(1 for code in status_codes if code == 400)

        return features

    async def _extract_ssh_features(self, ssh_events: List) -> Dict[str, float]:
        """Extract SSH-specific behavioral features"""
        features = {}

        # Credential analysis
        usernames = []
        passwords = []
        commands = []

        for event in ssh_events:
            try:
                raw_data = (
                    event.raw
                    if isinstance(event.raw, dict)
                    else json.loads(event.raw)
                    if event.raw
                    else {}
                )

                if event.eventid == "cowrie.login.failed":
                    if "username" in raw_data:
                        usernames.append(raw_data["username"])
                    if "password" in raw_data:
                        passwords.append(raw_data["password"])
                elif event.eventid == "cowrie.command.input":
                    if "input" in raw_data:
                        commands.append(raw_data["input"])

            except (json.JSONDecodeError, TypeError):
                continue

        # Credential diversity analysis
        features["username_diversity"] = len(set(usernames))
        features["password_diversity"] = len(set(passwords))
        features["command_diversity"] = len(set(commands))

        # Sophistication analysis
        if passwords:
            password_lengths = [len(str(p)) for p in passwords]
            features["password_complexity_avg"] = np.mean(password_lengths)
            features["password_complexity_variance"] = np.var(password_lengths)

        return features

    def _count_parameter_mutations(self, parameters: List[str]) -> int:
        """Count parameter mutation attempts"""
        if not parameters:
            return 0

        # Look for systematic parameter variations
        param_groups = defaultdict(list)
        for param in parameters:
            base_param = param.split("=")[0] if "=" in param else param
            param_groups[base_param].append(param)

        mutations = 0
        for param_name, variations in param_groups.items():
            if len(variations) > 3:  # Multiple variations of same parameter
                mutations += len(variations)

        return mutations

    def _calculate_attack_sophistication(self, attack_indicators: List[str]) -> float:
        """Calculate attack sophistication based on indicators"""
        if not attack_indicators:
            return 0.0

        sophistication_score = 0.0
        total_indicators = len(attack_indicators)

        # Score based on attack complexity
        for indicator in attack_indicators:
            indicator_lower = indicator.lower()

            # Basic attacks
            if any(basic in indicator_lower for basic in ["simple", "basic", "common"]):
                sophistication_score += 0.1
            # Intermediate attacks
            elif any(
                intermediate in indicator_lower
                for intermediate in ["bypass", "evasion", "encoding"]
            ):
                sophistication_score += 0.3
            # Advanced attacks
            elif any(
                advanced in indicator_lower
                for advanced in ["obfuscation", "polyglot", "advanced"]
            ):
                sophistication_score += 0.5
            else:
                sophistication_score += 0.2

        return min(sophistication_score / max(total_indicators, 1), 1.0)

    def _calculate_composite_score(self, anomaly_scores: Dict[str, float]) -> float:
        """Calculate composite threat score from individual detector scores"""
        if not anomaly_scores:
            return 0.0

        # Weighted scoring
        weights = {
            "rapid_enumeration": 0.25,
            "error_seeking": 0.25,
            "progressive_complexity": 0.20,
            "timing_anomalies": 0.15,
            "parameter_fuzzing": 0.15,
        }

        weighted_sum = 0.0
        total_weight = 0.0

        for detector_name, score in anomaly_scores.items():
            weight = weights.get(detector_name, 0.1)
            weighted_sum += score * weight
            total_weight += weight

        if total_weight == 0:
            return 0.0

        # Apply non-linear scaling for high-confidence detections
        composite_score = weighted_sum / total_weight

        # Boost score if multiple detectors agree
        active_detectors = sum(1 for score in anomaly_scores.values() if score > 0.3)
        if active_detectors >= 3:
            composite_score *= 1.2
        elif active_detectors >= 2:
            composite_score *= 1.1

        return min(composite_score, 1.0)

    def _create_behavioral_incident(
        self,
        src_ip: str,
        anomaly_scores: Dict[str, float],
        features: Dict[str, float],
        threat_score: float,
    ) -> BehavioralPattern:
        """Create behavioral pattern incident"""

        # Determine primary pattern
        primary_pattern = max(anomaly_scores.items(), key=lambda x: x[1])
        pattern_type = primary_pattern[0]
        confidence = primary_pattern[1]

        # Determine severity
        if threat_score > 0.8:
            severity = "high"
        elif threat_score > 0.6:
            severity = "medium"
        else:
            severity = "low"

        # Create description
        active_patterns = [
            name for name, score in anomaly_scores.items() if score > 0.3
        ]
        description = f"Behavioral anomaly: {', '.join(active_patterns)} (score: {threat_score:.2f})"

        return BehavioralPattern(
            pattern_type=pattern_type,
            src_ip=src_ip,
            severity=severity,
            confidence=confidence,
            indicators=anomaly_scores,
            features=features,
            description=description,
            threat_score=threat_score,
        )


# Global behavioral analyzer instance
behavioral_analyzer = BehaviorAnalyzer()

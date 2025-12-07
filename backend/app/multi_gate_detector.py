"""
Multi-Gate Detection System

Defense-in-depth approach to reduce false positives by passing detections
through multiple verification gates before creating incidents.

Gates:
1. Heuristic Pre-filter: Fast rule-based checks (microseconds)
2. ML Classification: Enhanced threat detector with temperature scaling
3. Specialist Verification: Binary classifiers for high-FP classes (1, 3, 4)
4. Vector Memory Check: Similarity to past false positives
5. Council Verification: Gemini/OpenAI for medium-confidence cases

Each gate can:
- PASS: Continue to next gate
- FAIL: Stop detection, no incident created
- ESCALATE: Increase confidence and continue
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .models import Event

logger = logging.getLogger(__name__)


class GateVerdict(Enum):
    """Possible verdicts from each gate."""

    PASS = "pass"
    FAIL = "fail"
    ESCALATE = "escalate"
    SKIP = "skip"  # Gate not applicable


@dataclass
class GateResult:
    """Result from a single gate."""

    gate_name: str
    verdict: GateVerdict
    confidence_modifier: float  # Multiplier for confidence (0.0-2.0)
    reason: str
    details: Dict[str, Any]
    processing_time_ms: float


@dataclass
class MultiGateResult:
    """Combined result from all gates."""

    passed_all_gates: bool
    final_confidence: float
    gate_results: List[GateResult]
    total_processing_time_ms: float
    threat_type: str
    threat_class: int
    should_create_incident: bool
    blocking_gate: Optional[str]  # Which gate blocked if any
    escalation_reasons: List[str]


class MultiGateDetector:
    """
    Multi-gate detection system that passes events through multiple
    verification stages to reduce false positives.
    """

    def __init__(self):
        self.logger = logger

        # Gate configuration
        self.gates_enabled = {
            "heuristic": True,
            "ml_classification": True,
            "specialist_verification": True,
            "vector_memory": True,
            "council": True,
        }

        # Thresholds for gate decisions
        self.heuristic_min_events = 1
        self.specialist_confirmation_threshold = 0.85
        self.vector_similarity_threshold = 0.90
        self.council_confidence_range = (0.50, 0.85)  # Only route if in this range

        # Classes that need specialist verification
        self.high_fp_classes = {1, 3, 4}  # DDoS, BruteForce, WebAttack

    async def detect(
        self,
        src_ip: str,
        events: List[Event],
        features: np.ndarray = None,
    ) -> MultiGateResult:
        """
        Run events through all detection gates.

        Args:
            src_ip: Source IP address
            events: List of events to analyze
            features: Pre-extracted feature vector (optional)

        Returns:
            MultiGateResult with combined verdict
        """
        start_time = datetime.now(timezone.utc)
        gate_results = []
        current_confidence = 0.0
        threat_type = "Unknown"
        threat_class = -1
        escalation_reasons = []

        # Extract features if not provided
        if features is None:
            try:
                from .ml_feature_extractor import ml_feature_extractor

                features = ml_feature_extractor.extract_features(src_ip, events)
            except Exception as e:
                self.logger.error(f"Feature extraction failed: {e}")
                features = np.zeros(79)

        # ===== GATE 1: Heuristic Pre-filter =====
        if self.gates_enabled["heuristic"]:
            gate_result = await self._gate_heuristic(src_ip, events)
            gate_results.append(gate_result)

            if gate_result.verdict == GateVerdict.FAIL:
                return self._build_result(
                    passed=False,
                    confidence=0.0,
                    gate_results=gate_results,
                    start_time=start_time,
                    threat_type="Normal",
                    threat_class=0,
                    blocking_gate="heuristic",
                    escalation_reasons=[],
                )
            elif gate_result.verdict == GateVerdict.ESCALATE:
                escalation_reasons.append(gate_result.reason)

        # ===== GATE 2: ML Classification =====
        if self.gates_enabled["ml_classification"]:
            gate_result, ml_prediction = await self._gate_ml_classification(
                src_ip, events, features
            )
            gate_results.append(gate_result)

            if ml_prediction:
                current_confidence = ml_prediction.confidence
                threat_type = ml_prediction.threat_type
                threat_class = ml_prediction.predicted_class

            if gate_result.verdict == GateVerdict.FAIL:
                return self._build_result(
                    passed=False,
                    confidence=current_confidence,
                    gate_results=gate_results,
                    start_time=start_time,
                    threat_type=threat_type,
                    threat_class=threat_class,
                    blocking_gate="ml_classification",
                    escalation_reasons=escalation_reasons,
                )
            elif gate_result.verdict == GateVerdict.ESCALATE:
                escalation_reasons.append(gate_result.reason)
                current_confidence *= gate_result.confidence_modifier

        # ===== GATE 3: Specialist Verification =====
        if (
            self.gates_enabled["specialist_verification"]
            and threat_class in self.high_fp_classes
        ):
            gate_result = await self._gate_specialist_verification(
                threat_class, features
            )
            gate_results.append(gate_result)

            if gate_result.verdict == GateVerdict.FAIL:
                # Specialist rejected - significantly reduce confidence
                current_confidence *= 0.3

                return self._build_result(
                    passed=False,
                    confidence=current_confidence,
                    gate_results=gate_results,
                    start_time=start_time,
                    threat_type=threat_type,
                    threat_class=threat_class,
                    blocking_gate="specialist_verification",
                    escalation_reasons=escalation_reasons,
                )
            elif gate_result.verdict == GateVerdict.ESCALATE:
                escalation_reasons.append(gate_result.reason)
                current_confidence *= gate_result.confidence_modifier

        # ===== GATE 4: Vector Memory Check =====
        if self.gates_enabled["vector_memory"]:
            gate_result = await self._gate_vector_memory(features, threat_type)
            gate_results.append(gate_result)

            if gate_result.verdict == GateVerdict.FAIL:
                # Similar to past FP - block
                return self._build_result(
                    passed=False,
                    confidence=current_confidence * 0.3,
                    gate_results=gate_results,
                    start_time=start_time,
                    threat_type=threat_type,
                    threat_class=threat_class,
                    blocking_gate="vector_memory",
                    escalation_reasons=escalation_reasons,
                )
            elif gate_result.verdict == GateVerdict.ESCALATE:
                escalation_reasons.append(gate_result.reason)

        # ===== GATE 5: Council Verification =====
        min_conf, max_conf = self.council_confidence_range
        if self.gates_enabled["council"] and min_conf <= current_confidence <= max_conf:
            gate_result = await self._gate_council_verification(
                src_ip, events, features, threat_type, current_confidence
            )
            gate_results.append(gate_result)

            if gate_result.verdict == GateVerdict.FAIL:
                return self._build_result(
                    passed=False,
                    confidence=current_confidence * 0.4,
                    gate_results=gate_results,
                    start_time=start_time,
                    threat_type=threat_type,
                    threat_class=threat_class,
                    blocking_gate="council",
                    escalation_reasons=escalation_reasons,
                )
            elif gate_result.verdict == GateVerdict.ESCALATE:
                escalation_reasons.append(gate_result.reason)
                current_confidence *= gate_result.confidence_modifier

        # All gates passed
        return self._build_result(
            passed=True,
            confidence=current_confidence,
            gate_results=gate_results,
            start_time=start_time,
            threat_type=threat_type,
            threat_class=threat_class,
            blocking_gate=None,
            escalation_reasons=escalation_reasons,
        )

    async def _gate_heuristic(self, src_ip: str, events: List[Event]) -> GateResult:
        """
        Gate 1: Fast heuristic pre-filter.
        Blocks obviously benign traffic, escalates suspicious patterns.
        """
        start = datetime.now(timezone.utc)

        # Check minimum events
        if len(events) < self.heuristic_min_events:
            return GateResult(
                gate_name="heuristic",
                verdict=GateVerdict.FAIL,
                confidence_modifier=0.0,
                reason=f"Insufficient events ({len(events)} < {self.heuristic_min_events})",
                details={"event_count": len(events)},
                processing_time_ms=(datetime.now(timezone.utc) - start).total_seconds()
                * 1000,
            )

        # Count attack indicators
        failed_logins = sum(
            1 for e in events if "login.failed" in (e.eventid or "").lower()
        )
        file_downloads = sum(
            1 for e in events if "file_download" in (e.eventid or "").lower()
        )
        commands = sum(1 for e in events if "command" in (e.eventid or "").lower())

        # Check for clear attack indicators
        if failed_logins >= 5 or file_downloads > 0 or commands >= 3:
            return GateResult(
                gate_name="heuristic",
                verdict=GateVerdict.ESCALATE,
                confidence_modifier=1.1,  # Slight boost
                reason=f"Attack indicators: {failed_logins} failed logins, {file_downloads} downloads, {commands} commands",
                details={
                    "failed_logins": failed_logins,
                    "file_downloads": file_downloads,
                    "commands": commands,
                },
                processing_time_ms=(datetime.now(timezone.utc) - start).total_seconds()
                * 1000,
            )

        # Pass without modification
        return GateResult(
            gate_name="heuristic",
            verdict=GateVerdict.PASS,
            confidence_modifier=1.0,
            reason="Heuristic checks passed",
            details={"event_count": len(events)},
            processing_time_ms=(datetime.now(timezone.utc) - start).total_seconds()
            * 1000,
        )

    async def _gate_ml_classification(
        self, src_ip: str, events: List[Event], features: np.ndarray
    ) -> Tuple[GateResult, Any]:
        """
        Gate 2: ML classification with temperature scaling.
        """
        start = datetime.now(timezone.utc)

        try:
            from .enhanced_threat_detector import enhanced_detector

            if not enhanced_detector.model:
                return (
                    GateResult(
                        gate_name="ml_classification",
                        verdict=GateVerdict.SKIP,
                        confidence_modifier=1.0,
                        reason="ML model not loaded",
                        details={},
                        processing_time_ms=(
                            datetime.now(timezone.utc) - start
                        ).total_seconds()
                        * 1000,
                    ),
                    None,
                )

            # Get prediction with temperature scaling and specialist verification
            prediction = await enhanced_detector.analyze_threat(
                src_ip, events, feature_vector=features.reshape(1, -1)
            )

            # Check if normal traffic
            if prediction.predicted_class == 0 and prediction.confidence > 0.95:
                return (
                    GateResult(
                        gate_name="ml_classification",
                        verdict=GateVerdict.FAIL,
                        confidence_modifier=0.0,
                        reason="ML classified as Normal with high confidence",
                        details={
                            "predicted_class": prediction.predicted_class,
                            "confidence": prediction.confidence,
                        },
                        processing_time_ms=(
                            datetime.now(timezone.utc) - start
                        ).total_seconds()
                        * 1000,
                    ),
                    prediction,
                )

            # Pass with classification result
            return (
                GateResult(
                    gate_name="ml_classification",
                    verdict=GateVerdict.PASS,
                    confidence_modifier=1.0,
                    reason=f"ML classification: {prediction.threat_type}",
                    details={
                        "predicted_class": prediction.predicted_class,
                        "threat_type": prediction.threat_type,
                        "confidence": prediction.confidence,
                        "uncertainty": prediction.uncertainty_score,
                    },
                    processing_time_ms=(
                        datetime.now(timezone.utc) - start
                    ).total_seconds()
                    * 1000,
                ),
                prediction,
            )

        except Exception as e:
            self.logger.error(f"ML classification gate failed: {e}")
            return (
                GateResult(
                    gate_name="ml_classification",
                    verdict=GateVerdict.SKIP,
                    confidence_modifier=1.0,
                    reason=f"ML classification error: {str(e)}",
                    details={"error": str(e)},
                    processing_time_ms=(
                        datetime.now(timezone.utc) - start
                    ).total_seconds()
                    * 1000,
                ),
                None,
            )

    async def _gate_specialist_verification(
        self, threat_class: int, features: np.ndarray
    ) -> GateResult:
        """
        Gate 3: Specialist binary classifier verification for high-FP classes.
        """
        start = datetime.now(timezone.utc)

        try:
            from .enhanced_threat_detector import enhanced_detector

            if not enhanced_detector.specialist_manager._loaded:
                return GateResult(
                    gate_name="specialist_verification",
                    verdict=GateVerdict.SKIP,
                    confidence_modifier=1.0,
                    reason="Specialist models not loaded",
                    details={},
                    processing_time_ms=(
                        datetime.now(timezone.utc) - start
                    ).total_seconds()
                    * 1000,
                )

            (
                confirmed,
                spec_conf,
                reason,
            ) = enhanced_detector.specialist_manager.verify_prediction(
                predicted_class=threat_class,
                features=features,
            )

            if confirmed:
                # Specialist confirmed - boost confidence
                return GateResult(
                    gate_name="specialist_verification",
                    verdict=GateVerdict.ESCALATE,
                    confidence_modifier=1.15,  # 15% boost for specialist confirmation
                    reason=f"Specialist confirmed (confidence: {spec_conf:.3f})",
                    details={
                        "specialist_confidence": spec_conf,
                        "verification_reason": reason,
                    },
                    processing_time_ms=(
                        datetime.now(timezone.utc) - start
                    ).total_seconds()
                    * 1000,
                )
            else:
                # Specialist rejected
                return GateResult(
                    gate_name="specialist_verification",
                    verdict=GateVerdict.FAIL,
                    confidence_modifier=0.3,
                    reason=f"Specialist rejected (confidence: {spec_conf:.3f})",
                    details={
                        "specialist_confidence": spec_conf,
                        "verification_reason": reason,
                    },
                    processing_time_ms=(
                        datetime.now(timezone.utc) - start
                    ).total_seconds()
                    * 1000,
                )

        except Exception as e:
            self.logger.error(f"Specialist verification gate failed: {e}")
            return GateResult(
                gate_name="specialist_verification",
                verdict=GateVerdict.SKIP,
                confidence_modifier=1.0,
                reason=f"Specialist verification error: {str(e)}",
                details={"error": str(e)},
                processing_time_ms=(datetime.now(timezone.utc) - start).total_seconds()
                * 1000,
            )

    async def _gate_vector_memory(
        self, features: np.ndarray, ml_prediction: str
    ) -> GateResult:
        """
        Gate 4: Check similarity to past false positives in vector memory.
        """
        start = datetime.now(timezone.utc)

        try:
            from .learning.vector_memory import check_similar_false_positives

            is_similar, fp_details = await check_similar_false_positives(
                features=features,
                ml_prediction=ml_prediction,
                threshold=self.vector_similarity_threshold,
            )

            if is_similar and fp_details:
                return GateResult(
                    gate_name="vector_memory",
                    verdict=GateVerdict.FAIL,
                    confidence_modifier=0.4,
                    reason=f"Similar to past FP (similarity: {fp_details['similarity_score']:.3f})",
                    details=fp_details,
                    processing_time_ms=(
                        datetime.now(timezone.utc) - start
                    ).total_seconds()
                    * 1000,
                )

            return GateResult(
                gate_name="vector_memory",
                verdict=GateVerdict.PASS,
                confidence_modifier=1.0,
                reason="No similar false positives found",
                details={},
                processing_time_ms=(datetime.now(timezone.utc) - start).total_seconds()
                * 1000,
            )

        except Exception as e:
            self.logger.debug(f"Vector memory gate skipped: {e}")
            return GateResult(
                gate_name="vector_memory",
                verdict=GateVerdict.SKIP,
                confidence_modifier=1.0,
                reason=f"Vector memory unavailable: {str(e)}",
                details={},
                processing_time_ms=(datetime.now(timezone.utc) - start).total_seconds()
                * 1000,
            )

    async def _gate_council_verification(
        self,
        src_ip: str,
        events: List[Event],
        features: np.ndarray,
        threat_type: str,
        current_confidence: float,
    ) -> GateResult:
        """
        Gate 5: Council of Models verification for medium-confidence cases.
        """
        start = datetime.now(timezone.utc)

        try:
            from .orchestrator.graph import create_initial_state
            from .orchestrator.workflow import orchestrate_incident

            # Create state for Council
            state = create_initial_state(
                src_ip=src_ip,
                events=[
                    {
                        "timestamp": str(e.ts) if e.ts else None,
                        "event_type": e.eventid or "unknown",
                        "dst_port": e.dst_port,
                    }
                    for e in events
                ],
                ml_prediction={
                    "class": threat_type,
                    "confidence": current_confidence,
                    "model": "enhanced_local",
                },
                raw_features=features.tolist(),
            )

            # Run through Council
            final_state = await orchestrate_incident(state)

            verdict = final_state.get("final_verdict", "INVESTIGATE")
            council_confidence = final_state.get("confidence_score", current_confidence)

            if verdict == "FALSE_POSITIVE":
                return GateResult(
                    gate_name="council",
                    verdict=GateVerdict.FAIL,
                    confidence_modifier=0.3,
                    reason=f"Council verdict: FALSE_POSITIVE",
                    details={
                        "council_verdict": verdict,
                        "council_confidence": council_confidence,
                        "reasoning": final_state.get("gemini_reasoning", "")[:200],
                    },
                    processing_time_ms=(
                        datetime.now(timezone.utc) - start
                    ).total_seconds()
                    * 1000,
                )
            elif verdict == "THREAT":
                return GateResult(
                    gate_name="council",
                    verdict=GateVerdict.ESCALATE,
                    confidence_modifier=1.2,
                    reason=f"Council confirmed threat",
                    details={
                        "council_verdict": verdict,
                        "council_confidence": council_confidence,
                    },
                    processing_time_ms=(
                        datetime.now(timezone.utc) - start
                    ).total_seconds()
                    * 1000,
                )
            else:
                # INVESTIGATE - pass without modification
                return GateResult(
                    gate_name="council",
                    verdict=GateVerdict.PASS,
                    confidence_modifier=1.0,
                    reason=f"Council verdict: INVESTIGATE",
                    details={
                        "council_verdict": verdict,
                        "council_confidence": council_confidence,
                    },
                    processing_time_ms=(
                        datetime.now(timezone.utc) - start
                    ).total_seconds()
                    * 1000,
                )

        except Exception as e:
            self.logger.warning(f"Council gate skipped: {e}")
            return GateResult(
                gate_name="council",
                verdict=GateVerdict.SKIP,
                confidence_modifier=1.0,
                reason=f"Council unavailable: {str(e)}",
                details={},
                processing_time_ms=(datetime.now(timezone.utc) - start).total_seconds()
                * 1000,
            )

    def _build_result(
        self,
        passed: bool,
        confidence: float,
        gate_results: List[GateResult],
        start_time: datetime,
        threat_type: str,
        threat_class: int,
        blocking_gate: Optional[str],
        escalation_reasons: List[str],
    ) -> MultiGateResult:
        """Build the final multi-gate result."""
        total_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

        # Determine if incident should be created
        # Must pass all gates and have sufficient confidence
        from .intelligent_detection import DetectionConfig

        config = DetectionConfig()
        threshold = config.confidence_thresholds.get(threat_class, 0.5)

        should_create = passed and confidence >= threshold and threat_class != 0

        return MultiGateResult(
            passed_all_gates=passed,
            final_confidence=confidence,
            gate_results=gate_results,
            total_processing_time_ms=total_time,
            threat_type=threat_type,
            threat_class=threat_class,
            should_create_incident=should_create,
            blocking_gate=blocking_gate,
            escalation_reasons=escalation_reasons,
        )


# Global instance
multi_gate_detector = MultiGateDetector()

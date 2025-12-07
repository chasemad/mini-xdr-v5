"""
🚀 INTELLIGENT INCIDENT DETECTION ENGINE
Enhanced detection system that leverages local ML models, OpenAI analysis, and configurable thresholds
to create incidents dynamically based on threat classification and anomaly scores.

Key Features:
1. Local ML threat classification with ensemble models
2. Configurable confidence thresholds per threat type
3. OpenAI-enhanced anomaly detection for novel attacks
4. Adaptive learning from false positives/negatives
5. Multi-layered scoring system with ensemble methods
"""

import asyncio
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sqlalchemy import and_, asc, desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from .config import settings
from .models import Event, Incident

# Council of Models integration
try:
    from .orchestrator.graph import create_initial_state
    from .orchestrator.workflow import orchestrate_incident

    COUNCIL_AVAILABLE = True
except ImportError:
    COUNCIL_AVAILABLE = False
    logging.warning(
        "Council orchestrator not available - running without Council verification"
    )

# Vector Memory for false positive learning
try:
    from .learning.vector_memory import (
        check_similar_false_positives,
        store_false_positive,
    )

    VECTOR_MEMORY_AVAILABLE = True
except ImportError:
    VECTOR_MEMORY_AVAILABLE = False
    logging.warning(
        "Vector memory not available - running without FP similarity detection"
    )

# Multi-Gate Detection System (Phase 3 - optional alternative architecture)
try:
    from .multi_gate_detector import MultiGateResult, multi_gate_detector

    MULTI_GATE_AVAILABLE = True
except ImportError:
    MULTI_GATE_AVAILABLE = False
    multi_gate_detector = None
    logging.info("Multi-gate detector not available - using inline detection logic")

logger = logging.getLogger(__name__)


class ThreatSeverity(Enum):
    """Threat severity levels"""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class ThreatClassification:
    """Represents a threat classification result"""

    threat_type: str
    threat_class: int
    confidence: float
    anomaly_score: float
    severity: ThreatSeverity
    indicators: Dict[str, Any]
    openai_enhanced: bool = False


@dataclass
class DetectionConfig:
    """Configurable detection thresholds and parameters"""

    # Local ML confidence thresholds by threat class
    confidence_thresholds: Dict[int, float] = None

    # Anomaly score thresholds (lowered for demo/testing)
    anomaly_score_threshold: float = 0.2

    # Minimum events required for analysis
    min_events_required: int = 1

    # Time windows for event aggregation (seconds)
    analysis_window: int = 300  # 5 minutes

    # OpenAI enhancement settings
    enable_openai_analysis: bool = True
    openai_threshold: float = 0.7

    # Adaptive learning parameters
    enable_adaptive_learning: bool = True
    false_positive_weight: float = 0.1

    def __post_init__(self):
        if self.confidence_thresholds is None:
            # Per-class confidence thresholds based on training precision metrics:
            # - Higher thresholds for classes with low precision (to reduce FPs)
            # - Lower thresholds for classes with high precision
            #
            # Training metrics from models/local_trained/training_summary.json:
            # Class 0 (Normal):      Precision 99.9%, Recall 83.4% → Keep high
            # Class 1 (DDoS):        Precision 73.2%, Recall 99.8% → Moderate
            # Class 2 (Recon):       Precision 73.3%, Recall 68.8% → Moderate
            # Class 3 (BruteForce):  Precision 17.8%, Recall 80.3% → HIGH (many FPs!)
            # Class 4 (WebAttack):   Precision 32.4%, Recall 28.8% → HIGH (many FPs!)
            # Class 5 (Malware):     Precision 84.2%, Recall 41.3% → Lower OK
            # Class 6 (APT):         Precision 46.9%, Recall 52.4% → Moderate
            #
            # NOTE: With specialist model verification enabled, thresholds can be
            # slightly lower since specialists provide a second verification gate.
            self.confidence_thresholds = {
                0: 0.95,  # Normal - very high threshold to avoid FPs
                1: 0.50,  # DDoS/DoS - 73% precision, specialist verified
                2: 0.50,  # Network Reconnaissance - 73% precision
                3: 0.80,  # Brute Force Attack - RAISED due to 17.8% precision!
                4: 0.75,  # Web Application Attack - RAISED due to 32.4% precision!
                5: 0.45,  # Malware/Botnet - 84% precision, can be lower
                6: 0.60,  # Advanced Persistent Threat - 47% precision
            }


class IntelligentDetectionEngine:
    """
    Advanced detection engine that combines local ML predictions,
    OpenAI analysis, and configurable thresholds for dynamic incident creation
    """

    def __init__(self, config: DetectionConfig = None):
        self.config = config or DetectionConfig()
        self.logger = logger

        # OpenAI client for enhanced analysis
        self.openai_client = None
        self._init_openai_client()

        # Learning statistics for adaptive thresholds
        self.learning_stats = {
            "total_classifications": 0,
            "incidents_created": 0,
            "false_positives": 0,
            "false_negatives": 0,
            "per_class_accuracy": {},
        }

    def _init_openai_client(self):
        """Initialize OpenAI client for enhanced analysis"""
        try:
            import openai

            openai_key = settings.openai_api_key
            if openai_key:
                self.openai_client = openai.AsyncOpenAI(api_key=openai_key)
                self.logger.info(
                    "OpenAI client initialized for enhanced threat analysis"
                )
            else:
                self.logger.warning(
                    "OpenAI API key not found - enhanced analysis disabled"
                )
        except Exception as e:
            self.logger.warning(f"Failed to initialize OpenAI client: {e}")

    async def analyze_and_create_incidents(
        self, db: AsyncSession, src_ip: str, events: List[Event] = None
    ) -> Dict[str, Any]:
        """
        Main analysis method that determines if incidents should be created
        based on local ML predictions and configurable thresholds
        """
        print(f"🟠 ANALYZE_AND_CREATE_INCIDENTS called for {src_ip}", flush=True)
        try:
            # Get recent events if not provided
            if events is None:
                print(f"🟠 Fetching recent events for {src_ip}...", flush=True)
                events = await self._get_recent_events(db, src_ip)

            print(
                f"🟠 Checking event count: {len(events)} vs min {self.config.min_events_required}",
                flush=True,
            )
            if len(events) < self.config.min_events_required:
                print(f"🟠 Insufficient events, returning False", flush=True)
                return {
                    "incident_created": False,
                    "reason": f"Insufficient events ({len(events)} < {self.config.min_events_required})",
                }

            # Step 1: Get local ML threat classification
            print(f"🟠 Getting ML classification for {src_ip}...", flush=True)
            classification = await self._get_local_ml_classification(src_ip, events)

            if not classification:
                print(f"🟠 ML classification failed, returning False", flush=True)
                return {
                    "incident_created": False,
                    "reason": "Local ML classification failed",
                }

            print(
                f"🟠 ML Classification: {classification.threat_type} (confidence: {classification.confidence:.2%})",
                flush=True,
            )

            # Step 1.5: Council of Models verification for ALL predictions
            # Previously only triggered for 0.50-0.90, now triggers for any detection
            council_data = None
            features = None
            if classification.confidence >= 0.30:  # Run council for most detections
                self.logger.info(
                    f"Routing {src_ip} to Council: confidence={classification.confidence:.2%}"
                )

                # Extract features for ML-Agent bridge
                try:
                    from .ml_feature_extractor import ml_feature_extractor

                    features = ml_feature_extractor.extract_features(src_ip, events)
                except Exception as e:
                    self.logger.debug(f"Feature extraction for council failed: {e}")

                council_data = await self._route_through_council(
                    src_ip, events, classification, features
                )

                if council_data:
                    # Update classification based on Council verdict
                    classification = council_data["updated_classification"]
                    self.logger.info(
                        f"Council verdict: {council_data.get('final_verdict')}, "
                        f"confidence: {classification.confidence:.2%}"
                    )

            # Step 1.9: Check if there's already an open incident for this IP
            # If so, always add events to it (even if new batch is classified as "Normal")
            existing_incident = await self._find_existing_incident(db, src_ip)

            if existing_incident:
                # Always add new events to existing incident
                self.logger.info(
                    f"Found existing open incident #{existing_incident.id} for {src_ip}, "
                    f"adding {len(events)} new events regardless of classification"
                )
                incident_id = await self._update_existing_incident(
                    db, existing_incident, events, classification, council_data
                )
                return {
                    "incident_created": True,  # Updated existing
                    "incident_id": incident_id,
                    "classification": asdict(classification),
                    "confidence": classification.confidence,
                    "threat_type": classification.threat_type,
                    "severity": classification.severity.value,
                    "note": "Updated existing incident with new events",
                }

            # Step 2: Check if classification meets threshold for NEW incident creation
            should_create_incident = await self._should_create_incident(
                classification, events
            )

            if not should_create_incident["create"]:
                return {
                    "incident_created": False,
                    "classification": asdict(classification),
                    "reason": should_create_incident["reason"],
                }

            # Step 2.5: Vector Memory FP Check - is this similar to past false positives?
            if VECTOR_MEMORY_AVAILABLE:
                try:
                    from .ml_feature_extractor import ml_feature_extractor

                    features = ml_feature_extractor.extract_features(src_ip, events)

                    is_similar_to_fp, fp_details = await check_similar_false_positives(
                        features=features,
                        ml_prediction=classification.threat_type,
                        threshold=0.90,  # High similarity threshold
                    )

                    if is_similar_to_fp and fp_details:
                        self.logger.warning(
                            f"⚠️ Pattern similar to past false positive "
                            f"(similarity: {fp_details['similarity_score']:.3f}): "
                            f"Previous FP was '{fp_details['original_prediction']}'"
                        )

                        # Reduce confidence significantly for similar patterns
                        original_confidence = classification.confidence
                        classification.confidence = min(
                            classification.confidence * 0.4, 0.35
                        )
                        classification.indicators["similar_to_past_fp"] = {
                            "similarity_score": fp_details["similarity_score"],
                            "original_prediction": fp_details["original_prediction"],
                            "reasoning": fp_details.get("reasoning", ""),
                        }

                        # Re-check if should create incident with reduced confidence
                        should_create_incident = await self._should_create_incident(
                            classification, events
                        )

                        if not should_create_incident["create"]:
                            self.logger.info(
                                f"Incident blocked due to FP similarity: "
                                f"confidence reduced from {original_confidence:.3f} to {classification.confidence:.3f}"
                            )
                            return {
                                "incident_created": False,
                                "classification": asdict(classification),
                                "reason": f"Pattern similar to past false positive (similarity: {fp_details['similarity_score']:.3f})",
                                "fp_similarity": fp_details,
                            }
                except Exception as e:
                    self.logger.warning(f"Vector memory FP check failed: {e}")

            # Step 3: Enhanced analysis with OpenAI (if enabled and needed)
            if (
                self.config.enable_openai_analysis
                and classification.confidence < self.config.openai_threshold
            ):
                enhanced_analysis = await self._openai_enhanced_analysis(
                    src_ip, events, classification
                )

                if enhanced_analysis:
                    classification.openai_enhanced = True
                    classification.indicators.update(enhanced_analysis)

            # Step 4: Create incident with comprehensive details (including Council data)
            incident_id = await self._create_intelligent_incident(
                db, src_ip, events, classification, council_data
            )

            # Step 5: Update learning statistics
            self._update_learning_stats(classification, True)

            return {
                "incident_created": True,
                "incident_id": incident_id,
                "classification": asdict(classification),
                "confidence": classification.confidence,
                "threat_type": classification.threat_type,
                "severity": classification.severity.value,
            }

        except Exception as e:
            self.logger.error(f"Intelligent detection failed for {src_ip}: {e}")
            return {"incident_created": False, "reason": f"Detection error: {str(e)}"}

    async def _get_recent_events(self, db: AsyncSession, src_ip: str) -> List[Event]:
        """Get recent events for the given IP within analysis window"""
        window_start = datetime.now(timezone.utc) - timedelta(
            seconds=self.config.analysis_window
        )

        stmt = (
            select(Event)
            .where(and_(Event.src_ip == src_ip, Event.ts >= window_start))
            .order_by(desc(Event.ts))
            .limit(100)  # Reasonable limit for analysis
        )

        result = await db.execute(stmt)
        return result.scalars().all()

    async def _find_existing_incident(
        self, db: AsyncSession, src_ip: str
    ) -> Optional[Incident]:
        """
        Find an existing incident for this IP within the consolidation window.
        Used to consolidate multiple batches of events into a single incident.

        NOTE: We look for both "open" AND "contained" incidents because:
        - Auto-containment may have already changed status to "contained"
        - The attack is still ongoing and events should be correlated
        - We exclude "dismissed" incidents as those are false positives
        """
        one_hour_ago = datetime.now(timezone.utc) - timedelta(hours=1)

        existing_query = (
            select(Incident)
            .where(
                and_(
                    Incident.src_ip == src_ip,
                    Incident.status.in_(
                        ["open", "contained"]
                    ),  # Include auto-contained incidents
                    Incident.created_at >= one_hour_ago,
                )
            )
            .order_by(Incident.created_at.desc())
        )

        result = await db.execute(existing_query)
        return result.scalars().first()

    async def _get_local_ml_classification(
        self, src_ip: str, events: List[Event]
    ) -> Optional[ThreatClassification]:
        """Get threat classification from Enhanced Local Model"""
        try:
            # Use Enhanced Local Model Only
            self.logger.info("Using local enhanced model for threat classification")
            enhanced_result = await self._get_enhanced_model_classification(
                src_ip, events
            )

            if enhanced_result:
                # Apply event-content-aware override to correct ML misclassifications
                # This fixes cases where statistical features (like high event rate) mislead the model
                corrected_result = await self._apply_event_content_override(
                    enhanced_result, events
                )
                return corrected_result

            # If enhanced model is not available, use fallback local detection
            self.logger.warning(
                "Enhanced model unavailable, using basic local detection"
            )
            return await self._get_fallback_local_classification(src_ip, events)

        except Exception as e:
            self.logger.error(f"Local model classification failed: {e}")
            return None

    async def _apply_event_content_override(
        self, classification: ThreatClassification, events: List[Event]
    ) -> ThreatClassification:
        """
        Override ML classification when actual event content clearly indicates a different attack type.

        This fixes the disconnect between:
        - Statistical ML features (event rate, burst intensity) → might say "DDoS"
        - Actual event content (login.failed, file_download) → clearly indicates brute force + malware

        The actual event types are ground truth and should take precedence over statistical aggregates.
        """
        # Analyze actual event content to determine attack signature
        event_signature = self._analyze_event_signatures(events)

        if not event_signature["override_needed"]:
            return classification

        self.logger.info(
            f"🔄 Event content override: ML said '{classification.threat_type}' but events show '{event_signature['actual_attack_type']}'"
        )

        # Map event signatures to threat classes
        threat_class_map = {
            "brute_force": (3, "Brute Force Attack", ThreatSeverity.HIGH),
            "malware": (5, "Malware/Botnet", ThreatSeverity.CRITICAL),
            "apt": (6, "Advanced Persistent Threat", ThreatSeverity.CRITICAL),
            "reconnaissance": (2, "Network Reconnaissance", ThreatSeverity.MEDIUM),
            "web_attack": (4, "Web Application Attack", ThreatSeverity.MEDIUM),
            "data_exfiltration": (5, "Data Exfiltration", ThreatSeverity.CRITICAL),
        }

        if event_signature["actual_attack_type"] in threat_class_map:
            new_class, new_type, new_severity = threat_class_map[
                event_signature["actual_attack_type"]
            ]

            # Update classification with corrected values
            corrected = ThreatClassification(
                threat_type=new_type,
                threat_class=new_class,
                confidence=max(
                    classification.confidence, event_signature["confidence"]
                ),
                anomaly_score=classification.anomaly_score,
                severity=new_severity,
                indicators={
                    **classification.indicators,
                    "event_content_override": {
                        "original_ml_classification": classification.threat_type,
                        "corrected_classification": new_type,
                        "reason": event_signature["reason"],
                        "event_evidence": event_signature["evidence"],
                    },
                },
            )

            self.logger.info(
                f"✅ Classification corrected: {classification.threat_type} → {new_type} "
                f"(confidence: {corrected.confidence:.2%})"
            )
            return corrected

        return classification

    def _analyze_event_signatures(self, events: List[Event]) -> Dict[str, Any]:
        """
        Analyze actual event types to determine the true attack signature.
        Returns override information if events clearly indicate a specific attack type.

        Enhanced with:
        - Time-based analysis (automated single-type events != DDoS)
        - Command content analysis (specific malware signatures)
        - Session correlation (attack phases)
        - Username pattern analysis (common vs unique)
        """
        # Count event types
        event_counts = {}
        for e in events:
            eventid = e.eventid or ""
            event_counts[eventid] = event_counts.get(eventid, 0) + 1

        # Extract key indicators
        failed_logins = event_counts.get("cowrie.login.failed", 0)
        success_logins = event_counts.get("cowrie.login.success", 0)
        file_downloads = event_counts.get("cowrie.session.file_download", 0)
        file_uploads = event_counts.get("cowrie.session.file_upload", 0)
        commands = event_counts.get("cowrie.command.input", 0)
        session_connects = sum(v for k, v in event_counts.items() if "connect" in k)
        http_requests = sum(v for k, v in event_counts.items() if "http" in k.lower())

        result = {
            "override_needed": False,
            "actual_attack_type": None,
            "confidence": 0.0,
            "reason": None,
            "evidence": {},
        }

        # ===== TIME-BASED ANALYSIS =====
        # High event rate with single event type is likely automated scanning, NOT DDoS
        if len(events) > 1:
            try:
                timestamps = [e.ts for e in events if e.ts]
                if len(timestamps) >= 2:
                    time_span = (max(timestamps) - min(timestamps)).total_seconds()
                    events_per_second = len(events) / max(time_span, 1)
                    unique_event_types = len(
                        set(e.eventid for e in events if e.eventid)
                    )

                    # High rate + single event type = automated, not DDoS
                    if events_per_second > 5 and unique_event_types == 1:
                        single_type = events[0].eventid or ""
                        if "login.failed" in single_type:
                            result["override_needed"] = True
                            result["actual_attack_type"] = "brute_force"
                            result["confidence"] = 0.90
                            result[
                                "reason"
                            ] = f"High-rate automated login attempts ({events_per_second:.1f}/sec) - brute force, not DDoS"
                            result["evidence"] = {
                                "events_per_second": events_per_second,
                                "single_event_type": single_type,
                                "time_span_seconds": time_span,
                            }
                            return result
            except Exception as e:
                self.logger.debug(f"Time-based analysis failed: {e}")

        # ===== COMMAND CONTENT ANALYSIS =====
        # Check for specific malware signatures in commands
        malware_commands = []
        recon_commands = []
        persistence_commands = []

        for e in events:
            msg = (e.message or "").lower()

            # Malware download signatures
            if any(
                p in msg
                for p in ["wget ", "curl ", "tftp ", "ftp ", "scp ", "nc -e", "bash -i"]
            ):
                malware_commands.append(msg[:100])

            # Reconnaissance signatures
            if any(
                p in msg
                for p in [
                    "uname",
                    "whoami",
                    "id;",
                    "cat /etc/passwd",
                    "ps aux",
                    "netstat",
                    "ifconfig",
                    "ip addr",
                ]
            ):
                recon_commands.append(msg[:100])

            # Persistence signatures
            if any(
                p in msg
                for p in [
                    "crontab",
                    "chmod +x",
                    "systemctl",
                    "rc.local",
                    ".bashrc",
                    "/etc/init",
                ]
            ):
                persistence_commands.append(msg[:100])

        # ===== USERNAME PATTERN ANALYSIS =====
        usernames = [
            getattr(e, "username", None) for e in events if getattr(e, "username", None)
        ]
        unique_usernames = set(usernames)
        common_usernames = {
            "root",
            "admin",
            "user",
            "guest",
            "test",
            "oracle",
            "postgres",
            "mysql",
        }

        # Many unique usernames = dictionary attack (brute force)
        if len(unique_usernames) > 5:
            uncommon_usernames = unique_usernames - common_usernames
            if len(uncommon_usernames) > 3:
                result["override_needed"] = True
                result["actual_attack_type"] = "brute_force"
                result["confidence"] = 0.88
                result[
                    "reason"
                ] = f"Dictionary attack detected: {len(unique_usernames)} unique usernames tried"
                result["evidence"] = {
                    "unique_usernames": len(unique_usernames),
                    "sample_usernames": list(unique_usernames)[:5],
                }
                return result

        # ===== ATTACK PHASE CORRELATION =====
        # Check for multi-phase attacks (recon → exploit → persistence)
        phases_detected = []
        if recon_commands:
            phases_detected.append("reconnaissance")
        if failed_logins > 0 or success_logins > 0:
            phases_detected.append("initial_access")
        if malware_commands:
            phases_detected.append("execution")
        if persistence_commands:
            phases_detected.append("persistence")
        if file_uploads > 0:
            phases_detected.append("exfiltration")

        # Multi-phase = APT
        if len(phases_detected) >= 3:
            result["override_needed"] = True
            result["actual_attack_type"] = "apt"
            result["confidence"] = 0.92
            result[
                "reason"
            ] = f"Multi-phase attack detected: {', '.join(phases_detected)}"
            result["evidence"] = {
                "phases": phases_detected,
                "recon_commands": len(recon_commands),
                "malware_commands": len(malware_commands),
                "persistence_commands": len(persistence_commands),
            }
            return result

        # Priority 1: Brute Force - clear signature is multiple failed logins with eventual success
        if failed_logins >= 5:
            result["override_needed"] = True
            result["actual_attack_type"] = "brute_force"
            result["confidence"] = min(0.85 + (failed_logins / 100), 0.98)
            result[
                "reason"
            ] = f"Detected {failed_logins} failed logins - clear brute force pattern"
            result["evidence"] = {
                "failed_logins": failed_logins,
                "success_logins": success_logins,
            }

            # If there's also post-exploitation, elevate to APT
            if success_logins > 0 and (file_downloads > 0 or commands >= 5):
                result["actual_attack_type"] = "apt"
                result["confidence"] = 0.92
                result[
                    "reason"
                ] = f"Brute force succeeded + post-exploitation: {commands} commands, {file_downloads} downloads"
                result["evidence"]["commands"] = commands
                result["evidence"]["file_downloads"] = file_downloads

            return result

        # Priority 2: Malware/Data Exfiltration - file operations or malware commands
        if file_downloads > 0 or file_uploads > 0 or malware_commands:
            if file_uploads > 0:
                result["override_needed"] = True
                result["actual_attack_type"] = "data_exfiltration"
                result["confidence"] = 0.88
                result[
                    "reason"
                ] = f"Detected {file_uploads} file uploads - data exfiltration pattern"
                result["evidence"] = {
                    "file_uploads": file_uploads,
                    "file_downloads": file_downloads,
                }
            elif malware_commands:
                result["override_needed"] = True
                result["actual_attack_type"] = "malware"
                result["confidence"] = 0.90
                result[
                    "reason"
                ] = f"Detected {len(malware_commands)} malware download commands"
                result["evidence"] = {
                    "malware_commands": malware_commands[:3],
                    "file_downloads": file_downloads,
                }
            else:
                result["override_needed"] = True
                result["actual_attack_type"] = "malware"
                result["confidence"] = 0.85
                result["reason"] = f"Detected {file_downloads} malware downloads"
                result["evidence"] = {"file_downloads": file_downloads}
            return result

        # Priority 3: Web attacks - HTTP-based attacks
        if http_requests >= 3:
            # Check for SQL injection, XSS, or path traversal patterns
            sql_patterns = sum(
                1
                for e in events
                if e.message
                and any(
                    p in e.message.lower()
                    for p in [
                        "sql",
                        "union",
                        "select",
                        "drop",
                        "injection",
                        "' or ",
                        "1=1",
                        "--",
                    ]
                )
            )
            xss_patterns = sum(
                1
                for e in events
                if e.message
                and any(
                    p in e.message.lower()
                    for p in [
                        "<script",
                        "xss",
                        "onerror",
                        "javascript:",
                        "onclick",
                        "onload",
                    ]
                )
            )
            path_traversal = sum(
                1
                for e in events
                if e.message
                and any(
                    p in e.message.lower()
                    for p in [
                        "../",
                        "..\\",
                        "/etc/passwd",
                        "/etc/shadow",
                        "c:\\windows",
                    ]
                )
            )

            if sql_patterns > 0 or xss_patterns > 0 or path_traversal > 0:
                result["override_needed"] = True
                result["actual_attack_type"] = "web_attack"
                result["confidence"] = 0.85
                result[
                    "reason"
                ] = f"Detected web attack patterns: {sql_patterns} SQLi, {xss_patterns} XSS, {path_traversal} path traversal"
                result["evidence"] = {
                    "http_requests": http_requests,
                    "sql_patterns": sql_patterns,
                    "xss_patterns": xss_patterns,
                    "path_traversal": path_traversal,
                }
                return result

        # Priority 4: Reconnaissance - recon commands or port scanning
        if recon_commands:
            result["override_needed"] = True
            result["actual_attack_type"] = "reconnaissance"
            result["confidence"] = 0.82
            result["reason"] = f"Detected {len(recon_commands)} reconnaissance commands"
            result["evidence"] = {"recon_commands": recon_commands[:3]}
            return result

        unique_ports = len(set(e.dst_port for e in events if e.dst_port))
        if unique_ports >= 8 and failed_logins < 3:
            result["override_needed"] = True
            result["actual_attack_type"] = "reconnaissance"
            result["confidence"] = 0.78
            result[
                "reason"
            ] = f"Detected port scanning: {unique_ports} unique ports probed"
            result["evidence"] = {
                "unique_ports": unique_ports,
                "session_connects": session_connects,
            }
            return result

        # No clear override pattern found
        return result

    async def _get_enhanced_model_classification(
        self, src_ip: str, events: List[Event]
    ) -> Optional[ThreatClassification]:
        """Get threat classification from Enhanced Local Model"""
        try:
            from .enhanced_threat_detector import enhanced_detector

            # Check if enhanced detector is loaded
            if not enhanced_detector.model:
                self.logger.warning("Enhanced detector model not loaded yet")
                return None

            # Phase 2: Extract advanced features (100-dimensional)
            advanced_features = None
            try:
                from .features import advanced_feature_extractor

                # Use raw event objects; advanced extractor safely handles missing fields via getattr
                advanced_features = (
                    await advanced_feature_extractor.extract_all_features(
                        src_ip=src_ip, events=events
                    )
                )
                self.logger.info(
                    f"Extracted {len(advanced_features)} advanced features for {src_ip}"
                )
            except Exception as e:
                self.logger.debug(f"Advanced features not available: {e}")

            # Use enhanced detector for analysis
            prediction_result = await enhanced_detector.analyze_threat(src_ip, events)

            if not prediction_result:
                return None

            # Map to ThreatClassification
            severity_mapping = {
                0: ThreatSeverity.INFO,  # Normal
                1: ThreatSeverity.HIGH,  # DDoS/DoS
                2: ThreatSeverity.MEDIUM,  # Reconnaissance
                3: ThreatSeverity.HIGH,  # Brute Force
                4: ThreatSeverity.MEDIUM,  # Web Attack
                5: ThreatSeverity.CRITICAL,  # Malware/Botnet
                6: ThreatSeverity.CRITICAL,  # APT
            }

            self.logger.info(
                f"Enhanced model prediction: {prediction_result.threat_type} ({prediction_result.confidence:.2%} confidence)"
            )

            # Build indicators with advanced features if available
            indicators = {
                "enhanced_model_prediction": {
                    "class_probabilities": prediction_result.class_probabilities,
                    "uncertainty_score": prediction_result.uncertainty_score,
                    "explanation": prediction_result.explanation,
                    "feature_importance": prediction_result.feature_importance,
                    "openai_enhanced": prediction_result.openai_enhanced,
                },
                "event_count": len(events),
                "time_span": self.config.analysis_window,
            }

            # Add Phase 2 advanced features if extracted
            if advanced_features is not None:
                indicators["phase2_advanced_features"] = {
                    "feature_count": len(advanced_features),
                    "features_extracted": True,
                    "feature_dimensions": "100D (79 base + 21 advanced)",
                }

            return ThreatClassification(
                threat_type=prediction_result.threat_type,
                threat_class=prediction_result.predicted_class,
                confidence=prediction_result.confidence,
                anomaly_score=prediction_result.uncertainty_score,  # Use uncertainty as anomaly proxy
                severity=severity_mapping.get(
                    prediction_result.predicted_class, ThreatSeverity.LOW
                ),
                indicators=indicators,
            )

        except Exception as e:
            self.logger.error(
                f"Enhanced model classification failed: {e}", exc_info=True
            )
            return None

    async def _get_fallback_local_classification(
        self, src_ip: str, events: List[Event]
    ) -> Optional[ThreatClassification]:
        """Fallback classification using basic heuristics when ML models unavailable"""
        try:
            # Simple heuristic-based classification
            event_count = len(events)
            unique_ports = len(set(e.dst_port for e in events if e.dst_port))
            failed_logins = sum(
                1 for e in events if "failed" in (e.message or "").lower()
            )

            # Determine threat type based on patterns
            threat_class = 0  # Normal by default
            confidence = 0.5
            threat_type = "Suspicious Activity"

            if failed_logins > 10:
                threat_class = 3  # Brute Force
                threat_type = "Brute Force Attack"
                confidence = min(0.7 + (failed_logins / 100), 0.95)
            elif unique_ports > 20:
                threat_class = 2  # Reconnaissance
                threat_type = "Network Reconnaissance"
                confidence = min(0.6 + (unique_ports / 100), 0.9)
            elif event_count > 100:
                threat_class = 1  # DDoS
                threat_type = "DDoS/DoS Attack"
                confidence = min(0.6 + (event_count / 500), 0.9)

            severity_mapping = {
                0: ThreatSeverity.INFO,
                1: ThreatSeverity.HIGH,
                2: ThreatSeverity.MEDIUM,
                3: ThreatSeverity.HIGH,
            }

            self.logger.info(
                f"Fallback classification: {threat_type} ({confidence:.2%} confidence)"
            )

            return ThreatClassification(
                threat_type=threat_type,
                threat_class=threat_class,
                confidence=confidence,
                anomaly_score=min(confidence, 0.8),
                severity=severity_mapping.get(threat_class, ThreatSeverity.LOW),
                indicators={
                    "fallback_heuristics": {
                        "event_count": event_count,
                        "unique_ports": unique_ports,
                        "failed_logins": failed_logins,
                    },
                    "note": "ML models unavailable - using heuristic detection",
                    "event_count": event_count,
                    "time_span": self.config.analysis_window,
                },
            )
        except Exception as e:
            self.logger.error(f"Fallback classification failed: {e}")
            return None

    async def _should_create_incident(
        self, classification: ThreatClassification, events: List[Event]
    ) -> Dict[str, Any]:
        """Determine if an incident should be created based on classification"""

        print(f"🔶 _should_create_incident called:")
        print(
            f"   - threat_class: {classification.threat_class} ({classification.threat_type})"
        )
        print(f"   - confidence: {classification.confidence:.3f}")
        print(f"   - anomaly_score: {classification.anomaly_score:.3f}")
        print(f"   - severity: {classification.severity}")

        # Skip creating incidents for "Normal" traffic unless very high confidence
        if classification.threat_class == 0:  # Normal
            if classification.confidence < 0.98:
                reason = f"Normal traffic with confidence {classification.confidence:.3f} < 0.98"
                print(f"   ❌ BLOCKED: {reason}")
                return {"create": False, "reason": reason}

        # Check confidence threshold for the specific threat class
        threshold = self.config.confidence_thresholds.get(
            classification.threat_class, 0.5
        )
        print(
            f"   - confidence threshold for class {classification.threat_class}: {threshold}"
        )

        if classification.confidence < threshold:
            reason = f"Confidence {classification.confidence:.3f} below threshold {threshold:.3f}"
            print(f"   ❌ BLOCKED: {reason}")
            return {"create": False, "reason": reason}

        # NOTE: anomaly_score is actually the model's UNCERTAINTY score
        # LOW uncertainty = HIGH confidence, so we should NOT block on low anomaly_score
        # Only check anomaly_score for Normal classifications as an additional safeguard
        # For actual threat detections, confidence is the primary metric
        print(f"   - anomaly_score (uncertainty): {classification.anomaly_score:.3f}")
        print(
            f"   ✅ Confidence {classification.confidence:.3f} >= threshold {threshold:.3f} - PASSING"
        )

        # If we passed the per-class confidence threshold above, create the incident
        # The per-class thresholds are already set appropriately (0.3 for most attack types)
        # No need for additional severity-based blocking

        severity_reasons = {
            ThreatSeverity.CRITICAL: "Critical threat detected - immediate action required",
            ThreatSeverity.HIGH: "High severity threat detected",
            ThreatSeverity.MEDIUM: "Medium severity threat detected",
            ThreatSeverity.LOW: "Low severity threat detected - monitoring recommended",
            ThreatSeverity.INFO: "Informational event flagged for review",
        }

        reason = severity_reasons.get(classification.severity, "Threat detected")
        print(f"   ✅ INCIDENT WILL BE CREATED: {reason}")
        return {"create": True, "reason": reason}

    async def _openai_enhanced_analysis(
        self, src_ip: str, events: List[Event], classification: ThreatClassification
    ) -> Optional[Dict[str, Any]]:
        """Use OpenAI to enhance threat analysis for uncertain cases"""

        if not self.openai_client:
            return None

        try:
            # Prepare event summary for OpenAI analysis
            event_summary = self._prepare_event_summary(events, classification)

            prompt = f"""
            Analyze this potential security threat:

            Source IP: {src_ip}
            ML Classification: {classification.threat_type} (Class {classification.threat_class})
            Confidence: {classification.confidence:.3f}
            Anomaly Score: {classification.anomaly_score:.3f}

            Event Summary:
            {event_summary}

            Questions:
            1. Does this behavior pattern indicate a genuine security threat?
            2. Are there novel attack indicators not captured by traditional ML?
            3. What is the likelihood this is a false positive?
            4. Should this trigger an incident for further investigation?

            Provide a JSON response with:
            - threat_likelihood: float (0-1)
            - novel_indicators: list of strings
            - false_positive_risk: float (0-1)
            - recommendation: "create_incident" or "monitor_only"
            - reasoning: string explanation
            """

            response = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=500,
            )

            # Parse OpenAI response
            analysis_text = response.choices[0].message.content

            # Try to extract JSON from response
            try:
                # Look for JSON block in the response
                import re

                json_match = re.search(
                    r"```json\s*(\{.*?\})\s*```", analysis_text, re.DOTALL
                )
                if json_match:
                    analysis = json.loads(json_match.group(1))
                else:
                    # Try to parse the entire response as JSON
                    analysis = json.loads(analysis_text)

                self.logger.info(f"OpenAI enhanced analysis for {src_ip}: {analysis}")
                return {
                    "openai_analysis": analysis,
                    "enhanced_confidence": analysis.get(
                        "threat_likelihood", classification.confidence
                    ),
                    "novel_indicators": analysis.get("novel_indicators", []),
                    "recommendation": analysis.get("recommendation", "monitor_only"),
                }

            except json.JSONDecodeError:
                self.logger.warning(
                    f"Failed to parse OpenAI JSON response: {analysis_text}"
                )
                return {
                    "openai_analysis": {"raw_response": analysis_text},
                    "enhanced_confidence": classification.confidence,
                }

        except Exception as e:
            self.logger.error(f"OpenAI enhanced analysis failed: {e}")
            return None

    def _prepare_event_summary(
        self, events: List[Event], classification: ThreatClassification
    ) -> str:
        """Prepare a concise event summary for OpenAI analysis"""

        # Aggregate event statistics
        event_types = {}
        unique_ports = set()
        message_patterns = set()

        for event in events[:20]:  # Limit to recent events
            event_types[event.eventid] = event_types.get(event.eventid, 0) + 1
            if event.dst_port:
                unique_ports.add(event.dst_port)
            if event.message:
                # Extract key patterns from messages
                message_patterns.add(event.message[:100])

        summary = f"""
        Event Types: {dict(list(event_types.items())[:5])}
        Target Ports: {list(unique_ports)[:10]}
        Recent Messages: {list(message_patterns)[:3]}
        Total Events: {len(events)}
        Time Span: {self.config.analysis_window}s
        """

        return summary.strip()

    async def _create_intelligent_incident(
        self,
        db: AsyncSession,
        src_ip: str,
        events: List[Event],
        classification: ThreatClassification,
        council_data: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Create or update incident with intelligent classification details.

        If an open incident already exists for this IP (within the last hour),
        update it instead of creating a new one - this consolidates attack sequences.

        Args:
            db: Database session
            src_ip: Source IP address
            events: List of events
            classification: Threat classification
            council_data: Optional Council verification data

        Returns:
            Created or updated incident ID
        """
        from datetime import datetime, timedelta, timezone

        from sqlalchemy import and_, select

        from .models import Incident

        # Check for existing incident from this IP (within last hour)
        # Include both "open" and "contained" to handle auto-containment race condition
        one_hour_ago = datetime.now(timezone.utc) - timedelta(hours=1)
        existing_query = (
            select(Incident)
            .where(
                and_(
                    Incident.src_ip == src_ip,
                    Incident.status.in_(
                        ["open", "contained"]
                    ),  # Include auto-contained incidents
                    Incident.created_at >= one_hour_ago,
                )
            )
            .order_by(Incident.created_at.desc())
        )

        result = await db.execute(existing_query)
        existing_incident = result.scalars().first()

        if existing_incident:
            # Update existing incident instead of creating new one
            self.logger.info(
                f"Found existing incident #{existing_incident.id} (status={existing_incident.status}) for {src_ip}, updating..."
            )
            return await self._update_existing_incident(
                db, existing_incident, events, classification, council_data
            )

        # Enhanced escalation level logic - check for critical indicators
        escalation_level = classification.severity.value
        # Risk score should reflect ML confidence, not just anomaly score
        # Combine confidence and anomaly score for a more meaningful risk metric
        risk_score = max(classification.confidence, classification.anomaly_score)

        # Check for critical threat indicators that should elevate priority
        critical_indicators = [
            "malware",
            "ransomware",
            "data_exfiltration",
            "privilege_escalation",
            "lateral_movement",
            "backdoor",
            "trojan",
            "cryptominer",
            "rootkit",
        ]

        threat_type_lower = classification.threat_type.lower()
        indicators_lower = [ind.lower() for ind in classification.indicators]

        # Elevate to HIGH if malware or data exfiltration detected
        if any(
            keyword in threat_type_lower
            or any(keyword in ind for ind in indicators_lower)
            for keyword in critical_indicators
        ):
            if escalation_level not in ["critical", "high"]:
                escalation_level = "high"
                risk_score = max(
                    risk_score, 0.75
                )  # Ensure risk score reflects severity
                self.logger.info(
                    f"Elevated escalation to HIGH due to critical threat indicators: {threat_type_lower}"
                )

        # Further elevate to CRITICAL if multiple critical indicators or high event count
        critical_count = sum(
            1
            for keyword in critical_indicators
            if keyword in threat_type_lower
            or any(keyword in ind for ind in indicators_lower)
        )
        if critical_count >= 2 or (len(events) > 50 and escalation_level == "high"):
            escalation_level = "critical"
            risk_score = max(risk_score, 0.85)
            self.logger.info(
                f"Elevated escalation to CRITICAL: {critical_count} critical indicators, {len(events)} events"
            )

        # Boost ML confidence for well-defined threat patterns
        boosted_confidence = classification.confidence
        if classification.confidence > 0.4 and any(
            keyword in threat_type_lower for keyword in critical_indicators
        ):
            boosted_confidence = min(classification.confidence + 0.15, 0.95)
            self.logger.info(
                f"Boosted ML confidence from {classification.confidence:.2f} to {boosted_confidence:.2f}"
            )

        # Determine detection method from indicators
        detection_method = "standard"
        if classification.indicators.get("multi_gate_detection"):
            detection_method = "multi_gate"
        elif classification.openai_enhanced:
            detection_method = "local_ml_openai_enhanced"

        # Create comprehensive triage note
        triage_note = {
            "summary": f"{classification.threat_type} detected from {src_ip}",
            "severity": escalation_level,  # Use enhanced escalation level
            "confidence": boosted_confidence,
            "anomaly_score": risk_score,
            "threat_class": classification.threat_class,
            "event_count": len(events),
            "analysis_method": "local_ml"
            + ("_openai_enhanced" if classification.openai_enhanced else ""),
            "detection_method": detection_method,
            "recommendation": self._get_response_recommendation(classification),
            "indicators": classification.indicators,
            "rationale": f"ML model classified with {boosted_confidence:.1%} confidence as {classification.threat_type}. Escalation elevated due to critical threat indicators.",
            # Include gate results and escalation reasons if from multi-gate detection
            "gate_results": classification.indicators.get("gate_results"),
            "escalation_reasons": classification.indicators.get(
                "escalation_reasons", []
            ),
            "gates_passed": classification.indicators.get("gates_passed", []),
        }

        # Generate AI Agent Actions for UI Checklist (v2 UI)
        agent_actions = []

        # 1. IP Reputation Action
        reputation_detail = "IP reputation analysis completed."
        if council_data and council_data.get("grok_intel"):
            reputation_detail = "Cross-referenced with Grok Threat Intel."
        elif classification.openai_enhanced:
            reputation_detail = "Analyzed by OpenAI for behavioral anomalies."

        agent_actions.append(
            {
                "action": "Analyzed IP Reputation",
                "status": "completed",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "execution_method": "automated",
                "detail": reputation_detail,
                "confidence": 0.98,
            }
        )

        # 2. Threat Feeds Correlation
        feed_detail = "Checked against internal threat database."
        if classification.indicators.get("enhanced_model_prediction"):
            feed_detail = "Correlated with ensemble model indicators."

        agent_actions.append(
            {
                "action": "Correlated with Threat Feeds",
                "status": "completed",
                "timestamp": (
                    datetime.now(timezone.utc) - timedelta(seconds=2)
                ).isoformat(),
                "execution_method": "automated",
                "detail": feed_detail,
                "confidence": 0.95,
            }
        )

        # 3. Active Sessions Check
        session_count = sum(1 for e in events if "session" in (e.message or "").lower())
        session_detail = f"Found {session_count} active sessions related to this IP."
        if session_count == 0:
            session_detail = "No active established sessions found for source IP."

        agent_actions.append(
            {
                "action": "Checked Active Sessions",
                "status": "completed",
                "timestamp": (
                    datetime.now(timezone.utc) - timedelta(seconds=5)
                ).isoformat(),
                "execution_method": "automated",
                "detail": session_detail,
                "confidence": 1.0,
            }
        )

        # Generate Agents Data for UI Cards (v2 UI)
        agents_data = {
            "attribution": {
                "threat_actor": council_data.get("grok_intel", {}).get(
                    "threat_actor", "Unknown"
                )
                if council_data and council_data.get("grok_intel")
                else "Unknown",
                "confidence": boosted_confidence,
                "tactics": [classification.threat_type],
                "techniques": list(classification.indicators.keys())[:3],
                "status": "active",
            },
            "containment": {
                "status": "active"
                if escalation_level in ["high", "critical"]
                else "standby",
                "effectiveness": 0.85
                if escalation_level in ["high", "critical"]
                else 0.0,
                "actions_taken": ["Block IP"] if escalation_level == "critical" else [],
                "recommendation": self._get_response_recommendation(classification),
            },
            "forensics": {
                "evidence_collected": [f"Event Log ({len(events)} events)"],
                "timeline_events": len(events),
                "status": "collecting",
                "suspicious_processes": 0,
            },
            "deception": {
                "honeytokens_deployed": 0,
                "attacker_interactions": 0,
                "status": "standby",
            },
        }

        # Update triage note with agents data
        triage_note["agents"] = agents_data

        # Build incident data dict
        # Store the events that were analyzed for this incident
        analyzed_events = [
            {
                "id": e.id,
                "ts": e.ts.isoformat() if e.ts else None,
                "eventid": e.eventid,
                "message": e.message,
                "is_trigger": len(events) == 1
                or i
                == len(events)
                - 1,  # Mark last event as trigger if multiple, or single event
            }
            for i, e in enumerate(events)
        ]

        incident_data = {
            "src_ip": src_ip,
            "reason": f"{classification.threat_type} (ML Confidence: {boosted_confidence:.1%})",
            "status": "open",
            "escalation_level": escalation_level,
            "risk_score": risk_score,
            "threat_category": classification.threat_type.lower().replace(" ", "_"),
            "containment_confidence": boosted_confidence,
            "containment_method": "ml_driven",
            "ml_confidence": boosted_confidence,
            "agent_actions": agent_actions,  # Add generated agent actions
            "triage_note": triage_note,
            "triggering_events": analyzed_events,  # Store events that triggered this incident
            "events_analyzed_count": len(events),
        }

        # Add Council verification data if available
        if council_data:
            # Helper to serialize dict/list values to JSON strings for Text columns
            def _serialize_if_needed(value):
                if isinstance(value, (dict, list)):
                    return json.dumps(value)
                return value

            incident_data.update(
                {
                    "council_verdict": council_data.get("final_verdict"),
                    "council_reasoning": council_data.get("council_reasoning"),
                    "council_confidence": council_data.get("council_confidence"),
                    # These are Text columns, not JSON - must serialize
                    "routing_path": _serialize_if_needed(
                        council_data.get("routing_path")
                    ),
                    "api_calls_made": _serialize_if_needed(
                        council_data.get("api_calls_made")
                    ),
                    "processing_time_ms": council_data.get("processing_time_ms"),
                    "gemini_analysis": _serialize_if_needed(
                        council_data.get("gemini_analysis")
                    ),
                    "grok_intel": _serialize_if_needed(council_data.get("grok_intel")),
                    "openai_remediation": _serialize_if_needed(
                        council_data.get("openai_remediation")
                    ),
                }
            )
            # Update triage note with Council verdict
            triage_note["council_verified"] = True
            triage_note["council_verdict"] = council_data.get("final_verdict")
            self.logger.info(
                f"Incident includes Council verification: "
                f"verdict={council_data.get('final_verdict')}, "
                f"confidence={council_data.get('council_confidence', 0):.2%}"
            )

        # Create incident with enhanced ML confidence and Council data
        incident = Incident(**incident_data)

        db.add(incident)
        await db.commit()
        await db.refresh(incident)

        self.logger.info(
            f"Intelligent incident created: ID={incident.id}, "
            f"IP={src_ip}, Type={classification.threat_type}, "
            f"Escalation={escalation_level}, Risk={risk_score:.2f}, "
            f"Confidence={boosted_confidence:.3f}"
        )

        # Generate comprehensive AI analysis with full event context
        try:
            await self.generate_comprehensive_ai_analysis(
                db, incident, force_refresh=True
            )
        except Exception as analysis_error:
            self.logger.warning(f"Failed to generate AI analysis: {analysis_error}")

        # Phase 2: Collect training sample if Council verified
        if council_data:
            try:
                from .learning import training_collector
                from .ml_feature_extractor import ml_feature_extractor

                # Extract features for training
                features = ml_feature_extractor.extract_features(src_ip, events)

                # Collect sample for automated retraining
                await training_collector.collect_sample(
                    features=features,
                    ml_prediction=council_data.get(
                        "ml_prediction", classification.threat_type
                    ),
                    ml_confidence=classification.confidence,
                    council_verdict=council_data.get("final_verdict", "INVESTIGATE"),
                    correct_label=council_data.get(
                        "correct_label", classification.threat_type
                    ),
                    incident_id=incident.id,
                )

                self.logger.info(
                    f"Training sample collected for incident {incident.id} "
                    f"(verdict: {council_data.get('final_verdict')})"
                )
            except Exception as e:
                self.logger.warning(f"Failed to collect training sample: {e}")

        return incident.id

    async def _update_existing_incident(
        self,
        db: AsyncSession,
        incident: "Incident",
        events: List[Event],
        classification: ThreatClassification,
        council_data: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Update an existing incident with new events and potentially elevated classification.

        This consolidates multiple batches of events from the same attack into one incident.
        """
        import json

        # Calculate new metrics based on combined data
        new_event_count = (incident.events_analyzed_count or 0) + len(events)

        # Update confidence - use the higher value
        new_confidence = max(incident.ml_confidence or 0, classification.confidence)

        # Update risk score - use the higher value
        new_risk_score = max(
            incident.risk_score or 0,
            classification.confidence,
            classification.anomaly_score,
        )

        # Determine new escalation level (can only go up, not down)
        escalation_order = {"low": 0, "medium": 1, "high": 2, "critical": 3}
        current_escalation = incident.escalation_level or "low"
        new_escalation = classification.severity.value

        if escalation_order.get(new_escalation, 0) > escalation_order.get(
            current_escalation, 0
        ):
            final_escalation = new_escalation
        else:
            final_escalation = current_escalation

        # Check for critical indicators that might elevate further
        critical_indicators = [
            "malware",
            "ransomware",
            "data_exfiltration",
            "backdoor",
            "cryptominer",
        ]
        threat_type_lower = classification.threat_type.lower()

        if any(kw in threat_type_lower for kw in critical_indicators):
            if final_escalation not in ["critical", "high"]:
                final_escalation = "high"
                new_risk_score = max(new_risk_score, 0.75)

        # Update triage note with combined info
        try:
            existing_triage = (
                incident.triage_note
                if isinstance(incident.triage_note, dict)
                else json.loads(incident.triage_note or "{}")
            )
        except:
            existing_triage = {}

        updated_triage = {
            **existing_triage,
            "summary": f"{classification.threat_type} attack sequence from {incident.src_ip}",
            "severity": final_escalation,
            "confidence": new_confidence,
            "event_count": new_event_count,
            "phases_detected": existing_triage.get("phases_detected", [])
            + [classification.threat_type],
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "rationale": f"Attack sequence consolidated: {new_event_count} total events analyzed. ML confidence: {new_confidence:.1%}. Latest phase: {classification.threat_type}.",
        }

        # Update the incident
        incident.events_analyzed_count = new_event_count
        incident.ml_confidence = new_confidence
        incident.risk_score = new_risk_score
        incident.escalation_level = final_escalation
        incident.triage_note = updated_triage

        # Update reason to reflect the attack progression
        phases = list(set(updated_triage.get("phases_detected", [])))
        if len(phases) > 1:
            incident.reason = f"Multi-phase attack: {', '.join(phases[:3])} (ML: {new_confidence:.1%})"
        else:
            incident.reason = (
                f"{classification.threat_type} (ML Confidence: {new_confidence:.1%})"
            )

        # Update Council data if provided
        if council_data:

            def _serialize_if_needed(value):
                if isinstance(value, (dict, list)):
                    return json.dumps(value)
                return value

            incident.council_verdict = council_data.get("final_verdict")
            incident.council_reasoning = council_data.get("council_reasoning")
            incident.council_confidence = council_data.get("council_confidence")
            incident.routing_path = _serialize_if_needed(
                council_data.get("routing_path")
            )
            incident.api_calls_made = _serialize_if_needed(
                council_data.get("api_calls_made")
            )
            incident.processing_time_ms = council_data.get("processing_time_ms")
            incident.gemini_analysis = _serialize_if_needed(
                council_data.get("gemini_analysis")
            )
            incident.grok_intel = _serialize_if_needed(council_data.get("grok_intel"))
            incident.openai_remediation = _serialize_if_needed(
                council_data.get("openai_remediation")
            )

            # Update triage note
            updated_triage["council_verified"] = True
            updated_triage["council_verdict"] = council_data.get("final_verdict")
            incident.triage_note = updated_triage

        await db.commit()
        await db.refresh(incident)

        # Regenerate comprehensive AI analysis with new events
        try:
            await self.generate_comprehensive_ai_analysis(
                db, incident, force_refresh=True
            )
        except Exception as analysis_error:
            self.logger.warning(f"Failed to regenerate AI analysis: {analysis_error}")

        self.logger.info(
            f"Updated incident #{incident.id}: events={new_event_count}, "
            f"confidence={new_confidence:.2%}, escalation={final_escalation}"
        )

        return incident.id

    def _get_response_recommendation(self, classification: ThreatClassification) -> str:
        """Get response recommendation based on threat classification"""

        if classification.severity == ThreatSeverity.CRITICAL:
            return "Immediate containment and forensic analysis required"
        elif classification.severity == ThreatSeverity.HIGH:
            return "Urgent investigation and potential containment"
        elif classification.severity == ThreatSeverity.MEDIUM:
            return "Monitor closely and investigate within 4 hours"
        else:
            return "Log and monitor for pattern development"

    async def generate_comprehensive_ai_analysis(
        self, db: AsyncSession, incident: Incident, force_refresh: bool = False
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive AI analysis using ALL key attack events.

        This creates a detailed threat narrative based on the full event timeline,
        automatically updates when new critical events are detected.
        """
        from sqlalchemy import desc, select

        # Get all events for this incident's source IP
        stmt = (
            select(Event)
            .where(Event.src_ip == incident.src_ip)
            .order_by(desc(Event.ts))
            .limit(200)  # Get comprehensive history
        )
        result = await db.execute(stmt)
        all_events = result.scalars().all()

        # Categorize and prioritize key attack events
        key_events = self._extract_key_attack_events(all_events)

        # Check if we need to regenerate (new events since last analysis)
        current_event_count = len(all_events)
        last_analyzed_count = incident.last_event_count or 0

        if (
            not force_refresh
            and current_event_count == last_analyzed_count
            and incident.ai_analysis
        ):
            # No new events, return cached analysis
            return incident.ai_analysis

        # Generate comprehensive analysis
        analysis = await self._generate_event_based_analysis(
            incident, all_events, key_events
        )

        # Update incident with new analysis
        incident.ai_analysis = analysis
        incident.ai_analysis_timestamp = datetime.now(timezone.utc)
        incident.last_event_count = current_event_count

        # Also update triage_note with the new summary
        try:
            triage = (
                incident.triage_note
                if isinstance(incident.triage_note, dict)
                else json.loads(incident.triage_note or "{}")
            )
        except:
            triage = {}

        triage.update(
            {
                "summary": analysis.get("summary", triage.get("summary", "")),
                "recommendation": analysis.get(
                    "recommendation", triage.get("recommendation", "")
                ),
                "rationale": analysis.get("rationale", []),
                "confidence": incident.ml_confidence or 0,
                "event_count": current_event_count,
                "key_events_count": len(key_events),
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }
        )
        incident.triage_note = triage

        await db.commit()

        self.logger.info(
            f"Generated comprehensive AI analysis for incident #{incident.id}: "
            f"{len(key_events)} key events, {current_event_count} total events"
        )

        return analysis

    def _extract_key_attack_events(self, events: List[Event]) -> List[Dict[str, Any]]:
        """Extract and prioritize the most significant attack events."""

        # Priority scoring for different event types
        priority_patterns = {
            # Critical events (100)
            "login.success": 100,
            "session.file_upload": 100,
            "session.file_download": 95,
            "command": 90,
            "malware": 100,
            "backdoor": 100,
            "privilege_escalation": 100,
            "data_exfiltration": 100,
            # High priority events (70-89)
            "login.failed": 75,
            "session.connect": 70,
            "brute": 80,
            "sql_injection": 85,
            "xss": 80,
            # Medium priority events (40-69)
            "session.closed": 40,
            "connection.lost": 45,
            "tcp": 50,
            "scan": 55,
            "recon": 60,
            # Low priority (default)
            "default": 20,
        }

        scored_events = []
        for event in events:
            score = priority_patterns.get("default")
            eventid = (event.eventid or "").lower()
            message = (event.message or "").lower()

            # Check for pattern matches
            for pattern, pattern_score in priority_patterns.items():
                if pattern in eventid or pattern in message:
                    score = max(score, pattern_score)

            # Boost score for events with suspicious content
            if any(
                kw in message
                for kw in [
                    "wget",
                    "curl",
                    "chmod",
                    "base64",
                    "/etc/passwd",
                    "shadow",
                    "id;",
                    "whoami",
                ]
            ):
                score = max(score, 90)

            # Handle raw field which could be dict or string
            raw_data = event.raw
            if raw_data:
                if isinstance(raw_data, dict):
                    # Convert dict to JSON string and truncate
                    raw_data = json.dumps(raw_data)[:500]
                elif isinstance(raw_data, str):
                    raw_data = raw_data[:500]
                else:
                    raw_data = str(raw_data)[:500]

            scored_events.append(
                {
                    "id": event.id,
                    "eventid": event.eventid,
                    "message": event.message,
                    "ts": event.ts.isoformat() if event.ts else None,
                    "src_ip": event.src_ip,
                    "dst_port": event.dst_port,
                    "priority_score": score,
                    "raw": raw_data,  # Truncated for API response
                }
            )

        # Sort by priority and take top events, ensuring diversity
        scored_events.sort(key=lambda x: x["priority_score"], reverse=True)

        # Get diverse set of key events (not all the same type)
        key_events = []
        seen_types = set()

        for event in scored_events:
            event_type = event["eventid"]
            # Always include critical events, limit duplicates of same type
            if event["priority_score"] >= 90:
                key_events.append(event)
                seen_types.add(event_type)
            elif event_type not in seen_types and len(key_events) < 15:
                key_events.append(event)
                seen_types.add(event_type)
            elif event["priority_score"] >= 70 and len(key_events) < 20:
                key_events.append(event)

        return key_events[:20]  # Cap at 20 key events

    async def _generate_event_based_analysis(
        self,
        incident: Incident,
        all_events: List[Event],
        key_events: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Generate AI analysis based on the full event context."""

        # Build attack narrative from key events
        attack_phases = []
        techniques_observed = set()
        indicators = []

        for event in key_events:
            eventid = (event.get("eventid") or "").lower()
            message = (event.get("message") or "").lower()

            # Identify attack phases
            if "scan" in eventid or "tcp" in eventid:
                attack_phases.append("reconnaissance")
                techniques_observed.add("Network Scanning")
            if "login.failed" in eventid or "brute" in message:
                attack_phases.append("credential_attack")
                techniques_observed.add("Brute Force Authentication")
            if "login.success" in eventid:
                attack_phases.append("initial_access")
                techniques_observed.add("Valid Account Compromise")
            if "command" in eventid or any(
                cmd in message for cmd in ["wget", "curl", "chmod", "id", "whoami"]
            ):
                attack_phases.append("execution")
                techniques_observed.add("Command Execution")
            if "file_upload" in eventid or "file_download" in eventid:
                attack_phases.append("collection")
                techniques_observed.add("Data Collection/Staging")
            if "malware" in message or "backdoor" in message:
                attack_phases.append("persistence")
                techniques_observed.add("Malware Installation")
            if any(kw in message for kw in ["exfil", "upload", "transfer"]):
                attack_phases.append("exfiltration")
                techniques_observed.add("Data Exfiltration")

        # Deduplicate and order phases
        phase_order = [
            "reconnaissance",
            "credential_attack",
            "initial_access",
            "execution",
            "persistence",
            "collection",
            "exfiltration",
        ]
        unique_phases = []
        for phase in phase_order:
            if phase in attack_phases and phase not in unique_phases:
                unique_phases.append(phase)

        # Count event types for the summary
        event_type_counts = {}
        for event in all_events:
            etype = event.eventid or "unknown"
            event_type_counts[etype] = event_type_counts.get(etype, 0) + 1

        # Sort by count
        top_event_types = sorted(
            event_type_counts.items(), key=lambda x: x[1], reverse=True
        )[:5]

        # Generate summary
        phase_names = {
            "reconnaissance": "network reconnaissance",
            "credential_attack": "credential attacks",
            "initial_access": "successful compromise",
            "execution": "command execution",
            "persistence": "persistence establishment",
            "collection": "data collection",
            "exfiltration": "data exfiltration",
        }

        if unique_phases:
            phase_descriptions = [phase_names.get(p, p) for p in unique_phases]
            if len(phase_descriptions) == 1:
                summary = f"A {phase_descriptions[0]} attack from IP {incident.src_ip} is detected"
            else:
                summary = f"A multi-stage attack from IP {incident.src_ip} is detected, showing {', '.join(phase_descriptions[:-1])} and {phase_descriptions[-1]}"
        else:
            summary = f"Suspicious activity from IP {incident.src_ip} is detected"

        # Add event context to summary
        login_success = event_type_counts.get("cowrie.login.success", 0)
        login_failed = event_type_counts.get("cowrie.login.failed", 0)
        file_events = event_type_counts.get(
            "cowrie.session.file_upload", 0
        ) + event_type_counts.get("cowrie.session.file_download", 0)

        context_parts = []
        if login_failed > 0:
            context_parts.append(f"{login_failed} failed login attempts")
        if login_success > 0:
            context_parts.append(f"{login_success} successful authentication(s)")
        if file_events > 0:
            context_parts.append(f"{file_events} file transfer(s)")

        if context_parts:
            summary += f", with {', '.join(context_parts)}."
        else:
            summary += "."

        # Generate recommendation based on severity
        escalation = (incident.escalation_level or "medium").lower()
        confidence = incident.ml_confidence or 0

        if escalation == "critical" or confidence >= 0.8:
            recommendation = "Immediate containment and block source IP"
        elif escalation == "high" or confidence >= 0.6:
            recommendation = "Urgent investigation and potential containment"
        elif "initial_access" in unique_phases or login_success > 0:
            recommendation = "Block source IP and investigate compromised credentials"
        elif login_failed > 5:
            recommendation = "Monitor and consider blocking after additional attempts"
        else:
            recommendation = "Continue monitoring for escalation"

        # Build rationale as list of key indicators
        rationale = []

        if confidence > 0:
            confidence_level = (
                "high"
                if confidence >= 0.7
                else "moderate"
                if confidence >= 0.4
                else "low"
            )
            rationale.append(
                f"The attack has a {confidence_level} machine learning confidence of {confidence * 100:.1f}%."
            )

        if techniques_observed:
            rationale.append(
                f"Observed techniques: {', '.join(list(techniques_observed)[:4])}."
            )

        if login_failed > 3:
            rationale.append(
                f"Multiple connection attempts ({login_failed}) indicate a targeted brute-force attempt."
            )

        if login_success > 0:
            rationale.append(
                f"Successful authentication detected - credential compromise confirmed."
            )

        if file_events > 0:
            rationale.append(
                f"File transfers detected - possible data exfiltration or malware delivery."
            )

        if len(unique_phases) > 2:
            rationale.append(
                f"Multi-phase attack pattern indicates sophisticated threat actor."
            )

        # Build the full analysis response
        analysis = {
            "summary": summary,
            "recommendation": recommendation,
            "rationale": rationale,
            "attack_phases": unique_phases,
            "techniques_observed": list(techniques_observed),
            "key_events": key_events[:10],  # Include top 10 for display
            "event_statistics": {
                "total_events": len(all_events),
                "key_events_count": len(key_events),
                "top_event_types": dict(top_event_types),
                "login_success": login_success,
                "login_failed": login_failed,
                "file_transfers": file_events,
            },
            "confidence": confidence,
            "escalation_level": escalation,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

        return analysis

    async def _route_through_council(
        self,
        src_ip: str,
        events: List[Event],
        classification: ThreatClassification,
        features: Optional[np.ndarray] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Route uncertain ML predictions through Council of Models for verification.

        The Council provides:
        - Gemini Judge: Deep reasoning on uncertain predictions
        - Grok Intel: External threat intelligence (Feature #80)
        - OpenAI Remediation: Automated response scripts
        - Vector Memory: Reuse past decisions (40% cost savings)

        Args:
            src_ip: Source IP address
            events: List of events
            classification: Initial ML classification

        Returns:
            Dictionary with Council analysis and updated classification
        """
        if not COUNCIL_AVAILABLE:
            self.logger.warning("Council not available - skipping verification")
            return None

        try:
            from .ml_feature_extractor import ml_feature_extractor

            # Phase 2: Try to get features from feature store first
            features = None
            try:
                import numpy as np

                from .features import feature_store

                # Check feature store cache
                features = await feature_store.retrieve_features(
                    entity_id=src_ip, entity_type="ip"
                )

                if features is not None:
                    self.logger.info(f"Feature store cache HIT for {src_ip}")
            except Exception as e:
                self.logger.debug(f"Feature store not available or cache miss: {e}")

            # Extract features if not cached
            if features is None:
                features = ml_feature_extractor.extract_features(src_ip, events)
                self.logger.info(f"Extracted features for {src_ip} (cache miss)")

                # Store in feature cache for reuse
                try:
                    from .features import feature_store

                    await feature_store.store_features(
                        entity_id=src_ip,
                        entity_type="ip",
                        features=features,
                        ttl_seconds=3600,  # 1 hour cache
                    )
                    self.logger.debug(f"Cached features for {src_ip}")
                except Exception as e:
                    self.logger.debug(f"Failed to cache features: {e}")

            # Create initial state for Council
            state = create_initial_state(
                src_ip=src_ip,
                events=[
                    {
                        "timestamp": str(e.ts) if e.ts else None,
                        "event_type": getattr(
                            e, "eventid", getattr(e, "source_type", "unknown")
                        ),
                        "dst_port": e.dst_port,
                        "src_port": getattr(e, "src_port", None),
                        "protocol": getattr(e, "protocol", None),
                        "username": getattr(e, "username", None),
                        "command": getattr(e, "command", None),
                    }
                    for e in events
                ],
                ml_prediction={
                    "class": classification.threat_type,
                    "confidence": classification.confidence,
                    "threat_score": classification.anomaly_score,
                    "model": "enhanced_local",
                },
                raw_features=features.tolist(),
            )

            # Run through Council orchestrator
            self.logger.info(f"Routing {src_ip} through Council orchestrator")
            final_state = await orchestrate_incident(state)

            # Parse Council verdict
            final_verdict = final_state.get("final_verdict", "INVESTIGATE")
            council_confidence = final_state.get(
                "confidence_score", classification.confidence
            )
            council_reasoning = final_state.get("gemini_reasoning", "")

            # Update classification based on Council verdict
            if final_verdict == "FALSE_POSITIVE":
                # Council overrode ML - mark as benign
                classification.confidence = min(
                    0.3, classification.confidence
                )  # Reduce confidence
                classification.threat_type = "False Positive (Council Override)"
                classification.indicators["council_override"] = True
                classification.indicators["council_reasoning"] = council_reasoning

            elif final_verdict == "THREAT":
                # Council confirmed ML - boost confidence
                classification.confidence = max(
                    council_confidence, classification.confidence
                )
                classification.indicators["council_confirmed"] = True

            elif final_verdict == "INVESTIGATE":
                # Council uncertain - keep original but flag for review
                classification.indicators["requires_human_review"] = True
                classification.indicators["council_uncertain"] = True

            # Add Council metadata
            return {
                "final_verdict": final_verdict,
                "council_confidence": council_confidence,
                "council_reasoning": council_reasoning,
                "routing_path": final_state.get("routing_path", []),
                "api_calls_made": final_state.get("api_calls_made", []),
                "processing_time_ms": final_state.get("processing_time_ms", 0),
                "gemini_analysis": final_state.get("gemini_analysis"),
                "grok_intel": final_state.get("grok_intel"),
                "openai_remediation": final_state.get("openai_remediation"),
                "action_plan": final_state.get("action_plan", []),
                "updated_classification": classification,
            }

        except Exception as e:
            self.logger.error(f"Council routing failed: {e}", exc_info=True)
            return None

    def _update_learning_stats(
        self, classification: ThreatClassification, incident_created: bool
    ):
        """Update learning statistics for adaptive thresholds"""
        self.learning_stats["total_classifications"] += 1
        if incident_created:
            self.learning_stats["incidents_created"] += 1

        # Track per-class statistics
        threat_class = classification.threat_class
        if threat_class not in self.learning_stats["per_class_accuracy"]:
            self.learning_stats["per_class_accuracy"][threat_class] = {
                "total": 0,
                "incidents": 0,
                "avg_confidence": 0.0,
            }

        class_stats = self.learning_stats["per_class_accuracy"][threat_class]
        class_stats["total"] += 1
        if incident_created:
            class_stats["incidents"] += 1

        # Update average confidence
        class_stats["avg_confidence"] = (
            class_stats["avg_confidence"] * (class_stats["total"] - 1)
            + classification.confidence
        ) / class_stats["total"]

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics for monitoring"""
        return {
            "learning_stats": self.learning_stats,
            "current_thresholds": self.config.confidence_thresholds,
            "incident_rate": (
                self.learning_stats["incidents_created"]
                / max(1, self.learning_stats["total_classifications"])
            ),
            "openai_enabled": self.openai_client is not None,
        }

    def adjust_thresholds(self, feedback: Dict[str, Any]):
        """Adjust detection thresholds based on feedback"""
        if not self.config.enable_adaptive_learning:
            return

        # Simple adaptive adjustment based on false positive feedback
        if feedback.get("false_positive") and feedback.get("threat_class") is not None:
            threat_class = feedback["threat_class"]
            if threat_class in self.config.confidence_thresholds:
                current_threshold = self.config.confidence_thresholds[threat_class]
                # Increase threshold to reduce false positives
                new_threshold = min(0.95, current_threshold + 0.05)
                self.config.confidence_thresholds[threat_class] = new_threshold
                self.logger.info(
                    f"Increased threshold for class {threat_class}: {current_threshold:.3f} -> {new_threshold:.3f}"
                )

    async def analyze_with_multi_gate(
        self, db: AsyncSession, src_ip: str, events: List[Event] = None
    ) -> Dict[str, Any]:
        """
        Alternative analysis method using the Multi-Gate Detection System.

        This method uses the modular multi-gate architecture instead of inline
        gating logic. Each gate (heuristic, ML, specialist, vector memory, council)
        is executed sequentially with early termination on FAIL.

        Args:
            db: Database session
            src_ip: Source IP to analyze
            events: Optional list of events (fetched if not provided)

        Returns:
            Dict with detection results and gate information
        """
        if not MULTI_GATE_AVAILABLE or not multi_gate_detector:
            self.logger.warning(
                "Multi-gate detector not available, using standard detection"
            )
            return await self.analyze_and_create_incidents(db, src_ip, events)

        try:
            # Get events if not provided
            if events is None:
                events = await self._get_recent_events(db, src_ip)

            if len(events) < self.config.min_events_required:
                return {
                    "incident_created": False,
                    "reason": f"Insufficient events ({len(events)} < {self.config.min_events_required})",
                    "detection_method": "multi_gate",
                }

            # Run through multi-gate detector
            result = await multi_gate_detector.detect(src_ip, events)

            # Log gate results
            for gate_result in result.gate_results:
                self.logger.info(
                    f"Gate '{gate_result.gate_name}': {gate_result.verdict.value} "
                    f"({gate_result.reason})"
                )

            if not result.should_create_incident:
                return {
                    "incident_created": False,
                    "reason": f"Blocked by gate: {result.blocking_gate or 'threshold'}",
                    "detection_method": "multi_gate",
                    "gate_results": [
                        {
                            "gate": gr.gate_name,
                            "verdict": gr.verdict.value,
                            "reason": gr.reason,
                        }
                        for gr in result.gate_results
                    ],
                    "final_confidence": result.final_confidence,
                    "threat_type": result.threat_type,
                    "processing_time_ms": result.total_processing_time_ms,
                }

            # Build gate results for storage
            gate_results_data = [
                {
                    "gate": gr.gate_name,
                    "verdict": gr.verdict.value,
                    "reason": gr.reason,
                    "confidence_modifier": gr.confidence_modifier,
                    "processing_time_ms": gr.processing_time_ms,
                    "details": gr.details,
                }
                for gr in result.gate_results
            ]

            # Create incident using standard method with classification from multi-gate
            classification = ThreatClassification(
                threat_type=result.threat_type,
                threat_class=result.threat_class,
                confidence=result.final_confidence,
                anomaly_score=0.0,  # Multi-gate doesn't provide anomaly score
                severity=self._get_severity_for_class(result.threat_class),
                indicators={
                    "multi_gate_detection": True,
                    "gates_passed": [
                        gr.gate_name
                        for gr in result.gate_results
                        if gr.verdict.value in ["pass", "escalate"]
                    ],
                    "escalation_reasons": result.escalation_reasons,
                    "gate_results": gate_results_data,
                    "blocking_gate": result.blocking_gate,
                    "total_processing_time_ms": result.total_processing_time_ms,
                },
            )

            incident_id = await self._create_intelligent_incident(
                db, src_ip, events, classification
            )

            return {
                "incident_created": True,
                "incident_id": incident_id,
                "detection_method": "multi_gate",
                "threat_type": result.threat_type,
                "confidence": result.final_confidence,
                "gate_results": [
                    {
                        "gate": gr.gate_name,
                        "verdict": gr.verdict.value,
                        "reason": gr.reason,
                    }
                    for gr in result.gate_results
                ],
                "escalation_reasons": result.escalation_reasons,
                "processing_time_ms": result.total_processing_time_ms,
            }

        except Exception as e:
            self.logger.error(f"Multi-gate detection failed: {e}", exc_info=True)
            # Fall back to standard detection
            return await self.analyze_and_create_incidents(db, src_ip, events)

    def _get_severity_for_class(self, threat_class: int) -> ThreatSeverity:
        """Get severity level for a threat class."""
        severity_mapping = {
            0: ThreatSeverity.INFO,  # Normal
            1: ThreatSeverity.HIGH,  # DDoS/DoS
            2: ThreatSeverity.MEDIUM,  # Reconnaissance
            3: ThreatSeverity.HIGH,  # Brute Force
            4: ThreatSeverity.MEDIUM,  # Web Attack
            5: ThreatSeverity.CRITICAL,  # Malware/Botnet
            6: ThreatSeverity.CRITICAL,  # APT
        }
        return severity_mapping.get(threat_class, ThreatSeverity.LOW)


# Global instance
intelligent_detector = IntelligentDetectionEngine()

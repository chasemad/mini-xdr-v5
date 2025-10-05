"""
🚀 INTELLIGENT INCIDENT DETECTION ENGINE
Enhanced detection system that leverages SageMaker ML, OpenAI analysis, and configurable thresholds
to create incidents dynamically based on threat classification and anomaly scores.

Key Features:
1. Direct SageMaker threat classification integration
2. Configurable confidence thresholds per threat type
3. OpenAI-enhanced anomaly detection for novel attacks
4. Adaptive learning from false positives/negatives
5. Multi-layered scoring system with ensemble methods
"""

import asyncio
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

from sqlalchemy import select, and_, func, desc, asc
from sqlalchemy.ext.asyncio import AsyncSession

from .models import Event, Incident
from .config import settings
from .secrets_manager import secrets_manager

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
    sagemaker_used: bool = False
    openai_enhanced: bool = False


@dataclass
class DetectionConfig:
    """Configurable detection thresholds and parameters"""
    # SageMaker ML confidence thresholds by threat class
    confidence_thresholds: Dict[int, float] = None

    # Anomaly score thresholds
    anomaly_score_threshold: float = 0.3

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
            # Default confidence thresholds per threat class
            self.confidence_thresholds = {
                0: 0.95,  # Normal - very high threshold to avoid FPs
                1: 0.6,   # DDoS/DoS - medium threshold
                2: 0.5,   # Network Reconnaissance - lower threshold
                3: 0.7,   # Brute Force Attack - high threshold
                4: 0.6,   # Web Application Attack - medium threshold
                5: 0.8,   # Malware/Botnet - high threshold
                6: 0.9    # APT - very high threshold due to criticality
            }


class IntelligentDetectionEngine:
    """
    Advanced detection engine that combines SageMaker ML predictions,
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
            'total_classifications': 0,
            'incidents_created': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'per_class_accuracy': {}
        }

    def _init_openai_client(self):
        """Initialize OpenAI client for enhanced analysis"""
        try:
            import openai
            # Get OpenAI API key from secrets manager
            openai_key = secrets_manager.get_secret("OPENAI_API_KEY")
            if openai_key:
                self.openai_client = openai.AsyncOpenAI(api_key=openai_key)
                self.logger.info("OpenAI client initialized for enhanced threat analysis")
            else:
                self.logger.warning("OpenAI API key not found - enhanced analysis disabled")
        except Exception as e:
            self.logger.warning(f"Failed to initialize OpenAI client: {e}")

    async def analyze_and_create_incidents(
        self,
        db: AsyncSession,
        src_ip: str,
        events: List[Event] = None
    ) -> Dict[str, Any]:
        """
        Main analysis method that determines if incidents should be created
        based on SageMaker predictions and configurable thresholds
        """
        try:
            # Get recent events if not provided
            if events is None:
                events = await self._get_recent_events(db, src_ip)

            if len(events) < self.config.min_events_required:
                return {
                    'incident_created': False,
                    'reason': f'Insufficient events ({len(events)} < {self.config.min_events_required})'
                }

            # Step 1: Get SageMaker threat classification
            classification = await self._get_sagemaker_classification(src_ip, events)

            if not classification:
                return {
                    'incident_created': False,
                    'reason': 'SageMaker classification failed'
                }

            # Step 2: Check if classification meets threshold for incident creation
            should_create_incident = await self._should_create_incident(classification, events)

            if not should_create_incident['create']:
                return {
                    'incident_created': False,
                    'classification': asdict(classification),
                    'reason': should_create_incident['reason']
                }

            # Step 3: Enhanced analysis with OpenAI (if enabled and needed)
            if (self.config.enable_openai_analysis and
                classification.confidence < self.config.openai_threshold):

                enhanced_analysis = await self._openai_enhanced_analysis(
                    src_ip, events, classification
                )

                if enhanced_analysis:
                    classification.openai_enhanced = True
                    classification.indicators.update(enhanced_analysis)

            # Step 4: Create incident with comprehensive details
            incident_id = await self._create_intelligent_incident(
                db, src_ip, events, classification
            )

            # Step 5: Update learning statistics
            self._update_learning_stats(classification, True)

            return {
                'incident_created': True,
                'incident_id': incident_id,
                'classification': asdict(classification),
                'confidence': classification.confidence,
                'threat_type': classification.threat_type,
                'severity': classification.severity.value
            }

        except Exception as e:
            self.logger.error(f"Intelligent detection failed for {src_ip}: {e}")
            return {
                'incident_created': False,
                'reason': f'Detection error: {str(e)}'
            }

    async def _get_recent_events(
        self,
        db: AsyncSession,
        src_ip: str
    ) -> List[Event]:
        """Get recent events for the given IP within analysis window"""
        window_start = datetime.now(timezone.utc) - timedelta(
            seconds=self.config.analysis_window
        )

        stmt = (
            select(Event)
            .where(
                and_(
                    Event.src_ip == src_ip,
                    Event.ts >= window_start
                )
            )
            .order_by(desc(Event.ts))
            .limit(100)  # Reasonable limit for analysis
        )

        result = await db.execute(stmt)
        return result.scalars().all()

    async def _get_sagemaker_classification(
        self,
        src_ip: str,
        events: List[Event]
    ) -> Optional[ThreatClassification]:
        """Get threat classification from Enhanced Local Model ONLY (No AWS)"""
        try:
            # Use Enhanced Local Model Only - NO AWS/SageMaker
            self.logger.info("Using local enhanced model for threat classification")
            enhanced_result = await self._get_enhanced_model_classification(src_ip, events)
            
            if enhanced_result:
                return enhanced_result
            
            # If enhanced model is not available, use fallback local detection
            self.logger.warning("Enhanced model unavailable, using basic local detection")
            return await self._get_fallback_local_classification(src_ip, events)

        except Exception as e:
            self.logger.error(f"Local model classification failed: {e}")
            return None

    async def _get_enhanced_model_classification(
        self,
        src_ip: str,
        events: List[Event]
    ) -> Optional[ThreatClassification]:
        """Get threat classification from Enhanced Local Model"""
        try:
            from .enhanced_threat_detector import enhanced_detector

            # Check if enhanced detector is loaded
            if not enhanced_detector.model:
                self.logger.warning("Enhanced detector model not loaded yet")
                return None

            # Use enhanced detector for analysis
            prediction_result = await enhanced_detector.analyze_threat(src_ip, events)

            if not prediction_result:
                return None

            # Map to ThreatClassification
            severity_mapping = {
                0: ThreatSeverity.INFO,      # Normal
                1: ThreatSeverity.HIGH,      # DDoS/DoS
                2: ThreatSeverity.MEDIUM,    # Reconnaissance
                3: ThreatSeverity.HIGH,      # Brute Force
                4: ThreatSeverity.MEDIUM,    # Web Attack
                5: ThreatSeverity.CRITICAL,  # Malware/Botnet
                6: ThreatSeverity.CRITICAL   # APT
            }

            self.logger.info(f"Enhanced model prediction: {prediction_result.threat_type} ({prediction_result.confidence:.2%} confidence)")

            return ThreatClassification(
                threat_type=prediction_result.threat_type,
                threat_class=prediction_result.predicted_class,
                confidence=prediction_result.confidence,
                anomaly_score=prediction_result.uncertainty_score,  # Use uncertainty as anomaly proxy
                severity=severity_mapping.get(prediction_result.predicted_class, ThreatSeverity.LOW),
                indicators={
                    'enhanced_model_prediction': {
                        'class_probabilities': prediction_result.class_probabilities,
                        'uncertainty_score': prediction_result.uncertainty_score,
                        'explanation': prediction_result.explanation,
                        'feature_importance': prediction_result.feature_importance,
                        'openai_enhanced': prediction_result.openai_enhanced
                    },
                    'event_count': len(events),
                    'time_span': self.config.analysis_window
                },
                sagemaker_used=False  # Local model used - NO AWS
            )

        except Exception as e:
            self.logger.error(f"Enhanced model classification failed: {e}", exc_info=True)
            return None
    
    async def _get_fallback_local_classification(
        self,
        src_ip: str,
        events: List[Event]
    ) -> Optional[ThreatClassification]:
        """Fallback classification using basic heuristics when ML models unavailable"""
        try:
            # Simple heuristic-based classification
            event_count = len(events)
            unique_ports = len(set(e.dst_port for e in events if e.dst_port))
            failed_logins = sum(1 for e in events if 'failed' in (e.message or '').lower())
            
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
            
            self.logger.info(f"Fallback classification: {threat_type} ({confidence:.2%} confidence)")
            
            return ThreatClassification(
                threat_type=threat_type,
                threat_class=threat_class,
                confidence=confidence,
                anomaly_score=min(confidence, 0.8),
                severity=severity_mapping.get(threat_class, ThreatSeverity.LOW),
                indicators={
                    'fallback_heuristics': {
                        'event_count': event_count,
                        'unique_ports': unique_ports,
                        'failed_logins': failed_logins
                    },
                    'note': 'ML models unavailable - using heuristic detection',
                    'event_count': event_count,
                    'time_span': self.config.analysis_window
                },
                sagemaker_used=False
            )
        except Exception as e:
            self.logger.error(f"Fallback classification failed: {e}")
            return None

    async def _should_create_incident(
        self,
        classification: ThreatClassification,
        events: List[Event]
    ) -> Dict[str, Any]:
        """Determine if an incident should be created based on classification"""

        # Skip creating incidents for "Normal" traffic unless very high confidence
        if classification.threat_class == 0:  # Normal
            if classification.confidence < 0.98:
                return {
                    'create': False,
                    'reason': f'Normal traffic with confidence {classification.confidence:.3f} < 0.98'
                }

        # Check confidence threshold for the specific threat class
        threshold = self.config.confidence_thresholds.get(
            classification.threat_class, 0.5
        )

        if classification.confidence < threshold:
            return {
                'create': False,
                'reason': f'Confidence {classification.confidence:.3f} below threshold {threshold:.3f}'
            }

        # Check anomaly score threshold
        if classification.anomaly_score < self.config.anomaly_score_threshold:
            return {
                'create': False,
                'reason': f'Anomaly score {classification.anomaly_score:.3f} below threshold {self.config.anomaly_score_threshold:.3f}'
            }

        # Additional checks for critical threats
        if classification.severity == ThreatSeverity.CRITICAL:
            # Always create incidents for critical threats with reasonable confidence
            if classification.confidence >= 0.6:
                return {'create': True, 'reason': 'Critical threat detected'}

        # High severity threats with good confidence
        if classification.severity == ThreatSeverity.HIGH:
            if classification.confidence >= 0.7:
                return {'create': True, 'reason': 'High severity threat with high confidence'}

        # Medium/Low threats need higher confidence or enhanced analysis
        if classification.severity in [ThreatSeverity.MEDIUM, ThreatSeverity.LOW]:
            if classification.confidence >= 0.8:
                return {'create': True, 'reason': 'Medium/Low threat with very high confidence'}
            else:
                return {
                    'create': False,
                    'reason': f'Medium/Low severity needs confidence >= 0.8, got {classification.confidence:.3f}'
                }

        return {'create': True, 'reason': 'Default creation criteria met'}

    async def _openai_enhanced_analysis(
        self,
        src_ip: str,
        events: List[Event],
        classification: ThreatClassification
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
                max_tokens=500
            )

            # Parse OpenAI response
            analysis_text = response.choices[0].message.content

            # Try to extract JSON from response
            try:
                # Look for JSON block in the response
                import re
                json_match = re.search(r'```json\s*(\{.*?\})\s*```', analysis_text, re.DOTALL)
                if json_match:
                    analysis = json.loads(json_match.group(1))
                else:
                    # Try to parse the entire response as JSON
                    analysis = json.loads(analysis_text)

                self.logger.info(f"OpenAI enhanced analysis for {src_ip}: {analysis}")
                return {
                    'openai_analysis': analysis,
                    'enhanced_confidence': analysis.get('threat_likelihood', classification.confidence),
                    'novel_indicators': analysis.get('novel_indicators', []),
                    'recommendation': analysis.get('recommendation', 'monitor_only')
                }

            except json.JSONDecodeError:
                self.logger.warning(f"Failed to parse OpenAI JSON response: {analysis_text}")
                return {
                    'openai_analysis': {'raw_response': analysis_text},
                    'enhanced_confidence': classification.confidence
                }

        except Exception as e:
            self.logger.error(f"OpenAI enhanced analysis failed: {e}")
            return None

    def _prepare_event_summary(
        self,
        events: List[Event],
        classification: ThreatClassification
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
        classification: ThreatClassification
    ) -> int:
        """Create incident with intelligent classification details"""

        # Create comprehensive triage note
        triage_note = {
            "summary": f"{classification.threat_type} detected from {src_ip}",
            "severity": classification.severity.value,
            "confidence": classification.confidence,
            "anomaly_score": classification.anomaly_score,
            "threat_class": classification.threat_class,
            "event_count": len(events),
            "analysis_method": "sagemaker_ml" + ("_openai_enhanced" if classification.openai_enhanced else ""),
            "recommendation": self._get_response_recommendation(classification),
            "indicators": classification.indicators,
            "rationale": f"ML model classified with {classification.confidence:.1%} confidence as {classification.threat_type}"
        }

        # Create incident with ML confidence
        incident = Incident(
            src_ip=src_ip,
            reason=f"{classification.threat_type} (ML Confidence: {classification.confidence:.1%})",
            status="open",
            escalation_level=classification.severity.value,
            risk_score=classification.anomaly_score,
            threat_category=classification.threat_type.lower().replace(" ", "_"),
            containment_confidence=classification.confidence,
            containment_method="ml_driven",
            ml_confidence=classification.confidence,  # Set ML confidence explicitly
            triage_note=triage_note
        )

        db.add(incident)
        await db.commit()
        await db.refresh(incident)

        self.logger.info(
            f"Intelligent incident created: ID={incident.id}, "
            f"IP={src_ip}, Type={classification.threat_type}, "
            f"Confidence={classification.confidence:.3f}"
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

    def _update_learning_stats(self, classification: ThreatClassification, incident_created: bool):
        """Update learning statistics for adaptive thresholds"""
        self.learning_stats['total_classifications'] += 1
        if incident_created:
            self.learning_stats['incidents_created'] += 1

        # Track per-class statistics
        threat_class = classification.threat_class
        if threat_class not in self.learning_stats['per_class_accuracy']:
            self.learning_stats['per_class_accuracy'][threat_class] = {
                'total': 0, 'incidents': 0, 'avg_confidence': 0.0
            }

        class_stats = self.learning_stats['per_class_accuracy'][threat_class]
        class_stats['total'] += 1
        if incident_created:
            class_stats['incidents'] += 1

        # Update average confidence
        class_stats['avg_confidence'] = (
            class_stats['avg_confidence'] * (class_stats['total'] - 1) + classification.confidence
        ) / class_stats['total']

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics for monitoring"""
        return {
            'learning_stats': self.learning_stats,
            'current_thresholds': self.config.confidence_thresholds,
            'incident_rate': (
                self.learning_stats['incidents_created'] /
                max(1, self.learning_stats['total_classifications'])
            ),
            'openai_enabled': self.openai_client is not None
        }

    def adjust_thresholds(self, feedback: Dict[str, Any]):
        """Adjust detection thresholds based on feedback"""
        if not self.config.enable_adaptive_learning:
            return

        # Simple adaptive adjustment based on false positive feedback
        if feedback.get('false_positive') and feedback.get('threat_class') is not None:
            threat_class = feedback['threat_class']
            if threat_class in self.config.confidence_thresholds:
                current_threshold = self.config.confidence_thresholds[threat_class]
                # Increase threshold to reduce false positives
                new_threshold = min(0.95, current_threshold + 0.05)
                self.config.confidence_thresholds[threat_class] = new_threshold
                self.logger.info(f"Increased threshold for class {threat_class}: {current_threshold:.3f} -> {new_threshold:.3f}")


# Global instance
intelligent_detector = IntelligentDetectionEngine()
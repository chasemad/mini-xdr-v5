from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from collections import defaultdict, Counter
from sqlalchemy import select, and_, func
from sqlalchemy.ext.asyncio import AsyncSession
from .models import Event, Incident
from .config import settings
import logging
import json

logger = logging.getLogger(__name__)


@dataclass
class AttackPattern:
    """Represents a detected attack pattern"""
    pattern_type: str
    src_ip: str
    severity: str
    confidence: float
    indicators: Dict[str, Any]
    event_count: int
    time_span: int
    description: str


class SlidingWindowDetector:
    """Detects SSH brute-force attacks using sliding window threshold"""
    
    def __init__(self, window_seconds: int = None, threshold: int = None):
        self.window_seconds = window_seconds or settings.fail_window_seconds
        self.threshold = threshold or settings.fail_threshold
        self.target_event_id = "cowrie.login.failed"
    
    async def evaluate(self, db: AsyncSession, src_ip: str) -> Optional[Dict[str, Any]]:
        """
        Evaluate if the src_ip has triggered the detection threshold
        
        Returns:
            Dict with incident details if threshold exceeded, None otherwise
        """
        # Check if there's already an open incident for this IP
        existing_incident = await self._get_open_incident(db, src_ip)
        if existing_incident:
            logger.debug(f"Open incident already exists for {src_ip}: {existing_incident.id}")
            return None
        
        # Count failed login attempts in the window
        window_start = datetime.now(timezone.utc) - timedelta(seconds=self.window_seconds)
        
        query = select(func.count(Event.id)).where(
            and_(
                Event.src_ip == src_ip,
                Event.eventid == self.target_event_id,
                Event.ts >= window_start
            )
        )
        
        result = await db.execute(query)
        fail_count = result.scalar()
        
        logger.info(f"SSH brute-force check: {src_ip} has {fail_count} failures in last {self.window_seconds}s (threshold: {self.threshold})")
        
        if fail_count >= self.threshold:
            # Create incident
            incident_data = {
                "src_ip": src_ip,
                "reason": f"SSH brute-force: {fail_count} failed attempts in {self.window_seconds}s",
                "status": "open",
                "auto_contained": False
            }
            
            incident = Incident(**incident_data)
            db.add(incident)
            await db.flush()  # Get the ID without committing
            
            logger.warning(f"NEW INCIDENT #{incident.id}: SSH brute-force from {src_ip}")
            
            return {
                "incident_id": incident.id,
                "src_ip": src_ip,
                "fail_count": fail_count,
                "threshold": self.threshold,
                "window_seconds": self.window_seconds
            }
        
        return None
    
    async def _get_open_incident(self, db: AsyncSession, src_ip: str) -> Optional[Incident]:
        """Check if there's an open incident for the given IP"""
        query = select(Incident).where(
            and_(
                Incident.src_ip == src_ip,
                Incident.status == "open"
            )
        ).order_by(Incident.created_at.desc())
        
        result = await db.execute(query)
        return result.scalars().first()


class AdvancedCorrelationEngine:
    """Advanced correlation engine for multi-chain attack detection"""
    
    def __init__(self):
        # Detection windows (seconds)
        self.short_window = 300      # 5 minutes
        self.medium_window = 1800    # 30 minutes  
        self.long_window = 3600      # 1 hour
        
        # Thresholds for different attack patterns
        self.thresholds = {
            'password_spray': {
                'min_unique_passwords': 10,
                'min_usernames': 1,
                'max_passwords_per_username': 50,
                'min_attempts': 15
            },
            'credential_stuffing': {
                'min_unique_pairs': 20,
                'min_attempts': 25,
                'unique_pair_ratio': 0.8  # 80% unique pairs
            },
            'username_enumeration': {
                'min_unique_usernames': 8,
                'max_passwords_per_username': 3,
                'min_attempts': 10
            },
            'distributed_brute_force': {
                'min_sessions': 3,
                'min_attempts': 20,
                'session_spread_ratio': 0.3  # Attempts spread across 30%+ of sessions
            },
            'slow_persistent': {
                'min_time_span': 900,  # 15 minutes
                'min_attempts': 15,
                'max_rate_per_minute': 2
            }
        }
    
    async def analyze_ip(self, db: AsyncSession, src_ip: str) -> List[AttackPattern]:
        """Perform comprehensive analysis of an IP's behavior"""
        patterns = []
        
        # Get events for analysis across different time windows
        events_medium = await self._get_failed_login_events(db, src_ip, self.medium_window)
        
        # Check each pattern type - focus on password spray first
        if pattern := await self._detect_password_spray(events_medium):
            patterns.append(pattern)
            
        return patterns
    
    async def _get_failed_login_events(self, db: AsyncSession, src_ip: str, window_seconds: int) -> List[Event]:
        """Get failed login events for an IP within time window"""
        window_start = datetime.now(timezone.utc) - timedelta(seconds=window_seconds)
        
        query = select(Event).where(
            and_(
                Event.src_ip == src_ip,
                Event.eventid == "cowrie.login.failed",
                Event.ts >= window_start
            )
        ).order_by(Event.ts.desc())
        
        result = await db.execute(query)
        return result.scalars().all()
    
    async def _detect_password_spray(self, events: List[Event]) -> Optional[AttackPattern]:
        """Detect password spray attacks"""
        if len(events) < self.thresholds['password_spray']['min_attempts']:
            return None
        
        # Extract usernames and passwords
        username_passwords = defaultdict(set)
        for event in events:
            try:
                raw_data = event.raw if isinstance(event.raw, dict) else json.loads(event.raw) if event.raw else {}
                username = raw_data.get('username', 'unknown')
                password = raw_data.get('password', 'unknown')
                username_passwords[username].add(password)
            except (json.JSONDecodeError, AttributeError, TypeError):
                continue
        
        # Check for password spray indicators
        total_usernames = len(username_passwords)
        total_unique_passwords = len(set().union(*username_passwords.values())) if username_passwords else 0
        
        # Look for many passwords against few usernames
        spray_indicators = []
        for username, passwords in username_passwords.items():
            if len(passwords) >= self.thresholds['password_spray']['min_unique_passwords']:
                spray_indicators.append(f"{username}: {len(passwords)} passwords")
        
        if (total_unique_passwords >= self.thresholds['password_spray']['min_unique_passwords'] and
            total_usernames <= 5 and  # Few usernames
            spray_indicators):
            
            confidence = min(0.95, total_unique_passwords / 100.0 + 0.3)
            severity = "high" if total_unique_passwords > 30 else "medium"
            
            return AttackPattern(
                pattern_type="password_spray",
                src_ip=events[0].src_ip,
                severity=severity,
                confidence=confidence,
                indicators={
                    "unique_passwords": total_unique_passwords,
                    "usernames_targeted": total_usernames,
                    "spray_details": spray_indicators,
                    "total_attempts": len(events)
                },
                event_count=len(events),
                time_span=int((events[0].ts - events[-1].ts).total_seconds()) if len(events) > 1 else 0,
                description=f"Password spray: {total_unique_passwords} unique passwords against {total_usernames} usernames"
            )
        
        return None


class WebAttackDetector:
    """Detects web application attacks"""
    
    def __init__(self, window_seconds: int = 300, threshold: int = 3):
        self.window_seconds = window_seconds  # 5 minutes
        self.threshold = threshold  # 3 attack indicators
    
    async def evaluate(self, db: AsyncSession, src_ip: str) -> Optional[Dict[str, Any]]:
        """
        Evaluate if the src_ip has triggered web attack detection
        """
        # Check if there's already an open incident for this IP
        existing_incident = await self._get_open_incident(db, src_ip)
        if existing_incident:
            logger.debug(f"Open incident already exists for {src_ip}: {existing_incident.id}")
            return None
        
        # Count web attack indicators in the window
        window_start = datetime.now(timezone.utc) - timedelta(seconds=self.window_seconds)
        
        # Look for events with attack indicators (from webhoneypot)
        query = select(Event).where(
            and_(
                Event.src_ip == src_ip,
                Event.ts >= window_start,
                Event.raw.contains("attack_indicators")  # Events with attack indicators
            )
        )
        
        result = await db.execute(query)
        events = result.scalars().all()
        
        # Analyze attack indicators
        attack_indicators = []
        sql_injection_count = 0
        admin_scan_count = 0
        
        for event in events:
            try:
                raw_data = event.raw if isinstance(event.raw, dict) else json.loads(event.raw) if event.raw else {}
                indicators = raw_data.get('attack_indicators', [])
                
                for indicator in indicators:
                    attack_indicators.append(indicator)
                    if 'sql' in indicator.lower() or 'injection' in indicator.lower():
                        sql_injection_count += 1
                    elif 'admin' in indicator.lower() or 'scan' in indicator.lower():
                        admin_scan_count += 1
                        
            except (json.JSONDecodeError, AttributeError, TypeError):
                continue
        
        total_indicators = len(attack_indicators)
        
        logger.info(f"Web attack check: {src_ip} has {total_indicators} attack indicators in last {self.window_seconds}s (threshold: {self.threshold})")
        
        if total_indicators >= self.threshold:
            # Determine attack type and severity
            attack_types = []
            if sql_injection_count > 0:
                attack_types.append(f"SQL injection ({sql_injection_count})")
            if admin_scan_count > 0:
                attack_types.append(f"Admin scan ({admin_scan_count})")
            
            attack_description = ", ".join(attack_types) if attack_types else "Web application attacks"
            severity = "high" if sql_injection_count > 0 else "medium"
            
            # Create incident
            incident_data = {
                "src_ip": src_ip,
                "reason": f"Web attack detected: {attack_description} ({total_indicators} indicators in {self.window_seconds//60}min)",
                "status": "open",
                "auto_contained": False
            }
            
            incident = Incident(**incident_data)
            db.add(incident)
            await db.flush()  # Get the ID without committing
            
            logger.warning(f"NEW WEB INCIDENT #{incident.id}: {attack_description} from {src_ip}")
            
            return {
                "incident_id": incident.id,
                "src_ip": src_ip,
                "attack_indicators": total_indicators,
                "threshold": self.threshold,
                "window_seconds": self.window_seconds,
                "attack_types": attack_types
            }
        
        return None
    
    async def _get_open_incident(self, db: AsyncSession, src_ip: str) -> Optional[Incident]:
        """Check if there's an open incident for the given IP"""
        query = select(Incident).where(
            and_(
                Incident.src_ip == src_ip,
                Incident.status == "open"
            )
        ).order_by(Incident.created_at.desc())
        
        result = await db.execute(query)
        return result.scalars().first()


# Import adaptive detection components
from .adaptive_detection import behavioral_analyzer
from .baseline_engine import baseline_engine
from .ml_engine import ml_detector

# Global detector instances
ssh_bruteforce_detector = SlidingWindowDetector()
web_attack_detector = WebAttackDetector()
correlation_engine = AdvancedCorrelationEngine()


class AdaptiveDetectionEngine:
    """Orchestrates all detection methods with intelligent scoring"""
    
    def __init__(self):
        # Existing detectors
        self.ssh_detector = ssh_bruteforce_detector
        self.web_detector = web_attack_detector
        self.correlation_engine = correlation_engine
        
        # New intelligent detectors
        self.behavior_analyzer = behavioral_analyzer
        self.ml_detector = ml_detector
        self.baseline_engine = baseline_engine
        
        # Scoring weights (tunable)
        self.weights = {
            'rule_based': 0.4,      # Traditional rules
            'behavioral': 0.3,       # Pattern analysis
            'ml_anomaly': 0.2,      # ML detection
            'statistical': 0.1       # Baseline deviation
        }
        
    async def comprehensive_analysis(self, db: AsyncSession, src_ip: str) -> Optional[Dict]:
        """Run comprehensive analysis using all detection methods"""
        
        detection_results = {}
        
        # Layer 1: Traditional rule-based detection (fast)
        ssh_result = await self.ssh_detector.evaluate(db, src_ip)
        web_result = await self.web_detector.evaluate(db, src_ip)
        
        if ssh_result or web_result:
            detection_results['rule_based'] = {
                'score': 1.0,
                'ssh_detection': ssh_result,
                'web_detection': web_result,
                'confidence': 0.9
            }
        
        # Layer 2: Behavioral pattern analysis
        behavior_result = await self.behavior_analyzer.analyze_ip_behavior(db, src_ip)
        if behavior_result:
            detection_results['behavioral'] = {
                'score': behavior_result.threat_score,
                'patterns': behavior_result.indicators,
                'confidence': behavior_result.confidence,
                'description': behavior_result.description
            }
        
        # Layer 3: ML anomaly detection  
        recent_events = await self._get_recent_events(db, src_ip, 60)
        if len(recent_events) >= 5:
            ml_score = await self.ml_detector.calculate_anomaly_score(src_ip, recent_events)
            if ml_score > 0.3:  # Only include significant ML scores
                detection_results['ml_anomaly'] = {
                    'score': ml_score,
                    'confidence': getattr(self.ml_detector, 'last_confidence', 0.5),
                    'model_status': self.ml_detector.get_model_status()
                }
        
        # Layer 4: Statistical baseline deviation
        baseline_deviation = await self.baseline_engine.calculate_deviation(db, src_ip)
        if baseline_deviation > 0.3:
            detection_results['statistical'] = {
                'score': baseline_deviation,
                'deviated_metrics': self.baseline_engine.last_deviations,
                'confidence': 0.7
            }
        
        # Combine all scores intelligently
        if detection_results:
            composite_incident = await self._create_composite_incident(db, src_ip, detection_results)
            return composite_incident
            
        return None
    
    async def _get_recent_events(self, db: AsyncSession, src_ip: str, minutes: int) -> List[Event]:
        """Get recent events for an IP"""
        window_start = datetime.now(timezone.utc) - timedelta(minutes=minutes)
        
        query = select(Event).where(
            and_(
                Event.src_ip == src_ip,
                Event.ts >= window_start
            )
        ).order_by(Event.ts.desc())
        
        result = await db.execute(query)
        return result.scalars().all()
    
    async def _create_composite_incident(self, db: AsyncSession, src_ip: str, detection_results: Dict) -> Dict:
        """Create composite incident from multiple detection layers"""
        
        # Calculate composite score
        composite_score = 0.0
        total_weight = 0.0
        detection_methods = []
        
        for method, result in detection_results.items():
            weight = self.weights.get(method, 0.1)
            score = result['score']
            confidence = result.get('confidence', 0.5)
            
            # Weight by confidence
            adjusted_weight = weight * confidence
            composite_score += score * adjusted_weight
            total_weight += adjusted_weight
            
            detection_methods.append(f"{method}({score:.2f})")
        
        if total_weight > 0:
            composite_score = composite_score / total_weight
        
        # Determine severity based on composite score and detection layers
        num_layers = len(detection_results)
        if composite_score > 0.8 or num_layers >= 3:
            severity = "high"
        elif composite_score > 0.6 or num_layers >= 2:
            severity = "medium"
        else:
            severity = "low"
        
        # Create comprehensive reason
        reason_parts = []
        
        if 'rule_based' in detection_results:
            rb_result = detection_results['rule_based']
            if rb_result.get('ssh_detection'):
                reason_parts.append("SSH brute-force")
            if rb_result.get('web_detection'):
                reason_parts.append("Web attack")
        
        if 'behavioral' in detection_results:
            behavior_desc = detection_results['behavioral'].get('description', 'Behavioral anomaly')
            reason_parts.append(behavior_desc)
        
        if 'ml_anomaly' in detection_results:
            ml_score = detection_results['ml_anomaly']['score']
            reason_parts.append(f"ML anomaly (score: {ml_score:.2f})")
        
        if 'statistical' in detection_results:
            stat_score = detection_results['statistical']['score']
            reason_parts.append(f"Statistical deviation (score: {stat_score:.2f})")
        
        reason = f"Adaptive detection: {'; '.join(reason_parts)} | Composite score: {composite_score:.2f}"
        
        # Check if there's already an open incident for this IP
        existing_incident = await self.ssh_detector._get_open_incident(db, src_ip)
        if existing_incident:
            logger.debug(f"Open incident already exists for {src_ip}: {existing_incident.id}")
            return {"incident_id": existing_incident.id}
        
        # Create new incident
        incident_data = {
            "src_ip": src_ip,
            "reason": reason,
            "status": "open",
            "auto_contained": False,
            "escalation_level": severity,
            "risk_score": composite_score,
            "threat_category": "adaptive_detection",
            "containment_confidence": composite_score,
            "containment_method": "ai_agent",
            "ensemble_scores": detection_results
        }
        
        incident = Incident(**incident_data)
        db.add(incident)
        await db.flush()  # Get the ID without committing
        
        logger.warning(f"NEW ADAPTIVE INCIDENT #{incident.id}: {reason}")
        
        return {
            "incident_id": incident.id,
            "src_ip": src_ip,
            "composite_score": composite_score,
            "detection_layers": list(detection_results.keys()),
            "severity": severity
        }


# Global adaptive engine instance  
adaptive_engine = AdaptiveDetectionEngine()


async def run_detection(db: AsyncSession, src_ip: str) -> Optional[int]:
    """
    Enhanced detection using adaptive engine with fallback to legacy detection
    
    Returns:
        Incident ID if a new incident was created, None otherwise
    """
    try:
        # Use new adaptive detection engine
        result = await adaptive_engine.comprehensive_analysis(db, src_ip)
        if result:
            return result['incident_id']
            
        # Fallback to legacy detection for compatibility
        return await _legacy_detection(db, src_ip)
        
    except Exception as e:
        logger.error(f"Adaptive detection error for {src_ip}: {e}")
        # Fallback to legacy detection on error
        return await _legacy_detection(db, src_ip)


async def _legacy_detection(db: AsyncSession, src_ip: str) -> Optional[int]:
    """
    Legacy detection system (original implementation)
    """
    try:
        # First run traditional brute-force detection
        detection_result = await ssh_bruteforce_detector.evaluate(db, src_ip)
        if detection_result:
            return detection_result["incident_id"]
        
        # Run web attack detection
        web_detection_result = await web_attack_detector.evaluate(db, src_ip)
        if web_detection_result:
            return web_detection_result["incident_id"]
        
        # If no specific attacks detected, run correlation analysis
        patterns = await correlation_engine.analyze_ip(db, src_ip)
        if patterns:
            # Create incident for the most severe pattern
            highest_severity_pattern = max(patterns, key=lambda p: {'low': 1, 'medium': 2, 'high': 3}[p.severity])
            incident_id = await _create_correlation_incident(db, highest_severity_pattern, patterns)
            return incident_id
            
    except Exception as e:
        logger.error(f"Legacy detection error for {src_ip}: {e}")
    
    return None


async def _create_correlation_incident(db: AsyncSession, primary_pattern: AttackPattern, all_patterns: List[AttackPattern]) -> int:
    """Create an incident based on correlation analysis"""
    
    # Check if there's already an open incident for this IP
    existing_incident = await ssh_bruteforce_detector._get_open_incident(db, primary_pattern.src_ip)
    if existing_incident:
        logger.debug(f"Open incident already exists for {primary_pattern.src_ip}: {existing_incident.id}")
        return existing_incident.id
    
    # Create comprehensive incident description
    pattern_descriptions = [p.description for p in all_patterns]
    reason = f"Multi-chain attack detected: {'; '.join(pattern_descriptions)}"
    
    incident_data = {
        "src_ip": primary_pattern.src_ip,
        "reason": reason,
        "status": "open",
        "auto_contained": False
    }
    
    incident = Incident(**incident_data)
    db.add(incident)
    await db.flush()  # Get the ID without committing
    
    logger.warning(f"NEW CORRELATION INCIDENT #{incident.id}: {primary_pattern.pattern_type} from {primary_pattern.src_ip}")
    
    return incident.id

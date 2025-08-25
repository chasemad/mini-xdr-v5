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


# Global detector instances
ssh_bruteforce_detector = SlidingWindowDetector()
correlation_engine = AdvancedCorrelationEngine()


async def run_detection(db: AsyncSession, src_ip: str) -> Optional[int]:
    """
    Run all detectors for the given source IP
    
    Returns:
        Incident ID if a new incident was created, None otherwise
    """
    try:
        # First run traditional brute-force detection
        detection_result = await ssh_bruteforce_detector.evaluate(db, src_ip)
        if detection_result:
            return detection_result["incident_id"]
        
        # If no traditional brute-force detected, run correlation analysis
        patterns = await correlation_engine.analyze_ip(db, src_ip)
        if patterns:
            # Create incident for the most severe pattern
            highest_severity_pattern = max(patterns, key=lambda p: {'low': 1, 'medium': 2, 'high': 3}[p.severity])
            incident_id = await _create_correlation_incident(db, highest_severity_pattern, patterns)
            return incident_id
            
    except Exception as e:
        logger.error(f"Detection error for {src_ip}: {e}")
    
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

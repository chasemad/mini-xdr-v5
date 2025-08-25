from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, List
from sqlalchemy import select, and_, func
from sqlalchemy.ext.asyncio import AsyncSession
from .models import Event, Incident
from .config import settings
import logging

logger = logging.getLogger(__name__)


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


# Global detector instance
ssh_bruteforce_detector = SlidingWindowDetector()


async def run_detection(db: AsyncSession, src_ip: str) -> Optional[int]:
    """
    Run all detectors for the given source IP
    
    Returns:
        Incident ID if a new incident was created, None otherwise
    """
    try:
        detection_result = await ssh_bruteforce_detector.evaluate(db, src_ip)
        if detection_result:
            return detection_result["incident_id"]
    except Exception as e:
        logger.error(f"Detection error for {src_ip}: {e}")
    
    return None

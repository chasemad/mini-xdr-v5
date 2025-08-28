"""
Multi-Source Log Ingestion with Agent Support
Handles ingestion from multiple log sources with validation and enrichment
"""
import asyncio
import json
import hmac
import hashlib
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from .models import Event, LogSource
from .ml_engine import EnsembleMLDetector
from .external_intel import ThreatIntelligence
from .config import settings

logger = logging.getLogger(__name__)


class MultiSourceIngestor:
    """Enhanced multi-source log ingestion with agent validation"""
    
    def __init__(self, ml_detector=None, threat_intel=None):
        self.ml_detector = ml_detector or EnsembleMLDetector()
        self.threat_intel = threat_intel or ThreatIntelligence()
        self.logger = logging.getLogger(__name__)
        
        # Parser mapping for different source types
        self.parsers = {
            'cowrie': self._parse_cowrie_event,
            'suricata': self._parse_suricata_event,
            'osquery': self._parse_osquery_event,
            'syslog': self._parse_syslog_event,
            'zeek': self._parse_zeek_event,
            'custom': self._parse_custom_event
        }
        
        # Validation settings
        self.validate_signatures = True
        self.enrich_with_intel = True
        self.calculate_ml_scores = True
    
    async def ingest_events(
        self, 
        source_type: str, 
        hostname: str, 
        events: List[Dict[str, Any]],
        db: AsyncSession,
        api_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Ingest events from multiple sources with validation and enrichment
        
        Args:
            source_type: Type of log source (cowrie, suricata, etc.)
            hostname: Source hostname
            events: List of event dictionaries
            db: Database session
            api_key: API key for validation
            
        Returns:
            Ingestion result summary
        """
        result = {
            "source_type": source_type,
            "hostname": hostname,
            "total_events": len(events),
            "processed": 0,
            "failed": 0,
            "errors": []
        }
        
        # Get or create log source
        log_source = await self._get_or_create_log_source(db, source_type, hostname)
        
        # Process events
        processed_events = []
        
        for event_data in events:
            try:
                # 1. Validate event signature if present
                if self.validate_signatures and api_key:
                    if not self._validate_event_signature(event_data, api_key):
                        result["failed"] += 1
                        result["errors"].append(f"Invalid signature for event from {hostname}")
                        continue
                
                # 2. Parse event based on source type
                parsed_event = await self._parse_event(source_type, event_data, hostname)
                if not parsed_event:
                    result["failed"] += 1
                    result["errors"].append(f"Failed to parse {source_type} event")
                    continue
                
                # 3. Enrich with threat intelligence
                if self.enrich_with_intel and parsed_event.get('src_ip'):
                    try:
                        intel_result = await self.threat_intel.lookup_ip(parsed_event['src_ip'])
                        parsed_event['threat_intel'] = {
                            'risk_score': intel_result.risk_score,
                            'category': intel_result.category,
                            'is_malicious': intel_result.is_malicious
                        }
                    except Exception as e:
                        self.logger.warning(f"Threat intel enrichment failed: {e}")
                
                # 4. Create Event object
                event_obj = Event(
                    src_ip=parsed_event.get('src_ip'),
                    dst_ip=parsed_event.get('dst_ip'),
                    dst_port=parsed_event.get('dst_port'),
                    eventid=parsed_event.get('eventid', 'unknown'),
                    message=parsed_event.get('message'),
                    raw=event_data,
                    source_type=source_type,
                    hostname=hostname,
                    signature=event_data.get('signature'),
                    agent_timestamp=event_data.get('agent_timestamp')
                )
                
                # 5. Calculate ML anomaly score (async)
                if self.calculate_ml_scores and event_obj.src_ip:
                    try:
                        # Get recent events for this IP to calculate score
                        recent_events = await self._get_recent_events_for_ip(
                            db, event_obj.src_ip, seconds=3600
                        )
                        anomaly_score = await self.ml_detector.calculate_anomaly_score(
                            event_obj.src_ip, recent_events
                        )
                        event_obj.anomaly_score = anomaly_score
                    except Exception as e:
                        self.logger.warning(f"ML scoring failed: {e}")
                
                processed_events.append(event_obj)
                result["processed"] += 1
                
            except Exception as e:
                result["failed"] += 1
                result["errors"].append(f"Event processing error: {str(e)}")
                self.logger.error(f"Failed to process event: {e}")
        
        # 6. Bulk insert events
        if processed_events:
            try:
                db.add_all(processed_events)
                await db.flush()  # Get IDs without committing
                
                # Update log source statistics
                log_source.events_processed += len(processed_events)
                log_source.last_event_ts = datetime.now(timezone.utc)
                
                await db.commit()
                
                self.logger.info(
                    f"Ingested {len(processed_events)} events from {source_type}:{hostname}"
                )
                
            except Exception as e:
                await db.rollback()
                result["failed"] += len(processed_events)
                result["processed"] = 0
                result["errors"].append(f"Database insertion failed: {str(e)}")
                self.logger.error(f"Failed to insert events: {e}")
        
        return result
    
    def _validate_event_signature(self, event_data: Dict[str, Any], api_key: str) -> bool:
        """Validate event signature for integrity"""
        if 'signature' not in event_data:
            return True  # No signature to validate
        
        try:
            # Extract signature and remove it from data for validation
            provided_signature = event_data.pop('signature')
            
            # Calculate expected signature
            event_json = json.dumps(event_data, sort_keys=True)
            expected_signature = hmac.new(
                api_key.encode(), 
                event_json.encode(), 
                hashlib.sha256
            ).hexdigest()
            
            # Restore signature to event data
            event_data['signature'] = provided_signature
            
            return hmac.compare_digest(provided_signature, expected_signature)
            
        except Exception as e:
            self.logger.error(f"Signature validation failed: {e}")
            return False
    
    async def _parse_event(
        self, 
        source_type: str, 
        event_data: Dict[str, Any], 
        hostname: str
    ) -> Optional[Dict[str, Any]]:
        """Parse event based on source type"""
        parser = self.parsers.get(source_type, self._parse_custom_event)
        
        try:
            return await parser(event_data, hostname)
        except Exception as e:
            self.logger.error(f"Failed to parse {source_type} event: {e}")
            return None
    
    async def _parse_cowrie_event(self, event_data: Dict[str, Any], hostname: str) -> Dict[str, Any]:
        """Parse Cowrie honeypot events"""
        return {
            'src_ip': event_data.get('src_ip') or event_data.get('srcip') or event_data.get('peer_ip'),
            'dst_ip': event_data.get('dst_ip') or event_data.get('dstip'),
            'dst_port': event_data.get('dst_port') or event_data.get('dstport'),
            'eventid': event_data.get('eventid', 'cowrie.unknown'),
            'message': event_data.get('message'),
            'session': event_data.get('session'),
            'username': event_data.get('username'),
            'password': event_data.get('password'),
            'timestamp': event_data.get('timestamp')
        }
    
    async def _parse_suricata_event(self, event_data: Dict[str, Any], hostname: str) -> Dict[str, Any]:
        """Parse Suricata IDS events"""
        event_type = event_data.get('event_type', 'unknown')
        
        # Extract network info
        src_ip = None
        dst_ip = None
        dst_port = None
        
        if 'src_ip' in event_data:
            src_ip = event_data['src_ip']
        if 'dest_ip' in event_data:
            dst_ip = event_data['dest_ip']
        if 'dest_port' in event_data:
            dst_port = event_data['dest_port']
        
        # Handle alert events
        message = ""
        if event_type == 'alert' and 'alert' in event_data:
            alert = event_data['alert']
            message = f"{alert.get('signature', 'Unknown')} - {alert.get('category', '')}"
        
        return {
            'src_ip': src_ip,
            'dst_ip': dst_ip,
            'dst_port': dst_port,
            'eventid': f'suricata.{event_type}',
            'message': message,
            'severity': event_data.get('alert', {}).get('severity'),
            'signature_id': event_data.get('alert', {}).get('signature_id'),
            'timestamp': event_data.get('timestamp')
        }
    
    async def _parse_osquery_event(self, event_data: Dict[str, Any], hostname: str) -> Dict[str, Any]:
        """Parse OSQuery events"""
        # OSQuery events are typically host-based, so src_ip might be the host IP
        return {
            'src_ip': event_data.get('host_ip') or event_data.get('local_address'),
            'dst_ip': event_data.get('remote_address'),
            'dst_port': event_data.get('remote_port'),
            'eventid': f"osquery.{event_data.get('name', 'unknown')}",
            'message': event_data.get('message') or f"OSQuery: {event_data.get('name', 'event')}",
            'action': event_data.get('action'),
            'columns': event_data.get('columns', {}),
            'timestamp': event_data.get('unixTime') or event_data.get('timestamp')
        }
    
    async def _parse_syslog_event(self, event_data: Dict[str, Any], hostname: str) -> Dict[str, Any]:
        """Parse generic syslog events"""
        message = event_data.get('message', '')
        
        # Try to extract IP from message
        import re
        ip_pattern = r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
        ips = re.findall(ip_pattern, message)
        src_ip = ips[0] if ips else None
        
        return {
            'src_ip': src_ip,
            'dst_ip': None,
            'dst_port': None,
            'eventid': f"syslog.{event_data.get('facility', 'unknown')}",
            'message': message,
            'severity': event_data.get('severity'),
            'facility': event_data.get('facility'),
            'timestamp': event_data.get('timestamp')
        }
    
    async def _parse_zeek_event(self, event_data: Dict[str, Any], hostname: str) -> Dict[str, Any]:
        """Parse Zeek/Bro network security monitor events"""
        return {
            'src_ip': event_data.get('id.orig_h') or event_data.get('src_ip'),
            'dst_ip': event_data.get('id.resp_h') or event_data.get('dst_ip'),
            'dst_port': event_data.get('id.resp_p') or event_data.get('dst_port'),
            'eventid': f"zeek.{event_data.get('_path', 'unknown')}",
            'message': f"Zeek {event_data.get('_path', 'event')} - {event_data.get('note', '')}",
            'connection_state': event_data.get('conn_state'),
            'service': event_data.get('service'),
            'timestamp': event_data.get('ts')
        }
    
    async def _parse_custom_event(self, event_data: Dict[str, Any], hostname: str) -> Dict[str, Any]:
        """Parse custom/unknown event formats"""
        # Best-effort parsing for unknown formats
        return {
            'src_ip': event_data.get('src_ip') or event_data.get('source_ip') or event_data.get('ip'),
            'dst_ip': event_data.get('dst_ip') or event_data.get('dest_ip'),
            'dst_port': event_data.get('dst_port') or event_data.get('port'),
            'eventid': event_data.get('eventid') or event_data.get('event_type') or 'custom.unknown',
            'message': event_data.get('message') or event_data.get('description') or str(event_data),
            'timestamp': event_data.get('timestamp') or event_data.get('@timestamp')
        }
    
    async def _get_or_create_log_source(
        self, 
        db: AsyncSession, 
        source_type: str, 
        hostname: str
    ) -> LogSource:
        """Get existing log source or create new one"""
        query = select(LogSource).where(
            LogSource.source_type == source_type,
            LogSource.hostname == hostname
        )
        
        result = await db.execute(query)
        log_source = result.scalars().first()
        
        if not log_source:
            log_source = LogSource(
                source_type=source_type,
                hostname=hostname,
                status="active"
            )
            db.add(log_source)
            await db.flush()
            
            self.logger.info(f"Created new log source: {source_type}:{hostname}")
        
        return log_source
    
    async def _get_recent_events_for_ip(
        self, 
        db: AsyncSession, 
        src_ip: str, 
        seconds: int = 3600
    ) -> List[Event]:
        """Get recent events for ML scoring"""
        from datetime import timedelta
        
        cutoff_time = datetime.now(timezone.utc) - timedelta(seconds=seconds)
        query = select(Event).where(
            Event.src_ip == src_ip,
            Event.ts >= cutoff_time
        ).order_by(Event.ts.desc()).limit(100)
        
        result = await db.execute(query)
        return result.scalars().all()
    
    async def get_source_statistics(self, db: AsyncSession) -> List[Dict[str, Any]]:
        """Get statistics for all log sources"""
        query = select(LogSource)
        result = await db.execute(query)
        sources = result.scalars().all()
        
        stats = []
        for source in sources:
            stats.append({
                'id': source.id,
                'source_type': source.source_type,
                'hostname': source.hostname,
                'status': source.status,
                'events_processed': source.events_processed,
                'events_failed': source.events_failed,
                'last_event_ts': source.last_event_ts.isoformat() if source.last_event_ts else None,
                'agent_version': source.agent_version
            })
        
        return stats


# Global multi-source ingestor instance
multi_ingestor = MultiSourceIngestor()

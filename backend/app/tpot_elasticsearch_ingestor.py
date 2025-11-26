"""
T-Pot Elasticsearch Event Ingestor
Pulls events from T-Pot Elasticsearch and ingests them into XDR
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Set

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from .config import settings
from .db import AsyncSessionLocal
from .intelligent_detection import IntelligentDetectionEngine
from .models import Event, Incident
from .multi_ingestion import multi_ingestor
from .tpot_connector import get_tpot_connector
from .triager import generate_default_triage, run_triage
from .trigger_evaluator import trigger_evaluator

logger = logging.getLogger(__name__)

# Global detection engine instance
_detection_engine = None


def get_detection_engine() -> IntelligentDetectionEngine:
    """Get or create the intelligent detection engine"""
    global _detection_engine
    if _detection_engine is None:
        _detection_engine = IntelligentDetectionEngine()
    return _detection_engine


class TPotElasticsearchIngestor:
    """Ingest T-Pot events from Elasticsearch into XDR database"""

    def __init__(self):
        self.connector = get_tpot_connector()
        self.is_running = False
        self.task = None
        self.last_ingested_timestamps = {}
        self.poll_interval = 10  # seconds

    async def start(self):
        """Start continuous ingestion from Elasticsearch"""
        if self.is_running:
            logger.warning("Elasticsearch ingestor already running")
            return

        self.is_running = True
        self.task = asyncio.create_task(self._ingestion_loop())
        logger.info("✅ T-Pot Elasticsearch ingestor started")

    async def stop(self):
        """Stop the ingestion loop"""
        self.is_running = False
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        logger.info("T-Pot Elasticsearch ingestor stopped")

    async def _ingestion_loop(self):
        """Continuous loop to pull and ingest events"""
        while self.is_running:
            try:
                if not self.connector.is_connected:
                    logger.debug("Waiting for T-Pot connection...")
                    await asyncio.sleep(self.poll_interval)
                    continue

                # Pull events from Elasticsearch
                await self._pull_and_ingest_events()

                # Wait before next poll
                await asyncio.sleep(self.poll_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in ingestion loop: {e}")
                await asyncio.sleep(self.poll_interval)

    async def _pull_and_ingest_events(self):
        """Pull recent events from Elasticsearch and ingest them"""
        try:
            # Query for events from last 1 minute
            query = {
                "query": {"range": {"@timestamp": {"gte": "now-1m"}}},
                "size": 100,
                "sort": [{"@timestamp": "asc"}],
            }

            result = await self.connector.query_elasticsearch(query)

            if not result["success"]:
                logger.debug(f"Elasticsearch query failed: {result.get('error')}")
                return

            hits = result["data"].get("hits", {}).get("hits", [])

            if not hits:
                logger.debug("No new events from Elasticsearch")
                return

            # Group events by honeypot type
            events_by_type = {}

            for hit in hits:
                source = hit["_source"]
                event_type = source.get("type", "unknown").lower()

                # Map T-Pot event types to our honeypot types
                honeypot_type = self._map_event_type(event_type, source)

                if honeypot_type not in events_by_type:
                    events_by_type[honeypot_type] = []

                # Convert to XDR event format
                xdr_event = self._convert_to_xdr_event(source, honeypot_type)
                if xdr_event:
                    events_by_type[honeypot_type].append(xdr_event)

            # Ingest events by type and collect unique IPs
            total_ingested = 0
            unique_ips: Set[str] = set()

            async with AsyncSessionLocal() as db:
                for honeypot_type, events in events_by_type.items():
                    if events:
                        # Collect unique source IPs for detection
                        for event in events:
                            if event.get("src_ip"):
                                unique_ips.add(event["src_ip"])

                        result = await multi_ingestor.ingest_events(
                            source_type=honeypot_type,
                            hostname=self.connector.host,
                            events=events,
                            db=db,
                        )
                        total_ingested += result.get("processed", 0)

            if total_ingested > 0:
                logger.info(
                    f"✅ Ingested {total_ingested} events from T-Pot Elasticsearch"
                )

                # Run intelligent detection for all unique source IPs
                if unique_ips:
                    await self._run_detection_for_ips(unique_ips)

        except Exception as e:
            logger.error(f"Failed to pull and ingest events: {e}")

    async def _run_detection_for_ips(self, unique_ips: Set[str]):
        """Run intelligent detection for all unique source IPs and trigger AI agents"""
        try:
            detection_engine = get_detection_engine()
            incidents_created = 0

            async with AsyncSessionLocal() as db:
                for src_ip in unique_ips:
                    try:
                        result = await detection_engine.analyze_and_create_incidents(
                            db=db, src_ip=src_ip
                        )
                        if result.get("incident_created"):
                            incidents_created += 1
                            incident_id = result.get("incident_id")
                            logger.info(
                                f"🚨 Incident created for {src_ip}: "
                                f"{result.get('classification', {}).get('threat_type', 'Unknown')}"
                            )

                            # Trigger AI agents and workflows for the new incident
                            if incident_id:
                                await self._trigger_agents_for_incident(
                                    db, incident_id, src_ip
                                )

                    except Exception as e:
                        logger.error(f"Detection failed for {src_ip}: {e}")

            if incidents_created > 0:
                logger.info(
                    f"🔴 Created {incidents_created} new incidents from T-Pot events"
                )

        except Exception as e:
            logger.error(f"Failed to run detection: {e}")

    async def _trigger_agents_for_incident(
        self, db: AsyncSession, incident_id: int, src_ip: str
    ):
        """Trigger AI agents and workflows for a newly created incident"""
        try:
            # Fetch the incident
            incident = (
                (await db.execute(select(Incident).where(Incident.id == incident_id)))
                .scalars()
                .first()
            )

            if not incident:
                logger.warning(f"Incident {incident_id} not found for agent triggering")
                return

            # Get recent events for context
            recent_events = await self._get_recent_events_for_ip(db, src_ip)

            logger.info(
                f"🤖 Triggering AI agents for incident #{incident_id} ({src_ip})"
            )

            # Step 1: Evaluate workflow triggers
            try:
                executed_workflows = (
                    await trigger_evaluator.evaluate_triggers_for_incident(
                        db, incident, recent_events
                    )
                )
                if executed_workflows:
                    logger.info(
                        f"✅ Executed {len(executed_workflows)} workflows for incident #{incident_id}: {executed_workflows}"
                    )
            except Exception as e:
                logger.error(
                    f"Failed to evaluate triggers for incident #{incident_id}: {e}"
                )

            # Step 2: Run triage analysis
            try:
                triage_input = {
                    "id": incident.id,
                    "src_ip": incident.src_ip,
                    "reason": incident.reason,
                    "status": incident.status,
                }
                event_summaries = [
                    {
                        "ts": e.ts.isoformat() if e.ts else None,
                        "eventid": e.eventid,
                        "message": e.message,
                        "source_type": e.source_type,
                    }
                    for e in recent_events
                ]

                triage_note = run_triage(triage_input, event_summaries)
                incident.triage_note = triage_note
                logger.info(f"✅ Triage completed for incident #{incident_id}")
            except Exception as e:
                logger.error(f"Triage failed for incident #{incident_id}: {e}")
                incident.triage_note = generate_default_triage(
                    {"id": incident_id, "src_ip": src_ip}, len(recent_events)
                )

            # Step 3: Run AI agent containment if enabled
            auto_contain_enabled = getattr(settings, "auto_contain", False)
            if auto_contain_enabled:
                try:
                    from .agents.containment_agent import ContainmentAgent

                    containment_agent = ContainmentAgent()
                    response = await containment_agent.orchestrate_response(
                        incident, recent_events, db
                    )
                    logger.info(
                        f"✅ AI containment agent triggered for incident #{incident_id}: {response.get('action', 'completed')}"
                    )
                except Exception as e:
                    logger.error(
                        f"AI containment failed for incident #{incident_id}: {e}"
                    )
            else:
                logger.info(
                    f"⏭️ Auto-contain disabled, skipping containment agent for incident #{incident_id}"
                )

            # Commit changes
            await db.commit()

            logger.info(f"🎯 AI agent processing complete for incident #{incident_id}")

        except Exception as e:
            logger.error(f"Failed to trigger agents for incident {incident_id}: {e}")

    async def _get_recent_events_for_ip(
        self, db: AsyncSession, src_ip: str, seconds: int = 600
    ) -> List[Event]:
        """Get recent events for an IP address"""
        since = datetime.now(timezone.utc) - timedelta(seconds=seconds)
        query = (
            select(Event)
            .where(and_(Event.src_ip == src_ip, Event.ts >= since))
            .order_by(Event.ts.desc())
            .limit(200)
        )

        result = await db.execute(query)
        return result.scalars().all()

    def _map_event_type(self, event_type: str, source: Dict[str, Any]) -> str:
        """Map T-Pot event type to our honeypot type"""
        # T-Pot uses 'type' field for honeypot identification
        type_mapping = {
            "suricata": "suricata",
            "cowrie": "cowrie",
            "dionaea": "dionaea",
            "wordpot": "wordpot",
            "elasticpot": "elasticpot",
            "redishoneypot": "redishoneypot",
            "mailoney": "mailoney",
            "sentrypeer": "sentrypeer",
        }

        return type_mapping.get(event_type, "custom")

    def _convert_to_xdr_event(
        self, source: Dict[str, Any], honeypot_type: str
    ) -> Dict[str, Any]:
        """Convert Elasticsearch event to XDR event format"""
        try:
            # Extract common fields
            event = {
                "timestamp": source.get("@timestamp"),
                "src_ip": source.get("src_ip") or source.get("source_ip"),
                "dst_ip": source.get("dest_ip") or source.get("destination_ip"),
                "dst_port": source.get("dest_port") or source.get("destination_port"),
            }

            # Honeypot-specific parsing
            if honeypot_type == "cowrie":
                event["eventid"] = source.get("eventid", "cowrie.session")
                event["username"] = source.get("username")
                event["password"] = source.get("password")
                event["session"] = source.get("session")

            elif honeypot_type == "suricata":
                alert = source.get("alert", {})
                event["eventid"] = "suricata.alert"
                event["message"] = alert.get("signature", "Suricata alert")
                event["severity"] = alert.get("severity")
                event["category"] = alert.get("category")

            elif honeypot_type == "dionaea":
                event["eventid"] = source.get("eventid", "dionaea.connection")
                event["src_port"] = source.get("src_port")

            else:
                # Generic event
                event["eventid"] = f"{honeypot_type}.event"
                event["message"] = source.get("message", f"{honeypot_type} event")

            # Filter out events without source IP (internal T-Pot traffic)
            if not event.get("src_ip"):
                return None

            # Filter out non-attack traffic (optional - can be tuned)
            if self._is_regular_traffic(event, source):
                return None

            return event

        except Exception as e:
            logger.debug(f"Failed to convert event: {e}")
            return None

    def _is_regular_traffic(
        self, event: Dict[str, Any], source: Dict[str, Any]
    ) -> bool:
        """Determine if this is regular traffic vs attack traffic"""

        # Filter out internal T-Pot traffic
        src_ip = event.get("src_ip", "")
        if src_ip.startswith("172.16.110.") or src_ip.startswith("10."):
            return True

        # DNS queries to known resolvers are normal
        if event.get("dst_port") == 53 and event.get("dst_ip") in [
            "8.8.8.8",
            "8.8.4.4",
            "1.1.1.1",
        ]:
            return True

        # Suricata informational events (not alerts) are normal
        if source.get("event_type") == "flow" and not source.get("alert"):
            return True

        return False


# Global instance
_ingestor = None


def get_elasticsearch_ingestor() -> TPotElasticsearchIngestor:
    """Get or create the Elasticsearch ingestor"""
    global _ingestor
    if _ingestor is None:
        _ingestor = TPotElasticsearchIngestor()
    return _ingestor


async def start_elasticsearch_ingestion():
    """Start the Elasticsearch ingestion process"""
    ingestor = get_elasticsearch_ingestor()
    await ingestor.start()


async def stop_elasticsearch_ingestion():
    """Stop the Elasticsearch ingestion process"""
    ingestor = get_elasticsearch_ingestor()
    await ingestor.stop()

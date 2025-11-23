"""
T-Pot Elasticsearch Event Ingestor
Pulls events from T-Pot Elasticsearch and ingests them into XDR
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict

from sqlalchemy.ext.asyncio import AsyncSession

from .db import AsyncSessionLocal
from .multi_ingestion import multi_ingestor
from .tpot_connector import get_tpot_connector

logger = logging.getLogger(__name__)


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

            # Ingest events by type
            total_ingested = 0
            async with AsyncSessionLocal() as db:
                for honeypot_type, events in events_by_type.items():
                    if events:
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

        except Exception as e:
            logger.error(f"Failed to pull and ingest events: {e}")

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

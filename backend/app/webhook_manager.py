"""
Webhook Management System for Mini-XDR
Handles webhook registration, delivery, and retry logic with HMAC security
"""

import os
import time
import hmac
import hashlib
import logging
import asyncio
import aiohttp
from typing import Optional, Dict, List, Any
from datetime import datetime
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


class WebhookManager:
    """Manages webhook subscriptions and deliveries with security and retry logic"""

    def __init__(self, signing_secret: Optional[str] = None):
        self.signing_secret = signing_secret or os.getenv('WEBHOOK_SIGNING_SECRET',
                                                          'mini-xdr-webhook-secret-2024')
        self.max_retries = 3
        self.retry_delays = [5, 15, 60]  # seconds

    def generate_signature(self, payload: bytes, timestamp: str) -> str:
        """Generate HMAC-SHA256 signature for webhook payload"""
        message = f"{timestamp}.{payload.decode()}"
        signature = hmac.new(
            self.signing_secret.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()
        return f"v1={signature}"

    async def deliver_webhook(
        self,
        url: str,
        event_type: str,
        payload: Dict[str, Any],
        webhook_id: Optional[int] = None,
        attempt: int = 1
    ) -> Dict[str, Any]:
        """
        Deliver webhook with HMAC signature and automatic retry logic

        Args:
            url: Webhook endpoint URL
            event_type: Type of event (incident.created, incident.contained, etc.)
            payload: Event payload data
            webhook_id: Database ID of webhook subscription
            attempt: Current delivery attempt number

        Returns:
            Delivery result with status and response info
        """
        timestamp = str(int(time.time()))
        payload_json = {
            "event_type": event_type,
            "timestamp": timestamp,
            "data": payload
        }

        payload_bytes = str(payload_json).encode()
        signature = self.generate_signature(payload_bytes, timestamp)

        headers = {
            "Content-Type": "application/json",
            "X-Mini-XDR-Event": event_type,
            "X-Mini-XDR-Signature": signature,
            "X-Mini-XDR-Timestamp": timestamp,
            "X-Mini-XDR-Delivery-Attempt": str(attempt),
            "User-Agent": "Mini-XDR-Webhook/1.0"
        }

        result = {
            "webhook_id": webhook_id,
            "url": url,
            "event_type": event_type,
            "attempt": attempt,
            "timestamp": datetime.utcnow().isoformat(),
            "success": False,
            "status_code": None,
            "response_time_ms": None,
            "error": None
        }

        try:
            start_time = time.time()

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=payload_json,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    response_time = (time.time() - start_time) * 1000

                    result["status_code"] = response.status
                    result["response_time_ms"] = round(response_time, 2)
                    result["success"] = 200 <= response.status < 300

                    if result["success"]:
                        logger.info(
                            f"✅ Webhook delivered successfully: {event_type} to {url} "
                            f"(attempt {attempt}, {response_time:.0f}ms)"
                        )
                    else:
                        result["error"] = f"HTTP {response.status}"
                        logger.warning(
                            f"⚠️ Webhook delivery failed: {event_type} to {url} "
                            f"(HTTP {response.status}, attempt {attempt})"
                        )

        except asyncio.TimeoutError:
            result["error"] = "Request timeout (10s)"
            logger.error(f"❌ Webhook timeout: {event_type} to {url} (attempt {attempt})")

        except aiohttp.ClientError as e:
            result["error"] = f"Connection error: {str(e)}"
            logger.error(f"❌ Webhook connection error: {url} - {e} (attempt {attempt})")

        except Exception as e:
            result["error"] = f"Unexpected error: {str(e)}"
            logger.error(f"❌ Webhook unexpected error: {url} - {e} (attempt {attempt})")

        # Retry logic for failed deliveries
        if not result["success"] and attempt < self.max_retries:
            delay = self.retry_delays[attempt - 1]
            logger.info(f"⏰ Retrying webhook delivery in {delay}s (attempt {attempt + 1}/{self.max_retries})")
            await asyncio.sleep(delay)
            return await self.deliver_webhook(url, event_type, payload, webhook_id, attempt + 1)

        return result

    async def trigger_webhooks(
        self,
        event_type: str,
        payload: Dict[str, Any],
        db: AsyncSession
    ) -> List[Dict[str, Any]]:
        """
        Trigger all registered webhooks for a specific event type

        Args:
            event_type: Event type to trigger webhooks for
            payload: Event data payload
            db: Database session

        Returns:
            List of delivery results
        """
        from .models import WebhookSubscription

        # Query active webhook subscriptions for this event type
        stmt = select(WebhookSubscription).where(
            WebhookSubscription.event_types.contains([event_type]),
            WebhookSubscription.is_active == True
        )
        result = await db.execute(stmt)
        webhooks = result.scalars().all()

        if not webhooks:
            logger.debug(f"No active webhooks found for event: {event_type}")
            return []

        logger.info(f"📤 Triggering {len(webhooks)} webhooks for event: {event_type}")

        # Deliver webhooks concurrently
        delivery_tasks = [
            self.deliver_webhook(
                webhook.url,
                event_type,
                payload,
                webhook.id
            )
            for webhook in webhooks
        ]

        results = await asyncio.gather(*delivery_tasks, return_exceptions=True)

        # Log delivery summary
        successful = sum(1 for r in results if isinstance(r, dict) and r.get("success"))
        logger.info(
            f"📊 Webhook delivery complete: {successful}/{len(webhooks)} successful "
            f"for event {event_type}"
        )

        return [r for r in results if isinstance(r, dict)]


# Global webhook manager instance
webhook_manager = WebhookManager()


async def trigger_incident_webhook(
    event_type: str,
    incident_data: Dict[str, Any],
    db: AsyncSession
):
    """
    Convenience function to trigger incident-related webhooks

    Supported event types:
    - incident.created: New incident detected
    - incident.contained: Incident contained automatically or manually
    - incident.escalated: Incident escalated to higher severity
    - response.executed: Response action executed
    - workflow.completed: Response workflow completed
    """
    return await webhook_manager.trigger_webhooks(event_type, incident_data, db)
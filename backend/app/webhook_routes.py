"""
Webhook Management API Routes for Mini-XDR
Provides endpoints for managing webhook subscriptions
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete
from typing import List, Optional
from pydantic import BaseModel, HttpUrl, Field
from datetime import datetime
import logging

from .database import get_db
from .webhook_manager import webhook_manager
from .security import verify_api_key

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/webhooks", tags=["webhooks"])


# Pydantic models for API
class WebhookCreate(BaseModel):
    """Request model for creating a webhook subscription"""
    url: HttpUrl = Field(..., description="Webhook endpoint URL")
    event_types: List[str] = Field(
        ...,
        description="List of event types to subscribe to",
        example=["incident.created", "incident.contained", "response.executed"]
    )
    name: Optional[str] = Field(None, description="Friendly name for this webhook")
    description: Optional[str] = Field(None, description="Description of webhook purpose")
    secret: Optional[str] = Field(None, description="Custom signing secret (optional)")


class WebhookUpdate(BaseModel):
    """Request model for updating a webhook subscription"""
    url: Optional[HttpUrl] = None
    event_types: Optional[List[str]] = None
    name: Optional[str] = None
    description: Optional[str] = None
    is_active: Optional[bool] = None


class WebhookResponse(BaseModel):
    """Response model for webhook subscription"""
    id: int
    url: str
    event_types: List[str]
    name: Optional[str]
    description: Optional[str]
    is_active: bool
    created_at: datetime
    last_triggered_at: Optional[datetime]
    delivery_success_count: int
    delivery_failure_count: int

    class Config:
        from_attributes = True


class WebhookDeliveryTest(BaseModel):
    """Request model for testing webhook delivery"""
    url: HttpUrl
    event_type: str = "test.ping"
    test_payload: Optional[dict] = None


@router.post("/subscriptions", response_model=WebhookResponse, status_code=status.HTTP_201_CREATED)
async def create_webhook_subscription(
    webhook: WebhookCreate,
    db: AsyncSession = Depends(get_db),
    _api_key: str = Depends(verify_api_key)
):
    """
    Create a new webhook subscription

    Subscribe to Mini-XDR events to receive real-time notifications:
    - **incident.created**: Triggered when a new incident is detected
    - **incident.contained**: Triggered when incident is contained
    - **incident.escalated**: Triggered when incident severity increases
    - **response.executed**: Triggered when a response action is executed
    - **workflow.completed**: Triggered when a response workflow completes
    """
    from .models import WebhookSubscription

    # Validate event types
    valid_events = [
        "incident.created", "incident.contained", "incident.escalated",
        "response.executed", "workflow.completed", "workflow.failed"
    ]

    invalid_events = [et for et in webhook.event_types if et not in valid_events]
    if invalid_events:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid event types: {invalid_events}. Valid events: {valid_events}"
        )

    # Create webhook subscription
    db_webhook = WebhookSubscription(
        url=str(webhook.url),
        event_types=webhook.event_types,
        name=webhook.name,
        description=webhook.description,
        signing_secret=webhook.secret or webhook_manager.signing_secret,
        is_active=True,
        delivery_success_count=0,
        delivery_failure_count=0
    )

    db.add(db_webhook)
    await db.commit()
    await db.refresh(db_webhook)

    logger.info(f"✅ Created webhook subscription #{db_webhook.id} for {webhook.url}")
    return db_webhook


@router.get("/subscriptions", response_model=List[WebhookResponse])
async def list_webhook_subscriptions(
    active_only: bool = False,
    db: AsyncSession = Depends(get_db),
    _api_key: str = Depends(verify_api_key)
):
    """List all webhook subscriptions"""
    from .models import WebhookSubscription

    stmt = select(WebhookSubscription)
    if active_only:
        stmt = stmt.where(WebhookSubscription.is_active == True)

    stmt = stmt.order_by(WebhookSubscription.created_at.desc())

    result = await db.execute(stmt)
    webhooks = result.scalars().all()

    return webhooks


@router.get("/subscriptions/{webhook_id}", response_model=WebhookResponse)
async def get_webhook_subscription(
    webhook_id: int,
    db: AsyncSession = Depends(get_db),
    _api_key: str = Depends(verify_api_key)
):
    """Get a specific webhook subscription by ID"""
    from .models import WebhookSubscription

    stmt = select(WebhookSubscription).where(WebhookSubscription.id == webhook_id)
    result = await db.execute(stmt)
    webhook = result.scalar_one_or_none()

    if not webhook:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Webhook subscription {webhook_id} not found"
        )

    return webhook


@router.patch("/subscriptions/{webhook_id}", response_model=WebhookResponse)
async def update_webhook_subscription(
    webhook_id: int,
    webhook_update: WebhookUpdate,
    db: AsyncSession = Depends(get_db),
    _api_key: str = Depends(verify_api_key)
):
    """Update a webhook subscription"""
    from .models import WebhookSubscription

    stmt = select(WebhookSubscription).where(WebhookSubscription.id == webhook_id)
    result = await db.execute(stmt)
    webhook = result.scalar_one_or_none()

    if not webhook:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Webhook subscription {webhook_id} not found"
        )

    # Update fields
    update_data = webhook_update.model_dump(exclude_unset=True)
    if "url" in update_data:
        update_data["url"] = str(update_data["url"])

    for field, value in update_data.items():
        setattr(webhook, field, value)

    await db.commit()
    await db.refresh(webhook)

    logger.info(f"✅ Updated webhook subscription #{webhook_id}")
    return webhook


@router.delete("/subscriptions/{webhook_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_webhook_subscription(
    webhook_id: int,
    db: AsyncSession = Depends(get_db),
    _api_key: str = Depends(verify_api_key)
):
    """Delete a webhook subscription"""
    from .models import WebhookSubscription

    stmt = delete(WebhookSubscription).where(WebhookSubscription.id == webhook_id)
    result = await db.execute(stmt)
    await db.commit()

    if result.rowcount == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Webhook subscription {webhook_id} not found"
        )

    logger.info(f"🗑️ Deleted webhook subscription #{webhook_id}")
    return None


@router.post("/test", status_code=status.HTTP_200_OK)
async def test_webhook_delivery(
    test: WebhookDeliveryTest,
    _api_key: str = Depends(verify_api_key)
):
    """
    Test webhook delivery to a URL

    Sends a test payload to verify webhook endpoint is working correctly
    """
    test_payload = test.test_payload or {
        "test": True,
        "message": "Mini-XDR webhook test",
        "timestamp": datetime.utcnow().isoformat()
    }

    result = await webhook_manager.deliver_webhook(
        url=str(test.url),
        event_type=test.event_type,
        payload=test_payload
    )

    if not result["success"]:
        raise HTTPException(
            status_code=status.HTTP_424_FAILED_DEPENDENCY,
            detail=f"Webhook test failed: {result.get('error', 'Unknown error')}"
        )

    return {
        "success": True,
        "message": "Webhook test successful",
        "result": result
    }


@router.get("/events")
async def list_available_webhook_events(_api_key: str = Depends(verify_api_key)):
    """List all available webhook event types"""
    return {
        "events": [
            {
                "type": "incident.created",
                "description": "Triggered when a new security incident is detected",
                "payload_example": {
                    "incident_id": 123,
                    "src_ip": "192.168.1.100",
                    "risk_score": 0.85,
                    "threat_category": "brute_force"
                }
            },
            {
                "type": "incident.contained",
                "description": "Triggered when an incident is successfully contained",
                "payload_example": {
                    "incident_id": 123,
                    "containment_method": "ip_block",
                    "contained_at": "2024-01-01T12:00:00Z"
                }
            },
            {
                "type": "incident.escalated",
                "description": "Triggered when incident severity increases",
                "payload_example": {
                    "incident_id": 123,
                    "old_severity": "medium",
                    "new_severity": "critical",
                    "reason": "successful_authentication_detected"
                }
            },
            {
                "type": "response.executed",
                "description": "Triggered when a response action is executed",
                "payload_example": {
                    "incident_id": 123,
                    "action_type": "block_ip",
                    "result": "success",
                    "executed_by": "auto_response_agent"
                }
            },
            {
                "type": "workflow.completed",
                "description": "Triggered when a response workflow completes",
                "payload_example": {
                    "workflow_id": "wf_123",
                    "incident_id": 123,
                    "status": "completed",
                    "actions_executed": 5,
                    "duration_seconds": 12.5
                }
            },
            {
                "type": "workflow.failed",
                "description": "Triggered when a response workflow fails",
                "payload_example": {
                    "workflow_id": "wf_123",
                    "incident_id": 123,
                    "status": "failed",
                    "error": "action_timeout",
                    "failed_at_step": 3
                }
            }
        ]
    }
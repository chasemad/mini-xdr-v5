"""
Agent API Routes - External agent communication endpoints

Handles agent enrollment, heartbeats, and telemetry submission.
"""

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from .agent_enrollment_service import AgentEnrollmentService
from .db import get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/agents", tags=["agents"])


# Request/Response models
class AgentEnrollRequest(BaseModel):
    """Request model for agent enrollment"""

    token: str
    agent_id: str
    hostname: str
    platform: str
    ip_address: Optional[str] = None
    metadata: Optional[Dict] = None


class AgentHeartbeatRequest(BaseModel):
    """Request model for agent heartbeat"""

    agent_id: str
    metrics: Optional[Dict] = None
    timestamp: Optional[str] = None


class AgentEventsRequest(BaseModel):
    """Request model for agent security events"""

    agent_id: str
    events: List[Dict]


class AgentEnrollResponse(BaseModel):
    """Response model for agent enrollment"""

    enrollment_id: int
    agent_id: str
    hostname: str
    platform: str
    status: str
    message: str = "Agent enrolled successfully"


class AgentHeartbeatResponse(BaseModel):
    """Response model for agent heartbeat"""

    status: str = "received"
    message: str = "Heartbeat received successfully"


class AgentEventsResponse(BaseModel):
    """Response model for agent events"""

    status: str = "received"
    events_processed: int = 0
    message: str = "Events received successfully"


@router.post("/enroll")
async def enroll_agent(request: AgentEnrollRequest, db: AsyncSession = Depends(get_db)):
    """
    Agent enrollment endpoint - called by agents during first contact

    This endpoint allows agents to register themselves with the XDR system
    using their enrollment token.
    """
    try:
        logger.info(
            f"Agent enrollment request: {request.agent_id} ({request.hostname})"
        )

        # Extract organization ID from token (format: aws-{org_id}-{asset_id}-{token})
        # For now, we'll use org ID 1 (Mini Corp) as default
        org_id = 1  # TODO: Extract from token or determine from agent_id

        # Create enrollment service
        enrollment_service = AgentEnrollmentService(org_id, db)

        # Register the agent
        result = await enrollment_service.register_agent(
            agent_token=request.token,
            agent_id=request.agent_id,
            hostname=request.hostname,
            platform=request.platform,
            ip_address=request.ip_address,
            metadata=request.metadata,
        )

        logger.info(f"Agent enrolled successfully: {request.agent_id}")

        return AgentEnrollResponse(
            enrollment_id=result["enrollment_id"],
            agent_id=result["agent_id"],
            hostname=result["hostname"],
            platform=result["platform"],
            status=result["status"],
        )

    except ValueError as e:
        logger.warning(f"Agent enrollment failed - invalid token: {e}")
        raise HTTPException(status_code=401, detail=str(e))
    except Exception as e:
        logger.error(f"Agent enrollment error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/heartbeat")
async def agent_heartbeat(
    request: AgentHeartbeatRequest, db: AsyncSession = Depends(get_db)
):
    """
    Agent heartbeat endpoint - called periodically by enrolled agents

    Updates the agent's last seen timestamp and collects system metrics.
    """
    try:
        logger.debug(f"Agent heartbeat: {request.agent_id}")

        # For now, we'll use org ID 1 (Mini Corp) as default
        org_id = 1

        # Create enrollment service
        enrollment_service = AgentEnrollmentService(org_id, db)

        # Update agent heartbeat
        # TODO: Implement heartbeat update in AgentEnrollmentService
        # For now, we'll just acknowledge receipt

        logger.debug(f"Heartbeat processed for agent: {request.agent_id}")

        return AgentHeartbeatResponse()

    except Exception as e:
        logger.error(f"Agent heartbeat error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/events")
async def submit_agent_events(
    request: AgentEventsRequest, db: AsyncSession = Depends(get_db)
):
    """
    Agent security events endpoint - called by agents to submit telemetry

    Accepts security events, process executions, network connections, etc.
    """
    try:
        logger.info(
            f"Agent events received: {request.agent_id} - {len(request.events)} events"
        )

        # Process events (store in incidents or telemetry table)
        events_processed = len(request.events)

        # TODO: Implement proper event processing and storage
        # For now, just log the events
        for event in request.events:
            logger.info(
                f"Security event from {request.agent_id}: {event.get('type')} - {event.get('timestamp')}"
            )

        return AgentEventsResponse(events_processed=events_processed)

    except Exception as e:
        logger.error(f"Agent events submission error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Legacy endpoint for backwards compatibility
@router.post("/check-in")
async def agent_check_in(request: Dict, db: AsyncSession = Depends(get_db)):
    """
    Legacy agent check-in endpoint
    """
    # Route to appropriate handler based on request content
    if "token" in request:
        # Enrollment request
        enroll_request = AgentEnrollRequest(**request)
        return await enroll_agent(enroll_request, db)
    else:
        # Heartbeat request
        heartbeat_request = AgentHeartbeatRequest(**request)
        return await agent_heartbeat(heartbeat_request, db)

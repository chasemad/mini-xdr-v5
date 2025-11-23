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


# =============================================================================
# AI Agent Coordination Endpoints (Phase 2)
# =============================================================================


class CoordinationHubStatusResponse(BaseModel):
    """Response model for coordination hub status"""

    status: str
    total_incidents_coordinated: int
    active_agents: List[str]
    coordination_metrics: Dict
    last_coordination_timestamp: Optional[str] = None


class AIAgentStatusResponse(BaseModel):
    """Response model for AI agent status"""

    agent_name: str
    status: str
    decisions_count: int
    performance_metrics: Dict
    last_active_timestamp: Optional[str] = None


class AgentDecisionResponse(BaseModel):
    """Response model for agent decisions"""

    agent_name: str
    decisions: List[Dict]
    total_count: int


class IncidentCoordinationResponse(BaseModel):
    """Response model for incident agent coordination"""

    incident_id: int
    coordination_status: str
    participating_agents: List[str]
    agent_decisions: Dict
    coordination_timeline: List[Dict]
    recommendations: List[str]


@router.get("/coordination-hub/status")
async def get_coordination_hub_status(db: AsyncSession = Depends(get_db)):
    """
    Get overall coordination hub status (Phase 2)

    Returns coordination hub metrics including active agents, coordination count,
    and performance statistics.
    """
    try:
        from .agents.coordination_hub import get_coordination_hub

        # Get coordination hub instance and status
        hub = await get_coordination_hub()
        status = hub.get_coordination_status()

        return CoordinationHubStatusResponse(
            status="operational",
            total_incidents_coordinated=status.get("total_coordinations", 0),
            active_agents=status.get("active_agents", []),
            coordination_metrics=status.get("metrics", {}),
            last_coordination_timestamp=status.get(
                "last_coordination_time", datetime.now(timezone.utc).isoformat()
            ),
        )

    except Exception as e:
        logger.error(f"Failed to get coordination hub status: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to retrieve coordination hub status"
        )


@router.get("/ai/{agent_name}/status")
async def get_ai_agent_status(agent_name: str, db: AsyncSession = Depends(get_db)):
    """
    Get status for a specific AI agent (Phase 2)

    Available agents: attribution, containment, forensics, deception, dlp,
    hunting, response_optimizer, effectiveness_tracker
    """
    try:
        # Map agent names to their instances
        agent_map = {}

        try:
            from .agents import (
                attribution_tracker,
                containment_orchestrator,
                deception_manager,
                forensics_investigator,
            )

            agent_map.update(
                {
                    "attribution": attribution_tracker,
                    "containment": containment_orchestrator,
                    "forensics": forensics_investigator,
                    "deception": deception_manager,
                }
            )
        except ImportError as e:
            logger.warning(f"Some agents not available: {e}")

        if agent_name not in agent_map:
            raise HTTPException(
                status_code=404,
                detail=f"Agent '{agent_name}' not found. Available: {list(agent_map.keys())}",
            )

        agent = agent_map[agent_name]

        # Get agent performance metrics if available
        performance_metrics = {}
        if hasattr(agent, "get_performance_metrics"):
            performance_metrics = agent.get_performance_metrics()

        # Get decision count
        decisions_count = 0
        if hasattr(agent, "decision_history"):
            decisions_count = len(agent.decision_history)

        return AIAgentStatusResponse(
            agent_name=agent_name,
            status="operational",
            decisions_count=decisions_count,
            performance_metrics=performance_metrics,
            last_active_timestamp=datetime.now(timezone.utc).isoformat(),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get AI agent status for {agent_name}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve agent status: {str(e)}"
        )


@router.get("/ai/{agent_name}/decisions")
async def get_ai_agent_decisions(
    agent_name: str, limit: int = 50, db: AsyncSession = Depends(get_db)
):
    """
    Get recent decisions made by a specific AI agent (Phase 2)

    Returns the decision history for the specified agent.
    """
    try:
        # Map agent names to their instances
        agent_map = {}

        try:
            from .agents import (
                attribution_tracker,
                containment_orchestrator,
                deception_manager,
                forensics_investigator,
            )

            agent_map.update(
                {
                    "attribution": attribution_tracker,
                    "containment": containment_orchestrator,
                    "forensics": forensics_investigator,
                    "deception": deception_manager,
                }
            )
        except ImportError as e:
            logger.warning(f"Some agents not available: {e}")

        if agent_name not in agent_map:
            raise HTTPException(
                status_code=404,
                detail=f"Agent '{agent_name}' not found. Available: {list(agent_map.keys())}",
            )

        agent = agent_map[agent_name]

        # Get decision history
        decisions = []
        if hasattr(agent, "decision_history"):
            decisions = agent.decision_history[-limit:]  # Get last N decisions

        # Convert dataclass decisions to dict
        decisions_list = []
        for decision in decisions:
            if hasattr(decision, "__dict__"):
                decisions_list.append(decision.__dict__)
            elif isinstance(decision, dict):
                decisions_list.append(decision)
            else:
                decisions_list.append({"raw_decision": str(decision)})

        return AgentDecisionResponse(
            agent_name=agent_name,
            decisions=decisions_list,
            total_count=len(decisions),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get decisions for {agent_name}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve decisions: {str(e)}"
        )


@router.get("/incidents/{incident_id}/coordination")
async def get_incident_agent_coordination(
    incident_id: int, db: AsyncSession = Depends(get_db)
):
    """
    Get agent coordination details for a specific incident (Phase 2)

    Returns which agents were involved, their decisions, and coordination timeline.
    """
    try:
        from sqlalchemy import select

        from .models import Incident

        # Fetch incident
        stmt = select(Incident).where(Incident.id == incident_id)
        result = await db.execute(stmt)
        incident = result.scalar_one_or_none()

        if not incident:
            raise HTTPException(
                status_code=404, detail=f"Incident {incident_id} not found"
            )

        # Extract agent coordination data from incident
        participating_agents = []
        agent_decisions = {}
        coordination_timeline = []
        recommendations = []

        # Check triage_note for agent data
        if incident.triage_note and isinstance(incident.triage_note, dict):
            triage = incident.triage_note

            # Extract agent participation
            if "agents" in triage:
                participating_agents = list(triage["agents"].keys())
                agent_decisions = triage["agents"]

            # Extract recommendations
            if "recommendation" in triage:
                recommendations.append(triage["recommendation"])
            if "action_plan" in triage:
                recommendations.extend(triage.get("action_plan", []))

        # Check for Council data
        coordination_status = "standard"
        if incident.council_verdict:
            coordination_status = "council_verified"

            # Add Council data to timeline
            if incident.council_reasoning:
                coordination_timeline.append(
                    {
                        "timestamp": incident.created_at.isoformat()
                        if incident.created_at
                        else None,
                        "event": "council_verification",
                        "details": incident.council_reasoning,
                        "verdict": incident.council_verdict,
                    }
                )

        return IncidentCoordinationResponse(
            incident_id=incident_id,
            coordination_status=coordination_status,
            participating_agents=participating_agents,
            agent_decisions=agent_decisions,
            coordination_timeline=coordination_timeline,
            recommendations=recommendations,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get coordination for incident {incident_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve incident coordination: {str(e)}",
        )

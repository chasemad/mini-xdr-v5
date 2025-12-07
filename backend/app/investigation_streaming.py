"""
Real-time streaming support for investigation results
Provides Server-Sent Events (SSE) for live updates
"""

import asyncio
import json
import logging
from typing import AsyncGenerator

from fastapi import Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .db import get_db
from .investigation_routes import _require_api_key
from .models import InvestigationResult

logger = logging.getLogger(__name__)


async def stream_investigation_results(
    incident_id: int, db: AsyncSession
) -> AsyncGenerator[str, None]:
    """
    Stream investigation results in real-time using Server-Sent Events.

    Args:
        incident_id: ID of the incident to stream results for
        db: Database session

    Yields:
        SSE-formatted messages with investigation result updates
    """
    last_id = 0

    # Send initial connection message
    yield f"data: {json.dumps({'type': 'connected', 'incident_id': incident_id})}\n\n"

    try:
        while True:
            # Query for new results since last check
            result = await db.execute(
                select(InvestigationResult)
                .where(InvestigationResult.incident_id == incident_id)
                .where(InvestigationResult.id > last_id)
                .order_by(InvestigationResult.id)
            )

            new_results = result.scalars().all()

            # Send each new result
            for investigation in new_results:
                data = {
                    "type": "investigation_result",
                    "id": investigation.id,
                    "investigation_id": investigation.investigation_id,
                    "tool_name": investigation.tool_name,
                    "tool_category": investigation.tool_category,
                    "status": investigation.status,
                    "severity": investigation.severity,
                    "confidence_score": investigation.confidence_score,
                    "started_at": investigation.started_at.isoformat()
                    if investigation.started_at
                    else None,
                    "completed_at": investigation.completed_at.isoformat()
                    if investigation.completed_at
                    else None,
                    "execution_time_ms": investigation.execution_time_ms,
                    "parameters": investigation.parameters,
                    "results": investigation.results,
                    "findings_count": investigation.findings_count,
                    "iocs_discovered": investigation.iocs_discovered,
                    "error_message": investigation.error_message,
                }

                yield f"data: {json.dumps(data)}\n\n"
                last_id = investigation.id

            # Wait before checking again (1 second polling interval)
            await asyncio.sleep(1)

    except asyncio.CancelledError:
        # Client disconnected
        logger.info(f"Stream cancelled for incident {incident_id}")
        yield f"data: {json.dumps({'type': 'disconnected'})}\n\n"
    except Exception as e:
        logger.error(f"Stream error for incident {incident_id}: {e}")
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"


async def get_investigation_stream(
    incident_id: int, db: AsyncSession = Depends(get_db)
) -> StreamingResponse:
    """
    FastAPI endpoint for streaming investigation results.

    Args:
        incident_id: ID of the incident
        db: Database session

    Returns:
        StreamingResponse with text/event-stream content type
    """
    # Verify incident exists
    from .models import Incident

    result = await db.execute(select(Incident).where(Incident.id == incident_id))
    incident = result.scalar_one_or_none()

    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")

    return StreamingResponse(
        stream_investigation_results(incident_id, db),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )


async def stream_agent_execution(
    agent_executor,
    messages: list,
) -> AsyncGenerator[str, None]:
    """
    Stream agent execution steps in real-time.

    This uses LangGraph's .astream() to send each agent step
    to the client as it happens.

    Args:
        agent_executor: LangGraph agent executor
        messages: Messages to send to the agent

    Yields:
        SSE-formatted messages with agent execution steps
    """
    try:
        # Stream agent execution
        async for event in agent_executor.astream({"messages": messages}):
            # Extract relevant data from event
            if isinstance(event, dict):
                # Send the event to client
                yield f"data: {json.dumps(event)}\n\n"

        # Send completion message
        yield f"data: {json.dumps({'type': 'completed'})}\n\n"

    except Exception as e:
        logger.error(f"Agent streaming error: {e}")
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

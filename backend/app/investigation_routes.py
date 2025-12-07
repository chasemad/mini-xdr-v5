# ==================== INVESTIGATION RESULTS API ENDPOINTS ====================
# These endpoints handle tool execution results and investigation findings

import logging
from datetime import datetime
from typing import Any, Dict

from fastapi import Depends, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .auth import get_current_user
from .db import get_db
from .models import Incident, InvestigationResult, User

logger = logging.getLogger(__name__)


def _require_api_key(request: Request):
    """Require API key for security"""
    import hmac

    from .config import settings

    if not settings.api_key:
        logger.error("API key must be configured for security")
        raise HTTPException(status_code=500, detail="API key must be configured")

    api_key = request.headers.get("x-api-key")
    if not api_key:
        raise HTTPException(status_code=401, detail="Missing API key header")

    # Use secure comparison to prevent timing attacks
    if not hmac.compare_digest(api_key, settings.api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")


async def get_incident_investigations(
    incident_id: int, http_request: Request, db: AsyncSession = Depends(get_db)
):
    """Get all investigation results for an incident"""
    _require_api_key(http_request)

    try:
        investigations = (
            (
                await db.execute(
                    select(InvestigationResult)
                    .where(InvestigationResult.incident_id == incident_id)
                    .order_by(InvestigationResult.created_at.desc())
                )
            )
            .scalars()
            .all()
        )

        return [
            {
                "id": inv.id,
                "investigation_id": inv.investigation_id,
                "tool_name": inv.tool_name,
                "tool_category": inv.tool_category,
                "status": inv.status,
                "started_at": inv.started_at.isoformat() if inv.started_at else None,
                "completed_at": inv.completed_at.isoformat()
                if inv.completed_at
                else None,
                "execution_time_ms": inv.execution_time_ms,
                "parameters": inv.parameters,
                "results": inv.results,
                "findings_count": inv.findings_count,
                "iocs_discovered": inv.iocs_discovered,
                "severity": inv.severity,
                "confidence_score": inv.confidence_score,
                "triggered_by": inv.triggered_by,
                "auto_triggered": inv.auto_triggered,
                "error_message": inv.error_message,
                "exported": inv.exported,
            }
            for inv in investigations
        ]

    except Exception as e:
        logger.error(f"Failed to get investigations for incident {incident_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def execute_investigation_tool(
    incident_id: int,
    request_data: Dict[str, Any],
    http_request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Execute an investigation tool for an incident"""
    _require_api_key(http_request)

    try:
        tool_name = request_data.get("tool_name")
        parameters = request_data.get("parameters", {})

        if not tool_name:
            raise HTTPException(status_code=400, detail="tool_name is required")

        # Verify incident exists
        incident = (
            (await db.execute(select(Incident).where(Incident.id == incident_id)))
            .scalars()
            .first()
        )

        if not incident:
            raise HTTPException(status_code=404, detail="Incident not found")

        # Create investigation result record
        investigation_id = (
            f"inv_{incident_id}_{tool_name}_{int(datetime.utcnow().timestamp())}"
        )

        investigation = InvestigationResult(
            investigation_id=investigation_id,
            incident_id=incident_id,
            tool_name=tool_name,
            tool_category=request_data.get("tool_category", "investigation"),
            status="running",
            parameters=parameters,
            triggered_by=f"analyst_{current_user.id}",
            triggered_by_user_id=current_user.id,
            auto_triggered=False,
        )

        db.add(investigation)
        await db.commit()
        await db.refresh(investigation)

        # Execute the tool asynchronously
        # For now, we'll simulate execution and update the record
        # In production, this would call the actual LangChain tool
        import asyncio

        await asyncio.sleep(0.1)  # Simulate async processing

        # Mock results for now - replace with actual tool execution
        mock_results = {
            "summary": f"{tool_name} execution completed",
            "findings": [
                {
                    "title": "Sample Finding",
                    "description": f"Results from {tool_name}",
                    "severity": "medium",
                    "iocs": [],
                }
            ],
            "recommendations": ["Continue monitoring"],
            "evidence": {},
        }

        investigation.status = "completed"
        investigation.completed_at = datetime.utcnow()
        investigation.execution_time_ms = 100
        investigation.results = mock_results
        investigation.findings_count = len(mock_results.get("findings", []))
        investigation.severity = "medium"
        investigation.confidence_score = 0.85

        await db.commit()
        await db.refresh(investigation)

        return {
            "success": True,
            "investigation_id": investigation.investigation_id,
            "status": investigation.status,
            "message": f"Investigation tool '{tool_name}' executed successfully",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to execute investigation tool: {e}")
        return {"success": False, "error": str(e)}


async def get_investigation_result(
    investigation_id: str, http_request: Request, db: AsyncSession = Depends(get_db)
):
    """Get a specific investigation result by ID"""
    _require_api_key(http_request)

    try:
        investigation = (
            (
                await db.execute(
                    select(InvestigationResult).where(
                        InvestigationResult.investigation_id == investigation_id
                    )
                )
            )
            .scalars()
            .first()
        )

        if not investigation:
            raise HTTPException(status_code=404, detail="Investigation not found")

        return {
            "id": investigation.id,
            "investigation_id": investigation.investigation_id,
            "incident_id": investigation.incident_id,
            "tool_name": investigation.tool_name,
            "tool_category": investigation.tool_category,
            "status": investigation.status,
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
            "severity": investigation.severity,
            "confidence_score": investigation.confidence_score,
            "triggered_by": investigation.triggered_by,
            "auto_triggered": investigation.auto_triggered,
            "error_message": investigation.error_message,
            "exported": investigation.exported,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get investigation {investigation_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def export_investigation_result(
    investigation_id: str,
    request_data: Dict[str, Any],
    http_request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Export investigation results in specified format"""
    _require_api_key(http_request)

    try:
        format_type = request_data.get("format", "json")  # json, markdown, pdf

        investigation = (
            (
                await db.execute(
                    select(InvestigationResult).where(
                        InvestigationResult.investigation_id == investigation_id
                    )
                )
            )
            .scalars()
            .first()
        )

        if not investigation:
            raise HTTPException(status_code=404, detail="Investigation not found")

        # Mark as exported
        if not investigation.exported:
            investigation.exported = True
            investigation.export_formats = investigation.export_formats or []
            if format_type not in investigation.export_formats:
                investigation.export_formats.append(format_type)
            await db.commit()

        # Generate export based on format
        if format_type == "json":
            return {
                "investigation_id": investigation.investigation_id,
                "tool_name": investigation.tool_name,
                "status": investigation.status,
                "results": investigation.results,
                "findings_count": investigation.findings_count,
                "iocs_discovered": investigation.iocs_discovered,
                "severity": investigation.severity,
                "confidence_score": investigation.confidence_score,
                "executed_at": investigation.started_at.isoformat()
                if investigation.started_at
                else None,
            }
        elif format_type == "markdown":
            # Generate markdown report
            markdown_content = f"""# Investigation Report: {investigation.tool_name}

**Investigation ID**: {investigation.investigation_id}
**Status**: {investigation.status}
**Severity**: {investigation.severity}
**Confidence**: {investigation.confidence_score * 100:.1f}%
**Executed**: {investigation.started_at.isoformat() if investigation.started_at else 'N/A'}

## Summary
{investigation.results.get('summary', 'No summary available') if investigation.results else 'No results'}

## Findings
"""
            if investigation.results and investigation.results.get("findings"):
                for i, finding in enumerate(investigation.results["findings"], 1):
                    markdown_content += (
                        f"\n### Finding {i}: {finding.get('title', 'Untitled')}\n"
                    )
                    markdown_content += f"**Severity**: {finding.get('severity', 'unknown').upper()}\n\n"
                    markdown_content += f"{finding.get('description', '')}\n"

            return {"format": "markdown", "content": markdown_content}
        else:
            raise HTTPException(
                status_code=400, detail=f"Unsupported format: {format_type}"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to export investigation {investigation_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

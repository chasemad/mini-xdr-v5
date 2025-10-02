"""
NLP Workflow Suggestion API Routes
Handles NLP-parsed workflow suggestions awaiting approval
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel

from .db import get_db
from .models import NLPWorkflowSuggestion, WorkflowTrigger, Incident
from .security import require_api_key
from .nlp_workflow_parser import parse_workflow_from_natural_language

router = APIRouter(prefix="/api/nlp-suggestions", tags=["nlp-suggestions"])


# Pydantic models
class NLPSuggestionCreate(BaseModel):
    prompt: str
    incident_id: Optional[int] = None


class NLPSuggestionResponse(BaseModel):
    id: int
    prompt: str
    incident_id: Optional[int]
    request_type: str
    priority: str
    confidence: float
    fallback_used: bool
    workflow_steps: List[dict]
    detected_actions: Optional[List[str]]
    missing_actions: Optional[List[str]]
    status: str
    reviewed_by: Optional[str]
    reviewed_at: Optional[datetime]
    trigger_id: Optional[int]
    parser_version: str
    parser_diagnostics: Optional[dict]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ApprovalRequest(BaseModel):
    trigger_name: str
    trigger_description: Optional[str] = None
    auto_execute: bool = False
    category: str = "nlp_generated"
    owner: Optional[str] = None


# API Endpoints
@router.post("/parse", response_model=NLPSuggestionResponse)
async def parse_nlp_workflow(
    data: NLPSuggestionCreate,
    db: AsyncSession = Depends(get_db),
    _: bool = Depends(require_api_key)
):
    """Parse natural language into workflow suggestion"""
    # Parse using NLP parser with incident context
    intent, explanation = await parse_workflow_from_natural_language(
        data.prompt,
        data.incident_id,
        db
    )

    # Extract detected actions
    detected_actions = [action['action_type'] for action in intent.actions]

    # Create suggestion record
    suggestion = NLPWorkflowSuggestion(
        prompt=data.prompt,
        incident_id=data.incident_id,
        request_type=intent.request_type,
        priority=intent.priority,
        confidence=intent.confidence,
        fallback_used=False,  # Implement fallback logic
        workflow_steps=intent.to_workflow_steps(),
        detected_actions=detected_actions,
        missing_actions=[],  # Implement missing action detection
        parser_version="v1.0",
        parser_diagnostics={
            "explanation": explanation,
            "confidence": intent.confidence,
            "approval_required": intent.approval_required
        }
    )

    db.add(suggestion)
    await db.commit()
    await db.refresh(suggestion)

    return suggestion


@router.get("/", response_model=List[NLPSuggestionResponse])
async def list_suggestions(
    status: Optional[str] = None,
    request_type: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
    _: bool = Depends(require_api_key)
):
    """List all NLP workflow suggestions"""
    query = select(NLPWorkflowSuggestion).order_by(desc(NLPWorkflowSuggestion.created_at))

    if status:
        query = query.where(NLPWorkflowSuggestion.status == status)
    if request_type:
        query = query.where(NLPWorkflowSuggestion.request_type == request_type)

    result = await db.execute(query)
    suggestions = result.scalars().all()
    return suggestions


@router.post("/{suggestion_id}/approve", response_model=dict)
async def approve_suggestion(
    suggestion_id: int,
    approval: ApprovalRequest,
    db: AsyncSession = Depends(get_db),
    _: bool = Depends(require_api_key)
):
    """Approve suggestion and create trigger"""
    # Get suggestion
    result = await db.execute(
        select(NLPWorkflowSuggestion).where(NLPWorkflowSuggestion.id == suggestion_id)
    )
    suggestion = result.scalar_one_or_none()

    if not suggestion:
        raise HTTPException(status_code=404, detail="Suggestion not found")

    if suggestion.status != "pending":
        raise HTTPException(status_code=400, detail="Suggestion already processed")

    # Create trigger from suggestion
    trigger = WorkflowTrigger(
        name=approval.trigger_name,
        description=approval.trigger_description or f"Auto-generated from NLP: {suggestion.prompt}",
        category=approval.category,
        enabled=True,
        auto_execute=approval.auto_execute,
        priority=suggestion.priority,
        status="active",
        conditions={"event_type": "manual", "source": "nlp"},
        playbook_name=f"nlp_{suggestion.id}",
        workflow_steps=suggestion.workflow_steps,
        source="nlp",
        source_prompt=suggestion.prompt,
        parser_confidence=suggestion.confidence,
        parser_version=suggestion.parser_version,
        request_type=suggestion.request_type,
        fallback_used=suggestion.fallback_used,
        owner=approval.owner,
        created_by="nlp_system",
    )

    db.add(trigger)

    # Update suggestion status
    suggestion.status = "approved"
    suggestion.reviewed_by = approval.owner or "system"
    suggestion.reviewed_at = datetime.utcnow()
    suggestion.trigger_id = trigger.id

    await db.commit()
    await db.refresh(trigger)

    return {
        "success": True,
        "trigger_id": trigger.id,
        "message": f"Trigger '{trigger.name}' created successfully"
    }


@router.post("/{suggestion_id}/dismiss")
async def dismiss_suggestion(
    suggestion_id: int,
    reason: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
    _: bool = Depends(require_api_key)
):
    """Dismiss a workflow suggestion"""
    result = await db.execute(
        select(NLPWorkflowSuggestion).where(NLPWorkflowSuggestion.id == suggestion_id)
    )
    suggestion = result.scalar_one_or_none()

    if not suggestion:
        raise HTTPException(status_code=404, detail="Suggestion not found")

    suggestion.status = "dismissed"
    suggestion.reviewed_by = "user"
    suggestion.reviewed_at = datetime.utcnow()

    if reason:
        suggestion.parser_diagnostics = suggestion.parser_diagnostics or {}
        suggestion.parser_diagnostics["dismissal_reason"] = reason

    await db.commit()

    return {"success": True, "message": "Suggestion dismissed"}


@router.get("/stats")
async def get_suggestion_stats(
    db: AsyncSession = Depends(get_db),
    _: bool = Depends(require_api_key)
):
    """Get statistics about NLP suggestions"""
    result = await db.execute(select(NLPWorkflowSuggestion))
    suggestions = result.scalars().all()

    total = len(suggestions)
    by_status = {}
    by_request_type = {}
    avg_confidence = 0

    for suggestion in suggestions:
        by_status[suggestion.status] = by_status.get(suggestion.status, 0) + 1
        by_request_type[suggestion.request_type] = by_request_type.get(suggestion.request_type, 0) + 1
        avg_confidence += suggestion.confidence

    if total > 0:
        avg_confidence /= total

    return {
        "total_suggestions": total,
        "by_status": by_status,
        "by_request_type": by_request_type,
        "avg_confidence": round(avg_confidence, 2),
        "pending_review": by_status.get("pending", 0)
    }

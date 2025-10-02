"""
Workflow Trigger Management API Routes
Handles CRUD operations for automatic workflow triggers
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel

from .db import get_db
from .models import WorkflowTrigger
from .security import require_api_key

router = APIRouter(prefix="/api/triggers", tags=["triggers"])


# Pydantic models for request/response
class TriggerCondition(BaseModel):
    event_type: Optional[str] = None
    threshold: Optional[int] = None
    window_seconds: Optional[int] = None
    pattern_match: Optional[str] = None
    risk_score_min: Optional[float] = None
    source: Optional[str] = None


class WorkflowStep(BaseModel):
    action_type: str
    parameters: dict
    timeout_seconds: Optional[int] = 30
    continue_on_failure: Optional[bool] = False


class TriggerCreate(BaseModel):
    name: str
    description: Optional[str] = None
    category: str
    enabled: bool = True
    auto_execute: bool = False
    priority: str = "medium"
    conditions: dict
    playbook_name: str
    workflow_steps: List[dict]
    cooldown_seconds: int = 60
    max_triggers_per_day: int = 100
    tags: Optional[List[str]] = None


class TriggerUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    category: Optional[str] = None
    enabled: Optional[bool] = None
    auto_execute: Optional[bool] = None
    priority: Optional[str] = None
    conditions: Optional[dict] = None
    playbook_name: Optional[str] = None
    workflow_steps: Optional[List[dict]] = None
    cooldown_seconds: Optional[int] = None
    max_triggers_per_day: Optional[int] = None
    tags: Optional[List[str]] = None


class TriggerResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    category: str
    enabled: bool
    auto_execute: bool
    priority: str
    conditions: dict
    playbook_name: str
    workflow_steps: List[dict]
    trigger_count: int
    success_count: int
    failure_count: int
    success_rate: float
    avg_response_time_ms: float
    last_triggered_at: Optional[datetime]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# API Endpoints
@router.get("/", response_model=List[TriggerResponse])
async def list_triggers(
    category: Optional[str] = None,
    enabled: Optional[bool] = None,
    db: AsyncSession = Depends(get_db),
    _: bool = Depends(require_api_key)
):
    """List all workflow triggers with optional filtering"""
    query = select(WorkflowTrigger).order_by(desc(WorkflowTrigger.created_at))

    if category:
        query = query.where(WorkflowTrigger.category == category)
    if enabled is not None:
        query = query.where(WorkflowTrigger.enabled == enabled)

    result = await db.execute(query)
    triggers = result.scalars().all()
    return triggers


@router.get("/{trigger_id}", response_model=TriggerResponse)
async def get_trigger(
    trigger_id: int,
    db: AsyncSession = Depends(get_db),
    _: bool = Depends(require_api_key)
):
    """Get a specific trigger by ID"""
    result = await db.execute(
        select(WorkflowTrigger).where(WorkflowTrigger.id == trigger_id)
    )
    trigger = result.scalar_one_or_none()

    if not trigger:
        raise HTTPException(status_code=404, detail="Trigger not found")

    return trigger


@router.post("/", response_model=TriggerResponse)
async def create_trigger(
    trigger_data: TriggerCreate,
    db: AsyncSession = Depends(get_db),
    _: bool = Depends(require_api_key)
):
    """Create a new workflow trigger"""
    # Check if trigger with same name exists
    result = await db.execute(
        select(WorkflowTrigger).where(WorkflowTrigger.name == trigger_data.name)
    )
    existing = result.scalar_one_or_none()

    if existing:
        raise HTTPException(status_code=400, detail="Trigger with this name already exists")

    # Create new trigger
    trigger = WorkflowTrigger(
        name=trigger_data.name,
        description=trigger_data.description,
        category=trigger_data.category,
        enabled=trigger_data.enabled,
        auto_execute=trigger_data.auto_execute,
        priority=trigger_data.priority,
        conditions=trigger_data.conditions,
        playbook_name=trigger_data.playbook_name,
        workflow_steps=trigger_data.workflow_steps,
        cooldown_seconds=trigger_data.cooldown_seconds,
        max_triggers_per_day=trigger_data.max_triggers_per_day,
        tags=trigger_data.tags
    )

    db.add(trigger)
    await db.commit()
    await db.refresh(trigger)

    return trigger


@router.put("/{trigger_id}", response_model=TriggerResponse)
async def update_trigger(
    trigger_id: int,
    trigger_data: TriggerUpdate,
    db: AsyncSession = Depends(get_db),
    _: bool = Depends(require_api_key)
):
    """Update an existing trigger"""
    result = await db.execute(
        select(WorkflowTrigger).where(WorkflowTrigger.id == trigger_id)
    )
    trigger = result.scalar_one_or_none()

    if not trigger:
        raise HTTPException(status_code=404, detail="Trigger not found")

    # Update fields
    update_data = trigger_data.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(trigger, key, value)

    trigger.updated_at = datetime.utcnow()

    await db.commit()
    await db.refresh(trigger)

    return trigger


@router.delete("/{trigger_id}")
async def delete_trigger(
    trigger_id: int,
    db: AsyncSession = Depends(get_db),
    _: bool = Depends(require_api_key)
):
    """Delete a trigger"""
    result = await db.execute(
        select(WorkflowTrigger).where(WorkflowTrigger.id == trigger_id)
    )
    trigger = result.scalar_one_or_none()

    if not trigger:
        raise HTTPException(status_code=404, detail="Trigger not found")

    await db.delete(trigger)
    await db.commit()

    return {"success": True, "message": f"Trigger {trigger.name} deleted"}


@router.post("/{trigger_id}/enable")
async def enable_trigger(
    trigger_id: int,
    db: AsyncSession = Depends(get_db),
    _: bool = Depends(require_api_key)
):
    """Enable a trigger"""
    result = await db.execute(
        select(WorkflowTrigger).where(WorkflowTrigger.id == trigger_id)
    )
    trigger = result.scalar_one_or_none()

    if not trigger:
        raise HTTPException(status_code=404, detail="Trigger not found")

    trigger.enabled = True
    trigger.updated_at = datetime.utcnow()

    await db.commit()
    await db.refresh(trigger)

    return {"success": True, "trigger": trigger}


@router.post("/{trigger_id}/disable")
async def disable_trigger(
    trigger_id: int,
    db: AsyncSession = Depends(get_db),
    _: bool = Depends(require_api_key)
):
    """Disable a trigger"""
    result = await db.execute(
        select(WorkflowTrigger).where(WorkflowTrigger.id == trigger_id)
    )
    trigger = result.scalar_one_or_none()

    if not trigger:
        raise HTTPException(status_code=404, detail="Trigger not found")

    trigger.enabled = False
    trigger.updated_at = datetime.utcnow()

    await db.commit()
    await db.refresh(trigger)

    return {"success": True, "trigger": trigger}


@router.get("/stats/summary")
async def get_trigger_stats(
    db: AsyncSession = Depends(get_db),
    _: bool = Depends(require_api_key)
):
    """Get overall trigger statistics"""
    result = await db.execute(select(WorkflowTrigger))
    triggers = result.scalars().all()

    total = len(triggers)
    enabled = len([t for t in triggers if t.enabled])
    auto_execute = len([t for t in triggers if t.auto_execute])
    total_triggers = sum(t.trigger_count for t in triggers)
    avg_success_rate = sum(t.success_rate for t in triggers) / total if total > 0 else 0

    return {
        "total_triggers": total,
        "enabled_triggers": enabled,
        "auto_execute_triggers": auto_execute,
        "total_executions": total_triggers,
        "avg_success_rate": round(avg_success_rate, 2)
    }


@router.post("/bulk/pause")
async def bulk_pause_triggers(
    trigger_ids: List[int],
    db: AsyncSession = Depends(get_db),
    _: bool = Depends(require_api_key)
):
    """Pause multiple triggers at once"""
    result = await db.execute(
        select(WorkflowTrigger).where(WorkflowTrigger.id.in_(trigger_ids))
    )
    triggers = result.scalars().all()

    for trigger in triggers:
        trigger.enabled = False
        trigger.status = "paused"
        trigger.updated_at = datetime.utcnow()

    await db.commit()
    return {"success": True, "paused_count": len(triggers)}


@router.post("/bulk/resume")
async def bulk_resume_triggers(
    trigger_ids: List[int],
    db: AsyncSession = Depends(get_db),
    _: bool = Depends(require_api_key)
):
    """Resume multiple triggers at once"""
    result = await db.execute(
        select(WorkflowTrigger).where(WorkflowTrigger.id.in_(trigger_ids))
    )
    triggers = result.scalars().all()

    for trigger in triggers:
        trigger.enabled = True
        trigger.status = "active"
        trigger.updated_at = datetime.utcnow()

    await db.commit()
    return {"success": True, "resumed_count": len(triggers)}


@router.post("/bulk/archive")
async def bulk_archive_triggers(
    trigger_ids: List[int],
    db: AsyncSession = Depends(get_db),
    _: bool = Depends(require_api_key)
):
    """Archive multiple triggers at once"""
    result = await db.execute(
        select(WorkflowTrigger).where(WorkflowTrigger.id.in_(trigger_ids))
    )
    triggers = result.scalars().all()

    for trigger in triggers:
        trigger.enabled = False
        trigger.status = "archived"
        trigger.updated_at = datetime.utcnow()

    await db.commit()
    return {"success": True, "archived_count": len(triggers)}


@router.post("/{trigger_id}/simulate")
async def simulate_trigger(
    trigger_id: int,
    test_data: Optional[dict] = None,
    db: AsyncSession = Depends(get_db),
    _: bool = Depends(require_api_key)
):
    """Run a dry-run simulation of a trigger without executing actions"""
    result = await db.execute(
        select(WorkflowTrigger).where(WorkflowTrigger.id == trigger_id)
    )
    trigger = result.scalar_one_or_none()

    if not trigger:
        raise HTTPException(status_code=404, detail="Trigger not found")

    # Simulate trigger evaluation
    simulation_result = {
        "trigger_id": trigger.id,
        "trigger_name": trigger.name,
        "would_execute": trigger.enabled and trigger.auto_execute,
        "conditions_met": True,  # In real implementation, evaluate conditions
        "workflow_steps": trigger.workflow_steps,
        "estimated_duration_seconds": len(trigger.workflow_steps) * 30,
        "required_approvals": not trigger.auto_execute,
        "safety_checks": {
            "destructive_actions": any(
                step.get("action_type") in ["terminate_process", "isolate_host", "reset_passwords"]
                for step in trigger.workflow_steps
            ),
            "approval_required": trigger.auto_execute is False,
            "within_rate_limit": trigger.trigger_count < trigger.max_triggers_per_day
        },
        "agent_readiness": {
            agent: "configured"
            for agent in (trigger.agent_requirements or [])
        },
        "simulation_timestamp": datetime.utcnow().isoformat()
    }

    return simulation_result


@router.patch("/{trigger_id}/settings")
async def update_trigger_settings(
    trigger_id: int,
    settings: dict,
    db: AsyncSession = Depends(get_db),
    _: bool = Depends(require_api_key)
):
    """Update trigger settings (auto_execute, priority, etc.)"""
    result = await db.execute(
        select(WorkflowTrigger).where(WorkflowTrigger.id == trigger_id)
    )
    trigger = result.scalar_one_or_none()

    if not trigger:
        raise HTTPException(status_code=404, detail="Trigger not found")

    # Update allowed settings
    if "auto_execute" in settings:
        trigger.auto_execute = settings["auto_execute"]
        trigger.last_editor = settings.get("editor", "system")

    if "priority" in settings:
        trigger.priority = settings["priority"]

    if "enabled" in settings:
        trigger.enabled = settings["enabled"]
        trigger.status = "active" if settings["enabled"] else "paused"

    trigger.updated_at = datetime.utcnow()

    await db.commit()
    await db.refresh(trigger)

    return {
        "success": True,
        "message": "Settings updated successfully",
        "trigger": {
            "id": trigger.id,
            "name": trigger.name,
            "auto_execute": trigger.auto_execute,
            "priority": trigger.priority,
            "enabled": trigger.enabled,
            "status": trigger.status
        }
    }

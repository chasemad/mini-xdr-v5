import uuid
from typing import Any, Dict, List, Optional

from app.db import AsyncSessionLocal, get_db
from app.models import Incident, ResponseWorkflow
from app.visual_workflow_executor import VisualWorkflowExecutor
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter(prefix="/api/workflows", tags=["visual-workflows"])


class WorkflowGraph(BaseModel):
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]


class SaveWorkflowRequest(BaseModel):
    incident_id: Optional[int] = None
    name: str
    graph: WorkflowGraph
    trigger_config: Optional[Dict[str, Any]] = None  # New field for system triggers


class ExecuteWorkflowRequest(BaseModel):
    incident_id: Optional[int] = None
    graph: Optional[WorkflowGraph] = None
    workflow_id: Optional[int] = None


async def run_workflow_background(
    incident_id: Optional[int],
    workflow_id: Optional[int],
    nodes: List[Dict],
    edges: List[Dict],
):
    """Background task wrapper to run workflow with its own session"""
    async with AsyncSessionLocal() as session:
        executor = VisualWorkflowExecutor(session, incident_id, workflow_id)
        await executor.execute_graph(nodes, edges)


@router.post("/save")
async def save_workflow(
    request: SaveWorkflowRequest, db: AsyncSession = Depends(get_db)
):
    """Save a visual workflow to the database"""

    # Check if incident exists if provided
    if request.incident_id:
        result = await db.execute(
            select(Incident).where(Incident.id == request.incident_id)
        )
        incident = result.scalar_one_or_none()
        if not incident:
            raise HTTPException(status_code=404, detail="Incident not found")

    # Create new workflow record
    workflow = ResponseWorkflow(
        workflow_id=f"wf-{uuid.uuid4()}",
        incident_id=request.incident_id,
        playbook_name=request.name,
        visual_graph=request.graph.dict(),
        status="saved",
        steps=[],  # Legacy field, can be empty
    )

    db.add(workflow)
    await db.commit()
    await db.refresh(workflow)

    # Handle Trigger Creation (System Workflow)
    if request.trigger_config:
        from app.models import WorkflowTrigger

        # Check if trigger exists or create new
        trigger = WorkflowTrigger(
            name=f"Trigger for {request.name}",
            category="system_automation",
            enabled=True,
            auto_execute=True,
            priority="medium",
            conditions=request.trigger_config,
            playbook_name=request.name,
            workflow_steps=[],  # Using visual_graph instead
            visual_graph=request.graph.dict(),
        )
        db.add(trigger)
        await db.commit()

    return {
        "success": True,
        "workflow_id": workflow.id,
        "message": "Workflow saved successfully",
    }


@router.get("/{workflow_id}")
async def get_workflow(workflow_id: int, db: AsyncSession = Depends(get_db)):
    """Load a saved workflow"""
    result = await db.execute(
        select(ResponseWorkflow).where(ResponseWorkflow.id == workflow_id)
    )
    workflow = result.scalar_one_or_none()

    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")

    return {
        "id": workflow.id,
        "name": workflow.playbook_name,
        "graph": workflow.visual_graph,
        "status": workflow.status,
        "execution_log": workflow.execution_log,
    }


@router.get("/")
async def list_workflows(
    incident_id: Optional[int] = None, db: AsyncSession = Depends(get_db)
):
    """List all saved workflows"""
    query = select(ResponseWorkflow)

    if incident_id:
        query = query.where(ResponseWorkflow.incident_id == incident_id)
    else:
        # If no incident_id, maybe return global ones?
        # For now let's return all or just global ones if we had a flag.
        # Let's return all for now to be safe, or maybe those with incident_id IS NULL for global?
        # The handoff says "Global workflows (no incident ID required)".
        # So let's return all if no filter, or maybe just global?
        # Let's return all for now.
        pass

    query = query.order_by(ResponseWorkflow.created_at.desc())
    result = await db.execute(query)
    workflows = result.scalars().all()

    return [
        {
            "id": wf.id,
            "name": wf.playbook_name,
            "incident_id": wf.incident_id,
            "created_at": wf.created_at,
            "status": wf.status,
        }
        for wf in workflows
    ]


@router.post("/run")
async def run_workflow(
    request: ExecuteWorkflowRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """Execute a visual workflow"""

    # 1. Determine Graph Source
    graph_data = None
    workflow_id = request.workflow_id
    incident_id = request.incident_id

    if request.graph:
        # Execute ad-hoc graph
        graph_data = request.graph.dict()

        # Save it first for tracking
        workflow = ResponseWorkflow(
            workflow_id=f"wf-{uuid.uuid4()}",
            incident_id=incident_id,
            playbook_name=f"Ad-hoc Execution {incident_id or 'System'}",
            visual_graph=graph_data,
            status="running",
            steps=[],
        )
        db.add(workflow)
        await db.commit()
        await db.refresh(workflow)
        workflow_id = workflow.id

    elif request.workflow_id:
        # Execute saved workflow
        result = await db.execute(
            select(ResponseWorkflow).where(ResponseWorkflow.id == request.workflow_id)
        )
        workflow = result.scalar_one_or_none()

        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")
        graph_data = workflow.visual_graph
        workflow.status = "running"
        await db.commit()

    else:
        raise HTTPException(
            status_code=400, detail="Must provide either graph or workflow_id"
        )

    # 2. Execute in Background
    # We pass the wrapper function to background_tasks
    background_tasks.add_task(
        run_workflow_background,
        incident_id,
        workflow_id,
        graph_data["nodes"],
        graph_data["edges"],
    )

    return {"success": True, "workflow_id": workflow_id, "status": "started"}

"""
Integration Configuration API Routes

Provides CRUD operations for managing security tool integrations
and querying their capabilities for the AI orchestration system.
"""

from typing import List, Optional

from app.database_models import IntegrationConfig
from app.db import get_db
from app.services.action_orchestrator import ActionOrchestrator
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter(prefix="/api/integrations", tags=["integrations"])


# Pydantic models for API
class IntegrationConfigCreate(BaseModel):
    vendor: str
    vendor_display_name: str
    category: str
    enabled: bool = True
    priority: int = 5
    capabilities: List[str]
    config: dict = {}


class IntegrationConfigUpdate(BaseModel):
    vendor_display_name: Optional[str] = None
    enabled: Optional[bool] = None
    priority: Optional[int] = None
    capabilities: Optional[List[str]] = None
    config: Optional[dict] = None
    health_status: Optional[str] = None


class IntegrationConfigResponse(BaseModel):
    id: int
    vendor: str
    vendor_display_name: str
    category: str
    enabled: bool
    priority: int
    capabilities: List[str]
    health_status: str
    total_executions: int
    successful_executions: int
    failed_executions: int
    average_response_time_ms: Optional[float]

    class Config:
        from_attributes = True


@router.get("/", response_model=List[IntegrationConfigResponse])
async def list_integrations(
    category: Optional[str] = None,
    enabled: Optional[bool] = None,
    db: AsyncSession = Depends(get_db),
):
    """List all integration configurations, optionally filtered"""
    query = select(IntegrationConfig)

    if category:
        query = query.where(IntegrationConfig.category == category)
    if enabled is not None:
        query = query.where(IntegrationConfig.enabled == enabled)

    query = query.order_by(IntegrationConfig.priority.asc())

    result = await db.execute(query)
    integrations = result.scalars().all()

    return integrations


@router.get("/{integration_id}", response_model=IntegrationConfigResponse)
async def get_integration(integration_id: int, db: AsyncSession = Depends(get_db)):
    """Get a specific integration configuration"""
    result = await db.execute(
        select(IntegrationConfig).where(IntegrationConfig.id == integration_id)
    )
    integration = result.scalar_one_or_none()

    if not integration:
        raise HTTPException(status_code=404, detail="Integration not found")

    return integration


@router.post("/", response_model=IntegrationConfigResponse)
async def create_integration(
    integration: IntegrationConfigCreate, db: AsyncSession = Depends(get_db)
):
    """Create a new integration configuration"""
    # Check if vendor already exists
    existing = await db.execute(
        select(IntegrationConfig).where(IntegrationConfig.vendor == integration.vendor)
    )
    if existing.scalar_one_or_none():
        raise HTTPException(
            status_code=400,
            detail=f"Integration for vendor '{integration.vendor}' already exists",
        )

    db_integration = IntegrationConfig(
        vendor=integration.vendor,
        vendor_display_name=integration.vendor_display_name,
        category=integration.category,
        enabled=integration.enabled,
        priority=integration.priority,
        capabilities=integration.capabilities,
        config=integration.config,
    )

    db.add(db_integration)
    await db.commit()
    await db.refresh(db_integration)

    return db_integration


@router.patch("/{integration_id}", response_model=IntegrationConfigResponse)
async def update_integration(
    integration_id: int,
    updates: IntegrationConfigUpdate,
    db: AsyncSession = Depends(get_db),
):
    """Update an integration configuration"""
    result = await db.execute(
        select(IntegrationConfig).where(IntegrationConfig.id == integration_id)
    )
    integration = result.scalar_one_or_none()

    if not integration:
        raise HTTPException(status_code=404, detail="Integration not found")

    # Apply updates
    update_data = updates.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(integration, field, value)

    await db.commit()
    await db.refresh(integration)

    return integration


@router.delete("/{integration_id}")
async def delete_integration(integration_id: int, db: AsyncSession = Depends(get_db)):
    """Delete an integration configuration"""
    result = await db.execute(
        select(IntegrationConfig).where(IntegrationConfig.id == integration_id)
    )
    integration = result.scalar_one_or_none()

    if not integration:
        raise HTTPException(status_code=404, detail="Integration not found")

    await db.delete(integration)
    await db.commit()

    return {"success": True, "message": f"Deleted integration '{integration.vendor}'"}


@router.get("/capabilities/{intent}")
async def get_vendors_for_intent(intent: str, db: AsyncSession = Depends(get_db)):
    """Get all vendors that can handle a specific intent"""
    orchestrator = ActionOrchestrator(db)
    vendors = await orchestrator.get_vendors_for_intent(intent)

    return {"intent": intent, "available_vendors": vendors, "count": len(vendors)}


@router.get("/health/status")
async def get_health_status(db: AsyncSession = Depends(get_db)):
    """Get health status of all integrations"""
    result = await db.execute(select(IntegrationConfig))
    integrations = result.scalars().all()

    health_summary = {"healthy": 0, "degraded": 0, "offline": 0, "unknown": 0}

    for integration in integrations:
        if not integration.enabled:
            continue
        status = integration.health_status
        if status in health_summary:
            health_summary[status] += 1

    return {
        "summary": health_summary,
        "total_enabled": sum(health_summary.values()),
        "integrations": [
            {
                "vendor": i.vendor,
                "vendor_display_name": i.vendor_display_name,
                "health_status": i.health_status,
                "last_check": i.last_health_check,
                "enabled": i.enabled,
            }
            for i in integrations
        ],
    }

"""
Seamless onboarding v2 API routes
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from ..auth import get_current_user
from ..db import get_db
from ..integrations.manager import IntegrationManager
from ..models import Organization, User
from .auto_discovery import AutoDiscoveryEngine
from .smart_deployment import SmartDeploymentEngine
from .validation import OnboardingValidator

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/onboarding/v2", tags=["Onboarding V2"])
ALLOWED_PROVIDERS = {"aws", "azure", "gcp"}


# ============================================================================
# Pydantic Models
# ============================================================================


class QuickStartRequest(BaseModel):
    """Request model for quick-start onboarding"""

    provider: str  # aws, azure, gcp
    credentials: Dict[str, Any]


class IntegrationSetupRequest(BaseModel):
    """Request model for integration setup"""

    provider: str
    credentials: Dict[str, Any]


# ============================================================================
# Helper Functions
# ============================================================================


async def get_organization(user: User, db: AsyncSession) -> Organization:
    """Get user's organization"""
    from sqlalchemy import select

    stmt = select(Organization).where(Organization.id == user.organization_id)
    result = await db.execute(stmt)
    org = result.scalars().first()

    if not org:
        raise HTTPException(status_code=404, detail="Organization not found")

    return org


async def update_org_onboarding_status(org_id: int, status: str, db: AsyncSession):
    """Update organization onboarding status"""
    from datetime import datetime, timezone

    from sqlalchemy import update

    from ..models import Organization

    stmt = (
        update(Organization)
        .where(Organization.id == org_id)
        .values(
            onboarding_status=status,
            onboarding_completed_at=datetime.now(timezone.utc)
            if status == "completed"
            else None,
        )
    )
    await db.execute(stmt)
    await db.commit()


# ============================================================================
# Background Task
# ============================================================================


async def auto_discover_and_deploy(org_id: int, provider: str, db: AsyncSession):
    """
    Background task for auto-discovery and deployment

    This runs asynchronously after the quick-start endpoint returns
    """
    logger.info(f"Starting background auto-discover-and-deploy for org {org_id}")

    try:
        # Step 1: Auto-discovery
        discovery_engine = AutoDiscoveryEngine(org_id, db)
        assets = await discovery_engine.discover_cloud_assets(provider)
        logger.info(f"Discovered {len(assets)} assets for org {org_id}")

        # Step 2: Smart deployment
        deployment_engine = SmartDeploymentEngine(org_id, db)
        await deployment_engine.deploy_to_assets(assets, provider)
        logger.info(f"Deployment initiated for org {org_id}")

        # Step 3: Validation
        validator = OnboardingValidator(org_id, db)
        # Run validation after a short delay to allow agents to check in
        # For now, we'll just prepare the validator - validation will happen via the progress endpoint
        logger.info(f"Validation ready for org {org_id}")

        # Update organization onboarding status
        await update_org_onboarding_status(org_id, "completed", db)
        logger.info(f"Onboarding completed for org {org_id}")

    except Exception as e:
        logger.error(f"Auto-onboarding failed for org {org_id}: {e}")
        await update_org_onboarding_status(org_id, "failed", db)


# ============================================================================
# API Endpoints
# ============================================================================


@router.post("/quick-start")
async def quick_start_onboarding(
    request: QuickStartRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    One-click onboarding with cloud integration

    This endpoint initiates seamless onboarding by:
    1. Setting up cloud provider integration
    2. Starting auto-discovery in the background
    3. Deploying agents automatically
    4. Validating the deployment
    """
    logger.info(
        f"Quick-start onboarding requested for {request.provider} by user {current_user.email}"
    )

    # Get organization
    org = await get_organization(current_user, db)

    if request.provider not in ALLOWED_PROVIDERS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported provider. Supported providers: aws, azure, gcp.",
        )

    # Verify seamless onboarding is enabled
    if org.onboarding_flow_version != "seamless":
        raise HTTPException(
            status_code=409,
            detail="Seamless onboarding not enabled for this organization. Contact support to enable it.",
        )

    # Initialize integration manager
    integration_mgr = IntegrationManager(org.id, db)

    # Validate and store credentials
    try:
        success = await integration_mgr.setup_integration(
            request.provider, request.credentials
        )
        if not success:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to authenticate with {request.provider}",
            )
    except Exception as e:
        logger.error(f"Integration setup failed: {e}")
        raise HTTPException(
            status_code=400, detail=f"Integration setup failed: {str(e)}"
        )

    # Update org status to in_progress
    await update_org_onboarding_status(org.id, "in_progress", db)

    # Start auto-discovery and deployment in background
    background_tasks.add_task(auto_discover_and_deploy, org.id, request.provider, db)

    return {
        "status": "initiated",
        "message": f"Auto-discovery started for {request.provider}. This may take a few minutes.",
        "estimated_completion": "5-10 minutes",
        "provider": request.provider,
        "organization_id": org.id,
    }


@router.get("/progress")
async def get_onboarding_progress(
    current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)
):
    """
    Get real-time onboarding progress

    Returns:
        Combined progress from discovery, deployment, and validation
    """
    org = await get_organization(current_user, db)

    # Get discovery progress
    discovery_engine = AutoDiscoveryEngine(org.id, db)
    discovery_status = await discovery_engine.get_status()

    # Get deployment progress
    deployment_engine = SmartDeploymentEngine(org.id, db)
    deployment_status = await deployment_engine.get_status()

    # Get validation status
    validator = OnboardingValidator(org.id, db)

    # Only run validation if deployment is complete
    if deployment_status["status"] == "completed":
        await validator.validate_deployment()

    validation_status = await validator.get_status()

    # Calculate overall progress
    overall_progress = (
        discovery_status["progress"] * 0.33
        + deployment_status["progress"] * 0.33
        + validation_status["progress"] * 0.34
    )

    # Determine overall status
    if validation_status["status"] == "completed":
        overall_status = "completed"
    elif (
        discovery_status["status"] == "failed"
        or deployment_status["status"] == "failed"
    ):
        overall_status = "failed"
    elif (
        discovery_status["status"] == "discovering"
        or deployment_status["status"] == "deploying"
    ):
        overall_status = "in_progress"
    else:
        overall_status = "not_started"

    return {
        "overall_status": overall_status,
        "overall_progress": int(overall_progress),
        "discovery": discovery_status,
        "deployment": deployment_status,
        "validation": validation_status,
        "organization_id": org.id,
    }


@router.get("/validation/summary")
async def get_validation_summary(
    current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)
):
    """Get detailed validation summary"""
    org = await get_organization(current_user, db)

    validator = OnboardingValidator(org.id, db)
    summary = await validator.get_validation_summary()

    return summary


@router.get("/assets")
async def get_discovered_assets(
    provider: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get discovered cloud assets

    Args:
        provider: Optional filter by provider (aws, azure, gcp)
    """
    if provider and provider not in ALLOWED_PROVIDERS:
        raise HTTPException(status_code=400, detail="Unsupported provider filter")

    org = await get_organization(current_user, db)

    discovery_engine = AutoDiscoveryEngine(org.id, db)
    assets = await discovery_engine.get_discovered_assets(provider=provider)

    return {"total": len(assets), "provider_filter": provider, "assets": assets}


@router.post("/assets/refresh")
async def refresh_asset_discovery(
    provider: str,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Refresh asset discovery for a provider"""
    if provider not in ALLOWED_PROVIDERS:
        raise HTTPException(status_code=400, detail="Unsupported provider")

    org = await get_organization(current_user, db)

    discovery_engine = AutoDiscoveryEngine(org.id, db)

    # Run refresh in background
    background_tasks.add_task(discovery_engine.refresh_discovery, provider)

    return {
        "status": "initiated",
        "message": f"Refreshing asset discovery for {provider}",
        "provider": provider,
    }


@router.get("/deployment/summary")
async def get_deployment_summary(
    current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)
):
    """Get deployment summary"""
    org = await get_organization(current_user, db)

    deployment_engine = SmartDeploymentEngine(org.id, db)
    summary = await deployment_engine.get_deployment_summary()

    return summary


@router.post("/deployment/retry")
async def retry_failed_deployments(
    provider: str,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Retry failed agent deployments"""
    if provider not in ALLOWED_PROVIDERS:
        raise HTTPException(status_code=400, detail="Unsupported provider")

    org = await get_organization(current_user, db)

    deployment_engine = SmartDeploymentEngine(org.id, db)

    # Run retry in background
    background_tasks.add_task(deployment_engine.retry_failed_deployments, provider)

    return {
        "status": "initiated",
        "message": f"Retrying failed deployments for {provider}",
        "provider": provider,
    }


@router.get("/deployment/health")
async def get_deployment_health(
    current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)
):
    """Check health of deployed agents"""
    org = await get_organization(current_user, db)

    deployment_engine = SmartDeploymentEngine(org.id, db)
    health = await deployment_engine.check_deployment_health()

    return health


@router.get("/integrations")
async def list_integrations(
    current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)
):
    """List all configured cloud integrations"""
    org = await get_organization(current_user, db)

    integration_mgr = IntegrationManager(org.id, db)
    integrations = await integration_mgr.list_integrations()

    return {"total": len(integrations), "integrations": integrations}


@router.post("/integrations/setup")
async def setup_integration(
    request: IntegrationSetupRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Setup a new cloud integration"""
    org = await get_organization(current_user, db)

    integration_mgr = IntegrationManager(org.id, db)

    try:
        success = await integration_mgr.setup_integration(
            request.provider, request.credentials
        )
        if not success:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to setup {request.provider} integration",
            )
    except Exception as e:
        logger.error(f"Integration setup failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "status": "success",
        "message": f"{request.provider} integration configured successfully",
        "provider": request.provider,
    }


@router.delete("/integrations/{provider}")
async def remove_integration(
    provider: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Remove a cloud integration"""
    org = await get_organization(current_user, db)

    integration_mgr = IntegrationManager(org.id, db)
    success = await integration_mgr.remove_integration(provider)

    if not success:
        raise HTTPException(
            status_code=404, detail=f"Integration for {provider} not found"
        )

    return {
        "status": "success",
        "message": f"{provider} integration removed",
        "provider": provider,
    }

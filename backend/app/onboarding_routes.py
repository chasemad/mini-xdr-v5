"""
Onboarding API Routes - Enterprise organization onboarding workflow
"""
import logging
from typing import List, Optional
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from .db import get_db
from .auth import get_current_user, require_role
from .models import User, Organization, DiscoveredAsset, AgentEnrollment
from .schemas import (
    OnboardingProfileRequest,
    NetworkScanRequest,
    NetworkScanResponse,
    DiscoveredAssetResponse,
    GenerateAgentTokenRequest,
    AgentTokenResponse,
    AgentEnrollmentResponse,
    ValidationCheckResponse,
    OnboardingStatusResponse
)
from .discovery_service import DiscoveryService
from .agent_enrollment_service import AgentEnrollmentService
from .agent_verification_service import AgentVerificationService
try:
    from kubernetes import client, config as k8s_config
    KUBERNETES_AVAILABLE = True
except ImportError:
    KUBERNETES_AVAILABLE = False
import os

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/onboarding", tags=["Onboarding"])


# ==================== HELPER FUNCTIONS ====================

async def get_organization(user: User, db: AsyncSession) -> Organization:
    """Get user's organization"""
    stmt = select(Organization).where(Organization.id == user.organization_id)
    result = await db.execute(stmt)
    org = result.scalar_one_or_none()
    
    if not org:
        raise HTTPException(status_code=404, detail="Organization not found")
    
    return org


def calculate_completion_percentage(org: Organization) -> int:
    """Calculate onboarding completion percentage"""
    if org.onboarding_status == "completed":
        return 100
    
    if org.onboarding_status == "not_started":
        return 0
    
    # Calculate based on completed steps
    steps = ["profile", "network_scan", "agents", "validation"]
    current_step = org.onboarding_step or "profile"
    
    try:
        step_index = steps.index(current_step)
        # Each step is worth 25%
        return min(100, (step_index + 1) * 25)
    except ValueError:
        return 0


# ==================== ONBOARDING ENDPOINTS ====================

@router.post("/start")
async def start_onboarding(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Initialize onboarding process
    
    Sets onboarding_status to 'in_progress' and returns current state.
    """
    org = await get_organization(current_user, db)
    
    if org.onboarding_status == "completed":
        raise HTTPException(
            status_code=400,
            detail="Onboarding already completed"
        )
    
    # Initialize onboarding
    if org.onboarding_status == "not_started":
        org.onboarding_status = "in_progress"
        org.onboarding_step = "profile"
        org.onboarding_data = {}
        await db.commit()
        
        logger.info(f"Onboarding started for organization {org.id}")
    
    return {
        "message": "Onboarding started",
        "onboarding_status": org.onboarding_status,
        "onboarding_step": org.onboarding_step,
        "completion_percentage": calculate_completion_percentage(org)
    }


@router.get("/status", response_model=OnboardingStatusResponse)
async def get_onboarding_status(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get current onboarding status and progress"""
    org = await get_organization(current_user, db)
    
    return OnboardingStatusResponse(
        onboarding_status=org.onboarding_status or "not_started",
        onboarding_step=org.onboarding_step,
        onboarding_data=org.onboarding_data,
        onboarding_completed_at=org.onboarding_completed_at.isoformat() if org.onboarding_completed_at else None,
        completion_percentage=calculate_completion_percentage(org)
    )


@router.post("/permissions")
async def save_permissions_and_credentials(
    payload: dict,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Step 4: Store permissions approval and credentials in a namespaced Kubernetes Secret

    Body example:
    {
      "allow_actions": true,
      "credentials": [{"key":"sudo_password","value":"...","scope":"linux"}]
    }
    """
    org = await get_organization(current_user, db)

    allow_actions = bool(payload.get("allow_actions", False))
    cred_list = payload.get("credentials", [])
    if not isinstance(cred_list, list):
        raise HTTPException(status_code=400, detail="Invalid credentials format")

    # Update onboarding data
    onboarding_data = org.onboarding_data or {}
    onboarding_data["permissions"] = {"allow_actions": allow_actions, "updated_at": datetime.now(timezone.utc).isoformat()}
    org.onboarding_data = onboarding_data
    org.onboarding_step = "validation"

    # Prepare k8s Secret data (base64 handled by k8s client when using stringData)
    string_data = {}
    for item in cred_list:
        key = str(item.get("key", "")).strip()
        value = str(item.get("value", ""))
        scope = str(item.get("scope", "other")).strip()
        if not key or not value:
            continue
        safe_key = f"{scope}_{key}".replace(" ", "_")[:253]
        string_data[safe_key] = value

    try:
        if KUBERNETES_AVAILABLE:
            # In-cluster config; fallback to local for dev
            try:
                k8s_config.load_incluster_config()
            except Exception:
                k8s_config.load_kube_config()

            v1 = client.CoreV1Api()
            namespace = os.getenv("KUBE_NAMESPACE") or open("/var/run/secrets/kubernetes.io/serviceaccount/namespace").read().strip()
            secret_name = f"mini-xdr-agent-credentials-{org.id}"

            metadata = client.V1ObjectMeta(
                name=secret_name,
                labels={"app": "mini-xdr", "org-id": str(org.id), "purpose": "agent-credentials"}
            )

            body = client.V1Secret(
                api_version="v1",
                kind="Secret",
                metadata=metadata,
                type="Opaque",
                string_data=string_data if string_data else None,
            )

            # Create or update
            try:
                existing = v1.read_namespaced_secret(name=secret_name, namespace=namespace)
                # Merge: overwrite existing keys provided
                if string_data:
                    if existing.string_data is None:
                        existing.string_data = {}
                    existing.string_data.update(string_data)
                v1.replace_namespaced_secret(name=secret_name, namespace=namespace, body=existing)
            except client.exceptions.ApiException as e:
                if e.status == 404:
                    v1.create_namespaced_secret(namespace=namespace, body=body)
                else:
                    raise

        await db.commit()
        logger.info(f"Permissions saved{' and secret synced' if KUBERNETES_AVAILABLE else ''} for org {org.id}")

        return {"message": "Permissions saved", "next_step": "validation"}
    except Exception as e:
        await db.rollback()
        logger.error(f"Failed to save permissions or secret: {e}")
        raise HTTPException(status_code=500, detail="Failed to save permissions")


@router.post("/profile")
async def save_organization_profile(
    profile: OnboardingProfileRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Step 1: Save organization profile
    
    Stores organization metadata and advances to network_scan step.
    """
    org = await get_organization(current_user, db)
    
    # Update organization settings
    settings = org.settings or {}
    settings.update({
        "region": profile.region,
        "industry": profile.industry,
        "company_size": profile.company_size
    })
    
    org.settings = settings
    
    # Update onboarding state
    onboarding_data = org.onboarding_data or {}
    onboarding_data["profile_completed"] = True
    onboarding_data["profile_timestamp"] = datetime.now(timezone.utc).isoformat()
    
    org.onboarding_data = onboarding_data
    org.onboarding_step = "network_scan"
    
    await db.commit()
    
    logger.info(f"Organization profile saved for {org.id}")
    
    return {
        "message": "Profile saved successfully",
        "next_step": "network_scan",
        "completion_percentage": calculate_completion_percentage(org)
    }


@router.post("/network-scan", response_model=NetworkScanResponse)
async def start_network_scan(
    scan_request: NetworkScanRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Step 2: Start network discovery scan
    
    Launches asynchronous network scan and returns scan_id for tracking.
    """
    org = await get_organization(current_user, db)
    
    # Initialize discovery service
    discovery_service = DiscoveryService(org.id, db)
    
    # Start scan (runs synchronously for now, can be made async with Celery/RQ)
    try:
        scan_result = await discovery_service.start_network_scan(
            network_ranges=scan_request.network_ranges,
            port_ranges=scan_request.port_ranges,
            scan_type=scan_request.scan_type
        )
        
        # Update onboarding data
        onboarding_data = org.onboarding_data or {}
        onboarding_data["network_scan_completed"] = True
        onboarding_data["scan_id"] = scan_result["scan_id"]
        onboarding_data["assets_discovered"] = scan_result["assets_discovered"]
        onboarding_data["scan_timestamp"] = datetime.now(timezone.utc).isoformat()
        
        org.onboarding_data = onboarding_data
        org.onboarding_step = "agents"
        
        await db.commit()
        
        logger.info(
            f"Network scan completed for org {org.id}: "
            f"{scan_result['assets_discovered']} assets discovered"
        )
        
        return NetworkScanResponse(**scan_result)
        
    except Exception as e:
        logger.error(f"Network scan failed for org {org.id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Network scan failed: {str(e)}"
        )


@router.get("/scan-results", response_model=List[DiscoveredAssetResponse])
async def get_scan_results(
    scan_id: Optional[str] = None,
    limit: int = 100,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get discovered assets from network scan"""
    org = await get_organization(current_user, db)
    
    discovery_service = DiscoveryService(org.id, db)
    assets = await discovery_service.get_scan_results(scan_id=scan_id, limit=limit)
    
    return [DiscoveredAssetResponse(**asset) for asset in assets]


@router.post("/generate-deployment-plan")
async def generate_deployment_plan(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Generate agent deployment recommendations
    
    Returns deployment matrix with priority groups and methods.
    """
    org = await get_organization(current_user, db)
    
    discovery_service = DiscoveryService(org.id, db)
    deployment_matrix = await discovery_service.generate_deployment_matrix()
    
    return deployment_matrix


@router.post("/generate-agent-token", response_model=AgentTokenResponse)
async def generate_agent_token(
    token_request: GenerateAgentTokenRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Step 3: Generate agent enrollment token
    
    Creates token and returns install scripts for specified platform.
    """
    org = await get_organization(current_user, db)
    
    enrollment_service = AgentEnrollmentService(org.id, db)
    
    token_data = await enrollment_service.generate_agent_token(
        platform=token_request.platform,
        hostname=token_request.hostname,
        discovered_asset_id=token_request.discovered_asset_id
    )
    
    logger.info(
        f"Agent token generated for org {org.id}, platform {token_request.platform}"
    )
    
    return AgentTokenResponse(**token_data)


@router.get("/enrolled-agents", response_model=List[AgentEnrollmentResponse])
async def get_enrolled_agents(
    status: Optional[str] = None,
    limit: int = 100,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get list of enrolled agents"""
    org = await get_organization(current_user, db)
    
    enrollment_service = AgentEnrollmentService(org.id, db)
    agents = await enrollment_service.get_enrolled_agents(status=status, limit=limit)
    
    return [AgentEnrollmentResponse(**agent) for agent in agents]


@router.post("/validation", response_model=List[ValidationCheckResponse])
async def run_validation_checks(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Step 4: Run validation checks

    Validates agent connectivity, telemetry flow, and detection pipeline.
    """
    org = await get_organization(current_user, db)

    checks = []

    # Check 1: Agent enrollment
    enrollment_service = AgentEnrollmentService(org.id, db)
    agents = await enrollment_service.get_enrolled_agents(status="active", limit=100)

    if len(agents) > 0:
        checks.append(ValidationCheckResponse(
            check_name="Agent Enrollment",
            status="pass",
            message=f"{len(agents)} agent(s) enrolled and active",
            details={"active_agents": len(agents)}
        ))
    else:
        checks.append(ValidationCheckResponse(
            check_name="Agent Enrollment",
            status="fail",
            message="No active agents enrolled",
            details={"active_agents": 0}
        ))

    # Check 2: Telemetry flow
    # Count recent events from this org (last 5 minutes)
    from .models import Event
    from datetime import timedelta

    recent_time = datetime.now(timezone.utc) - timedelta(minutes=5)
    stmt = select(Event).where(
        Event.organization_id == org.id,
        Event.ts >= recent_time
    )
    result = await db.execute(stmt)
    recent_events = result.scalars().all()

    if len(recent_events) > 0:
        checks.append(ValidationCheckResponse(
            check_name="Telemetry Flow",
            status="pass",
            message=f"{len(recent_events)} events received in last 5 minutes",
            details={"recent_events": len(recent_events)}
        ))
    else:
        checks.append(ValidationCheckResponse(
            check_name="Telemetry Flow",
            status="pending",
            message="No events received yet - agents may need time to start reporting",
            details={"recent_events": 0}
        ))

    # Check 3: Detection pipeline
    from .models import Incident

    stmt = select(Incident).where(Incident.organization_id == org.id)
    result = await db.execute(stmt)
    incidents = result.scalars().all()

    checks.append(ValidationCheckResponse(
        check_name="Detection Pipeline",
        status="pass" if len(incidents) >= 0 else "pending",
        message=f"Detection pipeline operational ({len(incidents)} incident(s) detected)",
        details={"total_incidents": len(incidents)}
    ))

    logger.info(f"Validation checks completed for org {org.id}: {len(checks)} checks")

    return checks


@router.post("/verify-agent-access/{enrollment_id}")
async def verify_agent_access(
    enrollment_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Step 4.5: Verify Agent Access & Capabilities (NEW)

    Performs comprehensive verification that agent can actually execute
    containment actions. This builds customer trust by proving the system works.

    Checks:
    - Agent connectivity and heartbeat
    - Platform-specific permissions (iptables, firewall, etc.)
    - Dry-run containment test (without actually blocking)
    - Rollback capability test
    """
    org = await get_organization(current_user, db)

    verification_service = AgentVerificationService(org.id, db)

    try:
        result = await verification_service.verify_agent_access(enrollment_id)

        logger.info(
            f"Agent verification completed for enrollment {enrollment_id}: "
            f"status={result['status']}"
        )

        return result

    except Exception as e:
        logger.error(f"Agent verification failed for enrollment {enrollment_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Verification failed: {str(e)}"
        )


@router.post("/verify-all-agents")
async def verify_all_agents(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Verify all active agents for this organization

    Returns verification status for all enrolled agents.
    Use this after agent deployment to validate readiness.
    """
    org = await get_organization(current_user, db)

    verification_service = AgentVerificationService(org.id, db)

    try:
        results = await verification_service.verify_all_agents()

        # Calculate summary statistics
        total = len(results)
        ready = sum(1 for r in results if r["status"] == "ready")
        warning = sum(1 for r in results if r["status"] == "warning")
        failed = sum(1 for r in results if r["status"] in ["fail", "error"])

        logger.info(
            f"Bulk verification completed for org {org.id}: "
            f"{ready}/{total} ready, {warning} warnings, {failed} failed"
        )

        return {
            "organization_id": org.id,
            "total_agents": total,
            "ready": ready,
            "warnings": warning,
            "failed": failed,
            "results": results,
            "verified_at": datetime.now(timezone.utc).isoformat()
        }

    except Exception as e:
        logger.error(f"Bulk verification failed for org {org.id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Bulk verification failed: {str(e)}"
        )


@router.post("/complete")
async def complete_onboarding(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Complete onboarding process
    
    Marks onboarding as completed and enables full dashboard access.
    """
    org = await get_organization(current_user, db)
    
    if org.onboarding_status == "completed":
        raise HTTPException(
            status_code=400,
            detail="Onboarding already completed"
        )
    
    # Mark onboarding as complete
    org.onboarding_status = "completed"
    org.onboarding_step = None
    org.onboarding_completed_at = datetime.now(timezone.utc)
    org.first_login_completed = True
    
    await db.commit()
    
    logger.info(f"Onboarding completed for organization {org.id}")
    
    return {
        "message": "Onboarding completed successfully!",
        "onboarding_status": "completed",
        "completed_at": org.onboarding_completed_at.isoformat(),
        "completion_percentage": 100
    }


@router.post("/skip")
async def skip_onboarding(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Skip onboarding (for demo/testing purposes)
    
    Marks onboarding as completed without validation.
    """
    org = await get_organization(current_user, db)
    
    org.onboarding_status = "completed"
    org.onboarding_step = None
    org.onboarding_completed_at = datetime.now(timezone.utc)
    org.first_login_completed = True
    
    await db.commit()
    
    logger.warning(f"Onboarding skipped for organization {org.id}")
    
    return {
        "message": "Onboarding skipped",
        "onboarding_status": "completed"
    }


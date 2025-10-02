"""
NLP Workflow API Routes
Natural language workflow creation endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import logging
import hmac

from .db import get_db
from .nlp_workflow_parser import parse_workflow_from_natural_language, get_nlp_parser
from .models import ResponseWorkflow, Incident
from .config import settings
from sqlalchemy import select

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/workflows/nlp", tags=["nlp-workflows"])


def verify_api_key(request: Request):
    """Verify API key from request headers"""
    if not settings.api_key:
        raise HTTPException(status_code=500, detail="API key must be configured")

    api_key = request.headers.get("x-api-key")
    if not api_key:
        raise HTTPException(status_code=401, detail="Missing API key header")

    if not hmac.compare_digest(api_key, settings.api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")

    return api_key


class NLPWorkflowRequest(BaseModel):
    """Request to create workflow from natural language"""
    text: str = Field(..., description="Natural language description of desired workflow")
    incident_id: Optional[int] = Field(None, description="Associated incident ID")
    auto_execute: bool = Field(False, description="Automatically execute the workflow")


class NLPWorkflowParseResponse(BaseModel):
    """Response from NLP parsing"""
    success: bool
    confidence: float
    priority: str
    actions_count: int
    actions: List[Dict[str, Any]]
    explanation: str
    approval_required: bool
    target_ip: Optional[str] = None
    conditions: Dict[str, Any] = {}


class NLPWorkflowCreateResponse(BaseModel):
    """Response from workflow creation"""
    success: bool
    workflow_id: Optional[str] = None
    workflow_db_id: Optional[int] = None
    message: str
    explanation: str
    actions_created: int


@router.post("/parse", response_model=NLPWorkflowParseResponse)
async def parse_natural_language_workflow(
    request: NLPWorkflowRequest,
    db: AsyncSession = Depends(get_db),
    _api_key: str = Depends(verify_api_key)
):
    """
    Parse natural language into structured workflow without creating it

    This endpoint allows users to preview what workflow will be created
    from their natural language description before committing to it.

    Example requests:
    - "Block IP 192.168.1.100 and isolate the affected host"
    - "Investigate brute force attack from 10.0.0.5, then contain if confirmed"
    - "Emergency: Reset all passwords and enable MFA for compromised accounts"
    """
    try:
        # Validate incident exists if provided
        if request.incident_id:
            stmt = select(Incident).where(Incident.id == request.incident_id)
            result = await db.execute(stmt)
            incident = result.scalar_one_or_none()

            if not incident:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Incident {request.incident_id} not found"
                )

        # Parse natural language with incident context
        intent, explanation = await parse_workflow_from_natural_language(
            request.text,
            request.incident_id,
            db
        )

        return NLPWorkflowParseResponse(
            success=True,
            confidence=intent.confidence,
            priority=intent.priority,
            actions_count=len(intent.actions),
            actions=intent.actions,
            explanation=explanation,
            approval_required=intent.approval_required,
            target_ip=intent.target_ip,
            conditions=intent.conditions
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"NLP parsing failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to parse natural language: {str(e)}"
        )


@router.post("/create", response_model=NLPWorkflowCreateResponse)
async def create_workflow_from_natural_language(
    request: NLPWorkflowRequest,
    db: AsyncSession = Depends(get_db),
    _api_key: str = Depends(verify_api_key)
):
    """
    Create and optionally execute a workflow from natural language

    This endpoint combines parsing and workflow creation into one step.
    Use this when you want to immediately create the workflow.

    Set auto_execute=true to also start execution immediately.
    """
    try:
        # Validate incident exists if provided
        if request.incident_id:
            stmt = select(Incident).where(Incident.id == request.incident_id)
            result = await db.execute(stmt)
            incident = result.scalar_one_or_none()

            if not incident:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Incident {request.incident_id} not found"
                )

        # Parse natural language with incident context
        intent, explanation = await parse_workflow_from_natural_language(
            request.text,
            request.incident_id,
            db
        )

        if len(intent.actions) == 0:
            return NLPWorkflowCreateResponse(
                success=False,
                message="Could not identify any actions from your request. Please be more specific.",
                explanation=explanation,
                actions_created=0
            )

        # Create workflow
        import uuid
        workflow_id = f"nlp_{uuid.uuid4().hex[:12]}"

        workflow = ResponseWorkflow(
            workflow_id=workflow_id,
            incident_id=request.incident_id,
            playbook_name=f"NLP Workflow: {request.text[:50]}...",
            playbook_version="v1.0",
            status="pending" if not request.auto_execute else "running",
            current_step=0,
            total_steps=len(intent.actions),
            progress_percentage=0.0,
            steps=intent.actions,
            ai_confidence=intent.confidence,
            auto_executed=request.auto_execute,
            approval_required=intent.approval_required,
            auto_rollback_enabled=True,
            execution_log=[{
                "timestamp": "now",
                "event": "workflow_created",
                "source": "nlp_parser",
                "natural_language_input": request.text,
                "confidence": intent.confidence
            }]
        )

        db.add(workflow)
        await db.commit()
        await db.refresh(workflow)

        # If auto-execute is requested and no approval needed, start execution
        if request.auto_execute and not intent.approval_required:
            # Import here to avoid circular dependency
            from .advanced_response_engine import get_response_engine
            response_engine = get_response_engine()

            # Execute workflow in background
            import asyncio
            asyncio.create_task(response_engine.execute_workflow(workflow.id, db))

            message = f"Workflow created and executing with {len(intent.actions)} actions"
        elif request.auto_execute and intent.approval_required:
            message = f"Workflow created but requires approval before execution ({len(intent.actions)} actions)"
        else:
            message = f"Workflow created with {len(intent.actions)} actions, ready for review"

        return NLPWorkflowCreateResponse(
            success=True,
            workflow_id=workflow_id,
            workflow_db_id=workflow.id,
            message=message,
            explanation=explanation,
            actions_created=len(intent.actions)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"NLP workflow creation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create workflow: {str(e)}"
        )


@router.get("/examples")
async def get_nlp_examples(_api_key: str = Depends(verify_api_key)):
    """
    Get example natural language workflow requests

    Returns a list of example phrases that the NLP parser can understand,
    organized by use case category.
    """
    return {
        "examples": [
            {
                "category": "Network Containment",
                "examples": [
                    "Block IP 192.168.1.100",
                    "Block 10.0.0.5 and deploy firewall rules",
                    "Ban the attacking IP and capture network traffic",
                    "Emergency: Block IP and alert analysts"
                ]
            },
            {
                "category": "Endpoint Response",
                "examples": [
                    "Isolate the compromised host",
                    "Quarantine host and terminate suspicious processes",
                    "Isolate endpoint and check database integrity",
                    "Emergency: Isolate all affected hosts immediately"
                ]
            },
            {
                "category": "Investigation",
                "examples": [
                    "Investigate SSH brute force behavior",
                    "Hunt for similar attacks in the environment",
                    "Analyze the threat and lookup threat intelligence",
                    "Investigate incident and create forensic case"
                ]
            },
            {
                "category": "Identity & Access",
                "examples": [
                    "Reset compromised user passwords",
                    "Revoke all active sessions and enforce MFA",
                    "Disable user account and reset password",
                    "Emergency: Reset all passwords and enable MFA"
                ]
            },
            {
                "category": "Multi-Step Workflows",
                "examples": [
                    "Block the attacker, isolate host, reset passwords, and alert analysts",
                    "Investigate the incident, hunt for similar attacks, then contain if confirmed",
                    "Emergency response: Block IP, isolate hosts, reset passwords, create incident case",
                    "Full ransomware response: Isolate hosts, block network traffic, backup data, alert team"
                ]
            },
            {
                "category": "Email Security",
                "examples": [
                    "Quarantine phishing email and block sender",
                    "Block sender domain and alert security team",
                    "Quarantine all emails from this sender"
                ]
            },
            {
                "category": "Cloud Security",
                "examples": [
                    "Deploy WAF rules to block attack patterns",
                    "Enable DLP policies for sensitive data",
                    "Update security groups and deploy firewall"
                ]
            }
        ],
        "tips": [
            "Be specific about IP addresses you want to target",
            "Use action words like 'block', 'isolate', 'investigate', 'alert'",
            "Combine multiple actions by separating with 'and' or 'then'",
            "Add 'emergency' or 'critical' for high-priority workflows",
            "Reference specific threats like 'ransomware', 'brute force', 'phishing'"
        ]
    }


@router.get("/capabilities")
async def get_nlp_capabilities(_api_key: str = Depends(verify_api_key)):
    """
    Get NLP parser capabilities and supported action types

    Returns information about what the NLP parser can understand
    and what response actions it can map to.
    """
    parser = get_nlp_parser()

    return {
        "supported_actions": {
            "network": [
                "block_ip", "unblock_ip", "deploy_firewall_rules",
                "deploy_waf_rules", "capture_network_traffic"
            ],
            "endpoint": [
                "isolate_host", "un_isolate_host", "terminate_process",
                "scan_endpoint", "disable_user_account"
            ],
            "forensics": [
                "investigate_behavior", "hunt_similar_attacks",
                "threat_intel_lookup", "capture_memory_dump"
            ],
            "identity": [
                "reset_passwords", "revoke_user_sessions",
                "enforce_mfa", "disable_user_account"
            ],
            "email": [
                "quarantine_email", "block_sender"
            ],
            "data": [
                "check_database_integrity", "backup_critical_data",
                "encrypt_sensitive_data", "enable_dlp"
            ],
            "communication": [
                "alert_security_analysts", "create_incident_case"
            ]
        },
        "supported_patterns": [
            "Action + IP address (e.g., 'block 192.168.1.1')",
            "Multiple actions chained (e.g., 'block IP and isolate host')",
            "Conditional logic (e.g., 'investigate then contain if confirmed')",
            "Priority indicators (e.g., 'emergency', 'critical', 'urgent')",
            "Threat type context (e.g., 'ransomware response', 'brute force')"
        ],
        "ai_enhanced": bool(parser.openai_api_key),
        "confidence_scoring": True,
        "approval_logic": True
    }
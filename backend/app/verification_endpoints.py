"""
T-Pot Action Verification Endpoints
API endpoints for verifying agent actions on honeypot
"""

from fastapi import HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import Dict, Any
import logging

from .db import get_db
from .models import Action, AdvancedResponseAction, Incident
from .tpot_verifier import get_verifier
from .config import settings
from datetime import datetime

logger = logging.getLogger(__name__)


async def verify_incident_actions(
    incident_id: int,
    db: AsyncSession
) -> Dict[str, Any]:
    """Verify all actions for an incident"""
    
    try:
        # Get incident
        incident = (await db.execute(
            select(Incident).where(Incident.id == incident_id)
        )).scalars().first()
        
        if not incident:
            return {"success": False, "error": "Incident not found"}
        
        # Get all actions for this incident
        basic_actions = (await db.execute(
            select(Action).where(Action.incident_id == incident_id)
        )).scalars().all()
        
        advanced_actions = (await db.execute(
            select(AdvancedResponseAction).where(AdvancedResponseAction.incident_id == incident_id)
        )).scalars().all()
        
        # Initialize verifier
        verifier = get_verifier({
            "host": settings.honeypot_host,
            "ssh_port": settings.honeypot_ssh_port,
            "user": settings.honeypot_user,
            "ssh_key_path": settings.expanded_ssh_key_path
        })
        
        if not verifier:
            return {"success": False, "error": "Verifier not configured"}
        
        # Prepare actions for verification
        actions_to_verify = []
        
        # Add basic actions
        for action in basic_actions:
            actions_to_verify.append({
                "id": action.id,
                "type": "basic",
                "action": action.action,
                "params": action.params or {},
                "status": action.result,
                "created_at": action.created_at.isoformat() if action.created_at else None
            })
        
        # Add advanced actions
        for action in advanced_actions:
            actions_to_verify.append({
                "id": action.id,
                "type": "advanced",
                "action": action.action_type,
                "params": action.parameters or {},
                "status": action.status,
                "created_at": action.created_at.isoformat() if action.created_at else None
            })
        
        # Verify all actions
        verification_results = await verifier.verify_multiple_actions(actions_to_verify)
        
        # Update actions with verification results
        for result in verification_results["results"]:
            action_id = result["action_id"]
            
            # Update basic actions
            for action in basic_actions:
                if action.id == action_id and result.get("type") == "basic":
                    action.verified_on_tpot = result.get("verified", False)
                    action.tpot_verification_timestamp = datetime.utcnow()
                    action.tpot_verification_details = result
            
            # Update advanced actions
            for action in advanced_actions:
                if action.id == action_id and result.get("type") == "advanced":
                    if not hasattr(action, 'tpot_verification_details'):
                        # Add verification to result_data
                        if action.result_data is None:
                            action.result_data = {}
                        action.result_data["tpot_verification"] = result
        
        await db.commit()
        
        return {
            "success": True,
            "incident_id": incident_id,
            **verification_results
        }
        
    except Exception as e:
        logger.error(f"Action verification failed: {e}")
        return {"success": False, "error": str(e)}


async def verify_single_action(
    action_id: int,
    action_type: str,
    db: AsyncSession
) -> Dict[str, Any]:
    """Verify a single action"""
    
    try:
        # Get action
        if action_type == "basic":
            action = (await db.execute(
                select(Action).where(Action.id == action_id)
            )).scalars().first()
            
            if not action:
                return {"success": False, "error": "Action not found"}
            
            action_dict = {
                "id": action.id,
                "action": action.action,
                "params": action.params or {},
                "status": action.result
            }
        else:
            action = (await db.execute(
                select(AdvancedResponseAction).where(AdvancedResponseAction.id == action_id)
            )).scalars().first()
            
            if not action:
                return {"success": False, "error": "Action not found"}
            
            action_dict = {
                "id": action.id,
                "action": action.action_type,
                "params": action.parameters or {},
                "status": action.status
            }
        
        # Initialize verifier
        verifier = get_verifier({
            "host": settings.honeypot_host,
            "ssh_port": settings.honeypot_ssh_port,
            "user": settings.honeypot_user,
            "ssh_key_path": settings.expanded_ssh_key_path
        })
        
        if not verifier:
            return {"success": False, "error": "Verifier not configured"}
        
        # Verify action
        verification = await verifier.verify_action(action_dict)
        
        # Update action with verification
        if action_type == "basic":
            action.verified_on_tpot = verification.get("verified", False)
            action.tpot_verification_timestamp = datetime.utcnow()
            action.tpot_verification_details = verification
        else:
            if action.result_data is None:
                action.result_data = {}
            action.result_data["tpot_verification"] = verification
        
        await db.commit()
        
        return {
            "success": True,
            "action_id": action_id,
            "action_type": action_type,
            **verification
        }
        
    except Exception as e:
        logger.error(f"Single action verification failed: {e}")
        return {"success": False, "error": str(e)}


async def get_tpot_status() -> Dict[str, Any]:
    """Get current T-Pot firewall status"""
    
    try:
        # Initialize verifier
        verifier = get_verifier({
            "host": settings.honeypot_host,
            "ssh_port": settings.honeypot_ssh_port,
            "user": settings.honeypot_user,
            "ssh_key_path": settings.expanded_ssh_key_path
        })
        
        if not verifier:
            return {"success": False, "error": "Verifier not configured"}
        
        # Get active blocks
        blocks = await verifier.get_active_blocks()
        
        return blocks
        
    except Exception as e:
        logger.error(f"Failed to get T-Pot status: {e}")
        return {"success": False, "error": str(e)}



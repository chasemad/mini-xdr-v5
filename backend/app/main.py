from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional, Union
import logging
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from .config import settings
from .db import get_db, init_db, AsyncSessionLocal
from .models import Event, Incident, Action
from .detect import run_detection
from .responder import block_ip, unblock_ip
from .triager import run_triage, generate_default_triage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global scheduler for background tasks
scheduler = AsyncIOScheduler()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting Mini-XDR backend...")
    await init_db()
    scheduler.start()
    
    # Add scheduled unblock processor
    scheduler.add_job(
        process_scheduled_unblocks,
        IntervalTrigger(seconds=30),
        id="process_scheduled_unblocks",
        replace_existing=True
    )
    
    yield
    
    # Shutdown
    logger.info("Shutting down Mini-XDR backend...")
    scheduler.shutdown()


app = FastAPI(
    title="Mini-XDR",
    description="SSH Brute-Force Detection and Response System",
    version="1.2.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global auto-contain setting (runtime configurable)
auto_contain_enabled = settings.auto_contain


def _require_api_key(request: Request):
    """Require API key for protected endpoints"""
    if not settings.api_key:
        return  # No API key configured, skip check
    
    api_key = request.headers.get("x-api-key")
    if api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")


async def _recent_events_for_ip(db: AsyncSession, src_ip: str, seconds: int = 600):
    """Get recent events for an IP address"""
    since = datetime.now(timezone.utc) - timedelta(seconds=seconds)
    query = select(Event).where(
        and_(Event.src_ip == src_ip, Event.ts >= since)
    ).order_by(Event.ts.desc()).limit(200)
    
    result = await db.execute(query)
    return result.scalars().all()


@app.post("/ingest/cowrie")
async def ingest_cowrie(
    events: Union[Dict[str, Any], List[Dict[str, Any]]],
    db: AsyncSession = Depends(get_db)
):
    """
    Ingest Cowrie honeypot events
    
    Accepts either a single event or list of events
    """
    # Normalize to list
    if isinstance(events, dict):
        events = [events]
    
    stored_count = 0
    incidents_detected = []
    last_src_ip = None
    
    for event_data in events:
        try:
            # Extract event fields
            event = Event(
                src_ip=event_data.get("src_ip") or event_data.get("srcip") or event_data.get("peer_ip"),
                dst_ip=event_data.get("dst_ip") or event_data.get("dstip"),
                dst_port=event_data.get("dst_port") or event_data.get("dstport"),
                eventid=event_data.get("eventid", "unknown"),
                message=event_data.get("message"),
                raw=event_data
            )
            
            if event.src_ip:
                last_src_ip = event.src_ip
            
            db.add(event)
            stored_count += 1
            
        except Exception as e:
            logger.error(f"Failed to process event: {e}")
            continue
    
    await db.commit()
    
    # Run detection on the last source IP seen
    incident_id = None
    if last_src_ip:
        incident_id = await run_detection(db, last_src_ip)
        
        if incident_id:
            incidents_detected.append(incident_id)
            
            # Get the incident for triage
            incident = (await db.execute(
                select(Incident).where(Incident.id == incident_id)
            )).scalars().first()
            
            if incident:
                # Run triage analysis
                recent_events = await _recent_events_for_ip(db, incident.src_ip)
                triage_input = {
                    "id": incident.id,
                    "src_ip": incident.src_ip,
                    "reason": incident.reason,
                    "status": incident.status
                }
                event_summaries = [
                    {
                        "ts": e.ts.isoformat() if e.ts else None,
                        "eventid": e.eventid,
                        "message": e.message
                    }
                    for e in recent_events
                ]
                
                try:
                    triage_note = run_triage(triage_input, event_summaries)
                except Exception as e:
                    logger.error(f"Triage failed: {e}")
                    triage_note = generate_default_triage(triage_input, len(event_summaries))
                
                incident.triage_note = triage_note
                
                # Auto-contain if enabled (with 10-second temporary block for testing)
                if auto_contain_enabled:
                    try:
                        status, detail = await block_ip(incident.src_ip, duration_seconds=10)
                        action = Action(
                            incident_id=incident.id,
                            action="block",
                            result="success" if status == "success" else "failed",
                            detail=detail,
                            params={"ip": incident.src_ip, "auto": True}
                        )
                        db.add(action)
                        
                        if status == "success":
                            incident.status = "contained"
                            incident.auto_contained = True
                            logger.info(f"Auto-contained incident #{incident.id}")
                        
                    except Exception as e:
                        logger.error(f"Auto-contain failed for incident #{incident.id}: {e}")
                
                await db.commit()
    
    return {
        "stored": stored_count,
        "detected": len(incidents_detected),
        "incident_id": incident_id
    }


@app.get("/incidents")
async def list_incidents(db: AsyncSession = Depends(get_db)):
    """List all incidents in reverse chronological order"""
    query = select(Incident).order_by(Incident.created_at.desc())
    result = await db.execute(query)
    incidents = result.scalars().all()
    
    return [
        {
            "id": inc.id,
            "created_at": inc.created_at.isoformat() if inc.created_at else None,
            "src_ip": inc.src_ip,
            "reason": inc.reason,
            "status": inc.status,
            "auto_contained": inc.auto_contained,
            "triage_note": inc.triage_note
        }
        for inc in incidents
    ]


@app.get("/incidents/{inc_id}")
async def get_incident_detail(inc_id: int, db: AsyncSession = Depends(get_db)):
    """Get detailed incident information"""
    incident = (await db.execute(
        select(Incident).where(Incident.id == inc_id)
    )).scalars().first()
    
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")
    
    # Get actions
    actions_query = select(Action).where(
        Action.incident_id == inc_id
    ).order_by(Action.created_at.desc())
    actions_result = await db.execute(actions_query)
    actions = actions_result.scalars().all()
    
    # Get recent events
    recent_events = await _recent_events_for_ip(db, incident.src_ip)
    
    return {
        "id": incident.id,
        "created_at": incident.created_at.isoformat() if incident.created_at else None,
        "src_ip": incident.src_ip,
        "reason": incident.reason,
        "status": incident.status,
        "auto_contained": incident.auto_contained,
        "triage_note": incident.triage_note,
        "actions": [
            {
                "id": a.id,
                "created_at": a.created_at.isoformat() if a.created_at else None,
                "action": a.action,
                "result": a.result,
                "detail": (a.detail or "")[:400],  # Truncate long details
                "params": a.params,
                "due_at": a.due_at.isoformat() if a.due_at else None
            }
            for a in actions
        ],
        "recent_events": [
            {
                "ts": e.ts.isoformat() if e.ts else None,
                "eventid": e.eventid,
                "message": e.message
            }
            for e in recent_events
        ]
    }


@app.post("/incidents/{inc_id}/unblock")
async def unblock_incident(
    inc_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    """Manually unblock an incident"""
    _require_api_key(request)
    
    incident = (await db.execute(
        select(Incident).where(Incident.id == inc_id)
    )).scalars().first()
    
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")
    
    # Execute unblock
    status, detail = await unblock_ip(incident.src_ip)
    
    # Record action
    action = Action(
        incident_id=inc_id,
        action="unblock",
        result="success" if status == "success" else "failed",
        detail=detail,
        params={"ip": incident.src_ip, "manual": True}
    )
    db.add(action)
    
    # Update incident status if successful
    if status == "success":
        incident.status = "open"  # Or could be "dismissed"
    
    await db.commit()
    
    return {"status": status, "detail": detail}


@app.post("/incidents/{inc_id}/contain")
async def contain_incident(
    inc_id: int,
    request: Request,
    duration_seconds: int = None,  # Optional query parameter for temporary blocking
    db: AsyncSession = Depends(get_db)
):
    """Manually contain an incident with optional temporary blocking"""
    _require_api_key(request)
    
    incident = (await db.execute(
        select(Incident).where(Incident.id == inc_id)
    )).scalars().first()
    
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")
    
    # Execute block (with optional duration)
    status, detail = await block_ip(incident.src_ip, duration_seconds)
    
    # Record action
    action = Action(
        incident_id=inc_id,
        action="block",
        result="success" if status == "success" else "failed",
        detail=detail,
        params={"ip": incident.src_ip, "manual": True}
    )
    db.add(action)
    
    # Update incident status if successful
    if status == "success":
        incident.status = "contained"
    
    await db.commit()
    
    return {"status": status, "detail": detail}


@app.post("/incidents/{inc_id}/schedule_unblock")
async def schedule_unblock(
    inc_id: int,
    minutes: int,
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    """Schedule an incident to be unblocked after specified minutes"""
    _require_api_key(request)
    
    if minutes < 1 or minutes > 1440:  # Max 24 hours
        raise HTTPException(status_code=400, detail="Minutes must be between 1 and 1440")
    
    incident = (await db.execute(
        select(Incident).where(Incident.id == inc_id)
    )).scalars().first()
    
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")
    
    # Create scheduled unblock action
    due_at = datetime.now(timezone.utc) + timedelta(minutes=minutes)
    action = Action(
        incident_id=inc_id,
        action="scheduled_unblock",
        result="pending",
        detail=f"Scheduled to unblock {incident.src_ip} in {minutes} minutes",
        params={"ip": incident.src_ip, "minutes": minutes},
        due_at=due_at
    )
    db.add(action)
    await db.commit()
    
    return {
        "status": "scheduled",
        "due_at": due_at.isoformat(),
        "minutes": minutes
    }


@app.get("/settings/auto_contain")
async def get_auto_contain():
    """Get auto-contain setting"""
    return {"enabled": auto_contain_enabled}


@app.post("/settings/auto_contain")
async def set_auto_contain(enabled: bool, request: Request):
    """Set auto-contain setting"""
    _require_api_key(request)
    
    global auto_contain_enabled
    auto_contain_enabled = enabled
    
    logger.info(f"Auto-contain setting changed to: {enabled}")
    
    return {"enabled": auto_contain_enabled}


async def process_scheduled_unblocks():
    """Background task to process scheduled unblocks"""
    try:
        async with AsyncSessionLocal() as db:
            # Find pending scheduled unblocks that are due
            now = datetime.now(timezone.utc)
            query = select(Action).where(
                and_(
                    Action.action == "scheduled_unblock",
                    Action.result == "pending",
                    Action.due_at <= now
                )
            )
            
            result = await db.execute(query)
            due_actions = result.scalars().all()
            
            for action in due_actions:
                try:
                    ip = action.params.get("ip") if action.params else None
                    if not ip:
                        continue
                    
                    # Execute unblock
                    status, detail = await unblock_ip(ip)
                    
                    # Create new unblock action
                    unblock_action = Action(
                        incident_id=action.incident_id,
                        action="unblock",
                        result="success" if status == "success" else "failed",
                        detail=f"Scheduled unblock: {detail}",
                        params={"ip": ip, "scheduled": True}
                    )
                    db.add(unblock_action)
                    
                    # Mark scheduled action as done
                    action.result = "done"
                    action.detail = f"Completed: {detail}"
                    
                    # Update incident status if successful
                    if status == "success":
                        incident = (await db.execute(
                            select(Incident).where(Incident.id == action.incident_id)
                        )).scalars().first()
                        if incident:
                            incident.status = "open"
                    
                    logger.info(f"Processed scheduled unblock for IP {ip}")
                    
                except Exception as e:
                    logger.error(f"Failed to process scheduled unblock {action.id}: {e}")
                    action.result = "failed"
                    action.detail = f"Failed: {str(e)}"
            
            await db.commit()
            
    except Exception as e:
        logger.error(f"Error in scheduled unblock processor: {e}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "auto_contain": auto_contain_enabled
    }


@app.get("/test/ssh")
async def test_ssh_connectivity():
    """Test SSH connectivity to honeypot"""
    from .responder import responder
    import subprocess
    import os
    
    try:
        # First, test basic network connectivity
        ping_result = subprocess.run(
            ['ping', '-c', '1', '-W', '3000', responder.host],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        ping_status = "success" if ping_result.returncode == 0 else "failed"
        ping_detail = ping_result.stdout.strip() if ping_result.returncode == 0 else ping_result.stderr.strip()
        
        # Test SSH connectivity
        ssh_status, ssh_detail = await responder.test_connection()
        
        # Get current environment info
        env_info = {
            "PATH": os.environ.get("PATH", "Not set"),
            "USER": os.environ.get("USER", "Not set"),
            "HOME": os.environ.get("HOME", "Not set"),
            "PWD": os.environ.get("PWD", "Not set")
        }
        
        return {
            "ssh_status": ssh_status,
            "ssh_detail": ssh_detail,
            "ping_status": ping_status,
            "ping_detail": ping_detail,
            "honeypot": f"{responder.username}@{responder.host}:{responder.port}",
            "environment": env_info,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"SSH test failed with exception: {e}")
        return {
            "ssh_status": "failed",
            "ssh_detail": f"SSH test exception: {str(e)}",
            "ping_status": "error",
            "ping_detail": "Could not test ping",
            "honeypot": f"{responder.username}@{responder.host}:{responder.port}",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True
    )

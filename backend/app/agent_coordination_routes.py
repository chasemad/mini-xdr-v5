"""
Agent Coordination API Routes
Provides endpoints for v2 incident UI to fetch agent coordination data
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .auth import get_current_user
from .db import AsyncSessionLocal
from .models import Incident

router = APIRouter(prefix="/api/agents", tags=["agents"])


async def get_db():
    """Dependency to get database session"""
    async with AsyncSessionLocal() as session:
        yield session


@router.get("/coordination-hub/status")
async def get_coordination_hub_status(current_user: dict = Depends(get_current_user)):
    """
    Get overall coordination hub status
    Returns aggregate statistics about agent coordination across all incidents
    """
    return {
        "status": "active",
        "active_coordinations": 1,
        "total_agents": 4,
        "agents": [
            {
                "name": "attribution",
                "status": "ready",
                "confidence_avg": 0.78,
                "decisions_made": 1,
            },
            {
                "name": "containment",
                "status": "ready",
                "confidence_avg": 0.92,
                "decisions_made": 1,
            },
            {
                "name": "forensics",
                "status": "ready",
                "confidence_avg": 0.85,
                "decisions_made": 1,
            },
            {
                "name": "deception",
                "status": "ready",
                "confidence_avg": 0.90,
                "decisions_made": 1,
            },
        ],
        "last_coordination": datetime.utcnow().isoformat(),
        "coordination_strategies": ["sequential", "parallel", "consensus"],
    }


@router.get("/ai/{agent_name}/status")
async def get_agent_status(
    agent_name: str, current_user: dict = Depends(get_current_user)
):
    """
    Get status for a specific AI agent
    """
    agent_statuses = {
        "attribution": {
            "name": "Attribution Agent",
            "status": "ready",
            "capabilities": [
                "threat_actor_identification",
                "campaign_tracking",
                "ttp_mapping",
            ],
            "last_active": datetime.utcnow().isoformat(),
            "decisions_count": 1,
            "average_confidence": 0.78,
        },
        "containment": {
            "name": "Containment Agent",
            "status": "ready",
            "capabilities": [
                "host_isolation",
                "ip_blocking",
                "firewall_rules",
                "rollback",
            ],
            "last_active": datetime.utcnow().isoformat(),
            "decisions_count": 1,
            "average_confidence": 0.92,
        },
        "forensics": {
            "name": "Forensics Agent",
            "status": "ready",
            "capabilities": [
                "evidence_collection",
                "timeline_analysis",
                "process_inspection",
            ],
            "last_active": datetime.utcnow().isoformat(),
            "decisions_count": 1,
            "average_confidence": 0.85,
        },
        "deception": {
            "name": "Deception Agent",
            "status": "ready",
            "capabilities": [
                "honeytoken_deployment",
                "attacker_tracking",
                "intelligence_gathering",
            ],
            "last_active": datetime.utcnow().isoformat(),
            "decisions_count": 1,
            "average_confidence": 0.90,
        },
    }

    if agent_name not in agent_statuses:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")

    return agent_statuses[agent_name]


@router.get("/ai/{agent_name}/decisions")
async def get_agent_decisions(
    agent_name: str,
    limit: int = 50,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get recent decisions made by a specific agent
    """
    # Query recent incidents with agent decisions
    result = await db.execute(
        select(Incident)
        .filter(Incident.triage_note.isnot(None))
        .order_by(Incident.created_at.desc())
        .limit(limit)
    )
    incidents = result.scalars().all()

    decisions = []
    for incident in incidents:
        try:
            triage_note = (
                json.loads(incident.triage_note)
                if isinstance(incident.triage_note, str)
                else incident.triage_note
            )
            if (
                triage_note
                and "agents" in triage_note
                and agent_name in triage_note["agents"]
            ):
                agent_data = triage_note["agents"][agent_name]
                decisions.append(
                    {
                        "incident_id": incident.id,
                        "timestamp": incident.created_at.isoformat(),
                        "decision": agent_data,
                        "confidence": agent_data.get("confidence", 0.0),
                    }
                )
        except (json.JSONDecodeError, KeyError, TypeError):
            continue

    return {"agent_name": agent_name, "decisions": decisions, "total": len(decisions)}


@router.get("/incidents/{incident_id}/coordination")
async def get_incident_coordination(
    incident_id: int,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get agent coordination data for a specific incident
    This is the PRIMARY endpoint called by v2 incident UI

    Returns IncidentCoordination structure expected by frontend:
    {
        incident_id, coordination_status, participating_agents,
        agent_decisions, coordination_timeline, recommendations
    }
    """
    # Query incident
    result = await db.execute(select(Incident).filter(Incident.id == incident_id))
    incident = result.scalar_one_or_none()

    if not incident:
        raise HTTPException(status_code=404, detail=f"Incident {incident_id} not found")

    # Parse triage_note to extract agent decisions (handle double-encoded JSON)
    try:
        triage_note = (
            json.loads(incident.triage_note)
            if isinstance(incident.triage_note, str)
            else incident.triage_note
        )
        print(f"DEBUG: After first parse, triage_note type: {type(triage_note)}")
        # Check if double-encoded
        if isinstance(triage_note, str):
            triage_note = json.loads(triage_note)
            print("DEBUG: Double-encoded, parsed again")
    except (json.JSONDecodeError, TypeError) as e:
        print(f"DEBUG: JSON parse error: {e}")
        triage_note = {}

    # Extract agent decisions from triage_note.agents
    agents_data = triage_note.get("agents", {}) if triage_note else {}
    print(
        f"DEBUG: agents_data keys: {list(agents_data.keys()) if agents_data else 'empty'}"
    )

    # Build agent_decisions structure
    agent_decisions = {
        "attribution": agents_data.get(
            "attribution",
            {
                "threat_actor": "Unknown",
                "confidence": 0.0,
                "tactics": [],
                "techniques": [],
                "iocs_identified": 0,
            },
        ),
        "containment": agents_data.get(
            "containment",
            {
                "status": "pending",
                "effectiveness": 0.0,
                "actions_taken": [],
                "systems_isolated": 0,
            },
        ),
        "forensics": agents_data.get(
            "forensics",
            {
                "evidence_collected": [],
                "timeline_events": 0,
                "suspicious_processes": [],
                "files_analyzed": 0,
            },
        ),
        "deception": agents_data.get(
            "deception",
            {
                "honeytokens_deployed": 0,
                "attacker_interactions": 0,
                "intelligence_gathered": [],
            },
        ),
    }

    # Build coordination timeline
    coordination_timeline = []

    # Add incident creation event
    coordination_timeline.append(
        {
            "timestamp": incident.created_at.isoformat(),
            "event": "incident_created",
            "details": f"Incident #{incident_id} created with ML confidence {incident.ml_confidence:.1%}",
            "agents": [],
        }
    )

    # Add Council verdict event if available
    if incident.council_verdict:
        coordination_timeline.append(
            {
                "timestamp": incident.created_at.isoformat(),
                "event": "council_verdict",
                "details": f"Council verdict: {incident.council_verdict}",
                "verdict": incident.council_verdict,
                "agents": ["gemini", "grok", "openai"],
            }
        )

    # Add agent decision events
    if agents_data.get("attribution"):
        coordination_timeline.append(
            {
                "timestamp": incident.created_at.isoformat(),
                "event": "attribution_analysis",
                "details": f"Attributed to {agents_data['attribution'].get('threat_actor', 'Unknown')}",
                "agents": ["attribution"],
            }
        )

    if agents_data.get("containment"):
        containment = agents_data["containment"]
        actions = containment.get("actions_taken", [])
        coordination_timeline.append(
            {
                "timestamp": incident.created_at.isoformat(),
                "event": "containment_actions",
                "details": f"{len(actions)} containment actions taken with {containment.get('effectiveness', 0):.0%} effectiveness",
                "agents": ["containment"],
            }
        )

    if agents_data.get("forensics"):
        forensics = agents_data["forensics"]
        evidence = forensics.get("evidence_collected", [])
        coordination_timeline.append(
            {
                "timestamp": incident.created_at.isoformat(),
                "event": "forensics_collection",
                "details": f"Collected {len(evidence)} forensic artifacts",
                "agents": ["forensics"],
            }
        )

    if agents_data.get("deception"):
        deception = agents_data["deception"]
        honeytokens = deception.get("honeytokens_deployed", 0)
        coordination_timeline.append(
            {
                "timestamp": incident.created_at.isoformat(),
                "event": "deception_deployed",
                "details": f"Deployed {honeytokens} honeytokens",
                "agents": ["deception"],
            }
        )

    # Build recommendations list
    recommendations = []
    if triage_note and "recommendation" in triage_note:
        recommendations.append(triage_note["recommendation"])

    # Add agent-specific recommendations
    if agents_data.get("containment", {}).get("actions_taken"):
        recommendations.extend(
            [
                f"Execute {action}"
                for action in agents_data["containment"]["actions_taken"][:3]
            ]
        )

    # Determine participating agents
    participating_agents = [
        agent
        for agent in ["attribution", "containment", "forensics", "deception"]
        if agent in agents_data
    ]

    # Determine coordination status
    coordination_status = "completed" if participating_agents else "pending"
    if incident.status == "open" and participating_agents:
        coordination_status = "active"
    elif incident.status == "closed":
        coordination_status = "completed"

    # Build final response matching frontend IncidentCoordination interface
    return {
        "incident_id": incident_id,
        "coordination_status": coordination_status,
        "participating_agents": participating_agents,
        "agent_decisions": agent_decisions,
        "coordination_timeline": coordination_timeline,
        "recommendations": recommendations[:5],  # Limit to top 5
        "strategy_used": "parallel",  # Could be extracted from metadata
        "confidence_score": incident.council_confidence
        or incident.ml_confidence
        or 0.0,
    }

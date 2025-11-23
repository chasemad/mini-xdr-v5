"""
System Health Check Routes
Comprehensive status for all Mini-XDR components
"""

import logging
import os
from datetime import datetime
from typing import Any, Dict

import psutil
from fastapi import APIRouter, Depends

from .agent_orchestrator import get_orchestrator
from .auth import get_current_user
from .config import settings
from .ml_engine import ml_detector
from .tpot_connector import get_tpot_connector

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/health", tags=["System Health"])


@router.get("/status")
async def get_system_health():
    """Get comprehensive system health status (no auth required)"""

    # T-Pot Status
    tpot_connector = get_tpot_connector()
    tpot_status = {
        "connected": tpot_connector.is_connected,
        "host": tpot_connector.host,
        "monitoring_active": len(tpot_connector.monitoring_tasks) > 0,
        "monitored_honeypots": list(tpot_connector.monitoring_tasks.keys()),
        "tunnels_active": list(tpot_connector.tunnels.keys()),
    }

    # ML Models Status
    ml_status = {
        "models_loaded": ml_detector.models_loaded
        if hasattr(ml_detector, "models_loaded")
        else False,
        "ensemble_ready": True,  # Ensemble is always available
        "detection_active": True,
    }

    # AI Agents Status
    try:
        orchestrator = await get_orchestrator()
        agents_status = {
            "orchestrator_active": orchestrator is not None,
            "available_agents": list(orchestrator.agents.keys())
            if orchestrator
            else [],
            "agents_count": len(orchestrator.agents) if orchestrator else 0,
        }
    except Exception as e:
        logger.warning(f"Could not get orchestrator status: {e}")
        agents_status = {
            "orchestrator_active": False,
            "available_agents": [],
            "agents_count": 0,
        }

    # System Resources
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage("/")

    system_resources = {
        "cpu_percent": cpu_percent,
        "memory_percent": memory.percent,
        "memory_available_gb": round(memory.available / (1024**3), 2),
        "disk_percent": disk.percent,
        "disk_free_gb": round(disk.free / (1024**3), 2),
    }

    # Process Info
    process = psutil.Process(os.getpid())
    process_info = {
        "pid": process.pid,
        "memory_mb": round(process.memory_info().rss / (1024**2), 2),
        "cpu_percent": process.cpu_percent(interval=0.1),
        "threads": process.num_threads(),
        "uptime_seconds": (
            datetime.now() - datetime.fromtimestamp(process.create_time())
        ).total_seconds(),
    }

    # Overall Health
    overall_healthy = (
        ml_status["ensemble_ready"]
        and agents_status["orchestrator_active"]
        and cpu_percent < 90
        and memory.percent < 90
    )

    return {
        "status": "healthy" if overall_healthy else "degraded",
        "timestamp": datetime.utcnow().isoformat(),
        "components": {
            "tpot_honeypot": tpot_status,
            "ml_models": ml_status,
            "ai_agents": agents_status,
            "api": {
                "running": True,
                "host": settings.api_host,
                "port": settings.api_port,
            },
        },
        "system": {"resources": system_resources, "process": process_info},
    }


@router.get("/tpot")
async def get_tpot_health(current_user: Dict = Depends(get_current_user)):
    """Get T-Pot specific health status"""

    connector = get_tpot_connector()

    health = {
        "connected": connector.is_connected,
        "host": connector.host,
        "ssh_port": connector.ssh_port,
        "monitoring": {
            "active": len(connector.monitoring_tasks) > 0,
            "honeypots": list(connector.monitoring_tasks.keys()),
            "count": len(connector.monitoring_tasks),
        },
        "tunnels": {
            "active": list(connector.tunnels.keys()),
            "count": len(connector.tunnels),
        },
    }

    if connector.is_connected:
        # Try to get container status
        try:
            container_result = await connector.get_container_status()
            if container_result["success"]:
                health["containers"] = {
                    "total": container_result["count"],
                    "running": len(
                        [
                            c
                            for c in container_result["containers"]
                            if "Up" in c["status"]
                        ]
                    ),
                }
        except Exception as e:
            logger.warning(f"Could not get container status: {e}")

    return health


@router.get("/agents")
async def get_agents_health(current_user: Dict = Depends(get_current_user)):
    """Get AI agents health status"""

    try:
        orchestrator = await get_orchestrator()

        if not orchestrator:
            return {
                "status": "unavailable",
                "message": "Agent orchestrator not initialized",
            }

        agents_info = {}
        for agent_name, agent in orchestrator.agents.items():
            agents_info[agent_name] = {
                "available": True,
                "type": type(agent).__name__,
                "description": getattr(agent, "description", "No description"),
            }

        return {
            "status": "healthy",
            "orchestrator": {
                "active": True,
                "version": getattr(orchestrator, "version", "1.0"),
            },
            "agents": agents_info,
            "total_agents": len(agents_info),
        }

    except Exception as e:
        logger.error(f"Failed to get agents health: {e}")
        return {"status": "error", "error": str(e)}


@router.get("/ml")
async def get_ml_health(current_user: Dict = Depends(get_current_user)):
    """Get ML models health status"""

    status = {
        "ensemble_detector": {"available": True, "models": []},
        "enhanced_detector": {"available": False},
    }

    # Check if models are loaded
    if hasattr(ml_detector, "xgb_model"):
        status["ensemble_detector"]["models"].append("xgboost")
    if hasattr(ml_detector, "isolation_forest"):
        status["ensemble_detector"]["models"].append("isolation_forest")
    if hasattr(ml_detector, "autoencoder_model"):
        status["ensemble_detector"]["models"].append("autoencoder")

    # Check enhanced detector
    try:
        from .enhanced_threat_detector import enhanced_detector

        if enhanced_detector and hasattr(enhanced_detector, "model"):
            status["enhanced_detector"]["available"] = True
            status["enhanced_detector"]["model_loaded"] = (
                enhanced_detector.model is not None
            )
    except Exception:
        pass

    return status


@router.get("/mcp")
async def get_mcp_health(current_user: Dict = Depends(get_current_user)):
    """Get MCP server health status"""

    # Check if MCP TypeScript server files exist
    mcp_files = ["app/mcp_server.ts", "app/mcp_server_http.ts"]

    mcp_status = {"files_present": [], "files_missing": []}

    for file in mcp_files:
        file_path = os.path.join(os.path.dirname(__file__), "..", file)
        if os.path.exists(file_path):
            mcp_status["files_present"].append(file)
        else:
            mcp_status["files_missing"].append(file)

    return {
        "status": "configured" if mcp_status["files_present"] else "not_found",
        "mcp_servers": mcp_status,
        "note": "MCP servers run via external process managers (npx, npm)",
    }

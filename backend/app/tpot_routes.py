"""
T-Pot Honeypot Management API Routes
"""

import logging
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from .auth import get_current_user
from .db import get_db
from .tpot_connector import get_tpot_connector

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/tpot", tags=["T-Pot Honeypot"])


# Request/Response Models


class BlockIPRequest(BaseModel):
    ip_address: str


class ContainerAction(BaseModel):
    container_name: str


class ElasticsearchQuery(BaseModel):
    query: Dict[str, Any]


# Status and Info Endpoints


@router.get("/status")
async def get_tpot_status():
    """Get T-Pot connection and monitoring status (public endpoint)"""
    try:
        connector = get_tpot_connector()
        stats = await connector.get_honeypot_stats()

        return {
            "status": "connected" if stats["connected"] else "disconnected",
            "host": stats["host"],
            "monitoring_honeypots": stats.get("monitoring", []),
            "active_tunnels": stats.get("tunnels", []),
            "containers": stats.get("containers", []),
            "blocked_ips": stats.get("blocked_ips", []),
            "blocked_count": stats.get("blocked_count", 0),
        }
    except Exception as e:
        logger.error(f"Failed to get T-Pot status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reconnect")
async def reconnect_tpot(
    current_user: Dict = Depends(get_current_user), db: AsyncSession = Depends(get_db)
):
    """Manually trigger T-Pot reconnection"""
    try:
        from .db import AsyncSessionLocal

        connector = get_tpot_connector()

        # Disconnect if already connected
        if connector.is_connected:
            logger.info("Disconnecting existing T-Pot connection...")
            await connector.disconnect()

        # Attempt reconnection
        logger.info("Attempting to reconnect to T-Pot...")
        success = await connector.connect()

        if success:
            # Start monitoring tasks with db session factory
            logger.info("Starting T-Pot monitoring tasks...")
            await connector.start_monitoring(AsyncSessionLocal)

            return {
                "success": True,
                "message": "Successfully reconnected to T-Pot",
                "host": connector.host,
                "port": connector.ssh_port,
            }
        else:
            return {
                "success": False,
                "message": "Failed to reconnect to T-Pot - check credentials and network connectivity",
                "host": connector.host,
                "port": connector.ssh_port,
            }
    except Exception as e:
        logger.error(f"Failed to reconnect to T-Pot: {e}")
        raise HTTPException(status_code=500, detail=f"Reconnection failed: {str(e)}")


@router.get("/containers")
async def get_containers(current_user: Dict = Depends(get_current_user)):
    """Get status of all T-Pot honeypot containers"""
    try:
        connector = get_tpot_connector()

        if not connector.is_connected:
            raise HTTPException(status_code=503, detail="Not connected to T-Pot")

        result = await connector.get_container_status()

        if result["success"]:
            return {"containers": result["containers"], "total": result["count"]}
        else:
            raise HTTPException(
                status_code=500, detail=result.get("error", "Unknown error")
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get containers: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/attacks/recent")
async def get_recent_attacks(minutes: int = 5):
    """Get recent attacks from T-Pot Elasticsearch (public endpoint)"""
    try:
        connector = get_tpot_connector()

        if not connector.is_connected:
            raise HTTPException(status_code=503, detail="Not connected to T-Pot")

        attacks = await connector.get_recent_attacks(minutes=minutes)

        return {"attacks": attacks, "count": len(attacks), "timeframe_minutes": minutes}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get recent attacks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Firewall Management Endpoints


@router.post("/firewall/block")
async def block_ip(
    request: BlockIPRequest, current_user: Dict = Depends(get_current_user)
):
    """Block an IP address on T-Pot firewall"""
    try:
        connector = get_tpot_connector()

        if not connector.is_connected:
            raise HTTPException(status_code=503, detail="Not connected to T-Pot")

        result = await connector.block_ip(request.ip_address)

        if result["success"]:
            return {
                "success": True,
                "message": f"IP {request.ip_address} blocked successfully",
                "action": result,
            }
        else:
            raise HTTPException(
                status_code=500, detail=result.get("error", "Failed to block IP")
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to block IP: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/firewall/unblock")
async def unblock_ip(
    request: BlockIPRequest, current_user: Dict = Depends(get_current_user)
):
    """Unblock an IP address on T-Pot firewall"""
    try:
        connector = get_tpot_connector()

        if not connector.is_connected:
            raise HTTPException(status_code=503, detail="Not connected to T-Pot")

        result = await connector.unblock_ip(request.ip_address)

        if result["success"]:
            return {
                "success": True,
                "message": f"IP {request.ip_address} unblocked successfully",
                "action": result,
            }
        else:
            raise HTTPException(
                status_code=500, detail=result.get("error", "Failed to unblock IP")
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to unblock IP: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/firewall/blocks")
async def get_blocked_ips(current_user: Dict = Depends(get_current_user)):
    """Get list of currently blocked IPs"""
    try:
        connector = get_tpot_connector()

        if not connector.is_connected:
            raise HTTPException(status_code=503, detail="Not connected to T-Pot")

        result = await connector.get_active_blocks()

        if result["success"]:
            return {"blocked_ips": result["blocked_ips"], "total": result["count"]}
        else:
            raise HTTPException(
                status_code=500, detail=result.get("error", "Failed to get blocks")
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get blocked IPs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Container Management Endpoints


@router.post("/containers/stop")
async def stop_container(
    request: ContainerAction, current_user: Dict = Depends(get_current_user)
):
    """Stop a T-Pot honeypot container"""
    try:
        connector = get_tpot_connector()

        if not connector.is_connected:
            raise HTTPException(status_code=503, detail="Not connected to T-Pot")

        result = await connector.stop_honeypot_container(request.container_name)

        if result["success"]:
            return {
                "success": True,
                "message": f"Container {request.container_name} stopped",
                "action": result,
            }
        else:
            raise HTTPException(
                status_code=500, detail=result.get("error", "Failed to stop container")
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to stop container: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/containers/start")
async def start_container(
    request: ContainerAction, current_user: Dict = Depends(get_current_user)
):
    """Start a T-Pot honeypot container"""
    try:
        connector = get_tpot_connector()

        if not connector.is_connected:
            raise HTTPException(status_code=503, detail="Not connected to T-Pot")

        result = await connector.start_honeypot_container(request.container_name)

        if result["success"]:
            return {
                "success": True,
                "message": f"Container {request.container_name} started",
                "action": result,
            }
        else:
            raise HTTPException(
                status_code=500, detail=result.get("error", "Failed to start container")
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start container: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Monitoring Control Endpoints


@router.post("/monitoring/start")
async def start_monitoring(
    honeypot_types: List[str] = None, current_user: Dict = Depends(get_current_user)
):
    """Start monitoring specific or all honeypots"""
    try:
        connector = get_tpot_connector()

        if not connector.is_connected:
            raise HTTPException(status_code=503, detail="Not connected to T-Pot")

        from .db import AsyncSessionLocal

        success = await connector.start_monitoring(AsyncSessionLocal, honeypot_types)

        if success:
            return {
                "success": True,
                "message": "Monitoring started",
                "monitoring": honeypot_types or "all",
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to start monitoring")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start monitoring: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/monitoring/stop")
async def stop_monitoring(
    honeypot_type: str = None, current_user: Dict = Depends(get_current_user)
):
    """Stop monitoring specific or all honeypots"""
    try:
        connector = get_tpot_connector()
        await connector.stop_monitoring(honeypot_type)

        return {
            "success": True,
            "message": f"Monitoring stopped for {honeypot_type or 'all'}",
        }

    except Exception as e:
        logger.error(f"Failed to stop monitoring: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Elasticsearch Query Endpoint


@router.post("/elasticsearch/query")
async def query_elasticsearch(
    request: ElasticsearchQuery, current_user: Dict = Depends(get_current_user)
):
    """Query T-Pot Elasticsearch directly"""
    try:
        connector = get_tpot_connector()

        if not connector.is_connected:
            raise HTTPException(status_code=503, detail="Not connected to T-Pot")

        result = await connector.query_elasticsearch(request.query)

        if result["success"]:
            return result["data"]
        else:
            raise HTTPException(
                status_code=500, detail=result.get("error", "Query failed")
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Elasticsearch query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Connection Management


@router.post("/connect")
async def connect_to_tpot(current_user: Dict = Depends(get_current_user)):
    """Establish connection to T-Pot"""
    try:
        connector = get_tpot_connector()

        if connector.is_connected:
            return {"success": True, "message": "Already connected to T-Pot"}

        success = await connector.connect()

        if success:
            # Set up tunnels
            await connector.setup_tunnels()

            return {"success": True, "message": "Connected to T-Pot successfully"}
        else:
            raise HTTPException(status_code=503, detail="Failed to connect to T-Pot")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to connect: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/disconnect")
async def disconnect_from_tpot(current_user: Dict = Depends(get_current_user)):
    """Disconnect from T-Pot"""
    try:
        connector = get_tpot_connector()
        await connector.disconnect()

        return {"success": True, "message": "Disconnected from T-Pot"}

    except Exception as e:
        logger.error(f"Failed to disconnect: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Command Execution Endpoint (Admin only)


class CommandRequest(BaseModel):
    command: str
    timeout: int = 30


@router.post("/execute")
async def execute_command(
    request: CommandRequest, current_user: Dict = Depends(get_current_user)
):
    """Execute a command on T-Pot (admin only)"""
    try:
        connector = get_tpot_connector()

        if not connector.is_connected:
            raise HTTPException(status_code=503, detail="Not connected to T-Pot")

        # Security check - only allow specific commands
        allowed_commands = [
            "ufw",
            "docker",
            "systemctl",
            "cat",
            "tail",
            "grep",
            "ls",
            "wc",
            "echo",
        ]

        command_start = request.command.split()[0] if request.command.split() else ""
        if not any(command_start.startswith(cmd) for cmd in allowed_commands):
            raise HTTPException(
                status_code=403,
                detail=f"Command not allowed: {command_start}. Allowed: {', '.join(allowed_commands)}",
            )

        result = await connector.execute_command(request.command, request.timeout)

        return {
            "success": result["success"],
            "output": result["output"],
            "stderr": result.get("stderr", ""),
            "exit_status": result.get("exit_status", -1),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Command execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

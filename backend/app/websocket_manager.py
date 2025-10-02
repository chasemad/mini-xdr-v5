"""
WebSocket Manager for Real-Time Updates
Handles WebSocket connections and broadcasts workflow/incident updates
"""

import asyncio
import json
import logging
from typing import Set, Dict, Any, Optional
from fastapi import WebSocket, WebSocketDisconnect
import weakref

logger = logging.getLogger(__name__)

class WebSocketManager:
    def __init__(self):
        # Use weak references to avoid memory leaks
        self.active_connections: Set[WebSocket] = set()
        self.connection_data: Dict[WebSocket, Dict[str, Any]] = {}

    async def connect(self, websocket: WebSocket, client_info: Optional[Dict[str, Any]] = None):
        """Accept WebSocket connection and store client info"""
        await websocket.accept()
        self.active_connections.add(websocket)

        # Store client information for targeted broadcasts
        self.connection_data[websocket] = client_info or {}

        logger.info(f"WebSocket connection established. Total connections: {len(self.active_connections)}")

        # Send initial connection confirmation
        await self.send_personal_message(websocket, {
            "type": "connection_established",
            "message": "WebSocket connected successfully",
            "timestamp": asyncio.get_event_loop().time()
        })

    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            self.connection_data.pop(websocket, None)
            logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")

    async def send_personal_message(self, websocket: WebSocket, message: Dict[str, Any]):
        """Send message to specific WebSocket connection"""
        try:
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"Error sending personal message: {e}")
            self.disconnect(websocket)

    async def broadcast(self, message: Dict[str, Any], message_type: str = "general"):
        """Broadcast message to all connected clients"""
        if not self.active_connections:
            return

        message_data = {
            "type": message_type,
            "data": message,
            "timestamp": asyncio.get_event_loop().time()
        }

        disconnected = []
        for websocket in self.active_connections.copy():
            try:
                await websocket.send_text(json.dumps(message_data))
            except Exception as e:
                logger.error(f"Error broadcasting to client: {e}")
                disconnected.append(websocket)

        # Clean up disconnected clients
        for ws in disconnected:
            self.disconnect(ws)

    async def broadcast_workflow_update(self, workflow_data: Dict[str, Any]):
        """Broadcast workflow-specific updates"""
        await self.broadcast(workflow_data, "workflow_update")

    async def broadcast_incident_update(self, incident_data: Dict[str, Any]):
        """Broadcast incident-specific updates"""
        await self.broadcast(incident_data, "incident_update")

    async def broadcast_system_status(self, status_data: Dict[str, Any]):
        """Broadcast system status updates"""
        await self.broadcast(status_data, "system_status")

    async def broadcast_execution_progress(self, execution_data: Dict[str, Any]):
        """Broadcast real-time execution progress"""
        await self.broadcast(execution_data, "execution_progress")

    def get_connection_count(self) -> int:
        """Get current number of active connections"""
        return len(self.active_connections)

# Global WebSocket manager instance
ws_manager = WebSocketManager()
"""
WebSocket manager for real-time updates to connected clients.
"""

from typing import Dict, List
from fastapi import WebSocket
import json
import asyncio
from datetime import datetime

from app.core.events import Event


class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""

    def __init__(self):
        """Initialize connection manager."""
        self.active_connections: Dict[str, List[WebSocket]] = {}
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket, discovery_id: str):
        """
        Accept a new WebSocket connection.

        Args:
            websocket: WebSocket connection
            discovery_id: Discovery ID to subscribe to
        """
        await websocket.accept()

        async with self._lock:
            if discovery_id not in self.active_connections:
                self.active_connections[discovery_id] = []
            self.active_connections[discovery_id].append(websocket)

        # Send connection confirmation
        await websocket.send_json({
            "type": "connected",
            "discovery_id": discovery_id,
            "timestamp": datetime.utcnow().isoformat(),
        })

    async def disconnect(self, websocket: WebSocket, discovery_id: str):
        """
        Disconnect a WebSocket.

        Args:
            websocket: WebSocket connection
            discovery_id: Discovery ID
        """
        async with self._lock:
            if discovery_id in self.active_connections:
                if websocket in self.active_connections[discovery_id]:
                    self.active_connections[discovery_id].remove(websocket)

                # Clean up empty lists
                if not self.active_connections[discovery_id]:
                    del self.active_connections[discovery_id]

    async def broadcast(self, discovery_id: str, message: dict):
        """
        Broadcast a message to all connected clients for a discovery.

        Args:
            discovery_id: Discovery ID
            message: Message dictionary to send
        """
        if discovery_id not in self.active_connections:
            return

        dead_connections = []

        for websocket in self.active_connections[discovery_id]:
            try:
                await websocket.send_json(message)
            except Exception as e:
                print(f"Error sending to websocket: {e}")
                dead_connections.append(websocket)

        # Clean up dead connections
        async with self._lock:
            for websocket in dead_connections:
                if websocket in self.active_connections.get(discovery_id, []):
                    self.active_connections[discovery_id].remove(websocket)

    async def broadcast_event(self, event: Event):
        """
        Broadcast an event to all connected clients.

        Args:
            event: Event to broadcast
        """
        message = {
            "type": event.event_type,
            "discovery_id": event.discovery_id,
            "timestamp": event.timestamp.isoformat(),
            "data": event.data,
        }
        await self.broadcast(event.discovery_id, message)

    def get_connection_count(self, discovery_id: str) -> int:
        """Get number of active connections for a discovery."""
        return len(self.active_connections.get(discovery_id, []))

    def get_all_discovery_ids(self) -> List[str]:
        """Get all discovery IDs with active connections."""
        return list(self.active_connections.keys())


# Global connection manager instance
_manager: ConnectionManager = None


def get_connection_manager() -> ConnectionManager:
    """Get or create the global ConnectionManager instance."""
    global _manager
    if _manager is None:
        _manager = ConnectionManager()
    return _manager

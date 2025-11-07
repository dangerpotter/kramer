"""
WebSocket endpoints for real-time updates.
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.services.websocket_manager import get_connection_manager

router = APIRouter()


@router.websocket("/ws/{discovery_id}")
async def websocket_endpoint(websocket: WebSocket, discovery_id: str):
    """
    WebSocket endpoint for real-time discovery updates.

    Args:
        websocket: WebSocket connection
        discovery_id: Discovery ID to subscribe to
    """
    manager = get_connection_manager()

    await manager.connect(websocket, discovery_id)

    try:
        # Keep connection alive and handle client messages
        while True:
            # Wait for messages from client (e.g., ping/pong)
            data = await websocket.receive_text()

            # Echo back for keep-alive
            await websocket.send_json({
                "type": "pong",
                "message": "Connection alive",
            })

    except WebSocketDisconnect:
        await manager.disconnect(websocket, discovery_id)
    except Exception as e:
        print(f"WebSocket error: {e}")
        await manager.disconnect(websocket, discovery_id)

"""WebSocket connection manager for live dashboard event streaming."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List, Set
from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger("monitoring.websocket")


class ConnectionManager:
    """Manages active WebSocket connections to broadcast live updates to dashboard clients."""

    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        """Accept new WebSocket connection."""
        await websocket.accept()
        async with self._lock:
            self.active_connections.add(websocket)
        logger.info(f"WebSocket client connected. Active connections: {len(self.active_connections)}")

    async def disconnect(self, websocket: WebSocket) -> None:
        """Remove WebSocket connection."""
        async with self._lock:
            self.active_connections.discard(websocket)
        logger.info(f"WebSocket client disconnected. Active connections: {len(self.active_connections)}")

    async def broadcast(self, message: Dict[str, Any]) -> None:
        """Broadcast JSON payload to all active WebSocket connections."""
        if not self.active_connections:
            return

        payload = json.dumps(message)
        dead_connections: List[WebSocket] = []

        async with self._lock:
            for connection in list(self.active_connections):
                try:
                    await connection.send_text(payload)
                except Exception as e:
                    logger.debug(f"Failed to send to client: {e}")
                    dead_connections.append(connection)

            for dead in dead_connections:
                self.active_connections.discard(dead)


ws_manager = ConnectionManager()

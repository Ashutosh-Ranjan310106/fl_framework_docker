"""Main FastAPI monitoring application entry point."""

from __future__ import annotations

from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from monitoring.app.api.clients import router as clients_router
from monitoring.app.api.metrics import router as metrics_router
from monitoring.app.api.rounds import router as rounds_router
from monitoring.app.api.routes import router as core_router
from monitoring.app.database.database import init_db
from monitoring.app.services.system_monitor import system_monitor
from monitoring.app.websocket.manager import ws_manager

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [Monitoring]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("monitoring.main")

DASHBOARD_DIR = Path(__file__).resolve().parent / "dashboard"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle: Initialize database and start system monitor."""
    logger.info("Initializing monitoring SQLite database...")
    init_db()

    logger.info("Starting background system resource monitor...")
    system_monitor.start()

    yield

    logger.info("Stopping system monitor...")
    system_monitor.stop()


app = FastAPI(
    title="Flower Federated Learning Monitor",
    description="Real-time monitoring console and REST API for Flower Federated Learning infrastructure.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Routers
app.include_router(core_router)
app.include_router(clients_router)
app.include_router(rounds_router)
app.include_router(metrics_router)


# WebSocket endpoint
@app.websocket("/ws/live")
async def websocket_live_endpoint(websocket: WebSocket):
    await ws_manager.connect(websocket)
    try:
        while True:
            # Keep connection alive; handle ping/messages from frontend
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        await ws_manager.disconnect(websocket)
    except Exception as e:
        logger.debug(f"WebSocket exception: {e}")
        await ws_manager.disconnect(websocket)


# Mount static assets (CSS, JS)
app.mount("/static", StaticFiles(directory=str(DASHBOARD_DIR)), name="static")


# Serve dashboard HTML at root
@app.get("/")
def serve_dashboard():
    index_file = DASHBOARD_DIR / "index.html"
    return FileResponse(str(index_file))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("monitoring.app.main:app", host="0.0.0.0", port=8000, reload=False)

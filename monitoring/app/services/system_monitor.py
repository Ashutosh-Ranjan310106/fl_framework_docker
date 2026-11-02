"""Background service periodically monitoring host and container resource utilization."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import logging
import os
from typing import Any, Dict
import psutil

from monitoring.app.database.database import SessionLocal
from monitoring.app.database import repository
from monitoring.app.websocket.manager import ws_manager

logger = logging.getLogger("monitoring.system_monitor")


class SystemMonitorService:
    """Collects system resources using psutil and broadcasts telemetry over WebSockets."""

    def __init__(self, interval_seconds: float = 2.0):
        self.interval = interval_seconds
        self._running = False
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        if not self._running:
            self._running = True
            self._task = asyncio.create_task(self._run_loop())
            logger.info(f"SystemMonitorService started with interval={self.interval}s")

    def stop(self) -> None:
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
        logger.info("SystemMonitorService stopped.")

    async def _run_loop(self) -> None:
        while self._running:
            try:
                cpu_percent = psutil.cpu_percent(interval=None)
                virtual_mem = psutil.virtual_memory()
                ram_percent = virtual_mem.percent
                ram_used_mb = virtual_mem.used / (1024 * 1024)
                ram_total_mb = virtual_mem.total / (1024 * 1024)

                payload = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "host_cpu_percent": float(cpu_percent),
                    "host_ram_percent": float(ram_percent),
                    "host_ram_used_mb": float(round(ram_used_mb, 1)),
                    "host_ram_total_mb": float(round(ram_total_mb, 1)),
                    "containers": {},
                }

                # Persist to SQLite
                db = SessionLocal()
                try:
                    repository.add_system_resource(
                        db=db,
                        host_cpu_percent=payload["host_cpu_percent"],
                        host_ram_percent=payload["host_ram_percent"],
                        host_ram_used_mb=payload["host_ram_used_mb"],
                        host_ram_total_mb=payload["host_ram_total_mb"],
                        containers=payload["containers"],
                    )
                except Exception as e:
                    logger.debug(f"Failed to persist system resource record: {e}")
                finally:
                    db.close()

                # Broadcast via WebSocket
                await ws_manager.broadcast({
                    "type": "system_resources",
                    "data": payload,
                })

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"Error in system resource collection: {e}")

            await asyncio.sleep(self.interval)


system_monitor = SystemMonitorService()

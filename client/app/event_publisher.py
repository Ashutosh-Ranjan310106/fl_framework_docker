"""Asynchronous / non-blocking background event publisher for Flower Client."""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Any, Dict, Optional
import requests

from shared.schemas.events import BaseEvent, EventSeverity, EventType

logger = logging.getLogger("client.event_publisher")


class ClientEventPublisher:
    """Publishes client-side telemetry events to FastAPI monitoring service via background worker thread."""

    def __init__(
        self,
        monitoring_url: str = "http://monitor:8000",
        experiment_id: str = "cifar10_baseline_001",
        client_id: str = "client_001",
        queue_size: int = 1000,
        enabled: bool = True,
    ):
        self.monitoring_url = monitoring_url.rstrip("/")
        self.endpoint = f"{self.monitoring_url}/api/internal/events"
        self.experiment_id = experiment_id
        self.client_id = client_id
        self.enabled = enabled
        self._queue: queue.Queue[BaseEvent] = queue.Queue(maxsize=queue_size)
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None

        if self.enabled:
            self._start_worker()

    def _start_worker(self) -> None:
        self._worker_thread = threading.Thread(
            target=self._worker_loop, daemon=True, name=f"ClientEventPublisherWorker-{self.client_id}"
        )
        self._worker_thread.start()
        logger.info(f"ClientEventPublisher started for client {self.client_id} -> {self.endpoint}")

    def _worker_loop(self) -> None:
        session = requests.Session()
        while not self._stop_event.is_set():
            try:
                event = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            success = False
            for attempt in range(3):
                try:
                    resp = session.post(
                        self.endpoint,
                        json=event.model_dump(mode="json"),
                        headers={"Content-Type": "application/json"},
                        timeout=2.5,
                    )
                    if 200 <= resp.status_code < 300:
                        success = True
                        break
                    else:
                        logger.debug(f"Client event publish failed (HTTP {resp.status_code}): {resp.text}")
                except Exception as e:
                    logger.debug(f"Attempt {attempt+1} failed to send client event: {e}")
                    time.sleep(0.3 * (attempt + 1))

            if not success:
                logger.warning(f"Failed to deliver client event {event.event_type} after 3 attempts.")

            self._queue.task_done()

    def publish(self, event: BaseEvent) -> None:
        """Enqueue event for background delivery."""
        if not self.enabled:
            return
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            logger.warning(f"Client event queue full. Dropped event: {event.event_type}")

    def emit(
        self,
        event_type: EventType,
        round_num: Optional[int] = None,
        severity: EventSeverity = EventSeverity.INFO,
        message: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Convenience method to construct and enqueue a client event."""
        event = BaseEvent(
            event_type=event_type,
            experiment_id=self.experiment_id,
            round=round_num,
            client_id=self.client_id,
            severity=severity,
            message=message or f"Client {self.client_id}: {event_type.value}",
            data=data or {},
        )
        self.publish(event)

    def close(self, timeout: float = 2.0) -> None:
        """Wait for pending events to flush and stop the worker."""
        if not self.enabled or not self._worker_thread:
            return
        try:
            self._queue.join()
        except Exception:
            pass
        self._stop_event.set()
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=timeout)

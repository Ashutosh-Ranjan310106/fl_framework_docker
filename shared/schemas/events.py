"""Event schemas for Flower Federated Learning Infrastructure and Monitoring Service."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid

from pydantic import BaseModel, Field


class EventType(str, Enum):
    # Experiment events
    EXPERIMENT_STARTED = "experiment_started"
    EXPERIMENT_COMPLETED = "experiment_completed"
    EXPERIMENT_FAILED = "experiment_failed"
    EXPERIMENT_STOPPED = "experiment_stopped"

    # Round lifecycle events
    ROUND_STARTED = "round_started"
    MODEL_DISTRIBUTION_STARTED = "model_distribution_started"
    CLIENT_UPDATES_STARTED = "client_updates_started"
    CLIENT_UPDATE_RECEIVED = "client_update_received"
    CLIENT_UPDATE_REJECTED = "client_update_rejected"
    AGGREGATION_STARTED = "aggregation_started"
    AGGREGATION_FINISHED = "aggregation_finished"
    EVALUATION_STARTED = "evaluation_started"
    EVALUATION_FINISHED = "evaluation_finished"
    ROUND_COMPLETED = "round_completed"

    # Client lifecycle & training events
    CLIENT_REGISTERED = "client_registered"
    CLIENT_CONNECTED = "client_connected"
    CLIENT_TRAINING_STARTED = "client_training_started"
    CLIENT_TRAINING_COMPLETED = "client_training_completed"
    CLIENT_UPDATE_SENT = "client_update_sent"
    CLIENT_WAITING = "client_waiting"
    CLIENT_DISCONNECTED = "client_disconnected"
    CLIENT_ERROR = "client_error"

    # System & generic
    SYSTEM_RESOURCES = "system_resources"
    ERROR_OCCURRED = "error_occurred"
    LOG_MESSAGE = "log_message"


class EventSeverity(str, Enum):
    INFO = "INFO"
    SUCCESS = "SUCCESS"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ClientStatus(str, Enum):
    REGISTERED = "Registered"
    CONNECTED = "Connected"
    TRAINING = "Training"
    SENDING_UPDATE = "Sending Update"
    WAITING = "Waiting"
    COMPLETED = "Completed"
    DISCONNECTED = "Disconnected"
    FAILED = "Failed"


class RoundStageStatus(str, Enum):
    WAITING = "WAITING"
    IN_PROGRESS = "IN PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class BaseEvent(BaseModel):
    """Base event payload shared across server, clients, and monitor."""
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: EventType
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    experiment_id: str
    round: Optional[int] = None
    client_id: Optional[str] = None
    severity: EventSeverity = EventSeverity.INFO
    message: Optional[str] = None
    data: Dict[str, Any] = Field(default_factory=dict)


class ClientTrainingMetrics(BaseModel):
    loss: float
    accuracy: float
    samples: int
    duration_seconds: float
    epochs: int = 1
    cpu_percent: Optional[float] = None
    ram_mb: Optional[float] = None


class GlobalEvaluationMetrics(BaseModel):
    loss: float
    accuracy: float
    samples: int
    duration_seconds: Optional[float] = None


class SystemResourceUsage(BaseModel):
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    host_cpu_percent: float
    host_ram_percent: float
    host_ram_used_mb: float
    host_ram_total_mb: float
    containers: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

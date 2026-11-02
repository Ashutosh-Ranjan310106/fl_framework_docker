"""Unit tests for shared event schemas and validation."""

import pytest
from shared.schemas.events import (
    BaseEvent,
    ClientStatus,
    ClientTrainingMetrics,
    EventSeverity,
    EventType,
    GlobalEvaluationMetrics,
    SystemResourceUsage,
)


def test_base_event_serialization():
    event = BaseEvent(
        event_type=EventType.CLIENT_TRAINING_COMPLETED,
        experiment_id="cifar10_baseline_001",
        round=1,
        client_id="client_001",
        severity=EventSeverity.SUCCESS,
        message="Client client_001 training completed.",
        data={"loss": 0.45, "accuracy": 0.78, "samples": 5000, "duration_seconds": 12.4},
    )

    dumped = event.model_dump()
    assert dumped["event_type"] == "client_training_completed"
    assert dumped["round"] == 1
    assert dumped["client_id"] == "client_001"
    assert dumped["data"]["samples"] == 5000


def test_client_training_metrics():
    metrics = ClientTrainingMetrics(
        loss=0.35,
        accuracy=0.88,
        samples=5000,
        duration_seconds=10.5,
        epochs=2,
    )
    assert metrics.loss == 0.35
    assert metrics.accuracy == 0.88
    assert metrics.epochs == 2


def test_system_resource_usage():
    usage = SystemResourceUsage(
        host_cpu_percent=25.5,
        host_ram_percent=60.2,
        host_ram_used_mb=9600.0,
        host_ram_total_mb=16000.0,
    )
    assert usage.host_cpu_percent == 25.5
    assert usage.host_ram_percent == 60.2

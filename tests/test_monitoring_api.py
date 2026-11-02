"""Integration tests for FastAPI Monitoring Service REST API."""

import uuid
import pytest
from fastapi.testclient import TestClient

from monitoring.app.main import app
from monitoring.app.database.database import init_db
from shared.schemas.events import BaseEvent, EventSeverity, EventType

init_db()
client = TestClient(app)


def test_health_check():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_experiment_endpoints():
    response = client.get("/api/experiment/current")
    assert response.status_code == 200
    data = response.json()
    assert "id" in data


def test_event_ingestion_and_retrieval():
    test_id = f"test_event_{uuid.uuid4().hex[:8]}"
    event_payload = {
        "event_id": test_id,
        "event_type": "client_training_completed",
        "experiment_id": "cifar10_baseline_001",
        "round": 1,
        "client_id": "client_001",
        "severity": "SUCCESS",
        "message": "Testing client event ingestion",
        "data": {
            "loss": 0.55,
            "accuracy": 0.72,
            "samples": 5000,
            "duration_seconds": 8.5,
        },
    }

    # Ingest event
    post_res = client.post("/api/internal/events", json=event_payload)
    assert post_res.status_code == 202

    # Verify event appears in /api/events
    get_res = client.get("/api/events?client_id=client_001")
    assert get_res.status_code == 200
    events = get_res.json()
    assert any(e["event_id"] == test_id for e in events)

    # Verify client appears in /api/clients
    client_res = client.get("/api/clients/client_001")
    assert client_res.status_code == 200
    client_data = client_res.json()
    assert client_data["client_id"] == "client_001"
    assert client_data["samples"] == 5000


def test_metrics_endpoints():
    res = client.get("/api/metrics/global")
    assert res.status_code == 200
    assert isinstance(res.json(), list)

    res_sys = client.get("/api/system/resources")
    assert res_sys.status_code == 200
    assert isinstance(res_sys.json(), list)

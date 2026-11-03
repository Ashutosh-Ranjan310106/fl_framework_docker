"""Tests for ReportGenerator and Reports & Volume Logs REST API endpoints."""

import os
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from monitoring.app.main import app
from server.app.report_generator import ReportGenerator

client = TestClient(app)


def test_report_generator_lifecycle(tmp_path):
    generator = ReportGenerator(
        experiment_id="test_exp_001",
        experiment_name="Test Experiment",
        num_rounds=2,
        strategy="FedAvg",
        results_dir=tmp_path,
    )

    # Record Round 1
    generator.record_round(
        round_num=1,
        loss=1.85,
        accuracy=0.35,
        samples=10000,
        duration_seconds=5.2,
        participating_clients=2,
        client_metrics=[
            {"client_id": "client_001", "samples": 5000, "loss": 1.88, "accuracy": 0.33},
            {"client_id": "client_002", "samples": 5000, "loss": 1.82, "accuracy": 0.37},
        ],
    )

    # Record Checkpoint
    ckpt_path = str(tmp_path / "models" / "test_exp_001_round_001.pt")
    generator.record_checkpoint(ckpt_path)

    # Record Round 2
    generator.record_round(
        round_num=2,
        loss=1.42,
        accuracy=0.52,
        samples=10000,
        duration_seconds=5.0,
        participating_clients=2,
        client_metrics=[
            {"client_id": "client_001", "samples": 5000, "loss": 1.45, "accuracy": 0.50},
            {"client_id": "client_002", "samples": 5000, "loss": 1.39, "accuracy": 0.54},
        ],
    )

    files = generator.generate_all_reports(status="COMPLETED")

    assert Path(files["json"]).exists()
    assert Path(files["markdown"]).exists()
    assert Path(files["csv"]).exists()

    # Verify JSON content
    data = generator.to_dict(status="COMPLETED")
    assert data["experiment_id"] == "test_exp_001"
    assert data["status"] == "COMPLETED"
    assert data["final_accuracy"] == 0.52
    assert len(data["rounds"]) == 2
    assert len(data["checkpoints"]) == 1

    # Verify Markdown contains tables
    md_content = Path(files["markdown"]).read_text(encoding="utf-8")
    assert "Global Accuracy & Loss Progression" in md_content
    assert "client_001" in md_content
    assert "Round 2" in md_content


def test_reports_and_logs_api(tmp_path, monkeypatch):
    monkeypatch.setenv("RESULTS_DIR", str(tmp_path))
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))

    # Create dummy log file
    log_dir = tmp_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    server_log = log_dir / "server.log"
    server_log.write_text(
        "[2026-11-02 10:00:00] [INFO] [Server-server_app]: Starting Flower Server\n"
        "[2026-11-02 10:00:05] [INFO] [Server-strategy]: Round 1 completed: Loss=1.85, Accuracy=0.35\n"
        "[2026-11-02 10:00:10] [INFO] [Server-strategy]: Round 2 completed: Loss=1.42, Accuracy=0.52\n",
        encoding="utf-8",
    )

    # Create dummy report
    gen = ReportGenerator(
        experiment_id="api_test_run",
        experiment_name="API Test Run",
        num_rounds=2,
        results_dir=tmp_path,
    )
    gen.record_round(round_num=1, loss=1.85, accuracy=0.35, samples=10000)
    gen.generate_all_reports(status="COMPLETED")

    # Test /api/reports list
    res_list = client.get("/api/reports")
    assert res_list.status_code == 200
    reports = res_list.json()
    assert any(r["experiment_id"] == "api_test_run" for r in reports)

    # Test /api/reports/{exp_id} detail
    res_detail = client.get("/api/reports/api_test_run")
    assert res_detail.status_code == 200
    detail = res_detail.json()
    assert detail["report"]["experiment_id"] == "api_test_run"
    assert "markdown" in detail
    assert "Global Accuracy & Loss Progression" in detail["markdown"]

    # Test /api/logs
    res_logs = client.get("/api/logs?limit=50")
    assert res_logs.status_code == 200
    log_data = res_logs.json()
    assert len(log_data["lines"]) >= 3
    assert any("Starting Flower Server" in line for line in log_data["lines"])

    # Test /api/logs with search filter
    res_search = client.get("/api/logs?search=Round 2")
    assert res_search.status_code == 200
    search_data = res_search.json()
    assert len(search_data["lines"]) == 1
    assert "Round 2 completed" in search_data["lines"][0]

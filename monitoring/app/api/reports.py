"""FastAPI router for Run Reports and Persistent Volume Logs."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query

router = APIRouter(prefix="/api", tags=["Reports & Logs"])
logger = logging.getLogger("monitoring.reports_api")


def _get_results_dir() -> Path:
    env_dir = os.getenv("RESULTS_DIR")
    if env_dir:
        return Path(env_dir)
    # Check default docker and local paths
    for p in [Path("/app/results"), Path("results")]:
        if p.exists():
            return p
    return Path("results")


def _get_log_dir() -> Path:
    env_dir = os.getenv("LOG_DIR")
    if env_dir:
        return Path(env_dir)
    for p in [Path("/app/logs"), Path("logs")]:
        if p.exists():
            return p
    return Path("logs")


@router.get("/reports", response_model=List[Dict[str, Any]])
def list_run_reports():
    """List all saved federated learning run reports from the volume."""
    results_dir = _get_results_dir()
    reports_dir = results_dir / "run_reports"
    reports: List[Dict[str, Any]] = []

    if not reports_dir.exists():
        return reports

    for json_file in sorted(reports_dir.glob("*_report.json"), reverse=True):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                reports.append({
                    "experiment_id": data.get("experiment_id", json_file.stem.replace("_report", "")),
                    "experiment_name": data.get("experiment_name", "FL Run"),
                    "status": data.get("status", "COMPLETED"),
                    "final_accuracy": data.get("final_accuracy", 0.0),
                    "final_loss": data.get("final_loss", 0.0),
                    "completed_rounds": data.get("completed_rounds", len(data.get("rounds", []))),
                    "start_time": data.get("start_time"),
                    "end_time": data.get("end_time"),
                    "has_markdown": (reports_dir / f"{data.get('experiment_id')}_report.md").exists(),
                    "has_csv": (reports_dir / f"{data.get('experiment_id')}_metrics.csv").exists(),
                })
        except Exception as e:
            logger.debug(f"Failed to read report {json_file}: {e}")

    return reports


@router.get("/reports/{experiment_id}")
def get_run_report(experiment_id: str):
    """Retrieve full JSON and Markdown details of a run report."""
    results_dir = _get_results_dir()
    reports_dir = results_dir / "run_reports"

    json_file = reports_dir / f"{experiment_id}_report.json"
    md_file = reports_dir / f"{experiment_id}_report.md"
    csv_file = reports_dir / f"{experiment_id}_metrics.csv"

    if not json_file.exists():
        # Check if latest_run_report matches
        latest_json = results_dir / "latest_run_report.json"
        if latest_json.exists():
            try:
                with open(latest_json, "r", encoding="utf-8") as f:
                    latest_data = json.load(f)
                    if latest_data.get("experiment_id") == experiment_id:
                        json_file = latest_json
                        md_file = results_dir / "latest_run_report.md"
            except Exception:
                pass

    if not json_file.exists():
        raise HTTPException(status_code=404, detail=f"Run report for '{experiment_id}' not found")

    with open(json_file, "r", encoding="utf-8") as f:
        report_data = json.load(f)

    markdown_content = ""
    if md_file.exists():
        with open(md_file, "r", encoding="utf-8") as f:
            markdown_content = f.read()

    return {
        "report": report_data,
        "markdown": markdown_content,
        "csv_available": csv_file.exists(),
    }


@router.get("/logs")
def get_server_logs(
    limit: int = Query(default=150, ge=10, le=1000),
    search: Optional[str] = None,
    source: str = "server",
):
    """
    Read real-time lines from volume-mounted logs (server or client).
    Allows monitoring window to display actual process logs stored in volume.
    """
    log_dir = _get_log_dir()
    log_file = log_dir / "server.log"
    if not log_file.exists():
        # Check subdirectories e.g. logs/server/server.log or stdout.log
        candidates = [
            log_dir / "server" / "server.log",
            log_dir / "server" / "stdout.log",
            log_dir / "server.log",
            log_dir / "monitoring.log",
        ]
        for c in candidates:
            if c.exists():
                log_file = c
                break

    if not log_file.exists():
        return {
            "source": source,
            "log_path": str(log_file),
            "lines": ["Waiting for server logs to be written to volume..."],
            "total_lines": 0,
        }

    try:
        with open(log_file, "r", encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()

        if search:
            search_lower = search.lower()
            filtered = [line.rstrip() for line in all_lines if search_lower in line.lower()]
        else:
            filtered = [line.rstrip() for line in all_lines]

        recent = filtered[-limit:]
        return {
            "source": source,
            "log_path": str(log_file),
            "lines": recent,
            "total_lines": len(all_lines),
        }
    except Exception as e:
        logger.error(f"Error reading log file {log_file}: {e}")
        return {
            "source": source,
            "log_path": str(log_file),
            "lines": [f"Error reading log file: {e}"],
            "total_lines": 0,
        }

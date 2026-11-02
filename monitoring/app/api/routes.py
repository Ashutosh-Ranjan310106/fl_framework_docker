"""Core FastAPI routes including health, experiment lifecycle, and internal event ingestion."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from monitoring.app.database.database import get_db
from monitoring.app.database import repository
from monitoring.app.services.event_processor import process_and_dispatch_event
from shared.schemas.events import BaseEvent

router = APIRouter(prefix="/api", tags=["Core"])


@router.get("/health")
def health_check():
    """Health check endpoint for Docker Compose and monitoring."""
    return {"status": "healthy", "service": "fl-monitoring-service"}


@router.get("/experiment/current", response_model=Optional[Dict[str, Any]])
def get_current_experiment(db: Session = Depends(get_db)):
    """Retrieve metadata and status of the current or most recent experiment."""
    exp = repository.get_current_experiment(db)
    if not exp:
        return {
            "id": "cifar10_baseline_001",
            "name": "CIFAR10 Flower Baseline",
            "status": "IDLE",
            "strategy": "fedavg",
            "num_rounds": 3,
            "start_time": None,
            "end_time": None,
            "config": {},
        }
    return exp


@router.get("/experiments", response_model=List[Dict[str, Any]])
def list_experiments(db: Session = Depends(get_db)):
    """List historical federated learning experiments."""
    return repository.get_all_experiments(db)


@router.get("/events", response_model=List[Dict[str, Any]])
def list_events(
    experiment_id: Optional[str] = None,
    client_id: Optional[str] = None,
    round_number: Optional[int] = None,
    event_type: Optional[str] = None,
    severity: Optional[str] = None,
    limit: int = Query(default=200, ge=1, le=1000),
    db: Session = Depends(get_db),
):
    """Retrieve chronological event feed with optional filtering."""
    return repository.get_events(
        db,
        experiment_id=experiment_id,
        client_id=client_id,
        round_number=round_number,
        event_type=event_type,
        severity=severity,
        limit=limit,
    )


@router.post("/internal/events", status_code=status.HTTP_202_ACCEPTED)
async def ingest_event(event: BaseEvent, db: Session = Depends(get_db)):
    """
    Internal event ingestion route used by Flower server and client processes.
    Validates Pydantic schema, persists to SQLite, and broadcasts via WebSockets.
    """
    result = await process_and_dispatch_event(event, db)
    return result

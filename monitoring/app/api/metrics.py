"""FastAPI router for training and system metrics endpoints."""

from __future__ import annotations

from typing import Any, Dict, List
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from monitoring.app.database.database import get_db
from monitoring.app.database import repository

router = APIRouter(prefix="/api", tags=["Metrics"])


@router.get("/metrics/global", response_model=List[Dict[str, Any]])
def get_global_metrics(experiment_id: str | None = None, db: Session = Depends(get_db)):
    """Retrieve global evaluation metrics over rounds."""
    if not experiment_id:
        curr = repository.get_current_experiment(db)
        if curr:
            experiment_id = curr["id"]
    return repository.get_global_metrics(db, experiment_id=experiment_id)


@router.get("/metrics/clients/{client_id}", response_model=List[Dict[str, Any]])
def get_client_metrics(client_id: str, experiment_id: str | None = None, db: Session = Depends(get_db)):
    """Retrieve training loss and accuracy history for a specific client."""
    if not experiment_id:
        curr = repository.get_current_experiment(db)
        if curr:
            experiment_id = curr["id"]
    return repository.get_client_metrics(db, client_id=client_id, experiment_id=experiment_id)


@router.get("/system/resources", response_model=List[Dict[str, Any]])
def get_system_resources(limit: int = 30, db: Session = Depends(get_db)):
    """Retrieve system resource utilization history (CPU, RAM)."""
    return repository.get_latest_system_resources(db, limit=limit)

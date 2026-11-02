"""FastAPI router for client monitoring endpoints."""

from __future__ import annotations

from typing import Any, Dict, List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from monitoring.app.database.database import get_db
from monitoring.app.database import repository

router = APIRouter(prefix="/api/clients", tags=["Clients"])


@router.get("", response_model=List[Dict[str, Any]])
def list_clients(experiment_id: str | None = None, db: Session = Depends(get_db)):
    """List all federated learning clients and their current status."""
    return repository.get_all_clients(db, experiment_id=experiment_id)


@router.get("/{client_id}", response_model=Dict[str, Any])
def get_client_detail(client_id: str, db: Session = Depends(get_db)):
    """Get detailed telemetry, dataset size, and status of an individual client."""
    client = repository.get_client(db, client_id)
    if not client:
        raise HTTPException(status_code=404, detail=f"Client '{client_id}' not found")

    metrics = repository.get_client_metrics(db, client_id)
    events = repository.get_events(db, client_id=client_id, limit=50)

    return {
        **client,
        "metrics_history": metrics,
        "recent_events": events,
    }

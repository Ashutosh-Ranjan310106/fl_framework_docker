"""FastAPI router for FL round monitoring endpoints."""

from __future__ import annotations

from typing import Any, Dict, List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from monitoring.app.database.database import get_db
from monitoring.app.database import repository

router = APIRouter(prefix="/api/rounds", tags=["Rounds"])


@router.get("", response_model=List[Dict[str, Any]])
def list_rounds(experiment_id: str | None = None, db: Session = Depends(get_db)):
    """List all federated rounds and their lifecycle status."""
    return repository.get_all_rounds(db, experiment_id=experiment_id)


@router.get("/{round_id}", response_model=Dict[str, Any])
def get_round_detail(round_id: int, experiment_id: str | None = None, db: Session = Depends(get_db)):
    """Get detailed information on a specific round including stage progress."""
    round_record = repository.get_round(db, round_number=round_id, experiment_id=experiment_id)
    if not round_record:
        raise HTTPException(status_code=404, detail=f"Round {round_id} not found")

    events = repository.get_events(db, round_number=round_id, limit=50)
    return {
        **round_record,
        "events": events,
    }

"""Database initialization and session handling with SQLite."""

from __future__ import annotations

import os
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from monitoring.app.database.models import Base

def _resolve_db_path() -> str:
    env_path = os.getenv("MONITORING_DB_PATH")
    if env_path:
        return env_path
    if Path("/app/results").exists():
        return "/app/results/monitoring.db"
    if Path("results").exists():
        return "results/monitoring.db"
    return "monitoring/data/monitoring.db"


DATABASE_PATH = _resolve_db_path()
Path(DATABASE_PATH).parent.mkdir(parents=True, exist_ok=True)

SQLALCHEMY_DATABASE_URL = f"sqlite:///{DATABASE_PATH}"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db() -> None:
    """Create all tables in SQLite database if they don't exist."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency for database sessions."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

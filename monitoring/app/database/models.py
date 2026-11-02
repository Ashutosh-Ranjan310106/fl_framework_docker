"""SQLAlchemy database models for persistence of FL telemetry and metrics."""

from __future__ import annotations

from datetime import datetime, timezone
from sqlalchemy import Column, Float, Integer, String, Text, Index
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class ExperimentRecord(Base):
    __tablename__ = "experiments"

    id = Column(String(100), primary_key=True)
    name = Column(String(200), nullable=False)
    status = Column(String(50), default="IDLE", nullable=False)
    strategy = Column(String(50), default="fedavg")
    num_rounds = Column(Integer, default=3)
    start_time = Column(String(50), default=lambda: datetime.now(timezone.utc).isoformat())
    end_time = Column(String(50), nullable=True)
    config_json = Column(Text, default="{}")


class ClientRecord(Base):
    __tablename__ = "clients"

    client_id = Column(String(100), primary_key=True)
    experiment_id = Column(String(100), nullable=False, index=True)
    role = Column(String(50), default="honest")
    status = Column(String(50), default="Registered")
    current_round = Column(Integer, default=0)
    samples = Column(Integer, default=0)
    last_loss = Column(Float, nullable=True)
    last_accuracy = Column(Float, nullable=True)
    cpu_percent = Column(Float, nullable=True)
    ram_mb = Column(Float, nullable=True)
    last_seen = Column(String(50), default=lambda: datetime.now(timezone.utc).isoformat())


class RoundRecord(Base):
    __tablename__ = "rounds"

    id = Column(Integer, primary_key=True, autoincrement=True)
    experiment_id = Column(String(100), nullable=False, index=True)
    round_number = Column(Integer, nullable=False, index=True)
    status = Column(String(50), default="IN PROGRESS")
    model_distribution_status = Column(String(50), default="WAITING")
    client_training_status = Column(String(50), default="WAITING")
    update_collection_status = Column(String(50), default="WAITING")
    aggregation_status = Column(String(50), default="WAITING")
    global_evaluation_status = Column(String(50), default="WAITING")
    participating_clients = Column(Integer, default=0)
    loss = Column(Float, nullable=True)
    accuracy = Column(Float, nullable=True)
    duration_seconds = Column(Float, nullable=True)
    start_time = Column(String(50), default=lambda: datetime.now(timezone.utc).isoformat())
    end_time = Column(String(50), nullable=True)

    __table_args__ = (
        Index("idx_round_exp_num", "experiment_id", "round_number", unique=True),
    )


class ClientMetricRecord(Base):
    __tablename__ = "client_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    experiment_id = Column(String(100), nullable=False, index=True)
    client_id = Column(String(100), nullable=False, index=True)
    round_number = Column(Integer, nullable=False, index=True)
    loss = Column(Float, nullable=False)
    accuracy = Column(Float, nullable=False)
    samples = Column(Integer, default=0)
    duration_seconds = Column(Float, default=0.0)
    epochs = Column(Integer, default=1)
    cpu_percent = Column(Float, nullable=True)
    ram_mb = Column(Float, nullable=True)
    timestamp = Column(String(50), default=lambda: datetime.now(timezone.utc).isoformat(), index=True)


class GlobalMetricRecord(Base):
    __tablename__ = "global_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    experiment_id = Column(String(100), nullable=False, index=True)
    round_number = Column(Integer, nullable=False, index=True)
    loss = Column(Float, nullable=False)
    accuracy = Column(Float, nullable=False)
    samples = Column(Integer, default=0)
    duration_seconds = Column(Float, default=0.0)
    timestamp = Column(String(50), default=lambda: datetime.now(timezone.utc).isoformat(), index=True)


class EventRecord(Base):
    __tablename__ = "events"

    event_id = Column(String(100), primary_key=True)
    event_type = Column(String(100), nullable=False, index=True)
    experiment_id = Column(String(100), nullable=False, index=True)
    round_number = Column(Integer, nullable=True, index=True)
    client_id = Column(String(100), nullable=True, index=True)
    severity = Column(String(20), default="INFO")
    message = Column(Text, nullable=False)
    data_json = Column(Text, default="{}")
    timestamp = Column(String(50), default=lambda: datetime.now(timezone.utc).isoformat(), index=True)


class SystemResourceRecord(Base):
    __tablename__ = "system_resources"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(String(50), default=lambda: datetime.now(timezone.utc).isoformat(), index=True)
    host_cpu_percent = Column(Float, nullable=False)
    host_ram_percent = Column(Float, nullable=False)
    host_ram_used_mb = Column(Float, nullable=False)
    host_ram_total_mb = Column(Float, nullable=False)
    containers_json = Column(Text, default="{}")

"""Database repository operations for experiments, clients, rounds, metrics, and events."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import desc

from monitoring.app.database.models import (
    ClientMetricRecord,
    ClientRecord,
    EventRecord,
    ExperimentRecord,
    GlobalMetricRecord,
    RoundRecord,
    SystemResourceRecord,
)
from shared.schemas.events import BaseEvent


def get_current_experiment(db: Session) -> Optional[Dict[str, Any]]:
    record = db.query(ExperimentRecord).order_by(desc(ExperimentRecord.start_time)).first()
    if not record:
        return None
    return {
        "id": record.id,
        "name": record.name,
        "status": record.status,
        "strategy": record.strategy,
        "num_rounds": record.num_rounds,
        "start_time": record.start_time,
        "end_time": record.end_time,
        "config": json.loads(record.config_json or "{}"),
    }


def get_all_experiments(db: Session) -> List[Dict[str, Any]]:
    records = db.query(ExperimentRecord).order_by(desc(ExperimentRecord.start_time)).all()
    return [
        {
            "id": r.id,
            "name": r.name,
            "status": r.status,
            "strategy": r.strategy,
            "num_rounds": r.num_rounds,
            "start_time": r.start_time,
            "end_time": r.end_time,
        }
        for r in records
    ]


def upsert_experiment(
    db: Session,
    exp_id: str,
    name: str,
    status: str = "RUNNING",
    strategy: str = "fedavg",
    num_rounds: int = 3,
    config: Optional[Dict[str, Any]] = None,
    end_time: Optional[str] = None,
) -> ExperimentRecord:
    record = db.query(ExperimentRecord).filter(ExperimentRecord.id == exp_id).first()
    if not record:
        record = ExperimentRecord(
            id=exp_id,
            name=name,
            status=status,
            strategy=strategy,
            num_rounds=num_rounds,
            config_json=json.dumps(config or {}),
        )
        db.add(record)
    else:
        record.status = status
        if name:
            record.name = name
        if end_time:
            record.end_time = end_time
        if config:
            record.config_json = json.dumps(config)
    db.commit()
    db.refresh(record)
    return record


def upsert_client(
    db: Session,
    client_id: str,
    experiment_id: str,
    status: Optional[str] = None,
    current_round: Optional[int] = None,
    samples: Optional[int] = None,
    last_loss: Optional[float] = None,
    last_accuracy: Optional[float] = None,
    cpu_percent: Optional[float] = None,
    ram_mb: Optional[float] = None,
    role: str = "honest",
) -> ClientRecord:
    record = db.query(ClientRecord).filter(ClientRecord.client_id == client_id).first()
    if not record:
        record = ClientRecord(
            client_id=client_id,
            experiment_id=experiment_id,
            role=role,
            status=status or "Registered",
            current_round=current_round or 0,
            samples=samples or 0,
            last_loss=last_loss,
            last_accuracy=last_accuracy,
            cpu_percent=cpu_percent,
            ram_mb=ram_mb,
        )
        db.add(record)
    else:
        if experiment_id:
            record.experiment_id = experiment_id
        if status:
            record.status = status
        if current_round is not None:
            record.current_round = current_round
        if samples is not None and samples > 0:
            record.samples = samples
        if last_loss is not None:
            record.last_loss = last_loss
        if last_accuracy is not None:
            record.last_accuracy = last_accuracy
        if cpu_percent is not None:
            record.cpu_percent = cpu_percent
        if ram_mb is not None:
            record.ram_mb = ram_mb
    db.commit()
    db.refresh(record)
    return record


def get_all_clients(db: Session, experiment_id: Optional[str] = None) -> List[Dict[str, Any]]:
    query = db.query(ClientRecord)
    if experiment_id:
        query = query.filter(ClientRecord.experiment_id == experiment_id)
    records = query.all()
    return [
        {
            "client_id": r.client_id,
            "experiment_id": r.experiment_id,
            "role": r.role,
            "status": r.status,
            "current_round": r.current_round,
            "samples": r.samples,
            "loss": r.last_loss,
            "accuracy": r.last_accuracy,
            "cpu": r.cpu_percent,
            "ram": r.ram_mb,
            "last_seen": r.last_seen,
        }
        for r in records
    ]


def get_client(db: Session, client_id: str) -> Optional[Dict[str, Any]]:
    r = db.query(ClientRecord).filter(ClientRecord.client_id == client_id).first()
    if not r:
        return None
    return {
        "client_id": r.client_id,
        "experiment_id": r.experiment_id,
        "role": r.role,
        "status": r.status,
        "current_round": r.current_round,
        "samples": r.samples,
        "loss": r.last_loss,
        "accuracy": r.last_accuracy,
        "cpu": r.cpu_percent,
        "ram": r.ram_mb,
        "last_seen": r.last_seen,
    }


def upsert_round(
    db: Session,
    experiment_id: str,
    round_number: int,
    status: Optional[str] = None,
    stage_name: Optional[str] = None,
    stage_status: Optional[str] = None,
    participating_clients: Optional[int] = None,
    loss: Optional[float] = None,
    accuracy: Optional[float] = None,
    duration_seconds: Optional[float] = None,
) -> RoundRecord:
    record = (
        db.query(RoundRecord)
        .filter(RoundRecord.experiment_id == experiment_id, RoundRecord.round_number == round_number)
        .first()
    )
    if not record:
        record = RoundRecord(
            experiment_id=experiment_id,
            round_number=round_number,
            status=status or "IN PROGRESS",
            participating_clients=participating_clients or 0,
            loss=loss,
            accuracy=accuracy,
            duration_seconds=duration_seconds,
        )
        db.add(record)
    else:
        if status:
            record.status = status
        if participating_clients is not None:
            record.participating_clients = participating_clients
        if loss is not None:
            record.loss = loss
        if accuracy is not None:
            record.accuracy = accuracy
        if duration_seconds is not None:
            record.duration_seconds = duration_seconds

    if stage_name and stage_status:
        attr_name = f"{stage_name}_status"
        if hasattr(record, attr_name):
            setattr(record, attr_name, stage_status)

    db.commit()
    db.refresh(record)
    return record


def get_all_rounds(db: Session, experiment_id: Optional[str] = None) -> List[Dict[str, Any]]:
    query = db.query(RoundRecord)
    if experiment_id:
        query = query.filter(RoundRecord.experiment_id == experiment_id)
    records = query.order_by(RoundRecord.round_number).all()
    return [
        {
            "id": r.id,
            "round_number": r.round_number,
            "status": r.status,
            "stages": {
                "model_distribution": r.model_distribution_status,
                "client_training": r.client_training_status,
                "update_collection": r.update_collection_status,
                "aggregation": r.aggregation_status,
                "global_evaluation": r.global_evaluation_status,
            },
            "participating_clients": r.participating_clients,
            "loss": r.loss,
            "accuracy": r.accuracy,
            "duration_seconds": r.duration_seconds,
            "start_time": r.start_time,
            "end_time": r.end_time,
        }
        for r in records
    ]


def get_round(db: Session, round_number: int, experiment_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    query = db.query(RoundRecord).filter(RoundRecord.round_number == round_number)
    if experiment_id:
        query = query.filter(RoundRecord.experiment_id == experiment_id)
    r = query.first()
    if not r:
        return None
    return {
        "id": r.id,
        "round_number": r.round_number,
        "status": r.status,
        "stages": {
            "model_distribution": r.model_distribution_status,
            "client_training": r.client_training_status,
            "update_collection": r.update_collection_status,
            "aggregation": r.aggregation_status,
            "global_evaluation": r.global_evaluation_status,
        },
        "participating_clients": r.participating_clients,
        "loss": r.loss,
        "accuracy": r.accuracy,
        "duration_seconds": r.duration_seconds,
        "start_time": r.start_time,
        "end_time": r.end_time,
    }


def add_client_metric(
    db: Session,
    experiment_id: str,
    client_id: str,
    round_number: int,
    loss: float,
    accuracy: float,
    samples: int,
    duration_seconds: float,
    epochs: int = 1,
    cpu_percent: Optional[float] = None,
    ram_mb: Optional[float] = None,
) -> ClientMetricRecord:
    existing = (
        db.query(ClientMetricRecord)
        .filter(
            ClientMetricRecord.experiment_id == experiment_id,
            ClientMetricRecord.client_id == client_id,
            ClientMetricRecord.round_number == round_number,
        )
        .first()
    )
    if existing:
        existing.loss = loss
        existing.accuracy = accuracy
        existing.samples = samples
        existing.duration_seconds = duration_seconds
        existing.epochs = epochs
        existing.cpu_percent = cpu_percent
        existing.ram_mb = ram_mb
        db.commit()
        db.refresh(existing)
        return existing

    metric = ClientMetricRecord(
        experiment_id=experiment_id,
        client_id=client_id,
        round_number=round_number,
        loss=loss,
        accuracy=accuracy,
        samples=samples,
        duration_seconds=duration_seconds,
        epochs=epochs,
        cpu_percent=cpu_percent,
        ram_mb=ram_mb,
    )
    db.add(metric)
    db.commit()
    db.refresh(metric)
    return metric


def get_client_metrics(db: Session, client_id: str) -> List[Dict[str, Any]]:
    records = (
        db.query(ClientMetricRecord)
        .filter(ClientMetricRecord.client_id == client_id)
        .order_by(ClientMetricRecord.round_number)
        .all()
    )
    return [
        {
            "round": r.round_number,
            "loss": r.loss,
            "accuracy": r.accuracy,
            "samples": r.samples,
            "duration_seconds": r.duration_seconds,
            "epochs": r.epochs,
            "cpu": r.cpu_percent,
            "ram": r.ram_mb,
            "timestamp": r.timestamp,
        }
        for r in records
    ]


def add_global_metric(
    db: Session,
    experiment_id: str,
    round_number: int,
    loss: float,
    accuracy: float,
    samples: int,
    duration_seconds: float = 0.0,
) -> GlobalMetricRecord:
    existing = (
        db.query(GlobalMetricRecord)
        .filter(
            GlobalMetricRecord.experiment_id == experiment_id,
            GlobalMetricRecord.round_number == round_number,
        )
        .first()
    )
    if existing:
        existing.loss = loss
        existing.accuracy = accuracy
        existing.samples = samples
        existing.duration_seconds = duration_seconds
        db.commit()
        db.refresh(existing)
        return existing

    metric = GlobalMetricRecord(
        experiment_id=experiment_id,
        round_number=round_number,
        loss=loss,
        accuracy=accuracy,
        samples=samples,
        duration_seconds=duration_seconds,
    )
    db.add(metric)
    db.commit()
    db.refresh(metric)
    return metric


def get_global_metrics(db: Session, experiment_id: Optional[str] = None) -> List[Dict[str, Any]]:
    query = db.query(GlobalMetricRecord)
    if experiment_id:
        query = query.filter(GlobalMetricRecord.experiment_id == experiment_id)
    records = query.order_by(GlobalMetricRecord.round_number).all()
    return [
        {
            "round": r.round_number,
            "loss": r.loss,
            "accuracy": r.accuracy,
            "samples": r.samples,
            "duration_seconds": r.duration_seconds,
            "timestamp": r.timestamp,
        }
        for r in records
    ]


def add_event(db: Session, event: BaseEvent) -> EventRecord:
    existing = db.query(EventRecord).filter(EventRecord.event_id == event.event_id).first()
    if existing:
        return existing

    record = EventRecord(
        event_id=event.event_id,
        event_type=event.event_type.value,
        experiment_id=event.experiment_id,
        round_number=event.round,
        client_id=event.client_id,
        severity=event.severity.value,
        message=event.message or "",
        data_json=json.dumps(event.data or {}),
        timestamp=event.timestamp,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def get_events(
    db: Session,
    experiment_id: Optional[str] = None,
    client_id: Optional[str] = None,
    round_number: Optional[int] = None,
    event_type: Optional[str] = None,
    severity: Optional[str] = None,
    limit: int = 200,
) -> List[Dict[str, Any]]:
    query = db.query(EventRecord)
    if experiment_id:
        query = query.filter(EventRecord.experiment_id == experiment_id)
    if client_id:
        query = query.filter(EventRecord.client_id == client_id)
    if round_number is not None:
        query = query.filter(EventRecord.round_number == round_number)
    if event_type:
        query = query.filter(EventRecord.event_type == event_type)
    if severity:
        query = query.filter(EventRecord.severity == severity)

    records = query.order_by(desc(EventRecord.timestamp)).limit(limit).all()
    return [
        {
            "event_id": r.event_id,
            "event_type": r.event_type,
            "experiment_id": r.experiment_id,
            "round": r.round_number,
            "client_id": r.client_id,
            "severity": r.severity,
            "message": r.message,
            "data": json.loads(r.data_json or "{}"),
            "timestamp": r.timestamp,
        }
        for r in reversed(records)  # chronological order
    ]


def add_system_resource(
    db: Session,
    host_cpu_percent: float,
    host_ram_percent: float,
    host_ram_used_mb: float,
    host_ram_total_mb: float,
    containers: Optional[Dict[str, Any]] = None,
) -> SystemResourceRecord:
    record = SystemResourceRecord(
        host_cpu_percent=host_cpu_percent,
        host_ram_percent=host_ram_percent,
        host_ram_used_mb=host_ram_used_mb,
        host_ram_total_mb=host_ram_total_mb,
        containers_json=json.dumps(containers or {}),
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def get_latest_system_resources(db: Session, limit: int = 30) -> List[Dict[str, Any]]:
    records = db.query(SystemResourceRecord).order_by(desc(SystemResourceRecord.timestamp)).limit(limit).all()
    return [
        {
            "timestamp": r.timestamp,
            "host_cpu_percent": r.host_cpu_percent,
            "host_ram_percent": r.host_ram_percent,
            "host_ram_used_mb": r.host_ram_used_mb,
            "host_ram_total_mb": r.host_ram_total_mb,
            "containers": json.loads(r.containers_json or "{}"),
        }
        for r in reversed(records)
    ]

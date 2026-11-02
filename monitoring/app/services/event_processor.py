"""Event processing logic to update database state and broadcast to WebSockets."""

from __future__ import annotations

from datetime import datetime, timezone
import logging
from typing import Any, Dict
from sqlalchemy.orm import Session

from monitoring.app.database import repository
from monitoring.app.websocket.manager import ws_manager
from shared.schemas.events import BaseEvent, ClientStatus, EventType, RoundStageStatus

logger = logging.getLogger("monitoring.event_processor")


async def process_and_dispatch_event(event: BaseEvent, db: Session) -> Dict[str, Any]:
    """
    1. Persist event in SQLite database.
    2. Update experiment, round, client, or metric records according to event type.
    3. Broadcast payload to connected WebSocket dashboard clients.
    """
    # 1. Persist raw event
    event_record = repository.add_event(db, event)

    data = event.data or {}
    exp_id = event.experiment_id
    round_num = event.round
    client_id = event.client_id
    timestamp = event.timestamp

    # 2. Process state updates
    if event.event_type == EventType.EXPERIMENT_STARTED:
        name = data.get("name", "Federated Learning Experiment")
        rounds = data.get("num_rounds", 3)
        strategy = data.get("strategy", "fedavg")
        repository.upsert_experiment(
            db=db,
            exp_id=exp_id,
            name=name,
            status="RUNNING",
            strategy=strategy,
            num_rounds=rounds,
            config=data,
        )

    elif event.event_type == EventType.EXPERIMENT_COMPLETED:
        repository.upsert_experiment(
            db=db,
            exp_id=exp_id,
            name="",
            status="COMPLETED",
            end_time=timestamp,
        )

    elif event.event_type == EventType.EXPERIMENT_FAILED:
        repository.upsert_experiment(
            db=db,
            exp_id=exp_id,
            name="",
            status="FAILED",
            end_time=timestamp,
        )

    elif event.event_type == EventType.ROUND_STARTED and round_num is not None:
        repository.upsert_round(
            db=db,
            experiment_id=exp_id,
            round_number=round_num,
            status="IN PROGRESS",
            stage_name="model_distribution",
            stage_status=RoundStageStatus.WAITING.value,
        )

    elif event.event_type == EventType.MODEL_DISTRIBUTION_STARTED and round_num is not None:
        repository.upsert_round(
            db=db,
            experiment_id=exp_id,
            round_number=round_num,
            stage_name="model_distribution",
            stage_status=RoundStageStatus.IN_PROGRESS.value,
        )

    elif event.event_type == EventType.CLIENT_UPDATES_STARTED and round_num is not None:
        count = data.get("received_count", 0)
        repository.upsert_round(
            db=db,
            experiment_id=exp_id,
            round_number=round_num,
            stage_name="model_distribution",
            stage_status=RoundStageStatus.COMPLETED.value,
            participating_clients=count,
        )
        repository.upsert_round(
            db=db,
            experiment_id=exp_id,
            round_number=round_num,
            stage_name="update_collection",
            stage_status=RoundStageStatus.COMPLETED.value,
        )

    elif event.event_type == EventType.AGGREGATION_STARTED and round_num is not None:
        clients_count = data.get("participating_clients")
        repository.upsert_round(
            db=db,
            experiment_id=exp_id,
            round_number=round_num,
            stage_name="aggregation",
            stage_status=RoundStageStatus.IN_PROGRESS.value,
            participating_clients=clients_count,
        )

    elif event.event_type == EventType.AGGREGATION_FINISHED and round_num is not None:
        repository.upsert_round(
            db=db,
            experiment_id=exp_id,
            round_number=round_num,
            stage_name="aggregation",
            stage_status=RoundStageStatus.COMPLETED.value,
        )

    elif event.event_type == EventType.EVALUATION_STARTED and round_num is not None:
        repository.upsert_round(
            db=db,
            experiment_id=exp_id,
            round_number=round_num,
            stage_name="global_evaluation",
            stage_status=RoundStageStatus.IN_PROGRESS.value,
        )

    elif event.event_type == EventType.EVALUATION_FINISHED and round_num is not None:
        loss = data.get("loss", 0.0)
        acc = data.get("accuracy", 0.0)
        samples = data.get("samples", 0)
        dur = data.get("duration_seconds", 0.0)
        repository.upsert_round(
            db=db,
            experiment_id=exp_id,
            round_number=round_num,
            stage_name="global_evaluation",
            stage_status=RoundStageStatus.COMPLETED.value,
            loss=loss,
            accuracy=acc,
        )
        repository.add_global_metric(
            db=db,
            experiment_id=exp_id,
            round_number=round_num,
            loss=loss,
            accuracy=acc,
            samples=samples,
            duration_seconds=dur,
        )

    elif event.event_type == EventType.ROUND_COMPLETED and round_num is not None:
        dur = data.get("duration_seconds")
        loss = data.get("loss")
        acc = data.get("accuracy")
        repository.upsert_round(
            db=db,
            experiment_id=exp_id,
            round_number=round_num,
            status="COMPLETED",
            loss=loss,
            accuracy=acc,
            duration_seconds=dur,
        )

    elif event.event_type == EventType.CLIENT_REGISTERED and client_id:
        samples = data.get("samples", 0)
        role = data.get("role", "honest")
        repository.upsert_client(
            db=db,
            client_id=client_id,
            experiment_id=exp_id,
            status=ClientStatus.REGISTERED.value,
            samples=samples,
            role=role,
        )

    elif event.event_type == EventType.CLIENT_CONNECTED and client_id:
        repository.upsert_client(
            db=db,
            client_id=client_id,
            experiment_id=exp_id,
            status=ClientStatus.CONNECTED.value,
        )

    elif event.event_type == EventType.CLIENT_TRAINING_STARTED and client_id:
        repository.upsert_client(
            db=db,
            client_id=client_id,
            experiment_id=exp_id,
            status=ClientStatus.TRAINING.value,
            current_round=round_num,
        )
        if round_num is not None:
            repository.upsert_round(
                db=db,
                experiment_id=exp_id,
                round_number=round_num,
                stage_name="client_training",
                stage_status=RoundStageStatus.IN_PROGRESS.value,
            )

    elif event.event_type == EventType.CLIENT_UPDATE_SENT and client_id:
        repository.upsert_client(
            db=db,
            client_id=client_id,
            experiment_id=exp_id,
            status=ClientStatus.SENDING_UPDATE.value,
            current_round=round_num,
        )

    elif event.event_type == EventType.CLIENT_TRAINING_COMPLETED and client_id:
        loss = data.get("loss", 0.0)
        acc = data.get("accuracy", 0.0)
        samples = data.get("samples", 0)
        dur = data.get("duration_seconds", 0.0)
        epochs = data.get("epochs", 1)
        cpu = data.get("cpu_percent")
        ram = data.get("ram_mb")

        repository.upsert_client(
            db=db,
            client_id=client_id,
            experiment_id=exp_id,
            status=ClientStatus.WAITING.value,
            current_round=round_num,
            samples=samples,
            last_loss=loss,
            last_accuracy=acc,
            cpu_percent=cpu,
            ram_mb=ram,
        )
        if round_num is not None:
            repository.add_client_metric(
                db=db,
                experiment_id=exp_id,
                client_id=client_id,
                round_number=round_num,
                loss=loss,
                accuracy=acc,
                samples=samples,
                duration_seconds=dur,
                epochs=epochs,
                cpu_percent=cpu,
                ram_mb=ram,
            )

    elif event.event_type == EventType.CLIENT_ERROR and client_id:
        repository.upsert_client(
            db=db,
            client_id=client_id,
            experiment_id=exp_id,
            status=ClientStatus.FAILED.value,
        )

    # 3. Construct broadcast message
    broadcast_msg = {
        "type": "event",
        "event": {
            "event_id": event.event_id,
            "event_type": event.event_type.value,
            "timestamp": event.timestamp,
            "experiment_id": event.experiment_id,
            "round": event.round,
            "client_id": event.client_id,
            "severity": event.severity.value,
            "message": event.message,
            "data": event.data,
        },
    }

    # Asynchronously broadcast to all open WebSocket connections
    await ws_manager.broadcast(broadcast_msg)

    return {"status": "accepted", "event_id": event.event_id}

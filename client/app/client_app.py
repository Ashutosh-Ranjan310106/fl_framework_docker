"""Flower ClientApp implementation for CIFAR-10 federated participant."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import sys
import time
from typing import Dict, List, Tuple
import flwr as fl
from flwr.client import ClientApp, NumPyClient

from client.app.client_identity import ClientConfig, load_client_config
from client.app.dataset_loader import get_client_dataloader
from client.app.event_publisher import ClientEventPublisher
from client.app.model import CIFAR10CNN, get_model_parameters, set_model_parameters
from client.app.trainer import evaluate_local, train_local
from shared.schemas.events import EventSeverity, EventType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [Client-%(name)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("client_app")

# Add file handler for persistent volume logging
log_dir = Path(os.getenv("LOG_DIR", "logs"))
if log_dir.exists() or os.getenv("LOG_DIR"):
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_dir / "client.log"), mode="a", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [Client-%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        logging.getLogger().addHandler(file_handler)
    except Exception:
        pass


class FlowerClient(NumPyClient):
    """Federated Learning client implementing NumPyClient interface."""

    def __init__(self, config: ClientConfig, publisher: ClientEventPublisher):
        self.config = config
        self.publisher = publisher
        self.model = CIFAR10CNN(num_classes=10)
        self.device = config.device

        # Load local partition
        logger.info(f"Loading client partition from {config.partition_path}...")
        self.train_loader = get_client_dataloader(
            partition_path=config.partition_path,
            batch_size=config.batch_size,
            shuffle=True,
        )
        self.num_samples = len(self.train_loader.dataset)
        logger.info(f"Client {config.client_id} successfully loaded {self.num_samples} training samples.")

        # Emit registration and connection events
        self.publisher.emit(
            event_type=EventType.CLIENT_REGISTERED,
            severity=EventSeverity.INFO,
            message=f"Client {config.client_id} registered with {self.num_samples} samples.",
            data={"samples": self.num_samples, "partition_path": config.partition_path, "role": config.role},
        )
        self.publisher.emit(
            event_type=EventType.CLIENT_CONNECTED,
            severity=EventSeverity.SUCCESS,
            message=f"Client {config.client_id} connected to Flower infrastructure.",
            data={"samples": self.num_samples},
        )

        self.current_round = 0

    def get_parameters(self, config: Dict[str, fl.common.Scalar]) -> List:
        return get_model_parameters(self.model)

    def fit(self, parameters: List, config: Dict[str, fl.common.Scalar]) -> Tuple[List, int, Dict]:
        # Extract server round and active experiment
        server_round = int(config.get("server_round", self.current_round + 1))
        self.current_round = server_round

        if "experiment_id" in config and config["experiment_id"]:
            self.publisher.experiment_id = str(config["experiment_id"])

        logger.info(f"--- Client {self.config.client_id} starting Round {server_round} (Exp: {self.publisher.experiment_id}) ---")
        self.publisher.emit(
            event_type=EventType.CLIENT_TRAINING_STARTED,
            round_num=server_round,
            severity=EventSeverity.INFO,
            message=f"Client {self.config.client_id} started training for round {server_round}.",
            data={"round": server_round, "epochs": self.config.local_epochs, "batch_size": self.config.batch_size},
        )

        # 1. Update model with global parameters
        set_model_parameters(self.model, parameters)

        # 2. Local training
        try:
            metrics = train_local(
                model=self.model,
                train_loader=self.train_loader,
                epochs=self.config.local_epochs,
                lr=self.config.learning_rate,
                optimizer_name=self.config.optimizer,
                device=self.device,
            )
        except Exception as e:
            logger.error(f"Training error in round {server_round}: {e}", exc_info=True)
            self.publisher.emit(
                event_type=EventType.CLIENT_ERROR,
                round_num=server_round,
                severity=EventSeverity.ERROR,
                message=f"Local training failed in round {server_round}: {e}",
                data={"error": str(e)},
            )
            raise e

        metrics["client_id"] = self.config.client_id

        logger.info(
            f"Client {self.config.client_id} finished Round {server_round}: "
            f"Loss={metrics['loss']:.4f}, Accuracy={metrics['accuracy']:.4f}, Time={metrics['duration_seconds']:.2f}s"
        )

        # 3. Emit training completed telemetry
        self.publisher.emit(
            event_type=EventType.CLIENT_TRAINING_COMPLETED,
            round_num=server_round,
            severity=EventSeverity.SUCCESS,
            message=f"Client {self.config.client_id} completed training for round {server_round}.",
            data=metrics,
        )

        # 4. Extract updated parameters and emit update sent
        updated_params = get_model_parameters(self.model)
        self.publisher.emit(
            event_type=EventType.CLIENT_UPDATE_SENT,
            round_num=server_round,
            severity=EventSeverity.INFO,
            message=f"Client {self.config.client_id} sent updated weights to server for round {server_round}.",
            data={"round": server_round, "samples": self.num_samples},
        )

        return updated_params, self.num_samples, metrics

    def evaluate(self, parameters: List, config: Dict[str, fl.common.Scalar]) -> Tuple[float, int, Dict]:
        set_model_parameters(self.model, parameters)
        loss, samples, metrics = evaluate_local(
            model=self.model,
            eval_loader=self.train_loader,
            device=self.device,
        )
        return float(loss), int(samples), metrics


def client_fn(context: fl.common.Context) -> fl.client.Client:
    """Flower ClientApp factory function."""
    client_config = load_client_config()
    publisher = ClientEventPublisher(
        monitoring_url=client_config.monitoring_url,
        client_id=client_config.client_id,
        enabled=client_config.monitoring_enabled,
    )
    flower_client = FlowerClient(config=client_config, publisher=publisher)
    return flower_client.to_client()


# Flower modern ClientApp
app = ClientApp(client_fn=client_fn)


def run_standalone_client():
    """Run client connecting over gRPC to server address with reconnect logic."""
    client_config = load_client_config()

    # Discover active experiment ID from monitoring service if not explicitly set
    active_exp_id = os.getenv("EXPERIMENT_ID")
    if not active_exp_id and client_config.monitoring_url:
        try:
            import requests
            resp = requests.get(f"{client_config.monitoring_url}/api/experiment/current", timeout=1.5)
            if resp.status_code == 200:
                data = resp.json()
                if data and data.get("id"):
                    active_exp_id = data["id"]
        except Exception:
            pass

    publisher = ClientEventPublisher(
        monitoring_url=client_config.monitoring_url,
        client_id=client_config.client_id,
        experiment_id=active_exp_id or "cifar10_baseline_001",
        enabled=client_config.monitoring_enabled,
    )

    flower_client = FlowerClient(config=client_config, publisher=publisher)
    logger.info(f"Connecting to Flower server at {client_config.server_address}...")

    max_retries = 30
    retry_delay = 2
    for attempt in range(1, max_retries + 1):
        try:
            fl.client.start_client(
                server_address=client_config.server_address,
                client=flower_client.to_client(),
            )
            logger.info(f"Client {client_config.client_id} finished execution.")
            break
        except Exception as e:
            logger.warning(f"Connection attempt {attempt}/{max_retries} failed: {e}. Retrying in {retry_delay}s...")
            publisher.emit(
                event_type=EventType.CLIENT_ERROR,
                severity=EventSeverity.WARNING,
                message=f"Connection attempt {attempt}/{max_retries} failed: {e}",
                data={"attempt": attempt, "error": str(e)},
            )
            time.sleep(retry_delay)
    else:
        logger.error(f"Could not connect to Flower server after {max_retries} attempts.")
        publisher.emit(
            event_type=EventType.CLIENT_ERROR,
            severity=EventSeverity.CRITICAL,
            message="Max connection retries exceeded.",
        )
        sys.exit(1)

    # Flush events on exit
    publisher.close()


if __name__ == "__main__":
    run_standalone_client()

"""Central Flower Server orchestration and ServerApp implementation."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict
import yaml
import flwr as fl
from flwr.server import ServerApp, ServerAppComponents, ServerConfig, start_server

from server.app.event_publisher import EventPublisher
from server.app.metrics_manager import MetricsManager
from server.app.model_manager import ModelManager
from server.app.round_manager import RoundManager
from server.app.strategy import MonitoredFedAvg
from shared.schemas.events import EventSeverity, EventType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [Server-%(name)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("server_app")


def load_server_config() -> Dict[str, Any]:
    """Load experiment configuration from YAML or environment variables."""
    config_path = os.getenv("EXPERIMENT_CONFIG_PATH", "experiments/default.yaml")
    cfg: Dict[str, Any] = {}

    if Path(config_path).exists():
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    exp_info = cfg.get("experiment", {})
    fl_info = cfg.get("federated_learning", {})
    mon_info = cfg.get("monitoring", {})

    return {
        "experiment_id": os.getenv("EXPERIMENT_ID", exp_info.get("id", "cifar10_baseline_001")),
        "experiment_name": exp_info.get("name", "CIFAR10 Flower Baseline"),
        "num_rounds": int(os.getenv("NUM_ROUNDS", fl_info.get("num_rounds", 3))),
        "min_fit_clients": int(os.getenv("MIN_FIT_CLIENTS", fl_info.get("min_fit_clients", 2))),
        "min_available_clients": int(os.getenv("MIN_AVAILABLE_CLIENTS", fl_info.get("min_available_clients", 2))),
        "fraction_fit": float(os.getenv("FRACTION_FIT", fl_info.get("fraction_fit", 1.0))),
        "fraction_evaluate": float(os.getenv("FRACTION_EVALUATE", fl_info.get("fraction_evaluate", 1.0))),
        "server_address": os.getenv("SERVER_ADDRESS", fl_info.get("server_address", "0.0.0.0:8080")),
        "monitoring_url": os.getenv("MONITORING_URL", mon_info.get("url", "http://monitor:8000")),
        "monitoring_enabled": mon_info.get("enabled", True),
        "checkpoint_dir": os.getenv("CHECKPOINT_DIR", "models/checkpoints"),
        "test_data_path": os.getenv("TEST_DATA_PATH", "datasets/original/cifar10/test/data.pt"),
    }


def server_fn(context: fl.common.Context) -> ServerAppComponents:
    """ServerApp components factory for modern Flower ServerApp."""
    cfg = load_server_config()

    publisher = EventPublisher(
        monitoring_url=cfg["monitoring_url"],
        experiment_id=cfg["experiment_id"],
        enabled=cfg["monitoring_enabled"],
    )
    round_manager = RoundManager(experiment_id=cfg["experiment_id"])
    model_manager = ModelManager(
        checkpoint_dir=cfg["checkpoint_dir"],
        test_data_path=cfg["test_data_path"],
    )
    metrics_manager = MetricsManager(experiment_id=cfg["experiment_id"])

    strategy = MonitoredFedAvg(
        event_publisher=publisher,
        round_manager=round_manager,
        model_manager=model_manager,
        metrics_manager=metrics_manager,
        fraction_fit=cfg["fraction_fit"],
        fraction_evaluate=cfg["fraction_evaluate"],
        min_fit_clients=cfg["min_fit_clients"],
        min_available_clients=cfg["min_available_clients"],
        num_rounds=cfg["num_rounds"],
    )

    config = ServerConfig(num_rounds=cfg["num_rounds"])
    return ServerAppComponents(strategy=strategy, config=config)


# Flower modern ServerApp
app = ServerApp(server_fn=server_fn)


def run_standalone_server():
    """Run Flower central server instance using gRPC."""
    cfg = load_server_config()

    logger.info("==================================================")
    logger.info(f" Starting Flower FL Server: {cfg['experiment_name']}")
    logger.info(f" Experiment ID: {cfg['experiment_id']}")
    logger.info(f" Target Rounds: {cfg['num_rounds']}")
    logger.info(f" Min Fit Clients: {cfg['min_fit_clients']}")
    logger.info(f" Server Address: {cfg['server_address']}")
    logger.info(f" Monitoring URL: {cfg['monitoring_url']}")
    logger.info("==================================================")

    publisher = EventPublisher(
        monitoring_url=cfg["monitoring_url"],
        experiment_id=cfg["experiment_id"],
        enabled=cfg["monitoring_enabled"],
    )

    # Emit experiment started
    publisher.emit(
        event_type=EventType.EXPERIMENT_STARTED,
        severity=EventSeverity.INFO,
        message=f"Experiment {cfg['experiment_name']} ({cfg['experiment_id']}) started.",
        data={
            "experiment_id": cfg["experiment_id"],
            "name": cfg["experiment_name"],
            "num_rounds": cfg["num_rounds"],
            "min_fit_clients": cfg["min_fit_clients"],
            "strategy": "fedavg",
        },
    )

    round_manager = RoundManager(experiment_id=cfg["experiment_id"])
    model_manager = ModelManager(
        checkpoint_dir=cfg["checkpoint_dir"],
        test_data_path=cfg["test_data_path"],
    )
    metrics_manager = MetricsManager(experiment_id=cfg["experiment_id"])

    strategy = MonitoredFedAvg(
        event_publisher=publisher,
        round_manager=round_manager,
        model_manager=model_manager,
        metrics_manager=metrics_manager,
        fraction_fit=cfg["fraction_fit"],
        fraction_evaluate=cfg["fraction_evaluate"],
        min_fit_clients=cfg["min_fit_clients"],
        min_available_clients=cfg["min_available_clients"],
        num_rounds=cfg["num_rounds"],
    )

    try:
        start_server(
            server_address=cfg["server_address"],
            config=ServerConfig(num_rounds=cfg["num_rounds"]),
            strategy=strategy,
        )
        logger.info("Flower Server finished training rounds.")
    except Exception as e:
        logger.error(f"Flower server encountered error: {e}", exc_info=True)
        publisher.emit(
            event_type=EventType.EXPERIMENT_FAILED,
            severity=EventSeverity.CRITICAL,
            message=f"Server terminated with error: {e}",
            data={"error": str(e)},
        )
        sys.exit(1)
    finally:
        publisher.close()


if __name__ == "__main__":
    run_standalone_server()

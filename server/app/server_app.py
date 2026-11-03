"""Central Flower Server orchestration and ServerApp implementation."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any, Dict, Optional, Tuple
import yaml
import flwr as fl
from flwr.server import ServerApp, ServerAppComponents, ServerConfig, start_server

from server.app.event_publisher import EventPublisher
from server.app.metrics_manager import MetricsManager
from server.app.model_manager import ModelManager
from server.app.report_generator import ReportGenerator
from server.app.round_manager import RoundManager
from server.app.strategy import MonitoredFedAvg
from shared.schemas.events import EventSeverity, EventType

# Configure base logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [Server-%(name)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("server_app")

# Add persistent volume file logging
LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))
try:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    server_log_file = LOG_DIR / "server.log"
    file_handler = logging.FileHandler(str(server_log_file), mode="a", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [Server-%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logging.getLogger().addHandler(file_handler)
except Exception as e:
    logger.warning(f"Could not initialize file logging in {LOG_DIR}: {e}")


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

    # Auto-generate unique timestamped run ID if not explicitly specified via environment
    if os.getenv("EXPERIMENT_ID"):
        exp_id = os.getenv("EXPERIMENT_ID")
    else:
        base_name = exp_info.get("id", "cifar10_run")
        run_ts = time.strftime("%Y%m%d_%H%M%S")
        exp_id = f"{base_name}_{run_ts}"

    return {
        "experiment_id": exp_id,
        "experiment_name": exp_info.get("name", "CIFAR10 Flower Baseline"),
        "num_rounds": int(os.getenv("NUM_ROUNDS", fl_info.get("num_rounds", 3))),
        "min_fit_clients": int(os.getenv("MIN_FIT_CLIENTS", fl_info.get("min_fit_clients", 2))),
        "min_available_clients": int(os.getenv("MIN_AVAILABLE_CLIENTS", fl_info.get("min_available_clients", 2))),
        "fraction_fit": float(os.getenv("FRACTION_FIT", fl_info.get("fraction_fit", 1.0))),
        "fraction_evaluate": float(os.getenv("FRACTION_EVALUATE", fl_info.get("fraction_evaluate", 1.0))),
        "server_address": os.getenv("SERVER_ADDRESS", fl_info.get("server_address", "0.0.0.0:8080")),
        "monitoring_url": os.getenv("MONITORING_URL", mon_info.get("url", "http://127.0.0.1:8000")),
        "monitoring_enabled": mon_info.get("enabled", True),
        "enable_embedded_monitor": os.getenv("ENABLE_EMBEDDED_MONITOR", "true").lower() in ("1", "true", "yes"),
        "continuous_mode": os.getenv("CONTINUOUS_MODE", "true").lower() in ("1", "true", "yes"),
        "monitor_host": os.getenv("MONITOR_HOST", "0.0.0.0"),
        "monitor_port": int(os.getenv("MONITOR_PORT", "8000")),
        "checkpoint_dir": os.getenv("CHECKPOINT_DIR", "models/checkpoints"),
        "results_dir": os.getenv("RESULTS_DIR", "results"),
        "log_dir": os.getenv("LOG_DIR", "logs"),
        "test_data_path": os.getenv("TEST_DATA_PATH", "datasets/original/cifar10/test/data.pt"),
    }


def start_embedded_monitoring_server(host: str = "0.0.0.0", port: int = 8000):
    """Launch embedded FastAPI monitoring server in a background daemon thread."""
    try:
        import requests
        import uvicorn
        from monitoring.app.main import app as monitor_app

        config = uvicorn.Config(
            app=monitor_app,
            host=host,
            port=port,
            log_level="warning",
            access_log=False,
        )
        server = uvicorn.Server(config)
        thread = threading.Thread(target=server.run, daemon=True, name="EmbeddedMonitorThread")
        thread.start()

        # Wait for health check
        for _ in range(30):
            time.sleep(0.1)
            try:
                res = requests.get(f"http://127.0.0.1:{port}/api/health", timeout=1)
                if res.status_code == 200:
                    logger.info(f"Embedded Monitoring Dashboard ready at http://{host}:{port}")
                    return server, thread
            except Exception:
                pass

        logger.warning(f"Monitoring server started on port {port} (health check timed out, proceeding).")
        return server, thread
    except Exception as e:
        logger.error(f"Failed to start embedded monitoring server: {e}", exc_info=True)
        return None, None


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
    report_generator = ReportGenerator(
        experiment_id=cfg["experiment_id"],
        experiment_name=cfg["experiment_name"],
        num_rounds=cfg["num_rounds"],
        results_dir=cfg["results_dir"],
    )

    strategy = MonitoredFedAvg(
        event_publisher=publisher,
        round_manager=round_manager,
        model_manager=model_manager,
        metrics_manager=metrics_manager,
        report_generator=report_generator,
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
    """Run Flower central server instance with persistent gRPC listening and integrated monitoring."""
    base_cfg = load_server_config()

    # Start embedded monitor if enabled (runs continuously in background)
    if base_cfg["enable_embedded_monitor"]:
        logger.info(f"Starting embedded Monitoring Console on port {base_cfg['monitor_port']}...")
        start_embedded_monitoring_server(
            host=base_cfg["monitor_host"], port=base_cfg["monitor_port"]
        )

    continuous_mode = base_cfg["continuous_mode"]

    while True:
        cfg = load_server_config()

        logger.info("==================================================")
        logger.info(f" Flower Server Ready: {cfg['experiment_name']}")
        logger.info(f" Experiment ID: {cfg['experiment_id']}")
        logger.info(f" Target Rounds: {cfg['num_rounds']}")
        logger.info(f" Min Fit Clients: {cfg['min_fit_clients']}")
        logger.info(f" Flower gRPC Port: {cfg['server_address']} (LISTENING)")
        logger.info(f" Web Dashboard: http://localhost:{cfg['monitor_port']}")
        logger.info(f" Results Volume: {cfg['results_dir']}")
        logger.info(f" Logs Volume: {cfg['log_dir']}")
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
        report_generator = ReportGenerator(
            experiment_id=cfg["experiment_id"],
            experiment_name=cfg["experiment_name"],
            num_rounds=cfg["num_rounds"],
            results_dir=cfg["results_dir"],
        )

        strategy = MonitoredFedAvg(
            event_publisher=publisher,
            round_manager=round_manager,
            model_manager=model_manager,
            metrics_manager=metrics_manager,
            report_generator=report_generator,
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
            logger.info("Flower Server completed all federated rounds.")

            # Finalize and persist reports to volume
            final_reports = report_generator.generate_all_reports(status="COMPLETED")
            logger.info("==================================================")
            logger.info(" Federated Learning Run Completed Successfully!")
            logger.info(f" Accuracy details & run reports saved to:")
            logger.info(f"   - JSON: {final_reports['json']}")
            logger.info(f"   - Markdown: {final_reports['markdown']}")
            logger.info(f"   - CSV: {final_reports['csv']}")
            logger.info(f" Monitoring Console is ACTIVE at http://localhost:{cfg['monitor_port']}")
            logger.info("==================================================")

        except KeyboardInterrupt:
            logger.info("Server received interrupt, exiting.")
            break
        except Exception as e:
            logger.error(f"Flower server encountered error: {e}", exc_info=True)
            publisher.emit(
                event_type=EventType.EXPERIMENT_FAILED,
                severity=EventSeverity.CRITICAL,
                message=f"Server terminated with error: {e}",
                data={"error": str(e)},
            )
            report_generator.generate_all_reports(status="FAILED")
            time.sleep(3)
        finally:
            publisher.close()

        if not continuous_mode:
            logger.info("Continuous mode disabled. Dashboard remaining active. Press CTRL+C to stop.")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                break
        else:
            logger.info("Starting new FL session — gRPC port 8080 remains open for next training run...")
            time.sleep(2)


if __name__ == "__main__":
    run_standalone_server()

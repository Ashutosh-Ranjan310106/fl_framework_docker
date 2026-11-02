"""Client identity and configuration resolution from CLI args, environment, and YAML."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml
from pydantic import BaseModel, Field


class ClientConfig(BaseModel):
    client_id: str = "client_001"
    role: str = "honest"
    partition_path: str = "/app/data"
    server_address: str = "server:8080"
    monitoring_url: str = "http://monitor:8000"
    monitoring_enabled: bool = True
    local_epochs: int = 2
    batch_size: int = 32
    learning_rate: float = 0.001
    optimizer: str = "adam"
    device: str = "cpu"


def _parse_cli_args() -> Dict[str, Any]:
    """Parse optional command-line flags."""
    parser = argparse.ArgumentParser(description="Flower Federated Client", add_help=False)
    parser.add_argument("--client-id", dest="client_id", type=str, default=None)
    parser.add_argument("--server", dest="server_address", type=str, default=None)
    parser.add_argument("--monitor", dest="monitoring_url", type=str, default=None)
    parser.add_argument("--partition", dest="partition_path", type=str, default=None)
    parser.add_argument("--role", dest="role", type=str, default=None)
    parser.add_argument("--epochs", dest="local_epochs", type=int, default=None)
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=None)
    parser.add_argument("--lr", dest="learning_rate", type=float, default=None)
    parser.add_argument("--device", dest="device", type=str, default=None)
    try:
        args, _ = parser.parse_known_args()
        return {k: v for k, v in vars(args).items() if v is not None}
    except Exception:
        return {}


def load_client_config(config_path: Optional[str | Path] = None) -> ClientConfig:
    """
    Resolve client configuration with hierarchy:
    1. CLI flags (highest priority)
    2. Environment variables
    3. YAML config file
    4. Defaults
    """
    data: Dict[str, Any] = {}

    # Check config path from env or argument
    path = config_path or os.getenv("CLIENT_CONFIG_PATH", "shared/configs/client_config.yaml")
    if Path(path).exists():
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
            c_info = raw.get("client", {})
            s_info = raw.get("server", {})
            m_info = raw.get("monitoring", {})
            t_info = raw.get("training", {})

            data = {
                "client_id": c_info.get("id"),
                "role": c_info.get("role"),
                "partition_path": c_info.get("partition_path"),
                "server_address": s_info.get("address"),
                "monitoring_url": m_info.get("url"),
                "monitoring_enabled": m_info.get("enabled", True),
                "local_epochs": t_info.get("local_epochs"),
                "batch_size": t_info.get("batch_size"),
                "learning_rate": t_info.get("learning_rate"),
                "optimizer": t_info.get("optimizer"),
                "device": t_info.get("device"),
            }
            # Clean None values
            data = {k: v for k, v in data.items() if v is not None}

    # Override with Environment Variables if set
    if os.getenv("CLIENT_ID"):
        data["client_id"] = os.getenv("CLIENT_ID")
    if os.getenv("CLIENT_ROLE"):
        data["role"] = os.getenv("CLIENT_ROLE")
    if os.getenv("PARTITION_PATH"):
        data["partition_path"] = os.getenv("PARTITION_PATH")
    if os.getenv("SERVER_ADDRESS"):
        data["server_address"] = os.getenv("SERVER_ADDRESS")
    if os.getenv("MONITORING_URL"):
        data["monitoring_url"] = os.getenv("MONITORING_URL")
    if os.getenv("LOCAL_EPOCHS"):
        data["local_epochs"] = int(os.getenv("LOCAL_EPOCHS"))
    if os.getenv("BATCH_SIZE"):
        data["batch_size"] = int(os.getenv("BATCH_SIZE"))
    if os.getenv("LEARNING_RATE"):
        data["learning_rate"] = float(os.getenv("LEARNING_RATE"))

    # Override with CLI arguments (highest priority)
    cli_args = _parse_cli_args()
    data.update(cli_args)

    return ClientConfig(**data)

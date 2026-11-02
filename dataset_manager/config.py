"""Configuration loader, constants, and logging setup for CIFAR-10 Dataset Manager."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict
import yaml

# Base directory for the project
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset constants
DATASET_NAME = "CIFAR-10"
NUM_CLASSES = 10
TOTAL_TRAIN_SAMPLES = 50000
TOTAL_TEST_SAMPLES = 10000
IMAGE_SHAPE = (3, 32, 32)
DEFAULT_SEED = 42

CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

# Standard storage paths
DATASETS_ROOT = PROJECT_ROOT / "datasets"
ORIGINAL_CIFAR10_DIR = DATASETS_ROOT / "original" / "cifar10"
ORIGINAL_TRAIN_DIR = ORIGINAL_CIFAR10_DIR / "train"
ORIGINAL_TEST_DIR = ORIGINAL_CIFAR10_DIR / "test"
PARTITIONS_ROOT = DATASETS_ROOT / "partitions" / "cifar10"
METADATA_DIR = DATASETS_ROOT / "metadata"
REGISTRY_FILE = METADATA_DIR / "dataset_registry.json"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default_config.yaml"


def setup_logger(name: str = "dataset_manager", level: int = logging.INFO) -> logging.Logger:
    """Set up and return a standardized logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger


def load_yaml_config(config_path: Path | str | None = None) -> Dict[str, Any]:
    """Load configuration from a YAML file, falling back to DEFAULT_CONFIG_PATH."""
    target_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not target_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {target_path}")

    with open(target_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config or {}


def save_yaml_config(data: Dict[str, Any], config_path: Path | str) -> None:
    """Save configuration data to a YAML file."""
    target_path = Path(config_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)

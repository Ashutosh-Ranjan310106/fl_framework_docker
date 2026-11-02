"""Manifest generator, loader, and fingerprinting for partition configurations."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from dataset_manager.config import (
    CIFAR10_CLASSES,
    DATASET_NAME,
    NUM_CLASSES,
    TOTAL_TEST_SAMPLES,
    TOTAL_TRAIN_SAMPLES,
    setup_logger,
)

logger = setup_logger("manifest")


def compute_config_fingerprint(config_dict: Dict[str, Any]) -> str:
    """Compute deterministic SHA-256 hash of configuration dictionary."""
    canonical_json = json.dumps(config_dict, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def create_partition_manifest(
    partition_strategy: str,
    num_clients: int,
    random_seed: int,
    strategy_params: Dict[str, Any],
    client_statistics: Dict[str, Dict[str, Any]],
    file_checksums: Dict[str, str],
    validation_results: Dict[str, Any],
    dataset_fingerprint: Optional[str] = None,
    total_train_samples: int = TOTAL_TRAIN_SAMPLES,
    total_test_samples: int = TOTAL_TEST_SAMPLES,
) -> Dict[str, Any]:
    """
    Construct a complete, self-contained partition manifest.
    Includes client breakdowns, fingerprints, checksums, and validation status.
    """
    sample_counts = [info["sample_count"] for info in client_statistics.values()]
    min_samples = min(sample_counts) if sample_counts else 0
    max_samples = max(sample_counts) if sample_counts else 0
    avg_samples = sum(sample_counts) / len(sample_counts) if sample_counts else 0.0
    total_allocated = sum(sample_counts)

    config_for_hash = {
        "dataset": DATASET_NAME,
        "partition_strategy": partition_strategy,
        "num_clients": num_clients,
        "random_seed": random_seed,
        "strategy_params": strategy_params,
        "total_train_samples": total_train_samples,
    }
    config_fingerprint = compute_config_fingerprint(config_for_hash)

    client_partitions = {}
    for cid, info in client_statistics.items():
        rel_path = f"{cid}/data.pt"
        client_partitions[cid] = {
            "client_id": cid,
            "sample_count": info["sample_count"],
            "partition_path": rel_path,
            "class_distribution": info.get("class_distribution", {}),
            "dominant_class": info.get("dominant_class"),
            "entropy": info.get("entropy"),
            "checksum": file_checksums.get(rel_path),
        }

    manifest = {
        "dataset": DATASET_NAME,
        "partition_strategy": partition_strategy,
        "num_clients": num_clients,
        "random_seed": random_seed,
        "strategy_params": strategy_params,
        "total_train_samples": total_train_samples,
        "total_test_samples": total_test_samples,
        "total_allocated_samples": total_allocated,
        "num_classes": NUM_CLASSES,
        "class_names": CIFAR10_CLASSES,
        "creation_timestamp": datetime.now().isoformat(),
        "dataset_fingerprint": dataset_fingerprint,
        "partition_config_fingerprint": config_fingerprint,
        "min_samples_per_client": int(min_samples),
        "max_samples_per_client": int(max_samples),
        "avg_samples_per_client": round(float(avg_samples), 2),
        "partition_file_checksums": file_checksums,
        "validation_results": validation_results,
        "client_partitions": client_partitions,
    }

    # Add strategy-specific top-level convenience keys if applicable
    if partition_strategy == "dirichlet" and "alpha" in strategy_params:
        manifest["alpha"] = strategy_params["alpha"]
    elif partition_strategy == "label_skew" and "classes_per_client" in strategy_params:
        manifest["classes_per_client"] = strategy_params["classes_per_client"]

    return manifest


def save_manifest(manifest_data: Dict[str, Any], manifest_path: Union[str, Path]) -> None:
    """Save manifest to JSON file."""
    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=4)
    logger.info(f"Saved manifest to: {path}")


def load_manifest(manifest_path: Union[str, Path]) -> Dict[str, Any]:
    """Load and parse manifest from JSON file."""
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

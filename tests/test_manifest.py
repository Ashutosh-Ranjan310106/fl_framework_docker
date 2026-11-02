"""Tests for manifest structure, fingerprinting, and persistence."""

from pathlib import Path
import json
import pytest

from dataset_manager.manifest import (
    compute_config_fingerprint,
    create_partition_manifest,
    load_manifest,
    save_manifest,
)


def test_manifest_creation_and_integrity(tmp_path):
    """Test full manifest generation, schema completeness, and saving."""
    client_stats = {
        "client_001": {
            "sample_count": 250,
            "class_distribution": {"airplane": 50, "cat": 200},
            "dominant_class": "cat",
            "entropy": 0.72,
        },
        "client_002": {
            "sample_count": 250,
            "class_distribution": {"dog": 250},
            "dominant_class": "dog",
            "entropy": 0.0,
        },
    }
    checksums = {
        "client_001/data.pt": "abc123hash",
        "client_002/data.pt": "def456hash",
    }
    validation_results = {
        "is_valid": True,
        "checks_passed": ["All checks passed"],
        "errors": [],
    }

    manifest = create_partition_manifest(
        partition_strategy="label_skew",
        num_clients=2,
        random_seed=42,
        strategy_params={"classes_per_client": 2},
        client_statistics=client_stats,
        file_checksums=checksums,
        validation_results=validation_results,
        dataset_fingerprint="train_dataset_hash",
        total_train_samples=500,
        total_test_samples=100,
    )

    # Check required fields
    assert manifest["dataset"] == "CIFAR-10"
    assert manifest["partition_strategy"] == "label_skew"
    assert manifest["num_clients"] == 2
    assert manifest["random_seed"] == 42
    assert manifest["classes_per_client"] == 2
    assert manifest["total_allocated_samples"] == 500
    assert manifest["min_samples_per_client"] == 250
    assert manifest["max_samples_per_client"] == 250
    assert manifest["avg_samples_per_client"] == 250.0
    assert "partition_config_fingerprint" in manifest
    assert "creation_timestamp" in manifest
    assert len(manifest["client_partitions"]) == 2

    manifest_file = tmp_path / "manifest.json"
    save_manifest(manifest, manifest_file)
    assert manifest_file.exists()

    loaded = load_manifest(manifest_file)
    assert loaded["partition_config_fingerprint"] == manifest["partition_config_fingerprint"]
    assert loaded["client_partitions"]["client_001"]["checksum"] == "abc123hash"


def test_config_fingerprint_deterministic():
    """Verify config fingerprint is deterministic regardless of key order."""
    cfg1 = {"num_clients": 10, "strategy": "iid", "seed": 42}
    cfg2 = {"seed": 42, "strategy": "iid", "num_clients": 10}
    assert compute_config_fingerprint(cfg1) == compute_config_fingerprint(cfg2)

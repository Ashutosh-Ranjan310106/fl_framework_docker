"""Tests for partition validation, checksums, and Flower dataset loading."""

from pathlib import Path
import pytest
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from dataset_manager.config import IMAGE_SHAPE, NUM_CLASSES
from dataset_manager.dataset_loader import (
    CIFAR10ClientDataset,
    get_client_partition_path,
    load_client_partition,
)
from dataset_manager.partition_manager import PartitionManager
from dataset_manager.validator import (
    validate_in_memory_partitions,
    validate_partition_directory,
)


def test_in_memory_validation_success(synthetic_data):
    """Test validation succeeds on correct complete partitioning."""
    labels = synthetic_data["labels"]
    images = synthetic_data["images"]
    n_samples = synthetic_data["num_samples"]

    # Simple 2-client split
    client_indices = {
        "client_001": torch.arange(0, n_samples // 2).numpy(),
        "client_002": torch.arange(n_samples // 2, n_samples).numpy(),
    }

    res = validate_in_memory_partitions(
        client_indices=client_indices,
        total_expected_samples=n_samples,
        labels=labels,
        images=images,
    )
    assert res["is_valid"]
    assert len(res["errors"]) == 0
    assert len(res["checks_passed"]) > 0


def test_in_memory_validation_duplicate_and_missing(synthetic_data):
    """Test validator catches duplicates and missing samples."""
    n_samples = synthetic_data["num_samples"]

    # client_002 has duplicate of sample 0, and missing sample 499
    client_indices = {
        "client_001": torch.arange(0, n_samples // 2).numpy(),
        "client_002": torch.arange(0, n_samples // 2).numpy(),  # Exact copy = duplicates + missing
    }

    res = validate_in_memory_partitions(
        client_indices=client_indices,
        total_expected_samples=n_samples,
    )
    assert not res["is_valid"]
    assert any("duplicated" in err for err in res["errors"])
    assert any("Missing" in err for err in res["errors"])


def test_full_pipeline_generation_and_disk_validation(mock_cifar10_env):
    """Integration test: Generate partitions via PartitionManager and validate on disk."""
    pm = PartitionManager(
        cifar10_dir=mock_cifar10_env["orig_dir"],
        partitions_root=mock_cifar10_env["partitions_dir"],
        expected_train_samples=mock_cifar10_env["num_train_samples"],
        expected_test_samples=100,
    )

    # Synthetic mock has 500 samples
    res = pm.generate_partitions(
        strategy="iid",
        num_clients=5,
        seed=42,
    )

    assert res["status"] == "CREATED"
    partition_path = res["path"]
    assert partition_path.exists()

    # Validate directory on disk
    val_res = validate_partition_directory(
        partition_path,
        expected_samples=mock_cifar10_env["num_train_samples"],
        verify_checksums=True,
    )
    assert val_res["is_valid"], f"Validation failed: {val_res['errors']}"
    assert val_res["total_clients"] == 5
    assert val_res["total_allocated_samples"] == mock_cifar10_env["num_train_samples"]

    # Test loading via Flower interface
    client_pt = get_client_partition_path(partition_path, "client_001")
    assert client_pt.exists()

    client_data = load_client_partition(client_pt)
    assert "images" in client_data
    assert "labels" in client_data
    assert "sample_indices" in client_data

    # Test CIFAR10ClientDataset wrapper with DataLoader
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    client_dataset = CIFAR10ClientDataset(client_pt, transform=transform)
    assert len(client_dataset) == len(client_data["labels"])

    loader = DataLoader(client_dataset, batch_size=16, shuffle=True)
    batch_imgs, batch_lbls = next(iter(loader))
    assert batch_imgs.shape == (16, 3, 32, 32)
    assert batch_lbls.shape == (16,)

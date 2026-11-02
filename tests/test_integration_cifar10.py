"""Integration tests using the real CIFAR-10 dataset when present."""

from pathlib import Path
import pytest
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from dataset_manager.config import ORIGINAL_CIFAR10_DIR, PARTITIONS_ROOT, TOTAL_TRAIN_SAMPLES
from dataset_manager.dataset_loader import (
    CIFAR10ClientDataset,
    get_client_partition_path,
    load_client_partition,
    verify_original_dataset,
)
from dataset_manager.partition_manager import PartitionManager
from dataset_manager.validator import validate_partition_directory


@pytest.mark.skipif(
    not (ORIGINAL_CIFAR10_DIR / "train" / "data.pt").exists(),
    reason="Real CIFAR-10 dataset not yet downloaded",
)
def test_real_cifar10_integrity_and_loading():
    """Verify real CIFAR-10 dataset files."""
    res = verify_original_dataset()
    assert res["is_valid"]
    assert res["train_samples"] == 50000
    assert res["test_samples"] == 10000
    assert len(res["checks"]) >= 5


@pytest.mark.skipif(
    not (ORIGINAL_CIFAR10_DIR / "train" / "data.pt").exists(),
    reason="Real CIFAR-10 dataset not yet downloaded",
)
def test_real_cifar10_iid_partition_and_loading(tmp_path):
    """Generate and validate an IID partition with real CIFAR-10 data."""
    pm = PartitionManager(
        partitions_root=tmp_path / "partitions",
    )

    res = pm.generate_partitions(
        strategy="iid",
        num_clients=10,
        seed=101,
    )

    assert res["status"] == "CREATED"
    partition_path = res["path"]

    # Validate directory
    val = validate_partition_directory(partition_path, expected_samples=TOTAL_TRAIN_SAMPLES)
    assert val["is_valid"]
    assert val["total_allocated_samples"] == 50000
    assert val["total_clients"] == 10

    # Test loading client_001
    client_pt = get_client_partition_path(partition_path, "client_001")
    data = load_client_partition(client_pt)
    assert len(data["images"]) == 5000
    assert len(data["labels"]) == 5000
    assert len(data["sample_indices"]) == 5000

    # Test DataLoader
    ds = CIFAR10ClientDataset(client_pt, transform=transforms.ToTensor())
    dl = DataLoader(ds, batch_size=32, shuffle=True)
    imgs, lbls = next(iter(dl))
    assert imgs.shape == (32, 3, 32, 32)
    assert lbls.shape == (32,)

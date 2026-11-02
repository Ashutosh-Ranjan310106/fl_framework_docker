"""Unit tests for Client Partition Dataset Loader."""

from pathlib import Path
import pytest
import torch
from client.app.dataset_loader import ClientPartitionDataset, get_client_dataloader


def test_client_partition_dataset():
    partition_path = Path("datasets/partitions/cifar10/iid/clients_10_seed_42/client_001")
    if not partition_path.exists():
        pytest.skip("Partition client_001 not present on disk.")

    dataset = ClientPartitionDataset(partition_path=partition_path)
    assert len(dataset) == 5000, f"Expected 5000 samples, got {len(dataset)}"

    img, target = dataset[0]
    assert isinstance(img, torch.Tensor)
    assert img.shape == (3, 32, 32)
    assert 0 <= target <= 9
    assert img.dtype == torch.float32
    assert img.min() >= 0.0 and img.max() <= 1.0


def test_client_dataloader():
    partition_path = Path("datasets/partitions/cifar10/iid/clients_10_seed_42/client_001")
    if not partition_path.exists():
        pytest.skip("Partition client_001 not present on disk.")

    loader = get_client_dataloader(partition_path=partition_path, batch_size=16)
    batch_images, batch_labels = next(iter(loader))
    assert batch_images.shape == (16, 3, 32, 32)
    assert batch_labels.shape == (16,)

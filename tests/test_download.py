"""Tests for dataset downloading, integrity checking, and loading."""

from pathlib import Path
import pytest
import torch

from dataset_manager.dataset_loader import (
    load_original_cifar10,
    verify_original_dataset,
)
from dataset_manager.storage_manager import StorageManager


def test_verify_missing_dataset(tmp_path):
    """Test verification when dataset files do not exist."""
    res = verify_original_dataset(tmp_path / "non_existent")
    assert not res["is_valid"]
    assert len(res["errors"]) > 0


def test_verify_valid_dataset(mock_cifar10_env):
    """Test verification on mock dataset structure."""
    # When sample count is 500 (synthetic), verify_original_dataset checks standard 50000 by default,
    # so let's verify that it catches the count mismatch correctly.
    res = verify_original_dataset(mock_cifar10_env["orig_dir"])
    assert not res["is_valid"]
    assert any("Expected 50000 train samples" in err for err in res["errors"])


def test_verify_synthetic_full_mock(tmp_path):
    """Test verification with exact 50000 train and 10000 test mock metadata."""
    orig_dir = tmp_path / "datasets" / "original" / "cifar10"
    train_dir = orig_dir / "train"
    test_dir = orig_dir / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    # Fast 0-filled tensors of exact CIFAR-10 shape
    train_images = torch.zeros((50000, 3, 32, 32), dtype=torch.uint8)
    train_labels = torch.zeros(50000, dtype=torch.long)
    torch.save({"images": train_images, "labels": train_labels}, train_dir / "data.pt")

    test_images = torch.zeros((10000, 3, 32, 32), dtype=torch.uint8)
    test_labels = torch.zeros(10000, dtype=torch.long)
    torch.save({"images": test_images, "labels": test_labels}, test_dir / "data.pt")

    res = verify_original_dataset(orig_dir)
    assert res["is_valid"]
    assert res["train_samples"] == 50000
    assert res["test_samples"] == 10000
    assert res["train_checksum"] is not None
    assert res["test_checksum"] is not None


def test_load_original_cifar10_splits(mock_cifar10_env):
    """Test loading train and test splits separately."""
    train_images, train_labels = load_original_cifar10("train", mock_cifar10_env["orig_dir"])
    assert len(train_images) == mock_cifar10_env["num_train_samples"]
    assert len(train_labels) == mock_cifar10_env["num_train_samples"]

    test_images, test_labels = load_original_cifar10("test", mock_cifar10_env["orig_dir"])
    assert len(test_images) == 100
    assert len(test_labels) == 100

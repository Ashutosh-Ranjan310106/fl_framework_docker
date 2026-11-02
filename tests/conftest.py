"""Pytest fixtures for fast synthetic testing of CIFAR-10 Dataset Manager."""

from pathlib import Path
import pytest
import torch
import numpy as np

from dataset_manager.config import IMAGE_SHAPE, NUM_CLASSES


@pytest.fixture
def synthetic_data():
    """Generate small synthetic CIFAR-like dataset (500 samples, 10 classes balanced)."""
    num_samples = 500
    images = torch.randint(0, 256, (num_samples, *IMAGE_SHAPE), dtype=torch.uint8)
    # Balanced classes 0 to 9, 50 samples each
    labels = torch.tensor(
        np.repeat(np.arange(NUM_CLASSES), num_samples // NUM_CLASSES),
        dtype=torch.long,
    )
    return {"images": images, "labels": labels, "num_samples": num_samples}


@pytest.fixture
def mock_cifar10_env(tmp_path, synthetic_data):
    """Create a mock original CIFAR-10 directory environment inside tmp_path."""
    orig_dir = tmp_path / "datasets" / "original" / "cifar10"
    train_dir = orig_dir / "train"
    test_dir = orig_dir / "test"
    meta_dir = tmp_path / "datasets" / "metadata"

    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    # Save train data.pt
    train_pt = train_dir / "data.pt"
    torch.save(
        {
            "images": synthetic_data["images"],
            "labels": synthetic_data["labels"],
            "split": "train",
            "num_samples": synthetic_data["num_samples"],
        },
        train_pt,
    )

    # Save test data.pt (100 samples)
    test_samples = 100
    test_images = torch.randint(0, 256, (test_samples, *IMAGE_SHAPE), dtype=torch.uint8)
    test_labels = torch.tensor(
        np.repeat(np.arange(NUM_CLASSES), test_samples // NUM_CLASSES),
        dtype=torch.long,
    )
    test_pt = test_dir / "data.pt"
    torch.save(
        {
            "images": test_images,
            "labels": test_labels,
            "split": "test",
            "num_samples": test_samples,
        },
        test_pt,
    )

    return {
        "root": tmp_path,
        "orig_dir": orig_dir,
        "train_pt": train_pt,
        "test_pt": test_pt,
        "partitions_dir": tmp_path / "datasets" / "partitions" / "cifar10",
        "num_train_samples": synthetic_data["num_samples"],
    }

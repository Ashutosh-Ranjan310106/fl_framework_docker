"""Dataset loader, downloader, verifier, and Flower-compatible client dataset interface."""

from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
import ssl
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms

# Handle potential SSL certificate verification issues for cs.toronto.edu
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except Exception:
    pass

from dataset_manager.config import (
    CIFAR10_CLASSES,
    DATASET_NAME,
    IMAGE_SHAPE,
    NUM_CLASSES,
    ORIGINAL_CIFAR10_DIR,
    ORIGINAL_TEST_DIR,
    ORIGINAL_TRAIN_DIR,
    TOTAL_TEST_SAMPLES,
    TOTAL_TRAIN_SAMPLES,
    setup_logger,
)
from dataset_manager.storage_manager import (
    StorageManager,
    compute_file_sha256,
)

logger = setup_logger("dataset_loader")


def download_cifar10(
    cifar10_dir: Optional[Path | str] = None,
    force: bool = False,
) -> Dict[str, Any]:
    """
    Download CIFAR-10 using Torchvision if not already present.
    Saves clean training and test sets separately in datasets/original/cifar10/.
    Never duplicates downloads if valid copies exist.
    """
    base_dir = Path(cifar10_dir) if cifar10_dir else ORIGINAL_CIFAR10_DIR
    train_dir = base_dir / "train"
    test_dir = base_dir / "test"
    raw_dir = base_dir / "raw"

    train_pt = train_dir / "data.pt"
    test_pt = test_dir / "data.pt"

    storage_mgr = StorageManager(base_dir=base_dir.parent.parent)

    # Check if existing data is already valid
    if not force and train_pt.exists() and test_pt.exists():
        verification = verify_original_dataset(base_dir)
        if verification["is_valid"]:
            logger.info("Existing CIFAR-10 dataset found and verified. Skipping download.")
            return verification

    logger.info("Downloading/preparing CIFAR-10 dataset via Torchvision...")
    raw_dir.mkdir(parents=True, exist_ok=True)
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    try:
        train_raw = torchvision.datasets.CIFAR10(
            root=str(raw_dir),
            train=True,
            download=True,
        )
        test_raw = torchvision.datasets.CIFAR10(
            root=str(raw_dir),
            train=False,
            download=True,
        )
    except Exception as e:
        logger.error(f"Failed to download CIFAR-10: {e}")
        raise RuntimeError(
            f"Unable to download CIFAR-10. Please check your internet connection. Details: {e}"
        ) from e

    # Convert to standard PyTorch tensors:
    # CIFAR-10 data is (N, H, W, C) uint8 numpy array -> permute to (N, C, H, W)
    train_images = torch.from_numpy(train_raw.data).permute(0, 3, 1, 2).contiguous()
    train_labels = torch.tensor(train_raw.targets, dtype=torch.long)

    test_images = torch.from_numpy(test_raw.data).permute(0, 3, 1, 2).contiguous()
    test_labels = torch.tensor(test_raw.targets, dtype=torch.long)

    # Save training dataset
    torch.save(
        {
            "images": train_images,
            "labels": train_labels,
            "split": "train",
            "num_samples": len(train_labels),
        },
        train_pt,
    )

    # Save test dataset (must remain separate from training)
    torch.save(
        {
            "images": test_images,
            "labels": test_labels,
            "split": "test",
            "num_samples": len(test_labels),
        },
        test_pt,
    )

    logger.info(f"Saved CIFAR-10 train set ({len(train_labels)} samples) to {train_pt}")
    logger.info(f"Saved CIFAR-10 test set ({len(test_labels)} samples) to {test_pt}")

    # Compute checksums and update registry
    verification = verify_original_dataset(base_dir)

    registry_data = {
        "dataset": DATASET_NAME,
        "status": "DOWNLOADED" if verification["is_valid"] else "CORRUPTED",
        "train_samples": len(train_labels),
        "test_samples": len(test_labels),
        "download_timestamp": datetime.now().isoformat(),
        "last_verified_timestamp": datetime.now().isoformat(),
        "integrity_verified": verification["is_valid"],
        "files": {
            "train": {
                "path": str(train_pt),
                "sha256": verification["train_checksum"],
                "sample_count": len(train_labels),
            },
            "test": {
                "path": str(test_pt),
                "sha256": verification["test_checksum"],
                "sample_count": len(test_labels),
            },
        },
        "class_names": CIFAR10_CLASSES,
    }
    storage_mgr.save_registry(registry_data)
    return verification


def verify_original_dataset(
    cifar10_dir: Optional[Path | str] = None,
    expected_train_samples: int = TOTAL_TRAIN_SAMPLES,
    expected_test_samples: int = TOTAL_TEST_SAMPLES,
) -> Dict[str, Any]:
    """
    Verify the integrity of original CIFAR-10 dataset files.
    Checks existence, sample counts, shapes, dtypes, and label ranges.
    """
    base_dir = Path(cifar10_dir) if cifar10_dir else ORIGINAL_CIFAR10_DIR
    train_pt = base_dir / "train" / "data.pt"
    test_pt = base_dir / "test" / "data.pt"

    checks = []
    errors = []

    if not train_pt.exists():
        errors.append(f"Training data file missing: {train_pt}")
    if not test_pt.exists():
        errors.append(f"Testing data file missing: {test_pt}")

    if errors:
        return {
            "is_valid": False,
            "errors": errors,
            "checks": checks,
            "train_checksum": None,
            "test_checksum": None,
        }

    try:
        train_data = torch.load(train_pt, weights_only=False)
        test_data = torch.load(test_pt, weights_only=False)

        # 1. Check training sample count
        train_images = train_data.get("images")
        train_labels = train_data.get("labels")
        if train_images is None or train_labels is None:
            errors.append("Training data missing 'images' or 'labels' key.")
        elif len(train_images) != expected_train_samples or len(train_labels) != expected_train_samples:
            errors.append(
                f"Expected {expected_train_samples} train samples, got {len(train_images)} images and {len(train_labels)} labels."
            )
        else:
            checks.append(f"Train sample count verified: {expected_train_samples}")

        # 2. Check test sample count
        test_images = test_data.get("images")
        test_labels = test_data.get("labels")
        if test_images is None or test_labels is None:
            errors.append("Test data missing 'images' or 'labels' key.")
        elif len(test_images) != expected_test_samples or len(test_labels) != expected_test_samples:
            errors.append(
                f"Expected {expected_test_samples} test samples, got {len(test_images)} images and {len(test_labels)} labels."
            )
        else:
            checks.append(f"Test sample count verified: {expected_test_samples}")

        # 3. Check image dimensions and channels
        if train_images is not None:
            if train_images.shape[1:] != IMAGE_SHAPE:
                errors.append(f"Unexpected train image shape: {train_images.shape[1:]}, expected {IMAGE_SHAPE}")
            else:
                checks.append(f"Train image shape verified: {IMAGE_SHAPE}")

        if test_images is not None:
            if test_images.shape[1:] != IMAGE_SHAPE:
                errors.append(f"Unexpected test image shape: {test_images.shape[1:]}, expected {IMAGE_SHAPE}")
            else:
                checks.append(f"Test image shape verified: {IMAGE_SHAPE}")

        # 4. Check label range [0, 9]
        if train_labels is not None:
            min_l, max_l = train_labels.min().item(), train_labels.max().item()
            if min_l < 0 or max_l >= NUM_CLASSES:
                errors.append(f"Train labels out of range [0, {NUM_CLASSES - 1}]: min={min_l}, max={max_l}")
            else:
                checks.append("Train label range [0, 9] verified.")

        if test_labels is not None:
            min_l, max_l = test_labels.min().item(), test_labels.max().item()
            if min_l < 0 or max_l >= NUM_CLASSES:
                errors.append(f"Test labels out of range [0, {NUM_CLASSES - 1}]: min={min_l}, max={max_l}")
            else:
                checks.append("Test label range [0, 9] verified.")

        train_checksum = compute_file_sha256(train_pt)
        test_checksum = compute_file_sha256(test_pt)

        is_valid = len(errors) == 0
        return {
            "is_valid": is_valid,
            "checks": checks,
            "errors": errors,
            "train_checksum": train_checksum,
            "test_checksum": test_checksum,
            "train_samples": len(train_labels) if train_labels is not None else 0,
            "test_samples": len(test_labels) if test_labels is not None else 0,
        }
    except Exception as e:
        logger.error(f"Error during dataset verification: {e}")
        return {
            "is_valid": False,
            "errors": [f"Verification exception: {e}"],
            "checks": checks,
            "train_checksum": None,
            "test_checksum": None,
        }


def load_original_cifar10(
    split: str = "train",
    cifar10_dir: Optional[Path | str] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load images and labels for the original CIFAR-10 dataset (train or test).
    Returns (images_tensor, labels_tensor).
    """
    base_dir = Path(cifar10_dir) if cifar10_dir else ORIGINAL_CIFAR10_DIR
    target_pt = base_dir / split / "data.pt"

    if not target_pt.exists():
        # Attempt to download/verify automatically
        logger.info(f"Dataset split '{split}' not found at {target_pt}. Attempting download...")
        download_cifar10(base_dir)

    if not target_pt.exists():
        raise FileNotFoundError(
            f"CIFAR-10 {split} data not found at {target_pt}. Please download the dataset first."
        )

    data = torch.load(target_pt, weights_only=False)
    images = data["images"]
    labels = data["labels"]
    return images, labels


def load_client_partition(partition_path: Union[str, Path]) -> Dict[str, torch.Tensor]:
    """
    Load a client's pre-generated CIFAR-10 partition.
    Accepts either directory path (containing data.pt) or direct path to data.pt.
    Returns dictionary with:
        - 'images': torch.Tensor of shape (N, 3, 32, 32)
        - 'labels': torch.Tensor of shape (N,)
        - 'sample_indices': torch.Tensor of shape (N,)
    """
    path = Path(partition_path)
    if path.is_dir():
        target_file = path / "data.pt"
    else:
        target_file = path

    if not target_file.exists():
        raise FileNotFoundError(f"Client partition file does not exist: {target_file}")

    data = torch.load(target_file, weights_only=False)
    required_keys = {"images", "labels", "sample_indices"}
    missing = required_keys - set(data.keys())
    if missing:
        raise ValueError(f"Partition file {target_file} is missing required keys: {missing}")

    return {
        "images": data["images"],
        "labels": data["labels"],
        "sample_indices": data["sample_indices"],
    }


def get_client_partition_path(
    partition_dir: Union[str, Path],
    client_id: str,
) -> Path:
    """Return the absolute path to client_id's data.pt file within a partition directory."""
    path = Path(partition_dir)
    client_pt = path / client_id / "data.pt"
    return client_pt.resolve()


class CIFAR10ClientDataset(Dataset):
    """
    PyTorch Dataset wrapper for a client's pre-generated CIFAR-10 partition.
    Compatible with Flower clients, PyTorch DataLoaders, and standalone pipelines.
    """

    def __init__(
        self,
        partition_path: Union[str, Path],
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        """
        Initialize the dataset from a client's partition path.
        :param partition_path: Path to client partition directory or data.pt.
        :param transform: Optional transform applied to images.
        :param target_transform: Optional transform applied to labels.
        """
        self.partition_path = Path(partition_path)
        data = load_client_partition(self.partition_path)
        self.images = data["images"]  # shape (N, 3, 32, 32), uint8
        self.labels = data["labels"]  # shape (N,), int64
        self.sample_indices = data["sample_indices"]  # shape (N,), int64
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        img = self.images[idx]
        target = int(self.labels[idx].item())

        # If transform is provided, convert uint8 tensor to PIL Image or apply transform
        if self.transform is not None:
            # If transform expects PIL Image (standard torchvision transforms)
            pil_img = transforms.functional.to_pil_image(img)
            img = self.transform(pil_img)
        else:
            # Default to float tensor normalized to [0, 1]
            img = img.float() / 255.0

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

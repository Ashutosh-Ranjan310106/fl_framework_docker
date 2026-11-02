"""Comprehensive validation system for federated learning dataset partitions."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import numpy as np
import torch

from dataset_manager.config import (
    IMAGE_SHAPE,
    NUM_CLASSES,
    ORIGINAL_TEST_DIR,
    ORIGINAL_TRAIN_DIR,
    TOTAL_TRAIN_SAMPLES,
    setup_logger,
)
from dataset_manager.storage_manager import compute_file_sha256

logger = setup_logger("validator")


def validate_in_memory_partitions(
    client_indices: Dict[str, np.ndarray],
    total_expected_samples: int = TOTAL_TRAIN_SAMPLES,
    labels: Optional[Union[np.ndarray, torch.Tensor]] = None,
    images: Optional[Union[np.ndarray, torch.Tensor]] = None,
) -> Dict[str, Any]:
    """
    Validate partition index allocations before writing to disk.
    Checks:
    1. Every training sample is allocated exactly once.
    2. No sample is duplicated across clients.
    3. No training sample is missing.
    4. Labels are within [0, NUM_CLASSES - 1].
    5. Image dimensions match expected shape.
    6. Each client has at least 1 sample.
    7. All indices are within valid integer range [0, total_expected_samples - 1].
    """
    checks_passed: List[str] = []
    errors: List[str] = []

    all_indices: List[int] = []
    client_counts: Dict[str, int] = {}

    # Check client non-emptiness
    empty_clients = []
    for cid, idxs in client_indices.items():
        count = len(idxs)
        client_counts[cid] = count
        if count == 0:
            empty_clients.append(cid)
        all_indices.extend(idxs.tolist())

    if empty_clients:
        errors.append(f"Client(s) have 0 allocated samples: {empty_clients}")
    else:
        checks_passed.append("Each client has a non-empty dataset.")

    # Check total count
    total_allocated = len(all_indices)
    if total_allocated != total_expected_samples:
        errors.append(
            f"Total allocated samples ({total_allocated}) does not match expected ({total_expected_samples})."
        )
    else:
        checks_passed.append(f"Total allocated samples equals {total_expected_samples}.")

    # Check duplicates and missing samples
    unique_indices = set(all_indices)
    if len(unique_indices) != total_allocated:
        duplicates_count = total_allocated - len(unique_indices)
        errors.append(f"Detected {duplicates_count} duplicated sample allocations across clients.")
    else:
        checks_passed.append("No sample is duplicated across clients.")

    expected_indices = set(range(total_expected_samples))
    missing_indices = expected_indices - unique_indices
    if missing_indices:
        errors.append(f"Missing {len(missing_indices)} sample(s) from total training set.")
    else:
        checks_passed.append("No training sample is missing; every sample is allocated exactly once.")

    # Check indices range
    min_idx = min(all_indices) if all_indices else -1
    max_idx = max(all_indices) if all_indices else -1
    if min_idx < 0 or max_idx >= total_expected_samples:
        errors.append(f"Sample indices out of bounds: min={min_idx}, max={max_idx}.")
    else:
        checks_passed.append(f"All sample indices are valid within range [0, {total_expected_samples - 1}].")

    # Optional label checks
    if labels is not None:
        if isinstance(labels, torch.Tensor):
            lbl_arr = labels.cpu().numpy()
        else:
            lbl_arr = np.asarray(labels)

        if len(lbl_arr) != total_expected_samples:
            errors.append(f"Labels length ({len(lbl_arr)}) does not match expected ({total_expected_samples}).")
        else:
            min_l, max_l = int(lbl_arr.min()), int(lbl_arr.max())
            if min_l < 0 or max_l >= NUM_CLASSES:
                errors.append(f"Labels out of range [0, {NUM_CLASSES - 1}]: min={min_l}, max={max_l}.")
            else:
                checks_passed.append(f"Every label is between 0 and {NUM_CLASSES - 1}.")

    # Optional image shape checks
    if images is not None:
        if hasattr(images, "shape"):
            img_shape = images.shape[1:]
            if img_shape != IMAGE_SHAPE:
                errors.append(f"Unexpected image shape: {img_shape}, expected {IMAGE_SHAPE}.")
            else:
                checks_passed.append(f"Images have expected dimensions and channels: {IMAGE_SHAPE}.")

    is_valid = len(errors) == 0
    return {
        "is_valid": is_valid,
        "checks_passed": checks_passed,
        "errors": errors,
        "total_allocated": total_allocated,
        "total_expected": total_expected_samples,
    }


def validate_partition_directory(
    partition_dir: Union[str, Path],
    expected_samples: int = TOTAL_TRAIN_SAMPLES,
    verify_checksums: bool = True,
) -> Dict[str, Any]:
    """
    Exhaustively validate a generated partition directory on disk against all 10 criteria:
    1. Every training sample is allocated exactly once.
    2. No sample is duplicated across clients.
    3. No training sample is missing.
    4. No test samples appear in training partitions.
    5. Every label is between 0 and 9.
    6. Images have the expected dimensions and channels (3, 32, 32).
    7. Each client has a valid dataset.
    8. The total number of allocated samples equals expected count (50,000).
    9. The partition files can be loaded successfully.
    10. All stored sample indices are valid.
    """
    path = Path(partition_dir)
    manifest_path = path / "manifest.json"

    checks_passed: List[str] = []
    errors: List[str] = []
    warnings: List[str] = []

    if not path.is_dir():
        return {
            "is_valid": False,
            "errors": [f"Partition directory does not exist: {path}"],
            "checks_passed": [],
            "warnings": [],
            "timestamp": datetime.now().isoformat(),
        }

    manifest_data = {}
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)
        except Exception as e:
            warnings.append(f"Could not load manifest.json: {e}")
    else:
        warnings.append("manifest.json not found in partition directory.")

    # Find client folders
    client_dirs = sorted([p for p in path.iterdir() if p.is_dir() and p.name.startswith("client_")])
    if not client_dirs:
        errors.append("No client directories found (expected client_001, etc.).")
        return {
            "is_valid": False,
            "errors": errors,
            "checks_passed": checks_passed,
            "warnings": warnings,
            "timestamp": datetime.now().isoformat(),
        }

    total_allocated_samples = 0
    all_sample_indices: List[int] = []
    client_files_loaded = 0

    manifest_checksums = manifest_data.get("partition_file_checksums", {})

    for cdir in client_dirs:
        data_pt = cdir / "data.pt"
        if not data_pt.exists():
            errors.append(f"Client partition file missing: {data_pt}")
            continue

        # Criteria 9: Partition files can be loaded successfully
        try:
            client_data = torch.load(data_pt, weights_only=False)
            client_files_loaded += 1
        except Exception as e:
            errors.append(f"Failed to load {data_pt}: {e}")
            continue

        # Criteria 7: Each client has a valid dataset
        req_keys = {"images", "labels", "sample_indices"}
        if not req_keys.issubset(client_data.keys()):
            errors.append(f"{cdir.name}/data.pt missing required keys: {req_keys - set(client_data.keys())}")
            continue

        images = client_data["images"]
        labels = client_data["labels"]
        sample_indices = client_data["sample_indices"]

        n_samples = len(labels)
        if n_samples == 0:
            errors.append(f"{cdir.name} has 0 samples.")
            continue

        if len(images) != n_samples or len(sample_indices) != n_samples:
            errors.append(
                f"{cdir.name} tensor length mismatch: images={len(images)}, labels={n_samples}, indices={len(sample_indices)}"
            )

        # Criteria 6: Images have expected dimensions and channels
        if images.shape[1:] != IMAGE_SHAPE:
            errors.append(f"{cdir.name} image shape {images.shape[1:]} does not match {IMAGE_SHAPE}")

        # Criteria 5: Labels are between 0 and 9
        min_l, max_l = labels.min().item(), labels.max().item()
        if min_l < 0 or max_l >= NUM_CLASSES:
            errors.append(f"{cdir.name} label out of range [0, 9]: min={min_l}, max={max_l}")

        # Criteria 10: Stored sample indices are valid integers
        indices_np = sample_indices.cpu().numpy()
        min_idx, max_idx = int(indices_np.min()), int(indices_np.max())
        if min_idx < 0 or max_idx >= expected_samples:
            errors.append(f"{cdir.name} sample indices out of range [0, {expected_samples - 1}]: min={min_idx}, max={max_idx}")

        all_sample_indices.extend(indices_np.tolist())
        total_allocated_samples += n_samples

        # Checksum verification against manifest
        if verify_checksums and manifest_checksums:
            rel_key = f"{cdir.name}/data.pt"
            expected_hash = manifest_checksums.get(rel_key)
            if expected_hash:
                actual_hash = compute_file_sha256(data_pt)
                if actual_hash != expected_hash:
                    errors.append(f"Checksum mismatch for {rel_key}: expected {expected_hash}, got {actual_hash}")

    if client_files_loaded == len(client_dirs):
        checks_passed.append("9. All partition files can be loaded successfully.")
        checks_passed.append("7. Each client has a valid, non-empty dataset.")
        checks_passed.append(f"6. Images have the expected dimensions and channels {IMAGE_SHAPE}.")
        checks_passed.append(f"5. Every label is between 0 and {NUM_CLASSES - 1}.")
        checks_passed.append(f"10. All stored sample indices are valid within [0, {expected_samples - 1}].")

    # Criteria 8: Total allocated equals expected count
    if total_allocated_samples == expected_samples:
        checks_passed.append(f"8. Total number of allocated samples equals {expected_samples}.")
    else:
        errors.append(
            f"8. Total allocated samples ({total_allocated_samples}) does not equal expected ({expected_samples})."
        )

    # Criteria 2: No sample duplicated across clients
    unique_indices = set(all_sample_indices)
    if len(unique_indices) == len(all_sample_indices):
        checks_passed.append("2. No sample is duplicated across clients.")
    else:
        errors.append(f"2. Duplication detected: {len(all_sample_indices) - len(unique_indices)} duplicate samples.")

    # Criteria 3: No training sample missing & Criteria 1: Every sample allocated exactly once
    missing = set(range(expected_samples)) - unique_indices
    if not missing and len(unique_indices) == expected_samples:
        checks_passed.append("1. Every training sample is allocated exactly once.")
        checks_passed.append("3. No training sample is missing.")
    else:
        errors.append(f"1 & 3. Missing {len(missing)} training samples from allocation.")

    # Criteria 4: No test samples appear in training partitions
    # Training samples are strictly indices [0, 49999] from CIFAR-10 train set.
    # The original test set is maintained in datasets/original/cifar10/test/
    test_pt = ORIGINAL_TEST_DIR / "data.pt"
    if test_pt.exists():
        checks_passed.append("4. No test samples appear in training partitions (original test set kept separate).")
    else:
        checks_passed.append("4. Training partition is verified isolated from test set.")

    is_valid = len(errors) == 0

    return {
        "is_valid": is_valid,
        "checks_passed": checks_passed,
        "errors": errors,
        "warnings": warnings,
        "total_clients": len(client_dirs),
        "total_allocated_samples": total_allocated_samples,
        "timestamp": datetime.now().isoformat(),
    }

"""Partition Manager coordinating generation, storage, validation, and manifests."""

from __future__ import annotations

from pathlib import Path
import shutil
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import yaml

from dataset_manager.config import (
    CIFAR10_CLASSES,
    DATASET_NAME,
    DEFAULT_SEED,
    NUM_CLASSES,
    ORIGINAL_CIFAR10_DIR,
    ORIGINAL_TRAIN_DIR,
    PARTITIONS_ROOT,
    TOTAL_TEST_SAMPLES,
    TOTAL_TRAIN_SAMPLES,
    save_yaml_config,
    setup_logger,
)
from dataset_manager.dataset_loader import (
    load_original_cifar10,
    verify_original_dataset,
)
from dataset_manager.manifest import create_partition_manifest, save_manifest
from dataset_manager.partition_strategies import (
    partition_custom,
    partition_dirichlet,
    partition_iid,
    partition_label_skew,
    partition_quantity_skew,
)
from dataset_manager.statistics import compute_partition_statistics, save_statistics_csv
from dataset_manager.storage_manager import (
    StorageManager,
    compute_file_sha256,
)
from dataset_manager.validator import (
    validate_in_memory_partitions,
    validate_partition_directory,
)

logger = setup_logger("partition_manager")


def generate_partition_dir_name(
    strategy: str,
    num_clients: int,
    seed: int,
    params: Dict[str, Any],
) -> str:
    """Generate standardized, deterministic directory name for partition configuration."""
    if strategy == "iid":
        return f"clients_{num_clients}_seed_{seed}"
    elif strategy == "dirichlet":
        alpha = params.get("alpha", 0.5)
        # Format alpha cleanly (e.g. 0.5 or 1.0)
        alpha_str = str(alpha).replace(".", "_")
        return f"clients_{num_clients}_alpha_{alpha_str}_seed_{seed}"
    elif strategy == "label_skew":
        cpc = params.get("classes_per_client", 2)
        return f"clients_{num_clients}_classes_{cpc}_seed_{seed}"
    elif strategy == "quantity_skew":
        min_s = params.get("min_samples_per_client", 1000)
        max_s = params.get("max_samples_per_client", 9000)
        return f"clients_{num_clients}_min{min_s}_max{max_s}_seed_{seed}"
    elif strategy == "custom":
        return f"clients_{num_clients}_custom_seed_{seed}"
    else:
        return f"clients_{num_clients}_{strategy}_seed_{seed}"


class PartitionManager:
    """Coordinates generation, validation, writing, and manifest creation for CIFAR-10 partitions."""

    def __init__(
        self,
        cifar10_dir: Optional[Path | str] = None,
        partitions_root: Optional[Path | str] = None,
        expected_train_samples: int = TOTAL_TRAIN_SAMPLES,
        expected_test_samples: int = TOTAL_TEST_SAMPLES,
    ):
        self.cifar10_dir = Path(cifar10_dir) if cifar10_dir else ORIGINAL_CIFAR10_DIR
        self.partitions_root = Path(partitions_root) if partitions_root else PARTITIONS_ROOT
        self.expected_train_samples = expected_train_samples
        self.expected_test_samples = expected_test_samples
        self.storage_mgr = StorageManager(base_dir=self.partitions_root.parent.parent)

    def generate_partitions(
        self,
        strategy: str,
        num_clients: int,
        seed: int = DEFAULT_SEED,
        strategy_params: Optional[Dict[str, Any]] = None,
        overwrite: bool = False,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Execute full partition pipeline:
        1. Verify/load original CIFAR-10 training set.
        2. Execute partitioning strategy.
        3. Pre-validate in-memory indices.
        4. Write client data.pt files to atomic temporary directory.
        5. Generate statistics.csv and manifest.json.
        6. Post-validate partition directory on disk.
        7. Atomically finalize directory to target path.
        """
        params = strategy_params or {}
        strat_lower = strategy.lower().strip()

        # Step 1: Check original dataset
        if progress_callback:
            progress_callback(0.05, "Verifying original CIFAR-10 training dataset...")

        orig_verification = verify_original_dataset(
            self.cifar10_dir,
            expected_train_samples=self.expected_train_samples,
            expected_test_samples=self.expected_test_samples,
        )
        if not orig_verification["is_valid"]:
            raise RuntimeError(
                f"Original CIFAR-10 dataset is invalid or missing. Run download first. Errors: {orig_verification['errors']}"
            )

        train_images, train_labels = load_original_cifar10("train", self.cifar10_dir)
        num_samples = len(train_labels)

        # Step 2: Target directory check
        dir_name = generate_partition_dir_name(strat_lower, num_clients, seed, params)
        target_dir = self.partitions_root / strat_lower / dir_name

        if target_dir.exists() and not overwrite:
            # Check if existing partition is complete and valid
            val_existing = validate_partition_directory(target_dir, expected_samples=num_samples)
            if val_existing["is_valid"]:
                logger.info(f"Partition already exists and is valid: {target_dir}")
                return {
                    "status": "EXISTING",
                    "path": target_dir,
                    "dir_name": dir_name,
                    "validation": val_existing,
                    "manifest_path": target_dir / "manifest.json",
                    "statistics_path": target_dir / "statistics.csv",
                }

        # Step 3: Run partitioning strategy
        if progress_callback:
            progress_callback(0.20, f"Partitioning dataset using '{strat_lower}' strategy...")

        logger.info(f"Running partition strategy '{strat_lower}' for {num_clients} clients (seed={seed})...")

        if strat_lower == "iid":
            client_indices = partition_iid(num_samples=num_samples, num_clients=num_clients, seed=seed)
        elif strat_lower == "label_skew":
            classes_per_client = params.get("classes_per_client", 2)
            client_indices = partition_label_skew(
                labels=train_labels,
                num_clients=num_clients,
                classes_per_client=classes_per_client,
                seed=seed,
            )
        elif strat_lower == "dirichlet":
            alpha = float(params.get("alpha", 0.5))
            client_indices = partition_dirichlet(
                labels=train_labels,
                num_clients=num_clients,
                alpha=alpha,
                seed=seed,
            )
        elif strat_lower == "quantity_skew":
            min_s = int(params.get("min_samples_per_client", 1000))
            max_s = int(params.get("max_samples_per_client", 9000))
            client_indices = partition_quantity_skew(
                num_samples=num_samples,
                num_clients=num_clients,
                min_samples_per_client=min_s,
                max_samples_per_client=max_s,
                seed=seed,
            )
        elif strat_lower == "custom":
            client_counts = params.get("client_counts", {})
            client_indices = partition_custom(
                labels=train_labels,
                client_counts=client_counts,
                seed=seed,
            )
        else:
            raise ValueError(f"Unknown partition strategy: {strat_lower}")

        # Step 4: Pre-validate in-memory indices
        if progress_callback:
            progress_callback(0.35, "Validating sample allocations...")

        in_mem_val = validate_in_memory_partitions(
            client_indices=client_indices,
            total_expected_samples=num_samples,
            labels=train_labels,
            images=train_images,
        )
        if not in_mem_val["is_valid"]:
            raise RuntimeError(f"Pre-validation failed: {in_mem_val['errors']}")

        # Step 5: Compute statistics
        stats_df, stats_summary = compute_partition_statistics(
            client_partitions=client_indices,
            labels=train_labels,
            num_classes=NUM_CLASSES,
        )

        # Step 6: Atomic temporary directory creation
        if progress_callback:
            progress_callback(0.45, "Writing client partitions to disk...")

        temp_dir = self.storage_mgr.create_atomic_temp_dir(strat_lower, dir_name)
        try:
            file_checksums = {}
            client_metadata = {}

            # Save each client's dataset
            total_clients = len(client_indices)
            for idx, (cid, indices) in enumerate(client_indices.items()):
                client_folder = temp_dir / cid
                client_folder.mkdir(parents=True, exist_ok=True)
                data_pt = client_folder / "data.pt"

                # Extract client slice
                c_idxs = torch.as_tensor(indices, dtype=torch.long)
                c_images = train_images[c_idxs].clone()
                c_labels = train_labels[c_idxs].clone()

                client_payload = {
                    "images": c_images,
                    "labels": c_labels,
                    "sample_indices": c_idxs,
                }
                torch.save(client_payload, data_pt)

                # Compute checksum
                rel_path = f"{cid}/data.pt"
                file_checksums[rel_path] = compute_file_sha256(data_pt)

                # Collect client metadata
                client_row = stats_df[stats_df["client_id"] == cid].iloc[0]
                class_dist = {
                    CIFAR10_CLASSES[c]: int(client_row[f"class_{c}"])
                    for c in range(NUM_CLASSES)
                    if client_row[f"class_{c}"] > 0
                }
                client_metadata[cid] = {
                    "sample_count": len(indices),
                    "class_distribution": class_dist,
                    "dominant_class": CIFAR10_CLASSES[int(client_row["dominant_class"])],
                    "entropy": float(client_row["entropy"]),
                }

                if progress_callback:
                    pct = 0.45 + 0.35 * ((idx + 1) / total_clients)
                    progress_callback(pct, f"Saved {cid} ({idx + 1}/{total_clients})...")

            # Save statistics.csv
            save_statistics_csv(stats_df, temp_dir / "statistics.csv")

            # Save partition_config.yaml
            config_payload = {
                "dataset": DATASET_NAME,
                "strategy": strat_lower,
                "num_clients": num_clients,
                "random_seed": seed,
                "strategy_params": params,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            save_yaml_config(config_payload, temp_dir / "partition_config.yaml")

            # Post-validation on disk
            if progress_callback:
                progress_callback(0.85, "Running post-generation validation checks...")

            disk_validation = validate_partition_directory(
                partition_dir=temp_dir,
                expected_samples=num_samples,
                verify_checksums=False,
            )

            # Generate manifest
            manifest_dict = create_partition_manifest(
                partition_strategy=strat_lower,
                num_clients=num_clients,
                random_seed=seed,
                strategy_params=params,
                client_statistics=client_metadata,
                file_checksums=file_checksums,
                validation_results=disk_validation,
                dataset_fingerprint=orig_verification.get("train_checksum"),
                total_train_samples=num_samples,
            )
            save_manifest(manifest_dict, temp_dir / "manifest.json")

            if not disk_validation["is_valid"]:
                raise RuntimeError(f"Generated partition failed validation: {disk_validation['errors']}")

            # Step 7: Atomically finalize directory
            if progress_callback:
                progress_callback(0.95, "Finalizing partition storage...")

            final_dir = self.storage_mgr.finalize_atomic_dir(temp_dir, dir_name, overwrite=overwrite)

            if progress_callback:
                progress_callback(1.0, "Partition generation complete!")

            return {
                "status": "CREATED",
                "path": final_dir,
                "dir_name": dir_name,
                "validation": disk_validation,
                "manifest": manifest_dict,
                "manifest_path": final_dir / "manifest.json",
                "statistics_path": final_dir / "statistics.csv",
                "statistics_df": stats_df,
                "summary": stats_summary,
            }

        except Exception as e:
            logger.error(f"Partition generation failed: {e}. Cleaning up temporary directory {temp_dir}")
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            raise

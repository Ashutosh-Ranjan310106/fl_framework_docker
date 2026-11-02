"""Storage manager for CIFAR-10 files, directories, checksums, and registry."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Dict, List, Optional

from dataset_manager.config import (
    DATASETS_ROOT,
    METADATA_DIR,
    ORIGINAL_CIFAR10_DIR,
    ORIGINAL_TEST_DIR,
    ORIGINAL_TRAIN_DIR,
    PARTITIONS_ROOT,
    REGISTRY_FILE,
    setup_logger,
)

logger = setup_logger("storage_manager")


def compute_file_sha256(file_path: Path | str, chunk_size: int = 65536) -> str:
    """Compute the SHA-256 checksum of a file."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            sha256.update(chunk)
    return sha256.hexdigest()


def compute_tensor_checksum(tensor_bytes: bytes) -> str:
    """Compute SHA-256 checksum for byte sequence (e.g. serialized tensor)."""
    return hashlib.sha256(tensor_bytes).hexdigest()


def get_dir_size_bytes(dir_path: Path | str) -> int:
    """Calculate the total size in bytes of a directory."""
    path = Path(dir_path)
    if not path.exists():
        return 0
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            fp = Path(root) / f
            if fp.is_file():
                try:
                    total += fp.stat().st_size
                except OSError:
                    pass
    return total


def format_size(size_bytes: int) -> str:
    """Format bytes into human-readable string (KB, MB, GB)."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024**2:
        return f"{size_bytes / 1024:.2f} KB"
    elif size_bytes < 1024**3:
        return f"{size_bytes / (1024**2):.2f} MB"
    else:
        return f"{size_bytes / (1024**3):.2f} GB"


def ensure_base_directories() -> None:
    """Ensure standard dataset directories exist."""
    ORIGINAL_TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    ORIGINAL_TEST_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    for strategy in ["iid", "dirichlet", "label_skew", "quantity_skew", "custom"]:
        (PARTITIONS_ROOT / strategy).mkdir(parents=True, exist_ok=True)


class StorageManager:
    """Manages files, partitions, atomic directories, and dataset registry."""

    def __init__(self, base_dir: Path | str = DATASETS_ROOT):
        self.base_dir = Path(base_dir)
        self.original_dir = self.base_dir / "original" / "cifar10"
        self.partitions_dir = self.base_dir / "partitions" / "cifar10"
        self.metadata_dir = self.base_dir / "metadata"
        self.registry_file = self.metadata_dir / "dataset_registry.json"
        self.ensure_dirs()

    def ensure_dirs(self) -> None:
        """Create necessary directories."""
        (self.original_dir / "train").mkdir(parents=True, exist_ok=True)
        (self.original_dir / "test").mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        for strat in ["iid", "dirichlet", "label_skew", "quantity_skew", "custom"]:
            (self.partitions_dir / strat).mkdir(parents=True, exist_ok=True)

    def load_registry(self) -> Dict[str, Any]:
        """Load dataset registry JSON or return initial defaults."""
        if not self.registry_file.exists():
            return {
                "dataset": "CIFAR-10",
                "status": "NOT_DOWNLOADED",
                "train_samples": 0,
                "test_samples": 0,
                "download_timestamp": None,
                "last_verified_timestamp": None,
                "integrity_verified": False,
                "files": {},
            }
        try:
            with open(self.registry_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading registry file: {e}")
            return {"dataset": "CIFAR-10", "status": "ERROR", "error": str(e)}

    def save_registry(self, registry_data: Dict[str, Any]) -> None:
        """Save dataset registry JSON."""
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        with open(self.registry_file, "w", encoding="utf-8") as f:
            json.dump(registry_data, f, indent=4)
        logger.info(f"Updated registry at {self.registry_file}")

    def list_partitions(self) -> List[Dict[str, Any]]:
        """Scan and list all existing partition directories with manifest summaries."""
        partitions = []
        if not self.partitions_dir.exists():
            return partitions

        strategies = ["iid", "dirichlet", "label_skew", "quantity_skew", "custom"]
        for strat in strategies:
            strat_dir = self.partitions_dir / strat
            if not strat_dir.exists():
                continue
            for item in strat_dir.iterdir():
                if item.is_dir() and not item.name.startswith(("_tmp_", ".")):
                    manifest_file = item / "manifest.json"
                    manifest_data = None
                    is_valid = False
                    if manifest_file.exists():
                        try:
                            with open(manifest_file, "r", encoding="utf-8") as f:
                                manifest_data = json.load(f)
                            is_valid = manifest_data.get("validation_results", {}).get("is_valid", False)
                        except Exception:
                            pass

                    size_bytes = get_dir_size_bytes(item)
                    partitions.append({
                        "name": item.name,
                        "strategy": strat,
                        "path": str(item),
                        "manifest_exists": manifest_file.exists(),
                        "manifest": manifest_data,
                        "is_valid": is_valid,
                        "size_bytes": size_bytes,
                        "size_formatted": format_size(size_bytes),
                        "num_clients": manifest_data.get("num_clients") if manifest_data else None,
                        "random_seed": manifest_data.get("random_seed") if manifest_data else None,
                        "created_at": manifest_data.get("creation_timestamp") if manifest_data else None,
                    })
        return sorted(partitions, key=lambda x: x.get("created_at") or "", reverse=True)

    def delete_partition(self, partition_path: Path | str) -> bool:
        """Safely delete a partition directory. NEVER deletes original dataset."""
        path = Path(partition_path).resolve()
        partitions_root = self.partitions_dir.resolve()

        # Strict safety check: Must be inside partitions_dir and not the partitions root itself
        if partitions_root not in path.parents or path == partitions_root:
            raise PermissionError(f"Cannot delete outside partitions directory: {path}")

        # Ensure we never touch the original dataset directory
        original_root = self.original_dir.resolve()
        if original_root in path.parents or path == original_root:
            raise PermissionError("Deleting original CIFAR-10 dataset is strictly prohibited!")

        if path.exists() and path.is_dir():
            shutil.rmtree(path)
            logger.info(f"Deleted partition at: {path}")
            return True
        return False

    def create_atomic_temp_dir(self, strategy: str, target_dir_name: str) -> Path:
        """Create a safe temporary directory under the strategy folder for atomic writes."""
        strat_dir = self.partitions_dir / strategy
        strat_dir.mkdir(parents=True, exist_ok=True)
        temp_dir_name = f"_tmp_{target_dir_name}_{int(time.time() * 1000)}"
        temp_path = strat_dir / temp_dir_name
        temp_path.mkdir(parents=True, exist_ok=False)
        return temp_path

    def finalize_atomic_dir(self, temp_dir: Path, target_dir_name: str, overwrite: bool = False) -> Path:
        """Atomically rename temporary directory to the target directory."""
        target_path = temp_dir.parent / target_dir_name
        if target_path.exists():
            if overwrite:
                shutil.rmtree(target_path)
            else:
                raise FileExistsError(f"Target partition directory already exists: {target_path}")

        temp_dir.rename(target_path)
        logger.info(f"Finalized partition directory at: {target_path}")
        return target_path

"""Tests for storage manager, directory safety, and partition deletion."""

from pathlib import Path
import pytest
import torch

from dataset_manager.storage_manager import StorageManager, compute_file_sha256, format_size


def test_storage_manager_initialization(tmp_path):
    """Test directory creation and registry initialization."""
    sm = StorageManager(base_dir=tmp_path / "datasets")
    assert (tmp_path / "datasets" / "original" / "cifar10" / "train").exists()
    assert (tmp_path / "datasets" / "original" / "cifar10" / "test").exists()
    assert (tmp_path / "datasets" / "partitions" / "cifar10").exists()

    reg = sm.load_registry()
    assert reg["dataset"] == "CIFAR-10"
    assert reg["status"] == "NOT_DOWNLOADED"


def test_compute_file_sha256(tmp_path):
    """Test SHA-256 calculation for file."""
    fpath = tmp_path / "test.bin"
    content = b"hello cifar10 federated learning"
    fpath.write_bytes(content)

    checksum = compute_file_sha256(fpath)
    assert len(checksum) == 64
    assert isinstance(checksum, str)


def test_format_size():
    """Test human readable size formatting."""
    assert format_size(500) == "500 B"
    assert format_size(2048) == "2.00 KB"
    assert format_size(5 * 1024 * 1024) == "5.00 MB"


def test_safe_partition_deletion(tmp_path):
    """Test deleting partition works, but attempting to delete original dataset is blocked."""
    sm = StorageManager(base_dir=tmp_path / "datasets")

    # Create dummy partition
    strat_dir = sm.partitions_dir / "iid" / "dummy_run"
    strat_dir.mkdir(parents=True, exist_ok=True)
    (strat_dir / "data.pt").write_text("data")

    assert strat_dir.exists()
    res = sm.delete_partition(strat_dir)
    assert res is True
    assert not strat_dir.exists()

    # Attempt to delete original dataset directory -> must raise PermissionError
    with pytest.raises(PermissionError, match="Cannot delete outside partitions directory"):
        sm.delete_partition(sm.original_dir)

    # Attempt to delete root partitions dir -> must raise PermissionError
    with pytest.raises(PermissionError, match="Cannot delete outside"):
        sm.delete_partition(sm.partitions_dir)

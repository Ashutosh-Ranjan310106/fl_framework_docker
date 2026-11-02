"""Unit tests for IID dataset partitioning strategy."""

import numpy as np
import pytest

from dataset_manager.partition_strategies import partition_iid


@pytest.mark.parametrize("num_clients", [2, 5, 10, 20, 50])
def test_iid_partition_allocation(num_clients):
    """Test IID partitioning produces exact allocation with no duplicates or omissions."""
    num_samples = 1000
    seed = 42

    partitions = partition_iid(num_samples=num_samples, num_clients=num_clients, seed=seed)

    assert len(partitions) == num_clients

    # Check client IDs
    for i in range(num_clients):
        expected_cid = f"client_{i+1:03d}"
        assert expected_cid in partitions

    all_indices = []
    expected_per_client = num_samples // num_clients
    for cid, idxs in partitions.items():
        assert isinstance(idxs, np.ndarray)
        assert len(idxs) in (expected_per_client, expected_per_client + 1)
        all_indices.extend(idxs.tolist())

    # Check total count
    assert len(all_indices) == num_samples

    # Check no duplicates
    assert len(set(all_indices)) == num_samples

    # Check all indices in [0, num_samples - 1]
    assert set(all_indices) == set(range(num_samples))


def test_iid_invalid_client_count():
    """Test IID partitioning raises ValueError for invalid client counts."""
    with pytest.raises(ValueError, match="at least 1"):
        partition_iid(num_samples=100, num_clients=0)

    with pytest.raises(ValueError, match="cannot be less than"):
        partition_iid(num_samples=5, num_clients=10)

"""Unit tests for Quantity Skew partitioning strategy."""

import numpy as np
import pytest

from dataset_manager.partition_strategies import partition_quantity_skew


def test_quantity_skew_allocation():
    """Test Quantity Skew partitions respect min/max constraints and allocate all samples."""
    num_samples = 1000
    num_clients = 10
    min_samples = 30
    max_samples = 250
    seed = 42

    partitions = partition_quantity_skew(
        num_samples=num_samples,
        num_clients=num_clients,
        min_samples_per_client=min_samples,
        max_samples_per_client=max_samples,
        seed=seed,
    )

    assert len(partitions) == num_clients

    all_indices = []
    sizes = []
    for cid, idxs in partitions.items():
        count = len(idxs)
        sizes.append(count)
        assert min_samples <= count <= max_samples, (
            f"Client {cid} count {count} outside [{min_samples}, {max_samples}]"
        )
        all_indices.extend(idxs.tolist())

    assert sum(sizes) == num_samples
    assert len(set(all_indices)) == num_samples
    assert set(all_indices) == set(range(num_samples))
    # Check that there is actual variance in quantities
    assert min(sizes) < max(sizes)


def test_quantity_skew_impossible_bounds():
    """Test that quantity bounds that cannot total the dataset size raise ValueError."""
    num_samples = 1000
    num_clients = 10

    # Min bound too high (10 * 120 = 1200 > 1000)
    with pytest.raises(ValueError, match="Impossible quantity skew"):
        partition_quantity_skew(
            num_samples=num_samples,
            num_clients=num_clients,
            min_samples_per_client=120,
            max_samples_per_client=200,
        )

    # Max bound too low (10 * 80 = 800 < 1000)
    with pytest.raises(ValueError, match="Impossible quantity skew"):
        partition_quantity_skew(
            num_samples=num_samples,
            num_clients=num_clients,
            min_samples_per_client=20,
            max_samples_per_client=80,
        )

"""Unit tests for Dirichlet-based partition distribution."""

import numpy as np
import pytest

from dataset_manager.partition_strategies import partition_dirichlet
from dataset_manager.statistics import calculate_entropy


@pytest.mark.parametrize("alpha", [0.1, 0.3, 0.5, 1.0, 10.0])
def test_dirichlet_allocation_all_alphas(synthetic_data, alpha):
    """Test Dirichlet partitioning produces valid, complete allocations across various alphas."""
    labels = synthetic_data["labels"]
    num_samples = synthetic_data["num_samples"]
    num_clients = 10
    seed = 42

    partitions = partition_dirichlet(
        labels=labels,
        num_clients=num_clients,
        alpha=alpha,
        seed=seed,
    )

    assert len(partitions) == num_clients

    all_indices = []
    for cid, idxs in partitions.items():
        assert len(idxs) > 0, f"Client {cid} received 0 samples with alpha={alpha}"
        all_indices.extend(idxs.tolist())

    assert len(all_indices) == num_samples
    assert len(set(all_indices)) == num_samples
    assert set(all_indices) == set(range(num_samples))


def test_dirichlet_skew_relationship(synthetic_data):
    """Verify that lower alpha yields lower entropy (higher skew) on average than higher alpha."""
    labels = synthetic_data["labels"].cpu().numpy()
    num_clients = 5
    seed = 42

    p_low = partition_dirichlet(labels=labels, num_clients=num_clients, alpha=0.1, seed=seed)
    p_high = partition_dirichlet(labels=labels, num_clients=num_clients, alpha=10.0, seed=seed)

    entropies_low = []
    for idxs in p_low.values():
        c_counts = np.bincount(labels[idxs], minlength=10)
        entropies_low.append(calculate_entropy(c_counts))

    entropies_high = []
    for idxs in p_high.values():
        c_counts = np.bincount(labels[idxs], minlength=10)
        entropies_high.append(calculate_entropy(c_counts))

    # Mean entropy for high alpha (more balanced) should be higher than low alpha (skewed)
    assert np.mean(entropies_high) > np.mean(entropies_low)


def test_dirichlet_invalid_alpha(synthetic_data):
    """Test that alpha <= 0 raises ValueError."""
    labels = synthetic_data["labels"]
    with pytest.raises(ValueError, match="must be positive"):
        partition_dirichlet(labels=labels, num_clients=5, alpha=0.0)

    with pytest.raises(ValueError, match="must be positive"):
        partition_dirichlet(labels=labels, num_clients=5, alpha=-0.5)

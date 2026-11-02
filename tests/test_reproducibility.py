"""Tests for strict reproducibility of dataset partitioning."""

import numpy as np
import pytest

from dataset_manager.partition_strategies import (
    partition_custom,
    partition_dirichlet,
    partition_iid,
    partition_label_skew,
    partition_quantity_skew,
)


def test_iid_reproducibility():
    """Verify IID produces bitwise identical partitions with same seed and different with different seeds."""
    num_samples = 1000
    num_clients = 10

    run1 = partition_iid(num_samples=num_samples, num_clients=num_clients, seed=42)
    run2 = partition_iid(num_samples=num_samples, num_clients=num_clients, seed=42)
    run3 = partition_iid(num_samples=num_samples, num_clients=num_clients, seed=999)

    for cid in run1:
        np.testing.assert_array_equal(run1[cid], run2[cid])

    # Different seed must differ
    diff_count = sum(not np.array_equal(run1[cid], run3[cid]) for cid in run1)
    assert diff_count > 0


def test_label_skew_reproducibility(synthetic_data):
    """Verify Label Skew produces identical partitions with same seed."""
    labels = synthetic_data["labels"]
    num_clients = 10
    classes_per_client = 2

    run1 = partition_label_skew(labels=labels, num_clients=num_clients, classes_per_client=classes_per_client, seed=42)
    run2 = partition_label_skew(labels=labels, num_clients=num_clients, classes_per_client=classes_per_client, seed=42)
    run3 = partition_label_skew(labels=labels, num_clients=num_clients, classes_per_client=classes_per_client, seed=123)

    for cid in run1:
        np.testing.assert_array_equal(run1[cid], run2[cid])

    diff_count = sum(not np.array_equal(run1[cid], run3[cid]) for cid in run1)
    assert diff_count > 0


def test_dirichlet_reproducibility(synthetic_data):
    """Verify Dirichlet produces identical partitions with same seed."""
    labels = synthetic_data["labels"]
    num_clients = 10
    alpha = 0.5

    run1 = partition_dirichlet(labels=labels, num_clients=num_clients, alpha=alpha, seed=42)
    run2 = partition_dirichlet(labels=labels, num_clients=num_clients, alpha=alpha, seed=42)
    run3 = partition_dirichlet(labels=labels, num_clients=num_clients, alpha=alpha, seed=777)

    for cid in run1:
        np.testing.assert_array_equal(run1[cid], run2[cid])

    diff_count = sum(not np.array_equal(run1[cid], run3[cid]) for cid in run1)
    assert diff_count > 0


def test_quantity_skew_reproducibility():
    """Verify Quantity Skew produces identical partitions with same seed."""
    num_samples = 1000
    num_clients = 10
    min_s = 50
    max_s = 200

    run1 = partition_quantity_skew(num_samples=num_samples, num_clients=num_clients, min_samples_per_client=min_s, max_samples_per_client=max_s, seed=42)
    run2 = partition_quantity_skew(num_samples=num_samples, num_clients=num_clients, min_samples_per_client=min_s, max_samples_per_client=max_s, seed=42)
    run3 = partition_quantity_skew(num_samples=num_samples, num_clients=num_clients, min_samples_per_client=min_s, max_samples_per_client=max_s, seed=888)

    for cid in run1:
        np.testing.assert_array_equal(run1[cid], run2[cid])

    diff_count = sum(not np.array_equal(run1[cid], run3[cid]) for cid in run1)
    assert diff_count > 0


def test_isolated_random_state():
    """Verify that external global random state mutations do not affect partitioned output."""
    num_samples = 500
    num_clients = 5

    # Run baseline
    baseline = partition_iid(num_samples=num_samples, num_clients=num_clients, seed=42)

    # Mess with global numpy and python random state
    np.random.seed(12345)
    _ = np.random.rand(100)

    # Run again with same seed=42
    after_mutation = partition_iid(num_samples=num_samples, num_clients=num_clients, seed=42)

    for cid in baseline:
        np.testing.assert_array_equal(baseline[cid], after_mutation[cid])

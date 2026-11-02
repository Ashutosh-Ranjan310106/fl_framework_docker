"""Unit tests for Non-IID Label Skew and Custom partitioning strategies."""

import numpy as np
import pytest
import torch

from dataset_manager.config import NUM_CLASSES
from dataset_manager.partition_strategies import partition_custom, partition_label_skew


def test_label_skew_allocation(synthetic_data):
    """Test Label Skew partitioning allocates samples restricted to specified classes per client."""
    labels = synthetic_data["labels"]
    num_samples = synthetic_data["num_samples"]
    num_clients = 10
    classes_per_client = 2
    seed = 42

    partitions = partition_label_skew(
        labels=labels,
        num_clients=num_clients,
        classes_per_client=classes_per_client,
        seed=seed,
    )

    assert len(partitions) == num_clients

    all_indices = []
    labels_np = labels.cpu().numpy()

    for cid, idxs in partitions.items():
        assert len(idxs) > 0
        all_indices.extend(idxs.tolist())

        # Verify client only contains at most `classes_per_client` distinct classes
        client_labels = labels_np[idxs]
        unique_client_classes = np.unique(client_labels)
        assert len(unique_client_classes) <= classes_per_client

    # Verify no duplicates, no missing samples, exact total count
    assert len(all_indices) == num_samples
    assert len(set(all_indices)) == num_samples
    assert set(all_indices) == set(range(num_samples))


def test_label_skew_invalid_config(synthetic_data):
    """Test validation errors for invalid label skew parameters."""
    labels = synthetic_data["labels"]

    # classes_per_client > num_classes
    with pytest.raises(ValueError, match="cannot exceed total classes"):
        partition_label_skew(labels=labels, num_clients=10, classes_per_client=15)

    # total slots < num_classes (e.g. 2 clients, 2 classes each = 4 slots < 10 classes)
    with pytest.raises(ValueError, match="Impossible to allocate all"):
        partition_label_skew(labels=labels, num_clients=2, classes_per_client=2)

    # classes_per_client < 1
    with pytest.raises(ValueError, match="at least 1"):
        partition_label_skew(labels=labels, num_clients=10, classes_per_client=0)


def test_custom_partition_valid(synthetic_data):
    """Test custom partition with valid specified client counts."""
    labels = synthetic_data["labels"]
    num_samples = synthetic_data["num_samples"]

    counts = {
        "client_001": 200,
        "client_002": 150,
        "client_003": 150,
    }
    assert sum(counts.values()) == num_samples

    partitions = partition_custom(labels=labels, client_counts=counts, seed=42)
    assert len(partitions) == 3
    assert len(partitions["client_001"]) == 200
    assert len(partitions["client_002"]) == 150
    assert len(partitions["client_003"]) == 150

    all_idxs = []
    for idxs in partitions.values():
        all_idxs.extend(idxs.tolist())
    assert len(set(all_idxs)) == num_samples


def test_custom_partition_mismatch(synthetic_data):
    """Test custom partition raises ValueError when counts don't sum to dataset size."""
    labels = synthetic_data["labels"]
    counts = {
        "client_001": 100,
        "client_002": 100,
    }
    with pytest.raises(ValueError, match="sample count mismatch"):
        partition_custom(labels=labels, client_counts=counts, seed=42)

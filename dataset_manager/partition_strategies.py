"""Dataset partitioning strategies for Federated Learning."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from dataset_manager.config import NUM_CLASSES, setup_logger

logger = setup_logger("partition_strategies")


def get_client_id(index: int) -> str:
    """Format 0-based client index into standardized client ID string: client_001, etc."""
    return f"client_{index + 1:03d}"


def partition_iid(
    num_samples: int,
    num_clients: int,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    IID Partitioning:
    Randomly and uniformly distribute training samples equally among clients.
    All samples are allocated exactly once without duplication or omission.
    """
    if num_clients < 1:
        raise ValueError("Number of clients must be at least 1.")
    if num_samples < num_clients:
        raise ValueError(f"Number of samples ({num_samples}) cannot be less than number of clients ({num_clients}).")

    rng = np.random.default_rng(seed)
    shuffled_indices = rng.permutation(num_samples)

    # Use array_split to handle cases where num_samples is not perfectly divisible
    client_chunks = np.array_split(shuffled_indices, num_clients)

    partitions: Dict[str, np.ndarray] = {}
    for idx, chunk in enumerate(client_chunks):
        cid = get_client_id(idx)
        partitions[cid] = np.array(chunk, dtype=np.int64)

    logger.info(f"Generated IID partitions for {num_clients} clients (seed={seed}).")
    return partitions


def partition_label_skew(
    labels: Union[np.ndarray, torch.Tensor, List[int]],
    num_clients: int,
    classes_per_client: int,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Non-IID Label Skew:
    Clients receive samples from a restricted subset of classes (classes_per_client).
    Preserves all training samples without duplicates or omissions.
    """
    if num_clients < 1:
        raise ValueError("Number of clients must be at least 1.")
    if classes_per_client < 1:
        raise ValueError("classes_per_client must be at least 1.")

    if isinstance(labels, torch.Tensor):
        labels_arr = labels.cpu().numpy()
    else:
        labels_arr = np.asarray(labels)

    num_samples = len(labels_arr)
    unique_classes = np.unique(labels_arr)
    num_classes = len(unique_classes)

    if classes_per_client > num_classes:
        raise ValueError(
            f"classes_per_client ({classes_per_client}) cannot exceed total classes in dataset ({num_classes})."
        )

    # To ensure every sample is allocated, each of the num_classes must be assigned to at least one client.
    total_slots = num_clients * classes_per_client
    if total_slots < num_classes:
        raise ValueError(
            f"Impossible to allocate all {num_classes} classes to {num_clients} clients with "
            f"{classes_per_client} classes per client (total slots {total_slots} < {num_classes}). "
            f"Please increase client count or classes per client."
        )

    rng = np.random.default_rng(seed)

    # Group sample indices by class
    class_indices: Dict[int, np.ndarray] = {}
    for c in unique_classes:
        c_int = int(c)
        c_idxs = np.where(labels_arr == c_int)[0]
        rng.shuffle(c_idxs)
        class_indices[c_int] = c_idxs

    # Assign classes to each client
    # First ensure every class appears at least once
    client_classes: List[set] = [set() for _ in range(num_clients)]

    # Distribute one slot for each unique class in shuffled client order
    permuted_clients = rng.permutation(num_clients)
    for i, c in enumerate(unique_classes):
        target_client = permuted_clients[i % num_clients]
        client_classes[target_client].add(int(c))

    # Fill remaining slots for clients that have fewer than classes_per_client
    all_classes_list = [int(c) for c in unique_classes]
    for client_idx in range(num_clients):
        while len(client_classes[client_idx]) < classes_per_client:
            available = [c for c in all_classes_list if c not in client_classes[client_idx]]
            chosen = rng.choice(available)
            client_classes[client_idx].add(chosen)

    # Now for each class, determine which clients have it, and split the class samples among them
    client_samples: Dict[str, List[int]] = {get_client_id(i): [] for i in range(num_clients)}

    for c in unique_classes:
        c_int = int(c)
        assigned_clients = [i for i in range(num_clients) if c_int in client_classes[i]]
        if not assigned_clients:
            # Defensive check, though ensured above
            assigned_clients = [rng.integers(0, num_clients)]

        c_idxs = class_indices[c_int]
        splits = np.array_split(c_idxs, len(assigned_clients))
        for client_idx, split_idxs in zip(assigned_clients, splits):
            cid = get_client_id(client_idx)
            client_samples[cid].extend(split_idxs.tolist())

    partitions: Dict[str, np.ndarray] = {}
    for cid, idxs in client_samples.items():
        rng.shuffle(idxs)
        partitions[cid] = np.array(idxs, dtype=np.int64)

    logger.info(
        f"Generated Label Skew partitions: {num_clients} clients, "
        f"{classes_per_client} classes/client (seed={seed})."
    )
    return partitions


def partition_dirichlet(
    labels: Union[np.ndarray, torch.Tensor, List[int]],
    num_clients: int,
    alpha: float,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Non-IID Dirichlet Partitioning:
    Sample Dirichlet distributions over classes for each client using parameter alpha.
    Lower alpha produces higher label skew.
    Guarantees every sample is allocated exactly once.
    """
    if num_clients < 1:
        raise ValueError("Number of clients must be at least 1.")
    if alpha <= 0:
        raise ValueError(f"Dirichlet alpha must be positive, got {alpha}.")

    if isinstance(labels, torch.Tensor):
        labels_arr = labels.cpu().numpy()
    else:
        labels_arr = np.asarray(labels)

    num_samples = len(labels_arr)
    unique_classes = np.unique(labels_arr)
    rng = np.random.default_rng(seed)

    client_samples: Dict[str, List[int]] = {get_client_id(i): [] for i in range(num_clients)}

    # For each class, draw Dirichlet proportions across all clients
    for c in unique_classes:
        c_int = int(c)
        c_idxs = np.where(labels_arr == c_int)[0]
        rng.shuffle(c_idxs)
        n_c = len(c_idxs)

        # Proportions across clients
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))

        # Use Hamilton / Largest Remainder method for exact integer sum matching n_c
        raw_counts = proportions * n_c
        int_counts = np.floor(raw_counts).astype(int)
        remainder = n_c - int_counts.sum()

        if remainder > 0:
            fractional_parts = raw_counts - int_counts
            # Give remaining samples to clients with largest fractional remainder
            top_indices = np.argsort(-fractional_parts)[:remainder]
            int_counts[top_indices] += 1

        # Slice indices and assign
        start = 0
        for client_idx in range(num_clients):
            cnt = int_counts[client_idx]
            if cnt > 0:
                cid = get_client_id(client_idx)
                client_samples[cid].extend(c_idxs[start : start + cnt].tolist())
                start += cnt

    # Ensure no client is completely empty
    for i in range(num_clients):
        cid = get_client_id(i)
        if len(client_samples[cid]) == 0:
            # Find client with largest number of samples and transfer 1 sample
            donor_cid = max(client_samples.keys(), key=lambda k: len(client_samples[k]))
            if len(client_samples[donor_cid]) > 1:
                moved = client_samples[donor_cid].pop()
                client_samples[cid].append(moved)

    partitions: Dict[str, np.ndarray] = {}
    for cid, idxs in client_samples.items():
        rng.shuffle(idxs)
        partitions[cid] = np.array(idxs, dtype=np.int64)

    logger.info(f"Generated Dirichlet partitions: {num_clients} clients, alpha={alpha} (seed={seed}).")
    return partitions


def partition_quantity_skew(
    num_samples: int,
    num_clients: int,
    min_samples_per_client: int,
    max_samples_per_client: int,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Quantity Skew Partitioning:
    Different clients receive varying sample counts between min_samples and max_samples.
    All num_samples are preserved and allocated without omission or duplicates.
    """
    if num_clients < 1:
        raise ValueError("Number of clients must be at least 1.")
    if min_samples_per_client < 1:
        raise ValueError("min_samples_per_client must be at least 1.")
    if max_samples_per_client < min_samples_per_client:
        raise ValueError(
            f"max_samples_per_client ({max_samples_per_client}) cannot be less than "
            f"min_samples_per_client ({min_samples_per_client})."
        )

    min_possible = num_clients * min_samples_per_client
    max_possible = num_clients * max_samples_per_client

    if num_samples < min_possible or num_samples > max_possible:
        raise ValueError(
            f"Impossible quantity skew configuration: {num_clients} clients with "
            f"min={min_samples_per_client} and max={max_samples_per_client} cannot partition {num_samples} samples. "
            f"Total samples must be between {min_possible} and {max_possible}."
        )

    rng = np.random.default_rng(seed)
    shuffled_indices = rng.permutation(num_samples)

    if min_samples_per_client == max_samples_per_client:
        sizes = [min_samples_per_client] * num_clients
    else:
        # Allocate min_samples to each client first
        sizes = np.full(num_clients, min_samples_per_client, dtype=int)
        excess_needed = num_samples - min_possible
        capacity = max_samples_per_client - min_samples_per_client

        # Sample random proportions for excess distribution
        alpha_vec = np.ones(num_clients)
        proportions = rng.dirichlet(alpha_vec)

        # Distribute excess proportionally with clipping to capacity
        raw_excess = proportions * excess_needed
        int_excess = np.floor(raw_excess).astype(int)
        int_excess = np.minimum(int_excess, capacity)

        current_total = int_excess.sum()
        remaining = excess_needed - current_total

        # Greedily assign remaining 1-by-1 to clients with capacity
        candidate_order = np.argsort(-(raw_excess - int_excess))
        for idx in candidate_order:
            if remaining <= 0:
                break
            space = capacity - int_excess[idx]
            add = min(space, remaining)
            int_excess[idx] += add
            remaining -= add

        # If any remaining due to capacity limits on top candidates, scan all
        if remaining > 0:
            for idx in range(num_clients):
                if remaining <= 0:
                    break
                space = capacity - int_excess[idx]
                if space > 0:
                    add = min(space, remaining)
                    int_excess[idx] += add
                    remaining -= add

        sizes += int_excess

    if sizes.sum() != num_samples:
        raise RuntimeError(f"Internal calculation error: partition sum {sizes.sum()} != {num_samples}")

    partitions: Dict[str, np.ndarray] = {}
    current_idx = 0
    for i, size in enumerate(sizes):
        cid = get_client_id(i)
        partitions[cid] = shuffled_indices[current_idx : current_idx + size].astype(np.int64)
        current_idx += size

    logger.info(
        f"Generated Quantity Skew partitions: {num_clients} clients, "
        f"min={min_samples_per_client}, max={max_samples_per_client} (seed={seed})."
    )
    return partitions


def partition_custom(
    labels: Union[np.ndarray, torch.Tensor, List[int]],
    client_counts: Dict[str, int],
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Custom Partitioning:
    Allocate exact sample counts per client specified by user dictionary: {'client_001': count, ...}.
    Validates that the sum matches total samples and no count is negative.
    """
    if isinstance(labels, torch.Tensor):
        labels_arr = labels.cpu().numpy()
    else:
        labels_arr = np.asarray(labels)

    num_samples = len(labels_arr)
    num_clients = len(client_counts)

    if num_clients < 1:
        raise ValueError("Custom configuration must specify at least 1 client.")

    for cid, cnt in client_counts.items():
        if cnt < 0:
            raise ValueError(f"Custom sample count for {cid} cannot be negative ({cnt}).")

    total_requested = sum(client_counts.values())
    if total_requested != num_samples:
        raise ValueError(
            f"Custom configuration sample count mismatch: sum of client samples ({total_requested}) "
            f"does not equal total dataset samples ({num_samples}). Discrepancy: {total_requested - num_samples}."
        )

    rng = np.random.default_rng(seed)
    shuffled_indices = rng.permutation(num_samples)

    partitions: Dict[str, np.ndarray] = {}
    current_idx = 0
    for cid, count in client_counts.items():
        partitions[cid] = shuffled_indices[current_idx : current_idx + count].astype(np.int64)
        current_idx += count

    logger.info(f"Generated Custom partitions for {num_clients} clients (seed={seed}).")
    return partitions

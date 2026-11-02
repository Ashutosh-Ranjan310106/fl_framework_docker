"""Partition statistics calculation, CSV export, and visualization utilities."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from dataset_manager.config import CIFAR10_CLASSES, NUM_CLASSES, setup_logger

logger = setup_logger("statistics")


def calculate_entropy(counts: Union[List[int], np.ndarray]) -> float:
    """Calculate Shannon entropy for a distribution of class counts."""
    counts_arr = np.asarray(counts, dtype=float)
    total = counts_arr.sum()
    if total == 0:
        return 0.0
    probs = counts_arr / total
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def compute_partition_statistics(
    client_partitions: Dict[str, np.ndarray],
    labels: Union[np.ndarray, torch.Tensor],
    num_classes: int = NUM_CLASSES,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Compute detailed sample counts, class distributions, and skew metrics per client.
    Returns:
        - DataFrame with per-client rows.
        - Summary dict with overall partition metrics.
    """
    if isinstance(labels, torch.Tensor):
        labels_arr = labels.cpu().numpy()
    else:
        labels_arr = np.asarray(labels)

    rows = []
    client_sizes = []
    all_class_counts = np.zeros(num_classes, dtype=int)

    for cid in sorted(client_partitions.keys()):
        sample_indices = client_partitions[cid]
        sample_count = len(sample_indices)
        client_sizes.append(sample_count)

        if sample_count > 0:
            client_labels = labels_arr[sample_indices]
            # Count each class
            class_counts = np.bincount(client_labels, minlength=num_classes)
        else:
            class_counts = np.zeros(num_classes, dtype=int)

        all_class_counts += class_counts
        present_classes = int(np.count_nonzero(class_counts))
        dominant_class = int(np.argmax(class_counts)) if sample_count > 0 else -1
        entropy = calculate_entropy(class_counts)

        row = {
            "client_id": cid,
            "sample_count": sample_count,
            "num_classes": present_classes,
            "dominant_class": dominant_class,
            "dominant_class_name": CIFAR10_CLASSES[dominant_class] if dominant_class >= 0 else "None",
            "entropy": round(entropy, 4),
        }
        for c in range(num_classes):
            row[f"class_{c}"] = int(class_counts[c])
            c_name = CIFAR10_CLASSES[c] if c < len(CIFAR10_CLASSES) else f"class_{c}"
            row[f"class_{c_name}"] = int(class_counts[c])

        rows.append(row)

    df = pd.DataFrame(rows)

    sizes_arr = np.array(client_sizes)
    summary = {
        "num_clients": len(client_partitions),
        "total_samples": int(sizes_arr.sum()),
        "min_samples": int(sizes_arr.min()) if len(sizes_arr) > 0 else 0,
        "max_samples": int(sizes_arr.max()) if len(sizes_arr) > 0 else 0,
        "mean_samples": float(round(sizes_arr.mean(), 2)) if len(sizes_arr) > 0 else 0.0,
        "std_samples": float(round(sizes_arr.std(), 2)) if len(sizes_arr) > 0 else 0.0,
        "class_totals": {
            CIFAR10_CLASSES[c] if c < len(CIFAR10_CLASSES) else f"class_{c}": int(all_class_counts[c])
            for c in range(num_classes)
        },
    }

    return df, summary


def save_statistics_csv(df: pd.DataFrame, csv_path: Union[str, Path]) -> None:
    """Save partition statistics DataFrame to a clean CSV file."""
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Standard columns to export
    base_cols = ["client_id", "sample_count", "num_classes", "dominant_class", "dominant_class_name", "entropy"]
    class_cols = [f"class_{c}" for c in range(NUM_CLASSES) if f"class_{c}" in df.columns]

    export_df = df[base_cols + class_cols]
    export_df.to_csv(path, index=False)
    logger.info(f"Saved partition statistics to: {path}")


def plot_samples_per_client(df: pd.DataFrame, title: str = "Samples per Client") -> plt.Figure:
    """Create a bar chart showing sample counts per client."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    clients = df["client_id"]
    counts = df["sample_count"]

    bars = ax.bar(clients, counts, color="#3498db", edgecolor="#2980b9", alpha=0.85)
    ax.set_xlabel("Client ID", fontsize=11, fontweight="bold")
    ax.set_ylabel("Number of Samples", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)

    # Rotate client labels if many
    if len(clients) > 15:
        ax.tick_params(axis="x", rotation=90)
    else:
        ax.tick_params(axis="x", rotation=45)

    mean_count = counts.mean()
    ax.axhline(mean_count, color="#e74c3c", linestyle="--", linewidth=1.5, label=f"Mean: {mean_count:.1f}")
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()
    return fig


def plot_class_distribution_heatmap(
    df: pd.DataFrame,
    title: str = "Class Distribution Heatmap (Clients vs Classes)",
) -> plt.Figure:
    """Create a heatmap matrix of sample counts per class across clients."""
    class_cols = [f"class_{c}" for c in range(NUM_CLASSES)]
    matrix = df[class_cols].values

    fig, ax = plt.subplots(figsize=(11, max(4.5, len(df) * 0.35)))
    cax = ax.imshow(matrix, cmap="YlGnBu", aspect="auto")

    ax.set_xticks(range(NUM_CLASSES))
    ax.set_xticklabels([f"{c}: {CIFAR10_CLASSES[c]}" for c in range(NUM_CLASSES)], rotation=45, ha="right")
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["client_id"])

    ax.set_xlabel("CIFAR-10 Class", fontsize=11, fontweight="bold")
    ax.set_ylabel("Client ID", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)

    cbar = fig.colorbar(cax, ax=ax)
    cbar.set_label("Sample Count", fontsize=10)
    plt.tight_layout()
    return fig


def plot_stacked_class_distribution(
    df: pd.DataFrame,
    title: str = "Class Composition per Client",
) -> plt.Figure:
    """Create a stacked bar chart displaying class composition per client."""
    fig, ax = plt.subplots(figsize=(12, 5))
    client_ids = df["client_id"]
    bottom = np.zeros(len(df))

    # Standard distinguishable colormap
    colors = plt.cm.tab10(np.linspace(0, 1, NUM_CLASSES))

    for c in range(NUM_CLASSES):
        col_name = f"class_{c}"
        if col_name in df.columns:
            counts = df[col_name].values
            label = f"{c}: {CIFAR10_CLASSES[c]}"
            ax.bar(client_ids, counts, bottom=bottom, color=colors[c], label=label, width=0.7)
            bottom += counts

    ax.set_xlabel("Client ID", fontsize=11, fontweight="bold")
    ax.set_ylabel("Total Samples", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)

    if len(client_ids) > 15:
        ax.tick_params(axis="x", rotation=90)
    else:
        ax.tick_params(axis="x", rotation=45)

    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", title="CIFAR-10 Classes")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    plt.tight_layout()
    return fig


def plot_overall_class_balance(df: pd.DataFrame, title: str = "Total Allocated Samples per Class") -> plt.Figure:
    """Create a bar chart checking the total allocated samples for each of the 10 classes."""
    class_cols = [f"class_{c}" for c in range(NUM_CLASSES)]
    totals = df[class_cols].sum()

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x_labels = [f"{c}\n{CIFAR10_CLASSES[c]}" for c in range(NUM_CLASSES)]
    bars = ax.bar(x_labels, totals.values, color="#2ecc71", edgecolor="#27ae60", alpha=0.85)

    ax.set_xlabel("CIFAR-10 Class", fontsize=11, fontweight="bold")
    ax.set_ylabel("Allocated Samples", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)

    # Expected count line (5,000 for standard CIFAR-10)
    expected_per_class = totals.mean()
    ax.axhline(expected_per_class, color="#e74c3c", linestyle="--", linewidth=1.5, label=f"Expected: {expected_per_class:.0f}")
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()
    return fig

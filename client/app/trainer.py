"""PyTorch training and evaluation loops for federated client."""

from __future__ import annotations

import logging
import os
import time
from typing import Dict, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from client.app.metrics import calculate_accuracy, get_system_usage

logger = logging.getLogger("client.trainer")

# Limit CPU thread contention across multiple client processes
torch.set_num_threads(int(os.getenv("TORCH_THREADS", "2")))


def train_local(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int = 2,
    lr: float = 0.001,
    optimizer_name: str = "adam",
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Train model locally on client dataset.
    Returns dictionary with training metrics.
    """
    dev = torch.device(device)
    model.to(dev)
    model.train()

    criterion = nn.CrossEntropyLoss()
    if optimizer_name.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    start_time = time.time()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_samples = 0

        for images, labels in train_loader:
            images, labels = images.to(dev), labels.to(dev)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            epoch_loss += loss.item() * batch_size
            _, predicted = torch.max(outputs.data, 1)
            epoch_correct += (predicted == labels).sum().item()
            epoch_samples += batch_size

        total_loss += epoch_loss
        total_correct += epoch_correct
        total_samples += epoch_samples

    duration = time.time() - start_time
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    cpu_usage, ram_mb = get_system_usage()

    return {
        "loss": float(avg_loss),
        "accuracy": float(accuracy),
        "samples": int(len(train_loader.dataset)),
        "duration_seconds": float(duration),
        "epochs": int(epochs),
        "cpu_percent": float(cpu_usage),
        "ram_mb": float(ram_mb),
    }


def evaluate_local(
    model: nn.Module,
    eval_loader: DataLoader,
    device: str = "cpu",
) -> Tuple[float, int, Dict[str, float]]:
    """Evaluate model locally on a dataloader."""
    dev = torch.device(device)
    model.to(dev)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in eval_loader:
            images, labels = images.to(dev), labels.to(dev)
            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += batch_size

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0

    return float(avg_loss), int(total_samples), {"accuracy": float(accuracy)}

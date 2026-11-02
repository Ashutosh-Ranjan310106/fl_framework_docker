"""Client metric helpers and resource usage tracking."""

from __future__ import annotations

import os
from typing import Dict, Tuple
import psutil
import torch


def get_system_usage() -> Tuple[float, float]:
    """Return (cpu_percent, ram_used_mb) for current client process."""
    process = psutil.Process(os.getpid())
    cpu_percent = psutil.cpu_percent(interval=None)
    mem_info = process.memory_info()
    ram_mb = mem_info.rss / (1024 * 1024)
    return float(cpu_percent), float(ram_mb)


def calculate_accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute classification accuracy between model outputs and targets."""
    _, predicted = torch.max(outputs.data, 1)
    total = targets.size(0)
    correct = (predicted == targets).sum().item()
    return float(correct / total) if total > 0 else 0.0

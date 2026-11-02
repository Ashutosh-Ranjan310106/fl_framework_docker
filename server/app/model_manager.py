"""Model management, checkpointing, and global evaluation for Flower Server."""

from __future__ import annotations

import logging
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from client.app.model import CIFAR10CNN, get_model_parameters, set_model_parameters

logger = logging.getLogger("server.model_manager")


class ModelManager:
    """Coordinates global model initialization, checkpointing, and evaluation."""

    def __init__(
        self,
        checkpoint_dir: str | Path = "models/checkpoints",
        test_data_path: Optional[str | Path] = "datasets/original/cifar10/test/data.pt",
        device: str = "cpu",
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device)
        self.model = CIFAR10CNN(num_classes=10).to(self.device)
        self.test_loader: Optional[DataLoader] = None

        if test_data_path and Path(test_data_path).exists():
            self._load_test_data(test_data_path)
        else:
            logger.warning(f"Test data not found at {test_data_path}. Global evaluation may be skipped.")

    def _load_test_data(self, test_path: str | Path) -> None:
        """Load test set for server-side evaluation."""
        data = torch.load(test_path, weights_only=False)
        images = data["images"].float() / 255.0
        labels = data["labels"]
        dataset = TensorDataset(images, labels)
        self.test_loader = DataLoader(dataset, batch_size=64, shuffle=False)
        logger.info(f"Loaded {len(dataset)} test samples for global evaluation.")

    def get_initial_parameters(self) -> List[np.ndarray]:
        """Return initial model parameters as list of numpy ndarrays."""
        return get_model_parameters(self.model)

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Set model parameters from numpy ndarrays."""
        set_model_parameters(self.model, parameters)

    def save_checkpoint(
        self,
        round_num: int,
        metrics: Optional[Dict[str, float]] = None,
        experiment_id: str = "cifar10_baseline_001",
    ) -> Path:
        """Save PyTorch checkpoint to disk."""
        checkpoint_name = f"{experiment_id}_round_{round_num:03d}.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        torch.save(
            {
                "round": round_num,
                "experiment_id": experiment_id,
                "model_state_dict": self.model.state_dict(),
                "metrics": metrics or {},
                "timestamp": time.time(),
            },
            checkpoint_path,
        )
        logger.info(f"Saved model checkpoint for round {round_num} to {checkpoint_path}")
        return checkpoint_path

    def evaluate_global(self, parameters: Optional[List[np.ndarray]] = None) -> Tuple[float, float, int]:
        """
        Evaluate current or provided parameters on the server-side test set.
        Returns: (loss, accuracy, num_samples)
        """
        if parameters is not None:
            self.set_parameters(parameters)

        if self.test_loader is None:
            logger.warning("No test loader configured for global evaluation.")
            return 0.0, 0.0, 0

        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = criterion(outputs, labels)

                batch_size = labels.size(0)
                total_loss += loss.item() * batch_size
                _, predicted = torch.max(outputs.data, 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += batch_size

        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        return float(avg_loss), float(accuracy), total_samples

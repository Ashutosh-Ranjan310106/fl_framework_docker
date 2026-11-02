"""PyTorch CNN model for CIFAR-10 Federated Learning."""

from __future__ import annotations

from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CIFAR10CNN(nn.Module):
    """
    Standard lightweight convolutional neural network for CIFAR-10.
    Input shape: (B, 3, 32, 32)
    Output shape: (B, 10)
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        # Block 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)  # -> 16x16

        # Block 2
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)  # -> 8x8

        # Fully connected
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.dropout = nn.Dropout(0.25)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(F.relu(self.bn2(self.conv2(x))))
        x = self.pool2(F.relu(self.bn3(self.conv3(x))))
        x = x.flatten(1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


def get_model_parameters(model: nn.Module) -> List[np.ndarray]:
    """Extract model parameters as a list of NumPy ndarrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_model_parameters(model: nn.Module, parameters: List[np.ndarray]) -> None:
    """Load model parameters from a list of NumPy ndarrays into model state dict."""
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)

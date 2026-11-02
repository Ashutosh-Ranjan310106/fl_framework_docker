"""Unit tests for CIFAR-10 PyTorch CNN model."""

import numpy as np
import pytest
import torch
from client.app.model import CIFAR10CNN, get_model_parameters, set_model_parameters


def test_model_forward():
    model = CIFAR10CNN(num_classes=10)
    batch_size = 4
    x = torch.randn(batch_size, 3, 32, 32)
    out = model(x)
    assert out.shape == (batch_size, 10), f"Expected shape (4, 10), got {out.shape}"


def test_model_parameters_extract_and_set():
    model1 = CIFAR10CNN(num_classes=10)
    model2 = CIFAR10CNN(num_classes=10)

    # Initially different weights
    params1 = get_model_parameters(model1)
    params2 = get_model_parameters(model2)

    diff = sum(np.sum(np.abs(p1 - p2)) for p1, p2 in zip(params1, params2))
    assert diff > 0.0, "Fresh models should have different initial random weights"

    # Set model2 weights to model1 weights
    set_model_parameters(model2, params1)
    params2_updated = get_model_parameters(model2)

    for p1, p2_up in zip(params1, params2_updated):
        np.testing.assert_allclose(p1, p2_up, atol=1e-6)

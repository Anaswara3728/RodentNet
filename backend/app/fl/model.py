from __future__ import annotations

from collections import OrderedDict
from typing import Iterable, Sequence

import numpy as np
import torch
from torch import nn


SHARED_PREFIXES = ("features.", "projection.")


class FedPerCifarNet(nn.Module):
    """Compact CIFAR10 network with shared representation and local personal head."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
        )
        self.personal_head = nn.Linear(64, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs)
        embedding = self.projection(features)
        return self.personal_head(embedding)


def shared_state_keys(model: nn.Module) -> list[str]:
    return [key for key in model.state_dict().keys() if key.startswith(SHARED_PREFIXES)]


def get_shared_parameters(model: nn.Module) -> list[np.ndarray]:
    state = model.state_dict()
    return [state[key].detach().cpu().numpy().copy() for key in shared_state_keys(model)]


def set_shared_parameters(model: nn.Module, parameters: Sequence[np.ndarray]) -> None:
    state = model.state_dict()
    keys = shared_state_keys(model)
    if len(keys) != len(parameters):
        raise ValueError(f"expected {len(keys)} shared tensors, received {len(parameters)}")

    updates = OrderedDict()
    for key, parameter in zip(keys, parameters):
        updates[key] = torch.tensor(parameter, dtype=state[key].dtype)
    state.update(updates)
    model.load_state_dict(state, strict=True)


def train_one_epoch(
    model: nn.Module,
    data_loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    *,
    device: torch.device,
    learning_rate: float,
) -> float:
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    batches = 0
    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        running_loss += float(loss.detach().cpu())
        batches += 1
    return running_loss / max(batches, 1)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    data_loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    *,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    loss_total = 0.0
    correct = 0
    examples = 0
    batches = 0
    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss_total += float(criterion(logits, labels).detach().cpu())
        correct += int((logits.argmax(dim=1) == labels).sum().detach().cpu())
        examples += int(labels.numel())
        batches += 1
    return loss_total / max(batches, 1), correct / max(examples, 1)

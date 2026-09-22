from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def load_cifar10_partition(
    *,
    data_dir: Path,
    client_index: int,
    num_clients: int,
    batch_size: int = 64,
    fake_data: bool = False,
) -> tuple[DataLoader, DataLoader]:
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ]
    )
    transform_eval = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ]
    )

    if fake_data:
        train_dataset = datasets.FakeData(
            size=2_000,
            image_size=(3, 32, 32),
            num_classes=10,
            transform=transform_train,
        )
        test_dataset = datasets.FakeData(
            size=600,
            image_size=(3, 32, 32),
            num_classes=10,
            transform=transform_eval,
        )
    else:
        train_dataset = datasets.CIFAR10(
            root=str(data_dir),
            train=True,
            download=True,
            transform=transform_train,
        )
        test_dataset = datasets.CIFAR10(
            root=str(data_dir),
            train=False,
            download=True,
            transform=transform_eval,
        )

    train_indices = list(range(client_index, len(train_dataset), num_clients))
    test_indices = list(range(client_index, len(test_dataset), num_clients))

    train_loader = DataLoader(
        Subset(train_dataset, train_indices),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        Subset(test_dataset, test_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader

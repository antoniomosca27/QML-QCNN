"""
loader.py
---------
Funzioni per scaricare/normalizzare medMNIST
e restituire DataLoader PyTorch.

Dipende da:
    pip install medmnist>=2.2.3
"""

from __future__ import annotations
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from medmnist import INFO
from medmnist.dataset import MedMNIST

# Parametri di default
_DEFAULT_DATASET = "pathmnist"  # uno dei 12 dataset di medMNIST
_IMAGE_SIZE = 28
_MEAN, _STD = 0.5, 0.5  # normalizzazione [-1,1]


def _get_transform() -> transforms.Compose:
    """Trasformazioni di base: tensor + normalizzazione."""
    return transforms.Compose(
        [
            transforms.ToTensor(),  # (0,255) → (0,1)
            transforms.Normalize(mean=[_MEAN] * 3, std=[_STD] * 3),
        ]
    )


def download_medmnist(name: str = _DEFAULT_DATASET, root: str | Path = "data/raw"):
    """
    Scarica (se necessario) il dataset scelto.

    Ritorna
    -------
    train_ds, test_ds : torch.utils.data.Dataset
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    info = INFO[name]
    DataClass = getattr(__import__("medmnist"), info["python_class"])

    train_ds = DataClass(
        split="train",
        transform=_get_transform(),
        download=True,
        root=str(root),
    )
    test_ds = DataClass(
        split="test",
        transform=_get_transform(),
        download=True,
        root=str(root),
    )
    return train_ds, test_ds


def split_train_val(train_ds, val_frac: float = 0.15, seed: int = 42):
    """Splitta il training set in train/val mantenendo la stratificazione."""
    val_len = int(len(train_ds) * val_frac)
    train_len = len(train_ds) - val_len
    torch.manual_seed(seed)
    train_subset, val_subset = random_split(train_ds, [train_len, val_len])
    return train_subset, val_subset


def get_dataloaders(
    batch_size: int = 128,
    dataset_name: str = _DEFAULT_DATASET,
    num_workers: int = 2,
):
    """Restituisce DataLoader per train/val/test."""
    train_ds, test_ds = download_medmnist(dataset_name)
    train_ds, val_ds = split_train_val(train_ds)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, val_loader, test_loader

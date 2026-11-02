"""Client-side PyTorch dataset loader for assigned CIFAR-10 partition."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional, Tuple, Union
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms


class ClientPartitionDataset(Dataset):
    """Loads a client's dedicated partition from data.pt."""

    def __init__(
        self,
        partition_path: Union[str, Path],
        transform: Optional[Callable] = None,
    ):
        self.path = Path(partition_path)
        if self.path.is_dir():
            target_pt = self.path / "data.pt"
        else:
            target_pt = self.path

        if not target_pt.exists():
            raise FileNotFoundError(f"Partition data file not found at: {target_pt}")

        data = torch.load(target_pt, weights_only=False)
        self.images = data["images"]  # shape (N, 3, 32, 32), uint8
        self.labels = data["labels"]  # shape (N,), int64
        self.sample_indices = data.get("sample_indices")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img = self.images[idx]
        target = int(self.labels[idx].item())

        if self.transform is not None:
            pil_img = transforms.functional.to_pil_image(img)
            img = self.transform(pil_img)
        else:
            # Default float tensor normalized to [0, 1]
            img = img.float() / 255.0

        return img, target


def get_client_dataloader(
    partition_path: Union[str, Path],
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Create a PyTorch DataLoader for the client's partition."""
    dataset = ClientPartitionDataset(partition_path=partition_path)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
    )

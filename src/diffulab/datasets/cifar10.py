import pickle
from pathlib import Path
from typing import List

import numpy as np
from numpy.typing import NDArray

from .base import DiffusionDataset


class CIFAR10Dataset(DiffusionDataset):
    """CIFAR10 dataset for diffusion models."""

    def __init__(
        self,
        data_path: str,
        batches_to_load: List[str] = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"],
    ):
        """Initialize the CIFAR10 dataset.

        Args:
            data_path: Path to the CIFAR10 data directory
            batches_to_load: List of batch files to load
        """
        super().__init__()
        self.data_path = Path(data_path)
        self.batches_to_load = batches_to_load
        self.images, self.labels = self.load_data()

    def load_data(self) -> tuple[NDArray[np.uint8], NDArray[np.int64]]:
        """Load CIFAR10 data from files."""
        images: list[NDArray[np.uint8]] = []
        labels: list[NDArray[np.int64]] = []

        for batch in self.batches_to_load:
            images_batch, labels_batch = self._load_cifar10_batch(self.data_path / batch)
            images.append(images_batch)
            labels.append(labels_batch)

        return np.concatenate(images, axis=0), np.concatenate(labels, axis=0)

    def _load_cifar10_batch(self, file: Path) -> tuple[NDArray[np.uint8], NDArray[np.int64]]:
        """Load a single CIFAR10 batch file."""
        with open(file, "rb") as f:
            batch = pickle.load(f, encoding="latin1")

        features = batch["data"]
        r = features[:, :1024].reshape(-1, 32, 32)
        g = features[:, 1024:2048].reshape(-1, 32, 32)
        b = features[:, 2048:].reshape(-1, 32, 32)

        # Create RGB images with shape (N, H, W, C)
        images = np.stack([r, g, b], axis=-1, dtype=np.uint8)
        labels = np.array(batch["labels"], dtype=np.int64)

        return images, labels

    def preprocess_image(self, image: NDArray) -> NDArray:
        """Preprocess CIFAR10 image: normalize and transpose to (C, H, W)."""
        # Convert to float and normalize to [-1, 1]
        normalized = (image.astype(np.float32) / 255.0 - 0.5) / 0.5

        # Transpose from (H, W, C) to (C, H, W)
        transposed = normalized.transpose(2, 0, 1)

        return transposed

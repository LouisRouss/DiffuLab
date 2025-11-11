import random
from pathlib import Path

import torch
import torchvision.transforms as transforms  # type: ignore[reportMissingTypeStub]
from streaming import StreamingDataset
from torch.utils.data import Dataset

from diffulab.datasets.base import BatchData


class ImageNetLatent(Dataset[BatchData]):
    """ImageNet dataset for diffusion models with latent and DINOv2 features."""

    def __init__(
        self,
        data_path: str,
        local: bool = True,
        batch_size: int = 64,
        split: str = "train",
    ) -> None:
        """Initialize the MNIST dataset.

        Args:
            data_path: Path to the dataset leading to MDS shard files
            local: Whether to use local files or remote files
            batch_size: Batch size for optimized streaming dataset and future data loading
            latent_scale: Scale factor for latent features
            split: Dataset split to use (train, val, test)
        """
        super().__init__()
        self.data_path = Path(data_path)
        self.dataset = StreamingDataset(
            remote=self.data_path.as_posix() if not local else None,
            local=self.data_path.as_posix() if local else None,
            batch_size=batch_size,
            split=split,
        )
        self.latent_scale: float | None = None
        self.transform = transforms.ToTensor()

    def set_latent_scale(self, scale: float) -> None:
        """Set the latent scale for the dataset."""
        self.latent_scale = scale

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> BatchData:
        """Get a sample from the dataset.
        Args:
            idx: Index of the sample to retrieve
        Returns:
            BatchData: A dictionary containing model inputs and extra features
        """
        assert self.latent_scale is not None, "Latent scale must be set before getting items"
        sample = self.dataset[idx]
        assert "latents" in sample, "Batch must contain 'latent' key, please precompute the latents before training"
        latent = torch.tensor(sample["latents"], dtype=torch.float32)
        batch_data: BatchData = {"model_inputs": {"x": latent * self.latent_scale}, "extra": {}}

        if "label" in sample:
            y = torch.tensor(sample["label"], dtype=torch.long)
            batch_data["model_inputs"]["y"] = y

        if "image" in sample:
            x0 = self.transform(sample["image"])
            batch_data["extra"]["x0"] = x0

        if "dst_features" in sample:
            dst_features = torch.tensor(sample["dst_features"], dtype=torch.float32)
            batch_data["extra"]["dst_features"] = dst_features

        return batch_data


class ImageNetNoisyLatent(ImageNetLatent):
    """ImageNet dataset for diffusion models with latent and DINOv2 features and added noise."""

    def __init__(
        self,
        data_path: str,
        local: bool = True,
        batch_size: int = 64,
        split: str = "train",
        noise_tau: float = 0.8,
    ) -> None:
        """Initialize the MNIST dataset with noise.

        Args:
            data_path: Path to the dataset leading to MDS shard files
            local: Whether to use local files or remote files
            batch_size: Batch size for optimized streaming dataset and future data loading
            latent_scale: Scale factor for latent features
            split: Dataset split to use (train, val, test)
            noise_std: Standard deviation of the Gaussian noise to add to latents
        """
        super().__init__(data_path, local, batch_size, split)
        self.noise_tau = noise_tau

    def __getitem__(self, idx: int) -> BatchData:
        """Get a sample from the dataset with added noise.
        Args:
            idx: Index of the sample to retrieve
        Returns:
            BatchData: A dictionary containing model inputs and extra features
        """
        batch_data = super().__getitem__(idx)
        latent = batch_data["model_inputs"]["x"]
        if self.noise_tau > 0:
            sigma = abs(random.gauss(0, self.noise_tau))
            noise = torch.randn_like(latent) * sigma
            latent = latent + noise
            batch_data["model_inputs"]["x"] = latent
        return batch_data

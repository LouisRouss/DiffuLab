from pathlib import Path

import torch
from streaming import StreamingDataset
from torch.utils.data import Dataset

from diffulab.datasets.base import BatchData


class ImageNetLatentREPA(Dataset[BatchData]):
    """ImageNet dataset for diffusion models with latent and DINOv2 features."""

    def __init__(
        self,
        data_path: str,
        local: bool = True,
        batch_size: int = 64,
        latent_scale: float = 0.18215,
        split: str = "train",
    ) -> None:
        """Initialize the MNIST dataset.

        Args:
            data_path: Path to the dataset leading to MDS shard files
            local: Whether to use local files or remote files
            batch_size: Batch size for optimized streaming dataset and future data loading
            latent_scale: Scale factor for latent features
        """
        super().__init__()
        self.data_path = Path(data_path)
        self.dataset = StreamingDataset(
            remote=self.data_path.as_posix() if not local else None,
            local=self.data_path.as_posix() if local else None,
            batch_size=batch_size,
            split=split,
        )
        self.latent_scale = latent_scale

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

        sample = self.dataset[idx]
        assert "latent" in sample, "Batch must contain 'latent' key, please precompute the latents before training"
        assert "label" in sample, "Batch must contain 'y' key, please add labels to the dataset"
        assert "dst_features" in sample, (
            "Batch must contain 'dst_features' key, please precompute the DINOv2 features before training"
        )
        latent = torch.tensor(sample["latent"], dtype=torch.float32)
        y = torch.tensor(sample["label"], dtype=torch.long)
        dst_features = torch.tensor(sample["dst_features"], dtype=torch.float32)

        batch_data: BatchData = {
            "model_inputs": {"x": latent * self.latent_scale, "y": y},
            "extra": {
                "dst_features": dst_features,
            },
        }

        return batch_data

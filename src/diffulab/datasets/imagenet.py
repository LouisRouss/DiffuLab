import math
import random
from pathlib import Path
from typing import Any, Generator, cast

import torch
import torchvision.transforms as transforms  # type: ignore[reportMissingTypeStub]
from PIL import Image
from streaming import StreamingDataset
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm

from diffulab.datasets.base import BatchData


class ImageNetLatentREPA(Dataset[BatchData]):
    """ImageNet dataset for diffusion models with latent and DINOv2 features."""

    def __init__(
        self,
        data_path: str,
        local: bool = True,
        batch_size: int = 64,
        split: str = "train",
    ) -> None:
        """Initialize the ImageNet dataset.

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
        assert "latent" in sample, "Batch must contain 'latent' key, please precompute the latents before training"
        assert "label" in sample, "Batch must contain 'y' key, please add labels to the dataset"

        latent = torch.tensor(sample["vision_latents"], dtype=torch.float32)
        y = torch.tensor(sample["label"], dtype=torch.long)

        batch_data: BatchData = {
            "model_inputs": {"x": latent * self.latent_scale, "y": y},
            "extra": {},
        }

        x0: torch.Tensor | None = None
        dst_features: torch.Tensor | None = None
        if "dst_features" not in sample:
            assert "image" in sample, "Batch must contain either 'dst_features' or 'image' key"
            x0 = self.transform(sample["image"])
            batch_data["extra"]["x0"] = x0
        else:
            dst_features = torch.tensor(sample["dst_features"], dtype=torch.float32)
            batch_data["extra"]["dst_features"] = dst_features

        return batch_data


class ImageNetmultiAR(Dataset[BatchData]):
    def __init__(
        self,
        data_path: str,
        local: bool = True,
        batch_size: int = 64,
        split: str = "train",
    ) -> None:
        self.latent_scale: float | None = None
        self.latent_bias: float = 0.0

        self.data_path = Path(data_path)
        self.dataset = StreamingDataset(
            remote=self.data_path.as_posix() if not local else None,
            local=self.data_path.as_posix() if local else None,
            batch_size=batch_size,
            split=split,
        )

        self.buckets: dict[tuple[int, int], list[int]] = {}
        for b, sample in enumerate(tqdm(self.dataset, desc="constructing buckets")):  # type: ignore
            w, h = cast(Image.Image, sample["image"]).size
            if (h, w) not in self.buckets:
                self.buckets[(h, w)] = []
            self.buckets[(h, w)].append(b)

        self.transform = transforms.ToTensor()

    def __len__(self) -> int:
        return sum(len(v) for v in self.buckets.values())

    def set_latent_scale(self, scale: float) -> None:
        """Set the latent scale for the dataset."""
        self.latent_scale = scale

    def set_latent_bias(self, bias: float) -> None:
        """Set the latent bias for the dataset."""
        self.latent_bias = bias

    # ADD RESIZE TO THE BUCKET SIZE FOR THE GIVEN ITEM (EVERYTHING NEED TO BE DONE OFFLINE)
    def __getitem__(self, idx: int) -> BatchData:
        """Get a sample from the dataset.
        Args:
            idx: Index of the sample to retrieve
        Returns:
            BatchData: A dictionary containing model inputs and extra features
        """
        assert self.latent_scale is not None, "Latent scale must be set before getting items"
        sample = self.dataset[idx]
        assert "vision_latents" in sample, (
            "Batch must contain 'vision_latents' key, please precompute the latents before training"
        )
        assert "caption" in sample, "Batch must contain 'caption' key, please add caption to the dataset"

        latent = torch.tensor(sample["vision_latents"], dtype=torch.float32)
        caption: str = sample["caption"]

        batch_data: BatchData = {
            "model_inputs": {
                "x": ((latent - self.latent_bias) * self.latent_scale).squeeze(),
                "initial_context": caption,
            },
            "extra": {},
        }

        x0: torch.Tensor | None = None
        dst_features: torch.Tensor | None = None
        assert "extra" in batch_data  # for pyright
        if "dst_features" not in sample:
            assert "image" in sample, "Batch must contain either 'dst_features' or 'image' key"
            x0 = self.transform(sample["image"])
            batch_data["extra"]["x0"] = x0
        else:
            dst_features = torch.tensor(sample["dst_features"], dtype=torch.float32)
            batch_data["extra"]["dst_features"] = dst_features

        return batch_data


def collate_fn(batch: list[BatchData]) -> BatchData:
    model_inputs: dict[str, Any] = {}
    extra: dict[str, Any] = {}

    # Collate model_inputs
    keys = batch[0]["model_inputs"].keys()
    for key in keys:
        if key == "initial_context":
            model_inputs[key] = [sample["model_inputs"].get(key, "") for sample in batch]
        else:
            model_inputs[key] = torch.stack([sample["model_inputs"][key] for sample in batch], dim=0)

    # Collate extras (only stack keys present in a sample)
    extra_keys: set[str] = set().union(*(sample.get("extra", {}).keys() for sample in batch))  # type: ignore
    for key in extra_keys:
        to_stack = [sample.get("extra", {}).get(key) for sample in batch if key in sample.get("extra", {})]
        extra[key] = torch.stack(to_stack, dim=0)  # type: ignore
    return BatchData(**{"model_inputs": model_inputs, "extra": extra})  # type: ignore


class MultiARBatchSampler(Sampler[list[int]]):
    def __init__(
        self, dataset: ImageNetmultiAR, batch_size: int, shuffle: bool = True, drop_last: bool = False
    ) -> None:
        if not hasattr(dataset, "buckets"):  # aspect ratio buckets is a dict with bucket_id -> list of indices
            raise ValueError("Dataset must have 'buckets' attribute for MultiARBatchSampler")
        self.shuffle = shuffle
        self.buckets = dataset.buckets
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self) -> Generator[list[int], Any, None]:
        # build all bucket-specific batches, then shuffle their order
        all_batches: list[list[int]] = []

        for _, idxs in self.buckets.items():
            idxs = idxs.copy()
            if self.shuffle:
                random.shuffle(idxs)

            for i in range(0, len(idxs), self.batch_size):
                batch = idxs[i : i + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                all_batches.append(batch)

        if self.shuffle:
            random.shuffle(all_batches)

        for batch in all_batches:
            yield batch

    def __len__(self) -> int:
        total: int = 0
        for idxs in self.buckets.values():
            if self.drop_last:
                total += len(idxs) // self.batch_size
            else:
                total += math.ceil(len(idxs) / self.batch_size)
        return total

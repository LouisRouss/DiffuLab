from abc import ABC, abstractmethod
from typing import Any, cast

import torch
import torch.nn as nn
import torchvision.transforms as transforms  # type: ignore[reportMissingTypeStub]
from PIL import Image
from streaming import StreamingDataset
from streaming.base import MDSWriter
from torch import Tensor
from torch.utils.data import DataLoader


class REPA(nn.Module, ABC):
    def __init__(self) -> None:
        super().__init__()  # type: ignore

    @property
    @abstractmethod
    def encoder(self) -> nn.Module:
        """
        The encoder module that processes the input tensor.
        This should be implemented in subclasses to return the specific encoder architecture.
        """

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """
        The dimension of the encoded representation.
        This should be implemented in subclasses to return the specific embedding dimension.
        """

    @abstractmethod
    def preprocess(self, x: Tensor) -> Tensor:
        """
        Preprocess the input tensor before encoding.

        Args:
            x (Tensor): Input tensor to be preprocessed.
            Assumed to be an image tensor with shape [N, C, H, W].

        Returns:
            Tensor: Preprocessed input tensor.
        """

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the encoder. Preprocesses the input tensor and returns the encoded representation.

        Args:
            x (Tensor): Input tensor to be encoded.

        Returns:
            Tensor: Encoded representation of the input tensor.
        """

    def compute_on_dataset(
        self,
        dataset_path: str,
        dst_path: str | tuple[str, str],
        local: bool = True,
        target_type: str = "float32",
        batch_size: int = 64,
    ) -> None:
        dataset = StreamingDataset(
            remote=dataset_path if not local else None,
            local=dataset_path if local else None,
            batch_size=batch_size,
        )
        assert dataset.shards, "Dataset has no shards."  # type: ignore

        def _collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
            """
            Custom collate function to handle the batch of data.
            Creates a list for each key from the batch items.
            """
            return {key: [item[key] for item in batch] for key in batch[0].keys()}

        dataloader = cast(
            DataLoader[dict[str, Any]],
            DataLoader(
                dataset,
                batch_size=batch_size,
                collate_fn=_collate_fn,
            ),
        )
        columns: dict[str, str] = {
            name: encoding
            for name, encoding in zip(dataset.shards[0].column_names, dataset.shards[0].column_encodings)  # type: ignore
        }

        image_key = next((col for col in columns if col.startswith("image")), None)
        if image_key is None:
            raise ValueError("Dataset must contain a column starting with 'image'.")
        if sum(col.startswith("image") for col in columns) > 1:
            raise ValueError(f"Dataset contains multiple columns starting with 'image'. Only one is allowed.")

        columns["repa_embedding"] = target_type

        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        transform = transforms.ToTensor()
        with MDSWriter(out=dst_path, columns=columns) as writer:
            for batch in dataloader:
                images = batch[image_key]
                assert all(isinstance(img, Image.Image) for img in images), "All images must be PIL Image objects"

                # Convert PIL images to tensors
                images = torch.stack([transform(img) for img in images])
                # Move to correct device and dtype
                images = images.to(dtype=dtype, device=device)
                # Compute embeddings
                embeddings = self.forward(images)
                embeddings_numpy = embeddings.cpu().numpy()  # type: ignore[reportUnknownMemberType]

                # Write each sample individually
                for i in range(embeddings.shape[0]):
                    sample: dict[str, Any] = {}
                    for key in batch.keys():
                        sample[key] = batch[key][i]
                    sample["repa_embedding"] = embeddings_numpy[i]
                    writer.write(sample)

from abc import ABC, abstractmethod
from typing import Any, cast

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from PIL import Image
from streaming import StreamingDataset
from streaming.base import MDSWriter
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import transforms  # type: ignore[reportMissingTypeStub]
from tqdm import tqdm


class VisionTower(nn.Module, ABC):
    def __init__(self, latent_scale: float | Tensor = 1.0, latent_bias: float | Tensor = 0) -> None:
        """
        Base class for vision towers, which are used to encode and decode images into latent representations.

        Args:
            - latent_scale (float | Tensor): Scale factor for the latent representation. Default is 1.0.
            - latent_bias (float | Tensor): Bias for the latent representation. Default is 0.
        """
        super().__init__()  # type: ignore
        self.latent_scale = latent_scale
        self.latent_bias = latent_bias

    @property
    @abstractmethod
    def compression_factor(self) -> int:
        """
        Compression factor of the AE.
        This should be implemented in subclasses to return the specific compression factor.
        """
        ...

    @property
    @abstractmethod
    def latent_channels(self) -> int:
        """
        Number of channels in the latent space.
        This should be implemented in subclasses to return the specific number of latent channels.
        """
        ...

    @abstractmethod
    def encode(self, x: Tensor) -> Tensor:
        """
        Encoding part of the VAE that encodes the input tensor into a latent representation.

        Args:
            x (Tensor): Input tensor to be encoded.

        Returns:
            Tensor: Encoded representation of the input tensor.
        """
        ...

    @abstractmethod
    def decode(self, z: Tensor) -> Tensor:
        """
        Decoding part of the VAE that decodes the latent representation back to the original space.
        Args:
            z (Tensor): Latent representation to be decoded.
        Returns:
            Tensor: Decoded representation of the latent tensor.
        """
        ...

    @abstractmethod
    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:  # type: ignore
        """
        Forward pass of the encoder.

        Args:
            x (Tensor): Input tensor to be encoded.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            forward (Tensor): Output tensor after encoding and decoding.
        """

    @torch.inference_mode()
    def compute_on_dataset(
        self,
        dataset_path: str,
        dst_path: str | tuple[str, str],
        local: bool = True,
        batch_size: int = 64,
        split: str | None = None,
        to_process_data_key: str | None = None,
        target_type: str = "float32",  # "float32" | "float16"
        column_target: str = "vision_latents",
    ) -> None:
        """
        Run the embedder over a MosaicML StreamingDataset and write an MDS with
        the original columns + one ndarray column per output of the embedder.

        Each output i is written to a column named f"{column_prefix}_{i}" with
        encoding "ndarray:<target_type>".

        Args:
            dataset_path: Path to existing StreamingDataset (local dir or remote URI).
            dst_path: Destination for MDSWriter (dir or (remote, local) tuple).
            local: If True, treat dataset_path as local; otherwise as remote.
            batch_size: Loader batch size (controls tokenization/forward work size).
            split: Optional split name.
            to_process_data_key: Explicit column name containing data to process. If None,
            target_type: "float32" or "float16" for output arrays.
            column_prefix: Prefix for destination embedding columns.
            progress_desc: TQDM description.
        """
        dataset = StreamingDataset(
            remote=dataset_path if not local else None,
            local=dataset_path if local else None,
            batch_size=batch_size,
            split=split,
        )
        assert dataset.shards, "Dataset has no shards."  # type: ignore

        def _collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
            return {key: [item[key] for item in batch] for key in batch[0].keys()}

        dataloader = cast(
            DataLoader[dict[str, Any]],
            DataLoader(dataset, batch_size=batch_size, collate_fn=_collate_fn),
        )

        source_columns: dict[str, str] = {
            name: encoding
            for name, encoding in zip(dataset.shards[0].column_names, dataset.shards[0].column_encodings)  # type: ignore
        }

        dst_columns = dict(source_columns)
        dst_columns[column_target] = f"ndarray:{target_type}"

        # device / dtype (best effort; works even if submodules are off-device)
        try:
            param = next(self.parameters())
            device = param.device
            dtype = param.dtype
        except StopIteration:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            logger.warning(f"device and dtype not found | defaulting to {device} and {dtype}.")

        transform = transforms.ToTensor()
        with MDSWriter(out=dst_path, columns=dst_columns) as writer:
            for batch in tqdm(dataloader, desc="computing embeddings", unit="batch"):
                to_process_data = batch[to_process_data_key]
                assert all(isinstance(img, Image.Image) for img in to_process_data), (
                    "All images must be PIL Image objects"
                )

                # Convert PIL images to tensors
                images = torch.stack([transform(img) for img in to_process_data])
                # Move to correct device and dtype
                images = images.to(dtype=dtype, device=device)

                # Forward -> tuple of Tensors
                output = self.encode(images)
                np_output = output.detach().to("cpu").float().numpy()  # type: ignore
                if target_type == "float32":
                    np_output = np_output.astype(np.float32)
                elif target_type == "float16":
                    np_output = np_output.astype(np.float16)
                else:
                    raise ValueError("target_type must be 'float32' or 'float16'")

                B = len(to_process_data)
                # Write per-sample rows, preserving original columns
                for i in range(B):
                    sample = {k: v[i] for k, v in batch.items()}
                    sample[column_target] = np_output[i]
                    writer.write(sample)

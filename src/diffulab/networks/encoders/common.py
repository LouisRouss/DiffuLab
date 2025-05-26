from abc import ABC, abstractmethod
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch import Tensor


class Encoder(nn.Module, ABC):
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
        Forward pass of the encoder.

        Args:
            x (Tensor): Input tensor to be encoded.

        Returns:
            Tensor: Encoded representation of the input tensor.
        """

    def compute_on_dataset(self, list_images: list[str], save_path: str) -> None:
        """
        Compute the encoded representations of a dataset using the encoder.

        Args:
            list_images (list[str]): List of image file paths to be encoded.
            save_path (str): Path to save the encoded representations.
        """
        self.eval()
        device = next(self.parameters()).device
        for image_path in list_images:
            image = Image.open(image_path).convert("RGB")
            image = (
                torch.tensor(image).permute(2, 0, 1).unsqueeze(0).to(device)
            )  # Convert to tensor and add batch dimension
            image = self.forward(image).squeeze(0)  # Remove batch dimension
            torch.save(image, Path(save_path) / f"{Path(image_path).stem}.pt")  # type: ignore[reportUnknownArgumentType]

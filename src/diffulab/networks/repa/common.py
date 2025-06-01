from abc import ABC, abstractmethod
from pathlib import Path

import torch.nn as nn
from torch import Tensor


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
        Forward pass of the encoder.

        Args:
            x (Tensor): Input tensor to be encoded.

        Returns:
            Tensor: Encoded representation of the input tensor.
        """

    def compute_on_dataset(self, dataset_path: Path, local: bool = True, batch_size: int = 64) -> None: ...

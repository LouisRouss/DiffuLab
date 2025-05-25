from abc import ABC, abstractmethod

import torch
import torch.nn as nn
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
        ...

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
        ...

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the encoder.

        Args:
            x (Tensor): Input tensor to be encoded.

        Returns:
            Tensor: Encoded representation of the input tensor.
        """
        x = self.preprocess(x)
        with torch.no_grad():
            x = self.encoder(x)
        return x

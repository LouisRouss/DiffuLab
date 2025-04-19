from abc import ABC, abstractmethod

import torch.nn as nn
from torch import Tensor


class Encoder(nn.Module, ABC):
    def __init__(self):
        super().__init__()  # type: ignore

    @abstractmethod
    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:  # type: ignore
        """
        Forward pass of the encoder.

        Args:
            x (Tensor): Input tensor to be encoded.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: Encoded representation of the input tensor.
        """

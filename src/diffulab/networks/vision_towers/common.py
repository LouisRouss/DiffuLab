from abc import ABC, abstractmethod

import torch.nn as nn
from torch import Tensor


class VisionTower(nn.Module, ABC):
    def __init__(self, latent_scale: float = 1.0) -> None:
        """
        Base class for vision towers, which are used to encode and decode images into latent representations.
        
        Args:
            - latent_scale (float): Scale factor for the latent representation. Default is 1.0.
        """
        super().__init__()  # type: ignore
        self.latent_scale = latent_scale

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

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch import Tensor


class VAE(nn.Module, ABC):
    def __init__(self):
        super().__init__()  # type: ignore

    def sample(self, mean: Tensor, std: Tensor, latent_scale: float = 1.0, latent_bias: float = 0.0) -> Tensor:
        """
        Sample from the latent space given the mean and variance.

        Args:
            mean (Tensor): Mean of the latent representation.
            std (Tensor): Standard_deviation of the latent representation.
            latent_scale (float, optional): Scale factor for the latent space. Defaults to 1.0.
            latent_bias (float, optional): Bias added to the latent space. Defaults to 0.0.

        Returns:
            Tensor: Sampled tensor from the latent space.
        """
        z = mean + std * torch.randn_like(mean)
        z = z * latent_scale + latent_bias
        return z

    @property
    @abstractmethod
    def compression_factor(self) -> int:
        """
        Compression factor of the VAE.
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
            Tensor: Encoded representation of the input tensor.
        """

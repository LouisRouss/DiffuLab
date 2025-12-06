from typing import cast

import torch
from diffusers.models.autoencoders.autoencoder_kl_flux2 import AutoencoderKLFlux2
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from jaxtyping import Float
from torch import Tensor

from diffulab.networks.vision_towers.common import VisionTower


class Flux2VAE(VisionTower):
    def __init__(self, model_name: str = "black-forest-labs/FLUX.2-dev") -> None:
        model: AutoencoderKLFlux2 = AutoencoderKLFlux2.from_pretrained(model_name, subfolder="vae")  # type: ignore[reportUnknownMemberType]
        model.eval()
        model.requires_grad_(False)
        super().__init__(
            latent_scale=1 / torch.sqrt(model.bn.running_var.view(1, -1, 1, 1) + model.config.batch_norm_eps),  # type: ignore
            latent_bias=model.bn.running_mean.view(1, -1, 1, 1),  # type: ignore
        )
        self.model = model
        self._compression_factor = 2 ** (len(self.model.encoder.down_blocks) - 1)
        self._latent_channels: int = self.model.config.latent_channels  # type: ignore

    @property
    def compression_factor(self) -> int:
        """
        Compression factor of the AE.
        This should be implemented in subclasses to return the specific compression factor.
        """
        return self._compression_factor

    @property
    def latent_channels(self) -> int:
        """
        Number of channels in the latent space.
        This should be implemented in subclasses to return the specific number of latent channels.
        """
        return self._latent_channels

    def encode(self, x: Float[Tensor, "batchsize 3 H W"]) -> Float[Tensor, "batchsize latent_channels H' W'"]:
        """
        Encoding part of the VAE that encodes the input tensor into a latent representation.

        Args:
            x (Tensor): Input tensor to be encoded.

        Returns:
            Tensor: Encoded representation of the input tensor.
        """
        x = x.float()
        x_min = x.min().item()
        x_max = x.max().item()

        if x_min >= 0.0 and x_max <= 1.0:
            pass
        elif x_min >= 0.0 and x_max <= 255.0 and x_max > 1.0:
            x = x / 255.0
        else:
            raise ValueError("Input tensor range is not supported. Expected 0–255 or 0–1")

        x = (x - 0.5) * 2.0
        posterior = cast(DiagonalGaussianDistribution, self.model.encode(x).latent_dist)  # type: ignore

        return posterior.sample()

    def decode(self, z: Float[Tensor, "batchsize latent_channels H' W'"]) -> Float[Tensor, "batchsize 3 H W"]:
        """
        Decoding part of the VAE that decodes the latent representation back to the original input space.

        Args:
            latent (Tensor): Latent representation to be decoded.

        Returns:
            Tensor: Decoded tensor in the original input space. Normalized to [-1, 1].
        """
        return self.model.decode(z).sample  # type: ignore

    def forward(self, x: Float[Tensor, "batchsize 3 H W"]) -> Float[Tensor, "batchsize latent_channels H W"]:
        """
        Forward pass through the DCAE, encoding and decoding the input tensor.

        Args:
            x (Tensor): Input tensor to be processed.

        Returns:
            Tensor: Output tensor after encoding and decoding.
        """
        z = self.encode(x)
        return self.decode(z)

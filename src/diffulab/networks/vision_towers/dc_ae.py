from diffusers.models.autoencoders.autoencoder_dc import AutoencoderDC
from jaxtyping import Float
from torch import Tensor

from diffulab.networks.vision_towers.common import VisionTower


class DCAE(VisionTower):
    def __init__(self, model_name: str = "mit-han-lab/dc-ae-f64c128-in-1.0-diffusers") -> None:
        super().__init__()
        self.model: AutoencoderDC = AutoencoderDC.from_pretrained(model_name)  # type: ignore[reportUnknownMemberType]
        self.model.eval()
        self.model.requires_grad_(False)

        self._compression_factor = self.model.spatial_compression_ratio
        self._latent_channels: int = self.model._internal_dict["latent_channels"]  # type: ignore

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
        if x.min() >= 0 and x.max() <= 1:  # Case: 0-1
            x = (x - 0.5) * 2.0
        elif x.min() >= 0 and x.max() <= 255:  # Case: 0-255
            x = x.float() / 255.0
            x = (x - 0.5) * 2.0
        return self.model.encode(x).latent  # type: ignore

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

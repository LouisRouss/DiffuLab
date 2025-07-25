from typing import cast

import timm
import torch
import torch.nn as nn
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch import Tensor
from torchvision.transforms import Normalize  # type: ignore[reportMissingTypeStub]

from diffulab.networks.repa.common import REPA


class DinoV2(REPA):
    native_resolution: int = 224
    base_patch_pixel_size: int = 14

    def __init__(
        self,
        dino_model: str = f"dinov2_vitl14_reg",
        resolution: int = 256,
        target_seq_len: int | None = None,
    ) -> None:
        super().__init__()

        self._encoder: nn.Module = torch.hub.load("facebookresearch/dinov2", dino_model)  # type: ignore
        del self._encoder.head
        self._encoder.eval()

        if not target_seq_len:
            self.inference_resolution: int = self.native_resolution * (resolution // 256)
            grid_size = self.inference_resolution // self.base_patch_pixel_size
        else:
            sqrt_val = target_seq_len**0.5
            if not sqrt_val.is_integer():
                raise ValueError(f"target_seq_len ({target_seq_len}) must be a square")
            grid_size = int(sqrt_val)
            self.inference_resolution = grid_size * self.base_patch_pixel_size

        self._encoder.pos_embed.data = timm.layers.pos_embed.resample_abs_pos_embed(
            self._encoder.pos_embed.data,  # type: ignore
            [grid_size, grid_size],
        )

        self._encoder.head = torch.nn.Identity()
        self._embedding_dim: int = self._encoder.embed_dim  # type: ignore

        self._encoder.requires_grad_(False)

    @property
    def encoder(self) -> nn.Module:
        """
        The encoder module that processes the input tensor.
        Returns:
            nn.Module: The DINO V2 encoder without the head.
        """
        return self._encoder

    @property
    def embedding_dim(self) -> int:
        """
        The dimension of the encoded representation.
        Returns:
            int: The embedding dimension of the DINO V2 encoder.
        """
        return self._embedding_dim

    def preprocess(self, x: Tensor) -> Tensor:
        # Normalize the input tensor to be between 0 and 1
        if x.min() >= 0 and x.max() <= 255:  # Case: 0-255
            x = x.float() / 255.0
        elif (x.min() >= -1.0 and x.min() < 0) and x.max() <= 1.0:  # Case: -1 to 1
            x = (x + 1.0) / 2.0
        elif x.min() >= 0.0 and x.max() <= 1.0:  # Case: 0-1
            pass
        else:
            raise ValueError("Input tensor range is not supported. Expected 0-255, 0-1 or -1 to 1.")

        x = x * 0.5 + 0.5
        x = Normalize(
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
        )(x)
        x = cast(
            Tensor,
            torch.nn.functional.interpolate(x, self.inference_resolution, mode="bicubic"),  # type: ignore[reportUnknownMemberType]
        )
        return x

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
            z = cast(Tensor, self.encoder.forward_features(x)["x_norm_patchtokens"])  # type: ignore
        return z

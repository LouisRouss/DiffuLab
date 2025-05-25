from typing import cast

import timm
import torch
import torch.nn as nn
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import Normalize  # type: ignore[reportMissingTypeStub]

from diffulab.networks.encoders.common import Encoder


class DinoV2(Encoder):
    def __init__(self, dino_model: str = f"dinov2_vitl14_reg", resolution: int = 256) -> None:
        super().__init__()
        self.resolution = resolution

        self._encoder: nn.Module = torch.hub.load("facebookresearch/dinov2", dino_model)  # type: ignore
        del self._encoder.head
        self._encoder.eval()

        # Resample the positional embedding to match the resolution
        patch_resolution = 16 * (resolution // 256)
        self._encoder.pos_embed.data = timm.layers.pos_embed.resample_abs_pos_embed(
            self._encoder.pos_embed.data,  # type: ignore
            [patch_resolution, patch_resolution],
        )

        self._encoder.head = torch.nn.Identity()

    @property
    def encoder(self) -> nn.Module:
        """
        The encoder module that processes the input tensor.
        Returns:
            nn.Module: The DINO V2 encoder without the head.
        """
        return self._encoder

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize the input tensor to be between 0 and 1
        if x.dtype == torch.uint8:  # Case: 0-255
            x = x.float() / 255.0
        elif x.min() >= -1.0 and x.max() <= 1.0:  # Case: -1 to 1
            x = (x + 1.0) / 2.0
        else:
            raise ValueError("Input tensor range is not supported. Expected 0-255 or -1 to 1.")

        x = x * 0.5 + 0.5
        x = Normalize(
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
        )(x)
        x = cast(torch.Tensor, torch.nn.functional.interpolate(x, 224 * (self.resolution // 256), mode="bicubic"))  # type: ignore[reportUnknownMemberType]
        return x

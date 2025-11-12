from typing import Any, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.spectral_norm import SpectralNorm
from transformers import ViTImageProcessor, ViTModel


class SpectralConv1d(nn.Conv1d):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)  # type: ignore
        SpectralNorm.apply(self, name="weight", n_power_iterations=1, dim=0, eps=1e-12)


class BatchNormLocal(nn.Module):
    def __init__(self, num_features: int, affine: bool = True, virtual_bs: int = 1, eps: float = 1e-6):
        super().__init__()  # type: ignore
        self.virtual_bs = virtual_bs
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.size()
        x = x.float()

        # Reshape batch into groups.
        G = np.ceil(x.size(0) / self.virtual_bs).astype(int)
        x = x.view(G, -1, x.size(-2), x.size(-1))
        # Calculate stats.
        mean = x.mean([1, 3], keepdim=True)
        var = x.var([1, 3], keepdim=True, unbiased=False)
        x = (x - mean) / (torch.sqrt(var + self.eps))

        if self.affine:
            x = x * self.weight[None, :, None] + self.bias[None, :, None]

        return x.view(shape)


class ResidualBlock(nn.Module):
    def __init__(self, fn: nn.Module):
        super().__init__()  # type: ignore
        self.fn = fn
        self.ratio = 1 / np.sqrt(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = x.float()
        return (self.fn(x).add(x)).mul_(self.ratio)


class RAEDiscriminator(nn.Module):
    def __init__(self, model_name: str = "facebook/dino-vits8", features_depth: list[int] = [2, 5, 8, 11]):
        super().__init__()  # type: ignore
        image_processor: ViTImageProcessor = ViTImageProcessor.from_pretrained(model_name, do_rescale=False)  # type: ignore
        self.register_buffer("image_mean", torch.tensor(image_processor.image_mean).view(1, -1, 1, 1))
        self.register_buffer("image_std", torch.tensor(image_processor.image_std).view(1, -1, 1, 1))
        self.image_size: tuple[int, int] = (image_processor.size["height"], image_processor.size["width"])

        self.dino_model: ViTModel = ViTModel.from_pretrained(model_name, add_pooling_layer=False)  # type: ignore
        self.features_depth = features_depth

        self.dino_model.eval()
        for param in self.dino_model.parameters():
            param.requires_grad = False

        channels = self.dino_model.config.hidden_size
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    SpectralConv1d(channels, channels, kernel_size=9, padding=9 // 2, padding_mode="circular"),
                    BatchNormLocal(channels, eps=1e-6),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    ResidualBlock(
                        nn.Sequential(
                            SpectralConv1d(channels, channels, kernel_size=9, padding=9 // 2, padding_mode="circular"),
                            BatchNormLocal(channels, eps=1e-6),
                            nn.LeakyReLU(negative_slope=0.2, inplace=True),
                        )
                    ),
                    SpectralConv1d(channels, 1, kernel_size=1, padding=0),
                )
                for _ in range(len(features_depth))
            ]
        )
        # train heads
        for p in self.heads:
            p.requires_grad_(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward pass through the discriminator
        Args:
            x (torch.Tensor): input images (B, C, H, W) in range [0, 1]
        Returns:
            torch.Tensor: discriminator outputs (B, sum of N_i)
        """
        inputs = x.float()
        inputs = cast(
            torch.Tensor,
            F.interpolate(inputs, size=self.image_size, mode="bilinear", align_corners=False, antialias=True),  # type: ignore
        )
        inputs = (inputs - self.image_mean.to(dtype=inputs.dtype, device=x.device)) / self.image_std.to(  # type: ignore
            dtype=inputs.dtype, device=inputs.device
        )
        inputs = inputs.to(dtype=x.dtype)  # type: ignore
        features: list[torch.Tensor] = list(
            self.dino_model(pixel_values=inputs, output_hidden_states=True).hidden_states
        )  # type: ignore
        activations = [
            features[i + 1][:, 1:, :].transpose(1, 2) for i in self.features_depth
        ]  # skip cls token and transpose to (B, C, N)
        outputs: list[torch.Tensor] = []
        for head, act in zip(self.heads, activations):
            out = head(act).view(x.shape[0], -1)
            outputs.append(out)
        return torch.cat(outputs, dim=1)

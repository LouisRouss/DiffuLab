import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class GroupNorm32(nn.GroupNorm):
    def forward(self, input: Tensor) -> Tensor:
        return super().forward(input.float()).type(input.dtype)


def normalization(channels: int) -> GroupNorm32:
    """
    Make a standard normalization layer.
    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(32, channels)


class Upsample(nn.Module):
    """
    From https://github.com/openai/guided-diffusion under MIT license as of 2024-18-08

    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.

    """

    def __init__(self, channels: int, use_conv: bool, out_channels: int | None = None) -> None:
        super().__init__()  # type:ignore
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv2d(self.channels, self.out_channels, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[1] == self.channels
        x = F.interpolate(x, scale_factor=2, mode="nearest")  # type: ignore
        if self.use_conv:
            x = self.conv(x)
        return x  # type: ignore


class Downsample(nn.Module):
    """
    From https://github.com/openai/guided-diffusion under MIT license as of 2024-18-08

    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    """

    def __init__(self, channels: int, use_conv: bool, out_channels: int | None = None) -> None:
        super().__init__()  # type:ignore
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        stride = 2
        if use_conv:
            self.op = nn.Conv2d(self.channels, self.out_channels, 3, stride=stride, padding=1)
        else:
            assert self.channels == self.out_channels
            self.op = nn.AvgPool2d(kernel_size=stride, stride=stride)

    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[1] == self.channels
        return self.op(x)


def timestep_embedding(timesteps: Tensor, dim: int, max_period: int = 10000) -> Tensor:
    """
    From https://github.com/openai/guided-diffusion under MIT license as of 2024-18-08

    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
        device=timesteps.device
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class LabelEmbed(nn.Module):
    def __init__(self, num_classes: int, embed_dim: int, classifier_free_guidance: bool = False) -> None:
        super().__init__()  # type: ignore
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.classifier_free_guidance = classifier_free_guidance

        if classifier_free_guidance:
            self.embedding = nn.Embedding(num_classes + 1, embed_dim)
        else:
            self.embedding = nn.Embedding(num_classes, embed_dim)

    def drop_labels(
        self,
        labels: Tensor,
        p: float,
    ) -> Tensor:
        """
        Randomly drop labels from a batch.
        :param labels: an [N] tensor of labels.
        :param p: the probability of dropping a label.
        :return: an [N] tensor of modified labels.
        """
        return torch.where(torch.rand(labels.size(), device=labels.device) < p, self.num_classes, labels)

    def forward(self, labels: Tensor, p: float = 0) -> Tensor:
        """
        Embed a batch of labels.
        :param labels: an [N] tensor of labels.
        :param p: the probability of dropping a label.
        :return: an [N x embed_dim] tensor of embeddings.
        """
        if p > 0:
            assert self.classifier_free_guidance, "Label dropout is only supported with classifier-free guidance."
            labels = self.drop_labels(labels, p)
        return self.embedding(labels)


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim: int = 32, base: int = 10_000) -> None:
        super().__init__()  # type: ignore
        self.dim = dim
        self.base = base
        # Create positional encodings
        self.register_buffer(
            name="theta",
            tensor=torch.pow(base, torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim).reciprocal(),
        )

        self.register_buffer(name="cos", tensor=torch.empty(0))
        self.register_buffer(name="sin", tensor=torch.empty(0))

    def _cache(self, seq_len: int) -> None:
        if seq_len < self.cos.shape[0]:
            return
        t = torch.arange(seq_len, device=self.device, dtype=torch.float32)
        freqs = torch.outer(t, self.theta)
        embs = torch.cat([freqs, freqs], dim=-1)
        self.cos = embs.cos().to(self.dtype)
        self.sin = embs.sin().to(self.dtype)

    def _neg_half(self, x: Tensor) -> Tensor:
        return torch.cat([-x[:, :, :, self.dim // 2 :], x[:, :, :, : self.dim // 2]], dim=-1)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        seq_len = q.shape[1]
        self._cache(seq_len)

        # [batch_size, seq_length, num_heads, head_dim] -> [batch_size, num_heads, seq_length, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)

        # Q rotation
        q_rope, q_pass = q[..., : self.dim], q[..., self.dim :]
        q_neg_half = self._neg_half(q_rope)
        q_rope = (q_rope * self.cos[:seq_len]) + (q_neg_half * self.sin[:seq_len])
        q_rot = torch.cat((q_rope, q_pass), dim=-1)

        # K rotation
        k_rope, k_pass = k[..., : self.dim], k[..., self.dim :]
        k_neg_half = self._neg_half(k_rope)
        k_rope = (k_rope * self.cos[:seq_len]) + (k_neg_half * self.sin[:seq_len])
        k_rot = torch.cat((k_rope, k_pass), dim=-1)

        # [batch_size, num_heads, seq_length, head_dim] -> [batch_size, seq_length, num_heads, head_dim]
        q_rot = q_rot.transpose(1, 2)
        k_rot = k_rot.transpose(1, 2)

        return q_rot, k_rot, v

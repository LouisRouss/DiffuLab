import math
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
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
        embeddings = self.embedding(labels).squeeze(1)
        return embeddings


class RotaryPositionalEmbedding(nn.Module):
    theta: Tensor
    cos: Tensor
    sin: Tensor

    def __init__(self, dim: int = 32, base: int = 10_000) -> None:
        super().__init__()  # type: ignore
        self.dim = dim
        self.base = base
        # Create positional encodings
        self.register_buffer(
            name="theta",
            tensor=torch.pow(base, torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim).reciprocal(),
        )

        self.cos = torch.empty(0, requires_grad=False)
        self.sin = torch.empty(0, requires_grad=False)

    def _cache(self, seq_len: int) -> None:
        if seq_len <= self.cos.shape[0]:
            return
        t = torch.arange(seq_len, dtype=torch.float32, device=self.theta.device)  # type: ignore
        freqs = torch.outer(t, self.theta)
        embs = torch.cat([freqs, freqs], dim=-1)
        with torch.no_grad():
            self.cos = embs.cos().float()
            self.sin = embs.sin().float()

    def _neg_half(self, x: Tensor) -> Tensor:
        return torch.cat([-x[:, :, :, self.dim // 2 :], x[:, :, :, : self.dim // 2]], dim=-1)

    def forward(
        self,
        q: Float[Tensor, "batch_size seq_len n_heads head_dim"],
        k: Float[Tensor, "batch_size seq_len n_heads head_dim"],
        v: Float[Tensor, "batch_size seq_len n_heads head_dim"],
    ) -> Tuple[
        Float[Tensor, "batch_size seq_len n_heads head_dim"],
        Float[Tensor, "batch_size seq_len n_heads head_dim"],
        Float[Tensor, "batch_size seq_len n_heads head_dim"],
    ]:
        seq_len = q.shape[1]
        self._cache(seq_len)
        cos = self.cos.to(device=q.device, dtype=q.dtype)
        sin = self.sin.to(device=q.device, dtype=q.dtype)

        # [batch_size, seq_length, num_heads, head_dim] -> [batch_size, num_heads, seq_length, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)

        # Q rotation
        q_rope, q_pass = q[..., : self.dim], q[..., self.dim :]
        q_neg_half = self._neg_half(q_rope)
        q_rope = (q_rope * cos[:seq_len]) + (q_neg_half * sin[:seq_len])
        q_rot = torch.cat((q_rope, q_pass), dim=-1)

        # K rotation
        k_rope, k_pass = k[..., : self.dim], k[..., self.dim :]
        k_neg_half = self._neg_half(k_rope)
        k_rope = (k_rope * cos[:seq_len]) + (k_neg_half * sin[:seq_len])
        k_rot = torch.cat((k_rope, k_pass), dim=-1)

        # [batch_size, num_heads, seq_length, head_dim] -> [batch_size, seq_length, num_heads, head_dim]
        q_rot = q_rot.transpose(1, 2)
        k_rot = k_rot.transpose(1, 2)

        return q_rot, k_rot, v


class RMSNorm(torch.nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm) module.

    Args:
        - dim (int): The dimension of the input tensor to be normalized.

    Attributes:
        - scale (torch.nn.Parameter): A learnable scaling parameter of shape (dim,).

    Methods:
        - forward(x: Tensor) -> Tensor:
            Applies RMS normalization to the input tensor.

    Example:
        >>> rms_norm = RMSNorm(dim=512)
        >>> input_tensor = torch.randn(10, 512)
        >>> output_tensor = rms_norm(input_tensor)
    """

    def __init__(self, dim: int):
        super().__init__()  # type: ignore
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Float[Tensor, "... dim"]) -> Float[Tensor, "... dim"]:
        x_dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)
        return (x * rrms).to(dtype=x_dtype) * self.scale


class QKNorm(nn.Module):
    """
    A neural network module that applies RMS normalization to query and key tensors.

    Args:
        dim (int): The dimension of the input tensors.

    Attributes:
        query_norm (RMSNorm): The RMS normalization layer for the query tensor.
        key_norm (RMSNorm): The RMS normalization layer for the key tensor.

    Methods:
        - forward(q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
            Applies RMS normalization to the query and key tensors, and ensures they have the same type as the value tensor.
            Args:
                - q (Tensor): The query tensor.
                - k (Tensor): The key tensor.
                - v (Tensor): The value tensor.
            Returns:
                - tuple[Tensor, Tensor]: The normalized query and key tensors, both converted to the type of the value tensor.
    Example:
        >>> qknorm = QKNorm(dim=512)
        >>> query_tensor = torch.randn(10, 25, 512)
        >>> key_tensor = torch.randn(10, 25, 512)
        >>> value_tensor = torch.randn(10, 25, 512)
        >>> normalized_query, normalized_key = qknorm(query_tensor, key_tensor, value_tensor)
    """

    def __init__(self, dim: int):
        super().__init__()  # type: ignore
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(
        self,
        q: Float[Tensor, "batch_size seq_len dim"],
        k: Float[Tensor, "batch_size seq_len dim"],
        v: Float[Tensor, "batch_size seq_len dim"],
    ) -> tuple[Float[Tensor, "batch_size seq_len dim"], Float[Tensor, "batch_size seq_len dim"]]:
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)


@dataclass
class ModulationOut:
    alpha: Tensor
    beta: Tensor
    gamma: Tensor
    delta: Tensor
    epsilon: Tensor
    zeta: Tensor


class Modulation(nn.Module):
    """
    A neural network module that applies a linear transformation to the input tensor
    and splits the output into six chunks.

    Attributes:
        lin (nn.Linear): A linear layer that transforms the input tensor.

    Methods:
        __init__(dim: int):
            Initializes the Modulation module with the specified dimension.

        forward(vec: Tensor) -> ModulationOut:
            Applies the linear transformation to the input tensor, followed by
            the SiLU activation function, and splits the result into six chunks.

    Args:
        dim (int): The dimension of the input tensor.

    Example:
        >>> modulation = Modulation(dim=512)
        >>> input_tensor = torch.randn(10, 512)
        >>> output = modulation(input_tensor)
        >>> print(output.alpha.shape)  # Output: torch.Size([10, 1, 512])
    """

    def __init__(self, dim: int):
        super().__init__()  # type: ignore
        self.lin = nn.Linear(dim, 6 * dim, bias=True)

    def forward(self, vec: Float[Tensor, "... dim"]) -> ModulationOut:
        out = self.lin(nn.functional.silu(vec))
        if len(out.shape) == 2:
            out = out.unsqueeze(1)
        out = out.chunk(6, dim=-1)
        return ModulationOut(*out)


def modulate(x: Tensor, scale: Tensor, shift: Tensor) -> Tensor:
    return x * (1 + scale) + shift

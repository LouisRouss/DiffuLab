import math
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor


class GroupNorm32(nn.GroupNorm):
    def forward(self, input: Tensor) -> Tensor:
        return super().forward(input.float()).type(input.dtype)


def normalization(channels: int) -> GroupNorm32:
    """
    Make a standard normalization layer.

    Args:
        channels (int): number of input channels.
    Returns:
        GroupNorm32: an nn.Module for normalization.
    """
    return GroupNorm32(32, channels)


class Upsample(nn.Module):
    """
    From https://github.com/openai/guided-diffusion under MIT license as of 2024-18-08

    An upsampling layer with an optional convolution.
    """

    def __init__(self, channels: int, use_conv: bool, out_channels: int | None = None) -> None:
        """
        Args:
            channels (int): channels in the inputs and outputs.
            use_conv (bool): a bool determining if a convolution is applied.
            out_channels (int | None): if provided, the number of output channels.
        """
        super().__init__()  # type:ignore
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv2d(self.channels, self.out_channels, 3, padding=1)

    def forward(
        self, x: Float[Tensor, "batch_size channels height width"]
    ) -> Float[Tensor, "batch_size channels 2*height 2*width"]:
        assert x.shape[1] == self.channels
        x = F.interpolate(x, scale_factor=2, mode="nearest")  # type: ignore
        if self.use_conv:
            x = self.conv(x)
        return x  # type: ignore


class Downsample(nn.Module):
    """
    From https://github.com/openai/guided-diffusion under MIT license as of 2024-18-08

    A downsampling layer with an optional convolution.
    """

    def __init__(self, channels: int, use_conv: bool, out_channels: int | None = None) -> None:
        """
        Args:
            channels (int): channels in the inputs and outputs.
            use_conv (bool): a bool determining if a convolution is applied.
            out_channels (int | None): if provided, the number of output channels.
        """
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

    def forward(
        self, x: Float[Tensor, "batch_size channels height width"]
    ) -> Float[Tensor, "batch_size channels height//2 width//2"]:
        assert x.shape[1] == self.channels
        return self.op(x)


def timestep_embedding(
    timesteps: Float[Tensor, "batch_size"], dim: int, max_period: int = 10000
) -> Float[Tensor, "batch_size dim"]:
    """
    From https://github.com/openai/guided-diffusion under MIT license as of 2024-18-08

    Create sinusoidal timestep embeddings.

    Args:
        timesteps (Tensor): a 1D tensor of timesteps.
        dim (int): the dimension of the output embeddings.
        max_period (int): the maximum period for the sinusoidal functions.
    Returns:
        Tensor: a 2D tensor of shape [len(timesteps), dim] containing the sinusoidal embeddings.
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
        """
        Args:
            num_classes (int): the number of classes.
            embed_dim (int): the dimension of the embeddings.
            classifier_free_guidance (bool): if True, the embedding layer will have an extra class for classifier-free guidance.
        """
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
        labels: Int[Tensor, "batch_size"],
        p: float,
    ) -> Tensor:
        """
        Randomly drop labels from a batch.

        Args:
            labels (Tensor): an [N] tensor of labels.
            p (float): the probability of dropping a label.
        Returns:
            Tensor: a tensor of labels with some labels randomly replaced by the extra class
        """
        return torch.where(torch.rand(labels.size(), device=labels.device) < p, self.num_classes, labels)

    def forward(self, labels: Int[Tensor, "batch_size"], p: float = 0) -> Float[Tensor, "batch_size embed_dim"]:
        """
        Embed a batch of labels.
        Args:
            labels (Tensor): a [N] tensor of labels.
            p (float): the probability of dropping a label. If greater than 0, labels will be randomly replaced by the extra class.
        Returns:
            Tensor: an [N, embed_dim] tensor of embeddings.
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
    """
    Rotary Positional Embedding (RoPE) module.
    This module applies rotary positional encoding to the query and key tensors in a multi-head attention mechanism.
    """

    def __init__(self, dim: int = 32, base: int = 10_000) -> None:
        """
        Args:
            dim (int): the dimension of the positional encoding.
            base (int): the base for the exponential decay of frequencies.
        """
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
        """
        Cache the cosine and sine values for the given sequence length.
        Args:
            seq_len (int): the length of the sequence for which to cache the values.
        """
        if seq_len <= self.cos.shape[0]:
            return
        t = torch.arange(seq_len, dtype=torch.float32, device=self.theta.device)  # type: ignore
        freqs = torch.outer(t, self.theta)
        embs = torch.cat([freqs, freqs], dim=-1)
        with torch.no_grad():
            self.cos = embs.cos().float()
            self.sin = embs.sin().float()

    def _neg_half(
        self, x: Float[Tensor, "batch_size num_heads seq_length head_dim"]
    ) -> Float[Tensor, "batch_size num_heads seq_length head_dim"]:
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
        """
        Apply Rotary Positional Encoding to the query and key tensors.

        Args:
            q (Tensor): the query tensor of shape [batch_size, seq_len, n_heads, head_dim].
            k (Tensor): the key tensor of shape [batch_size, seq_len, n_heads, head_dim].
            v (Tensor): the value tensor of shape [batch_size, seq_len, n_heads, head_dim].
        Returns:
            Tuple[Tensor, Tensor, Tensor]: the rotated query and key tensors, and the unchanged value tensor.
        """
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
        __init__(embedding_dim: int, input_dim: int):
            Initializes the Modulation module with the specified dimensions.

        forward(vec: Tensor) -> ModulationOut:
            Applies the linear transformation to the input tensor, followed by
            the SiLU activation function, and splits the result into six chunks.

    Args:
        embedding_dim (int): The dimension of the input embedding tensor.
        input_dim (int): The dimension that determines the output size (6 * input_dim).

    Example:
        >>> modulation = Modulation(embedding_dim=256, input_dim=128)
        >>> input_tensor = torch.randn(10, 256)
        >>> output = modulation(input_tensor)
        >>> print(output.alpha.shape)  # Output: torch.Size([10, 1, 128])
    """

    def __init__(self, embedding_dim: int, input_dim: int):
        super().__init__()  # type: ignore
        self.lin = nn.Linear(embedding_dim, 6 * input_dim, bias=True)

    def forward(self, vec: Float[Tensor, "B dim"] | Float[Tensor, "B seq_len dim"]) -> ModulationOut:
        out = self.lin(nn.functional.silu(vec))
        if out.dim() == 2:
            out = out[:, None, :]
        out = out.chunk(6, dim=-1)

        return ModulationOut(*out)


def modulate(x: Tensor, scale: Tensor, shift: Tensor) -> Tensor:
    return x * (1 + scale) + shift

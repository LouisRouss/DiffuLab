# Recoded from scratch from https://arxiv.org/pdf/2403.03206, if you see any error please report it to the author of the repository

import logging
from typing import Any

import torch
import torch.nn as nn
from einops import rearrange
from jaxtyping import Bool, Float, Int
from torch import Tensor
from torch.utils.checkpoint import checkpoint  # type: ignore

from diffulab.networks.denoisers.common import Denoiser, ModelOutput
from diffulab.networks.embedders.common import ContextEmbedder, ContextEmbedderOutput
from diffulab.networks.utils.nn import (
    LabelEmbed,
    Modulation,
    ModulationOut,
    PackedSwiGLU,
    QKNorm,
    RotaryPositionalEmbeddingNDim,
    get_cos_sin_ndim_grid,
    modulate,
    timestep_embedding,
)
from diffulab.networks.utils.utils import zero_module


class DiTAttention(nn.Module):
    """
    DiTAttention is a multi-head self attention mechanism with rotary positional embeddings.

    Args:
        inner_dim (int): Dimension of the input.
        num_heads (int): Number of attention heads.
        rope_axes_dim (list[int]): List of dimensions for rotary positional embeddings.

    Attributes:
        num_heads (int): Number of attention heads.
        head_dim (int): Dimension of each attention head.
        scale (float): Scaling factor for attention scores.
        qkv (nn.Linear): Linear layer for query, key, and value projection of input.
        qk_norm (QKNorm): Normalization layer for input query and key.
        rope (RotaryPositionalEmbedding): Rotary positional embedding layer.
        proj_out (nn.Linear): Linear layer for projecting the output.

    Methods:
        forward(input: Tensor, context: Tensor) -> tuple[Tensor, Tensor]:
            Forward pass of the attention mechanism.

            Args:
                input (Tensor): Input tensor of shape (batch_size, seq_len, input_dim).

            Returns:
                Tensor: output tensor.
    Example:
        >>> dit_attention = DiTAttention(inner_dim=512, num_heads=8)
        >>> input_tensor = torch.randn(10, 25, 512)
        >>> cos_sin_rope = (torch.randn(25, 256), torch.randn(25, 256))
        >>> output = dit_attention(input_tensor, cos_sin_rope)

    """

    def __init__(self, inner_dim: int, num_heads: int, rope_axes_dim: list[int]) -> None:
        super().__init__()  # type: ignore
        self.num_heads = num_heads
        self.head_dim = inner_dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(inner_dim, 3 * inner_dim)
        self.qk_norm = QKNorm(inner_dim)
        self.rope = RotaryPositionalEmbeddingNDim(axes_dim=rope_axes_dim)
        self.proj_out = nn.Linear(inner_dim, inner_dim)

    def forward(
        self,
        input: Float[Tensor, "batch_size seq_len dim"],
        cos_sin_rope: tuple[Float[Tensor, "seq_len dim/2"], Float[Tensor, "seq_len dim/2"]],
        attn_mask: Bool[Tensor, "batch_size seq_len"] | Int[Tensor, "batch_size seq_len"] | None = None,
    ) -> Float[Tensor, "batch_size seq_len dim"]:
        input_q, input_k, input_v = self.qkv(input).chunk(3, dim=-1)
        input_q, input_k = self.qk_norm(input_q, input_k, input_v)

        q, k, v = (
            rearrange(input_q, "b n (h d) -> b n h d", h=self.num_heads),
            rearrange(input_k, "b n (h d) -> b n h d", h=self.num_heads),
            rearrange(input_v, "b n (h d) -> b n h d", h=self.num_heads),
        )
        q, k, v = self.rope(q=q, k=k, v=v, cos_sin=cos_sin_rope)
        q, k, v = map(lambda x: rearrange(x, "b n h d -> b h n d"), [q, k, v])

        attn_output = nn.functional.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            scale=self.scale,
            attn_mask=attn_mask.bool() if attn_mask is not None else None,
        )

        attn_output = rearrange(attn_output, "b h n d -> b n (h d)")

        output: Tensor = self.proj_out(attn_output)

        return output


class MMDiTAttention(nn.Module):
    """
    MMDiTAttention is a multi-head attention mechanism with rotary positional embeddings.

    Args:
        inner_dim (int): Dimension of the input.
        num_heads (int): Number of attention heads.
        rope_axes_dim (list[int]): List of dimensions for rotary positional embeddings.

    Attributes:
        num_heads (int): Number of attention heads.
        head_dim (int): Dimension of each attention head.
        scale (float): Scaling factor for attention scores.
        qkv_input (nn.Linear): Linear layer for query, key, and value projection of input.
        qkv_context (nn.Linear): Linear layer for query, key, and value projection of context.
        qk_norm_input (QKNorm): Normalization layer for input query and key.
        qk_norm_context (QKNorm): Normalization layer for context query and key.
        rope (RotaryPositionalEmbedding): Rotary positional embedding layer.
        input_proj_out (nn.Linear): Linear layer for projecting the output of the input.
        context_proj_out (nn.Linear): Linear layer for projecting the output of the context.

    Methods:
        forward
            Forward pass of the attention mechanism.
            Args:
                input (Tensor): Input tensor of shape (batch_size, seq_len, input_dim).
                context (Tensor): Context tensor of shape (batch_size, seq_len, context_dim).
                cos_sin_rope (tuple[Tensor, Tensor]): Tuple containing cosine and sine tensors for rotary embeddings.
                attn_mask (Tensor): Attention mask tensor.

            Returns:
                tuple[Tensor, Tensor]: Tuple containing the output tensors for input and context.

    Example:
        >>> mmdit_attention = MMDiTAttention(context_dim=512, input_dim=512, dim=512, num_heads=8)
        >>> input_tensor = torch.randn(10, 25, 512)
        >>> context_tensor = torch.randn(10, 32, 512)
        >>> cos_sin_rope = (torch.randn(57, 256), torch.randn(57, 256))
        >>> output_input, output_context = mmdit_attention(input_tensor, context_tensor, cos_sin_rope)
        >>> print(output_input.shape)  # Output: torch.Size([10, 25, 512])
        >>> print(output_context.shape)  # Output: torch.Size([10, 32, 512])
    """

    def __init__(
        self,
        inner_dim: int,
        num_heads: int,
        rope_axes_dim: list[int],
    ):
        super().__init__()  # type: ignore
        self.num_heads = num_heads
        self.head_dim = inner_dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv_input = nn.Linear(inner_dim, 3 * inner_dim)
        self.qkv_context = nn.Linear(inner_dim, 3 * inner_dim)
        self.qk_norm_input = QKNorm(inner_dim)
        self.qk_norm_context = QKNorm(inner_dim)

        self.rope = RotaryPositionalEmbeddingNDim(axes_dim=rope_axes_dim)

        self.input_proj_out = nn.Linear(inner_dim, inner_dim)
        self.context_proj_out = nn.Linear(inner_dim, inner_dim)

    def forward(
        self,
        input: Float[Tensor, "batch_size seq_len_input inner_dim"],
        context: Float[Tensor, "batch_size seq_len_context inner_dim"],
        cos_sin_rope: tuple[Float[Tensor, "seq_len dim/2"], Float[Tensor, "seq_len dim/2"]],
        attn_mask: Bool[Tensor, "batch_size seq_len_context"] | Int[Tensor, "batch_size seq_len_context"] | None = None,
    ) -> tuple[Float[Tensor, "batch_size seq_len input_dim"], Float[Tensor, "batch_size seq_len context_dim"]]:
        input_q, input_k, input_v = self.qkv_input(input).chunk(3, dim=-1)
        context_q, context_k, context_v = self.qkv_context(context).chunk(3, dim=-1)

        input_q, input_k = self.qk_norm_input(input_q, input_k, input_v)
        context_q, context_k = self.qk_norm_context(context_q, context_k, context_v)

        q, k, v = (
            rearrange(torch.cat([context_q, input_q], dim=1), "b n (h d) -> b n h d", h=self.num_heads),
            rearrange(torch.cat([context_k, input_k], dim=1), "b n (h d) -> b n h d", h=self.num_heads),
            rearrange(torch.cat([context_v, input_v], dim=1), "b n (h d) -> b n h d", h=self.num_heads),
        )
        q, k, v = self.rope(q=q, k=k, v=v, cos_sin=cos_sin_rope)
        q, k, v = map(lambda x: rearrange(x, "b n h d -> b h n d"), [q, k, v])

        if attn_mask is not None:
            attn_mask = torch.cat(
                [
                    attn_mask.bool(),
                    torch.ones(attn_mask.size(0), input.size(1), device=attn_mask.device).bool(),
                ],
                dim=1,
            )
            attn_mask = attn_mask[:, None, None, :]

        attn_output = nn.functional.scaled_dot_product_attention(
            query=q, key=k, value=v, scale=self.scale, attn_mask=attn_mask
        )

        attn_output = rearrange(attn_output, "b h n d -> b n (h d)")

        input_output = self.input_proj_out(attn_output[:, context.size(1) :, :])
        context_output = self.context_proj_out(attn_output[:, : context.size(1), :])
        return input_output, context_output


class DiTBlock(nn.Module):
    """
    DiTBlock is a neural network module that performs modulation, normalization,
    attention, and multi-layer perceptron (MLP) operations on input tensor.

    Args:
        input_dim (int): Dimension of the input tensor.
        hidden_dim (int): Dimension of the hidden layer in the attention mechanism.
        embedding_dim (int): Dimension of the embedding used in modulation.
        num_heads (int): Number of attention heads.
        mlp_ratio (int): Ratio used to determine the size of the MLP layers.
        rope_axes_dim (list[int]): List of dimensions for rotary positional embeddings.
        use_checkpoint (bool): Whether to use gradient checkpointing for memory efficiency. Default is False.

    Methods:
        forward
            Performs the forward pass of the MMDiTBlock.
            Args:
                input (Tensor): The input tensor.
                y (Tensor): An additional tensor, not used in the current implementation.
                cos_sin_rope (tuple): Tuple containing cosine and sine tensors for rotary embeddings.
            Returns:
                Tuple[Tensor, Tensor]: The processed input tensor.

    Example:
        >>> dit_block = DiTBlock(input_dim=512, hidden_dim=512, embedding_dim=512, num_heads=8, mlp_ratio=4)
        >>> input_tensor = torch.randn(10, 25, 512)
        >>> y = torch.randn(10, 512)
        >>> cos_sin_rope = (torch.randn(25, 256), torch.randn(25, 256))
        >>> output_tensor = dit_block(input_tensor, y)
        >>> print(output_tensor.shape)  # Output: torch.Size([10, 25, 512])
    """

    def __init__(
        self,
        inner_dim: int,
        embedding_dim: int,
        num_heads: int,
        mlp_ratio: int,
        rope_axes_dim: list[int],
        use_checkpoint: bool = False,
    ):
        super().__init__()  # type: ignore
        self.modulation = Modulation(embedding_dim, inner_dim)
        self.norm_1 = nn.LayerNorm(inner_dim)
        self.attention = DiTAttention(inner_dim, num_heads, rope_axes_dim=rope_axes_dim)
        self.norm_2 = nn.LayerNorm(inner_dim)
        self.mlp_input = nn.Sequential(
            nn.Linear(inner_dim, mlp_ratio * inner_dim * 2),
            PackedSwiGLU(),
            nn.Linear(mlp_ratio * inner_dim, inner_dim),
        )
        self.use_checkpoint = use_checkpoint

    def forward(
        self,
        input: Float[Tensor, "batch_size seq_len inner_dim"],
        y: Float[Tensor, "batch_size embedding_dim"],
        cos_sin_rope: tuple[Float[Tensor, "seq_len dim/2"], Float[Tensor, "seq_len dim/2"]],
    ) -> Float[Tensor, "batch_size seq_len inner_dim"]:
        """
        Forward pass of the DiTBlock applying modulation and attention mechanisms.
        Args:
            - input (Tensor): The input tensor to be processed
            - y (Tensor): The conditioning tensor used for modulation
            - cos_sin_rope (tuple): Tuple containing cosine and sine tensors for rotary embeddings
        Returns:
            Tensor: The processed input tensor with residual connection
        """
        return (
            checkpoint(self._forward, *(input, y, cos_sin_rope), use_reentrant=False)
            if self.use_checkpoint
            else self._forward(input, y, cos_sin_rope)
        )  # type: ignore

    def _forward(
        self,
        input: Float[Tensor, "batch_size seq_len embedding_dim"],
        y: Float[Tensor, "batch_size embedding_dim"] | Float[Tensor, "batch_size seq_len embedding_dim"],
        cos_sin_rope: tuple[Float[Tensor, "seq_len dim/2"], Float[Tensor, "seq_len dim/2"]],
    ) -> Tensor:
        modulation: ModulationOut = self.modulation(y)

        modulated_input = (
            input
            + self.attention(
                modulate(self.norm_1(input), scale=modulation.alpha, shift=modulation.beta), cos_sin_rope=cos_sin_rope
            )
            * modulation.gamma
        )

        modulated_input = modulated_input + (
            self.mlp_input(modulate(self.norm_2(modulated_input), scale=modulation.delta, shift=modulation.epsilon))
            * modulation.zeta
        )

        return modulated_input


class MMDiTBlock(nn.Module):
    """
    MMDiTBlock is a neural network module that performs modulation, normalization,
    attention, and multi-layer perceptron (MLP) operations on input and context tensors.

    Args:
        inner_dim (int): Dimension of the input tensor.
        embedding_dim (int): Dimension of the embedding used in modulation.
        num_heads (int): Number of attention heads.
        mlp_ratio (int): Ratio used to determine the size of the MLP layers.
        rope_axes_dim (list[int]): List of dimensions for rotary positional embeddings.
        use_checkpoint (bool): Whether to use gradient checkpointing for memory efficiency. Default is False.

    Methods:
        forward
            Performs the forward pass of the MMDiTBlock.
            Args:
                input (Tensor): The input tensor.
                y (Tensor): An additional tensor, not used in the current implementation.
                context (Tensor): The context tensor.
                cos_sin_rope (tuple): Tuple containing cosine and sine tensors for rotary embeddings.
                attn_mask (Tensor): Attention mask tensor.
            Returns:
                Tuple[Tensor, Tensor]: The modulated input and context tensors.

    Example:
        >>> mmdit_block = MMDiTBlock(inner_dim=512, embedding_dim=512, num_heads=8, mlp_ratio=4, rope_axes_dim=[256])
        >>> input_tensor = torch.randn(10, 25, 512)
        >>> y = torch.randn(10, 512)
        >>> context_tensor = torch.randn(10, 32, 512)
        >>> cos_sin_rope = (torch.randn(57, 256), torch.randn(57, 256))
        >>> output_input, output_context = mmdit_block(input_tensor, y, context_tensor, cos_sin_rope)
        >>> print(output_input.shape)  # Output: torch.Size([10, 25, 512])
        >>> print(output_context.shape)  # Output: torch.Size([10, 32, 512])
    """

    def __init__(
        self,
        inner_dim: int,
        embedding_dim: int,
        num_heads: int,
        mlp_ratio: int,
        rope_axes_dim: list[int],
        use_checkpoint: bool = False,
    ):
        super().__init__()  # type: ignore
        self.modulation_context = Modulation(embedding_dim, inner_dim)
        self.modulation_input = Modulation(embedding_dim, inner_dim)

        self.context_norm_1 = nn.LayerNorm(inner_dim)
        self.input_norm_1 = nn.LayerNorm(inner_dim)

        self.attention = MMDiTAttention(inner_dim, num_heads, rope_axes_dim=rope_axes_dim)

        self.context_norm_2 = nn.LayerNorm(inner_dim)
        self.input_norm_2 = nn.LayerNorm(inner_dim)

        self.mlp_context = nn.Sequential(
            nn.Linear(inner_dim, mlp_ratio * inner_dim * 2),
            PackedSwiGLU(),
            nn.Linear(mlp_ratio * inner_dim, inner_dim),
        )
        self.mlp_input = nn.Sequential(
            nn.Linear(inner_dim, mlp_ratio * inner_dim * 2),
            PackedSwiGLU(),
            nn.Linear(mlp_ratio * inner_dim, inner_dim),
        )
        self.use_checkpoint = use_checkpoint

    def forward(
        self,
        input: Float[Tensor, "batch_size seq_len_input inner_dim"],
        y: Float[Tensor, "batch_size embedding_dim"],
        context: Float[Tensor, "batch_size seq_len_context inner_dim"],
        cos_sin_rope: tuple[Float[Tensor, "seq_len dim/2"], Float[Tensor, "seq_len dim/2"]],
        attn_mask: Bool[Tensor, "batch_size seq_len_context"] | Int[Tensor, "batch_size seq_len_context"] | None = None,
    ) -> tuple[
        Float[Tensor, "batch_size seq_len_input embedding_dim"], Float[Tensor, "batch_size seq_len_context context_dim"]
    ]:
        """
        Forward pass of the MMDiT module applying modulation and attention mechanisms.
        Args:
            - input (Tensor): The input tensor to be processed
            - y (Tensor): The conditioning tensor used for modulation
            - context (Tensor): The context tensor to be processed alongside input
            - cos_sin_rope (tuple): Tuple containing cosine and sine tensors for rotary embeddings
            - attn_mask (Tensor | None): Optional attention mask for context
        Returns:
            tuple: A tuple containing:
                - Tensor: The modulated and processed input tensor with residual connection
                - Tensor: The modulated and processed context tensor with residual connection
        The forward pass applies the following operations:
        1. Input and context modulation using separate modulation networks
        2. Cross-attention between modulated input and context
        3. Secondary modulation and normalization
        4. MLP processing with modulation
        5. Residual connections for both input and context paths
        """
        return (
            checkpoint(self._forward, *(input, y, context, cos_sin_rope, attn_mask), use_reentrant=False)
            if self.use_checkpoint
            else self._forward(input, y, context, cos_sin_rope, attn_mask)
        )  # type: ignore

    def _forward(
        self,
        input: Float[Tensor, "batch_size seq_len embedding_dim"],
        y: Float[Tensor, "batch_size embedding_dim"],
        context: Float[Tensor, "batch_size seq_len context_dim"],
        cos_sin_rope: tuple[Float[Tensor, "seq_len dim/2"], Float[Tensor, "seq_len dim/2"]],
        attn_mask: Bool[Tensor, "batch_size seq_len_context"] | Int[Tensor, "batch_size seq_len_context"] | None = None,
    ):
        modulation_input: ModulationOut = self.modulation_input(y)
        modulation_context: ModulationOut = self.modulation_context(y)

        modulated_input = modulate(self.input_norm_1(input), scale=modulation_input.alpha, shift=modulation_input.beta)
        modulated_context = modulate(
            self.context_norm_1(context), scale=modulation_context.alpha, shift=modulation_context.beta
        )

        modulated_input, modulated_context = self.attention(
            modulated_input, modulated_context, cos_sin_rope=cos_sin_rope, attn_mask=attn_mask
        )
        modulated_input = input + modulated_input * modulation_input.gamma
        modulated_context = context + modulated_context * modulation_context.gamma

        modulated_input = (
            modulated_input
            + self.mlp_input(
                modulate(
                    self.input_norm_2(modulated_input), scale=modulation_input.delta, shift=modulation_input.epsilon
                )
            )
            * modulation_input.zeta
        )
        modulated_context = (
            modulated_context
            + self.mlp_context(
                modulate(
                    self.context_norm_2(modulated_context),
                    scale=modulation_context.delta,
                    shift=modulation_context.epsilon,
                )
            )
            * modulation_context.zeta
        )

        return modulated_input, modulated_context


class MMDiTSingleStreamBlock(nn.Module):
    def __init__(
        self,
        inner_dim: int,
        embedding_dim: int,
        num_heads: int,
        mlp_ratio: int,
        rope_axes_dim: list[int],
        use_checkpoint: bool = False,
    ):
        super().__init__()  # type: ignore
        self.mlp = nn.Sequential(
            nn.Linear(inner_dim, mlp_ratio * inner_dim * 2),
            PackedSwiGLU(),
            nn.Linear(mlp_ratio * inner_dim, inner_dim),
        )
        self.attention = DiTAttention(inner_dim, num_heads, rope_axes_dim=rope_axes_dim)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(embedding_dim, 3 * inner_dim))
        self.norm = nn.LayerNorm(inner_dim)
        self.use_checkpoint = use_checkpoint

    def forward(
        self,
        input: Float[Tensor, "batch_size seq_len inner_dim"],
        y: Float[Tensor, "batch_size embedding_dim"],
        context: Float[Tensor, "batch_size seq_len inner_dim"],
        cos_sin_rope: tuple[Float[Tensor, "seq_len dim/2"], Float[Tensor, "seq_len dim/2"]],
        attn_mask: Bool[Tensor, "batch_size seq_len_context"] | Int[Tensor, "batch_size seq_len_context"] | None = None,
    ) -> tuple[
        Float[Tensor, "batch_size seq_len_input inner_dim"], Float[Tensor, "batch_size seq_len_context inner_dim"]
    ]:
        return (
            checkpoint(self._forward, *(input, y, context, cos_sin_rope, attn_mask), use_reentrant=False)
            if self.use_checkpoint
            else self._forward(input, y, context, cos_sin_rope, attn_mask)
        )  # type: ignore

    def _forward(
        self,
        input: Float[Tensor, "batch_size seq_len inner_dim"],
        y: Float[Tensor, "batch_size embedding_dim"],
        context: Float[Tensor, "batch_size seq_len inner_dim"],
        cos_sin_rope: tuple[Float[Tensor, "seq_len dim/2"], Float[Tensor, "seq_len dim/2"]],
        attn_mask: Bool[Tensor, "batch_size seq_len_context"] | Int[Tensor, "batch_size seq_len_context"] | None = None,
    ):
        latents = torch.cat([context, input], dim=1)
        if attn_mask is not None:
            attn_mask = torch.cat(
                [
                    attn_mask.bool(),
                    torch.ones(attn_mask.size(0), input.size(1), device=attn_mask.device).bool(),
                ],
                dim=1,
            )
            attn_mask = attn_mask[:, None, None, :]

        modulation = self.modulation(y)
        if modulation.dim() == 2:
            modulation = modulation[:, None, :]
        alpha, beta, gamma = modulation.chunk(3, dim=-1)
        modulated_latents = modulate(self.norm(latents), scale=alpha, shift=beta)

        latents = (
            latents
            + (
                self.attention(modulated_latents, cos_sin_rope=cos_sin_rope, attn_mask=attn_mask)
                + self.mlp(modulated_latents)
            )
            * gamma
        )
        return latents[:, context.size(1) :, :], latents[:, : context.size(1), :]


class ModulatedLastLayer(nn.Module):
    def __init__(self, embedding_dim: int, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()  # type: ignore
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(embedding_dim, 2 * hidden_size))

    def forward(self, x: Float[Tensor, "batch_size seq_len dim"], vec: Float[Tensor, "batch_size dim"]) -> Tensor:
        modulation = self.adaLN_modulation(vec)
        if modulation.dim() == 2:
            modulation = modulation[:, None, :]
        alpha, beta = modulation.chunk(2, dim=-1)
        x = modulate(self.norm_final(x), scale=alpha, shift=beta)
        x = self.linear(x)
        return x


class MMDiT(Denoiser):
    """
    Multimodal DiT architecture following https://arxiv.org/pdf/2403.03206

    This module implements a DiT-style transformer that can run in two modes:
    - simple_dit=True: a single-stream DiT with label conditioning only (no multimodal context).
    - simple_dit=False: an MMDiT with self attention with contextual tokens from a `context_embedder`.

    In both modes the input image is patchified with a convolutional projection, processed by a stack
    of DiT/MMDiT blocks, then projected back to per-patch predictions and finally unpatchified to
    the image space via a modulation-aware last layer.

    Args:
        simple_dit (bool): If True, use DiT blocks with class-label conditioning only (no context
            or cross-attention). If False, use MMDiT blocks with cross-attention to contextual tokens
            produced by `context_embedder`. Default: False.
        input_channels (int): Number of channels of the main input x. Default: 3.
        output_channels (int | None): Number of channels to predict. If None, equals `input_channels`.
            Default: None.
        inner_dim (int): Token/patch embedding width for the stream. Default: 4096.
        embedding_dim (int): Conditioning embedding width (for timestep/labels/pooled context) used
            by modulation layers and the last prediction layer. Default: 4096.
        num_heads (int): Number of attention heads in each block. Default: 16.
        mlp_ratio (int): Expansion ratio for the MLP in each block. Default: 4.
        patch_size (int): Side length P of square patches. Images are projected with stride P. Default: 16.
        depth (int): Number of DiT/MMDiT blocks. Default: 38.
        context_dim (int): Model width for contextual tokens after `context_embed` when
            `simple_dit=False`. Ignored when `simple_dit=True`. Default: 4096.
        rope_base (int): Base frequency for RoPE. Default: 10000.
        partial_rotary_factor (float): Fraction of each head dimension using RoPE.
            1.0 means full rotary. Default: 1.0.
        rope_axes_dim (list[int] | None): List of dimensions for rotary positional embeddings.
            When `simple_dit=True`, should contain 2 integers for H and W axes. When `simple_dit=False`,
            should contain 3 integers for L, H, W axes. If None, defaults are used based on
            partial_rotary_factor and the heads_dim. Default: None
        frequency_embedding (int): Size of the Fourier timestep embedding before the time MLP.
            Default: 256.
        n_classes (int | None): Number of classes for label conditioning in `simple_dit` mode.
            Required to use classifier-free guidance with labels. Must be None when using
            a `context_embedder`. Default: None.
        classifier_free (bool): Enables classifier-free guidance. In `simple_dit`, it applies to
            dropped labels; in MMDiT mode, it is forwarded to the context embedder which may drop
            context. Default: False.
        context_embedder (ContextEmbedder | None): When `simple_dit=False`, a module returning
            `ContextEmbedderOutput`. Must be provided for text/image conditioning and must be None
            when `simple_dit=True`. If the embedder returns pooled and token embeddings
            (`n_output == 2`), pooled features are fused into the timestep embedding via an MLP.
            Default: None.
        use_checkpoint (bool): Enable torch.utils.checkpoint in blocks to trade compute for memory.
            Default: False.
    """

    def __init__(
        self,
        simple_dit: bool = False,
        input_channels: int = 3,
        output_channels: int | None = None,
        inner_dim: int = 4096,
        embedding_dim: int = 4096,
        num_heads: int = 16,
        mlp_ratio: int = 4,
        patch_size: int = 16,
        depth: int = 38,
        n_single_stream_blocks: int = 0,
        rope_base: int = 10_000,
        partial_rotary_factor: float = 1,
        rope_axes_dim: list[int] | None = None,
        frequency_embedding: int = 256,
        n_classes: int | None = None,
        classifier_free: bool = False,
        context_embedder: ContextEmbedder | None = None,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        assert not (n_classes is not None and context_embedder is not None), (
            "n_classes and context_embedder cannot both be specified"
        )
        self.simple_dit = simple_dit
        self.patch_size = patch_size
        self.input_channels = input_channels
        if not output_channels:
            output_channels = input_channels
        self.output_channels = output_channels
        self.context_embedder = context_embedder
        self.frequency_embedding = frequency_embedding
        self.rope_base = rope_base

        self.n_classes = n_classes
        self.classifier_free = classifier_free

        heads_dim = inner_dim // num_heads
        if not self.simple_dit:
            assert self.context_embedder is not None, "for MMDiT context embedder must be provided"
            assert isinstance(self.context_embedder.output_size, tuple) and all(
                isinstance(i, int) for i in self.context_embedder.output_size
            ), "context_embedder.output_size must be a tuple of integers"

            self.pooled_embedding = False
            self.mlp_pooled_context = None
            if self.context_embedder.n_output == 2:
                self.pooled_embedding = True
                self.mlp_pooled_context = nn.Sequential(
                    nn.Linear(self.context_embedder.output_size[0], embedding_dim * 2),
                    nn.SiLU(),
                    nn.Linear(embedding_dim * 2, embedding_dim),
                )
                self.context_embed = nn.Linear(self.context_embedder.output_size[1], inner_dim)
            else:
                assert self.context_embedder.n_output == 1
                self.context_embed = nn.Linear(self.context_embedder.output_size[0], inner_dim)
            if rope_axes_dim is None:
                rope_axes_dim = [
                    int((partial_rotary_factor * heads_dim) // 3),  # L for text, set to 0 for image tokens
                    int((partial_rotary_factor * heads_dim) // 3),  # H set to 0 for text
                    int((partial_rotary_factor * heads_dim) // 3),  # W set to 0 for text
                ]

        else:
            self.label_embed = (
                LabelEmbed(self.n_classes, embedding_dim, self.classifier_free) if self.n_classes is not None else None
            )
            if rope_axes_dim is None:
                rope_axes_dim = [
                    int((partial_rotary_factor * heads_dim) // 2),  # H
                    int((partial_rotary_factor * heads_dim) // 2),  # W
                ]
            if n_single_stream_blocks > 0:
                logging.warning(
                    "n_single_stream_blocks is ignored when simple_dit=True. All blocks are single-stream DiT blocks."
                )
                n_single_stream_blocks = depth

        self.rope_axes_dim = rope_axes_dim
        self.last_layer = ModulatedLastLayer(
            embedding_dim=embedding_dim,
            hidden_size=inner_dim,
            patch_size=self.patch_size,
            out_channels=self.output_channels,
        )
        self.time_embed = nn.Sequential(
            nn.Linear(self.frequency_embedding, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

        self.conv_proj = nn.Conv2d(self.input_channels, inner_dim, kernel_size=self.patch_size, stride=self.patch_size)

        self.layers = nn.ModuleList(
            [
                MMDiTBlock(
                    inner_dim=inner_dim,
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    rope_axes_dim=self.rope_axes_dim,
                    use_checkpoint=use_checkpoint,
                )
                if not self.simple_dit
                else DiTBlock(
                    inner_dim=inner_dim,
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    rope_axes_dim=self.rope_axes_dim,
                    use_checkpoint=use_checkpoint,
                )
                for _ in range(depth - n_single_stream_blocks)
            ]
            + [
                MMDiTSingleStreamBlock(
                    inner_dim=inner_dim,
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    rope_axes_dim=self.rope_axes_dim,
                    use_checkpoint=use_checkpoint,
                )
                for _ in range(n_single_stream_blocks)
            ]
        )

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:  # type: ignore
                nn.init.constant_(module.bias, 0)
        if isinstance(module, Modulation):
            zero_module(module)
        if isinstance(module, ModulatedLastLayer):
            zero_module(module.adaLN_modulation)

    def patchify(
        self, x: Float[Tensor, "batch_size channels height width"]
    ) -> Float[Tensor, "batch_size num_patches patch_dim"]:
        """
        Convert image tensor into patches using convolutional projection.
        Args:
            x (Tensor): Input tensor of shape (B, C, H, W)
        Returns:
            Tensor: Patchified tensor of shape (B, num_patches, patch_dim)
        """
        _, _, H, W = x.shape
        self.original_size = (H, W)

        x = self.conv_proj(x)
        _, _, Hp, Wp = x.shape
        self.grid_size = (Hp, Wp)
        x = rearrange(x, "b c h w -> b (h w) c")

        return x

    def unpatchify(
        self, x: Float[Tensor, "batch_size seq_len patch_dim"]
    ) -> Float[Tensor, "batch_size channels height width"]:
        """
        Convert patches back into the original image tensor.
        Args:
            x (Tensor): Patchified tensor of shape (B, num_patches, patch_dim)
        Returns:
            Tensor: Reconstructed image tensor of shape (B, C, H, W)
        """
        # Reshape the tensor to the original image dimensions
        x = rearrange(
            x,
            "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
            h=self.grid_size[0],
            w=self.grid_size[1],
            p1=self.patch_size,
            p2=self.patch_size,
            c=self.output_channels,
        )
        return x

    def mmdit_forward(
        self,
        x: Float[Tensor, "batch_size seq_len patch_dim"],
        timesteps: Float[Tensor, "batch_size"],
        initial_context: Any | None = None,
        p: float = 0.0,
        intermediate_features: bool = False,
    ) -> ModelOutput:
        assert self.context_embedder is not None, "for MMDiT context embedder must be provided"
        emb = self.time_embed(timestep_embedding(timesteps, self.frequency_embedding))
        context_output: ContextEmbedderOutput = self.context_embedder(initial_context, p)
        if self.pooled_embedding:
            assert self.mlp_pooled_context is not None, (
                "for MMDiT with pooled context, mlp_pooled_context must be defined"
            )
            assert "pooled_embeddings" in context_output, "pooled embeddings must be in context_output"
            context_pooled = context_output["pooled_embeddings"]
            emb = self.mlp_pooled_context(context_pooled) + emb

        context = context_output["embeddings"]
        context: Tensor = self.context_embed(context)
        attn_mask = context_output.get("attn_mask", None)

        # pos_ids: [S, n_axes] positional IDs along each axis for rope
        # in mmdit attention we concat with context first. Context have 0,0 for h w
        # text: (t>0, 0, 0)
        text_pos_ids = torch.stack(
            [
                torch.arange(1, context.shape[1] + 1, device=x.device),
                torch.zeros(context.shape[1], device=x.device, dtype=torch.long),
                torch.zeros(context.shape[1], device=x.device, dtype=torch.long),
            ],
            dim=-1,
        )

        # image: (0, h, w)
        img_pos_ids = torch.stack(
            torch.meshgrid(
                torch.zeros(1, device=x.device, dtype=torch.long),
                torch.arange(self.grid_size[0], device=x.device),
                torch.arange(self.grid_size[1], device=x.device),
                indexing="ij",
            ),
            dim=-1,
        ).view(-1, 3)

        pos_ids = torch.cat([text_pos_ids, img_pos_ids], dim=0).unsqueeze(0).repeat(x.size(0), 1, 1)
        cos_sin_rope = get_cos_sin_ndim_grid(pos_ids, base=self.rope_base, axes_dim=self.rope_axes_dim)

        features: list[Tensor] | None = [] if intermediate_features else None
        # Pass through each layer sequentially
        for layer in self.layers:
            x, context = layer(x, emb, context, cos_sin_rope=cos_sin_rope, attn_mask=attn_mask)
            if features:
                features.append(x)

        x = self.last_layer(x, emb)
        if features:
            features.append(x)
        model_output: ModelOutput = {"x": x}
        if features:
            model_output["features"] = features
        return model_output

    def simple_dit_forward(
        self,
        x: Float[Tensor, "batch_size seq_len patch_dim"],
        timestep: Float[Tensor, "batch_size"],
        p: float = 0.0,
        y: Int[Tensor, "batch_size"] | None = None,
        intermediate_features: bool = False,
    ) -> ModelOutput:
        if p > 0:
            assert self.n_classes, (
                "probability of dropping for classifier free guidance is only available if a number of classes is set"
            )

        emb = self.time_embed(timestep_embedding(timestep, self.frequency_embedding))
        if self.label_embed is not None:
            emb = emb + self.label_embed(y, p)

        # pos_ids: [B, S, n_axes] positional IDs along each axis for rope
        pos_ids = (
            torch.stack(
                torch.meshgrid(
                    [
                        torch.arange(self.grid_size[0], device=x.device),
                        torch.arange(self.grid_size[1], device=x.device),
                    ],
                    indexing="ij",
                ),
                dim=-1,
            )
            .view(-1, 2)
            .unsqueeze(0)
            .repeat(x.size(0), 1, 1)
        )
        cos_sin_rope = get_cos_sin_ndim_grid(pos_ids, base=self.rope_base, axes_dim=self.rope_axes_dim)

        # Pass through each layer sequentially
        features: list[Tensor] | None = [] if intermediate_features else None
        for layer in self.layers:
            x = layer(x, emb, cos_sin_rope=cos_sin_rope)
            if features:
                features.append(x)

        x = self.last_layer(x, emb)
        if features:
            features.append(x)
        model_output: ModelOutput = {"x": x}
        if features:
            model_output["features"] = features
        return model_output

    def forward(
        self,
        x: Float[Tensor, "batch_size channels height width"],
        timesteps: Float[Tensor, "batch_size"],
        initial_context: Any | None = None,
        p: float = 0.0,
        y: Int[Tensor, "batch_size"] | None = None,
        x_context: Tensor | None = None,
        intermediate_features: bool = False,
    ) -> ModelOutput:
        assert not (initial_context is not None and y is not None), "initial_context and y cannot both be specified"
        if p > 0:
            assert self.classifier_free, (
                "probability of dropping for classifier free guidance is only available if model is set up to be classifier free"
            )
        if x_context is not None:
            x = torch.cat([x, x_context], dim=1)

        x = self.patchify(x)
        if self.simple_dit:
            model_output = self.simple_dit_forward(x, timesteps, p, y, intermediate_features)
        else:
            model_output = self.mmdit_forward(x, timesteps, initial_context, p, intermediate_features)
        model_output["x"] = self.unpatchify(model_output["x"])

        return model_output

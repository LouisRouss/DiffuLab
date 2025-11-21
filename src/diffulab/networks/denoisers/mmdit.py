# Recoded from scratch from https://arxiv.org/pdf/2403.03206, if you see any error please report it to the author of the repository

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
    QKNorm,
    RMSNorm,
    RotaryPositionalEmbedding,
    RotaryPositionalEmbedding2D,
    modulate,
    timestep_embedding,
)


class DiTAttention(nn.Module):
    """
    DiTAttention is a multi-head self attention mechanism with rotary positional embeddings.

    Args:
        input_dim (int): Dimension of the input.
        dim (int): Inner Attention dimension.
        num_heads (int): Number of attention heads.
        partial_rotary_factor (float, optional): Factor for partial rotary embeddings. Default is 0.5.
        base (int, optional): Base for rotary positional embeddings. Default is 10000.

    Attributes:
        num_heads (int): Number of attention heads.
        head_dim (int): Dimension of each attention head.
        scale (float): Scaling factor for attention scores.
        qkv (nn.Linear): Linear layer for query, key, and value projection of input.
        qk_norm (QKNorm): Normalization layer for input query and key.
        partial_rotary_factor (float): Factor for partial rotary embeddings.
        rotary_dim (int): Dimension for rotary embeddings.
        rope (RotaryPositionalEmbedding): Rotary positional embedding layer.
        input_proj (nn.Linear): Linear layer for projecting the output of the input.

    Methods:
        forward(input: Tensor, context: Tensor) -> tuple[Tensor, Tensor]:
            Forward pass of the attention mechanism.

            Args:
                input (Tensor): Input tensor of shape (batch_size, seq_len, input_dim).

            Returns:
                Tensor: output tensor.
    Example:
        >>> dit_attention = DiTAttention(input_dim=512, dim=512, num_heads=8)
        >>> input_tensor = torch.randn(10, 25, 512)
        >>> output_tensor = dit_attention(input_tensor)
        >>> print(output_tensor.shape)  # Output: torch.Size([10, 25, 512])
    """

    def __init__(
        self,
        input_dim: int,
        dim: int,
        num_heads: int,
        partial_rotary_factor: float = 1,
        base: int = 10000,
    ) -> None:
        super().__init__()  # type: ignore
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(input_dim, 3 * dim)
        self.qk_norm = QKNorm(dim)

        self.partial_rotary_factor = partial_rotary_factor
        self.rotary_dim = int(self.head_dim * self.partial_rotary_factor)
        self.rope = RotaryPositionalEmbedding2D(dim=self.rotary_dim, base=base)

        self.proj_out = nn.Linear(dim, input_dim)

    def forward(
        self,
        input: Float[Tensor, "batch_size seq_len dim"],
        shape: tuple[int, int] | None = None,
    ) -> Float[Tensor, "batch_size seq_len dim"]:
        input_q, input_k, input_v = self.qkv(input).chunk(3, dim=-1)
        input_q, input_k = self.qk_norm(input_q, input_k, input_v)

        q, k, v = (
            rearrange(input_q, "b n (h d) -> b n h d", h=self.num_heads),
            rearrange(input_k, "b n (h d) -> b n h d", h=self.num_heads),
            rearrange(input_v, "b n (h d) -> b n h d", h=self.num_heads),
        )
        q, k, v = self.rope(q=q, k=k, v=v, shape=shape)
        q, k, v = map(lambda x: rearrange(x, "b n h d -> b h n d"), [q, k, v])

        attn_output = nn.functional.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            scale=self.scale,
        )

        attn_output = rearrange(attn_output, "b h n d -> b n (h d)")

        output: Tensor = self.proj_out(attn_output)

        return output


class MMDiTAttention(nn.Module):
    """
    MMDiTAttention is a multi-head attention mechanism with rotary positional embeddings.

    Args:
        context_dim (int): Dimension of the context input.
        input_dim (int): Dimension of the input.
        dim (int): Inner Attention dimension.
        num_heads (int): Number of attention heads.
        partial_rotary_factor (float, optional): Factor for partial rotary embeddings. Default is 0.5.
        base (int, optional): Base for rotary positional embeddings. Default is 10000.

    Attributes:
        num_heads (int): Number of attention heads.
        head_dim (int): Dimension of each attention head.
        scale (float): Scaling factor for attention scores.
        qkv_input (nn.Linear): Linear layer for query, key, and value projection of input.
        qkv_context (nn.Linear): Linear layer for query, key, and value projection of context.
        qk_norm_input (QKNorm): Normalization layer for input query and key.
        qk_norm_context (QKNorm): Normalization layer for context query and key.
        partial_rotary_factor (float): Factor for partial rotary embeddings.
        rotary_dim (int): Dimension for rotary embeddings.
        rope (RotaryPositionalEmbedding): Rotary positional embedding layer.
        input_proj_out (nn.Linear): Linear layer for projecting the output of the input.
        context_proj_out (nn.Linear): Linear layer for projecting the output of the context.

    Methods:
        forward(input: Tensor, context: Tensor) -> tuple[Tensor, Tensor]:
            Forward pass of the attention mechanism.

            Args:
                input (Tensor): Input tensor of shape (batch_size, seq_len, input_dim).
                context (Tensor): Context tensor of shape (batch_size, seq_len, context_dim).

            Returns:
                tuple[Tensor, Tensor]: Tuple containing the output tensors for input and context.

    Example:
        >>> mmdit_attention = MMDiTAttention(context_dim=512, input_dim=512, dim=512, num_heads=8)
        >>> input_tensor = torch.randn(10, 25, 512)
        >>> context_tensor = torch.randn(10, 32, 512)
        >>> output_input, output_context = mmdit_attention(input_tensor, context_tensor)
        >>> print(output_input.shape)  # Output: torch.Size([10, 25, 512])
        >>> print(output_context.shape)  # Output: torch.Size([10, 32, 512])
    """

    def __init__(
        self,
        context_dim: int,
        input_dim: int,
        dim: int,
        num_heads: int,
        partial_rotary_factor_input: float = 1,
        base_input: int = 100,
        partial_rotary_factor_context: float = 1,
        base_context: int = 10000,
    ):
        super().__init__()  # type: ignore
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv_input = nn.Linear(input_dim, 3 * dim)
        self.qkv_context = nn.Linear(context_dim, 3 * dim)
        self.qk_norm_input = QKNorm(dim)
        self.qk_norm_context = QKNorm(dim)

        rotary_dim_input = int(self.head_dim * partial_rotary_factor_input)
        self.rope_input = RotaryPositionalEmbedding2D(dim=rotary_dim_input, base=base_input)

        rotary_dim_context = int(self.head_dim * partial_rotary_factor_context)
        self.rope_context = RotaryPositionalEmbedding(dim=rotary_dim_context, base=base_context)

        self.input_proj_out = nn.Linear(dim, input_dim)
        self.context_proj_out = nn.Linear(dim, context_dim)

    def forward(
        self,
        input: Float[Tensor, "batch_size seq_len_input input_dim"],
        context: Float[Tensor, "batch_size seq_len_context context_dim"],
        attn_mask: Bool[Tensor, "batch_size seq_len_context"] | Int[Tensor, "batch_size seq_len_context"] | None = None,
        shape: tuple[int, int] | None = None,
    ) -> tuple[Float[Tensor, "batch_size seq_len input_dim"], Float[Tensor, "batch_size seq_len context_dim"]]:
        input_q, input_k, input_v = self.qkv_input(input).chunk(3, dim=-1)
        context_q, context_k, context_v = self.qkv_context(context).chunk(3, dim=-1)

        input_q, input_k = self.qk_norm_input(input_q, input_k, input_v)
        context_q, context_k = self.qk_norm_context(context_q, context_k, context_v)

        input_q, input_k, input_v = (
            rearrange(input_q, "b n (h d) -> b n h d", h=self.num_heads),
            rearrange(input_k, "b n (h d) -> b n h d", h=self.num_heads),
            rearrange(input_v, "b n (h d) -> b n h d", h=self.num_heads),
        )
        context_q, context_k, context_v = (
            rearrange(context_q, "b n (h d) -> b n h d", h=self.num_heads),
            rearrange(context_k, "b n (h d) -> b n h d", h=self.num_heads),
            rearrange(context_v, "b n (h d) -> b n h d", h=self.num_heads),
        )

        input_q, input_k, input_v = self.rope_input(q=input_q, k=input_k, v=input_v, shape=shape)
        context_q, context_k, context_v = self.rope_context(q=context_q, k=context_k, v=context_v)

        q, k, v = (
            rearrange(torch.cat([context_q, input_q], dim=1), "b n h d -> b h n d"),
            rearrange(torch.cat([context_k, input_k], dim=1), "b n h d -> b h n d"),
            rearrange(torch.cat([context_v, input_v], dim=1), "b n h d -> b h n d"),
        )

        if attn_mask is not None:
            attn_mask = torch.cat(
                [
                    attn_mask.bool(),
                    torch.ones(attn_mask.size(0), input.size(1), device=attn_mask.device).bool(),
                ],
                dim=1,
            )
            b, n = attn_mask.shape
            attn_mask = attn_mask[:, None, None, :].expand(b, 1, n, n)

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

    Methods:
        forward(input: Tensor, y: Tensor, context: Tensor) -> Tuple[Tensor, Tensor]:
            Performs the forward pass of the MMDiTBlock.
            Args:
                input (Tensor): The input tensor.
                y (Tensor): The conditioning tensor used for modulation
            Returns:
                Tuple[Tensor, Tensor]: The processed input tensor.

    Example:
        >>> dit_block = DiTBlock(input_dim=512, hidden_dim=512, embedding_dim=512, num_heads=8, mlp_ratio=4)
        >>> input_tensor = torch.randn(10, 25, 512)
        >>> y = torch.randn(10, 512)
        >>> output_tensor = dit_block(input_tensor, y)
        >>> print(output_tensor.shape)  # Output: torch.Size([10, 25, 512])
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        embedding_dim: int,
        num_heads: int,
        mlp_ratio: int,
        partial_rotary_factor: float = 1,
        base: int = 100,
        use_checkpoint: bool = False,
    ):
        super().__init__()  # type: ignore
        self.modulation = Modulation(embedding_dim, input_dim)
        self.norm_1 = nn.RMSNorm(input_dim)
        self.attention = DiTAttention(
            input_dim, hidden_dim, num_heads, partial_rotary_factor=partial_rotary_factor, base=base
        )
        self.norm_2 = nn.RMSNorm(input_dim)
        self.mlp_input = nn.Sequential(
            nn.Linear(input_dim, mlp_ratio * input_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_ratio * input_dim, input_dim),
        )
        self.use_checkpoint = use_checkpoint

    def forward(
        self,
        input: Float[Tensor, "batch_size seq_len embedding_dim"],
        y: Float[Tensor, "batch_size embedding_dim"],
        shape: tuple[int, int] | None = None,
    ) -> Float[Tensor, "batch_size seq_len input_dim"]:
        """
        Forward pass of the DiTBlock applying modulation and attention mechanisms.
        Args:
            - input (Tensor): The input tensor to be processed
            - y (Tensor): The conditioning tensor used for modulation
        Returns:
            Tensor: The processed input tensor with residual connection
        """
        return (
            checkpoint(self._forward, *(input, y, shape), use_reentrant=False)
            if self.use_checkpoint
            else self._forward(input, y, shape)
        )  # type: ignore

    def _forward(
        self,
        input: Float[Tensor, "batch_size seq_len embedding_dim"],
        y: Float[Tensor, "batch_size embedding_dim"] | Float[Tensor, "batch_size seq_len embedding_dim"],
        shape: tuple[int, int] | None = None,
    ) -> Tensor:
        modulation: ModulationOut = self.modulation(y)

        modulated_input = (
            input
            + self.attention(modulate(self.norm_1(input), scale=modulation.alpha, shift=modulation.beta), shape=shape)
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
        context_dim (int): Dimension of the context tensor.
        input_dim (int): Dimension of the input tensor.
        hidden_dim (int): Dimension of the hidden layer in the attention mechanism.
        embedding_dim (int): Dimension of the embedding used in modulation.
        num_heads (int): Number of attention heads.
        mlp_ratio (int): Ratio used to determine the size of the MLP layers.

    Methods:
        forward(input: Tensor, y: Tensor, context: Tensor) -> Tuple[Tensor, Tensor]:
            Performs the forward pass of the MMDiTBlock.
            Args:
                input (Tensor): The input tensor.
                y (Tensor): An additional tensor, not used in the current implementation.
                context (Tensor): The context tensor.
            Returns:
                Tuple[Tensor, Tensor]: The modulated input and context tensors.

    Example:
        >>> mmdit_block = MMDiTBlock(context_dim=512, input_dim=512, hidden_dim=512, embedding_dim=512, num_heads=8, mlp_ratio=4)
        >>> input_tensor = torch.randn(10, 25, 512)
        >>> y = torch.randn(10, 512)
        >>> context_tensor = torch.randn(10, 32, 512)
        >>> output_input, output_context = mmdit_block(input_tensor, y, context_tensor)
        >>> print(output_input.shape)  # Output: torch.Size([10, 25, 512])
        >>> print(output_context.shape)  # Output: torch.Size([10, 32, 512])
    """

    def __init__(
        self,
        context_dim: int,
        input_dim: int,
        hidden_dim: int,
        embedding_dim: int,
        num_heads: int,
        mlp_ratio: int,
        partial_rotary_factor_input: float = 1,
        base_input: int = 100,
        partial_rotary_factor_context: float = 1,
        base_context: int = 10000,
        use_checkpoint: bool = False,
    ):
        super().__init__()  # type: ignore
        self.modulation_context = Modulation(embedding_dim, context_dim)
        self.modulation_input = Modulation(embedding_dim, input_dim)

        self.context_norm_1 = RMSNorm(context_dim)
        self.input_norm_1 = RMSNorm(input_dim)

        self.attention = MMDiTAttention(
            context_dim,
            input_dim,
            hidden_dim,
            num_heads,
            partial_rotary_factor_input,
            base_input,
            partial_rotary_factor_context,
            base_context,
        )

        self.context_norm_2 = RMSNorm(context_dim)
        self.input_norm_2 = RMSNorm(input_dim)

        self.mlp_context = nn.Sequential(
            nn.Linear(context_dim, mlp_ratio * context_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_ratio * context_dim, context_dim),
        )
        self.mlp_input = nn.Sequential(
            nn.Linear(input_dim, mlp_ratio * input_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_ratio * input_dim, input_dim),
        )
        self.use_checkpoint = use_checkpoint

    def forward(
        self,
        input: Float[Tensor, "batch_size seq_len_input embedding_dim"],
        y: Float[Tensor, "batch_size embedding_dim"],
        context: Float[Tensor, "batch_size seq_len_context context_dim"],
        attn_mask: Bool[Tensor, "batch_size seq_len_context"] | Int[Tensor, "batch_size seq_len_context"] | None = None,
        shape: tuple[int, int] | None = None,
    ) -> tuple[
        Float[Tensor, "batch_size seq_len_input embedding_dim"], Float[Tensor, "batch_size seq_len_context context_dim"]
    ]:
        """
        Forward pass of the MMDiT module applying modulation and attention mechanisms.
        Args:
            - input (Tensor): The input tensor to be processed
            - y (Tensor): The conditioning tensor used for modulation
            - context (Tensor): The context tensor to be processed alongside input
            - attn_mask (Tensor | None): Optional attention mask for context
            - shape (tuple[int, int] | None): Optional shape for rotary embeddings
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
            checkpoint(self._forward, *(input, y, context, attn_mask, shape), use_reentrant=False)
            if self.use_checkpoint
            else self._forward(input, y, context, attn_mask, shape)
        )  # type: ignore

    def _forward(
        self,
        input: Float[Tensor, "batch_size seq_len embedding_dim"],
        y: Float[Tensor, "batch_size embedding_dim"],
        context: Float[Tensor, "batch_size seq_len context_dim"],
        attn_mask: Bool[Tensor, "batch_size seq_len_context"] | Int[Tensor, "batch_size seq_len_context"] | None = None,
        shape: tuple[int, int] | None = None,
    ):
        modulation_input: ModulationOut = self.modulation_input(y)
        modulation_context: ModulationOut = self.modulation_context(y)

        modulated_input = modulate(self.input_norm_1(input), scale=modulation_input.alpha, shift=modulation_input.beta)
        modulated_context = modulate(
            self.context_norm_1(context), scale=modulation_context.alpha, shift=modulation_context.beta
        )

        modulated_input, modulated_context = self.attention(
            modulated_input, modulated_context, attn_mask=attn_mask, shape=shape
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


class ModulatedLastLayer(nn.Module):
    def __init__(self, embedding_dim: int, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()  # type: ignore
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(embedding_dim, 2 * hidden_size, bias=True))

    def forward(self, x: Float[Tensor, "batch_size seq_len dim"], vec: Float[Tensor, "batch_size dim"]) -> Tensor:
        alpha, beta = self.adaLN_modulation(vec).chunk(2, dim=1)
        x = modulate(self.norm_final(x), scale=alpha[:, None, :], shift=beta[:, None, :])
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
        input_dim (int): Token/patch embedding width for the image stream. Also the hidden size used
            before the final per-patch projection. Default: 4096.
        hidden_dim (int): Inner attention dimension for attention projections. Default: 4096.
        embedding_dim (int): Conditioning embedding width (for timestep/labels/pooled context) used
            by modulation layers and the last prediction layer. Default: 4096.
        num_heads (int): Number of attention heads in each block. Default: 16.
        mlp_ratio (int): Expansion ratio for the MLP in each block. Default: 4.
        patch_size (int): Side length P of square patches. Images are projected with stride P. Default: 16.
        depth (int): Number of DiT/MMDiT blocks. Default: 38.
        context_dim (int): Model width for contextual tokens after `context_embed` when
            `simple_dit=False`. Ignored when `simple_dit=True`. Default: 4096.
        partial_rotary_factor (float): Fraction of each head dimension using RoPE.
            1.0 means full rotary. Default: 1.0.
        frequency_embedding (int): Size of the Fourier timestep embedding before the time MLP.
            Default: 256.
        n_classes (int | None): Number of classes for label conditioning in `simple_dit` mode.
            Required to use classifier-free guidance with labels. Must be None when using
            a `context_embedder`. Default: None.
        classifier_free (bool): Enables classifier-free guidance. In `simple_dit`, it applies to
            dropped labels; in MMDiT mode, it is forwarded to the context embedder which may drop
            context. Default: False.
        context_embedder (ContextEmbedder | None): When `simple_dit=False`, a module returning
            `ContextEmbedderOutput`. Must be provided for text conditioning and must be None
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
        input_dim: int = 4096,
        hidden_dim: int = 4096,
        embedding_dim: int = 4096,
        num_heads: int = 16,
        mlp_ratio: int = 4,
        patch_size: int = 16,
        depth: int = 38,
        context_dim: int = 4096,
        partial_rotary_factor_input: float = 1,
        base_input: int = 100,
        partial_rotary_factor_context: float = 1,
        base_context: int = 10000,
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

        self.n_classes = n_classes
        self.classifier_free = classifier_free

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
                    nn.Linear(self.context_embedder.output_size[0], embedding_dim),
                    nn.SiLU(),
                    nn.Linear(embedding_dim, embedding_dim),
                )
                self.context_embed = nn.Linear(self.context_embedder.output_size[1], context_dim)
            else:
                assert self.context_embedder.n_output == 1
                self.context_embed = nn.Linear(self.context_embedder.output_size[0], context_dim)
        else:
            self.label_embed = (
                LabelEmbed(self.n_classes, embedding_dim, self.classifier_free) if self.n_classes is not None else None
            )

        self.last_layer = ModulatedLastLayer(
            embedding_dim=embedding_dim,
            hidden_size=input_dim,
            patch_size=self.patch_size,
            out_channels=self.output_channels,
        )
        self.time_embed = nn.Sequential(
            nn.Linear(self.frequency_embedding, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

        self.conv_proj = nn.Conv2d(self.input_channels, input_dim, kernel_size=self.patch_size, stride=self.patch_size)

        self.layers = nn.ModuleList(
            [
                MMDiTBlock(
                    context_dim=context_dim,
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    partial_rotary_factor_input=partial_rotary_factor_input,
                    base_input=base_input,
                    partial_rotary_factor_context=partial_rotary_factor_context,
                    base_context=base_context,
                    use_checkpoint=use_checkpoint,
                )
                if not self.simple_dit
                else DiTBlock(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    partial_rotary_factor=partial_rotary_factor_input,
                    base=base_input,
                    use_checkpoint=use_checkpoint,
                )
                for _ in range(depth)
            ]
        )

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:  # type: ignore
                nn.init.constant_(module.bias, 0)

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
        context = self.context_embed(context)
        attn_mask = context_output.get("attn_mask", None)

        features: list[Tensor] | None = [] if intermediate_features else None
        # Pass through each layer sequentially
        for layer in self.layers:
            x, context = layer(x, emb, context, attn_mask=attn_mask, shape=self.grid_size)
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

        features: list[Tensor] | None = [] if intermediate_features else None
        # Pass through each layer sequentially
        for layer in self.layers:
            x = layer(x, emb, shape=self.grid_size)
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

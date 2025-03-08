# Recoded from scratch from https://arxiv.org/pdf/2403.03206, if you see any error please report it to the author of the repository

from dataclasses import dataclass

import torch
import torch.nn as nn
from einops import rearrange  # type: ignore
from torch import Tensor

from diffulab.networks.denoisers.common import Denoiser
from diffulab.networks.embedders.common import ContextEmbedder
from diffulab.networks.utils.nn import RotaryPositionalEmbedding, timestep_embedding


class RMSNorm(torch.nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm) module.

    Args:
        dim (int): The dimension of the input tensor to be normalized.

    Attributes:
        scale (torch.nn.Parameter): A learnable scaling parameter of shape (dim,).

    Methods:
        forward(x: Tensor) -> Tensor:
            Applies RMS normalization to the input tensor.

    Example:
        >>> rms_norm = RMSNorm(dim=512)
        >>> input_tensor = torch.randn(10, 512)
        >>> output_tensor = rms_norm(input_tensor)
    """

    def __init__(self, dim: int):
        super().__init__()  # type: ignore
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor):
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
        forward(q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
            Applies RMS normalization to the query and key tensors, and ensures they have the same type as the value tensor.
            Args:
                q (Tensor): The query tensor.
                k (Tensor): The key tensor.
                v (Tensor): The value tensor.
            Returns:
                tuple[Tensor, Tensor]: The normalized query and key tensors, both converted to the type of the value tensor.
    """

    def __init__(self, dim: int):
        super().__init__()  # type: ignore
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)


class MMDiTAttention(nn.Module):
    """
    MMDiTAttention is a multi-head attention mechanism with rotary positional embeddings.

    Args:
        context_dim (int): Dimension of the context input.
        input_dim (int): Dimension of the input.
        dim (int): Dimension of the model.
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
    """

    def __init__(
        self,
        context_dim: int,
        input_dim: int,
        dim: int,
        num_heads: int,
        partial_rotary_factor: float = 0.5,
        base: int = 10000,
    ):
        super().__init__()  # type: ignore
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv_input = nn.Linear(input_dim, 3 * dim)
        self.qkv_context = nn.Linear(context_dim, 3 * dim)
        self.qk_norm_input = QKNorm(dim)
        self.qk_norm_context = QKNorm(dim)

        self.partial_rotary_factor = partial_rotary_factor
        self.rotary_dim = int(self.head_dim * self.partial_rotary_factor)
        self.rope = RotaryPositionalEmbedding(dim=self.head_dim, base=base)

        self.input_proj_out = nn.Linear(dim, input_dim)
        self.context_proj_out = nn.Linear(dim, context_dim)

    def forward(self, input: Tensor, context: Tensor) -> tuple[Tensor, Tensor]:
        input_q, input_k, input_v = self.qkv_input(input).chunk(3, dim=-1)
        context_q, context_k, context_v = self.qkv_context(context).chunk(3, dim=-1)

        input_q, input_k = self.qk_norm_input(input_q, input_k, input_v)
        context_q, context_k = self.qk_norm_context(context_q, context_k, context_v)

        q, k, v = (
            torch.cat([context_q, input_q], dim=1).view(-1, self.num_heads, self.head_dim),
            torch.cat([context_k, input_k], dim=1).view(-1, self.num_heads, self.head_dim),
            torch.cat([context_v, input_v], dim=1).view(-1, self.num_heads, self.head_dim),
        )
        q, k, v = self.rope(q=q, k=k, v=v)
        q, k, v = map(lambda x: rearrange(x, "b n h d -> b n (h d)"), [q, k, v])

        attn_weights = (q @ k.transpose(-2, -1)) * self.scale
        attn_weights = attn_weights.softmax(dim=-1)

        attn_output = attn_weights @ v

        input_output = self.input_proj_out(attn_output[:, context.size(1) :, :])
        context_output = self.context_proj_out(attn_output[:, : context.size(1), :])

        return input_output, context_output


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
    """

    def __init__(self, dim: int):
        super().__init__()  # type: ignore
        self.lin = nn.Linear(dim, 6 * dim, bias=True)

    def forward(self, vec: Tensor) -> ModulationOut:
        out = self.lin(nn.functional.silu(vec))[:, None, :].chunk(6, dim=-1)

        return ModulationOut(*out)


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
    """

    def __init__(
        self, context_dim: int, input_dim: int, hidden_dim: int, embedding_dim: int, num_heads: int, mlp_ratio: int
    ):
        super().__init__()  # type: ignore
        self.modulation_context = Modulation(embedding_dim)
        self.modulation_input = Modulation(embedding_dim)

        self.context_norm_1 = nn.LayerNorm(context_dim)
        self.input_norm_1 = nn.LayerNorm(input_dim)

        self.attention = MMDiTAttention(context_dim, input_dim, hidden_dim, num_heads)

        self.context_norm_2 = nn.LayerNorm(context_dim)
        self.input_norm_2 = nn.LayerNorm(input_dim)

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

    def forward(self, input: Tensor, y: Tensor, context: Tensor):
        modulation_input = self.modulation_input(input)
        modulation_context = self.modulation_context(context)

        modulated_input = (modulation_input.alpha * self.input_norm_1(input)) + modulation_input.beta
        modulated_context = (modulation_context.alpha * self.context_norm_1(context)) + modulation_context.beta

        modulated_input, modulated_context = self.attention(modulated_input, modulated_context)
        modulated_input = input + modulated_input * modulation_input.gamma
        modulated_context = context + modulated_context * modulation_context.gamma

        modulated_input = (modulation_input.delta * self.input_norm_1(modulated_input)) + modulation_input.epsilon
        modulated_context = (
            modulation_context.delta * self.context_norm_1(modulated_context)
        ) + modulation_context.epsilon

        modulated_input = modulation_input.zeta * self.mlp_input(modulated_input)
        modulated_context = modulation_context.zeta * self.mlp_context(modulated_context)

        return modulated_input + input, modulated_context + context


class LastLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()  # type: ignore
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x: Tensor, vec: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        x = self.linear(x)
        return x


class MMDiT(Denoiser):
    """
    architecture following https://arxiv.org/pdf/2403.03206
    """

    def __init__(
        self,
        context_embedder: ContextEmbedder,
        context_dim: int = 4096,
        input_channels: int = 3,
        output_channels: int | None = None,
        input_dim: int = 4096,
        hidden_dim: int = 4096,
        embedding_dim: int = 4096,
        num_heads: int = 16,
        mlp_ratio: int = 4,
        patch_size: int = 16,
        depth: int = 38,
    ):
        assert context_embedder.n_output == 2, "for MMDiT context embedder should provide 2 embeddings"
        assert isinstance(context_embedder.output_size, tuple) and all(
            isinstance(i, int) for i in context_embedder.output_size
        ), "context_embedder.output_size must be a tuple of integers, (embeddings provided should be one dimensional)"

        self.patch_size = patch_size
        self.input_channels = input_channels
        if not output_channels:
            output_channels = input_channels
        self.output_channels = output_channels
        self.context_embedder = context_embedder

        self.pooled_embed = nn.Sequential(
            nn.Linear(context_embedder.output_size[0], embedding_dim),  # type: ignore
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.time_embed = nn.Sequential(
            nn.Linear(self.input_channels, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.context_embed = nn.Linear(context_embedder.output_size[1], context_dim)  # type: ignore
        self.conv_proj = nn.Conv2d(self.input_channels, input_dim, kernel_size=self.patch_size, stride=self.patch_size)

        self.layers = nn.Sequential(*[
            MMDiTBlock(
                context_dim=context_dim,
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
            )
            for _ in range(depth)
        ])

        self.last_layer = LastLayer(
            hidden_size=input_dim, patch_size=self.patch_size, out_channels=self.output_channels
        )

    def patchify(self, x: Tensor) -> Tensor:
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
        x = rearrange(x, "b c h w -> b (h w) c")

        return x

    def unpatchify(self, x: Tensor) -> Tensor:
        """
        Convert patches back into the original image tensor.
        Args:
            x (Tensor): Patchified tensor of shape (B, num_patches, patch_dim)
        Returns:
            Tensor: Reconstructed image tensor of shape (B, C, H, W)
        """

        H, W = self.original_size

        x = rearrange(x, "b (h w) c -> b c h w", h=H // self.patch_size, w=W // self.patch_size)
        x = rearrange(x, "b c (h p1) (w p2) -> b c (h p1) (w p2)", p1=self.patch_size, p2=self.patch_size)

        return x[:, :, :H, :W]

    def forward(
        self,
        x: Tensor,
        timesteps: Tensor,
        context: Tensor | None = None,
        p: float = 0.0,
        x_context: Tensor | None = None,
    ) -> Tensor:
        if x_context is not None:
            x = torch.cat([x, x_context], dim=1)

        emb = self.time_embed(timestep_embedding(timesteps, self.input_channels))

        if context is not None:
            context_pooled, context = self.context_embedder(context, p)
            context_pooled = self.pooled_embed(context_pooled) + emb
            context = self.context_embed(context)
        else:
            context_pooled = emb.clone()
            context = x.clone()

        x = self.patchify(x)
        x, context = self.layers(x, context_pooled, context)

        x = self.last_layer(x, context_pooled)
        x = self.unpatchify(x)
        return x

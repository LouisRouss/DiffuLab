# Recoded from scratch from https://arxiv.org/pdf/2403.03206, if you see any error please report it to the author of the repository

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from einops import rearrange
from jaxtyping import Float, Int
from torch import Tensor

from diffulab.networks.denoisers.common import Denoiser
from diffulab.networks.embedders.common import ContextEmbedder
from diffulab.networks.utils.nn import LabelEmbed, RotaryPositionalEmbedding, timestep_embedding


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
        partial_rotary_factor: float = 0.5,
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
        self.rope = RotaryPositionalEmbedding(dim=self.head_dim, base=base)

        self.proj_out = nn.Linear(dim, input_dim)

    def forward(self, input: Float[Tensor, "batch_size seq_len dim"]) -> Float[Tensor, "batch_size seq_len dim"]:
        input_q, input_k, input_v = self.qkv(input).chunk(3, dim=-1)
        input_q, input_k = self.qk_norm(input_q, input_k, input_v)

        q, k, v = (
            rearrange(input_q, "b n (h d) -> b n h d", h=self.num_heads),
            rearrange(input_k, "b n (h d) -> b n h d", h=self.num_heads),
            rearrange(input_v, "b n (h d) -> b n h d", h=self.num_heads),
        )
        q, k, v = self.rope(q=q, k=k, v=v)
        q, k, v = map(lambda x: rearrange(x, "b n h d -> b n (h d)"), [q, k, v])

        attn_weights: Tensor = (q @ k.transpose(-2, -1)) * self.scale
        attn_weights = attn_weights.softmax(dim=-1)

        attn_output = attn_weights @ v

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

    def forward(
        self,
        input: Float[Tensor, "batch_size seq_len input_dim"],
        context: Float[Tensor, "batch_size seq_len context_dim"],
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

    Example:
        >>> modulation = Modulation(dim=512)
        >>> input_tensor = torch.randn(10, 512)
        >>> output = modulation(input_tensor)
        >>> print(output.alpha.shape)  # Output: torch.Size([10, 512])
    """

    def __init__(self, dim: int):
        super().__init__()  # type: ignore
        self.lin = nn.Linear(dim, 6 * dim, bias=True)

    def forward(self, vec: Float[Tensor, "... dim"]) -> ModulationOut:
        out = self.lin(nn.functional.silu(vec))[:, None, :].chunk(6, dim=-1)

        return ModulationOut(*out)


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
                y (Tensor): An additional tensor, not used in the current implementation.
                context (Tensor): The context tensor.
            Returns:
                Tuple[Tensor, Tensor]: The processed input tensor.

    Example:
        >>> dit_block = DiTBlock(input_dim=512, hidden_dim=512, embedding_dim=512, num_heads=8, mlp_ratio=4)
        >>> input_tensor = torch.randn(10, 25, 512)
        >>> y = torch.randn(10, 512)
        >>> output_tensor = dit_block(input_tensor, y)
        >>> print(output_tensor.shape)  # Output: torch.Size([10, 25, 512])
    """

    def __init__(self, input_dim: int, hidden_dim: int, embedding_dim: int, num_heads: int, mlp_ratio: int):
        super().__init__()  # type: ignore
        self.modulation = Modulation(embedding_dim)
        self.norm_1 = nn.LayerNorm(input_dim)
        self.attention = DiTAttention(input_dim, hidden_dim, num_heads)
        self.input_norm_2 = nn.LayerNorm(input_dim)
        self.mlp_input = nn.Sequential(
            nn.Linear(input_dim, mlp_ratio * input_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_ratio * input_dim, input_dim),
        )

    def forward(
        self, input: Float[Tensor, "batch_size seq_len embedding_dim"], y: Float[Tensor, "batch_size embedding_dim"]
    ) -> Tensor:
        modulation: ModulationOut = self.modulation(y)
        modulated_input = (modulation.alpha * self.norm_1(input)) + modulation.beta

        modulated_input = self.attention(modulated_input)
        modulated_input = input + modulated_input * modulation.gamma

        modulated_input = (modulation.delta * self.input_norm_2(modulated_input)) + modulation.epsilon

        modulated_input = modulation.zeta * self.mlp_input(modulated_input)

        return modulated_input + input


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

    def forward(
        self,
        input: Float[Tensor, "batch_size seq_len embedding_dim"],
        y: Float[Tensor, "batch_size embedding_dim"],
        context: Float[Tensor, "batch_size seq_len context_dim"],
    ):
        """
        Forward pass of the MMDiT module applying modulation and attention mechanisms.
        Args:
            - input (Tensor): The input tensor to be processed
            - y (Tensor): The conditioning tensor used for modulation
            - context (Tensor): The context tensor to be processed alongside input
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
        modulation_input: ModulationOut = self.modulation_input(y)
        modulation_context: ModulationOut = self.modulation_context(y)

        modulated_input = (modulation_input.alpha * self.input_norm_1(input)) + modulation_input.beta
        modulated_context = (modulation_context.alpha * self.context_norm_1(context)) + modulation_context.beta

        modulated_input, modulated_context = self.attention(modulated_input, modulated_context)
        modulated_input = input + modulated_input * modulation_input.gamma
        modulated_context = context + modulated_context * modulation_context.gamma

        modulated_input = (modulation_input.delta * self.input_norm_2(modulated_input)) + modulation_input.epsilon
        modulated_context = (
            modulation_context.delta * self.context_norm_2(modulated_context)
        ) + modulation_context.epsilon

        modulated_input = modulation_input.zeta * self.mlp_input(modulated_input)
        modulated_context = modulation_context.zeta * self.mlp_context(modulated_context)

        return modulated_input + input, modulated_context + context


class ModulatedLastLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()  # type: ignore
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x: Float[Tensor, "batch_size seq_len dim"], vec: Float[Tensor, "batch_size dim"]) -> Tensor:
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
        n_classes: int | None = None,
        classifier_free: bool = False,
        context_embedder: ContextEmbedder | None = None,
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

        self.n_classes = n_classes
        self.classifier_free = classifier_free

        if not self.simple_dit:
            assert self.context_embedder is not None, "for MMDiT context embedder must be provided"
            assert self.context_embedder.n_output == 2, "for MMDiT context embedder should provide 2 embeddings"
            assert isinstance(self.context_embedder.output_size, tuple) and all(
                isinstance(i, int) for i in self.context_embedder.output_size
            ), (
                "context_embedder.output_size must be a tuple of integers, (embeddings provided should be one dimensional)"
            )
            self.mlp_pooled_context = nn.Sequential(
                nn.Linear(self.context_embedder.output_size[0], embedding_dim),
                nn.SiLU(),
                nn.Linear(embedding_dim, embedding_dim),
            )
            self.context_embed = nn.Linear(self.context_embedder.output_size[1], context_dim)

            self.last_layer = ModulatedLastLayer(
                hidden_size=input_dim, patch_size=self.patch_size, out_channels=self.output_channels
            )
        else:
            self.label_embed = (
                LabelEmbed(self.n_classes, embedding_dim, self.classifier_free) if self.n_classes is not None else None
            )
            self.last_layer = nn.Sequential(
                nn.LayerNorm(input_dim, elementwise_affine=False, eps=1e-6),
                nn.Linear(input_dim, self.patch_size * self.patch_size * self.output_channels, bias=True),
            )

        self.time_embed = nn.Sequential(
            nn.Linear(self.input_channels, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

        self.conv_proj = nn.Conv2d(self.input_channels, input_dim, kernel_size=self.patch_size, stride=self.patch_size)

        # Fix the ModuleList initialization - it was taking *[] which is incorrect
        self.layers = nn.ModuleList([
            MMDiTBlock(
                context_dim=context_dim,
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
            )
            if not self.simple_dit
            else DiTBlock(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
            )
            for _ in range(depth)
        ])

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
        H, W = self.original_size
        patch_size = self.patch_size
        p = self.output_channels

        # Calculate number of patches in height and width dimensions
        h = H // patch_size
        w = W // patch_size

        # Reshape the tensor to the original image dimensions
        x = rearrange(x, "b (h w) (p1 p2 c) -> b c (h p1) (w p2)", h=h, w=w, p1=patch_size, p2=patch_size, c=p)
        return x

    def mmdit_forward(
        self,
        x: Float[Tensor, "batch_size channels height width"],
        timesteps: Float[Tensor, "batch_size"],
        initial_context: Any | None = None,
        p: float = 0.0,
    ) -> Tensor:
        assert self.context_embedder is not None, "for MMDiT context embedder must be provided"
        x = self.patchify(x)
        emb = self.time_embed(timestep_embedding(timesteps, self.input_channels))
        context_pooled, context = self.context_embedder(initial_context, p)
        context_pooled = self.mlp_pooled_context(context_pooled) + emb
        context = self.context_embed(context)
        # Pass through each layer sequentially
        for layer in self.layers:
            x, context = layer(x, context_pooled, context)

        x = self.last_layer(x, context_pooled)
        x = self.unpatchify(x)
        return x

    def simple_dit_forward(
        self,
        x: Float[Tensor, "batch_size channels height width"],
        timestep: Float[Tensor, "batch_size"],
        p: float = 0.0,
        y: Int[Tensor, "batch_size"] | None = None,
    ):
        if p > 0:
            assert self.n_classes, (
                "probability of dropping for classifier free guidance is only available if a number of classes is set"
            )
        x = self.patchify(x)

        emb = self.time_embed(timestep_embedding(timestep, self.input_channels))
        if self.label_embed is not None:
            emb = emb + self.label_embed(y, p)

        # Pass through each layer sequentially
        for layer in self.layers:
            x = layer(x, emb)

        x = self.last_layer(x)

        x = self.unpatchify(x)
        return x

    def forward(
        self,
        x: Float[Tensor, "batch_size channels height width"],
        timesteps: Float[Tensor, "batch_size"],
        initial_context: Any | None = None,
        p: float = 0.0,
        y: Int[Tensor, "batch_size"] | None = None,
        x_context: Tensor | None = None,
    ) -> Tensor:
        assert not (initial_context is not None and y is not None), "initial_context and y cannot both be specified"
        if p > 0:
            assert self.classifier_free, (
                "probability of dropping for classifier free guidance is only available if model is set up to be classifier free"
            )
        if x_context is not None:
            x = torch.cat([x, x_context], dim=1)
        if self.simple_dit:
            return self.simple_dit_forward(x, timesteps, p, y)
        else:
            return self.mmdit_forward(x, timesteps, initial_context, p)

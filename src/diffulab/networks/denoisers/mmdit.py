# Recoded from scratch from https://arxiv.org/pdf/2403.03206, if you see any error please report it to the author of the repository

from typing import Any

import torch
import torch.nn as nn
from einops import rearrange
from jaxtyping import Float, Int
from torch import Tensor

from diffulab.networks.denoisers.common import Denoiser
from diffulab.networks.embedders.common import ContextEmbedder
from diffulab.networks.utils.nn import (
    LabelEmbed,
    Modulation,
    ModulationOut,
    QKNorm,
    RMSNorm,
    RotaryPositionalEmbedding,
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
        self.rope = RotaryPositionalEmbedding(dim=self.rotary_dim, base=base)

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
        q, k, v = map(lambda x: rearrange(x, "b n h d -> b h n d"), [q, k, v])

        attn_weights: Tensor = (q @ k.transpose(-2, -1)) * self.scale
        attn_weights = attn_weights.softmax(dim=-1)
        attn_output = attn_weights @ v

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
        partial_rotary_factor: float = 1,
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
        self.rope = RotaryPositionalEmbedding(dim=self.rotary_dim, base=base)

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
        q, k, v = map(lambda x: rearrange(x, "b n h d -> b h n d"), [q, k, v])

        attn_weights = (q @ k.transpose(-2, -1)) * self.scale
        attn_weights = attn_weights.softmax(dim=-1)
        attn_output = attn_weights @ v

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

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        embedding_dim: int,
        num_heads: int,
        mlp_ratio: int,
        partial_rotary_factor: float = 1,
    ):
        super().__init__()  # type: ignore
        self.modulation = Modulation(embedding_dim)
        self.norm_1 = nn.RMSNorm(input_dim)
        self.attention = DiTAttention(input_dim, hidden_dim, num_heads)
        self.norm_2 = nn.RMSNorm(input_dim)
        self.mlp_input = nn.Sequential(
            nn.Linear(input_dim, mlp_ratio * input_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_ratio * input_dim, input_dim),
        )

    def forward(
        self, input: Float[Tensor, "batch_size seq_len embedding_dim"], y: Float[Tensor, "batch_size embedding_dim"]
    ) -> Tensor:
        modulation: ModulationOut = self.modulation(y)

        modulated_input = (
            input
            + self.attention(modulate(self.norm_1(input), scale=modulation.alpha, shift=modulation.beta))
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
        self, context_dim: int, input_dim: int, hidden_dim: int, embedding_dim: int, num_heads: int, mlp_ratio: int
    ):
        super().__init__()  # type: ignore
        self.modulation_context = Modulation(embedding_dim)
        self.modulation_input = Modulation(embedding_dim)

        self.context_norm_1 = RMSNorm(context_dim)
        self.input_norm_1 = RMSNorm(input_dim)

        self.attention = MMDiTAttention(context_dim, input_dim, hidden_dim, num_heads)

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

        modulated_input = modulate(self.input_norm_1(input), scale=modulation_input.alpha, shift=modulation_input.beta)
        modulated_context = modulate(
            self.context_norm_1(context), scale=modulation_context.alpha, shift=modulation_context.beta
        )

        modulated_input, modulated_context = self.attention(modulated_input, modulated_context)
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
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()  # type: ignore
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x: Float[Tensor, "batch_size seq_len dim"], vec: Float[Tensor, "batch_size dim"]) -> Tensor:
        alpha, beta = self.adaLN_modulation(vec).chunk(2, dim=1)
        x = modulate(self.norm_final(x), scale=alpha[:, None, :], shift=beta[:, None, :])
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
            ), "context_embedder.output_size must be a tuple of integers"
            self.mlp_pooled_context = nn.Sequential(
                nn.Linear(self.context_embedder.output_size[0], embedding_dim),
                nn.SiLU(),
                nn.Linear(embedding_dim, embedding_dim),
            )
            self.context_embed = nn.Linear(self.context_embedder.output_size[1], context_dim)
        else:
            self.label_embed = (
                LabelEmbed(self.n_classes, embedding_dim, self.classifier_free) if self.n_classes is not None else None
            )

        self.last_layer = ModulatedLastLayer(
            hidden_size=input_dim, patch_size=self.patch_size, out_channels=self.output_channels
        )
        self.input_dim = input_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.input_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

        self.conv_proj = nn.Conv2d(self.input_channels, input_dim, kernel_size=self.patch_size, stride=self.patch_size)

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
        x: Float[Tensor, "batch_size seq_len patch_dim"],
        timesteps: Float[Tensor, "batch_size"],
        initial_context: Any | None = None,
        p: float = 0.0,
    ) -> Tensor:
        assert self.context_embedder is not None, "for MMDiT context embedder must be provided"
        emb = self.time_embed(timestep_embedding(timesteps, self.input_dim))
        context_pooled, context = self.context_embedder(initial_context, p)
        context_pooled = self.mlp_pooled_context(context_pooled) + emb
        context = self.context_embed(context)
        # Pass through each layer sequentially
        for layer in self.layers:
            x, context = layer(x, context_pooled, context)

        x = self.last_layer(x, context_pooled)
        return x

    def simple_dit_forward(
        self,
        x: Float[Tensor, "batch_size seq_len patch_dim"],
        timestep: Float[Tensor, "batch_size"],
        p: float = 0.0,
        y: Int[Tensor, "batch_size"] | None = None,
    ) -> Tensor:
        if p > 0:
            assert self.n_classes, (
                "probability of dropping for classifier free guidance is only available if a number of classes is set"
            )

        emb = self.time_embed(timestep_embedding(timestep, self.input_dim))
        if self.label_embed is not None:
            emb = emb + self.label_embed(y, p)

        # Pass through each layer sequentially
        for layer in self.layers:
            x = layer(x, emb)

        x = self.last_layer(x, emb)
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

        x = self.patchify(x)
        if self.simple_dit:
            x = self.simple_dit_forward(x, timesteps, p, y)
        else:
            x = self.mmdit_forward(x, timesteps, initial_context, p)
        x = self.unpatchify(x)
        return x

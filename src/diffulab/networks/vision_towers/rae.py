import torch
import torch.nn as nn
from einops import rearrange
from jaxtyping import Float
from torch import Tensor
from torch.utils.checkpoint import checkpoint  # type: ignore

from diffulab.networks.utils.nn import QKNorm, RotaryPositionalEmbedding


class Attention(nn.Module):
    """
    Multi-head self-attention mechanism with rotary positional embeddings.

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
    """

    def __init__(
        self,
        input_dim: int,
        dim: int,
        num_heads: int,
        partial_rotary_factor: float = 1,
        base: int = 10000,
        dropout: float = 0.0,
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
        self.dropout = dropout

        self.proj_out = nn.Linear(dim, input_dim)

    def forward(self, input: Float[Tensor, "batch_size seq_len dim"]) -> Float[Tensor, "batch_size seq_len dim"]:
        input_q, input_k, input_v = self.qkv(input).chunk(3, dim=-1)
        input_q, input_k = self.qk_norm(input_q, input_k, input_v)

        q, k, v = (
            rearrange(input_q, "b n (h d) -> b n h d", h=self.num_heads),
            rearrange(input_k, "b n (h d) -> b n h d", h=self.num_heads),
            rearrange(input_v, "b n (h d) -> b n h d", h=self.num_heads),
        )
        # Remove CLS token for rotary embeddings
        q[:, 1:], k[:, 1:], v = self.rope(q=q[:, 1:], k=k[:, 1:], v=v)
        q, k, v = map(lambda x: rearrange(x, "b n h d -> b h n d"), [q, k, v])

        attn_output = nn.functional.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            scale=self.scale,
            dropout_p=self.dropout,
        )

        attn_output = rearrange(attn_output, "b h n d -> b n (h d)")

        output: Tensor = self.proj_out(attn_output)

        return output


class TransformerBlock(nn.Module):
    """
    Transformer block with attention mechanisms.

    Args:
        input_dim (int): Dimension of the input tensor.
        hidden_dim (int): Dimension of the hidden layer in the attention mechanism.
        num_heads (int): Number of attention heads.
        mlp_ratio (int): Ratio used to determine the size of the MLP layers.

    Methods:
        forward(input: Tensor, y: Tensor, context: Tensor) -> Tuple[Tensor, Tensor]:
            Performs the forward pass of the MMDiTBlock.
            Args:
                input (Tensor): The input tensor.
                context (Tensor): The context tensor.
            Returns:
                Tuple[Tensor, Tensor]: The processed input tensor.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: int,
        partial_rotary_factor: float = 1,
        use_checkpoint: bool = False,
        dropout_attn: float = 0,
        dropout_mlp: float = 0,
    ):
        super().__init__()  # type: ignore
        self.norm_1 = nn.RMSNorm(input_dim)
        self.attention = Attention(
            input_dim, hidden_dim, num_heads, partial_rotary_factor=partial_rotary_factor, dropout=dropout_attn
        )
        self.norm_2 = nn.RMSNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, mlp_ratio * input_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_ratio * input_dim, input_dim),
        )
        self.use_checkpoint = use_checkpoint
        self.dropout = nn.Dropout(dropout_mlp)

    def forward(
        self,
        input: Float[Tensor, "batch_size seq_len embedding_dim"],
    ) -> Float[Tensor, "batch_size seq_len input_dim"]:
        """
        Forward pass of the DiTBlock applying modulation and attention mechanisms.
        Args:
            - input (Tensor): The input tensor to be processed
            - y (Tensor): The conditioning tensor used for modulation
        Returns:
            Tensor: The processed input tensor with residual connection
        """
        return checkpoint(self._forward, *(input), use_reentrant=False) if self.use_checkpoint else self._forward(input)  # type: ignore

    def _forward(
        self,
        input: Float[Tensor, "batch_size seq_len embedding_dim"],
    ) -> Float[Tensor, "batch_size seq_len embedding_dim"]:
        input = input + self.attention(self.norm_1(input))
        input = input + self.dropout(self.mlp(self.norm_2(input)))
        return input


class RAEDecoder(nn.Module):
    """
    ViT decoder for SSL based encoders. Uses MAE logic during training.
    """

    def __init__(
        self,
        out_size: tuple[int, int] = (256, 256),
        out_channels: int = 3,
        encoder_dim: int = 768,
        input_dim: int = 1152,
        hidden_dim: int = 1152,
        num_heads: int = 16,
        mlp_ratio: int = 4,
        patch_size: int = 16,
        depth: int = 28,
        partial_rotary_factor: float = 1,
        use_checkpoint: bool = False,
        dropout_attn: float = 0,
        dropout_mlp: float = 0,
    ):
        super().__init__()  # type: ignore
        self.input_proj = nn.Linear(encoder_dim, input_dim)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    partial_rotary_factor=partial_rotary_factor,
                    use_checkpoint=use_checkpoint,
                    dropout_attn=dropout_attn,
                    dropout_mlp=dropout_mlp,
                )
                for _ in range(depth)
            ]
        )

        self.last_layer = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, patch_size * patch_size * out_channels),
        )
        self.apply(self._init_weights)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, input_dim))
        self.patch_size = patch_size
        self.out_size = out_size
        self.out_channels = out_channels

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:  # type: ignore
                nn.init.constant_(module.bias, 0)

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
        H, W = self.out_size
        patch_size = self.patch_size
        p = self.out_channels

        # Calculate number of patches in height and width dimensions
        h = H // patch_size
        w = W // patch_size

        # Reshape the tensor to the original image dimensions
        x = rearrange(x, "b (h w) (p1 p2 c) -> b c (h p1) (w p2)", h=h, w=w, p1=patch_size, p2=patch_size, c=p)
        return x

    def forward(
        self,
        x: Float[Tensor, "batch_size seq_len encoder_dim"],
    ) -> Float[Tensor, "batch_size channels height width"]:
        x = self.input_proj(x)
        x = torch.cat([self.cls_token.repeat(x.size(0), 1, 1), x], dim=1)

        for layer in self.layers:
            x = layer(x)

        x = self.last_layer(x)
        x = self.unpatchify(x)
        return x

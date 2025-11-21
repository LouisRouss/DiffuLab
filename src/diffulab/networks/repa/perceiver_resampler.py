# This implementation has been adapted from https://github.com/lucidrains/flamingo-pytorch/tree/main
# Under MIT license

from typing import cast

import torch
from einops import rearrange, repeat
from jaxtyping import Float
from torch import Tensor, einsum, nn

from diffulab.networks.utils.nn import RotaryPositionalEmbedding2D


class PerceiverRotaryPositionalEmbedding(RotaryPositionalEmbedding2D):
    """Rotary positional embedding applied only to keys.

    This subclass restricts rotary application to the key tensor (``k``) while
    passing queries and values through unchanged. It caches precomputed cosine
    and sine tables via the parent implementation.

    Args:
        dim (int): Rotary embedding dimension (applied to the leading slice of the key head dimension).
        base (int): Base for rotary frequency computation.
    """

    def __init__(self, dim: int = 32, base: int = 10_000) -> None:
        super().__init__(dim, base)  # type: ignore

    def forward(
        self,
        q: Float[Tensor, "batch_size seq_len n_heads head_dim"],
        k: Float[Tensor, "batch_size seq_len n_heads head_dim"],
        v: Float[Tensor, "batch_size seq_len n_heads head_dim"],
        shape: tuple[int, int] | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Apply rotary embedding to the key tensor only.

        Args:
            q (Tensor): Query tensor of shape ``[B, N, H, D]``.
            k (Tensor): Key tensor of shape ``[B, N, H, D]``.
            v (Tensor): Value tensor of shape ``[B, N, H, D]``.

        Returns:
            tuple: ``(q, k_rot, v)`` with identical shapes where ``k_rot`` has
            the first ``dim`` key channels rotated.
        """
        seq_len = k.shape[1]
        if shape is None:
            shape = (int(seq_len**0.5), int(seq_len**0.5))
        height, width = shape
        assert height * width == seq_len, "Sequence length does not match provided shape."
        self._cache(height, width)

        cos = self.cos.to(device=q.device, dtype=q.dtype)  # [S, dim]
        sin = self.sin.to(device=q.device, dtype=q.dtype)  # [S, dim]

        # [batch_size, seq_length, num_heads, head_dim] -> [batch_size, num_heads, seq_length, head_dim]
        k = k.transpose(1, 2)

        # K rotation
        k_rope, k_pass = k[..., : self.dim], k[..., self.dim :]
        k_rope = self._apply_rotary(k_rope, cos, sin)
        k_rot = torch.cat((k_rope, k_pass), dim=-1)

        # [batch_size, num_heads, seq_length, head_dim] -> [batch_size, seq_length, num_heads, head_dim]
        k_rot = k_rot.transpose(1, 2)

        return q, k_rot, v


def FeedForward(dim: int, mult: float = 4) -> nn.Sequential:
    """Feed-forward MLP block with GELU activation.

    Structure: LayerNorm -> Linear(dim, dim*mult) -> GELU -> Linear(dim*mult, dim).

    Args:
        dim (int): Input / output embedding dimension.
        mult (int): Width multiplier for the hidden layer.

    Returns:
        nn.Sequential: Configured feed-forward network.
    """
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim), nn.Linear(dim, inner_dim, bias=False), nn.GELU(), nn.Linear(inner_dim, dim, bias=False)
    )


class PerceiverAttention(nn.Module):
    """Cross-attention layer between inputs and a set of latent tokens.

    Queries are derived from latent tokens; keys/values are concatenation of
    (projected) input tokens and latent tokens (self-attention augmentation).

    Rotary positional embedding (optionally partial) is applied only to the key
    originating from the input tokens.

    Args:
        dim (int): Embedding dimension of inputs and latents.
        head_dim (int): Dimension per attention head.
        num_heads (int): Number of attention heads.
        partial_rotary_factor (float): Fraction (0..1] of ``head_dim`` receiving rotary embedding.
    """

    def __init__(self, dim: int, head_dim: int = 64, num_heads: int = 8, partial_rotary_factor: float = 1) -> None:
        super().__init__()  # type: ignore[reportUnknownMemberType]
        self.scale = head_dim**-0.5
        self.num_heads = num_heads
        inner_dim = head_dim * num_heads

        self.norm_x = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

        rotary_dim = int(head_dim * partial_rotary_factor)
        self.rope = PerceiverRotaryPositionalEmbedding(dim=rotary_dim)

    def forward(
        self, x: Float[Tensor, "batch n dim"], latents: Float[Tensor, "batch m dim"]
    ) -> Float[Tensor, "batch m dim"]:
        """
        Perform Perceiver cross/self attention update on latent tokens.

        Args:
            x (Tensor): Input sequence embeddings of shape ``[B, N, D]``.
            latents (Tensor): Current latent tokens of shape ``[B, M, D]``.

        Returns:
            Tensor: Updated latent tokens of shape ``[B, M, D]``.
        """
        x = self.norm_x(x)
        latents = self.norm_latents(latents)

        q = self.to_q(latents)
        k_x, v_x = self.to_kv(x).chunk(2, dim=-1)
        k_latent, v_latent = self.to_kv(latents).chunk(2, dim=-1)

        q, k_x, v_x = (
            rearrange(q, "b n (h d) -> b n h d", h=self.num_heads),
            rearrange(k_x, "b n (h d) -> b n h d", h=self.num_heads),
            rearrange(v_x, "b n (h d) -> b n h d", h=self.num_heads),
        )
        q, k_x, v_x = self.rope(q=q, k=k_x, v=v_x)
        q, k_x, v_x = map(lambda x: rearrange(x, "b n h d -> b h n d"), [q, k_x, v_x])

        k_latent, v_latent = (
            rearrange(k_latent, "b m (h d) -> b h m d", h=self.num_heads),
            rearrange(v_latent, "b m (h d) -> b h m d", h=self.num_heads),
        )

        k = torch.cat((k_x, k_latent), dim=2)  # [b h n+m d]
        v = torch.cat((v_x, v_latent), dim=2)

        q = q * self.scale

        # attention
        sim = einsum("... i d, ... j d  -> ... i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("... i j, ... j d -> ... i d", attn, v)
        out = rearrange(out, "b h m d -> b m (h d)")
        return self.to_out(out)


class PerceiverResampler(nn.Module):
    """
    Stack of Perceiver attention + feed-forward layers producing latents.

    A learned set of ``num_latents`` tokens is iteratively refined via cross-
    attention with the input sequence followed by a feed-forward block (each
    with residual connections). The final normalized latent set is returned.

    Args:
        dim (int): Embedding dimension for inputs and latents.
        depth (int): Number of attention + feed-forward layers.
        head_dim (int): Dimension per attention head.
        num_heads (int): Number of attention heads.
        ff_mult (int): Multiplier for hidden dimension in feed-forward layers.
        num_latents (int): Number of learned latent tokens to maintain.
    """

    def __init__(
        self, dim: int, depth: int, head_dim: int = 64, num_heads: int = 8, ff_mult: int = 4, num_latents: int = 16
    ):
        super().__init__()  # type: ignore[reportUnknownMemberType]
        self.latents = nn.Parameter(torch.randn(num_latents, dim))

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, head_dim=head_dim, num_heads=num_heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Float[Tensor, "batch n dim"]) -> Float[Tensor, "batch m dim"]:
        """Encode an input sequence into a fixed set of latent tokens.

        Args:
            x: Input embeddings of shape ``[B, N, D]``.

        Returns:
            Tensor: Latent tokens of shape ``[B, M, D]`` where ``M = num_latents``.
        """
        latents = repeat(self.latents, "n d -> b n d", b=x.shape[0])

        for attn, ff in self.layers:  # type: ignore
            latents = cast(Tensor, attn(x, latents) + latents)
            latents = cast(Tensor, ff(latents) + latents)

        return self.norm(latents)

# This implementation has been adapted from https://github.com/lucidrains/flamingo-pytorch/tree/main
# Under MIT license

from typing import Any, cast

import torch
from einops import rearrange, repeat
from jaxtyping import Float
from torch import Tensor, einsum, nn

from diffulab.networks.utils.nn import RotaryPositionalEmbedding


def exists(val: Any) -> bool:
    """
    Check if an object is not None.
    """
    return val is not None


# ROPE only on k
class PerceiverRotaryPositionalEmbedding(RotaryPositionalEmbedding):
    def __init__(self, dim: int = 32, base: int = 10_000) -> None:
        super().__init__(dim, base)  # type: ignore

    def forward(
        self,
        q: Float[Tensor, "batch_size seq_len n_heads head_dim"],
        k: Float[Tensor, "batch_size seq_len n_heads head_dim"],
        v: Float[Tensor, "batch_size seq_len n_heads head_dim"],
    ) -> tuple[
        Float[Tensor, "batch_size seq_len n_heads head_dim"],
        Float[Tensor, "batch_size seq_len n_heads head_dim"],
        Float[Tensor, "batch_size seq_len n_heads head_dim"],
    ]:
        seq_len = k.shape[1]
        self._cache(seq_len)
        cos = self.cos.to(device=k.device, dtype=k.dtype)
        sin = self.sin.to(device=k.device, dtype=k.dtype)

        # [batch_size, seq_length, num_heads, head_dim] -> [batch_size, num_heads, seq_length, head_dim]
        k = k.transpose(1, 2)

        # K rotation
        k_rope, k_pass = k[..., : self.dim], k[..., self.dim :]
        k_neg_half = self._neg_half(k_rope)
        k_rope = (k_rope * cos[:seq_len]) + (k_neg_half * sin[:seq_len])
        k_rot = torch.cat((k_rope, k_pass), dim=-1)

        # [batch_size, num_heads, seq_length, head_dim] -> [batch_size, seq_length, num_heads, head_dim]
        k_rot = k_rot.transpose(1, 2)

        return q, k_rot, v


def FeedForward(dim: int, mult: float = 4) -> nn.Sequential:
    """
    A simple feed-forward module with a GELU activation function.
    Args:
        dim (int): Input and output dimension of the network.
        mult (float): Multiplier for the inner dimension.
    Returns:
        nn.Sequential: A sequential model containing LayerNorm, Linear, GELU, and another Linear layer.
    """
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim), nn.Linear(dim, inner_dim, bias=False), nn.GELU(), nn.Linear(inner_dim, dim, bias=False)
    )


class PerceiverAttention(nn.Module):
    def __init__(self, dim: int, head_dim: int = 64, num_heads: int = 8, partial_rotary_factor: int = 1) -> None:
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
        latents = repeat(self.latents, "n d -> b n d", b=x.shape[0])

        for attn, ff in self.layers:  # type: ignore
            latents = cast(Tensor, attn(x, latents) + latents)
            latents = cast(Tensor, ff(latents) + latents)

        return self.norm(latents)

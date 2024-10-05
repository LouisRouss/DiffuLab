from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()  # type: ignore
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        x_dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)
        return (x * rrms).to(dtype=x_dtype) * self.scale


class QKNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()  # type: ignore
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)


class MMDiTAttention(nn.Module):
    def __init__(self, context_dim: int, input_dim: int, dim: int, num_heads: int):
        super().__init__()  # type: ignore
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv_context = nn.Linear(context_dim, 3 * dim)
        self.qkv_input = nn.Linear(input_dim, 3 * dim)

        self.qk_norm_context = QKNorm(dim)
        self.qk_norm_input = QKNorm(dim)

        self.context_proj_out = nn.Linear(dim, context_dim)
        self.input_proj_out = nn.Linear(dim, input_dim)

    def forward(self, input: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        input_q, input_k, input_v = self.qkv_input(input).chunk(3, dim=-1)
        context_q, context_k, context_v = self.qkv_context(context).chunk(3, dim=-1)

        input_q, input_k = self.qk_norm_input(input_q, input_k, input_v)
        context_q, context_k = self.qk_norm_context(context_q, context_k, context_v)

        q, k, v = (
            torch.cat([context_q, input_q], dim=1),
            torch.cat([context_k, input_k], dim=1),
            torch.cat([context_v, input_v], dim=1),
        )

        attn_weights = (q @ k.transpose(-2, -1)) * self.scale
        attn_weights = attn_weights.softmax(dim=-1)

        attn_output = attn_weights @ v

        input_output = self.input_proj_out(attn_output[:, context.size(1) :, :])
        context_output = self.context_proj_out(attn_output[:, : context.size(1), :])

        return input_output, context_output


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: int, activation: Callable[..., nn.Module] = nn.GELU):
        super().__init__()  # type: ignore
        self.fc1 = nn.Linear(dim, mlp_ratio * dim)
        self.fc2 = nn.Linear(mlp_ratio * dim, dim)
        self.silu = activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.silu(self.fc1(x)))


@dataclass
class ModulationOut:
    alpha: torch.Tensor
    beta: torch.Tensor
    gamma: torch.Tensor
    delta: torch.Tensor
    epsilon: torch.Tensor
    zeta: torch.Tensor


class Modulation(nn.Module):
    def __init__(self, dim: int):
        super().__init__()  # type: ignore
        self.lin = nn.Linear(dim, 6 * dim, bias=True)

    def forward(self, vec: torch.Tensor) -> ModulationOut:
        out = self.lin(nn.functional.silu(vec))[:, None, :].chunk(6, dim=-1)

        return ModulationOut(*out)


class MMDiTBlock(nn.Module):
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

        self.mlp_context = MLP(context_dim, mlp_ratio)
        self.mlp_input = MLP(input_dim, mlp_ratio)

    def forward(self, input: torch.Tensor, y: torch.Tensor, context: torch.Tensor):
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

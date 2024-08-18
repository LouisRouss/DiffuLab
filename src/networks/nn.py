import math
from abc import abstractmethod
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def exists(val: Any) -> bool:
    return val is not None


def default(val: Any, d: Any) -> Any:
    return (val) if exists(val) else (d)


def timestep_embedding(timesteps: Tensor, dim: int, max_period: int = 10000) -> Tensor:
    """
    From https://github.com/openai/guided-diffusion under MIT license as of 2024-18-08

    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
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


def checkpoint(func: Callable[..., Tensor], inputs: Any, params: Any, flag: bool) -> Any:
    """
    From https://github.com/openai/guided-diffusion under MIT license as of 2024-18-08

    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.
    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)  # type: ignore
    else:
        return func(*inputs)


class CheckpointFunction(torch.autograd.Function):
    """
    From https://github.com/openai/guided-diffusion under MIT license as of 2024-18-08
    """

    @staticmethod
    def forward(
        ctx: Any,
        run_function: Callable[..., tuple[Tensor, ...]],
        length: int,
        *args: Tensor,
    ) -> tuple[Tensor, ...]:
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx: Any, *output_grads: Tensor):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with torch.enable_grad():
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads


class GroupNorm32(nn.GroupNorm):
    def forward(self, input: Tensor) -> Tensor:
        return super().forward(input.float()).type(input.dtype)


def normalization(channels: int) -> GroupNorm32:
    """
    Make a standard normalization layer.
    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(32, channels)


def zero_module(module: nn.Module) -> nn.Module:
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x: Tensor, emb: Tensor) -> Tensor:
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class ContextBlock(nn.Module):
    """
    Any module where forward() takes context embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x: Tensor, context: Tensor) -> Tensor:
        """
        Apply the module to `x` given `context` embeddings.
        """


class EmbedSequential(nn.Sequential, TimestepBlock, ContextBlock):  # type: ignore
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x: Tensor, emb: Tensor, context: Tensor | None = None) -> Tensor:  # type:ignore
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, ContextBlock):
                x = layer(x, context)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    From https://github.com/openai/guided-diffusion under MIT license as of 2024-18-08

    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.

    """

    def __init__(self, channels: int, use_conv: bool, out_channels: int | None = None) -> None:
        super().__init__()  # type:ignore
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv2d(self.channels, self.out_channels, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[1] == self.channels
        x = F.interpolate(x, scale_factor=2, mode="nearest")  # type: ignore
        if self.use_conv:
            x = self.conv(x)
        return x  # type: ignore


class Downsample(nn.Module):
    """
    From https://github.com/openai/guided-diffusion under MIT license as of 2024-18-08

    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    """

    def __init__(self, channels: int, use_conv: bool, out_channels: int | None = None) -> None:
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

    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    From https://github.com/openai/guided-diffusion under MIT license as of 2024-18-08

    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels: int,
        emb_channels: int,
        dropout: float,
        out_channels: int | None = None,
        use_conv: bool = False,
        use_scale_shift_norm: bool = False,
        use_checkpoint: bool = False,
        up: bool = False,
        down: bool = False,
    ) -> None:
        super().__init__()  # type:ignore
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            nn.Conv2d(channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False)
            self.x_upd = Upsample(channels, False)
        elif down:
            self.h_upd = Downsample(channels, False)
            self.x_upd = Downsample(channels, False)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = nn.Conv2d(channels, self.out_channels, 3, padding=1)
        else:
            self.skip_connection = nn.Conv2d(channels, self.out_channels, 1)

    def forward(self, x: Tensor, emb: Tensor) -> Tensor:
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(self._forward, (x, emb), self.parameters(), self.use_checkpoint)

    def _forward(self, x: Tensor, emb: Tensor) -> Tensor:
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class AttentionBlock(ContextBlock):
    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        inner_channels: int = -1,
        dropout: float = 0.0,
        norm: Callable[..., nn.Module] = normalization,
        use_checkpoint: bool = False,
    ):
        super().__init__()  # type: ignore

        self.use_checkpoint = use_checkpoint
        self.channels = channels
        self.inner_channels = channels if inner_channels == -1 else inner_channels
        self.num_heads = num_heads
        assert self.inner_channels % self.num_heads == 0, "inner_channels must be divisible by num_heads"
        self.dim_head = self.inner_channels // num_heads
        self.scale = self.dim_head**-0.5

        self.norm = norm(channels)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Conv1d(channels, self.inner_channels, 1)
        self.to_kv = nn.Conv1d(channels, self.inner_channels * 2, 1)

        self.to_out = nn.Sequential(nn.Linear(self.inner_channels, self.channels), nn.Dropout(dropout))

    def forward(self, x: Tensor, context: Tensor | None = None) -> Tensor:
        return checkpoint(self._forward, (x, context), self.parameters(), True if self.use_checkpoint else False)

    def _forward(self, x: Tensor, context: Tensor | None = None) -> Tensor:
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        context = default(context, self.norm(x))

        qkv = torch.cat(self.to_q(self.norm(x)), *self.to_kv(context).chunk(2, dim=1), dim=1)
        length = qkv.shape[-1]
        q, k, v = qkv.chunk(3, dim=1)

        dots = torch.einsum(
            "bct,bcs->bts",
            (q * scale).view(b * self.num_heads, self.dim_head, length),  # type: ignore
            (k * scale).view(b * self.num_heads, self.dim_head, length),  # type: ignore
        )
        attn = self.attend(dots.float()).type(dots.dtype)
        attn = self.dropout(attn)

        out = torch.einsum("bts,bcs->bct", attn, v.reshape(b * self.num_heads, self.dim_head, length))
        out = out.reshape(b, -1, length)
        out = self.to_out(out)
        return (x + out).reshape(b, c, *spatial)


class LabelEmbed(nn.Module):
    def __init__(self, num_classes: int, embed_dim: int, classifier_free_guidance: bool = False) -> None:
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
        labels: Tensor,
        p: float,
    ) -> Tensor:
        """
        Randomly drop labels from a batch.
        :param labels: an [N] tensor of labels.
        :param p: the probability of dropping a label.
        :return: an [N] tensor of modified labels.
        """
        return torch.where(torch.rand_like(labels) < p, self.num_classes, labels)

    def forward(self, labels: Tensor, p: float = 0) -> Tensor:
        """
        Embed a batch of labels.
        :param labels: an [N] tensor of labels.
        :param p: the probability of dropping a label.
        :return: an [N x embed_dim] tensor of embeddings.
        """
        if p > 0:
            assert self.classifier_free_guidance, "Label dropout is only supported with classifier-free guidance."
            labels = self.drop_labels(labels, p)
        return self.embedding(labels)


class PatchEmbed(nn.Module):
    def __init__(self, image_size: int, patch_size: int, in_channels: int, embed_dim: int) -> None:
        super().__init__()  # type: ignore
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_patches = (image_size // patch_size) ** 2
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: Tensor) -> Tensor:
        x = self.projection(x)  # shape: (batch_size, embed_dim, num_patches, num_patches)
        x = x.flatten(2)  # shape: (batch_size, embed_dim, num_patches^2)
        x = x.transpose(1, 2)  # shape: (batch_size, num_patches^2, embed_dim)
        return x


class SinCosPositionalEmbedding(nn.Module):
    """
    SinCosPositionalEmbedding module for adding sinusoidal positional embeddings to input tensors.

    Args:
        embed_size (int): The size of the embedding.

    Attributes:
        embed_size (int): The size of the embedding.
        max_len_cached (int): The maximum length of the positional embedding cache.
        pe (Tensor): The positional embedding matrix.

    Methods:
        _create_pe_matrix(max_len: int) -> Tensor:
            Creates the positional embedding matrix.
        _update_pe_cache(current_len: int) -> None:
            Updates the positional embedding cache if necessary.
        forward(x: Tensor, with_cls_token: bool = False) -> Tensor:
            Forward pass of the SinCosPositionalEmbedding module.

    """

    def __init__(self, embed_size: int):
        super().__init__()  # type: ignore
        self.embed_size = embed_size
        self.max_len_cached: int = 1  # Start with a minimal cache
        self.register_buffer("pe", self._create_pe_matrix(self.max_len_cached))

    def _create_pe_matrix(self, max_len: int) -> Tensor:
        """
        Creates the positional embedding matrix.

        Args:
            max_len (int): The maximum length of the positional embedding.

        Returns:
            Tensor: The positional embedding matrix.

        """

        """
        Updates the positional embedding cache if necessary.

        Args:
            current_len (int): The current length of the input tensor.

        Returns:
            None

        """
        pe = torch.zeros(max_len, self.embed_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.embed_size, 2).float() * (-math.log(10000.0) / self.embed_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # Add batch dimension

    def _update_pe_cache(self, current_len: int) -> None:
        if current_len > self.max_len_cached:
            self.pe = self._create_pe_matrix(current_len)
            self.max_len_cached = current_len

    def forward(self, x: Tensor, with_cls_token: bool = False) -> Tensor:
        """
        Forward pass of the SinCosPositionalEmbedding module.

        Args:
            x (Tensor): The input tensor.
            with_cls_token (bool, optional): Whether a cls token is included in the input tensor. Defaults to False.

        Returns:
            Tensor: The output tensor.

        """
        if self.cls_token:
            cls_token, x = x[:, :1], x[:, 1:]

        seq_len = x.size(1)
        self._update_pe_cache(seq_len)  # Update cache if necessary
        x = x + self.pe[:, :seq_len]

        if self.cls_token:
            x = torch.cat([cls_token, x], dim=1)  # Prepend the cls token # type: ignore
        return x

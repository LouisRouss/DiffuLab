from abc import abstractmethod
from typing import Callable

import torch
import torch.nn as nn
from torch import Tensor

from networks.common import Denoiser, contextEmbedder
from networks.utils.nn import (
    Downsample,
    LabelEmbed,
    Upsample,
    normalization,
    timestep_embedding,
)
from networks.utils.utils import checkpoint, default, zero_module


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
            if isinstance(layer, TimestepBlock) and isinstance(layer, ContextBlock):
                x = layer(x, emb, context)
            elif isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, ContextBlock):
                x = layer(x, context)
            else:
                x = layer(x)
        return x


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
        q_bias: bool = True,
        kv_bias: bool = True,
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

        self.to_q = nn.Conv1d(channels, self.inner_channels, 1, bias=q_bias)
        self.to_kv = nn.Conv1d(channels, self.inner_channels * 2, 1, bias=kv_bias)

        self.to_out = nn.Sequential(nn.Linear(self.inner_channels, self.channels), nn.Dropout(dropout))

    def forward(self, x: Tensor, context: Tensor | None = None) -> Tensor:
        return checkpoint(
            self._forward,
            (x, context),
            self.parameters(),
            True if self.use_checkpoint else False,
        )

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


class UNetModel(Denoiser):
    """
    Inspired by https://github.com/openai/guided-diffusion under MIT license as of 2024-18-08

    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor, for image colorization : Y_channels + X_channels .
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        image_size: list[int],
        in_channels: int,
        model_channels: int,
        out_channels: int,
        num_res_blocks: int,
        attention_resolutions: list[int],
        dropout: float = 0,
        channel_mult: str = "1, 2, 4, 8",
        conv_resample: bool = True,
        use_checkpoint: bool = False,
        use_fp16: bool = False,
        num_heads: int = 1,
        num_head_channels: int = -1,
        use_scale_shift_norm: bool = False,
        resblock_updown: bool = False,
        n_classes: int | None = None,
        classifier_free: bool = False,
        context_embedder: contextEmbedder | None = None,
    ):
        super().__init__()  # type: ignore
        assert (n_classes is None) != (
            context_embedder is None
        ), "n_classes and context_embedder cannot both be specified or both be None"
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult: list[int] = eval(f"[{channel_mult}]")
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = torch.bfloat16 if use_fp16 else torch.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.context_embedder = context_embedder
        self.classifier_free = classifier_free

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        self.label_embed = (
            LabelEmbed(n_classes, model_channels, self.classifier_free) if n_classes is not None else None
        )

        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList([EmbedSequential(nn.Conv2d(in_channels, ch, 3, padding=1))])
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers: list[nn.Module] = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            num_heads=num_heads,
                            dropout=dropout,
                            use_checkpoint=use_checkpoint,
                        )
                    )
                self.input_blocks.append(EmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    EmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(ch, conv_resample, out_channels=out_ch)
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = EmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                num_heads=num_heads,
                dropout=dropout,
                use_checkpoint=use_checkpoint,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            num_heads=num_heads,
                            dropout=dropout,
                            use_checkpoint=use_checkpoint,
                        ),
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(EmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(nn.Conv2d(input_ch, out_channels, 3, padding=1)),
        )

    def forward(
        self, x: Tensor, timesteps: Tensor, y: Tensor | None = None, context: Tensor | None = None, p: float = 0.0
    ) -> Tensor:
        """
        Apply the model to an input batch.
        :param x: a [N x C x ...] Tensor of noisy image.
        :param timesteps: a 1-D batch of timesteps.
        :param y: a [N x C x ...] Tensor of labels.
        :param p: the probability of dropping the ground-truth label in the label embedding.
        :param context: a [N x C x ...] Tensor of context for CrossAttention, can be images, text etc...
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        assert (context is not None) == (
            self.context_embedder is not None
        ), "must specify context if and only if the model is context-conditional"

        hs: list[Tensor] = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        if self.label_embed is not None:
            emb = emb + self.label_embed(y, p)
        if self.context_embedder is not None:
            context = self.context_embedder(context, p)
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h: Tensor = module(h, emb=emb, context=context)
            hs.append(h)
        h = self.middle_block(h, emb=emb, context=context)
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb=emb, context=context)
        return self.out(h)

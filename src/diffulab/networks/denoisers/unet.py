from abc import abstractmethod
from typing import Any, Callable

import torch
import torch.nn as nn
from einops import rearrange
from jaxtyping import Bool, Float, Int
from torch import Tensor
from torch.utils.checkpoint import checkpoint  # type: ignore

from diffulab.networks.denoisers.common import Denoiser, ModelOutput
from diffulab.networks.embedders.common import ContextEmbedder, ContextEmbedderOutput
from diffulab.networks.utils.nn import (
    Downsample,
    LabelEmbed,
    Upsample,
    normalization,
    timestep_embedding,
)
from diffulab.networks.utils.utils import default, zero_module


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(
        self, x: Float[Tensor, "batch_size channels height width"], emb: Float[Tensor, "batch_size emb_channels"]
    ) -> Float[Tensor, "batch_size channels height width"]:
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class ContextBlock(nn.Module):
    """
    Any module where forward() takes context embeddings as a second argument and
    attnention mask as an optional third argument.
    """

    @abstractmethod
    def forward(
        self,
        x: Float[Tensor, "batch_size channels height width"],
        context: Float[Tensor, "batch_size context_channels context_length"],
        attn_mask: Bool[Tensor, "batch_size seq_len_context"] | Int[Tensor, "batch_size seq_len_context"] | None = None,
    ) -> Float[Tensor, "batch_size channels height width"]:
        """
        Apply the module to `x` given `context` embeddings.
        """


class EmbedSequential(nn.Sequential, TimestepBlock, ContextBlock):  # type: ignore
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(  # type:ignore
        self,
        x: Float[Tensor, "batch_size channels height width"],
        emb: Float[Tensor, "batch_size emb_channels"],
        context: Float[Tensor, "batch_size context_channels context_length"] | None = None,
        attn_mask: Bool[Tensor, "batch_size seq_len_context"] | Int[Tensor, "batch_size seq_len_context"] | None = None,
    ) -> Float[Tensor, "batch_size channels height width"]:
        for layer in self:
            if isinstance(layer, TimestepBlock) and isinstance(layer, ContextBlock):
                x = layer(x, emb, context, attn_mask=attn_mask)
            elif isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, ContextBlock):
                x = layer(x, context, attn_mask=attn_mask)
            else:
                x = layer(x)
        return x


class ResBlock(TimestepBlock):
    """
    Residual block used in the U-Net denoiser, optionally performing up/down sampling
    and channel projection. Adapted (with modifications) from OpenAI's guided-diffusion
    implementation (MIT License).

    This block:
        * Applies GroupNorm + SiLU + Conv2d to the input.
        * Optionally upsamples or downsamples the residual and main branch in a synchronized way.
        * Incorporates a timestep (or generic conditioning) embedding through an MLP that produces
            either additive bias or (scale, shift) parameters for Feature-wise Linear Modulation
            (FiLM-like) when use_scale_shift_norm is True.
        * Applies dropout and a zero-initialized output convolution (zero_module) to stabilize training.
        * Adds a learned (or identity) skip connection; can use a 1x1 or 3x3 projection if the
            channel count changes (configurable via use_conv).
        * Supports gradient checkpointing to reduce memory usage at the cost of extra compute.

    Args:
            channels (int): Number of input feature channels.
            emb_channels (int): Dimensionality of the conditioning (timestep) embedding provided to forward.
            dropout (float): Dropout probability applied before the final convolution in the main branch.
            out_channels (int | None, optional): Number of output channels. Defaults to channels if None.
            use_conv (bool, optional): If True and out_channels differs from channels, use a 3x3 convolution in the
                skip path; otherwise a 1x1 convolution is used.
            use_scale_shift_norm (bool, optional): If True, interpret the embedding MLP output as (scale, shift) and apply them
                after normalization (FiLM style). If False, embedding is added directly to the
                feature map (broadcast).
            use_checkpoint (bool, optional): If True, wrap the forward pass in torch.utils.checkpoint to trade compute for
                reduced memory.
            up (bool, optional): If True, perform learned nearest-neighbor-like upsampling on both residual and
                skip paths before the main convolution.
            down (bool, optional): If True, perform strided (anti-aliased) downsampling on both residual and skip
                paths before the main convolution.
    Notes:
            * zero_module on the final convolution helps the network start as an identity,
                often stabilizing diffusion model training.
            * use_scale_shift_norm implements the "scale-shift" variant seen in certain
                diffusion implementations (similar to adaptive GroupNorm).

    Raises:
            RuntimeError: If both up and down are True (mutually exclusive expectation).
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

    def forward(
        self, x: Float[Tensor, "batch_size channels height width"], emb: Float[Tensor, "batch_size emb_channels"]
    ) -> Float[Tensor, "batch_size channels height width"]:
        """
        Apply the module to an input feature map, conditioned on a (timestep) embedding.

        This method optionally uses gradient checkpointing (torch.utils.checkpoint.checkpoint)
        to reduce memory consumption during the backward pass if self.use_checkpoint is True.

        Args:
            x (torch.Tensor): Input feature tensor of shape (batch_size, channels, height, width).
                Represents the current latent / feature map to be processed.
            emb (torch.Tensor): Conditioning embedding tensor of shape (batch_size, emb_channels),
                typically a timestep or diffusion/noise level embedding already projected to the
                expected dimensionality.

        Returns:
            torch.Tensor: Output feature tensor of shape (batch_size, channels, height, width),
            transformed by the underlying block while incorporating the conditioning embedding.

        Raises:
            RuntimeError: If checkpointing fails due to incompatible inputs.
            ValueError: If input dimensionalities do not match model expectations.

        Notes:
            - Gradient checkpointing trades extra forward compute for reduced activation memory.
            - The underlying computation is delegated to self._forward(x, emb).
            - The channel dimension of x is preserved.
        """
        return (
            checkpoint(self._forward, *(x, emb), use_reentrant=False) if self.use_checkpoint else self._forward(x, emb)
        )  # type: ignore

    def _forward(
        self, x: Float[Tensor, "batch_size channels height width"], emb: Float[Tensor, "batch_size emb_channels"]
    ) -> Float[Tensor, "batch_size channels height width"]:
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
        context_channels: int | None = None,
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
        self.context_channels = context_channels or channels
        self.inner_channels = channels if inner_channels == -1 else inner_channels
        self.num_heads = num_heads
        assert self.inner_channels % self.num_heads == 0, "inner_channels must be divisible by num_heads"
        self.dim_head = self.inner_channels // num_heads
        self.scale = self.dim_head**-0.5

        self.norm_x = norm(self.channels)
        self.norm_context = norm(self.context_channels)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Conv1d(self.channels, self.inner_channels, 1, bias=q_bias)
        self.to_kv = nn.Conv1d(self.context_channels, self.inner_channels * 2, 1, bias=kv_bias)

        self.to_out = nn.Sequential(nn.Conv1d(self.inner_channels, self.channels, 1), nn.Dropout(dropout))

    def forward(
        self,
        x: Float[Tensor, "batch_size channels height width"],
        context: Float[Tensor, "batch_size context_channels context_length"] | None = None,
        attn_mask: Bool[Tensor, "batch_size seq_len_context"] | Int[Tensor, "batch_size seq_len_context"] | None = None,
    ) -> Float[Tensor, "batch_size channels height width"]:
        return (
            checkpoint(
                self._forward,
                *(x, context, attn_mask),
                use_reentrant=False,
            )
            if self.use_checkpoint
            else self._forward(x, context, attn_mask)
        )  # type: ignore

    def _forward(
        self,
        x: Float[Tensor, "batch_size channels height width"],
        context: Float[Tensor, "batch_size context_channels context_length"] | None = None,
        attn_mask: Bool[Tensor, "batch_size seq_len_context"] | Int[Tensor, "batch_size seq_len_context"] | None = None,
    ) -> Float[Tensor, "batch_size channels height width"]:
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        context = default(context, x)

        q = self.to_q(self.norm_x(x))  # (b, inner_channels, x_len)
        k, v = self.to_kv(self.norm_context(context)).chunk(2, dim=1)  # (b, inner_channels, context_len) each

        q, k, v = (
            rearrange(q, "b (h d) n -> b h n d", h=self.num_heads),
            rearrange(k, "b (h d) n  -> b h n d", h=self.num_heads),
            rearrange(v, "b (h d) n  -> b h n d", h=self.num_heads),
        )

        if attn_mask is not None:
            b, n = attn_mask.shape
            attn_mask = attn_mask[:, None, None, :].expand(b, 1, x.shape[2], n)

        out = nn.functional.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            scale=self.scale,
            attn_mask=attn_mask,
        )
        out = rearrange(out, "b h n d -> b (h d) n")
        out = self.to_out(out)
        return (x + out).reshape(b, c, *spatial)


class GEGLU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()  # type: ignore[reportUnknownMemberType]
        self.proj = nn.Conv1d(in_channels, out_channels * 2, kernel_size=1)

    def forward(
        self, x: Float[Tensor, "batch_size in_channels seq_len"]
    ) -> Float[Tensor, "batch_size out_channels seq_len"]:
        x_proj = self.proj(x)
        x, gate = x_proj.chunk(2, dim=1)
        return x * torch.nn.functional.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, in_channels: int, inner_channels: int, dropout: float = 0.0):
        super().__init__()  # type: ignore[reportUnknownMemberType]
        self.net = nn.Sequential(
            GEGLU(in_channels, inner_channels),
            nn.Dropout(dropout),
            nn.Conv1d(inner_channels, in_channels, kernel_size=1),
        )
        self.norm = normalization(in_channels)

    def forward(
        self, x: Float[Tensor, "batch_size channels height width"]
    ) -> Float[Tensor, "batch_size channels height width"]:
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        h = self.norm(x)
        h = self.net(h)
        return (x + h).reshape(b, c, *spatial)


class TransformerAttentionBlock(ContextBlock):
    def __init__(
        self,
        channels: int,
        context_channels: int | None = None,
        num_heads: int = 8,
        inner_channels: int = -1,
        dropout: float = 0.0,
        norm: Callable[..., nn.Module] = normalization,
        use_checkpoint: bool = False,
        q_bias: bool = True,
        kv_bias: bool = True,
        mlp_ratio: int = 4,
    ):
        super().__init__()  # type: ignore[reportUnknownMemberType]
        self.self_attn = AttentionBlock(
            channels,
            context_channels=None,
            num_heads=num_heads,
            inner_channels=inner_channels,
            dropout=dropout,
            norm=norm,
            use_checkpoint=use_checkpoint,
            q_bias=q_bias,
            kv_bias=kv_bias,
        )
        self.cross_attn = AttentionBlock(
            channels,
            context_channels=context_channels,
            num_heads=num_heads,
            inner_channels=inner_channels,
            dropout=dropout,
            norm=norm,
            use_checkpoint=use_checkpoint,
            q_bias=q_bias,
            kv_bias=kv_bias,
        )
        self.ff = FeedForward(channels, channels * mlp_ratio, dropout)

    def forward(
        self,
        x: Float[Tensor, "batch_size channels height width"],
        context: Float[Tensor, "batch_size context_channels context_length"] | None = None,
        attn_mask: Bool[Tensor, "batch_size seq_len_context"] | Int[Tensor, "batch_size seq_len_context"] | None = None,
    ) -> Float[Tensor, "batch_size channels height width"]:
        # residuals are handled in the submodules
        h = self.self_attn(x)
        h = self.cross_attn(h, context=context, attn_mask=attn_mask)
        return self.ff(h)


class TransformerBlock(ContextBlock):
    def __init__(
        self,
        channels: int,
        context_channels: int | None = None,
        num_heads: int = 8,
        inner_channels: int = -1,
        dropout: float = 0.0,
        norm: Callable[..., nn.Module] = normalization,
        use_checkpoint: bool = False,
        q_bias: bool = True,
        kv_bias: bool = True,
        mlp_ratio: int = 4,
        depth: int = 1,
    ):
        super().__init__()  # type: ignore[reportUnknownMemberType]
        self.channels = channels
        self.context_channels = context_channels or channels
        self.inner_channels = channels if inner_channels == -1 else inner_channels
        self.num_heads = num_heads
        assert self.inner_channels % self.num_heads == 0, "inner_channels must be divisible by num_heads"
        self.dim_head = self.inner_channels // num_heads

        self.norm_x = norm(self.channels)
        self.proj_in = nn.Conv2d(self.channels, self.inner_channels, kernel_size=1, stride=1, padding=0)
        self.attn_blocks = nn.ModuleList(
            [
                TransformerAttentionBlock(
                    channels=self.inner_channels,
                    context_channels=self.context_channels,
                    num_heads=num_heads,
                    dropout=dropout,
                    norm=norm,
                    use_checkpoint=use_checkpoint,
                    q_bias=q_bias,
                    kv_bias=kv_bias,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(depth)
            ]
        )
        self.proj_out = nn.Conv2d(self.inner_channels, self.channels, kernel_size=1, stride=1, padding=0)

    def forward(
        self,
        x: Float[Tensor, "batch_size channels height width"],
        context: Float[Tensor, "batch_size context_channels context_length"],
        attn_mask: Bool[Tensor, "batch_size seq_len_context"] | Int[Tensor, "batch_size seq_len_context"] | None = None,
    ) -> Float[Tensor, "batch_size channels height width"]:
        assert context is not None, "TransformerBlock requires context input"
        h = self.norm_x(x)
        h = self.proj_in(h)
        for attn_block in self.attn_blocks:
            h = attn_block(h, context=context, attn_mask=attn_mask)
        h = self.proj_out(h)
        return x + h


class UNetModel(Denoiser):
    """
    U-Net denoiser with optional class-label and/or multimodal context conditioning.

    This model is a configurable U-Net backbone with:
      - timestep embeddings added via ResBlocks (FiLM-style optional scale/shift),
      - optional class-label conditioning for classifier-free guidance,
      - optional cross-attention conditioning via a ContextEmbedder (Transformer blocks),
      - optional gradient checkpointing for reduced memory usage.

    The spatial encoder-decoder is organized into stages specified by `channel_mult` and
    `num_res_blocks`, with attention (self-attn or cross-attn) inserted at resolutions
    defined in `attention_resolutions`. If a `context_embedder` is provided (n_output=1),
    cross-attention is used; otherwise, self-attention is used.

    Args:
        image_size (list[int]): Spatial size [H, W] the model expects. Inputs must match this
            size at training/inference.
        in_channels (int): Number of input channels consumed by the first convolution. If you
            plan to provide `x_context` to forward (concatenated to x), set this to the
            combined channel count (C + C_ctx) to avoid shape mismatch.
        model_channels (int): Base channel width. Each level scales this by the factors in
            `channel_mult`.
        out_channels (int): Number of output channels predicted by the final conv head.
        num_res_blocks (int): Number of residual blocks per resolution level (encoder and decoder).
        attention_resolutions (list[int]): Downsampling factors at which to insert attention
            blocks. The running downsample factor starts at 1 at the highest resolution,
            doubles at each downsample, and halves on upsample. If the current factor `ds`
            is in this list, an attention (or transformer) block is added at that stage.
        dropout (float, default=0.0): Dropout probability used inside ResBlocks and attention FFNs.
        channel_mult (str, default="1, 2, 4, 8"): Comma-separated multipliers determining the
            channel width at each level: channels[level] = model_channels * channel_mult[level].
        conv_resample (bool, default=True): If True, use convolutional up/downsampling ops
            in the simple Upsample/Downsample paths.
        use_checkpoint (bool, default=False): Enable torch.utils.checkpoint to trade compute
            for reduced activation memory in supported blocks.
        num_heads (int, default=1): Number of heads used by attention blocks.
        use_scale_shift_norm (bool, default=False): If True, ResBlocks interpret the embedding MLP
            output as (scale, shift) for FiLM-like modulation; otherwise the embedding is added.
        resblock_updown (bool, default=False): If True, perform up/down sampling within ResBlocks
            (learned, anti-aliased). If False, use separate Upsample/Downsample modules.
        n_classes (int | None, default=None): Number of classes for class-conditional training. If
            set, a label embedding is added to the timestep embedding in forward.
        classifier_free (bool, default=False): Enable classifier-free guidance. When True, labels
            can be randomly dropped with probability `p` during forward. Requires `n_classes` set.
        context_embedder (ContextEmbedder | None, default=None): Optional embedder for external
            conditioning (e.g., text). Must have `n_output == 1` and returns:
              - embeddings: [B, C_ctx, L_ctx]
              - attn_mask (optional): [B, L_ctx]
            When provided, cross-attention Transformer blocks are used at attention stages.
        transformer_depth (int, default=1): Number of stacked attention+MLP sub-blocks inside each
            TransformerBlock inserted at attention resolutions.

    Notes:
        - Timestep embeddings are produced by `timestep_embedding(t, model_channels)` and
          projected to `time_embed_dim = 4 * model_channels`.
        - If `n_classes` is set, a LabelEmbed module adds label conditioning to the time embedding.
        - With classifier-free guidance enabled, labels may be dropped in forward with probability `p`.
        - If `context_embedder` is provided, it must have `n_output == 1`; its embeddings and
          optional mask are fed to cross-attention Transformer blocks at configured resolutions.
        - If you pass `x_context` to forward, ensure `in_channels` already accounts for those
          extra channels (i.e., set `in_channels = C + C_ctx`).
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
        num_heads: int = 1,
        use_scale_shift_norm: bool = False,
        resblock_updown: bool = False,
        n_classes: int | None = None,
        classifier_free: bool = False,
        context_embedder: ContextEmbedder | None = None,
        transformer_depth: int = 1,
    ):
        super().__init__()  # type: ignore
        assert not (n_classes is not None and context_embedder is not None), (
            "n_classes and context_embedder cannot both be specified"
        )

        if context_embedder:
            assert context_embedder.n_output == 1, (
                "For UNet please provide a context embedder with n_output=1 (context and attention mask)"
            )  # context and attention mask

        self.context_channels = None if context_embedder is None else context_embedder.output_size[0]
        self.transformer_depth = transformer_depth
        self.use_context = self.context_channels is not None

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
        self.num_heads = num_heads
        self.context_embedder = context_embedder
        self.classifier_free = classifier_free
        self.n_classes = n_classes

        self.time_embed_dim = self.model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(self.model_channels, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )

        self.label_embed = (
            LabelEmbed(self.n_classes, self.time_embed_dim, self.classifier_free)
            if self.n_classes is not None
            else None
        )

        ch = input_ch = int(self.channel_mult[0] * self.model_channels)
        self.input_blocks = nn.ModuleList([EmbedSequential(nn.Conv2d(in_channels, ch, 3, padding=1))])
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(self.channel_mult):
            for _ in range(num_res_blocks):
                layers: list[nn.Module] = [
                    ResBlock(
                        ch,
                        self.time_embed_dim,
                        dropout,
                        out_channels=int(mult * self.model_channels),
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * self.model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            num_heads=num_heads,
                            dropout=dropout,
                            use_checkpoint=use_checkpoint,
                        )
                        if not self.use_context
                        else TransformerBlock(
                            ch,
                            context_channels=self.context_channels,
                            num_heads=num_heads,
                            dropout=dropout,
                            use_checkpoint=use_checkpoint,
                            depth=self.transformer_depth,
                        )
                    )
                self.input_blocks.append(EmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(self.channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    EmbedSequential(
                        ResBlock(
                            ch,
                            self.time_embed_dim,
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
                self.time_embed_dim,
                dropout,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                context_channels=self.context_channels,
                num_heads=num_heads,
                dropout=dropout,
                use_checkpoint=use_checkpoint,
            )
            if not self.use_context
            else TransformerBlock(
                ch,
                context_channels=self.context_channels,
                num_heads=num_heads,
                dropout=dropout,
                use_checkpoint=use_checkpoint,
                depth=self.transformer_depth,
            ),
            ResBlock(
                ch,
                self.time_embed_dim,
                dropout,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(self.channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        self.time_embed_dim,
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
                            context_channels=self.context_channels,
                            num_heads=num_heads,
                            dropout=dropout,
                            use_checkpoint=use_checkpoint,
                        )
                        if not self.use_context
                        else TransformerBlock(
                            ch,
                            context_channels=self.context_channels,
                            num_heads=num_heads,
                            dropout=dropout,
                            use_checkpoint=use_checkpoint,
                            depth=self.transformer_depth,
                        ),
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            self.time_embed_dim,
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
        self,
        x: Float[Tensor, "batch_size channels height width"],
        timesteps: Float[Tensor, "batch_size"],
        y: Int[Tensor, "batch_size"] | None = None,
        context: Any | None = None,
        p: float = 0.0,
        x_context: Float[Tensor, "batch_size channels height width"] | None = None,
    ) -> ModelOutput:
        """
        Apply the U-Net denoiser to a batch of (noisy) images, optionally conditioned on
        class labels and/or external context (e.g. text or images) with support for
        classifier-free guidance and auxiliary input concatenation.

        Args:
            x (torch.Tensor): Noisy input images of shape (N, C, H, W). The spatial
                size (H, W) must match the model's configured image_size.
            timesteps (torch.Tensor): 1D tensor of shape (N,) containing diffusion
                timesteps (typically integers or floats in a predefined range).
            y (torch.Tensor | None): Optional class label tensor of shape (N,) or a
                shape accepted by the underlying label embedding module. Must be
                provided iff the model is class-conditional (self.n_classes is not None).
            context (Any | None): Optional conditioning input consumed by the
                context_embedder (e.g. tokenized text, image features). Must be
                provided if the model is context-conditional (self.context_embedder
                is not None).
            p (float, default=0.0): Probability of dropping labels (and/or context,
                depending on embedder implementation) for classifier-free guidance.
                Requires classifier_free = True and n_classes to be set when > 0.
            x_context (torch.Tensor | None): Optional tensor of shape (N, C_ctx, H, W)
                concatenated channel-wise to x before the first block. Useful for
                conditioning on an aligned spatial map (e.g. segmentation, guidance
                image, mask).

        Returns:
            ModelOutput: A dictionary with key:
                - "x": torch.Tensor of denoised outputs (shape (N, C_out, H, W)),
                  where C_out is determined by the final convolution/output head.

        Raises:
            AssertionError:
                - If (H, W) of x does not match model image_size.
                - If y is (not) provided inconsistently with class-conditional setup.
                - If context is (not) provided inconsistently with context-conditional setup.
                - If p > 0 but classifier-free guidance or n_classes are not properly configured.

        Notes:
            - Time embeddings are generated via timestep_embedding and passed through
              a learned projection (time_embed).
            - If label embedding is enabled, its contribution is added to the time embedding.
            - When classifier-free guidance is active (p > 0), labels (and/or context)
              may be randomly replaced/dropped during embedding to enable guidance at sampling.
            - Intermediate activations are stored (hs) to form skip connections in the
              decoder path.

        Example:
        ```python
            output = model.forward(
                x=noisy_images,
                timesteps=torch.randint(0, T, (batch_size,)),
                y=labels,
                context=text_tokens,
                p=0.1,
            pred = output["x"]
        ```
        """
        assert list(x.shape[2:]) == self.image_size, (
            f"Input shape {x.shape[2:]} does not match model image size {self.image_size}"
        )

        assert (y is not None) == (self.n_classes is not None), (
            "must specify y if and only if the model is class-conditional"
        )

        assert (context is not None) == (self.context_embedder is not None), (
            "must specify context if and only if the model is context-conditional"
        )

        if p > 0:
            assert self.classifier_free, (
                "probability of dropping for classifier free guidance is only available if model is set up to be classifier free"
            )
            assert self.n_classes, (
                "probability of dropping for classifier free guidance is only available if a number of classes is set"
            )
        hs: list[Tensor] = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        if self.label_embed is not None:
            emb = emb + self.label_embed(y, p)

        attn_mask = None
        if self.context_embedder is not None:
            context_output: ContextEmbedderOutput = self.context_embedder(context, p)
            context = context_output["embeddings"]
            attn_mask = context_output.get("attn_mask")

        if x_context is not None:
            x = torch.cat([x, x_context], dim=1)
        h = x
        for module in self.input_blocks:
            h: Tensor = module(h, emb=emb, context=context, attn_mask=attn_mask)
            hs.append(h)
        h = self.middle_block(h, emb=emb, context=context, attn_mask=attn_mask)
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb=emb, context=context, attn_mask=attn_mask)
        return {"x": self.out(h)}

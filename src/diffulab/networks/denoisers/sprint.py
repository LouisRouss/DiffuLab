import logging
from typing import Any

import torch
import torch.nn as nn
from einops import rearrange
from jaxtyping import Float, Int
from torch import Tensor

from diffulab.networks.denoisers.common import Denoiser, ModelOutput
from diffulab.networks.denoisers.mmdit import DiTBlock, MMDiTBlock, MMDiTSingleStreamBlock, ModulatedLastLayer
from diffulab.networks.embedders.common import ContextEmbedder, ContextEmbedderOutput
from diffulab.networks.utils.nn import (
    LabelEmbed,
    Modulation,
    get_cos_sin_ndim_grid,
    timestep_embedding,
)
from diffulab.networks.utils.utils import zero_module


class SprintDiT(Denoiser):
    """
    (mm)DiT with sprint integration (https://arxiv.org/pdf/2510.21986).

    Args:
        simple_dit (bool): If True, use DiT blocks with class-label conditioning only (no context
            or cross-attention). If False, use MMDiT blocks with cross-attention to contextual tokens
            produced by `context_embedder`. Default: False.
        input_channels (int): Number of channels of the main input x. Default: 3.
        output_channels (int | None): Number of channels to predict. If None, equals `input_channels`.
            Default: None.
        inner_dim (int): Token/patch embedding width for the stream. Default: 4096.
        num_heads (int): Number of attention heads in each block. Default: 16.
        mlp_ratio (int): Expansion ratio for the MLP in each block. Default: 4.
        patch_size (int): Side length P of square patches. Images are projected with stride P. Default: 16.
        context_dim (int): Model width for contextual tokens after `context_embed` when
            `simple_dit=False`. Ignored when `simple_dit=True`. Default: 4096.
        encoder_depth (int): Number of transformer blocks in the encoder. Default: 2.
        deep_layers_depth (int): Number of transformer blocks in the deep layers path. Default: 8.
        decoder_depth (int): Number of transformer blocks in the decoder. Default: 2.
        rope_base (int): Base frequency for RoPE. Default: 10000.
        partial_rotary_factor (float): Fraction of each head dimension using RoPE.
            1.0 means full rotary. Default: 1.0.
        rope_axes_dim (list[int] | None): List of dimensions for rotary positional embeddings.
            When `simple_dit=True`, should contain 2 integers for H and W axes. When `simple_dit=False`,
            should contain 3 integers for L, H, W axes. If None, defaults are used based on
            partial_rotary_factor and the heads_dim. Default: None
        frequency_embedding (int): Size of the Fourier timestep embedding before the time MLP.
            Default: 256.
        n_classes (int | None): Number of classes for label conditioning in `simple_dit` mode.
            Required to use classifier-free guidance with labels. Must be None when using
            a `context_embedder`. Default: None.
        classifier_free (bool): Enables classifier-free guidance. In `simple_dit`, it applies to
            dropped labels; in MMDiT mode, it is forwarded to the context embedder which may drop
            context. Additionally for Sprint, drops the deep layers path (path free guidance). Default: False.
        context_embedder (ContextEmbedder | None): When `simple_dit=False`, a module returning
            `ContextEmbedderOutput`. Must be provided for text/image conditioning and must be None
            when `simple_dit=True`. If the embedder returns pooled and token embeddings
            (`n_output == 2`), pooled features are fused into the timestep embedding via an MLP.
            Default: None.
        use_checkpoint (bool): Enable torch.utils.checkpoint in blocks to trade compute for memory.
            Default: False.
        drop_rate (float): Fraction of image tokens to drop in the deep layers path during training.
            Default: 0.75.
    """

    def __init__(
        self,
        simple_dit: bool = False,
        input_channels: int = 3,
        output_channels: int | None = None,
        inner_dim: int = 768,
        embedding_dim: int = 768,
        num_heads: int = 12,
        mlp_ratio: int = 4,
        patch_size: int = 16,
        encoder_depth: int = 2,
        deep_layers_depth: int = 8,
        n_single_stream_blocks: int = 0,
        decoder_depth: int = 2,
        rope_base: int = 10_000,
        partial_rotary_factor: float = 1,
        rope_axes_dim: list[int] | None = None,
        frequency_embedding: int = 256,
        n_classes: int | None = None,
        classifier_free: bool = False,
        context_embedder: ContextEmbedder | None = None,
        use_checkpoint: bool = False,
        drop_rate: float = 0.75,
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
        self.frequency_embedding = frequency_embedding
        self.rope_base = rope_base

        self.n_classes = n_classes
        self.classifier_free = classifier_free

        self.mask_token = nn.Parameter(torch.zeros(1, 1, inner_dim))
        self.drop_rate = drop_rate

        heads_dim = inner_dim // num_heads
        if not self.simple_dit:
            assert self.context_embedder is not None, "for dit with text context embedder must be provided"
            assert isinstance(self.context_embedder.output_size, tuple) and all(
                isinstance(i, int) for i in self.context_embedder.output_size
            ), "context_embedder.output_size must be a tuple of integers"

            self.pooled_embedding = False
            self.mlp_pooled_context = None
            if self.context_embedder.n_output == 2:
                self.pooled_embedding = True
                self.mlp_pooled_context = nn.Sequential(
                    nn.Linear(self.context_embedder.output_size[0], embedding_dim * 2),
                    nn.SiLU(),
                    nn.Linear(embedding_dim * 2, embedding_dim),
                )
                self.context_embed = nn.Linear(self.context_embedder.output_size[1], inner_dim, bias=False)
            else:
                assert self.context_embedder.n_output == 1
                self.context_embed = nn.Linear(self.context_embedder.output_size[0], inner_dim, bias=False)
            if rope_axes_dim is None:
                rope_axes_dim = [
                    int((partial_rotary_factor * heads_dim) // 3),  # L for text, set to 0 for image tokens
                    int((partial_rotary_factor * heads_dim) // 3),  # H set to 0 for text
                    int((partial_rotary_factor * heads_dim) // 3),  # W set to 0 for text
                ]
        else:
            self.label_embed = (
                LabelEmbed(self.n_classes, embedding_dim, self.classifier_free) if self.n_classes is not None else None
            )
            if rope_axes_dim is None:
                rope_axes_dim = [
                    int((partial_rotary_factor * heads_dim) // 2),  # H
                    int((partial_rotary_factor * heads_dim) // 2),  # W
                ]
            if n_single_stream_blocks > 0:
                logging.warning(
                    "n_single_stream_blocks is ignored when simple_dit=True. All blocks are single-stream DiT blocks."
                )
                n_single_stream_blocks = encoder_depth + deep_layers_depth + decoder_depth

        self.rope_axes_dim = rope_axes_dim

        self.time_embed = nn.Sequential(
            nn.Linear(self.frequency_embedding, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

        self.conv_proj = nn.Conv2d(
            self.input_channels, inner_dim, kernel_size=self.patch_size, stride=self.patch_size, bias=False
        )

        self.fuse = nn.Linear(inner_dim * 2, inner_dim, bias=False)
        if not self.simple_dit:
            self.fuse_context = nn.Linear(2 * inner_dim, inner_dim, bias=False)
        self.last_layer = ModulatedLastLayer(
            embedding_dim=embedding_dim,
            hidden_size=inner_dim,
            patch_size=self.patch_size,
            out_channels=self.output_channels,
        )

        # --------------
        # encoder layers
        # --------------
        self.layers = nn.ModuleList(  # name compatibility for RePA
            [
                MMDiTBlock(
                    inner_dim=inner_dim,
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    rope_axes_dim=self.rope_axes_dim,
                    use_checkpoint=use_checkpoint,
                )
                if not self.simple_dit
                else DiTBlock(
                    inner_dim=inner_dim,
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    rope_axes_dim=self.rope_axes_dim,
                    use_checkpoint=use_checkpoint,
                )
                for _ in range(encoder_depth)
            ]
        )

        # --------------
        # deep layers
        # --------------
        self.deep_layers = nn.ModuleList(
            [
                MMDiTBlock(
                    inner_dim=inner_dim,
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    rope_axes_dim=self.rope_axes_dim,
                    use_checkpoint=use_checkpoint,
                )
                if not self.simple_dit
                else DiTBlock(
                    inner_dim=inner_dim,
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    rope_axes_dim=self.rope_axes_dim,
                    use_checkpoint=use_checkpoint,
                )
                for _ in range(deep_layers_depth - n_single_stream_blocks)
            ]
            + [
                MMDiTSingleStreamBlock(
                    inner_dim=inner_dim,
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    rope_axes_dim=self.rope_axes_dim,
                )
            ]
        )

        # --------------
        # decoder layers
        # --------------
        self.decoder_layers = nn.ModuleList(
            [
                MMDiTBlock(
                    inner_dim=inner_dim,
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    rope_axes_dim=self.rope_axes_dim,
                    use_checkpoint=use_checkpoint,
                )
                if not self.simple_dit
                else DiTBlock(
                    inner_dim=inner_dim,
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    rope_axes_dim=self.rope_axes_dim,
                    use_checkpoint=use_checkpoint,
                )
                for _ in range(decoder_depth)
            ]
        )

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:  # type: ignore
                nn.init.constant_(module.bias, 0)
        if isinstance(module, Modulation):
            zero_module(module)
        if isinstance(module, ModulatedLastLayer):
            zero_module(module.adaLN_modulation)

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
        _, _, Hp, Wp = x.shape
        self.grid_size = (Hp, Wp)
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
        # Reshape the tensor to the original image dimensions
        x = rearrange(
            x,
            "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
            h=self.grid_size[0],
            w=self.grid_size[1],
            p1=self.patch_size,
            p2=self.patch_size,
            c=self.output_channels,
        )
        return x

    def drop_tokens(
        self,
        x: Float[Tensor, "batch_size seq_len patch_dim"],
        cos_sin_rope: tuple[Float[Tensor, "batch_size seq_len rope_dim"], ...],
    ) -> tuple[
        Float[Tensor, "batch_size kept_seq_len patch_dim"],
        Int[Tensor, "batch_size kept_seq_len"],
        tuple[Float[Tensor, "batch_size kept_seq_len rope_dim"], ...],
    ]:
        """
        Drop a fraction of tokens during training for Sprint.
        Args:
            x (Tensor): Input tensor of shape (B, seq_len, patch_dim)
            cos_sin_rope (tuple): Tuple of RoPE tensors, each of shape (B, seq_len, rope_dim)
        Returns:
            x_dropped (Tensor): Tensor with dropped tokens of shape (B, kept_seq_len, patch_dim)
            kept_indices (Tensor): Indices of kept tokens of shape (B, kept_seq_len)
            cos_sin_rope_dropped (tuple): Tuple of RoPE tensors for kept tokens,
                each of shape (B, kept_seq_len, rope_dim)
        """
        B, S, D = x.shape

        if not self.training:
            return x, torch.arange(S, device=x.device).expand(B, S), cos_sin_rope

        k = max(1, int(S * (1.0 - float(self.drop_rate))))
        scores = torch.rand((B, S), device=x.device, dtype=torch.float32)
        kept_indices = torch.topk(scores, k=k, dim=1, largest=True, sorted=False).indices.to(torch.long)
        order = torch.argsort(kept_indices, dim=1)
        kept_indices = torch.gather(kept_indices, dim=1, index=order)

        x_dropped = torch.gather(x, dim=1, index=kept_indices.unsqueeze(-1).expand(B, k, D))
        cos_sin_rope_dropped = tuple(
            torch.gather(rope, dim=1, index=kept_indices.unsqueeze(-1).expand(B, k, rope.shape[-1]))
            for rope in cos_sin_rope
        )

        return x_dropped, kept_indices, cos_sin_rope_dropped

    def restore_tokens(
        self,
        x_dropped: Float[Tensor, "batch_size kept_seq_len patch_dim"],
        kept_indices: Int[Tensor, "batch_size kept_seq_len"],
        path_drop_p: float = 0.0,
    ) -> Float[Tensor, "batch_size seq_len patch_dim"]:
        """
        Restore dropped tokens to their original positions using a mask token.
        Args:
            x_dropped (Tensor): Tensor with dropped tokens of shape (B, kept_seq_len, patch_dim)
            kept_indices (Tensor): Indices of kept tokens of shape (B, kept_seq_len)
            path_drop_p (float): Probability of dropping the dense path during training. Default: 0.0
        Returns:
            x_full (Tensor): Tensor with restored tokens of shape (B, seq_len, patch_dim)
        """
        H, W = self.grid_size
        B, _, D = x_dropped.shape
        S = H * W

        # Create full tensor filled with mask token
        mask_token = self.mask_token.to(dtype=x_dropped.dtype)
        x_full = mask_token.expand(B, S, D).clone()

        # Scatter kept tokens back to their original positions
        kept_indices_expanded = kept_indices.unsqueeze(-1).expand(-1, -1, D)  # (B, num_kept, D)
        x_full.scatter_(dim=1, index=kept_indices_expanded, src=x_dropped)

        if path_drop_p > 0:
            drop_mask = torch.rand(B, device=x_full.device) < path_drop_p
            x_full = torch.where(drop_mask[:, None, None], mask_token.expand_as(x_full), x_full)

        return x_full

    def _forward_mmdit(
        self,
        x: Float[Tensor, "batch_size seq_len patch_dim"],
        timesteps: Float[Tensor, "batch_size"],
        initial_context: Any | None = None,
        p: float = 0.0,
        intermediate_features: bool = False,
    ) -> ModelOutput:
        """
        Forward pass through the Sprint mmDiT model.
        Args:
            x (Tensor): Input tensor of shape (B, seq_len, patch_dim)
            timesteps (Tensor): Timestep tensor of shape (B,)
            initial_context (Any, optional): Initial context for the context embedder. Defaults to None.
            p (float, optional): Probability for classifier/path-free guidance. Defaults to 0.0.
            intermediate_features (bool, optional): Whether to return intermediate features. Defaults to False.
        Returns:
            ModelOutput: Output dictionary containing the final output and optional intermediate features.

        """
        assert self.context_embedder is not None, "for MMDiT context embedder must be provided"
        emb = self.time_embed(timestep_embedding(timesteps, self.frequency_embedding))
        context_output: ContextEmbedderOutput = self.context_embedder(initial_context, p)
        if self.pooled_embedding:
            assert self.mlp_pooled_context is not None, (
                "for MMDiT with pooled context, mlp_pooled_context must be defined"
            )
            assert "pooled_embeddings" in context_output, "pooled embeddings must be in context_output"
            context_pooled = context_output["pooled_embeddings"]
            emb = self.mlp_pooled_context(context_pooled) + emb

        context = context_output["embeddings"]
        context = self.context_embed(context)
        attn_mask = context_output.get("attn_mask", None)

        # pos_ids: [S, n_axes] positional IDs along each axis for rope
        # in mmdit attention we concat with context first. Context have 0,0 for h w
        # text: (t>0, 0, 0)
        text_pos_ids = torch.stack(
            [
                torch.arange(1, context.shape[1] + 1, device=x.device),
                torch.zeros(context.shape[1], device=x.device, dtype=torch.long),
                torch.zeros(context.shape[1], device=x.device, dtype=torch.long),
            ],
            dim=-1,
        )

        # image: (0, h, w)
        img_pos_ids = torch.stack(
            torch.meshgrid(
                torch.zeros(1, device=x.device, dtype=torch.long),
                torch.arange(self.grid_size[0], device=x.device),
                torch.arange(self.grid_size[1], device=x.device),
                indexing="ij",
            ),
            dim=-1,
        ).view(-1, 3)

        pos_ids = torch.cat([text_pos_ids, img_pos_ids], dim=0).unsqueeze(0).repeat(x.size(0), 1, 1)
        cos_sin_rope = get_cos_sin_ndim_grid(pos_ids, base=self.rope_base, axes_dim=self.rope_axes_dim)

        features: list[Tensor] | None = [] if intermediate_features else None

        # Pass through each encoder layer sequentially
        for layer in self.layers:
            x, context = layer(x, emb, context, cos_sin_rope=cos_sin_rope, attn_mask=attn_mask)
            if features is not None:
                features.append(x)
        encoder_context = context.clone()

        # drop tokens and forward through deep layers
        cos_sin_rope_img = tuple(rope[:, text_pos_ids.shape[0] :] for rope in cos_sin_rope)
        x_dropped, kept_indices, cos_sin_rope_img_dropped = self.drop_tokens(x, cos_sin_rope=cos_sin_rope_img)
        cos_sin_rope_dropped = tuple(
            torch.cat([rope[:, : text_pos_ids.shape[0]], cos_sin_rope_img_dropped[i]], dim=1)
            for i, rope in enumerate(cos_sin_rope)
        )
        if p < 1:
            for layer in self.deep_layers:
                x_dropped, context = layer(
                    x_dropped, emb, context, cos_sin_rope=cos_sin_rope_dropped, attn_mask=attn_mask
                )  # type: ignore
                if features is not None:
                    features.append(x_dropped)
            x_restored = self.restore_tokens(x_dropped, kept_indices, p)
        else:
            x_restored = self.mask_token.expand_as(x).clone()

        # fuse deep layers output with residual connection from encoder
        x_fused = self.fuse(torch.cat([x_restored, x], dim=-1))
        context_fused = self.fuse_context(torch.cat([context, encoder_context], dim=-1))  # type: ignore

        for layer in self.decoder_layers:
            x_fused, context_fused = layer(
                x_fused,
                emb,
                context_fused,
                cos_sin_rope=cos_sin_rope,
                attn_mask=attn_mask,  # type: ignore
            )
            if features is not None:
                features.append(x_fused)

        x_fused = self.last_layer(x_fused, emb)
        if features is not None:
            features.append(x_fused)

        x_output = self.unpatchify(x_fused)

        output: ModelOutput = {"x": x_output}
        if features is not None:
            output["features"] = features

        return output

    def _forward_dit(
        self,
        x: Float[Tensor, "batch_size seq_len patch_dim"],
        timestep: Float[Tensor, "batch_size"],
        p: float = 0.0,
        y: Int[Tensor, "batch_size"] | None = None,
        intermediate_features: bool = False,
    ) -> ModelOutput:
        """
        Forward pass through the Sprint DiT model.
        Args:
            x (Tensor): Input tensor of shape (B, seq_len, patch_dim)
            timestep (Tensor): Timestep tensor of shape (B,)
            p (float, optional): Probability for classifier/path-free guidance. Defaults to 0.0
            y (Tensor, optional): Class labels. Defaults to None.
            intermediate_features (bool, optional): Whether to return intermediate features. Defaults to False.
        Returns:
            ModelOutput: Output dictionary containing the final output and optional intermediate features.
        """
        if p > 0:
            assert self.n_classes, (
                "probability of dropping for classifier free guidance is only available if a number of classes is set"
            )

        emb = self.time_embed(timestep_embedding(timestep, self.frequency_embedding))
        if self.label_embed is not None:
            emb = emb + self.label_embed(y, p)

        # pos_ids: [S, n_axes] positional IDs along each axis for rope
        pos_ids = (
            torch.stack(
                torch.meshgrid(
                    [
                        torch.arange(self.grid_size[0], device=x.device),
                        torch.arange(self.grid_size[1], device=x.device),
                    ],
                    indexing="ij",
                ),
                dim=-1,
            )
            .view(-1, 2)
            .unsqueeze(0)
            .repeat(x.size(0), 1, 1)
        )
        cos_sin_rope = get_cos_sin_ndim_grid(pos_ids, base=self.rope_base, axes_dim=self.rope_axes_dim)

        features: list[Tensor] | None = [] if intermediate_features else None
        # Pass through each layer sequentially
        for layer in self.layers:
            x = layer(x, emb, cos_sin_rope=cos_sin_rope)
            if features is not None:
                features.append(x)

        x_dropped, kept_indices, cos_sin_rope_dropped = self.drop_tokens(x, cos_sin_rope=cos_sin_rope)

        if p < 1:
            for layer in self.deep_layers:
                x_dropped = layer(x_dropped, emb, cos_sin_rope=cos_sin_rope_dropped)
                if features is not None:
                    features.append(x_dropped)
            x_restored = self.restore_tokens(x_dropped, kept_indices, p)
        else:
            x_restored = self.mask_token.expand_as(x).clone()

        x_fused = self.fuse(torch.cat([x_restored, x], dim=-1))

        for layer in self.decoder_layers:
            x_fused = layer(x_fused, emb, cos_sin_rope=cos_sin_rope)
            if features is not None:
                features.append(x_fused)

        x_fused = self.last_layer(x_fused, emb)
        if features is not None:
            features.append(x_fused)

        x_output = self.unpatchify(x_fused)

        output: ModelOutput = {"x": x_output}
        if features is not None:
            output["features"] = features

        return output

    def forward(
        self,
        x: Float[Tensor, "batch_size channels height width"],
        timesteps: Float[Tensor, "batch_size"],
        initial_context: Any | None = None,
        p: float = 0.0,
        y: Int[Tensor, "batch_size"] | None = None,
        x_context: Tensor | None = None,
        intermediate_features: bool = False,
    ) -> ModelOutput:
        """
        Forward pass through the Sprint model.
        Args:
            x (Tensor): Input tensor of shape (B, C, H, W)
            timesteps (Tensor): Timestep tensor of shape (B,)
            initial_context (Any, optional): Initial context for the context embedder. Defaults to None.
            p (float, optional): Probability for classifier/path-free guidance. Defaults to 0.0
            y (Tensor, optional): Class labels. Defaults to None.
            x_context (Tensor, optional): Additional context tensor to concatenate with input. Defaults to None.
            intermediate_features (bool, optional): Whether to return intermediate features. Defaults to False.
            path_drop_p (float, optional): Probability of dropping the dense path. Defaults to 0.0.
        Returns:
            ModelOutput: Output dictionary containing the final output and optional intermediate features.
        """
        assert not (initial_context is not None and y is not None), "initial_context and y cannot both be specified"
        if p > 0:
            assert self.classifier_free, (
                "probability of dropping for classifier free guidance is only available if model is set up to be classifier free"
            )
        if x_context is not None:
            x = torch.cat([x, x_context], dim=1)

        encoder_input = self.patchify(x)

        if self.simple_dit:
            return self._forward_dit(encoder_input, timesteps, p, y, intermediate_features)

        return self._forward_mmdit(encoder_input, timesteps, initial_context, p, intermediate_features)

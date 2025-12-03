# Recoded from scratch taking inspiration from https://arxiv.org/pdf/2504.05741
# Some changes were made to the architecture to reuse DiT and MMDiT components
# e.g double stream

from typing import Any

import torch
import torch.nn as nn
from einops import rearrange
from jaxtyping import Float, Int
from torch import Tensor

from diffulab.networks.denoisers.common import Denoiser, ModelOutput
from diffulab.networks.denoisers.mmdit import DiTBlock, MMDiTBlock
from diffulab.networks.embedders.common import ContextEmbedder, ContextEmbedderOutput
from diffulab.networks.utils.nn import (
    LabelEmbed,
    get_cos_sin_ndim_grid,
    modulate,
    timestep_embedding,
)


class ModulatedLastLayerDDT(nn.Module):
    def __init__(self, embedding_dim: int, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()  # type: ignore
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(embedding_dim, 2 * hidden_size, bias=True))

    def forward(
        self, x: Float[Tensor, "batch_size seq_len dim"], vec: Float[Tensor, "batch_size seq_len dim"]
    ) -> Tensor:
        alpha, beta = self.adaLN_modulation(vec).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), scale=alpha, shift=beta)
        x = self.linear(x)
        return x


class DDT(Denoiser):
    """
    Decoder-Encoder architecture following https://arxiv.org/pdf/2504.05741

    This module implements a dual-stream DDT: an encoder stream that consumes the noisy input
    (and optional context) and a lightweight decoder stream conditioned on the encoder output
    and timestep. It reuses DiT/MMDiT blocks and supports both label-only and multimodal conditioning.

    Args:
        simple_ddt (bool): If True, uses a DiT-style encoder with class-label conditioning only
            (no multimodal context). If False, uses an MMDiT-style encoder. Default: False.
        input_channels (int): Number of channels of the main input x. Default: 3.
        output_channels (int | None): Number of channels to predict. If None, equals
            `input_channels`. Default: None.
        input_dim (int): Token/patch embedding width for both encoder and decoder streams.
            Also used as the hidden size of the last prediction layer. Default: 768.
        hidden_dim (int): Inner attention dimension used by DiT/MMDiT attention blocks.
            Default: 768.
        num_heads (int): Number of attention heads in each block. Default: 12.
        mlp_ratio (int): Expansion ratio for the MLP in each block. Default: 4.
        patch_size (int): Side length P of square patches. Images are projected with stride P
            in both encoder and decoder streams. Default: 16.
        context_dim (int): Model width of contextual tokens after `context_embed` when
            `simple_ddt=False`. Ignored when `simple_ddt=True`. Default: 1024.
        encoder_depth (int): Number of DiT/MMDiT blocks in the encoder. Default: 8.
        decoder_depth (int): Number of DiT blocks in the decoder. Default: 4.
        partial_rotary_factor (float): Fraction of each head dimension using RoPE.
            1.0 means full rotary. Default: 1.0.
        frequency_embedding (int): Size of the Fourier timestep embedding before the time MLP.
            Default: 256.
        n_classes (int | None): Number of classes for label conditioning in `simple_ddt` mode.
            Required to use classifier-free guidance with labels. Default: None.
        classifier_free (bool): Enables classifier-free guidance. In `simple_ddt`, it applies to
            dropped labels; in MMDiT mode, it is forwarded to the context embedder which may drop
            context. Default: False.
        context_embedder (ContextEmbedder | None): When `simple_ddt=False`, a module returning
            `ContextEmbedderOutput`. Must be provided for text/image conditioning. Must be None
            when `simple_ddt=True`. If the embedder returns pooled and token embeddings
            (`n_output == 2`), pooled features are fused into the timestep embedding via an MLP.
            Default: None.
        use_checkpoint (bool): Enable torch.utils.checkpoint in blocks to trade compute for memory.
            Default: False.
    """

    def __init__(
        self,
        simple_ddt: bool = False,
        input_channels: int = 3,
        output_channels: int | None = None,
        input_dim: int = 768,
        hidden_dim: int = 768,
        num_heads: int = 12,
        mlp_ratio: int = 4,
        patch_size: int = 16,
        context_dim: int = 1024,
        encoder_depth: int = 8,
        decoder_depth: int = 4,
        rope_base: int = 10_000,
        partial_rotary_factor: float = 1,
        rope_axes_dim: list[int] | None = None,
        frequency_embedding: int = 256,
        n_classes: int | None = None,
        classifier_free: bool = False,
        context_embedder: ContextEmbedder | None = None,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        assert not (n_classes is not None and context_embedder is not None), (
            "n_classes and context_embedder cannot both be specified"
        )
        self.simple_ddt = simple_ddt
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

        heads_dim = hidden_dim // num_heads
        if not self.simple_ddt:
            assert self.context_embedder is not None, "for ddt with text context embedder must be provided"
            assert isinstance(self.context_embedder.output_size, tuple) and all(
                isinstance(i, int) for i in self.context_embedder.output_size
            ), "context_embedder.output_size must be a tuple of integers"

            self.pooled_embedding = False
            self.mlp_pooled_context = None
            if self.context_embedder.n_output == 2:
                self.pooled_embedding = True
                self.mlp_pooled_context = nn.Sequential(
                    nn.Linear(self.context_embedder.output_size[0], input_dim),
                    nn.SiLU(),
                    nn.Linear(input_dim, input_dim),
                )
                self.context_embed = nn.Linear(self.context_embedder.output_size[1], context_dim)
            else:
                assert self.context_embedder.n_output == 1
                self.context_embed = nn.Linear(self.context_embedder.output_size[0], context_dim)
            if rope_axes_dim is None:
                rope_axes_dim = [
                    int((partial_rotary_factor * heads_dim) // 3),  # L for text, set to 0 for image tokens
                    int((partial_rotary_factor * heads_dim) // 3),  # H set to 0 for text
                    int((partial_rotary_factor * heads_dim) // 3),  # W set to 0 for text
                ]
        else:
            self.label_embed = (
                LabelEmbed(self.n_classes, input_dim, self.classifier_free) if self.n_classes is not None else None
            )
            if rope_axes_dim is None:
                rope_axes_dim = [
                    int((partial_rotary_factor * heads_dim) // 2),  # H
                    int((partial_rotary_factor * heads_dim) // 2),  # W
                ]

        self.rope_axes_dim = rope_axes_dim
        self.last_layer = ModulatedLastLayerDDT(
            embedding_dim=input_dim,
            hidden_size=input_dim,
            patch_size=self.patch_size,
            out_channels=self.output_channels,
        )
        self.time_embed = nn.Sequential(
            nn.Linear(self.frequency_embedding, input_dim),
            nn.SiLU(),
            nn.Linear(input_dim, input_dim),
        )

        self.conv_proj_encoder = nn.Conv2d(
            self.input_channels, input_dim, kernel_size=self.patch_size, stride=self.patch_size
        )
        self.conv_proj_decoder = nn.Conv2d(
            self.input_channels, input_dim, kernel_size=self.patch_size, stride=self.patch_size
        )

        # --------------
        # encoder layers
        # --------------
        self.layers = nn.ModuleList(
            [
                MMDiTBlock(
                    context_dim=context_dim,
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    embedding_dim=input_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    rope_axes_dim=self.rope_axes_dim,
                    use_checkpoint=use_checkpoint,
                )
                if not self.simple_ddt
                else DiTBlock(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    embedding_dim=input_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    rope_axes_dim=self.rope_axes_dim,
                    use_checkpoint=use_checkpoint,
                )
                for _ in range(encoder_depth)
            ]
        )

        # --------------
        # decoder layers
        # --------------
        self.decoder_layers = nn.ModuleList(
            [
                DiTBlock(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    embedding_dim=input_dim,
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

    def patchify(
        self, x: Float[Tensor, "batch_size channels height width"], encoder: bool = True
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

        x = self.conv_proj_encoder(x) if encoder else self.conv_proj_decoder(x)
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

    def encode_mmddt(
        self,
        x: Float[Tensor, "batch_size seq_len patch_dim"],
        timesteps: Float[Tensor, "batch_size"],
        initial_context: Any | None = None,
        p: float = 0.0,
        intermediate_features: bool = False,
    ) -> ModelOutput:
        """
        Forward pass through the encoder of the mmDDT model.
        Args:
            x (Tensor): Input tensor of shape (B, seq_len, patch_dim)
            timesteps (Tensor): Timestep tensor of shape (B,)
            initial_context (Any, optional): Initial context for the context embedder. Defaults to None.
            p (float, optional): Probability for classifier-free guidance. Defaults to 0.0.
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
                torch.arange(1, context.shape[1], device=x.device),
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

        pos_ids = torch.cat([text_pos_ids, img_pos_ids], dim=0)
        cos_sin_rope = get_cos_sin_ndim_grid(pos_ids, base=self.rope_base, axes_dim=self.rope_axes_dim)

        features: list[Tensor] | None = [] if intermediate_features else None
        # Pass through each layer sequentially
        for layer in self.layers:
            x, context = layer(x, emb, context, cos_sin_rope=cos_sin_rope, attn_mask=attn_mask)
            if features is not None:
                features.append(x)

        encoder_output: ModelOutput = {"x": x}
        if features:
            encoder_output["features"] = features
        return encoder_output

    def encode_ddt(
        self,
        x: Float[Tensor, "batch_size seq_len patch_dim"],
        timestep: Float[Tensor, "batch_size"],
        p: float = 0.0,
        y: Int[Tensor, "batch_size"] | None = None,
        intermediate_features: bool = False,
    ) -> ModelOutput:
        """
        Forward pass through the encoder of the DDT model.
        Args:
            x (Tensor): Input tensor of shape (B, seq_len, patch_dim)
            timestep (Tensor): Timestep tensor of shape (B,)
            p (float, optional): Probability for classifier-free guidance. Defaults to 0.0
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
        pos_ids = torch.stack(
            torch.meshgrid(
                [torch.arange(self.grid_size[0], device=x.device), torch.arange(self.grid_size[1], device=x.device)],
                indexing="ij",
            ),
            dim=-1,
        ).view(-1, 2)
        cos_sin_rope = get_cos_sin_ndim_grid(pos_ids, base=self.rope_base, axes_dim=self.rope_axes_dim)

        features: list[Tensor] | None = [] if intermediate_features else None
        # Pass through each layer sequentially
        for layer in self.layers:
            x = layer(x, emb, cos_sin_rope=cos_sin_rope)
            if features is not None:
                features.append(x)

        encoder_output: ModelOutput = {"x": x}
        if features:
            encoder_output["features"] = features
        return encoder_output

    def decode(
        self,
        x: Float[Tensor, "batch_size seq_len patch_dim"],
        encoder_output: Float[Tensor, "batch_size seq_len patch_dim"],
        timesteps: Float[Tensor, "batch_size"],
        intermediate_features: bool = False,
    ) -> ModelOutput:
        """
        Forward pass through the decoder of the DDT model.
        Args:
            x (Tensor): Input tensor of shape (B, seq_len, patch_dim)
            encoder_output (Tensor): Encoder output tensor of shape (B, seq_len, patch_dim)
            timesteps (Tensor): Timestep tensor of shape (B,)
            intermediate_features (bool, optional): Whether to return intermediate features. Defaults to False.
        Returns:
            ModelOutput: Output dictionary containing the final output and optional intermediate features.
        """
        emb = self.time_embed(timestep_embedding(timesteps, self.frequency_embedding))[:, None, :]
        encoder_output = nn.functional.silu(encoder_output + emb)

        # pos_ids: [S, n_axes] positional IDs along each axis for rope
        pos_ids = torch.stack(
            torch.meshgrid(
                [torch.arange(self.grid_size[0], device=x.device), torch.arange(self.grid_size[1], device=x.device)],
                indexing="ij",
            ),
            dim=-1,
        ).view(-1, 2)
        cos_sin_rope = get_cos_sin_ndim_grid(pos_ids, base=self.rope_base, axes_dim=self.rope_axes_dim)

        features: list[Tensor] | None = [] if intermediate_features else None
        for layer in self.decoder_layers:
            x = layer(x, encoder_output, cos_sin_rope=cos_sin_rope)
            if features is not None:
                features.append(x)

        x = self.last_layer(x, encoder_output)

        decoder_output: ModelOutput = {"x": x}
        if features:
            decoder_output["features"] = features
        return decoder_output

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
        Forward pass through the DDT model.
        Args:
            x (Tensor): Input tensor of shape (B, C, H, W)
            timesteps (Tensor): Timestep tensor of shape (B,)
            initial_context (Any, optional): Initial context for the context embedder. Defaults to None.
            p (float, optional): Probability for classifier-free guidance. Defaults to 0.0
            y (Tensor, optional): Class labels. Defaults to None.
            x_context (Tensor, optional): Additional context tensor to concatenate with input. Defaults to None.
            intermediate_features (bool, optional): Whether to return intermediate features. Defaults to False.
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

        encoder_input = self.patchify(x, encoder=True)
        if self.simple_ddt:
            encoder_output = self.encode_ddt(encoder_input, timesteps, p, y, intermediate_features)
        else:
            encoder_output = self.encode_mmddt(encoder_input, timesteps, initial_context, p, intermediate_features)

        decoder_input = self.patchify(x, encoder=False)
        decoder_output = self.decode(decoder_input, encoder_output["x"], timesteps, intermediate_features)

        decoder_output["x"] = self.unpatchify(decoder_output["x"])

        if "features" in encoder_output:
            if "features" in decoder_output:
                decoder_output["features"] = encoder_output["features"] + decoder_output["features"]

        return decoder_output

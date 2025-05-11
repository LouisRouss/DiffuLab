# See https://arxiv.org/abs/2504.05741
# Some parts of the code are adapted from the official DDT implementation:
# https://github.com/MCG-NJU/DDT (no license specified)

from typing import Any

import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor

from diffulab.networks.denoisers.mmdit import DiTBlock, MMDiT
from diffulab.networks.embedders.common import ContextEmbedder
from diffulab.networks.utils.nn import timestep_embedding


class MMDDT(MMDiT):
    """
    Architecture for the ddt model following https://arxiv.org/abs/2504.05741.
    Adding the possibility to train with text as the mmdit model.
    """

    def __init__(
        self,
        simple_ddt: bool = False,
        input_channels: int = 3,
        output_channels: int | None = None,
        input_dim: int = 1152,
        hidden_dim: int = 1152,
        embedding_dim: int = 1152,
        num_heads: int = 16,
        mlp_ratio: int = 4,
        patch_size: int = 2,
        depth_encoder: int = 22,
        depth_decoder: int = 6,
        context_dim: int = 1152,
        n_classes: int | None = None,
        classifier_free: bool = False,
        context_embedder: ContextEmbedder | None = None,
    ):
        super().__init__(
            simple_dit=simple_ddt,
            input_channels=input_channels,
            output_channels=output_channels,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            patch_size=patch_size,
            depth=depth_encoder,
            context_dim=context_dim,
            n_classes=n_classes,
            classifier_free=classifier_free,
            context_embedder=context_embedder,
        )
        self.simple_ddt = simple_ddt
        self.depth_encoder = depth_encoder
        self.depth_decoder = depth_decoder
        self.depth = depth_encoder + depth_decoder

        self.decoder_layers = nn.ModuleList(
            [
                DiTBlock(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(depth_decoder)
            ]
        )
        self.decoder_layers.apply(self._init_weights)

    def simple_ddt_encode(
        self,
        x: Float[Tensor, "batch_size seq_len patch_dim"],
        timestep: Float[Tensor, "batch_size"],
        p: float = 0.0,
        y: Int[Tensor, "batch_size"] | None = None,
    ) -> Tensor:
        """
        Encode input patchified tensor using a simple Decoupled Diffusion Transformer approach.
        Args:
            x (Tensor): Input patchified tensor of shape [batch_size, seq_len, patch_dim].
            timestep (Tensor): Timestep tensor of shape [batch_size].
            p (float, optional): Probability of dropping class labels for classifier-free guidance. Defaults to 0.0.
            y (Tensor, optional): Class labels tensor of shape [batch_size]. Defaults to None.
        Returns:
            Tensor: Encoded tensor with the same shape as input, with time embeddings added.
        """

        if p > 0:
            assert self.n_classes, (
                "probability of dropping for classifier free guidance is only available if a number of classes is set"
            )
        t_emb: Tensor = self.time_embed(timestep_embedding(timestep, self.input_dim))
        emb = t_emb + self.label_embed(y, p) if self.label_embed is not None else t_emb
        # Pass through each layer sequentially
        for layer in self.layers:
            x = layer(x, emb)
        return x + t_emb.view(x.shape[0], -1, x.shape[-1])

    def mmddt_encode(
        self,
        x: Float[Tensor, "batch_size seq_len patch_dim"],
        timesteps: Float[Tensor, "batch_size"],
        initial_context: Any | None = None,
        p: float = 0.0,
    ) -> Tensor:
        """
        Encode input patchified tensor using a MMDiT approach.
        Args:
            x (Tensor): Input patchified tensor with shape [batch_size, seq_len, patch_dim]
            timesteps (Tensor): Timestep values for diffusion process with shape [batch_size]
            initial_context (Any, optional): Optional context information to condition the model
            p (float, optional): Dropout probability for context processing, default is 0.0
        Returns:
            Encoded tensor with the same shape as input x
        Note:
            This method requires a context_embedder to be defined for the model
        """
        assert self.context_embedder is not None, "for MMDiT context embedder must be provided"
        t_emb: Tensor = self.time_embed(timestep_embedding(timesteps, self.input_dim))
        context_pooled, context = self.context_embedder(initial_context, p)
        context_pooled = self.mlp_pooled_context(context_pooled) + t_emb
        context = self.context_embed(context)
        # Pass through each layer sequentially
        for layer in self.layers:
            x, context = layer(x, context_pooled, context)
        return x + t_emb.view(x.shape[0], -1, x.shape[-1])

    def forward(
        self,
        x: Float[Tensor, "batch_size channels height width"],
        timesteps: Float[Tensor, "batch_size"],
        initial_context: Any | None = None,
        p: float = 0.0,
        y: Int[Tensor, "batch_size"] | None = None,
        x_context: Tensor | None = None,
        encoder_features: Tensor | None = None,
    ) -> Tensor:
        """
        Forward pass through the DDT architecture.
        Args:
            x (Tensor): Input image tensor of shape [batch_size, channels, height, width].
            timesteps (Tensor): Timestep tensor of shape [batch_size].
            initial_context (Any, optional): Optional initial context for MMDDT encoding.
            p (float, optional): Probability of dropping context for classifier-free guidance. Defaults to 0.0.
            y (Tensor, optional): Optional class labels of shape [batch_size]. Used for conditional generation.
            x_context (Tensor, optional): Optional context tensor to concatenate with input x. Can be B&W image for coloriation,
                low rresolution image for super-resolution, etc...
            encoder_features (Tensor, optional): Optional pre-computed encoder features.
        Returns:
            Tensor: Denoised output tensor of the same shape as input x.
        """
        assert not (initial_context is not None and y is not None), "initial_context and y cannot both be specified"
        if p > 0:
            assert self.classifier_free, (
                "probability of dropping for classifier free guidance is only available if model is set up to be classifier free"
            )
        if x_context is not None:
            x = torch.cat([x, x_context], dim=1)

        x = self.patchify(x)
        if not encoder_features:
            encoder_features = (
                self.simple_ddt_encode(x, timesteps, p=p, y=y)
                if self.simple_ddt
                else self.mmddt_encode(x, timesteps, initial_context=initial_context, p=p)
            )

        for layer in self.decoder_layers:
            x = layer(x, encoder_features)
        x = self.last_layer(x, encoder_features)
        x = self.unpatchify(x)
        return x

# See https://arxiv.org/abs/2504.05741
# Some parts of the code are adapted from the official DDT implementation:
# https://github.com/MCG-NJU/DDT (no license specified)
from typing import Any

import torch.nn as nn
from einops import rearrange
from jaxtyping import Float, Int
from torch import Tensor

from diffulab.networks.denoisers.common import Denoiser
from diffulab.networks.denoisers.mmdit import DiTBlock, MMDiTBlock, ModulatedLastLayer
from diffulab.networks.embedders.common import ContextEmbedder
from diffulab.networks.utils.nn import LabelEmbed, timestep_embedding


class MMDDT(Denoiser):
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
        depth_encoder: int = 4,
        depth_decoder: int = 24,
        context_dim: int = 1152,
        n_classes: int | None = None,
        classifier_free: bool = False,
        context_embedder: ContextEmbedder | None = None,
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
        self.depth_encoder = depth_encoder
        self.depth_decoder = depth_decoder
        self.depth = depth_encoder + depth_decoder

        self.n_classes = n_classes
        self.classifier_free = classifier_free

        if not self.simple_ddt:
            assert self.context_embedder is not None, "for mmDDT context embedder must be provided"
            assert self.context_embedder.n_output == 2, "for mmDDT context embedder should provide 2 embeddings"
            assert isinstance(self.context_embedder.output_size, tuple) and all(
                isinstance(i, int) for i in self.context_embedder.output_size
            ), "context_embedder.output_size must be a tuple of integers"
            self.mlp_pooled_context = nn.Sequential(
                nn.Linear(self.context_embedder.output_size[0], embedding_dim),
                nn.SiLU(),
                nn.Linear(embedding_dim, embedding_dim),
            )
            self.context_embed = nn.Linear(self.context_embedder.output_size[1], context_dim)
        else:
            self.label_embed = (
                LabelEmbed(self.n_classes, embedding_dim, self.classifier_free) if self.n_classes is not None else None
            )

        self.last_layer = ModulatedLastLayer(
            hidden_size=input_dim, patch_size=self.patch_size, out_channels=self.output_channels
        )

        self.input_dim = input_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.input_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

        self.conv_proj = nn.Conv2d(self.input_channels, input_dim, kernel_size=self.patch_size, stride=self.patch_size)

        self.encoder_layers = nn.ModuleList([
            MMDiTBlock(
                context_dim=context_dim,
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
            )
            if not self.simple_dit
            else DiTBlock(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
            )
            for _ in range(depth_encoder)
        ])

        self.decoder_layers = nn.ModuleList([
            DiTBlock(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
            )
            for _ in range(depth_decoder)
        ])
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:  # type: ignore
                nn.init.constant_(module.bias, 0)

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
        H, W = self.original_size
        patch_size = self.patch_size
        p = self.output_channels

        # Calculate number of patches in height and width dimensions
        h = H // patch_size
        w = W // patch_size

        # Reshape the tensor to the original image dimensions
        x = rearrange(x, "b (h w) (p1 p2 c) -> b c (h p1) (w p2)", h=h, w=w, p1=patch_size, p2=patch_size, c=p)
        return x

    def simple_ddt_encode(
        self,
        x: Float[Tensor, "batch_size seq_len patch_dim"],
        timestep: Float[Tensor, "batch_size"],
        p: float = 0.0,
        y: Int[Tensor, "batch_size"] | None = None,
    ) -> Tensor:
        if p > 0:
            assert self.n_classes, (
                "probability of dropping for classifier free guidance is only available if a number of classes is set"
            )

        t_emb = self.time_embed(timestep_embedding(timestep, self.input_dim))
        emb = t_emb + self.label_embed(y, p) if self.label_embed is not None else t_emb

        # Pass through each layer sequentially
        for layer in self.encoder_layers:
            x = layer(x, emb)

        return nn.functional.silu(x + t_emb)

    def mmddt_encode(
        self,
        x: Float[Tensor, "batch_size seq_len patch_dim"],
        timesteps: Float[Tensor, "batch_size"],
        initial_context: Any | None = None,
        p: float = 0.0,
    ) -> Tensor:
        assert self.context_embedder is not None, "for MMDiT context embedder must be provided"
        t_emb = self.time_embed(timestep_embedding(timesteps, self.input_dim))
        context_pooled, context = self.context_embedder(initial_context, p)
        context_pooled = self.mlp_pooled_context(context_pooled) + t_emb
        context = self.context_embed(context)
        # Pass through each layer sequentially
        for layer in self.encoder_layers:
            x, context = layer(x, context_pooled, context)

        return nn.functional.silu(x + t_emb)

    def forward(
        self,
        x: Float[Tensor, "batch_size channels height width"],
        timesteps: Float[Tensor, "batch_size"],
        initial_context: Any | None = None,
        p: float = 0.0,
        y: Int[Tensor, "batch_size"] | None = None,
        x_context: Tensor | None = None,
        encoder_features: Tensor | None = None,
    ) -> Tensor: ...

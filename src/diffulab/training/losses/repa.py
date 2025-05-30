from typing import Any

import torch
from jaxtyping import Float
from torch import Tensor, nn

from diffulab.networks.denoisers.mmdit import MMDiT
from diffulab.networks.repa.common import REPA
from diffulab.networks.repa.dinov2 import DinoV2
from diffulab.training.losses.common import LossFunction


class RepaLoss(LossFunction):
    encoder_registry: dict[str, type[REPA]] = {"dinov2": DinoV2}

    def __init__(
        self,
        denoiser: MMDiT,  # maybe allow other denoisers in the future
        repa_encoder: str = "dinov2",
        encoder_args: dict[str, Any] = {},
        alignment_layer: int = 8,
        denoiser_dimension: int = 256,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()

        assert repa_encoder in self.encoder_registry, (
            f"Encoder {repa_encoder} is not supported. Available encoders: {list(self.encoder_registry.keys())}"
        )

        self.denoiser = denoiser
        self.repa_encoder = self.encoder_registry[repa_encoder](**encoder_args)

        self.proj = nn.Sequential(
            nn.Linear(denoiser_dimension, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.repa_encoder.embedding_dim),
        )
        self.denoiser.layers[alignment_layer - 1].register_forward_hook(self._forward_hook)
        self.src_features: Tensor | None = None

    def _forward_hook(self, net: nn.Module, input: tuple[Any, ...], output: Tensor) -> None:
        """
        Hook to capture the output of the specified layer during the forward pass.
        """
        self.src_features = output

    def forward(self, x0: Float[Tensor, "batch 3 H W"]) -> Tensor:
        assert self.src_features is not None, "Source features are not computed. Ensure the forward hook is registered."
        with torch.no_grad():
            dst_features = self.repa_encoder(
                x0
            )  # batch size seqlen embedding_dim # SEE HOW TO HANDLE THE PRE COMPUTING OF FEATURES
        cos_sim = torch.nn.functional.cosine_similarity(self.proj(self.src_features), dst_features, dim=-1)
        loss = 1 - cos_sim.mean()
        return loss

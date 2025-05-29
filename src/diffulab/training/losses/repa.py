from typing import Any

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
        layer_dimension: int = 256,
    ) -> None:
        super().__init__()

        assert repa_encoder in self.encoder_registry, (
            f"Encoder {repa_encoder} is not supported. Available encoders: {list(self.encoder_registry.keys())}"
        )

        self.denoiser = denoiser
        self.repa_encoder = self.encoder_registry[repa_encoder](**encoder_args)

        self.proj = nn.Linear(layer_dimension, self.repa_encoder.embedding_dim)
        self.denoiser.layers[alignment_layer - 1].register_forward_hook(self._forward_hook)

    def _forward_hook(self, net: nn.Module, input: tuple[Any, ...], output: Tensor) -> None:
        """
        Hook to capture the output of the specified layer during the forward pass.
        """
        projected_output = self.proj(output)
        self._repa_output = projected_output

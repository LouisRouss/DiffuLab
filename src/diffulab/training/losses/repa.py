from typing import Any, TypedDict

import torch
from jaxtyping import Float
from torch import Tensor, nn

from diffulab.networks.denoisers.mmdit import MMDiT
from diffulab.networks.repa.common import REPA
from diffulab.networks.repa.dinov2 import DinoV2
from diffulab.networks.repa.perceiver_resampler import PerceiverResampler
from diffulab.training.losses.common import LossFunction


class ResamplerParams(TypedDict):
    dim: int
    depth: int
    head_dim: int
    num_heads: int
    ff_mult: int
    num_latents: int


class RepaLoss(LossFunction):
    encoder_registry: dict[str, type[REPA]] = {"dinov2": DinoV2}

    def __init__(
        self,
        denoiser: MMDiT,
        repa_encoder: str = "dinov2",
        encoder_args: dict[str, Any] = {},
        alignment_layer: int = 8,
        denoiser_dimension: int = 256,
        hidden_dim: int = 256,
        load_dino: bool = True,  # whether to load the DINO model weights, if precomputed features are used no need to load it
        embedding_dim: int = 768,  # dimension of the DINO features
        use_resampler: bool = False,  # whether to use the perceiver resampler
        resampler_params: ResamplerParams | None = None,
        coeff: float = 1.0,  # weight for the loss
    ) -> None:
        super().__init__()

        assert repa_encoder in self.encoder_registry, (
            f"Encoder {repa_encoder} is not supported. Available encoders: {list(self.encoder_registry.keys())}"
        )
        if not isinstance(denoiser, MMDiT):  # type: ignore[reportUnnecessaryIsInstance]
            raise TypeError(
                f"Denoiser must be an instance of MMDiT, got {type(denoiser)} instead. REPA isn't implemented for other denoisers yet."
            )

        self.denoiser = denoiser
        self.repa_encoder: REPA | None = None
        if load_dino:
            self.repa_encoder = self.encoder_registry[repa_encoder](**encoder_args)

        self.proj = nn.Sequential(
            nn.Linear(denoiser_dimension, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.repa_encoder.embedding_dim if self.repa_encoder else embedding_dim),
        )

        self.resampler: PerceiverResampler | None = None
        if use_resampler:
            assert resampler_params is not None, (
                "Resampler parameters must be provided when using the perceiver resampler."
            )
            self.resampler = PerceiverResampler(
                **resampler_params,
            )
        self.alignment_layer = alignment_layer
        self.hook_handle = None
        self._register_hook(self.denoiser)
        self.src_features: Tensor | None = None
        self.coeff = coeff

    def _register_hook(self, model: MMDiT) -> None:
        """Register the forward hook on the specified layer of the model."""
        self._unregister_hook()  # Ensure no previous hook is registered
        self.hook_handle = model.layers[self.alignment_layer - 1].register_forward_hook(self._forward_hook)

    def _unregister_hook(self) -> None:
        """Remove the forward hook."""
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None

    def set_model(self, model: MMDiT) -> None:
        """Switch the hook to a different model (e.g., EMA model)."""
        self._register_hook(model)

    def _forward_hook(self, net: nn.Module, input: tuple[Any, ...], output: Tensor) -> None:
        """
        Hook to capture the output of the specified layer during the forward pass.
        """
        self.src_features = output

    def forward(
        self,
        x0: Float[Tensor, "batch 3 H W"] | None = None,
        dst_features: Float[Tensor, "batch seq_len n_dim"] | None = None,
    ) -> Tensor:
        assert self.src_features is not None, "Source features are not computed. Ensure the forward hook is registered."
        assert x0 is not None or dst_features is not None, "Either x0 or dst_features must be provided."
        if dst_features is None:
            assert self.repa_encoder is not None, "REPA encoder must be initialized to compute features."
            with torch.no_grad():
                dst_features = self.repa_encoder(
                    x0
                )  # batch size seqlen embedding_dim # SEE HOW TO HANDLE THE PRE COMPUTING OF FEATURES
        assert dst_features is not None, "Destination features must be provided or computed."

        projected_src_features: Tensor = self.proj(self.src_features)

        if self.resampler is not None:
            projected_src_features = self.resampler(projected_src_features)

        cos_sim = torch.nn.functional.cosine_similarity(projected_src_features, dst_features, dim=-1)  # type: ignore
        loss = 1 - cos_sim.mean()
        return self.coeff * loss

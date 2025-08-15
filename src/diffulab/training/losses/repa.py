from typing import Any, Callable, TypedDict
from weakref import WeakKeyDictionary

import torch
from jaxtyping import Float
from torch import Tensor, nn
from torch.utils.hooks import RemovableHandle

from diffulab.networks.denoisers.mmdit import MMDiT
from diffulab.networks.repa.common import REPA
from diffulab.networks.repa.dinov2 import DinoV2
from diffulab.networks.repa.perceiver_resampler import PerceiverResampler
from diffulab.training.losses.common import LossFunction

try:
    from torch._dynamo import disable as _dynamo_disable  # type: ignore
except Exception:

    def _dynamo_disable(fn: Any) -> Any:
        return fn


class ResamplerParams(TypedDict):
    dim: int
    depth: int
    head_dim: int
    num_heads: int
    ff_mult: int
    num_latents: int


class RepaLoss(LossFunction):
    encoder_registry: dict[str, type[REPA]] = {"dinov2": DinoV2}
    name: str = "RepaLoss"

    def __init__(
        self,
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
        self._handles: "WeakKeyDictionary[nn.Module, RemovableHandle]" = WeakKeyDictionary()
        self._features: "WeakKeyDictionary[nn.Module, torch.Tensor]" = WeakKeyDictionary()
        self._active_model: nn.Module | None = None
        self._hook_layer_idx = self.alignment_layer - 1  # as before
        self.coeff = coeff

    @_dynamo_disable
    def _make_forward_hook(self, key_model: MMDiT) -> Callable[[nn.Module, tuple[Any, ...], torch.Tensor], None]:
        def _hook(_mod: nn.Module, _inp: tuple[Any, ...], out: torch.Tensor):
            self._features[key_model] = out

        return _hook

    def _attach_once(self, model: MMDiT) -> None:
        if model in self._handles:
            return
        layer = model.layers[self._hook_layer_idx]
        handle = layer.register_forward_hook(self._make_forward_hook(model))  # type: ignore
        self._handles[model] = handle

    def set_model(self, model: MMDiT) -> None:  # type: ignore
        """
        Select which model's captured features to use. If we haven't attached to this model yet,
        attach exactly once and keep the handle for the lifetime of the process.
        """
        self._attach_once(model)
        self._active_model = model

    def _unregister_all(self) -> None:
        for h in list(self._handles.values()):
            h.remove()
        self._handles.clear()
        self._features.clear()
        self._active_model = None

    def forward(
        self,
        x0: Float[Tensor, "batch 3 H W"] | None = None,
        dst_features: Float[Tensor, "batch seq_len n_dim"] | None = None,
    ) -> Tensor:
        if self._active_model is None or self._active_model not in self._features:
            raise RuntimeError(
                "REPA: no captured features for the active model. Did you call set_model(...) and run a forward pass?"
            )
        assert x0 is not None or dst_features is not None, "Either x0 or dst_features must be provided."
        if dst_features is None:
            assert self.repa_encoder is not None, "REPA encoder must be initialized to compute features."
            with torch.no_grad():
                dst_features = self.repa_encoder(
                    x0
                )  # batch size seqlen embedding_dim # SEE HOW TO HANDLE THE PRE COMPUTING OF FEATURES
        assert dst_features is not None, "Destination features must be provided or computed."

        src_features = self._features[self._active_model]
        projected_src_features: Tensor = self.proj(src_features)

        if self.resampler is not None:
            projected_src_features = self.resampler(projected_src_features)

        cos_sim = torch.nn.functional.cosine_similarity(projected_src_features, dst_features, dim=-1)  # type: ignore
        loss = 1 - cos_sim.mean()
        return self.coeff * loss

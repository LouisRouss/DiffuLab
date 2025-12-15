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
    """Representation Alignment (REPA) loss.

    Aligns intermediate features from a denoiser (MMDiT) to features from an
    external vision encoder (e.g., DINOv2) using a projection MLP and, optionally,
    a Perceiver resampler. Denoiser features are captured via a forward hook on a
    specified transformer block and compared to encoder features using cosine
    similarity. The loss is averaged over the sequence dimension and scaled by
    ``coeff``.

    Typical usage:
        loss_fn = RepaLoss(...)
        loss_fn.set_model(denoiser)
        # Run a forward pass through the denoiser to populate captured features
        loss = loss_fn(x0=batch_images)  # or pass dst_features=...

    Args:
        repa_encoder: Key of the encoder to instantiate. Supported values are
            keys of ``encoder_registry``, e.g. "dinov2".
        encoder_args: Keyword arguments forwarded to the encoder constructor.
        alignment_layer: 1-based index of the MMDiT layer from which to capture
            features.
        denoiser_dimension: Feature dimensionality of the denoiser at the
            alignment layer.
        hidden_dim: Hidden size of the projection MLP.
        load_dino: Whether to instantiate and load the encoder. Set to ``False``
            when precomputed ``dst_features`` will be supplied at call time.
        embedding_dim: Target embedding dimensionality when the encoder is not
            instantiated (i.e., when ``load_dino=False``).
        use_resampler: Whether to apply a :class:`PerceiverResampler` after the
            projection MLP.
        resampler_params: Configuration for the :class:`PerceiverResampler`.
            Required if ``use_resampler=True``.
        coeff: Multiplicative weight applied to the returned loss value.

    Attributes:
        repa_encoder: The instantiated encoder or ``None`` if
            ``load_dino=False``.
        proj: Projection MLP mapping denoiser features to the encoder embedding
            space.
        resampler: Optional :class:`PerceiverResampler` applied after the
            projection.
        alignment_layer: 1-based index of the hooked MMDiT layer.
        coeff: Multiplicative weight applied to the returned loss.
    """

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
        self._hook_layer_idx = self.alignment_layer - 1
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
        """Register the model to capture features from a specific layer.

        This attaches a forward hook to the specified ``alignment_layer`` of the
        provided model (only once). A forward pass on ``model`` must be executed
        after calling this method so that features are captured before computing
        the loss.

        Args:
            model (MMDiT): The model whose intermediate features will be
                aligned to the encoder features.
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
        """Compute the REPA cosine-similarity loss.

        Either provide input images via ``x0`` to compute destination features
        with the encoder, or pass precomputed ``dst_features`` directly.

        Args:
            x0 (Tensor): Input images of shape ``[B, 3, H, W]`` used to compute encoder
                features when an encoder is available.
            dst_features (Tensor): Precomputed encoder features of shape ``[B, S, D]``.
                If provided, ``x0`` is ignored.

        Returns:
            Tensor: A scalar tensor containing the REPA loss.

        Raises:
            RuntimeError: If no captured features are available for the active
                model. Ensure ``set_model(...)`` was called and a forward pass
                on the model was executed first.
            AssertionError: If neither ``x0`` nor ``dst_features`` is provided.
        """
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
        if isinstance(src_features, tuple):
            src_features = src_features[0]
        projected_src_features: Tensor = self.proj(src_features)

        if self.resampler is not None:
            projected_src_features = self.resampler(projected_src_features)

        cos_sim = torch.nn.functional.cosine_similarity(projected_src_features, dst_features, dim=-1)  # type: ignore
        loss = 1 - cos_sim.mean()  # type: ignore
        return self.coeff * loss  # type: ignore

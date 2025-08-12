from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict, cast

import torch
from accelerate import Accelerator  # type: ignore
from jaxtyping import Float
from torch import Tensor, nn

from diffulab.networks.denoisers.mmdit import MMDiT
from diffulab.networks.repa.common import REPA
from diffulab.networks.repa.dinov2 import DinoV2
from diffulab.networks.repa.perceiver_resampler import PerceiverResampler
from diffulab.training.losses.common import LossFunction

if TYPE_CHECKING:
    from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel
    from torch.nn.parallel import DistributedDataParallel


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
        self._hook_handle = None
        self.src_features: Tensor | None = None
        self.coeff = coeff

    def _register_hook(self, model: MMDiT) -> None:
        """Register the forward hook on the specified layer of the model."""
        self._unregister_hook()  # Ensure no previous hook is registered
        self._hook_handle = model.layers[self.alignment_layer - 1].register_forward_hook(self._forward_hook)

    def _unregister_hook(self) -> None:
        """Remove the forward hook."""
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    def set_model(self, model: MMDiT) -> None:
        """Switch the hook to a different model (e.g., EMA model)."""
        self._register_hook(model)

    def _forward_hook(self, net: nn.Module, input: tuple[Any, ...], output: Tensor) -> None:
        """
        Hook to capture the output of the specified layer during the forward pass.
        """
        self.src_features = output

    def save(self, path: str | Path, accelerator: Accelerator) -> None:
        """
        Save state dict containing projection (and resampler if present).

        Args:
            path (str | Path): Path to save the loss function.
            accelerator (Accelerator | None): Accelerator instance for distributed training. Uses
                accelerator.save if provided.
        """
        file_path = Path(path) / "RepaLoss.pt"

        unwrapped_proj = cast(nn.Module, accelerator.unwrap_model(self.proj))  # type: ignore
        merged_state = {}
        for k, v in unwrapped_proj.state_dict().items():
            merged_state[f"proj.{k}"] = v
        if self.resampler is not None:
            unwrapped_resampler = cast(nn.Module, accelerator.unwrap_model(self.resampler))  # type: ignore
            for k, v in unwrapped_resampler.state_dict().items():
                merged_state[f"resampler.{k}"] = v

        accelerator.save(merged_state, file_path)  # type: ignore

    def accelerate_prepare(
        self, accelerator: Accelerator
    ) -> "list[nn.Module | DistributedDataParallel | FullyShardedDataParallel]":
        """
        Prepare the loss function for distributed training.

        Args:
            accelerator (Accelerator): Accelerator instance for distributed training.
        """
        trainable_modules: "list[nn.Module | DistributedDataParallel | FullyShardedDataParallel]" = []
        self.proj = accelerator.prepare_model(self.proj)  # type: ignore
        trainable_modules.append(self.proj)  # type: ignore
        if self.resampler is not None:
            self.resampler = accelerator.prepare_model(self.resampler)  # type: ignore
            trainable_modules.append(self.resampler)  # type: ignore
        if self.repa_encoder is not None:
            self.repa_encoder = accelerator.prepare_model(self.repa_encoder)  # type: ignore

        return trainable_modules

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

        projected_src_features: Tensor = self.proj(self.src_features)  # type: ignore

        if self.resampler is not None:
            projected_src_features = self.resampler(projected_src_features)

        cos_sim = torch.nn.functional.cosine_similarity(projected_src_features, dst_features, dim=-1)  # type: ignore
        loss = 1 - cos_sim.mean()
        return self.coeff * loss

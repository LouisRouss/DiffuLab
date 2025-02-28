from typing import Any

from torch import Tensor

from diffulab.diffuse.diffusion import Diffusion
from diffulab.diffuse.modelizations.flow import Flow
from diffulab.diffuse.modelizations.gaussian_diffusion import GaussianDiffusion
from diffulab.networks.denoisers.common import Denoiser, ModelInput


class Diffuser:
    model_registry: dict[str, type[Diffusion]] = {"rectified_flow": Flow, "gaussian_diffusion": GaussianDiffusion}

    def __init__(
        self,
        denoiser: Denoiser,
        model_type: str = "rectified_flow",
        n_steps: int = 50,
        sampling_method: str = "euler",
        extra_args: dict[str, Any] = {},
    ):
        self.model_type = model_type
        self.denoiser = denoiser
        self.n_steps = n_steps

        if self.model_type in self.model_registry:
            self.diffusion = self.model_registry[self.model_type](
                n_steps=n_steps, sampling_method=sampling_method, **extra_args
            )
        else:
            raise NotImplementedError(f"Model type {self.model_type} is not implemented")

    def eval(self) -> None:
        self.denoiser.eval()

    def train(self) -> None:
        self.denoiser.train()

    def draw_timesteps(self, batch_size: int) -> Tensor:
        return self.diffusion.draw_timesteps(batch_size=batch_size)

    def compute_loss(self, model_inputs: ModelInput, timesteps: Tensor, noise: Tensor | None = None) -> Tensor:
        return self.diffusion.compute_loss(self.denoiser, model_inputs, timesteps, noise)

    def set_steps(self, n_steps: int, **extra_args: dict[str, Any]) -> None:
        self.diffusion.set_steps(n_steps, **extra_args)  # type: ignore

    def generate(
        self,
        data_shape: tuple[int, ...],
        model_inputs: ModelInput,
        use_tqdm: bool = True,
        clamp_x: bool = True,
        guidance_scale: float = 0,
        n_steps: int | None = None,
        **kwargs: dict[str, Any],
    ) -> Tensor:
        return self.diffusion.denoise(
            self.denoiser,
            data_shape,
            model_inputs,
            use_tqdm=use_tqdm,
            clamp_x=clamp_x,
            guidance_scale=guidance_scale,
            n_steps=n_steps,
            **kwargs,
        )

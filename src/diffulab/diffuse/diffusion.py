from abc import ABC, abstractmethod
from typing import Any

import torch

from torch import Tensor
from tqdm import tqdm

from diffulab.networks.denoisers.common import Denoiser


class Diffusion(ABC):
    def init(self, n_steps: int, sampling_method: str = "euler", schedule: str = "linear"):
        self.timesteps: list[float] = []
        self.steps: int = n_steps
        self.sampling_method = sampling_method

    @abstractmethod
    def set_steps(self, n_steps: int, schedule: str) -> None:
        pass

    @abstractmethod
    def one_step_denoise(
        self,
        model: Denoiser,
        model_inputs: dict[str, Any],
        t_prev: float,
        t_curr: float,
        guidance_scale: float,
        clamp_x: bool,
    ) -> Tensor:
        pass

    @abstractmethod
    def compute_loss(
        self, model: Denoiser, model_inputs: dict[str, Any], timesteps: Tensor, noise: Tensor | None = None
    ) -> Tensor:
        pass

    @abstractmethod
    def add_noise(self, x: Tensor, timesteps: Tensor, noise: Tensor | None = None) -> tuple[Tensor, Tensor]:
        pass

    def draw_timesteps(self, batch_size: int) -> Tensor:
        return torch.rand((batch_size), dtype=torch.float32)

    @torch.inference_mode()
    def denoise(
        self,
        model: Denoiser,
        data_shape: tuple[int, ...],
        model_inputs: dict[str, Any] = {},
        use_tqdm: bool = True,
        clamp_x: bool = True,
        guidance_scale: float = 10,
    ) -> Tensor:
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        x = torch.randn(data_shape, device=device, dtype=dtype)
        for t_curr, t_prev in tqdm(
            zip(self.timesteps[:-1], self.timesteps[1:]),
            desc="generating image",
            total=self.steps,
            disable=not use_tqdm,
            leave=False,
        ):
            model_inputs["x"] = x
            x = self.one_step_denoise(
                model,
                model_inputs,
                t_curr=t_curr,
                t_prev=t_prev,
                guidance_scale=guidance_scale,
                clamp_x=clamp_x,
            )
        return x


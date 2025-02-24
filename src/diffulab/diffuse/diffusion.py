from abc import ABC, abstractmethod
from typing import Any

from torch import Tensor

from diffulab.networks.denoisers.common import Denoiser, ModelInput


class Diffusion(ABC):
    def __init__(self, n_steps: int, sampling_method: str = "euler", schedule: str = "linear"):
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
        model_inputs: ModelInput,
        guidance_scale: float,
        clamp_x: bool,
        *args: Any,
        **kwargs: Any,
    ) -> Tensor:
        pass

    @abstractmethod
    def compute_loss(
        self, model: Denoiser, model_inputs: ModelInput, timesteps: Tensor, noise: Tensor | None = None
    ) -> Tensor:
        pass

    @abstractmethod
    def add_noise(self, x: Tensor, timesteps: Tensor, noise: Tensor | None = None) -> tuple[Tensor, Tensor]:
        pass

    @abstractmethod
    def denoise(
        self,
        model: Denoiser,
        data_shape: tuple[int, ...],
        model_inputs: ModelInput,
        use_tqdm: bool = True,
        clamp_x: bool = True,
        guidance_scale: float = 10,
    ) -> Tensor:
        pass

    @abstractmethod
    def draw_timesteps(self, batch_size: int) -> Tensor:
        pass

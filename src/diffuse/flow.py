import math
from abc import ABC, abstractmethod
from typing import Any, Callable

import numpy as np
import torch
import torch.distributions as dist
from numpy.typing import NDArray
from torch import Tensor

from diffuse.utils import extract_into_tensor
from networks.common import Denoiser


class Flow(ABC):
    def init(self, n_steps: int, method: str = "euler"):
        self.set_steps(n_steps)

    def add_noise(self, x: Tensor, timesteps: Tensor, noise: Tensor | None = None) -> Tensor:
        if noise is None:
            noise = torch.randn_like(x)
        assert noise.shape == x.shape
        assert timesteps.shape[0] == x.shape[0]
        at = extract_into_tensor(self.at, timesteps, x.shape)  # type: ignore
        bt = extract_into_tensor(self.bt, timesteps, x.shape)  # type: ignore
        z_t = at * x + bt * noise
        return z_t

    @abstractmethod
    def set_steps(self, n_steps: int) -> None:
        pass


class Straight(Flow):
    def __init__(self, n_steps: int = 50, method: str = "euler"):
        super().init(n_steps=n_steps, method=method)

    def set_steps(self, n_steps: int) -> None:
        self.at = 1 - np.linspace(0, 1, n_steps)
        self.bt = np.linspace(0, 1, n_steps)
        self.steps = n_steps


class EDM(Flow):
    def __init__(self, mean: float = -1.2, std: float = 1.2, n_steps: int = 50, method: str = "euler"):
        self.mean = mean
        self.std = std
        super().init(n_steps=n_steps, method=method)

    def set_steps(self, n_steps: int) -> None:
        self.at = np.ones((n_steps))
        self.bt = self.compute_bt(n_steps)
        self.steps = n_steps

    def compute_bt(self, n_steps: int) -> NDArray[np.float32]:
        normal_dist = dist.Normal(self.mean, self.std)
        # Compute the inverse CDF (quantile function) of the normal distribution for each t
        quantile_values: Tensor = normal_dist.icdf(torch.linspace(0, 1, steps=n_steps))  # type: ignore
        # Compute b_t as the exponential of the quantile values
        bt = torch.exp(quantile_values)  # type: ignore
        return bt.to(torch.float32).numpy()  # type: ignore


class Cosine(Flow):
    def __init__(self, n_steps: int = 50, method: str = "euler"):
        super().init(n_steps=n_steps, method=method)

    def set_steps(self, n_steps: int) -> None:
        self.at = np.cos(np.linspace(0, 1, n_steps) * math.pi / 2)
        self.bt = np.sin(np.linspace(0, 1, n_steps) * math.pi / 2)
        self.steps = n_steps


class DDPM(Flow):
    """
    Largely inspired by https://github.com/openai/guided-diffusion/tree/main under MIT license as of 10/08/2024
    """

    def __init__(
        self,
        learn_sigma: bool = False,
        rescale_timesteps: bool = False,
        noise_schedule: str = "linear",
        n_steps: int = 1000,
        method: str = "euler",
    ):
        self.noise_schedule = noise_schedule
        self.learn_sigma = learn_sigma
        self.rescale_timesteps = rescale_timesteps
        super().init(n_steps=n_steps, method=method)

    def set_steps(self, n_steps: int) -> None:
        betas = self.get_beta_schedule(n_steps)
        alphas = 1.0 - betas
        self.at = np.cumprod(alphas, axis=0)
        self.bt = np.sqrt(1.0 - self.at)
        self.steps = n_steps

    def betas_for_alpha_bar(
        self, alpha_bar: Callable[[float], float], n_steps: int, max_beta: float = 0.999
    ) -> NDArray[np.float64]:
        """
        Create a beta schedule that discretizes the given alpha_t_bar function,
        which defines the cumulative product of (1-beta) over time from t = [0,1].

        :param num_diffusion_timesteps: the number of betas to produce.
        :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                        produces the cumulative product of (1-beta) up to that
                        part of the diffusion process.
        :param max_beta: the maximum beta to use; use values lower than 1 to
                        prevent singularities.
        """
        betas: list[float] = []
        for i in range(n_steps):
            t1 = i / n_steps
            t2 = (i + 1) / n_steps
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
        return np.array(betas)

    def get_beta_schedule(self, n_steps: int) -> NDArray[np.float64]:
        """
        Get a pre-defined beta schedule for the given name.

        The beta schedule library consists of beta schedules which remain similar
        in the limit of num_diffusion_timesteps.
        Beta schedules may be added, but should not be removed or changed once
        they are committed to maintain backwards compatibility.
        """
        if self.noise_schedule == "linear":
            # Linear schedule from Ho et al, extended to work for any number of
            # diffusion steps.
            scale = 1000 / n_steps
            beta_start = scale * 0.0001
            beta_end = scale * 0.02
            return np.linspace(beta_start, beta_end, n_steps, dtype=np.float64)  # type: ignore
        elif self.noise_schedule == "cosine":
            return self.betas_for_alpha_bar(lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2, n_steps)
        else:
            raise NotImplementedError(f"unknown beta schedule: {self.noise_schedule}")


class Diffuser:
    def __init__(self, denoiser: Denoiser, method: str = "rectified_flow"):
        self.method = method
        self.denoiser = denoiser

        if self.method == "rectified_flow":
            self.flow = Straight()

    def add_noise(self, x: Tensor, timesteps: Tensor, noise: Tensor | None = None) -> Tensor:
        return self.flow.add_noise(x, timesteps, noise)

    def one_step_denoise(self, model_inputs: dict[str, Any]) -> Tensor: ...

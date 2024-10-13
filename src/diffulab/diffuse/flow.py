import math
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch
import torch.distributions as dist
import torch.nn as nn
from numpy.typing import NDArray
from torch import Tensor

from diffulab.diffuse.utils import extract_into_tensor
from diffulab.networks.common import Denoiser


class Flow(ABC):
    def init(self, n_steps: int, sampling_method: str = "euler"):
        self.at = np.empty((n_steps), dtype=np.float32)
        self.bt = np.empty((n_steps), dtype=np.float32)
        self.dat = np.empty((n_steps), dtype=np.float32)
        self.dbt = np.empty((n_steps), dtype=np.float32)
        self.wt = np.empty((n_steps), dtype=np.float32)

        self.set_steps(n_steps)
        self.sampling_method = sampling_method
        self.dlambda = 2 * (self.dat / self.at - self.dbt / self.bt)

    @abstractmethod
    def set_steps(self, n_steps: int) -> None:
        pass

    @abstractmethod
    def one_step_denoise(self, model: Denoiser, model_inputs: dict[str, Any], timesteps: Tensor) -> Tensor:
        pass

    @abstractmethod
    def draw_timesteps(self, batch_size: int) -> Tensor:
        pass

    def compute_loss(
        self, model: Denoiser, model_inputs: dict[str, Any], timesteps: Tensor, noise: Tensor | None = None
    ) -> Tensor:
        model_inputs["x"], noise = self.add_noise(model_inputs["x"], timesteps, noise)
        prediction = model(**model_inputs, timesteps=timesteps)
        loss = (
            -1
            / 2
            * extract_into_tensor(self.wt, timesteps, model_inputs["x"].shape)
            * extract_into_tensor(self.dlambda, timesteps, model_inputs["x"].shape)
            * nn.functional.mse_loss(prediction, noise)
        )
        return loss

    def add_noise(self, x: Tensor, timesteps: Tensor, noise: Tensor | None = None) -> tuple[Tensor, Tensor]:
        if noise is None:
            noise = torch.randn_like(x)
        assert noise.shape == x.shape
        assert timesteps.shape[0] == x.shape[0]
        at = extract_into_tensor(self.at, timesteps, x.shape)  # type: ignore
        bt = extract_into_tensor(self.bt, timesteps, x.shape)  # type: ignore
        z_t = at * x + bt * noise
        return z_t, noise

    @torch.no_grad()  # type: ignore
    def denoise(
        self,
        model: Denoiser,
        model_inputs: dict[str, Any],
        data_shape: tuple[int, ...],
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        x = torch.randn(data_shape, device=device, dtype=dtype)
        timesteps: Tensor = torch.ones((data_shape[0]), device=device, dtype=dtype)
        for n in range(self.steps):  # type: ignore
            model_inputs["x"] = x
            x = self.one_step_denoise(model, model_inputs, timesteps)  # type: ignore
            timesteps -= 1 / self.steps  # type: ignore
        return x


class Straight(Flow):
    def __init__(self, n_steps: int = 50, sampling_method: str = "euler"):
        super().init(n_steps=n_steps, sampling_method=sampling_method)

    def set_steps(self, n_steps: int) -> None:
        self.at = 1 - np.linspace(0, 1, n_steps)
        self.bt = np.linspace(0, 1, n_steps)
        self.dat = np.full((n_steps), -1, dtype=np.float32)
        self.dbt = np.full((n_steps), 1, dtype=np.float32)
        self.wt = self.bt[:-1] / self.at[:-1]
        self.steps = n_steps

    def one_step_denoise(self, model: Denoiser, model_inputs: dict[str, Any], timesteps: Tensor) -> Tensor:
        if self.sampling_method == "euler":
            prediction = model(**model_inputs, timesteps=timesteps)
            x = model_inputs["x"] - prediction * 1 / self.steps
            return x
        else:
            raise NotImplementedError

    def draw_timesteps(self, batch_size: int) -> Tensor:
        return torch.rand((batch_size), dtype=torch.float32)


class EDM(Flow):
    def __init__(self, mean: float = -1.2, std: float = 1.2, n_steps: int = 50, sampling_method: str = "euler"):
        self.mean = mean
        self.std = std
        super().init(n_steps=n_steps, sampling_method=sampling_method)

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

    def one_step_denoise(self, model: Denoiser, model_inputs: dict[str, Any], timesteps: Tensor) -> Tensor:
        if self.sampling_method == "euler":
            prediction = model(**model_inputs, timesteps=timesteps)
            bt = extract_into_tensor(self.bt, timesteps=timesteps, broadcast_shape=prediction.shape)
            bt_prev = extract_into_tensor(
                self.bt, timesteps=timesteps - (1 / self.steps), broadcast_shape=prediction.shape
            )
            x = model_inputs["x"] - (bt - bt_prev) * prediction
            return x
        # elif self.sampling_method == "ddpm":
        #     ...
        else:
            raise NotImplementedError

    def draw_timesteps(self, batch_size: int) -> Tensor:
        return torch.rand((batch_size), dtype=torch.float32)


class Cosine(Flow):
    def __init__(self, n_steps: int = 50, sampling_method: str = "euler"):
        super().init(n_steps=n_steps, sampling_method=sampling_method)

    def set_steps(self, n_steps: int) -> None:
        self.at = np.cos(np.linspace(0, 1, n_steps) * math.pi / 2)
        self.bt = np.sin(np.linspace(0, 1, n_steps) * math.pi / 2)
        self.steps = n_steps

    def one_step_denoise(self, model: Denoiser, model_inputs: dict[str, Any], timesteps: Tensor) -> Tensor:
        if self.sampling_method == "euler":
            prediction = model(**model_inputs, timesteps=timesteps)
            x = model_inputs["x"] - (
                math.pi / 2 * extract_into_tensor(self.at, timesteps, prediction.shape)
            ) * prediction * (1 / self.steps)
            return x
        # elif self.sampling_method == "ddpm":
        #     ...
        else:
            raise NotImplementedError

    def draw_timesteps(self, batch_size: int) -> Tensor:
        return torch.rand((batch_size), dtype=torch.float32)


class Diffuser:
    def __init__(
        self,
        denoiser: Denoiser,
        model_type: str = "rectified_flow",
        edm_mean: float = -1.2,
        edm_std: float = 1.2,
        n_steps: int = 50,
        sampling_method: str = "euler",
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        self.model_type = model_type
        self.denoiser = denoiser
        self.device = device
        self.dtype = dtype

        if self.model_type == "rectified_flow":
            self.flow = Straight(n_steps=n_steps, sampling_method=sampling_method)

        elif self.model_type == "edm":
            self.flow = EDM(mean=edm_mean, std=edm_std, n_steps=n_steps, sampling_method=sampling_method)

        elif self.model_type == "cosine":
            self.flow = Cosine(n_steps=n_steps, sampling_method=sampling_method)

        else:
            raise NotImplementedError

    def add_noise(self, x: Tensor, timesteps: Tensor, noise: Tensor | None = None) -> tuple[Tensor, Tensor]:
        return self.flow.add_noise(x, timesteps, noise)

    def one_step_denoise(self, model_inputs: dict[str, Any], timesteps: Tensor) -> Tensor:
        return self.flow.one_step_denoise(self.denoiser, model_inputs, timesteps)

    def denoise(
        self,
        model_inputs: dict[str, Any],
        data_shape: tuple[int, ...],
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        return self.flow.denoise(self.denoiser, model_inputs, data_shape, device, dtype)

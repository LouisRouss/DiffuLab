import math
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch

# import torch.distributions as dist
import torch.nn as nn

# from numpy.typing import NDArray
from torch import Tensor

from diffulab.diffuse.utils import extract_into_tensor
from diffulab.networks.common import Denoiser


class Flow(ABC):
    def init(self, n_steps: int, sampling_method: str = "euler"):
        self.at = np.empty((n_steps), dtype=np.float32)
        self.bt = np.empty((n_steps), dtype=np.float32)
        self.log_snr = np.empty((n_steps), dtype=np.float32)
        self.dlog_snr = np.empty((n_steps), dtype=np.float32)
        self.wt = np.empty((n_steps), dtype=np.float32)
        self.timesteps: list[float] = []
        self.steps: int = n_steps

        self.set_steps(n_steps)
        self.sampling_method = sampling_method

    @abstractmethod
    def set_steps(self, n_steps: int) -> None:
        pass

    @abstractmethod
    def draw_timesteps(self, batch_size: int) -> Tensor:
        pass

    @abstractmethod
    def get_v(self, model: Denoiser, model_inputs: dict[str, Any], t_curr: float) -> Tensor:
        pass

    def one_step_denoise(self, model: Denoiser, model_inputs: dict[str, Any], t_prev: float, t_curr: float) -> Tensor:
        v = self.get_v(model, model_inputs, t_curr)
        if self.sampling_method == "euler":
            x_t_minus_one = model_inputs["x"] - v * 1 / self.steps
            return x_t_minus_one
        else:
            raise NotImplementedError

    def compute_loss(
        self, model: Denoiser, model_inputs: dict[str, Any], timesteps: Tensor, noise: Tensor | None = None
    ) -> Tensor:
        model_inputs["x"], noise = self.add_noise(model_inputs["x"], timesteps, noise)
        prediction = model(**model_inputs, timesteps=timesteps)
        loss = (
            -1
            / 2
            * extract_into_tensor(self.wt, timesteps, model_inputs["x"].shape)
            * extract_into_tensor(self.dlog_snr, timesteps, model_inputs["x"].shape)
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
        data_shape: tuple[int, ...],
        model_inputs: dict[str, Any] = {},
    ) -> Tensor:
        assert len(self.timesteps) == self.steps, "Please set the number of steps before denoising."
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        x = torch.randn(data_shape, device=device, dtype=dtype)
        for t_curr, t_prev in zip(self.timesteps[:-1], self.timesteps[1:]):
            model_inputs["x"] = x
            x = self.one_step_denoise(model, model_inputs, t_curr=t_curr, t_prev=t_prev)
        return x


class Straight(Flow):
    def __init__(self, n_steps: int = 50, sampling_method: str = "euler"):
        super().init(n_steps=n_steps, sampling_method=sampling_method)

    def set_steps(self, n_steps: int) -> None:
        self.at = 1 - np.linspace(0, 1, n_steps)
        self.bt = np.linspace(0, 1, n_steps)
        dat = np.full((n_steps), -1, dtype=np.float32)
        dbt = np.full((n_steps), 1, dtype=np.float32)
        self.log_snr = np.log(self.at**2 / self.bt**2)
        self.dlog_snr = 2 * (dat / self.at - dbt / self.bt)
        self.wt = self.bt[:-1] / self.at[:-1]
        self.timesteps: list[float] = torch.linspace(1, 0, n_steps + 1).tolist()  # type: ignore
        self.steps = n_steps

    def get_v(self, model: Denoiser, model_inputs: dict[str, Any], t_curr: float) -> Tensor:
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        timesteps = torch.full((model_inputs["x"].shape[0],), t_curr, device=device, dtype=dtype)
        prediction = model(**model_inputs, timesteps=timesteps)
        return prediction

    def draw_timesteps(self, batch_size: int) -> Tensor:
        return torch.rand((batch_size), dtype=torch.float32)


class Cosine(Flow):
    def __init__(self, n_steps: int = 50, sampling_method: str = "euler", prediction_type: str = "v"):
        self.prediction_type = "v"
        super().init(n_steps=n_steps, sampling_method=sampling_method)

    def set_steps(self, n_steps: int) -> None:
        self.at = np.cos(np.linspace(0, 1, n_steps) * math.pi / 2)
        self.bt = np.sin(np.linspace(0, 1, n_steps) * math.pi / 2)
        dat = -math.pi / 2 * np.sin(np.linspace(0, 1, n_steps) * math.pi / 2)
        dbt = math.pi / 2 * np.cos(np.linspace(0, 1, n_steps) * math.pi / 2)
        self.log_snr = np.log(self.at**2 / self.bt**2)
        self.dlog_snr = 2 * (dat / self.at - dbt / self.bt)
        if self.prediction_type == "v":
            self.wt = np.exp(-self.log_snr / 2)  # for v-prediction
        else:
            raise NotImplementedError
        self.timesteps: list[float] = torch.linspace(1, 0, n_steps + 1).tolist()  # type: ignore
        self.steps = n_steps

    def get_v(self, model: Denoiser, model_inputs: dict[str, Any], t_curr: float) -> Tensor:
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        timesteps = torch.full((model_inputs["x"].shape[0],), t_curr, device=device, dtype=dtype)
        if self.prediction_type == "v":
            prediction = model(**model_inputs, timesteps=timesteps)
            return (math.pi / 2) * prediction
        else:
            raise NotImplementedError

    def draw_timesteps(self, batch_size: int) -> Tensor:
        return torch.rand((batch_size), dtype=torch.float32)


# class EDM(Flow):
#     def __init__(self, mean: float = -1.2, std: float = 1.2, n_steps: int = 50, sampling_method: str = "euler"):
#         self.mean = mean
#         self.std = std
#         super().init(n_steps=n_steps, sampling_method=sampling_method)

#     def set_steps(self, n_steps: int) -> None:
#         self.at = np.ones((n_steps))
#         self.bt = self.compute_bt(n_steps)
#         dat = np.zeros((n_steps))
#         dbt = self.compute_dbt(n_steps)
#         self.log_snr = np.log(self.at**2 / self.bt**2)
#         self.dlog_snr = 2 * (dat / self.at - dbt / self.bt)
#         self.wt = self.log_snr * (np.exp(-self.log_snr) + 0.5**2)  # F prediction
#         self.steps = n_steps

#     def compute_bt(self, n_steps: int) -> NDArray[np.float32]:
#         normal_dist = dist.Normal(self.mean, self.std)
#         # Compute the inverse CDF (quantile function) of the normal distribution for each t
#         quantile_values: Tensor = normal_dist.icdf(torch.linspace(0, 1, steps=n_steps))  # type: ignore
#         # Compute b_t as the exponential of the quantile values
#         bt = torch.exp(quantile_values)  # type: ignore
#         return bt.to(torch.float32).numpy()  # type: ignore

#     def compute_dbt(self, n_steps: int) -> NDArray[np.float32]:
#         # First, compute bt as before
#         bt = self.compute_bt(n_steps)
#         # Calculate the finite difference approximation of the derivative
#         # dbt/dt ≈ (b_t+1 - b_t) / delta_t
#         delta_t = 1.0 / (n_steps - 1)  # Assuming time step is evenly spaced
#         dbt = np.diff(bt) / delta_t  # np.diff gives b_t+1 - b_t
#         dbt = np.append(dbt, dbt[-1])  # Extrapolate last value by repeating the last difference
#         return dbt.astype(np.float32)

#     def draw_timesteps(self, batch_size: int) -> Tensor:
#         return torch.rand((batch_size), dtype=torch.float32)


class Diffuser:
    def __init__(
        self,
        denoiser: Denoiser,
        model_type: str = "rectified_flow",
        edm_mean: float = -1.2,
        edm_std: float = 1.2,
        n_steps: int = 50,
        sampling_method: str = "euler",
    ):
        self.model_type = model_type
        self.denoiser = denoiser
        self.n_steps = n_steps

        if self.model_type == "rectified_flow":
            self.flow = Straight(n_steps=n_steps, sampling_method=sampling_method)

        # elif self.model_type == "edm":
        #     self.flow = EDM(mean=edm_mean, std=edm_std, n_steps=n_steps, sampling_method=sampling_method)

        elif self.model_type == "cosine":
            self.flow = Cosine(n_steps=n_steps, sampling_method=sampling_method)

        else:
            raise NotImplementedError

    def eval(self) -> None:
        self.denoiser.eval()

    def train(self) -> None:
        self.denoiser.train()

    def draw_timesteps(self, batch_size: int) -> Tensor:
        return self.flow.draw_timesteps(batch_size=batch_size)

    def compute_loss(self, model_inputs: dict[str, Any], timesteps: Tensor, noise: Tensor | None = None) -> Tensor:
        return self.flow.compute_loss(self.denoiser, model_inputs, timesteps, noise)

    def set_steps(self, n_steps: int):
        self.flow.set_steps(n_steps)

    def generate(
        self,
        data_shape: tuple[int, ...],
        model_inputs: dict[str, Any] = {},
    ) -> Tensor:
        return self.flow.denoise(self.denoiser, data_shape, model_inputs)

from abc import ABC, abstractmethod
from typing import Any

import torch

import torch.nn as nn
from torch import Tensor
from tqdm import tqdm

from diffulab.networks.denoisers.common import Denoiser


class Diffusion(ABC):
    def init(self, n_steps: int, sampling_method: str = "euler"):
        self.timesteps: list[float] = []
        self.steps: int = n_steps
        self.set_steps(n_steps, schedule="linear")
        self.sampling_method = sampling_method

    @abstractmethod
    def set_steps(self, n_steps: int, schedule:str) -> None:
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

class Flow(Diffusion):
    def __init__(self, n_steps: int = 50, sampling_method: str = "euler"):
        super().init(n_steps=n_steps, sampling_method=sampling_method)

    def set_steps(self, n_steps: int, schedule: str = "linear") -> None:
        if schedule == "linear":
            self.timesteps: list[float] = torch.linspace(1, 0, n_steps + 1).tolist()  # type: ignore
            self.steps = n_steps
        else:
            raise NotImplementedError("Only linear schedule is supported for the moment")

    def at(self, timesteps: Tensor) -> Tensor:
        return 1 - timesteps

    def bt(self, timesteps: Tensor) -> Tensor:
        return timesteps

    def dat(self, timesteps: Tensor) -> Tensor:
        return torch.full_like(timesteps, -1)

    def dbt(self, timesteps: Tensor) -> Tensor:
        return torch.full_like(timesteps, 1)

    def log_snr(self, timesteps: Tensor) -> Tensor:
        return torch.log(self.at(timesteps) ** 2 / self.bt(timesteps) ** 2)

    def dlog_snr(self, timesteps: Tensor) -> Tensor:
        return 2 * (self.dat(timesteps) / self.at(timesteps) - self.dbt(timesteps) / self.bt(timesteps))

    def wt(self, timesteps: Tensor) -> Tensor:
        return self.bt(timesteps) / self.at(timesteps)

    def get_v(self, model: Denoiser, model_inputs: dict[str, Any], t_curr: float) -> Tensor:
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        timesteps = torch.full((model_inputs["x"].shape[0],), t_curr, device=device, dtype=dtype)
        prediction = model(**model_inputs, timesteps=timesteps)
        # return prediction - model_inputs["x"]
        v = prediction * self.dlog_snr(timesteps) * self.bt(timesteps) * (1 / 2) + model_inputs["x"] * self.dat(
            timesteps
        ) / self.at(timesteps)
        return v
    
    def one_step_denoise(
        self,
        model: Denoiser,
        model_inputs: dict[str, Any],
        t_prev: float,
        t_curr: float,
        guidance_scale: float,
        clamp_x: bool,
    ) -> Tensor:
        v = self.get_v(model, {**model_inputs, "p": 0}, t_curr)
        if guidance_scale > 0:
            v_dropped = self.get_v(model, {**model_inputs, "p": 1}, t_curr)
            v = v + guidance_scale * (v - v_dropped)
        if self.sampling_method == "euler":
            x_t_minus_one : Tensor = model_inputs["x"] - v * (t_prev - t_curr)
        else: # different methods to be implemented maybe in the generic class instead
            raise NotImplementedError
        if clamp_x:
            x_t_minus_one = x_t_minus_one.clamp(0, 1)
        return x_t_minus_one

    def compute_loss(
        self, model: Denoiser, model_inputs: dict[str, Tensor], timesteps: Tensor, noise: Tensor | None = None
    ) -> Tensor:
        model_inputs["x"], noise = self.add_noise(model_inputs["x"], timesteps, noise)
        prediction = model(**model_inputs, timesteps=timesteps)
        loss = (
            -1
            / 2
            * self.wt(timesteps)
            * self.dlog_snr(timesteps)
            * nn.functional.mse_loss(prediction, noise, reduction="none").mean(dim=list(range(1, prediction.dim())))
        ).mean()
        return loss

    def add_noise(self, x: Tensor, timesteps: Tensor, noise: Tensor | None = None) -> tuple[Tensor, Tensor]:
        if noise is None:
            noise = torch.randn_like(x)
        assert noise.shape == x.shape
        assert timesteps.shape[0] == x.shape[0]
        at = self.at(timesteps).view(-1, *([1] * (x.dim() - 1))).to(x.device)
        bt = self.bt(timesteps).view(-1, *([1] * (x.dim() - 1))).to(x.device)
        z_t = at * x + bt * noise
        return z_t, noise




class Diffuser:
    model_registry = {
        "rectified_flow": Flow,
    }
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
            self.diffusion = self.model_registry[self.model_type](n_steps=n_steps, sampling_method=sampling_method, **extra_args)
        else:
            raise NotImplementedError(f"Model type {self.model_type} is not implemented")

    def eval(self) -> None:
        self.denoiser.eval()

    def train(self) -> None:
        self.denoiser.train()

    def draw_timesteps(self, batch_size: int) -> Tensor:
        return self.diffusion.draw_timesteps(batch_size=batch_size)

    def compute_loss(self, model_inputs: dict[str, Any], timesteps: Tensor, noise: Tensor | None = None) -> Tensor:
        return self.diffusion.compute_loss(self.denoiser, model_inputs, timesteps, noise)

    def set_steps(self, n_steps: int):
        self.diffusion.set_steps(n_steps)

    def generate(
        self,
        data_shape: tuple[int, ...],
        model_inputs: dict[str, Any] = {},
    ) -> Tensor:
        return self.diffusion.denoise(self.denoiser, data_shape, model_inputs)

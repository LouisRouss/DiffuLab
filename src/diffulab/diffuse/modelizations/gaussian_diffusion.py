# In part inspired from https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/gaussian_diffusion.py under MIT license as of 2025-03-02

import enum
import math
from typing import Callable

import torch
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm

from diffulab.diffuse.diffusion import Diffusion
from diffulab.diffuse.utils import extract_into_tensor
from diffulab.networks.denoisers.common import Denoiser, ModelInput


class MeanType(enum.Enum):
    EPSILON = "epsilon"
    XSTART = "xstart"
    XPREV = "xprev"


class ModelVarType(enum.Enum):
    LEARNED = "learned"
    FIXED_SMALL = "fixed_small"
    FIXED_LARGE = "fixed_large"
    LEARNED_RANGE = "learned_range"


class GaussianDiffusion(Diffusion):
    def __init__(
        self,
        n_steps: int = 50,
        original_steps: int = 1000,
        sampling_method: str = "ddpm",
        schedule: str = "linear",
        mean_type: str = "epsilon",
        variance_type: str = "fixed_small",
    ):
        super().__init__(n_steps=n_steps, sampling_method=sampling_method, schedule=schedule)
        if mean_type not in MeanType._value2member_map_:
            raise ValueError(f"mean_type must be one of {[e.value for e in MeanType]}")
        if variance_type not in ModelVarType._value2member_map_:
            raise ValueError(f"variance_type must be one of {[e.value for e in ModelVarType]}")
        self.mean_type = mean_type
        self.var_type = variance_type

    def set_steps(self, n_steps: int, schedule: str = "linear") -> None:
        self.timesteps: list[float] = torch.linspace(n_steps, 0, n_steps + 1, dtype=torch.int32).tolist()  # type: ignore
        self.steps = n_steps
        self.betas = self._get_variance_schedule(n_steps, schedule)
        self.alphas = 1 - self.betas
        self.alphas_bar = self.alphas.cumprod(dim=0)
        self.alphas_bar_prev = torch.cat([torch.tensor([1.0], dtype=torch.float64), self.alphas_bar[:-1]])
        self.alphas_bar_next = torch.cat([self.alphas_bar[1:], torch.tensor([0.0], dtype=torch.float64)])

        # utils for computation
        self.sqrt_alphas_bar = self.alphas_bar.sqrt()
        self.posterior_variance = self.betas * (1.0 - self.alphas_bar_prev) / (1.0 - self.alphas_bar)
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]])
        )
        self.posterior_mean_coef1 = self.betas * (self.alphas_bar_prev).sqrt() / (1.0 - self.alphas_bar)
        self.posterior_mean_coef2 = (1.0 - self.alphas_bar_prev) * self.alphas.sqrt() / (1.0 - self.alphas_bar)

    def _get_variance_schedule(self, n_steps: int, variance_schedule: str = "linear") -> Tensor:
        if variance_schedule == "linear":
            # Linear schedule from Ho et al, extended to work for any number of
            # diffusion steps.
            scale = 1000 / n_steps
            beta_start = scale * 0.0001
            beta_end = scale * 0.02
            return torch.linspace(beta_start, beta_end, n_steps, dtype=torch.float64)
        elif variance_schedule == "cosine":
            return self._betas_for_alpha_bar(
                n_steps,
                lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
            )
        else:
            raise NotImplementedError(f"unknown beta schedule: {variance_schedule}")

    def _betas_for_alpha_bar(
        self, n_steps: int, alpha_bar: Callable[[float], float], max_beta: float = 0.999
    ) -> Tensor:
        betas: list[float] = []
        for i in range(n_steps):
            t1 = i / n_steps
            t2 = (i + 1) / n_steps
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
        return torch.tensor(betas, dtype=torch.float64)

    def _get_x_start_from_x_prev(self, x_prev: Tensor, x: Tensor, t: int) -> Tensor:
        x_start = (1.0 / self.posterior_mean_coef1[t]) * x_prev + (1.0 / self.posterior_mean_coef2[t]) * x
        return x_start

    def _get_x_start_from_eps(self, eps: Tensor, x: Tensor, t: int) -> Tensor:
        x_start = (1.0 / self.sqrt_alphas_bar[t]) * x - (
            (1.0 - self.alphas_bar[t]).sqrt() / self.sqrt_alphas_bar[t]
        ) * eps
        return x_start

    def _get_mean_from_x_start(self, x: Tensor, x_start: Tensor, t: int) -> Tensor:
        mean = self.posterior_mean_coef1[t] * x_start + self.posterior_mean_coef2[t] * x
        return mean

    def _get_x_prev_from_mean_var(self, mean: Tensor, var: Tensor, t: int) -> Tensor:
        if t > 0:
            return mean + torch.randn_like(mean) * var.sqrt()
        else:  # no noise for ultimate timestep
            return mean

    def _get_eps_from_xstart(self, x_start: Tensor, x: Tensor, t: int) -> Tensor:
        eps = (1.0 / (1 - self.alphas_bar[t]).sqrt()) * (x - self.sqrt_alphas_bar[t] * x_start)
        return eps

    def _get_p_mean_var(
        self, prediction: Tensor, x: Tensor, t: int, clamp_x: bool = False
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if self.var_type in [ModelVarType.FIXED_SMALL.value, ModelVarType.FIXED_LARGE.value]:
            model_output = prediction
        elif self.var_type in [ModelVarType.LEARNED.value, ModelVarType.LEARNED_RANGE.value]:
            assert prediction.shape[1] % 2 == 0
            model_output, log_var = torch.chunk(prediction, 2, dim=1)
        else:
            raise ValueError(f"Unknown model var type: {self.var_type}")

        # extract mean
        if self.mean_type == MeanType.XPREV.value:
            x_start = self._get_x_start_from_x_prev(model_output, x, t)
        elif self.mean_type == MeanType.XSTART.value:
            x_start = model_output
        elif self.mean_type == MeanType.EPSILON.value:
            x_start = self._get_x_start_from_eps(model_output, x, t)
        else:
            raise ValueError(f"Unknown mean type: {self.mean_type}")
        if clamp_x:
            x_start = torch.clamp(x_start, -1, 1)
        mean = self._get_mean_from_x_start(x, x_start, t)

        # extract variance
        if self.var_type == ModelVarType.FIXED_SMALL.value:
            var, log_var = self.posterior_variance[t], self.posterior_log_variance_clipped[t]
        elif self.var_type == ModelVarType.FIXED_LARGE.value:
            var, log_var = (
                torch.cat([self.posterior_variance[1:2], self.betas[1:]]),
                torch.cat([self.posterior_log_variance_clipped[1:2], self.betas[1:]]),
            )
            var, log_var = var[t], log_var[t]
        elif self.var_type == ModelVarType.LEARNED.value:
            var = log_var.exp()  # type: ignore
        elif self.var_type == ModelVarType.LEARNED_RANGE.value:
            min_log = self.posterior_log_variance_clipped[t]
            max_log = self.betas[t].log()
            log_var = (log_var + 1) / 2  # type: ignore
            log_var = log_var * max_log + (1 - log_var) * min_log
            var = log_var.exp()
        else:
            raise ValueError(f"Unknown model var type: {self.var_type}")

        return mean, var, log_var, x_start  # type: ignore

    def _get_mean_for_ddim_guidance(
        self, x: Tensor, x_start: Tensor, t: int, grad: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        eps = self._get_eps_from_xstart(x, x_start, t)
        eps = eps - (1 - self.alphas_bar[t]).sqrt() * grad

        x_start = self._get_x_start_from_eps(eps, x, t)
        mean = self._get_mean_from_x_start(x, x_start, t)

        return mean, x_start, eps

    def _sample_x_prev_ddim(self, x: Tensor, eps: Tensor, t: int) -> Tensor:
        x_prev = (
            self.alphas_bar_prev[t].sqrt() * ((x - (1 - self.alphas_bar[t]).sqrt() * eps) / self.sqrt_alphas_bar[t])
            + (1 - self.alphas_bar_prev[t]).sqrt() * eps
        )
        return x_prev

    def classifier_grad(
        self, x: Tensor, y: Tensor, t: Tensor, classifier: Callable[[Tensor, Tensor], Tensor]
    ) -> Tensor:
        x = x.detach().requires_grad_()
        logits = classifier(x, t)
        logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
        logprobs_y = logprobs[torch.arange(y.size(0)), y]
        return torch.autograd.grad(logprobs_y.sum(), x)[0]

    def draw_timesteps(self, batch_size: int) -> Tensor:
        return torch.randint(0, self.steps, (batch_size,), dtype=torch.int32)

    def one_step_denoise(
        self,
        model: Denoiser,
        model_inputs: ModelInput,
        t: int,
        clamp_x: bool = False,
        guidance_scale: float = 0.0,
        classifier_free: bool = True,
        classifier: Callable[[Tensor, Tensor], Tensor] | None = None,
    ) -> Tensor:
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        timesteps = torch.full((model_inputs["x"].shape[0],), t, device=device, dtype=dtype)
        prediction = model(**{**model_inputs, "p": 0}, timesteps=timesteps)

        if classifier_free and guidance_scale > 0:
            prediction_uncond = model(**{**model_inputs, "p": 1}, timesteps=timesteps)
            prediction = prediction + guidance_scale * (prediction - prediction_uncond)

        mean, var, _, x_start = self._get_p_mean_var(prediction, model_inputs["x"], t, clamp_x)

        if not classifier_free and guidance_scale > 0:
            assert classifier is not None
            if "y" in model_inputs:
                grad = guidance_scale * self.classifier_grad(prediction, model_inputs["y"], timesteps, classifier)
            elif "context" in model_inputs:
                grad = guidance_scale * self.classifier_grad(prediction, model_inputs["context"], timesteps, classifier)
            else:
                raise ValueError("No context or label provided for the classifier")

            if self.sampling_method == "ddpm":
                mean = mean + grad
                x_prev = self._get_x_prev_from_mean_var(mean, var, t)
            elif self.sampling_method == "ddim":
                mean, x_start, eps = self._get_mean_for_ddim_guidance(model_inputs["x"], x_start, t, grad)
                x_prev = self._sample_x_prev_ddim(model_inputs["x"], eps, t)
            else:
                raise NotImplementedError(
                    f"Classifier guidance not implemented for sampling method: {self.sampling_method}"
                )
        else:
            x_prev = self._get_x_prev_from_mean_var(mean, var, t)

        return x_prev

    # maybe change it to avoid inplace change of the dict
    def denoise(
        self,
        model: Denoiser,
        data_shape: tuple[int, ...],
        model_inputs: ModelInput,
        use_tqdm: bool = True,
        clamp_x: bool = True,
        guidance_scale: float = 0,
        classifier_free: bool = True,
        classifier: Callable[[Tensor, Tensor], Tensor] | None = None,
    ) -> Tensor:
        if "x" not in model_inputs:
            model_inputs["x"] = torch.randn(
                data_shape, device=next(model.parameters()).device, dtype=next(model.parameters()).dtype
            )
        for t in tqdm(
            list(range(self.steps))[::-1],
            desc="generating image",
            total=self.steps,
            disable=not use_tqdm,
            leave=False,
        ):
            model_inputs["x"] = self.one_step_denoise(
                model=model,
                model_inputs=model_inputs,
                t=t,
                clamp_x=clamp_x,
                guidance_scale=guidance_scale,
                classifier_free=classifier_free,
                classifier=classifier,
            )
        return model_inputs["x"]

    def compute_loss(
        self, model: Denoiser, model_inputs: ModelInput, timesteps: Tensor, noise: Tensor | None = None
    ) -> Tensor:
        model_inputs["x"], noise = self.add_noise(model_inputs["x"], timesteps, noise)
        prediction = model(**model_inputs, timesteps=timesteps)
        loss = nn.functional.mse_loss(prediction, noise, reduction="mean")
        return loss

    def add_noise(self, x: Tensor, timesteps: Tensor, noise: Tensor | None = None) -> tuple[Tensor, Tensor]:
        if noise is None:
            noise = torch.randn_like(x)
        assert noise.shape == x.shape
        assert timesteps.shape[0] == x.shape[0]
        x_t = (
            extract_into_tensor(self.sqrt_alphas_bar, timesteps, x.shape) * x
            + (1 - extract_into_tensor(self.alphas_bar, timesteps, noise.shape)).sqrt() * noise
        )
        return x_t, noise

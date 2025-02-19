from typing import Callable

import torch

from torch import Tensor
import math

from diffulab.diffuse.diffusion import Diffusion
# In part adapted from https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/gaussian_diffusion.py

class GaussianDiffusion(Diffusion):
    def __init__(self, n_steps: int = 50, sampling_method: str = "euler", schedule: str = "linear"):
        super().init(n_steps=n_steps, sampling_method=sampling_method, schedule=schedule)
        self.set_steps(n_steps, schedule=schedule)

    def set_steps(self, n_steps: int, schedule: str = "linear") -> None: 
        self.timesteps: list[float] = torch.linspace(n_steps, 0, n_steps + 1, dtype=torch.int32).tolist() # type: ignore
        self.steps = n_steps
        self.betas = self._get_variance_schedule(n_steps, schedule)
        self.alphas = 1 - self.betas
        self.alpha_bar = self.alphas.cumprod(dim=0)
        self.alpha_bar_prev = torch.cat([torch.tensor([1.0], dtype=torch.float64), self.alpha_bar[:-1]])
        self.alpha_bar_next = torch.cat([self.alpha_bar[1:], torch.tensor([0.0], dtype=torch.float64)])
    
    def _get_variance_schedule(self, n_steps: int, variance_schedule: str = "linear") -> Tensor:
        if variance_schedule == "linear":
            # Linear schedule from Ho et al, extended to work for any number of
            # diffusion steps.
            scale = 1000 / n_steps
            beta_start = scale * 0.0001
            beta_end = scale * 0.02
            return torch.linspace(
                beta_start, beta_end, n_steps, dtype=torch.float64
            )
        elif variance_schedule == "cosine":
            return self._betas_for_alpha_bar(
                n_steps,
                lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
            )
        else:
            raise NotImplementedError(f"unknown beta schedule: {variance_schedule}")
        
    def _betas_for_alpha_bar(self, n_steps: int, alpha_bar: Callable[[float], float], max_beta : float =0.999) -> Tensor:
        betas : list[float]= []
        for i in range(n_steps):
            t1 = i / n_steps
            t2 = (i + 1) / n_steps
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
        return torch.tensor(betas, dtype=torch.float64)
    
    
    
if __name__ == "__main__":
    gaussian_diffusion = GaussianDiffusion(n_steps=50, sampling_method="euler", schedule="linear", variance_schedule="linear")
    print(gaussian_diffusion.timesteps)
    print(gaussian_diffusion.betas)
    print(gaussian_diffusion.alphas)
    print(gaussian_diffusion.alpha_bar)
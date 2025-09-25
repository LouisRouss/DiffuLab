import torch
from torch import Tensor

from diffulab.diffuse.samplers import StepResult
from diffulab.diffuse.samplers.flow.common import Sampler


class EulerMaruyama(Sampler):
    name = "euler_maruyama"

    def __init__(self, eta: float = 0.7) -> None:
        super().__init__()
        self.eta = eta
        self.tmax = None

    def set_steps(self, timesteps: list[float]):
        """
        Set tmax based on the provided timesteps.
        Args:
            timesteps (list[float]): The list of timesteps
        """
        self.tmax = timesteps[1]

    def step(self, x_t: Tensor, v: Tensor, t_curr: float, t_prev: float, x_prev: Tensor | None = None) -> StepResult:
        """
        Perform one step of the reverse diffusion process using the Euler-Maruyama method.

        Args:
            x_t (Tensor): The current state tensor at time t_curr.
            v (Tensor): The velocity field tensor at time t_curr.
            t_curr (float): The current timestep in the diffusion process.
            t_prev (float): The previous timestep in the diffusion process.
            x_prev (Tensor, optional): If provided, this tensor will be used as the previous state instead of sampling a new one.

        Returns:
            StepResult: A dictionary containing the results of the step, including:

        """
        assert self.tmax is not None, "set_steps must be called before step"
        sigma: float = ((t_curr / (1 - min(t_curr, self.tmax))) ** 0.5) * self.eta
        x_prev_mean = x_t - (v + sigma**2 / (2 * t_curr) * (x_t + (1 - t_curr) * v)) * (t_curr - t_prev)
        x_prev_std = torch.tensor(sigma * (t_curr - t_prev) ** 0.5, device=x_t.device)
        if x_prev is None:
            noise = torch.randn_like(x_t)
            x_prev = x_prev_mean + x_prev_std * noise

        assert x_prev is not None  # for python type checking
        estimated_x0 = x_t - v * t_curr
        logprob = -(
            (x_prev.detach() - x_prev_mean) ** 2 / (2 * x_prev_std**2)
            + torch.log(x_prev_std)
            + 0.5 * torch.log(torch.tensor(2 * torch.pi, device=x_prev.device))
        )

        return StepResult(
            x_prev=x_prev, x_prev_mean=x_prev_mean, x_prev_std=x_prev_std, estimated_x0=estimated_x0, logprob=logprob
        )

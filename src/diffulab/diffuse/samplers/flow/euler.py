from torch import Tensor

from diffulab.diffuse.samplers import StepResult
from diffulab.diffuse.samplers.flow.common import Sampler


class Euler(Sampler):
    name = "euler"

    def __init__(self) -> None:
        super().__init__()

    def set_steps(self, timesteps: list[float]) -> None:
        """
        Set the sampler timesteps for eventual parameters computation.

        Args:
            timesteps (list[float]): A list of timesteps
        """
        pass

    def step(self, x_t: Tensor, v: Tensor, t_curr: float, t_prev: float) -> StepResult:
        """
        Perform one step of the reverse diffusion process using the Euler method.

        Args:
            x_t (Tensor): The current state tensor at time t_curr.
            v (Tensor): The velocity field tensor at time t_curr.
            t_curr (float): The current timestep in the diffusion process.
            t_prev (float): The previous timestep in the diffusion process.

        Returns:
            StepResult: A dictionary containing the results of the step, including:
                - x_prev (Tensor): The updated state tensor at time t_prev.
                - estimated_x0 (Tensor, optional): Estimated original data
        """
        dt = t_curr - t_prev  # positive dt
        x_prev = x_t - v * dt
        estimated_x0 = x_t - v * t_curr

        return StepResult(x_prev=x_prev, estimated_x0=estimated_x0)

from abc import ABC, abstractmethod
from typing import Any

from torch import Tensor

from diffulab.diffuse.samplers import StepResult


class Sampler(ABC):
    name: str

    def __init__(self) -> None:
        pass

    @abstractmethod
    def step(self, x_t: Tensor, v: Tensor, t_curr: float, t_prev: float, *args: Any, **kwargs: Any) -> StepResult:
        """
        Perform one step of the reverse diffusion process.

        Args:
            x_t (Tensor): The current state tensor at time t_curr.
            v (Tensor): The velocity field tensor at time t_curr inferred by the model.
            t_curr (float): The current timestep in the diffusion process.
            t_prev (float): The previous timestep in the diffusion process.
            *args: Additional positional arguments for specific sampler implementations.
            **kwargs: Additional keyword arguments for specific sampler implementations.

        Returns:
            StepResult: A dictionary containing the results of the step, including:
                - x_prev (Tensor): The updated state tensor at time t_prev.
                - x_prev_mean (Tensor, optional): The mean of the updated state before adding noise.
                - x_prev_std (Tensor, optional): The standard deviation of the noise added.
                - logprobs (Tensor, optional): Log probabilities if applicable.
                - estimated_x0 (Tensor, optional): Estimated original data if applicable.
        """
        pass

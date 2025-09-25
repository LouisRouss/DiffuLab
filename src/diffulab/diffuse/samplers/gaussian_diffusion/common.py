from abc import ABC, abstractmethod
from typing import Any

from torch import Tensor

from diffulab.diffuse.samplers import StepResult


class Sampler(ABC):
    name: str

    def __init__(self) -> None:
        pass

    @abstractmethod
    def set_steps(self, betas: Tensor) -> None:
        """
        Set the diffusion parameters based on the provided beta schedule.

        Args:
            betas (Tensor): A 1D tensor containing the beta values for each timestep in the diffusion process.
        """
        pass

    @abstractmethod
    def step(
        self, model_prediction: Tensor, timesteps: Tensor, xt: Tensor, clamp_x: bool = False, *args: Any, **kwargs: Any
    ) -> StepResult:
        """
        Perform one step of the reverse diffusion process.

        Args:
            model_prediction (Tensor): The model's prediction at the current timestep.
            timesteps (Tensor): The current timestep in the diffusion process.
            xt (Tensor): The current state tensor at the current timestep.
            clamp_x (bool, optional): Whether to clamp the predicted x0 to a valid range
            *args: Additional positional arguments specific to the sampler.
            **kwargs: Additional keyword arguments specific to the sampler.

        Returns:
            StepResult: A dictionary containing the results of the step, including:
                - x_prev (Tensor): The updated state tensor at the previous timestep.
                - estimated_x0 (Tensor, optional): Estimated original data.
        """
        pass

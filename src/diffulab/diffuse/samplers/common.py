from abc import ABC, abstractmethod
from typing import Any, NotRequired, Required, TypedDict

from torch import Tensor


class StepResult(TypedDict):
    x_prev: Required[Tensor]
    estimated_x0: Required[Tensor]
    x_prev_mean: NotRequired[Tensor]
    x_prev_std: NotRequired[Tensor]
    logprob: NotRequired[Tensor]


class Sampler(ABC):
    name: str

    def __init__(self) -> None:
        pass

    @abstractmethod
    def set_steps(self, *args: Any, **kwargs: Any) -> None:
        """
        Set parameters based on the provided schedule.
        """

    @abstractmethod
    def step(self, *args: Any, **kwargs: Any) -> StepResult:
        """
        Perform one step of the reverse process
        """
        pass

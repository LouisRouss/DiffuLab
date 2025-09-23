from typing import NotRequired, Required, TypedDict

from torch import Tensor


class StepResult(TypedDict):
    x_prev: Required[Tensor]
    estimated_x0: Required[Tensor]
    x_prev_mean: NotRequired[Tensor]
    x_prev_std: NotRequired[Tensor]
    logprob: NotRequired[Tensor]

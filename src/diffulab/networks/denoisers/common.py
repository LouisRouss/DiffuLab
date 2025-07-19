from abc import ABC, abstractmethod
from typing import Any, NotRequired, Required, TypedDict

import torch.nn as nn
from torch import Tensor


class ModelInput(TypedDict, total=False):
    x: Required[Tensor]  # input tensor
    p: NotRequired[float]  # probabilily of label dropping
    y: NotRequired[Tensor]  # class labels
    context: NotRequired[Tensor]  # context information, can be text image etc


class ModelOutput(TypedDict, total=False):
    x: Required[Tensor]  # output tensor
    features: NotRequired[list[Tensor]]  # list of features from intermediate layers
    repa_features: NotRequired[list[Tensor]]  # list of features after projection for REPA alignment


class Denoiser(nn.Module, ABC):
    def __init__(self):
        super().__init__()  # type: ignore

    @abstractmethod
    def forward(self, x: Tensor, timesteps: Tensor, *args: Any, **kwargs: Any) -> ModelOutput:
        """
        Apply the model to an input batch.
        :param x: a [N x C x ...] Tensor of noisy image.
        :param timesteps: a 1-D batch of timesteps.
        :return: a dictionary containing "x" the tensor
        output and potentially other outputs such as intermediate features.
        """

from abc import ABC, abstractmethod
from typing import Any, NotRequired, Required, TypedDict

import torch.nn as nn
from torch import Tensor


class ModelInput(TypedDict, total=False):
    x: Required[Tensor]  # input tensor
    p: NotRequired[float]  # probabilily of label dropping
    y: NotRequired[Tensor]  # class labels
    context: NotRequired[Tensor]  # context information, can be text image etc
    x_context: NotRequired[Tensor]  # additional image context, will be concatenated to x


class ModelOutput(TypedDict, total=False):
    x: Required[Tensor]  # output tensor
    features: NotRequired[list[Tensor]]  # list of features from intermediate layers
    repa_features: NotRequired[list[Tensor]]  # list of features after projection for REPA alignment


class ModelInputGRPO(TypedDict, total=False):
    x: NotRequired[Tensor]  # noisy input tensor
    p: NotRequired[float]  # probabilily of label dropping
    context: Required[Tensor]  # context information, can be text image etc
    x_context: NotRequired[Tensor]  # additional image context, will be concatenated to x


class ExtraInputGRPO(TypedDict, total=False):
    captions: Required[list[str]]  # list of captions for the batch


class Denoiser(nn.Module, ABC):
    classifier_free: bool

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

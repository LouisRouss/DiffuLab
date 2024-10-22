from abc import ABC, abstractmethod

import torch.nn as nn
from torch import Tensor


class Denoiser(nn.Module, ABC):
    def __init__(self):
        super().__init__()  # type: ignore

    @abstractmethod
    def forward(self, x: Tensor, timesteps: Tensor, *args, **kwargs) -> Tensor:  # type: ignore
        """
        Apply the model to an input batch.
        :param x: a [N x C x ...] Tensor of noisy image.
        :param timesteps: a 1-D batch of timesteps.
        :return: an [N x C x ...] Tensor of outputs.
        """

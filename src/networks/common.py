from abc import ABC, abstractmethod

import torch.nn as nn
from torch import Tensor


class Denoiser(nn.Module, ABC):
    def __init__(self):
        super().__init__()  # type: ignore

    @abstractmethod
    def forward(
        self, x: Tensor, timesteps: Tensor, y: Tensor | None = None, context: Tensor | None = None, p: float = 0.0
    ) -> Tensor:
        """
        Apply the model to an input batch.
        :param x: a [N x C x ...] Tensor of noisy image.
        :param timesteps: a 1-D batch of timesteps.
        :param y: a [N x C x ...] Tensor of labels.
        :param context: a [N x C x ...] Tensor of context for CrossAttention, can be images, text etc...
        :param p: the probability of droping the ground-truth label in the label embedding.
        :return: an [N x C x ...] Tensor of outputs.
        """

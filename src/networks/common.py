from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch import Tensor


class Denoiser(nn.Module, ABC):
    def __init__(self):
        super().__init__()  # type: ignore

    @abstractmethod
    def forward(
        self,
        x: Tensor,
        timesteps: Tensor,
        y: Tensor | None = None,
        context: Tensor | None = None,
        p: float = 0.0,
    ) -> Tensor:
        """
        Apply the model to an input batch.
        :param x: a [N x C x ...] Tensor of noisy image.
        :param timesteps: a 1-D batch of timesteps.
        :param y: a [N x C x ...] Tensor of labels.
        :param context: a [N x C x ...] Tensor of context for CrossAttention, can be images, text etc...
        :param p: the probability of dropping the ground-truth label / context
        :return: an [N x C x ...] Tensor of outputs.
        """


class contextEmbedder(nn.Module, ABC):
    def __init__(self):
        super().__init__()  # type: ignore

    def drop_labels(
        self,
        labels: Tensor,
        p: float,
    ) -> Tensor:
        """
        Randomly drop labels from a batch.
        :param labels: an [N] tensor of labels.
        :param p: the probability of dropping a label.
        :return: an [N] tensor of modified labels.
        """
        return torch.where(torch.rand_like(labels) < p, self.num_classes, labels)

    @abstractmethod
    def forward(self, context: Tensor, p: float):
        """
        Apply the model to an input batch.
        :param context: a [N x C x ...] Tensor of context.
        :param p: the probability of dropping the context.
        :return: an [N x C x ...] Tensor of outputs.
        """
        pass

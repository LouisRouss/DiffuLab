from abc import ABC, abstractmethod

import torch
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


class contextEmbedder(nn.Module, ABC):
    def __init__(self):
        super().__init__()  # type: ignore

    @property
    @abstractmethod
    def n_output(self) -> int:
        """
        Represents the number of output embedding the embedder is returning.
        """
        pass

    @property
    @abstractmethod
    def output_size(self) -> tuple[tuple[int] | int, ...]:
        """
        Represents the dimension of each output embedding.
        """
        pass

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

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor


class ContextEmbedder(nn.Module, ABC):
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
    def forward(self, context: Any, p: float) -> tuple[Tensor, ...]:
        """
        Apply the model to an input batch.
        :param context: context, can be a tensor or a list of str for example.
        :param p: the probability of dropping the context.
        :return: an [N x C x ...] Tensor of outputs.
        """
        pass

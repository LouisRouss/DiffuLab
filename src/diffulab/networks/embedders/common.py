from abc import ABC, abstractmethod
from typing import Any

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

    @abstractmethod
    def drop_context(
        self,
        drop_context: Any,
        p: float,
    ) -> Any:
        """
        Randomly drop drop_context from a batch.
        :param drop_context: an [N] sequence of context.
        :param p: the probability of dropping a context.
        :return: an [N] sequence of modified context.
        """
        pass

    @abstractmethod
    def forward(self, context: Any, p: float) -> tuple[Tensor, ...]:
        """
        Apply the model to an input batch.
        :param context: context, can be a tensor or a list of str for example.
        :param p: the probability of dropping the context.
        :return: an [N x C x ...] Tensor of outputs.
        """
        pass

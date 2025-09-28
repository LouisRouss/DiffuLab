from abc import ABC, abstractmethod
from typing import Any

import torch.nn as nn
from torch import Tensor


class ContextEmbedder(nn.Module, ABC):
    _n_output: int
    _output_size: tuple[int, ...]

    def __init__(self):
        super().__init__()  # type: ignore

    @property
    def n_output(self) -> int:
        """
        Represents the number of output embedding the embedder is returning.
        """
        return self._n_output

    @property
    def output_size(self) -> tuple[int, ...]:
        """
        Represents the dimension of each output embedding.
        """
        return self._output_size

    @abstractmethod
    def drop_conditions(
        self,
        context: Any,
        p: float,
    ) -> Any:
        """
        Randomly drop drop_context from a batch.

        Args:
            drop_context (Any): a sequence of context.
            p (float): the probability of dropping a context.
        Returns
            Any: the same sequence of context with some elements randomly dropped.
        """
        pass

    @abstractmethod
    def forward(self, context: Any, p: float) -> tuple[Tensor, ...]:
        """
        Apply the model to an input batch.

        Args:
            context (Any): the input batch, can be a tensor or a list of str for example.
            p (float): the probability of dropping the context.
        Returns:
            tuple[Tensor, ...]: a tuple of tensors representing the embeddings.
        """
        pass

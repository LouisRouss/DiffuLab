from typing import cast

import torch
from torch import Tensor

from diffulab.networks.embedders.common import ContextEmbedder


class PreComputedEmbedder(ContextEmbedder):
    """
    An embedder that does not load any model weights and expects precomputed embeddings to be provided.

    This embedder is useful when embeddings are computed offline or using a different service.

    Args:
        n_output (int): The number of output embeddings.
        output_size (tuple[int, ...]): The dimension of each output embedding.
    """

    def __init__(self, n_output: int, output_size: tuple[int, ...], zero_embedding: tuple[Tensor] | Tensor) -> None:
        super().__init__()
        self._n_output = n_output
        self._output_size = output_size
        self.zero_embedding = zero_embedding
        if isinstance(self.zero_embedding, Tensor):
            self.zero_embedding = self.zero_embedding.detach().requires_grad_(False)
        else:
            self.zero_embedding = tuple(z.detach().requires_grad_(False) for z in self.zero_embedding)

    def drop_conditions(
        self,
        context: tuple[Tensor] | Tensor,
        p: float,
    ) -> tuple[Tensor, ...]:
        """
        Randomly drop context from a batch.

        Args:
            drop_context (Any): a sequence of context.
            p (float): the probability of dropping a context.
        Returns
            Tensor | tuple[Tensor]: the same sequence of context with some elements randomly dropped.
             and replaced by zero embeddings.
        """
        if isinstance(context, Tensor):
            assert isinstance(self.zero_embedding, Tensor)
            mask = (torch.rand(context.shape[0], device=context.device) > p).float().unsqueeze(-1)
            return cast(tuple[Tensor], (context * mask + self.zero_embedding.to(context.device) * (1 - mask),))
        else:
            assert len(context) == self._n_output
            assert isinstance(self.zero_embedding, tuple)
            batch_size = context[0].shape[0]
            mask = (torch.rand(batch_size, device=context[0].device) > p).float().unsqueeze(-1)
            return cast(
                tuple[Tensor],
                tuple(c * mask + ze.to(c.device) * (1 - mask) for c, ze in zip(context, self.zero_embedding)),
            )  # type: ignore[reportUnknownArgumentType]

    @torch.inference_mode()
    def forward(self, context: tuple[Tensor] | Tensor, p: float) -> tuple[Tensor, ...]:
        """
        Apply the model to an input batch.

        Args:
            context (Tensor | tuple[Tensor]): the input batch of precomputed embeddings.
            p (float): the probability of dropping the context.
        Returns:
            Tensor | tuple[Tensor]: a tensor or a tuple of tensors representing the embeddings.
        """
        if p > 0:
            return self.drop_conditions(context, p)
        return context if isinstance(context, tuple) else (context,)

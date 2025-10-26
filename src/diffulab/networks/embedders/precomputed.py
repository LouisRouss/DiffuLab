import torch
from jaxtyping import Float
from torch import Tensor

from diffulab.networks.embedders.common import ContextEmbedder, ContextEmbedderOutput


class PreComputedEmbedder(ContextEmbedder):
    """
    An embedder that does not load any model weights and expects precomputed embeddings to be provided.

    This embedder is useful when embeddings are computed offline or using a different service.

    Args:
        n_output (int): The number of output embeddings.
        output_size (tuple[int, ...]): The dimension of each output embedding.
        zero_embedding (Float[Tensor, "seq_len dim"]): A tensor representing the zero embedding to use when dropping context.

    """

    def __init__(
        self, n_output: int, output_size: tuple[int, ...], zero_embedding: Float[Tensor, "seq_len dim"]
    ) -> None:
        super().__init__()
        self._n_output = n_output
        self._output_size = output_size
        self.zero_embedding = zero_embedding

    def drop_conditions(
        self,
        context: ContextEmbedderOutput,
        p: float,
    ) -> ContextEmbedderOutput:
        """
        Randomly drop context from a batch.

        Args:
            context (ContextEmbedderOutput): Context embeddings.
            p (float): the probability of dropping a context.
        Returns
            ContextEmbedderOutput: the same context embeddings with some elements randomly dropped.
        """
        embedding = context["embeddings"]
        batch_size, seq_len = embedding.shape[:2]
        mask = (torch.rand(batch_size, device=embedding.device) > p).float()[:, None, None]

        # first handle sequence embeddings
        zero_emb = torch.cat(
            [
                self.zero_embedding,
                torch.zeros(
                    seq_len - self.zero_embedding.shape[0],
                    self.zero_embedding.shape[1],
                    device=self.zero_embedding.device,
                ),
            ],
            dim=0,
        )
        embedding = embedding * mask + zero_emb.unsqueeze(0) * (torch.ones_like(mask) - mask)
        attn_mask = context.get("attn_mask") or torch.ones(batch_size, embedding.shape[1], device=embedding.device)
        attn_mask[mask.squeeze() == 0, self.zero_embedding.shape[0] :] = (
            0.0  # disable attention to padding tokens for dropped contexts
        )
        return {
            "embeddings": embedding,
            "attn_mask": attn_mask,
        }

    @torch.inference_mode()
    def forward(self, context: ContextEmbedderOutput, p: float = 0) -> ContextEmbedderOutput:
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
        return context

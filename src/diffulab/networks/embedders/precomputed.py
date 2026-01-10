from pathlib import Path

import torch

from diffulab.networks.embedders.common import ContextEmbedder, ContextEmbedderOutput


class PrecomputedEmbedder(ContextEmbedder):
    def __init__(self, path_null_embedding: Path | str, null_embedding_seq_len: int) -> None:
        super().__init__()
        self.null_embedding = torch.load(path_null_embedding).squeeze()
        self.null_embedding_mask = torch.cat(
            [
                torch.ones(null_embedding_seq_len, dtype=torch.bool),
                torch.zeros(self.null_embedding.shape[0] - null_embedding_seq_len, dtype=torch.bool),
            ],
            dim=0,
        )
        self._output_size = (self.null_embedding.shape[-1],)  # type: ignore
        self._n_output = 1

    def drop_conditions(self, context: ContextEmbedderOutput, p: float) -> ContextEmbedderOutput:
        batch_size = context["embeddings"].shape[0]
        device, dtype = context["embeddings"].device, context["embeddings"].dtype

        drop_mask = torch.rand(batch_size, device=device) < p
        null_emb = self.null_embedding.to(device=device, dtype=dtype)
        null_mask = self.null_embedding_mask.to(device=device)

        embeddings = torch.where(
            drop_mask[:, None, None], null_emb.unsqueeze(0).expand(batch_size, -1, -1), context["embeddings"]
        )
        attn_mask = torch.where(
            drop_mask[:, None],
            null_mask.unsqueeze(0).expand(batch_size, -1),
            context["attn_mask"],  # type: ignore
        )

        return {"embeddings": embeddings, "attn_mask": attn_mask}

    def forward(self, context: ContextEmbedderOutput, p: float = 0) -> ContextEmbedderOutput:
        dropped_context = self.drop_conditions(context, p)
        return dropped_context

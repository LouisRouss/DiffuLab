import gc
import random
from typing import TYPE_CHECKING, cast

import torch
from jaxtyping import Bool, Float
from open_clip import create_model_from_pretrained, get_tokenizer  # type: ignore
from torch import Tensor
from transformers import AutoTokenizer, CLIPTextModel, T5EncoderModel, T5Tokenizer

from diffulab.networks.embedders.common import ContextEmbedder, ContextEmbedderOutput

if TYPE_CHECKING:
    from open_clip import CLIP, SimpleTokenizer  # type: ignore
    from transformers.models.clip.tokenization_clip_fast import CLIPTokenizerFast


class SD3TextEmbedder(ContextEmbedder):
    """
    Composite text embedder combining CLIP-L/14, CLIP-bigG/14 and T5 XXL.

    This class mirrors the text conditioning strategy used by Stable Diffusion 3
    where multiple powerful language / text encoders are fused. Each encoder can
    be toggled on/off at construction.

    The forward path returns two tensors:

    * pooled: Concatenated pooled CLIP embeddings of shape ``[B, 2048]``
      (``768 + 1280``) when both CLIP encoders are enabled.
    * full_encoding: A composite sequence embedding formed by (a) concatenating
      the token-level hidden states of CLIP-L/14 and CLIP-bigG/14 along the last
      dimension with zero padding to match the T5 hidden size (4096), then (b)
      concatenating that padded CLIP block with the T5 hidden states along the
      sequence dimension. Resulting shape is ``[B, N_clip + N_t5, 4096]`` where
      ``N_clip`` is the (padded) CLIP token length and ``N_t5`` the T5 token length.

    Args:
        device (str | torch.device): Torch device or device string for model instantiation & tensors.

    """

    def __init__(self, device: str | torch.device = "cuda") -> None:
        super().__init__()
        self.device = device
        self._output_size = (2048, 4096)  # pooled, full_encoding
        self._n_output = 2

        # load l_14
        self.clip_l14 = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", device_map=device)  # type: ignore
        self.tokenizer_l14: CLIPTokenizerFast = AutoTokenizer.from_pretrained(  # type: ignore
            "openai/clip-vit-large-patch14", device_map=device
        )
        self.clip_l14.eval()
        self.clip_l14.requires_grad_(False)

        # load g_14
        self.clip_g14 = cast(
            CLIP,
            create_model_from_pretrained("hf-hub:laion/CLIP-ViT-bigG-14-laion2B-39B-b160k", device=device)[0],  # type: ignore
        )
        self.tokenizer_g14 = cast(SimpleTokenizer, get_tokenizer("hf-hub:laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"))
        del self.clip_g14.visual
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.clip_g14.eval()
        self.clip_g14.requires_grad_(False)

        # load t5
        self.t5 = T5EncoderModel.from_pretrained("google/t5-v1_1-xl", from_tf=True)  # type: ignore
        self.tokenizer_t5 = cast(
            T5Tokenizer,
            T5Tokenizer.from_pretrained(  # type: ignore
                "google/t5-v1_1-xl", clean_up_tokenization_spaces=True, legacy=False
            ),
        )
        self.t5.eval()  # pyright: ignore[reportUnknownMemberType]
        self.t5.requires_grad_(False)  # pyright: ignore[reportUnknownMemberType]

    def get_l14_embeddings(
        self, context: list[str]
    ) -> tuple[
        Float[Tensor, "batch_size seq_len 768"], Float[Tensor, "batch_size 768"], Bool[Tensor, "batch_size seq_len"]
    ]:
        """Compute CLIP ViT-L/14 token and pooled embeddings. Return attention mask too.

        Args:
            context (list[str]): List of input prompt strings.

        Returns:
            tuple[Tensor, Tensor]:
                * last_hidden_state: Shape ``[B, N_ctx, 768]``
                * pooled_output: Shape ``[B, 768]`` (CLIP text projection output)
                * attention_mask: Shape ``[B, N_ctx]``
        """
        inputs_l14 = self.tokenizer_l14(context, return_tensors="pt", padding=True).to(self.device)
        outputs_l14 = self.clip_l14(**inputs_l14)
        last_hidden_state = outputs_l14["last_hidden_state"]  # [batch_size, n_ctx, 768]
        pooled_output = outputs_l14["pooler_output"]  # [batch_size, 768]
        attention_mask = cast(
            Tensor,
            inputs_l14.attention_mask.bool(),  # type: ignore
        )  # [batch_size, n_ctx]
        return last_hidden_state, pooled_output, attention_mask

    def get_g14_embeddings(
        self, context: list[str]
    ) -> tuple[
        Float[Tensor, "batch_size seq_len 1280"], Float[Tensor, "batch_size 1280"], Bool[Tensor, "batch_size seq_len"]
    ]:
        """Compute CLIP ViT-bigG/14 token and pooled embeddings.

        Args:
            context (list[str]): List of input prompt strings.

        Returns:
            tuple[Tensor, Tensor]:
                * last_hidden_state: Shape ``[B, N_ctx, 1280]``
                * pooled_output: Shape ``[B, 1280]``
                * attention_mask: Shape ``[B, N_ctx]``
        """
        inputs_g14 = self.tokenizer_g14(context).to(self.device)
        x = self.clip_g14.token_embedding(inputs_g14)  # type: ignore # [batch_size, n_ctx, d_model]
        x = x + self.clip_g14.positional_embedding  # type: ignore
        x = self.clip_g14.transformer(x, attn_mask=self.clip_g14.attn_mask)  # type: ignore
        last_hidden_state = cast(Tensor, self.clip_g14.ln_final(x))  # type: ignore # [batch_size, n_ctx, 1280]

        eot_positions = (inputs_g14 == self.tokenizer_g14.eot_token_id).float().argmax(dim=1)  # [batch_size]
        _, seq_len = inputs_g14.shape
        arange = torch.arange(seq_len, device=inputs_g14.device).unsqueeze(0)  # [1, seq_len]
        attention_mask = (arange <= eot_positions.unsqueeze(1)).bool()

        # Max pooling
        mask = attention_mask.unsqueeze(-1)  # [B, N, 1]
        neg_inf = torch.finfo(last_hidden_state.dtype).min
        masked_hidden = last_hidden_state.masked_fill(~mask, neg_inf)
        pooled_output = masked_hidden.max(dim=1).values

        return last_hidden_state, pooled_output, attention_mask

    def get_t5_embeddings(
        self, context: list[str]
    ) -> tuple[Float[Tensor, "batch_size seq_len 4096"], Bool[Tensor, "batch_size seq_len"]]:
        """Compute T5 XXL token-level embeddings.

        Args:
            context (list[str]): List of input prompt strings.

        Returns:
            tuple[Tensor, Tensor]:
                * last_hidden_state: Shape ``[B, N_ctx, 4096]``
                * attention_mask: Shape ``[B, N_ctx]``

        """
        inputs_t5_list: dict[str, list[int]] = self.tokenizer_t5(context, padding=True)  # type: ignore
        inputs_t5: dict[str, Tensor] = {
            key: torch.tensor(value, dtype=torch.long, device=self.device)
            for key, value in inputs_t5_list.items()  # type: ignore
        }
        last_hidden_state: Tensor = self.t5(**inputs_t5)["last_hidden_state"]  # [batch_size, n_ctx, 4096]
        attention_mask = inputs_t5["attention_mask"].bool()  # [batch_size, n_ctx]
        return last_hidden_state, attention_mask

    def get_embeddings(self, context: list[str]) -> dict[str, Tensor]:
        """Obtain embeddings

        Args:
            context (list[str]): List of input prompt strings.

        Returns:
            dict[str, Tensor]: Mapping with keys among ``{"l14", "l14_pooled", "g14", "g14_pooled", "t5"}``.
        """
        assert self.loaded_weights, "Cannot get embeddings when model weights are not loaded."
        outputs: dict[str, Tensor] = {}

        outputs_l14 = self.get_l14_embeddings(context)
        outputs["l14"] = outputs_l14[0]
        outputs["l14_pooled"] = outputs_l14[1]
        outputs["l14_attn_mask"] = outputs_l14[2]

        outputs_g14 = self.get_g14_embeddings(context)
        outputs["g14"] = outputs_g14[0]
        outputs["g14_pooled"] = outputs_g14[1]
        outputs["g14_attn_mask"] = outputs_g14[2]

        outputs_t5 = self.get_t5_embeddings(context)
        outputs["t5"] = outputs_t5[0]
        outputs["t5_attn_mask"] = outputs_t5[1]
        return outputs

    def drop_conditions(self, context: list[str], p: float) -> list[str]:
        """Randomly drop prompts for classifier-free guidance style training.

        Args:
            context (list[str]): Original list of prompt strings.
            p (float): Drop probability.

        Returns:
            list[str]: context with some entries replaced by empty strings.
        """
        return ["" if random.random() < p else c for c in context]

    def forward(self, context: list[str], p: float = 0) -> ContextEmbedderOutput:
        """Forward pass producing pooled and composite sequence embeddings.

        Args:
            context: List of input prompt strings.
            p: Probability to drop each prompt (per encoder) for classifier-free guidance.

        Returns:
            ContextEmbedderOutput: Dictionary with keys:
                * embeddings: Composite sequence embeddings, shape ``[B, N_clip + N_t5, 4096]``
                * pooled_embeddings: Concatenated CLIP pooled embeddings, shape ``[B, 2048]``
                * attn_mask: Attention mask for the composite embeddings, shape ``[B, N_clip + N_t5]``
        """
        context = self.drop_conditions(context, p)
        embeddings = self.get_embeddings(context)
        pooled = torch.cat([embeddings["l14_pooled"], embeddings["g14_pooled"]], dim=-1)
        full_encoding_clip = torch.cat(
            [
                embeddings["l14"],
                embeddings["g14"],
                torch.zeros(embeddings["t5"].shape[-1] - (embeddings["l14"].shape[-1] + embeddings["g14"].shape[-1])),
            ],
            dim=-1,
        )
        assert embeddings["l14_attn_mask"] == embeddings["g14_attn_mask"], (
            "Mismatched CLIP attention masks."
        )  # should be same tokenizer
        full_encoding = torch.cat([full_encoding_clip, embeddings["t5"]], dim=-2)
        attention_mask = torch.cat([embeddings["l14_attn_mask"], embeddings["t5_attn_mask"]], dim=-1).bool()
        return ContextEmbedderOutput(embeddings=full_encoding, pooled_embeddings=pooled, attn_mask=attention_mask)

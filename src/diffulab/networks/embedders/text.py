import gc
import random
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from jaxtyping import Float
from open_clip import create_model_from_pretrained, get_tokenizer  # type: ignore
from torch import Tensor
from transformers import AutoModel, AutoTokenizer, CLIPTextModel, T5EncoderModel

from diffulab.networks.embedders.common import ContextEmbedder

if TYPE_CHECKING:
    from transformers import Qwen2TokenizerFast, Qwen3Model


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
        self.tokenizer_l14 = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14", device_map=device)  # type: ignore

        # load g_14
        self.clip_g14 = create_model_from_pretrained(  # type: ignore
            "hf-hub:laion/CLIP-ViT-bigG-14-laion2B-39B-b160k", device=device
        )[0]  # type: ignore
        self.tokenizer_g14 = get_tokenizer("hf-hub:laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")
        if hasattr(self.clip_g14, "visual"):  # type: ignore
            del self.clip_g14.visual  # type: ignore
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # load t5
        self.t5 = T5EncoderModel.from_pretrained("google/t5-v1_1-xl", from_tf=True)  # type: ignore
        self.tokenizer_t5 = AutoTokenizer.from_pretrained(  # type: ignore
            "google/t5-v1_1-xl", clean_up_tokenization_spaces=True, legacy=False
        )

    def dict_to_device(self, d: dict[str, Tensor]) -> dict[str, Tensor]:
        """Move all tensors in a dictionary to the configured device.

        Args:
            d (dict[str, Tensor]): Mapping of string keys to tensors.

        Returns:
            dict[str, Tensor]: Same mapping with tensors moved to ``self.device``.
        """
        return {k: v.to(self.device) for k, v in d.items()}

    def get_l14_embeddings(
        self, context: list[str]
    ) -> tuple[Float[Tensor, "batch_size seq_len 768"], Float[Tensor, "batch_size 768"]]:
        """Compute CLIP ViT-L/14 token and pooled embeddings.

        Args:
            context (list[str]): List of input prompt strings.

        Returns:
            tuple[Tensor, Tensor]:
                * last_hidden_state: Shape ``[B, N_ctx, 768]``
                * pooled_output: Shape ``[B, 768]`` (CLIP text projection output)
        """
        inputs_l14 = self.dict_to_device(self.tokenizer_l14(context, return_tensors="pt", padding=True))  # type: ignore
        outputs_l14 = self.clip_l14(**inputs_l14)  # type: ignore
        last_hidden_state = outputs_l14["last_hidden_state"]  # [batch_size, n_ctx, 768]
        pooled_output = outputs_l14["pooler_output"]  # [batch_size, 768]
        return last_hidden_state, pooled_output

    def get_g14_embeddings(
        self, context: list[str]
    ) -> tuple[Float[Tensor, "batch_size seq_len 1280"], Float[Tensor, "batch_size 1280"]]:
        """Compute CLIP ViT-bigG/14 token and pooled embeddings.

        Args:
            context (list[str]): List of input prompt strings.

        Returns:
            tuple[Tensor, Tensor]:
                * last_hidden_state: Shape ``[B, N_ctx, 1280]``
                * pooled_output: Shape ``[B, 1280]`` (selection at EOS token)
        """
        inputs_g14 = self.tokenizer_g14(context).to(self.device)
        cast_dtype = self.clip_g14.get_cast_dtype()  # type: ignore

        x = self.clip_g14.token_embedding(inputs_g14).to(cast_dtype)  # [batch_size, n_ctx, d_model] # type: ignore
        x = x + self.clip_g14.positional_embedding.to(cast_dtype)  # type: ignore
        x = self.clip_g14.transformer(x, attn_mask=self.attn_mask)  # type: ignore
        last_hidden_state: Tensor = self.clip_g14.ln_final(x)  # [batch_size, n_ctx, 1280] # type: ignore
        pooled_output: Tensor = last_hidden_state[  # type: ignore
            torch.arange(last_hidden_state.shape[0]),  # type: ignore
            inputs_g14.to(dtype=torch.int, device=self.device).argmax(dim=-1),
        ]

        return last_hidden_state, pooled_output  # type: ignore

    def get_t5_embeddings(self, context: list[str]) -> Float[Tensor, "batch_size seq_len 4096"]:
        """Compute T5 XXL token-level embeddings.

        Args:
            context (list[str]): List of input prompt strings.

        Returns:
            Tensor: Last hidden state of shape ``[B, N_ctx, 4096]``.
        """
        inputs_t5_list: dict[str, list[int]] = self.tokenizer_t5(context)  # type: ignore
        inputs_t5: dict[str, Tensor] = {
            key: torch.tensor(value, dtype=torch.long, device=self.device)
            for key, value in inputs_t5_list.items()  # type: ignore
        }
        last_hidden_state: Tensor = self.t5(**inputs_t5)["last_hidden_state"]  # [batch_size, n_ctx, 4096]
        return last_hidden_state

    def get_embeddings(self, context: list[str]) -> dict[str, Tensor]:
        """Obtain embeddings

        Args:
            context (list[str]): List of input prompt strings.

        Returns:
            dict[str, Tensor]: Mapping with keys among ``{"l14", "l14_pooled", "g14", "g14_pooled", "t5"}``.
        """
        assert self.loaded_weights, "Cannot get embeddings when model weights are not loaded."
        outputs: dict[str, Tensor] = {}

        outputs_l14 = self.get_l14_embeddings(context)  # [batch_size, n_ctx, 768]
        outputs["l14"] = outputs_l14[0]
        outputs["l14_pooled"] = outputs_l14[1]

        outputs_g14 = self.get_g14_embeddings(context)  # [batch_size, n_ctx, 1280]
        outputs["g14"] = outputs_g14[0]
        outputs["g14_pooled"] = outputs_g14[1]

        outputs_t5 = self.get_t5_embeddings(context)  # [batch_size, n_ctx, 4096]
        outputs["t5"] = outputs_t5
        return outputs

    def precompute_embeddings_from_list(
        self, context: list[str], batch_size: int = 8, path_to_save: str | Path = Path.home() / "saved_embeddings"
    ) -> None:
        """Compute embeddings for a list of prompts.

        Each prompt's pooled CLIP embedding and composite full encoding are
        stored separately as ``<index>.pt`` along with a ``<index>.txt`` file
        containing the raw prompt string to allow later reconstruction.

        Existing cached indices are detected so new prompts append seamlessly.

        Args:
            context: List of prompt strings to encode.
            batch_size: Number of prompts per forward pass. ``-1`` processes all at once.
            path_to_save: Directory where embeddings will be written / appended.
        """
        path_to_save = Path(path_to_save)
        path_to_save.mkdir(parents=True, exist_ok=True)

        # Check for existing embeddings and set the starting index
        existing_files = list(path_to_save.glob("*.pt"))
        if existing_files:
            existing_indices = [int(f.stem) for f in existing_files if f.stem.isdigit()]
            begins = max(existing_indices) + 1
        else:
            begins = 0

        if batch_size == -1:
            batch_size = len(context)

        for i in range(0, len(context), batch_size):
            embeddings = self.forward(context[i : i + batch_size], 0.0)
            for b in range(batch_size):
                torch.save(  # type: ignore
                    {"pooled": embeddings[0][b], "full_encoding": embeddings[1][b]},
                    path_to_save / f"{begins + i + b}.pt",
                )
                with (path_to_save / f"{begins + i + b}.txt").open("r") as f:
                    f.write(context[i + b])

    def drop_conditions(self, context: list[str], p: float) -> list[str]:
        """Randomly drop prompts for classifier-free guidance style training.

        Args:
            context (list[str]): Original list of prompt strings.
            p (float): Drop probability.

        Returns:
            list[str]: context with some entries replaced by empty strings.
        """
        return ["" if random.random() < p else c for c in context]

    def forward(
        self, context: list[str], p: float
    ) -> tuple[Float[Tensor, "batch_size 2048"], Float[Tensor, "batch_size seq_len 4096"]]:
        """Forward pass producing pooled and composite sequence embeddings.

        Args:
            context: List of input prompt strings.
            p: Probability to drop each prompt (per encoder) for classifier-free guidance.

        Returns:
            tuple[Tensor, Tensor]:
                * pooled: Shape ``[B, 2048]`` concatenated pooled CLIP features.
                * full_encoding: Shape ``[B, seq_len, 4096]`` composite sequence embedding.
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
        full_encoding = torch.cat([full_encoding_clip, embeddings["t5"]], dim=-2)
        return pooled, full_encoding


class QwenEmbedding(ContextEmbedder):
    model_registry: dict[str, int] = {
        "Qwen/Qwen3-Embedding-0.6B": 1024,
        "Qwen/Qwen3-Embedding-4B": 2560,
        "Qwen/Qwen3-Embedding-8B": 4096,
    }

    def __init__(self, device: str | torch.device = "cuda", model_id: str = "Qwen/Qwen3-Embedding-4B") -> None:
        super().__init__()
        assert model_id in self.model_registry, f"Model {model_id} not in registry {list(self.model_registry.keys())}"
        self.tokenizer: "Qwen2TokenizerFast" = AutoTokenizer.from_pretrained(model_id, padding_side="left")  # type: ignore[reportUnknownMemberType]
        self.model: "Qwen3Model" = AutoModel.from_pretrained(model_id)  # type: ignore[reportUnknownMemberType]
        self._n_output = 1
        self._output_size = (self.model_registry[model_id],)

    @staticmethod
    def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def get_embeddings(self, context: list[str]) -> Tensor:
        tokenized_context = self.tokenizer(
            context,
            padding=True,
            truncation=True,
            max_length=8192,
            return_tensors="pt",
        )
        tokenized_context.to(self.model.device)
        outputs = self.model(**tokenized_context)
        embeddings = self.last_token_pool(outputs.last_hidden_state, tokenized_context["attention_mask"])  # type: ignore[reportArgumentType]
        return embeddings

    def drop_conditions(self, context: list[str], p: float) -> list[str]:
        """Randomly drop prompts for classifier-free guidance style training.

        Args:
            context (list[str]): Original list of prompt strings.
            p (float): Drop probability.

        Returns:
            list[str]: context with some entries replaced by empty strings.
        """
        return ["" if random.random() < p else c for c in context]

    def forward(self, context: list[str], p: float) -> tuple[Float[Tensor, "batch_size dim"]]:
        """Forward pass producing pooled and composite sequence embeddings.

        Args:
            context: List of input prompt strings.
            p: Probability to drop each prompt (per encoder) for classifier-free guidance.

        Returns:
            tuple[Tensor, Tensor]:
                * pooled: Shape ``[B, 2048]`` concatenated pooled CLIP features.
                * full_encoding: Shape ``[B, seq_len, 4096]`` composite sequence embedding.
        """
        context = self.drop_conditions(context, p)
        embeddings = self.get_embeddings(context)
        return (embeddings,)

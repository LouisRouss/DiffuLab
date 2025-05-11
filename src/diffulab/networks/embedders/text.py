import gc
import random
from pathlib import Path

import torch
from open_clip import create_model_from_pretrained, get_tokenizer  # type: ignore
from torch import Tensor
from transformers import AutoTokenizer, CLIPTextModel, T5EncoderModel  # type: ignore

from diffulab.networks.embedders.common import ContextEmbedder


class SD3TextEmbedder(ContextEmbedder):
    def __init__(
        self, device: str | torch.device = "cuda", load_l14: bool = True, load_g14: bool = True, load_t5: bool = True
    ) -> None:
        super().__init__()
        self.device = device

        self.load_l14 = load_l14
        if load_l14:
            self.clip_l14 = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", device_map=device)  # type: ignore
            self.tokenizer_l14 = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14", device_map=device)  # type: ignore

        self.load_g14 = load_g14
        if load_g14:
            self.clip_g14 = create_model_from_pretrained(  # type: ignore
                "hf-hub:laion/CLIP-ViT-bigG-14-laion2B-39B-b160k", device=device
            )[0]  # type: ignore
            self.tokenizer_g14 = get_tokenizer("hf-hub:laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")
            if hasattr(self.clip_g14, "visual"):  # type: ignore
                del self.clip_g14.visual  # type: ignore
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        self.load_t5 = load_t5
        if load_t5:
            self.t5 = T5EncoderModel.from_pretrained("google/t5-v1_1-xxl")  # type: ignore
            self.tokenizer_t5 = AutoTokenizer.from_pretrained(  # type: ignore
                "google/t5-v1_1-xxl", clean_up_tokenization_spaces=True, legacy=False
            )

        self.outputs_size = (2048, 4096)

    @property
    def n_output(self) -> int:
        return 2

    @property
    def output_size(self) -> tuple[int, int]:
        return (2048, 4096)

    def dict_to_device(self, d: dict[str, Tensor]) -> dict[str, Tensor]:
        return {k: v.to(self.device) for k, v in d.items()}

    def get_l14_embeddings(self, context: list[str]) -> tuple[Tensor, Tensor]:
        inputs_l14 = self.dict_to_device(self.tokenizer_l14(context, return_tensors="pt", padding=True))  # type: ignore
        outputs_l14 = self.clip_l14(**inputs_l14)  # type: ignore
        last_hidden_state = outputs_l14["last_hidden_state"]  # [batch_size, n_ctx, 768]
        pooled_output = outputs_l14["pooler_output"]  # [batch_size, 768]
        return last_hidden_state, pooled_output

    def get_g14_embeddings(self, context: list[str]) -> Tensor:
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

    def get_t5_embeddings(self, context: list[str]) -> Tensor:
        inputs_t5_list: dict[str, list[int]] = self.tokenizer_t5(context)  # type: ignore
        inputs_t5: dict[str, Tensor] = {
            key: torch.tensor(value, dtype=torch.long, device=self.device)
            for key, value in inputs_t5_list.items()  # type: ignore
        }
        last_hidden_state: Tensor = self.t5(**inputs_t5)["last_hidden_state"]  # [batch_size, n_ctx, 4096]
        return last_hidden_state

    def get_embeddings(
        self, context_l14: list[str], context_g14: list[str], context_t5: list[str]
    ) -> dict[str, Tensor]:
        outputs: dict[str, Tensor] = {}
        if self.load_l14:
            outputs_l14 = self.get_l14_embeddings(context_l14)  # [batch_size, n_ctx, 768]
            outputs["l14"] = outputs_l14[0]
            outputs["l14_pooled"] = outputs_l14[1]
        if self.load_g14:
            outputs_g14 = self.get_g14_embeddings(context_g14)  # [batch_size, n_ctx, 1280]
            outputs["g14"] = outputs_g14[0]
            outputs["g14_pooled"] = outputs_g14[1]
        if self.load_t5:
            outputs_t5 = self.get_t5_embeddings(context_t5)  # [batch_size, n_ctx, 4096]
            outputs["t5"] = outputs_t5
        return outputs

    def create_cache_from_list(
        self, context: list[str], batch_size: int = 8, path_to_save: str | Path = Path.home() / "saved_embeddings"
    ) -> None:
        path_to_save = Path(path_to_save)
        path_to_save.mkdir(parents=True, exist_ok=True)

        # Check for existing embeddings and set the starting index
        existing_files = list(path_to_save.glob("*.pth"))
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
                    path_to_save / f"{begins + i + b}.pth",
                )
                with (path_to_save / f"{begins + i + b}.txt").open("r") as f:
                    f.write(context[i + b])

    def drop_conditions(self, context: list[str], p: float) -> tuple[list[str], list[str], list[str]]:
        context_l14 = ["" if random.random() < p else c for c in context]
        context_g14 = ["" if random.random() < p else c for c in context]
        context_t5 = ["" if random.random() < p else c for c in context]
        return context_l14, context_g14, context_t5

    def forward(self, context: list[str], p: float) -> tuple[Tensor, Tensor]:
        assert self.load_l14 and self.load_g14 and self.load_t5
        context_l14, context_g14, context_t5 = self.drop_conditions(context, p)
        embeddings = self.get_embeddings(context_l14, context_g14, context_t5)
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

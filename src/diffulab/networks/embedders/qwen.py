import random
from typing import cast

import torch
from torch import Tensor
from transformers import Qwen2Tokenizer, Qwen3VLForConditionalGeneration

from diffulab.networks.embedders.common import ContextEmbedder, ContextEmbedderOutput


class QwenTextEmbedder(ContextEmbedder):
    def __init__(
        self, model_id: str = "Qwen/Qwen3-VL-2B-Instruct", device: str | torch.device = "cuda", max_length: int = 1024
    ) -> None:
        super().__init__()
        self.model: Qwen3VLForConditionalGeneration = Qwen3VLForConditionalGeneration.from_pretrained(  # type: ignore
            model_id, device_map=device, dtype="auto"
        )
        self.model.eval()
        self.model.requires_grad_(False)
        self.tokenizer: Qwen2Tokenizer = Qwen2Tokenizer.from_pretrained(model_id, device_map=device)  # type: ignore
        self.prompt_template: str = self._get_prompt_template()
        self.prompt_template_encoder_start_idx = 34
        self.max_length = max_length

        self._output_size: tuple[int] = (self.model.config.text_config.hidden_size,)
        self._n_output = 1

    def _get_prompt_template(self) -> str:
        return (
            "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, "
            "spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    def drop_conditions(self, context: list[str], p: float) -> list[str]:
        """Randomly drop prompts for classifier-free guidance style training.

        Args:
            context (list[str]): Original list of prompt strings.
            p (float): Drop probability.

        Returns:
            list[str]: context with some entries replaced by empty strings.
        """
        return ["" if random.random() < p else c for c in context]

    @torch._dynamo.disable  # type: ignore[reportUnknownMemberType]
    def forward(self, context: list[str], p: float = 0) -> ContextEmbedderOutput:
        """Compute Qwen text embeddings.

        Args:
            context (list[str]): List of input prompt strings.
            p (float): Probability of dropping the condition (replacing with empty string).

        Returns:
            ContextEmbedderOutput:
                Sequence embeddings of shape ``[B, N_ctx, D]`` where ``N_ctx`` is the number of tokens after prompt template
                and ``D`` is the model hidden size.
                Attention mask of shape ``[B, N_ctx]`` indicating non-padding tokens.

        """
        context_dropped = self.drop_conditions(context, p)
        prompt_texts = [self.prompt_template.format(c) for c in context_dropped]
        tokens = self.tokenizer(
            prompt_texts,
            max_length=self.max_length + self.prompt_template_encoder_start_idx,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.model.device)

        embeddings = cast(
            Tensor,
            self.model(
                input_ids=tokens.input_ids,  # type: ignore
                attention_mask=tokens.attention_mask,  # type: ignore
                output_hidden_states=True,
            ).hidden_states[-1],
        )

        embeddings = embeddings[:, self.prompt_template_encoder_start_idx :, :]
        attn_mask = tokens.attention_mask[:, self.prompt_template_encoder_start_idx :]  # type: ignore

        return ContextEmbedderOutput(
            embeddings=embeddings,
            attn_mask=attn_mask,  # type: ignore
        )

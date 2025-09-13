import itertools
import re
from typing import TYPE_CHECKING, Literal, TypedDict, Union, cast  # added TypedDict related imports

import numpy as np
import torch
from einops import rearrange
from jaxtyping import Float
from numpy.typing import NDArray
from PIL import Image
from qwen_vl_utils import process_vision_info  # type: ignore[reportUnknownVariableType]
from torch import Tensor
from transformers import (  # type: ignore[reportMissingTypeStubs]
    AutoProcessor,
    AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration,
)

from diffulab.networks.rewards.common import RewardModel

if TYPE_CHECKING:
    from transformers import Qwen2_5_VLProcessor, Qwen2TokenizerFast  # type: ignore[reportMissingTypeStubs]


class ImageContent(TypedDict):
    type: Literal["image"]
    image: Image.Image


class TextContent(TypedDict):
    type: Literal["text"]
    text: str


ContentItem = Union[ImageContent, TextContent]


class ChatMessage(TypedDict):
    role: Literal["user", "assistant", "system"]
    content: list[ContentItem]


# TODO: vectorize everything and optimize inference


class PrefGRPORewardModel(RewardModel):
    model_registry = {
        "cot_7b": "CodeGoat24/UnifiedReward-Think-qwen-7b",
        "3b": "CodeGoat24/UnifiedReward-2.0-qwen-3b",
        "7b": "CodeGoat24/UnifiedReward-2.0-qwen-7b",
        "32b": "CodeGoat24/UnifiedReward-2.0-qwen-32b",
        "72b": "CodeGoat24/UnifiedReward-2.0-qwen-72b",
    }

    def __init__(self, version: str = "7b", n_image_per_prompt: int = 16):
        super().__init__()
        self.use_cot = version.startswith("cot")
        self.version = version
        assert self.version in self.model_registry, (
            f"Unsupported model version: {self.version}, available versions: {list(self.model_registry.keys())}"
        )
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(  # type: ignore[reportUnknownMemberType]
            self.model_registry[self.version], dtype="auto", device_map="auto"
        )
        self.processor: Qwen2_5_VLProcessor = AutoProcessor.from_pretrained(self.model_registry[self.version])  # type: ignore[reportUnknownMemberType]
        self.tokenizer: Qwen2TokenizerFast = AutoTokenizer.from_pretrained(self.model_registry[self.version])  # type: ignore[reportUnknownMemberType]
        self.n_image_per_prompt = n_image_per_prompt

    def set_n_image_per_prompt(self, n: int) -> None:
        self.n_image_per_prompt = n

    @staticmethod
    def _extract_cot_answer(text: str) -> str | None:
        """Extract the content inside <answer>...</answer> tags (case-insensitive)."""
        match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

        return None

    @staticmethod
    def _parse_scores(raw: str) -> dict[str, dict[str, float]]:
        """
        Parse blocks like:
        Alignment Score:
            Image 1: 0.45
            Image 2: 0.54
        Coherence Score:
            ...
        into a nested dict: {section: {image_label: float, ...}, ...}

        Args:
            raw (str): The raw text to parse.
        Returns:
            dict[str, dict[str, float]]: A dictionary mapping section names to dictionaries of image scores.

        Example:
            ```python
            text = '''
            Alignment Score:
            Image 1: 0.4537
            Image 2: 0.5463

            Coherence Score:
            Image 1: 0.6000
            Image 2: 0.4000
            '''
            scores = parse_scores(text)
            # scores == {
            #   "Alignment Score": {"Image 1": 0.4537, "Image 2": 0.5463},
            #   "Coherence Score": {"Image 1": 0.6000, "Image 2": 0.4000}
            # }
            ```
        """

        def _unescape_newlines(s: str) -> str:
            """
            Un-escape newline characters in a string, if any.
            E.g. "Hello\\nWorld" -> "Hello
            World"
            """
            return bytes(s, "utf-8").decode("unicode_escape") if "\\n" in s else s

        text = _unescape_newlines(raw).strip()
        sections: dict[str, dict[str, float]] = {}
        current = None

        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue

            m = re.match(r"^(.*?\S)\s*:\s*$", line)
            if m and ("Score" in m.group(1)):
                current = m.group(1)
                sections[current] = {}
                continue

            m = re.match(r"^Image\s+(\d+)\s*:\s*([+-]?\d+(?:\.\d+)?)$", line)
            if m and current:
                label = f"Image {m.group(1)}"
                sections[current][label] = float(m.group(2))

        return sections

    @staticmethod
    def convert_to_image(image: Float[Tensor, "C H W"]) -> Image.Image:
        """
        Convert a tensor image to a PIL Image.
        Assume

        Args:
            image (Float[Tensor, "C H W"]): The input tensor image.
        Returns:
            Image.Image: The converted PIL Image.
        """
        image = ((image * 0.5 + 0.5).clamp(0, 1) * 255).byte()
        np_image: NDArray[np.uint8] = rearrange(image.cpu(), "C H W -> H W C").numpy()  # type: ignore[reportUnknownMemberType]
        return Image.fromarray(np_image)

    def get_template(self, prompt: str) -> str:
        if self.use_cot:
            return f"""Given a caption and two images generated based on this caption, please analyze in detail the
                two provided images. Evaluate them on various dimensions such as semantic consistency (how closely the image
                content aligns with the caption), aesthetics (composition, color usage, artistic expression), authenticity
                (realism and attention to detail), and any other factors you deem relevant. For each evaluation dimension,
                provide a score between 1-10 for both images (e.g., Image 1: 8/10, Image 2: 6/10) and provide a concise
                rationale for the score. Calculate the total score for each image by summing all dimension scores. Use a
                chain-of-thought process to detail your reasoning steps, and enclose all your detailed reasoning within
                <think> and </think> tags. Then, in the <answer> tag, output exactly one of the following strings: 'Image 1
                is better' or 'Image 2 is better' based on the total scores. No additional text is allowed in the <answer>
                section.\n\nExample output format:\n<think>\n1. Semantic consistency: Image 1 (9/10) - ...; Image 2 (7/10)
                - ...\n2. Aesthetics: Image 2 (8/10) - ...; Image 1 (8/10) - ...\n3. Authenticity: Image 1 (8/10) - ...;
                Image 2 (5/10) - ...\n[Additional dimensions if any]: Image 2 (8/10) - ...; Image 1 (6/10) - ...\nTotal
                score:\nImage 1: 9+8+8+6=31\nImage 2: 7+8+5+8=28\n</think>\n<answer>Image 1 is better</answer>\n**Note:
                In the example above, scores and the final answer are placeholders meant only to demonstrate the format.
                Your actual evaluation should be based on the quality of two given images.**\n\nYour task is provided as
                follows:\nText Caption: [{prompt}]"""

        return f"""You are presented with two generated images (Image 1 and Image 2) along with a shared text caption.
            Your task is to comparatively evaluate the two images across three specific dimensions:\n\n
            - Alignment Score: How well each image matches the caption in terms of content.\n
            - Coherence Score: How logically consistent and visually coherent each image is (absence of visual glitches, distorted objects, etc.).\n
            - Style Score: How aesthetically appealing each image is, regardless of caption accuracy.\n\n
            For each dimension, you must assign a relative score to Image 1 and Image 2, such that:\n
            - Each score is a float between 0 and 1 (inclusive).\n
            - The scores for Image 1 and Image 2 must sum to exactly 1.0 for each dimension.\n
            - The higher the score, the better that image is in the corresponding dimension *relative to the other*.\n\n
            This format emphasizes comparative quality rather than absolute evaluation.\n\n
            Please provide your evaluation in the format below:\n\n
            Alignment Score:\n
             Image 1: X\n
             Image 2: Y\n\n
            Coherence Score:\n
             Image 1: X\n
             Image 2: Y\n\n
            Style Score:\n
             Image 1: X\n
             Image 2: Y\n\n
            Your task is provided as follows:\n
            Text Caption: [{prompt}]"""

    def get_message(self, image_1: Image.Image, image_2: Image.Image, prompt: str) -> list[ChatMessage]:
        template = self.get_template(prompt)
        message: list[ChatMessage] = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_1},
                    {"type": "image", "image": image_2},
                    {"type": "text", "text": template},
                ],
            }
        ]
        return message

    def _assess_winner(self, output: str) -> Literal[0, 1] | None:
        if self.use_cot:
            # Prefer explicit <answer> tag extraction; fall back to whole output if missing
            answer = self._extract_cot_answer(output) or output
            answer_lower = answer.lower()
            if "image 1 is better" in answer_lower:
                return 0
            if "image 2 is better" in answer_lower:
                return 1
        else:
            scores = self._parse_scores(output)
            if not scores:
                return
            total_img1 = sum(section["Image 1"] for section in scores.values() if "Image 1" in section)
            total_img2 = sum(section["Image 2"] for section in scores.values() if "Image 2" in section)
            if total_img1 > total_img2:
                return 0
            if total_img2 > total_img1:
                return 1
        return None

    def parse_and_aggregate(
        self,
        outputs: list[str],
        pairs: torch.Tensor,
        P: int,
    ) -> tuple[Float[Tensor, "P n_image_per_prompt"], Float[Tensor, "P n_image_per_prompt"]]:
        """
        Aggregate pairwise preferences into per-image win and compare counts.

        Args:
            outputs: Model text outputs, ordered by prompt first, then pair index.
            pairs: Tensor of shape (n_pairs, 2) with image indices per prompt (0..n_images_per_prompt-1).
            P: Number of prompts in the batch.

        Returns:
            win_count: (P, n_images_per_prompt) float tensor where each entry is the accumulated wins
                       (ties contribute 0.5).
            compare_count: (P, n_images_per_prompt) int tensor with number of comparisons per image.
        """
        assert pairs.ndim == 2 and pairs.shape[1] == 2, "pairs must have shape (n_pairs, 2)"
        n_pairs = pairs.shape[0]

        # Initialize counts
        win_count = torch.zeros(P, self.n_image_per_prompt, dtype=torch.float32)
        compare_count = torch.zeros(P, self.n_image_per_prompt, dtype=torch.int32)

        # Ensure pairs on CPU for indexing ops; we only need indices here
        pairs_cpu = pairs.to("cpu")

        for i, output in enumerate(outputs):
            p = i // n_pairs  # prompt index
            j = i % n_pairs  # pair index within the prompt
            # Extract explicit Python ints for stable typing
            idx1 = int(pairs_cpu[j, 0].item())
            idx2 = int(pairs_cpu[j, 1].item())

            # Each image in the pair participates in one comparison
            compare_count[p, idx1] += 1
            compare_count[p, idx2] += 1

            winner = self._assess_winner(output)
            if winner is None:
                # tie: split the point
                win_count[p, idx1] += 0.5
                win_count[p, idx2] += 0.5
            elif winner == 0:
                win_count[p, idx1] += 1.0
            else:  # winner == 1
                win_count[p, idx2] += 1.0

        return win_count, compare_count

    def compute_reward(
        self,
        win_count: Float[Tensor, "P n_image_per_prompt"],
        compare_count: Float[Tensor, "P n_image_per_prompt"],
        advantage_per_prompt: bool = True,
    ) -> Float[Tensor, "n_images"]:
        """
        Compute standardized rewards from win and compare counts.

        """
        compare_count = compare_count.to(torch.float32)
        win_rates = torch.where(compare_count > 0, win_count / compare_count, torch.zeros_like(win_count))

        if advantage_per_prompt:
            mean = win_rates.mean(dim=1, keepdim=True)
            std = win_rates.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
            rewards = (win_rates - mean) / std
        else:
            mean = win_rates.mean()
            std = win_rates.std(unbiased=False).clamp_min(1e-6)
            rewards = (win_rates - mean) / std
        return torch.tensor(rewards, dtype=torch.float32)

    @torch.inference_mode()
    def forward(
        self,
        images: Float[Tensor, "batch n_channels height width"],
        context: list[str],
        advantage_per_prompt: bool = True,
    ) -> Float[Tensor, "batch"]:
        """
        Compute per-image standardized rewards from pairwise preference judgments within each prompt group.
        This method groups the input batch into P prompts, each containing `self.n_image_per_prompt` images.
        For every prompt, it forms all unordered image pairs, builds chat inputs with the corresponding
        text context, queries the underlying vision-language model to obtain pairwise preferences, aggregates
        the results into per-image win rates, and returns z-scored rewards either per prompt or globally.
        Args:
            images (torch.Tensor): Float tensor of shape (B, C, H, W) containing a batch of images.
                The batch size B must be divisible by `self.n_image_per_prompt`. Images are implicitly
                partitioned into P = B // `self.n_image_per_prompt` prompts.
            context (list[str]): A list of length P with the textual context (e.g., prompt) for each
                group of images. Each context entry is replicated across all intra-group pairs.
            advantage_per_prompt (bool, optional): If True, compute a per-prompt z-score (mean/std computed
                across the images within each prompt). If False, compute a global z-score across all images
                in the batch. Defaults to True.
        Returns:
            torch.Tensor: Float tensor of shape (B,) on the same device as `images`, containing the
                standardized reward for each image.
        Raises:
            AssertionError: If B % `self.n_image_per_prompt` != 0.
            RuntimeError: Propagated from underlying model/processor during tokenization or generation.
            ValueError: Propagated from parsing/aggregation if model outputs cannot be interpreted.
        Notes:
            - Pairs are formed as all unordered combinations of the `self.n_image_per_prompt` images
              within each prompt (r=2).
            - Win rates are computed as wins / comparisons per image; images with zero comparisons
              receive a zero win rate before standardization.
            - Standard deviation uses unbiased=False and is clamped with a minimum of 1e-6 to avoid
              division by zero.
            - Inference is batched with an upper bound of B pairwise conversations per generation call
              and uses max_new_tokens=4096.
        """
        B, C, H, W = images.shape
        assert B % self.n_image_per_prompt == 0, (
            f"Batch size {images.shape[0]} is not divisible by n_image_per_prompt {self.n_image_per_prompt}"
        )

        P = B // self.n_image_per_prompt

        imgs = images.view(P, self.n_image_per_prompt, C, H, W)
        pairs = torch.combinations(torch.arange(self.n_image_per_prompt, device=images.device), r=2)

        trg_tensor = imgs[:, pairs]
        trg_tensor = trg_tensor.reshape(-1, 2, C, H, W)  # (P * n_pairs, 2, C, H, W)

        context_extended = list(itertools.chain.from_iterable(itertools.repeat(c, pairs.shape[0]) for c in context))

        data_to_process: list[list[ChatMessage]] = [
            self.get_message(self.convert_to_image(pair[0]), self.convert_to_image(pair[1]), pair_context)
            for pair, pair_context in zip(trg_tensor, context_extended)
        ]

        outputs: list[str] = []
        # Batch processing respecing the original batch size
        for batch in range(0, len(data_to_process), B):
            batch_data = data_to_process[batch : batch + B]
            chat_input = self.processor.apply_chat_template(  # type: ignore[reportArgumentType]
                batch_data,
                tokenize=False,
                add_generation_prompt=True,
            )
            image_inputs, _ = process_vision_info(batch_data)  # type: ignore[reportArgumentType]

            inputs = self.processor(text=chat_input, images=image_inputs, return_tensors="pt", padding=True).to(  # type: ignore[reportUnknownMemberType]
                self.model.device
            )

            generated_ids = self.model.generate(**inputs, max_new_tokens=4096)  # type: ignore[reportUnknownMemberType]
            generated_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)]
            outputs.extend(cast(list[str], self.processor.batch_decode(generated_trimmed, skip_special_tokens=True)))  # type: ignore[reportUnknownMemberType]

        # Aggregate pairwise preferences into per-image counts per prompt
        win_count, compare_count = self.parse_and_aggregate(outputs, pairs, P=P)
        rewards = self.compute_reward(win_count, compare_count, advantage_per_prompt)
        rewards_flat = rewards.reshape(-1).to(images.device)
        return rewards_flat

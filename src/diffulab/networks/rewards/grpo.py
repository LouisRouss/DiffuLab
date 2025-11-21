import re
from typing import TYPE_CHECKING, Literal, TypedDict, Union, cast  # added TypedDict related imports

import numpy as np
import torch
from einops import rearrange
from jaxtyping import Float
from PIL import Image
from qwen_vl_utils import process_vision_info  # type: ignore[reportUnknownVariableType]
from torch import Tensor
from transformers import (  # type: ignore[reportMissingTypeStubs]
    AutoProcessor,
    CLIPModel,
    CLIPProcessor,
    Qwen2_5_VLForConditionalGeneration,
)

from diffulab.networks.rewards.common import RewardModel

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from transformers import BatchEncoding, CLIPOutput, Qwen2_5_VLProcessor  # type: ignore


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


class PrefGRPORewardModel(RewardModel):
    model_registry = {
        "cot_7b": "CodeGoat24/UnifiedReward-Think-qwen-7b",
        "3b": "CodeGoat24/UnifiedReward-2.0-qwen-3b",
        "7b": "CodeGoat24/UnifiedReward-2.0-qwen-7b",
        "32b": "CodeGoat24/UnifiedReward-2.0-qwen-32b",
        "72b": "CodeGoat24/UnifiedReward-2.0-qwen-72b",
    }

    def __init__(
        self,
        version: str = "7b",
        n_image_per_prompt: int = 16,
        advantage_clip_max: float = 5.0,
        use_clip: bool = False,
        lambda_base: float = 0.7,
        lambda_clip: float = 1.4,
        clip_model_id: str = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    ):
        super().__init__(n_image_per_prompt)
        self.use_cot = version.startswith("cot")
        self.version = version
        assert self.version in self.model_registry, (
            f"Unsupported model version: {self.version}, available versions: {list(self.model_registry.keys())}"
        )

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(  # type: ignore[reportUnknownMemberType]
            self.model_registry[self.version], dtype="auto", device_map="auto"
        )
        self.processor: "Qwen2_5_VLProcessor" = AutoProcessor.from_pretrained(self.model_registry[self.version])  # type: ignore[reportUnknownMemberType]
        for param in self.model.parameters():  # type: ignore
            param.requires_grad = False

        self.clip_model: None | CLIPModel = None
        self.clip_processor: None | CLIPProcessor = None
        self.use_clip = use_clip
        if self.use_clip:
            self.clip_model = CLIPModel.from_pretrained(clip_model_id)  # type: ignore
            self.clip_processor = CLIPProcessor.from_pretrained(clip_model_id)  # type: ignore
            self.clip_model.eval()
            for param in self.clip_model.parameters():  # type: ignore
                param.requires_grad = False

        self.advantage_clip_max = advantage_clip_max
        self.lambda_base = lambda_base
        self.lambda_clip = lambda_clip

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
        np_image: "NDArray[np.uint8]" = rearrange(image.cpu(), "C H W -> H W C").numpy()  # type: ignore[reportUnknownMemberType]
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
            win_count: (P, n_image_per_prompt) float tensor where each entry is the accumulated wins
                       (ties contribute 0.5).
            compare_count: (P, n_image_per_prompt) int tensor with number of comparisons per image.
        """
        assert self.n_image_per_prompt is not None, (
            "n_image_per_prompt must be set before calling parse_and_aggregate()"
        )
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

    def compute_advantages(
        self,
        advantages: Float[Tensor, "P n_image_per_prompt"],
        advantage_per_prompt: bool = True,
    ) -> Float[Tensor, "P n_image_per_prompt"]:
        """
        Compute per-image standardized advantages from win and compare counts.
        Args:
            advantages (Float[Tensor, "P n_image_per_prompt"]): Float tensor with per-image advantage.
            advantage_per_prompt (bool, optional): If True, compute a per-prompt z-score (mean/std computed
                across the images within each prompt). If False, compute a global z-score across all images
                in the batch. Defaults to True.
        Returns:
            Float[Tensor, "P n_image_per_prompt"]: Float tensor of shape (P, n_image_per_prompt,) with standardized advantages.
        Raises:
            ValueError: If shapes of win_count and compare_count do not match.
        Notes:
            - Standard deviation uses unbiased=False and is clamped with a minimum of 1e-6 to avoid
              division by zero.
        """
        if advantage_per_prompt:
            mean = advantages.mean(dim=1, keepdim=True)
            std = advantages.std(dim=1, keepdim=True).clamp_min(1e-6)
            advantages = (advantages - mean) / std
        else:
            mean = advantages.mean()
            std = advantages.std().clamp_min(1e-6)
            advantages = (advantages - mean) / std
        return advantages

    @torch.inference_mode()
    def compute_pairwise_advantages(
        self,
        images: Float[Tensor, "n_group images_per_prompt n_channels height width"],
        context: list[str],
        advantage_per_prompt: bool = True,
    ) -> Float[Tensor, "n_group images_per_prompt"]:
        """
        Compute per-image standardized rewards from pairwise preference judgments within each prompt group.
        This method groups the input batch into P prompts, each containing `self.n_image_per_prompt` images.
        For every prompt, it forms all unordered image pairs, builds chat inputs with the corresponding
        text context, queries the underlying vision-language model to obtain pairwise preferences, aggregates
        the results into per-image win rates, and returns z-scored rewards either per prompt or globally.
        Args:
            images (torch.Tensor): Float tensor of shape (P, N, C, H, W) containing a batch of images.
            context (list[str]): A list of length P*N with the textual context (e.g., prompt) for each
                image.
            advantage_per_prompt (bool, optional): If True, compute a per-prompt z-score (mean/std computed
                across the images within each prompt). If False, compute a global z-score across all images
                in the batch. Defaults to True.
        Returns:
            torch.Tensor: Float tensor of shape (P, N) on the same device as `images`, containing the
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
        P, n_images, C, H, W = images.shape
        assert n_images == self.n_image_per_prompt, (
            f"Expected {self.n_image_per_prompt} images per prompt, but got {n_images}"
        )
        assert len(context) == P * n_images, f"Expected {P * n_images} context entries, but got {len(context)}"

        pairs = torch.combinations(torch.arange(self.n_image_per_prompt, device=images.device), r=2)
        trg_tensor = images[:, pairs]
        trg_tensor = trg_tensor.reshape(-1, 2, C, H, W)  # (P * n_pairs, 2, C, H, W)

        data_to_process: list[list[ChatMessage]] = [
            self.get_message(self.convert_to_image(pair[0]), self.convert_to_image(pair[1]), pair_context)
            for pair, pair_context in zip(trg_tensor, context)
        ]

        outputs: list[str] = []
        # Batch processing respecting the original batch size
        for batch in range(0, len(data_to_process), P):
            batch_data = data_to_process[batch : batch + P]
            chat_input = self.processor.apply_chat_template(  # type: ignore[reportArgumentType]
                batch_data,  # type: ignore[reportArgumentType]
                tokenize=False,
                add_generation_prompt=True,
            )
            image_inputs, _ = process_vision_info(batch_data)  # type: ignore[reportArgumentType]

            inputs = self.processor(text=chat_input, images=image_inputs, return_tensors="pt", padding=True).to(  # type: ignore[reportUnknownMemberType]
                self.model.device
            )

            generated_ids = self.model.generate(**inputs)  # type: ignore[reportUnknownMemberType]
            generated_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)]
            outputs.extend(cast(list[str], self.processor.batch_decode(generated_trimmed, skip_special_tokens=True)))  # type: ignore[reportUnknownMemberType]

        # Aggregate pairwise preferences into per-image counts per prompt
        win_count, compare_count = self.parse_and_aggregate(outputs, pairs, P=P)
        compare_count = compare_count.to(torch.float32)
        advantages = torch.where(compare_count > 0, win_count / compare_count, torch.zeros_like(win_count))
        advantages = self.compute_advantages(advantages, advantage_per_prompt)
        return advantages.to(images.device)

    @torch.inference_mode()
    def compute_clip_advantages(
        self,
        images: Float[Tensor, "n_group images_per_prompt n_channels height width"],
        context: list[str],
        advantage_per_prompt: bool = True,
    ) -> Float[Tensor, "n_group images_per_prompt"]:
        """
        Compute per-image standardized rewards using CLIP cosine similarity between images and text context.
        Args:
            images (torch.Tensor): Float tensor of shape (P, N, C, H, W) containing a batch of images.
            context (list[str]): A list of length P*N with the textual context (e.g., prompt) for each
                image.
            advantage_per_prompt (bool, optional): If True, compute a per-prompt z-score (mean/std computed
                across the images within each prompt). If False, compute a global z-score across all images
                in the batch. Defaults to True.
        Returns:
            torch.Tensor: Float tensor of shape (P, N) on the same device as `images`, containing the
                standardized reward for each image.
        Raises:
            AssertionError: If B % `self.n_image_per_prompt` != 0 or if CLIP model/processor are not initialized.
            RuntimeError: Propagated from underlying model/processor during tokenization or forward pass.
        Notes:
            - Inference is batched with an upper bound of B images per forward call.
            - Standard deviation uses unbiased=False and is clamped with a minimum of 1e-6 to avoid
              division by zero.
        """
        assert self.clip_model is not None, "Clip model is not initialized"
        assert self.clip_processor is not None, "Clip processor is not initialized"
        device = next(self.clip_model.parameters()).device

        P, n_images, C, H, W = images.shape
        images = images.reshape(-1, C, H, W)

        advantages = torch.zeros((images.shape[0],), dtype=torch.float32, device=device)
        for batch in range(0, len(images), P):
            batch_images = [self.convert_to_image(img) for img in images[batch : batch + P]]
            batch_context = context[batch : batch + P]

            inputs = cast(
                "BatchEncoding",
                self.clip_processor(text=batch_context, images=batch_images, return_tensors="pt", padding=True).to(  # type: ignore
                    self.clip_model.device
                ),
            )
            outputs: "CLIPOutput" = self.clip_model(**inputs)
            logits_per_image = cast(Tensor, outputs.logits_per_image)  # type: ignore
            cosine_sims = logits_per_image / self.clip_model.logit_scale.exp()
            advantages[batch : batch + P] = cosine_sims

        advantages = advantages.reshape(P, n_images)
        advantages = self.compute_advantages(advantages, advantage_per_prompt)
        return advantages

    def forward(
        self,
        images: Float[Tensor, "batch n_channels height width"],
        context: list[str],
        advantage_per_prompt: bool = True,
    ) -> Float[Tensor, "batch"]:
        """
        Compute per-image standardized rewards from pairwise preference judgments within each prompt group.
        Args:
            images (torch.Tensor): Float tensor of shape (B, C, H, W) containing a batch of images.
            context (list[str]): A list of length B with the textual context (e.g., prompt) for each
                image.
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
            - This method reshapes the input batch into P prompts of N images each, where
              N=`self.n_image_per_prompt` and P=B/N.
            - Rewards are computed by combining pairwise preference judgments and optional CLIP
              cosine similarities, followed by z-scoring either per prompt or globally.
        """
        B, C, H, W = images.shape
        assert self.n_image_per_prompt is not None, "n_image_per_prompt must be set before calling forward()"
        assert B % self.n_image_per_prompt == 0, (
            f"Batch size {images.shape[0]} is not divisible by n_image_per_prompt {self.n_image_per_prompt}"
        )
        assert len(context) == B // self.n_image_per_prompt, (
            f"Length of context {len(context)} does not match number of prompts {B // self.n_image_per_prompt}"
        )

        # P also corresponds to the original batch size
        P = B // self.n_image_per_prompt
        images = images.reshape(P, self.n_image_per_prompt, C, H, W)

        advantages = self.compute_pairwise_advantages(
            images=images,
            context=context,
            advantage_per_prompt=advantage_per_prompt,
        )

        if self.use_clip:
            clip_advantages = self.compute_clip_advantages(
                images=images,
                context=context,
                advantage_per_prompt=advantage_per_prompt,
            )
            advantages = self.lambda_base * advantages + self.lambda_clip * clip_advantages

        advantages = advantages.clamp(-self.advantage_clip_max, self.advantage_clip_max)
        return advantages.reshape(B)

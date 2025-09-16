from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor


class RewardModel(nn.Module, ABC):
    def __init__(self, n_image_per_prompt: int | None = None):
        super().__init__()  # type: ignore
        self._n_image_per_prompt = n_image_per_prompt

    @abstractmethod
    @torch.inference_mode()
    def forward(self, images: Float[Tensor, "batch n_channels height width"], context: Any) -> Float[Tensor, "batch"]:
        """
        Apply the model to an input batch.

        Args:
            images (torch.Tensor): the input batch.
            context (Any): additional context information.
        Returns:
            torch.Tensor: a tensor representing the reward for the given input batch.
        """
        pass

    def set_n_image_per_prompt(self, n: int) -> None:
        self._n_image_per_prompt = n

    @property
    @abstractmethod
    def n_image_per_prompt(self) -> int | None:
        """
        Returns the number of images per prompt that the model has generated.

        Returns:
            int or None: the number of images per prompt, or None if not set.
        """
        return self._n_image_per_prompt

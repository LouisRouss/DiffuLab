from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor


class RewardModel(nn.Module, ABC):
    def __init__(self):
        super().__init__()  # type: ignore

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

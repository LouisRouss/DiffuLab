from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn
from PIL import Image


class RewardModel(nn.Module, ABC):
    def __init__(self):
        super().__init__()  # type: ignore

    @abstractmethod
    def forward(self, images: list[Image.Image], context: Any) -> torch.Tensor:
        """
        Apply the model to an input batch.

        Args:
            x (torch.Tensor): the input batch.
            *args (Any): additional positional arguments.
            **kwargs (Any): additional keyword arguments.
        Returns:
            torch.Tensor: a tensor representing the reward for the given input batch.
        """
        pass

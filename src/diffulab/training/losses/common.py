from abc import ABC
from typing import TYPE_CHECKING

import torch.nn as nn

if TYPE_CHECKING:
    from diffulab.networks.denoisers import Denoiser


class LossFunction(ABC, nn.Module):
    name: str = "extra_loss"

    def __init__(self) -> None:
        super().__init__()  # type: ignore

    def set_model(self, model: "Denoiser") -> None:
        """
        Set the model for the loss function.

        Args:
            model (nn.Module): The model to set.
        """
        # By default, doesn't do anything.
        pass

from abc import ABC
from pathlib import Path

import torch.nn as nn
from accelerate import Accelerator  # type: ignore


class LossFunction(ABC, nn.Module):  # to be completed
    def __init__(self) -> None:
        super().__init__()  # type: ignore

    def save(self, path: str | Path, accelerator: Accelerator | None) -> None:
        """
        Save eventual learnable parameters of the loss function.

        Args:
            path (str | Path): Path to save the loss function.
            accelerator (Accelerator | None): Accelerator instance for distributed training. Uses
                accelerator.save if provided
        """
        # By default, doesn't save anything.
        pass

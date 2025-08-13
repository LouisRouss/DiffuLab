from abc import ABC
from pathlib import Path
from typing import TYPE_CHECKING

import torch.nn as nn

if TYPE_CHECKING:
    from accelerate import Accelerator  # type: ignore
    from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel
    from torch.nn.parallel import DistributedDataParallel

    from diffulab.networks.denoisers import Denoiser


class LossFunction(ABC, nn.Module):  # to be completed
    def __init__(self) -> None:
        super().__init__()  # type: ignore

    def save(self, path: str | Path, accelerator: "Accelerator") -> None:
        """
        Save eventual learnable parameters of the loss function.

        Args:
            path (str | Path): Path to save the loss function.
            accelerator (Accelerator | None): Accelerator instance for distributed training. Uses
                accelerator.save if provided
        """
        # By default, doesn't save anything.
        pass

    def accelerate_prepare(
        self, accelerator: "Accelerator"
    ) -> "list[nn.Module | DistributedDataParallel | FullyShardedDataParallel]":
        """
        Prepare the loss function for distributed training.

        Args:
            accelerator (Accelerator): Accelerator instance for distributed training.
        """
        # By default, doesn't prepare anything.
        return []

    def set_model(self, model: "Denoiser") -> None:
        """
        Set the model for the loss function.

        Args:
            model (nn.Module): The model to set.
        """
        # By default, doesn't do anything.
        pass

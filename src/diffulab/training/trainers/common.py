import os
from abc import ABC, abstractmethod
from datetime import datetime
from math import ceil, sqrt
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, cast

import numpy as np
import torch
import wandb
from accelerate import Accelerator  # type: ignore [stub file not found]
from accelerate.utils import TorchDynamoPlugin  # type: ignore [stub file not found]
from ema_pytorch import EMA  # type: ignore [stub file not found]
from numpy.typing import NDArray
from torch import Tensor
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torchvision.utils import make_grid  # type: ignore

from diffulab.datasets.base import BatchData
from diffulab.networks.denoisers.common import Denoiser, ModelInput

if TYPE_CHECKING:
    from diffulab.diffuse import Diffuser
HOME_PATH = Path.home()


class Trainer(ABC):
    """
    An abstract training class for diffusion models with support for distributed training, mixed precision, gradient accumulation,
    and EMA model averaging.
    It uses the Hugging Face Accelerate library for distributed training.
    Training and validation loops need to be implemented in subclasses.
    Args:
        n_epoch (int): Number of training epochs.
        gradient_accumulation_step (int, optional): Number of steps to accumulate gradients. Defaults to 1.
        precision_type (str, optional): Type of precision for mixed precision training ("no", "fp16", "bf16").
            Defaults to "no".
        save_path (str | Path, optional): Path to save model checkpoints and logs. Defaults to a timestamped
            directory under ~/experiments.
        project_name (str, optional): Name of the project for logging purposes. Defaults to "my_project".
        run_config (dict[str, Any] | None, optional): Configuration dictionary for the training run.
            Defaults to None.
        init_kwargs (dict[str, Any], optional): Additional initialization arguments for trackers.
            Defaults to {}.
        use_ema (bool, optional): Whether to use Exponential Moving Average of the model weights.
            Defaults to False.
        ema_rate (float, optional): Decay rate for EMA. Defaults to 0.999.
        ema_update_after_step (int, optional): Number of steps before starting EMA updates.
            Defaults to 100.
        ema_update_every (int, optional): Frequency of EMA updates. Defaults to 1.
        compile (bool, optional): Whether to compile the model using TorchDynamo. Defaults to True.
        dynam_plugin_kwargs (dict[str, Any], optional): Additional arguments for the TorchDynamo plugin.
            Defaults to {}.
    Attributes:
        n_epoch (int): Number of training epochs.
        use_ema (bool): Whether EMA is enabled.
        ema_rate (float): EMA decay rate.
        ema_update_after_step (int): Steps before EMA updates begin.
        ema_update_every (int): EMA update frequency.
        accelerator (Accelerator): Hugging Face Accelerator instance.
        save_path (Path): Path for saving outputs.
    Methods:
        training_step: Performs a single training step.
        move_dict_to_device: Moves batch data to the appropriate device.
        validation_step: Performs a single validation step.
        save_model: Saves model checkpoints.
        log_images: Logs generated images during validation.
        train: Main training loop.
    """

    def __init__(
        self,
        n_epoch: int,
        gradient_accumulation_step: int = 1,
        precision_type: str = "no",
        save_path: str | Path = Path.home() / "experiments" / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        project_name: str = "my_project",
        run_config: dict[str, Any] | None = None,
        init_kwargs: dict[str, Any] = {},
        use_ema: bool = False,
        ema_rate: float = 0.999,
        ema_update_after_step: int = 0,
        ema_update_every: int = 10,
        compile: bool = False,
        dynamo_plugin_kwargs: dict[str, Any] = {
            "backend": "inductor",
            "mode": "default",
            "fullgraph": False,
            "dynamic": True,
        },
    ):
        assert (HOME_PATH / ".cache" / "huggingface" / "accelerate" / "default_config.yaml").exists(), (
            "please run `accelerate config` first in the CLI and save the config at the default location"
        )
        self.n_epoch = n_epoch
        self.use_ema = use_ema
        self.ema_rate = ema_rate
        self.ema_update_after_step = ema_update_after_step * gradient_accumulation_step
        self.ema_update_every = ema_update_every * gradient_accumulation_step
        dynamo_plugin = TorchDynamoPlugin(**dynamo_plugin_kwargs) if compile else None
        self.compile = compile
        self.accelerator = Accelerator(
            split_batches=True,
            mixed_precision=precision_type,
            gradient_accumulation_steps=gradient_accumulation_step,
            log_with="wandb",
            dynamo_plugin=dynamo_plugin,
        )
        self.save_path = Path(save_path) / project_name
        Path(self.save_path).mkdir(parents=True, exist_ok=True)
        os.environ["WANDB_DIR"] = str(self.save_path / "wandb")
        Path(os.environ["WANDB_DIR"]).mkdir(parents=True, exist_ok=True)
        self.accelerator.init_trackers(project_name=project_name, config=run_config, init_kwargs=init_kwargs)  # type: ignore

    def move_dict_to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        """
        Moves all tensor values in a dictionary to the appropriate device.
        This method recursively traverses a dictionary and moves any tensor values to the
        device specified by the accelerator, while leaving non-tensor values unchanged.
        Args:
            batch (dict[str, Any]): A dictionary where values may be tensors
                that need to be moved to the appropriate device.
        Returns:
            dict[str, Any]: A new dictionary with the same structure as the input, but with all
                tensor values moved to the accelerator's device.
        """
        return {k: v.to(self.accelerator.device) if isinstance(v, Tensor) else v for k, v in batch.items()}

    def save_model(
        self,
        optimizer: Optimizer,
        diffuser: "Diffuser",
        ema_denoiser: EMA | None = None,
        scheduler: LRScheduler | None = None,
    ) -> None:
        """
        Saves model checkpoints and training state.
        This method saves the current state of the model, optimizer, EMA model (if used), and
        scheduler (if used) to the specified save path. It handles distributed training states
        by properly unwrapping models before saving.
        Args:
            optimizer (Optimizer): The optimizer used for training.
            diffuser (Diffuser): The diffusion model wrapper containing the denoiser model.
            ema_denoiser (Denoiser | None, optional): EMA version of the denoiser model.
                If provided, its state will be saved. Defaults to None.
            scheduler (LRScheduler | None, optional): Learning rate scheduler.
                If provided, its state will be saved. Defaults to None.
        Note:
            The following files are created in self.save_path:
            - denoiser.pt: Main model state dict
            - optimizer.pt: Optimizer state dict
            - ema.pt: EMA model state dict (if EMA is used)
            - scheduler.pt: Scheduler state dict (if scheduler is used)
        """
        unwrapped_denoiser: Denoiser = self.accelerator.unwrap_model(diffuser.denoiser)  # type: ignore
        state_dict = cast(dict[str, Tensor], unwrapped_denoiser.state_dict())  # type: ignore
        if self.compile:
            state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        self.accelerator.save(state_dict, self.save_path / "denoiser.pt")  # type: ignore

        self.accelerator.save(optimizer.optimizer.state_dict(), self.save_path / "optimizer.pt")  # type: ignore

        if ema_denoiser is not None:
            unwrapped_ema = self.accelerator.unwrap_model(ema_denoiser)  # type: ignore
            self.accelerator.save(unwrapped_ema.ema_model.state_dict(), self.save_path / "ema.pt")  # type: ignore

        if scheduler is not None:
            self.accelerator.save(scheduler.scheduler.state_dict(), self.save_path / "scheduler.pt")  # type: ignore

        for extra_loss in diffuser.extra_losses:
            unwrapped_loss = self.accelerator.unwrap_model(extra_loss)  # type: ignore
            state_dict = cast(dict[str, Tensor], unwrapped_loss.state_dict())  # type: ignore
            if self.compile:
                state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
            self.accelerator.save(state_dict, self.save_path / f"{unwrapped_loss.name}.pt")  # type: ignore

    @torch.no_grad()  # type: ignore
    def log_images(
        self,
        diffuser: "Diffuser",
        val_dataloader: Iterable[BatchData],
        epoch: int,
        val_steps: int = 50,
        guidance_scale: float = 0,
    ) -> None:
        """
        Logs generated images during the validation phase.
        This method generates and logs sample images using the current state of the diffusion model.
        It temporarily adjusts the number of diffusion steps for faster validation image generation,
        generates samples, and logs them using wandb (Weights & Biases).
        Args:
            diffuser (Diffuser): The diffusion model wrapper used for generating samples.
            val_dataloader (Iterable[BatchData]): An iterator providing validation batches.
                Each batch should contain at least an 'x' key with the input data.
            epoch (int): Current training epoch number, used for logging.
            val_steps (int, optional): Number of diffusion steps to use for validation
                image generation. Using fewer steps speeds up generation. Defaults to 50.
            guidance_scale (float, optional): Guidance scale for classifier-free guidance.
                Defaults to 0.
        Note:
            - The method temporarily changes the number of diffusion steps to val_steps
              for faster generation, then restores the original number of steps.
            - Generated images are normalized from [-1, 1] to [0, 1] range before logging.
            - Images are logged to wandb with the key 'val/images'.
        """
        batch: ModelInput = next(iter(val_dataloader))["model_inputs"]
        x: Tensor = batch.pop("x")  # type: ignore
        original_steps = diffuser.n_steps
        diffuser.set_steps(val_steps)
        images = diffuser.generate(data_shape=x.shape, model_inputs=batch, guidance_scale=guidance_scale)["x"]
        images = (images * 0.5 + 0.5).clamp(0, 1).cpu()

        grid = make_grid(images, nrow=int(ceil(sqrt(images.shape[0]))), padding=2)
        np_grid: NDArray[np.uint8] = (grid * 255).round().byte().permute(1, 2, 0).numpy()  # type: ignore
        to_log = wandb.Image(np_grid, caption="Validation Images")
        self.accelerator.log({"val/images": to_log}, step=epoch + 1, log_kwargs={"wandb": {"commit": True}})  # type: ignore
        diffuser.set_steps(original_steps)

    @abstractmethod
    def training_step(self, *args: Any, **kwargs: Any) -> None:
        """
        Performs a single training step.
        This method should be implemented in subclasses to define the specific operations
        that occur during a single training step, including forward pass, loss computation,
        backward pass, and optimizer step.
        """
        ...

    @abstractmethod
    def validation_step(self, *args: Any, **kwargs: Any) -> None:
        """
        Performs a single validation step.
        This method should be implemented in subclasses to define the specific operations
        that occur during a single validation step, including forward pass and loss computation.
        """
        ...

    @abstractmethod
    def train(self, *args: Any, **kwargs: Any) -> None:
        """
        Main training loop.
        This method should be implemented in subclasses to define the overall training process,
        including iterating over epochs and batches, calling training and validation steps,
        logging metrics, and saving model checkpoints.
        """
        ...

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

import torch
import wandb
from accelerate import Accelerator  # type: ignore [stub file not found]
from ema_pytorch import EMA  # type: ignore [stub file not found]
from torch import Tensor
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from tqdm import tqdm

from diffulab.datasets.base import BatchData
from diffulab.networks.denoisers.common import Denoiser, ModelInput
from diffulab.training.utils import AverageMeter

if TYPE_CHECKING:
    from diffulab.diffuse.diffuser import Diffuser

HOME_PATH = Path.home()


class Trainer:
    """
    A training class that handles the training loop for diffusion models with support for distributed training,
    mixed precision, gradient accumulation, and EMA model averaging.
    This class provides a complete training pipeline including validation, logging, and model checkpointing.
    It uses the Hugging Face Accelerate library for distributed training and handles both conditional and
    unconditional diffusion model training.
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
        ema_update_after_step: int = 100,
        ema_update_every: int = 1,
    ):
        assert (HOME_PATH / ".cache" / "huggingface" / "accelerate" / "default_config.yaml").exists(), (
            "please run `accelerate config` first in the CLI and save the config at the default location"
        )
        self.n_epoch = n_epoch
        self.use_ema = use_ema
        self.ema_rate = ema_rate
        self.ema_update_after_step = ema_update_after_step * gradient_accumulation_step
        self.ema_update_every = ema_update_every * gradient_accumulation_step
        self.accelerator = Accelerator(
            split_batches=True,
            mixed_precision=precision_type,
            gradient_accumulation_steps=gradient_accumulation_step,
            log_with="wandb",
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
        return {
            k: v.to(self.accelerator.device) if isinstance(v, Tensor) else v
            for k, v in batch.items()  # type: ignore
        }

    def training_step(
        self,
        diffuser: "Diffuser",
        optimizer: Optimizer,
        batch: BatchData,
        tracker: AverageMeter,
        p_classifier_free_guidance: float = 0,
        scheduler: LRScheduler | None = None,
        per_batch_scheduler: bool = False,
        ema_denoiser: EMA | None = None,
    ) -> None:
        """
        Performs a single training step for the diffusion model.
        This method executes one complete training iteration including forward pass, loss computation,
        backpropagation, and optimizer updates. It also handles EMA updates and learning rate scheduling
        if enabled.
        Args:
            diffuser (Diffuser): The diffusion model wrapper that handles the diffusion process.
            optimizer (Optimizer): The optimizer used for updating model parameters.
            batch (BatchData): A dictionary containing the training batch data, must include 'x' key
                for input data.
            tracker (AverageMeter): Tracks and logs training metrics like loss values.
            p_classifier_free_guidance (float, optional): Probability of using classifier-free guidance
                during training. Defaults to 0.
            scheduler (LRScheduler | None, optional): Learning rate scheduler. Defaults to None.
            per_batch_scheduler (bool, optional): Whether to step the scheduler after each batch.
                Defaults to False.
            ema_denoiser (EMA | None, optional): Exponential Moving Average model for parameter
                averaging. Defaults to None.
        Note:
            - The method automatically handles device placement through the accelerator.
            - Gradient accumulation is handled by the accelerator if configured.
            - Loss values are automatically tracked and can be accessed through the tracker.
        """
        optimizer.zero_grad()
        model_inputs = batch["model_inputs"]
        timesteps = diffuser.draw_timesteps(model_inputs["x"].shape[0]).to(self.accelerator.device)
        model_inputs.update({"p": p_classifier_free_guidance})
        losses = diffuser.compute_loss(
            model_inputs=model_inputs, timesteps=timesteps, extra_args=batch.get("extra", {})
        )
        for key, loss in losses.items():
            tracker.update(loss.item(), key=f"train/{key}")
        loss = sum(losses.values())
        self.accelerator.backward(loss)  # type: ignore
        optimizer.step()
        if scheduler is not None and per_batch_scheduler:
            scheduler.step()
        if ema_denoiser is not None:
            ema_denoiser.update()

    @torch.no_grad()  # type: ignore
    def validation_step(
        self,
        diffuser: "Diffuser",
        val_batch: BatchData,
        tracker: AverageMeter,
        ema_eval: Denoiser | None = None,
    ) -> None:
        """
        Performs a validation step for the diffusion model.
        This method computes the validation loss for a batch of data, supporting both standard
        and EMA (Exponential Moving Average) model evaluation.
        Args:
            diffuser (Diffuser): The diffusion model wrapper containing both the model and
                diffusion process.
            val_batch (ModelInput): A dictionary containing the validation batch data,
                must include 'x' key for input data.
            tracker (AverageMeter): Tracks and logs validation metrics like loss values.
            ema_eval (Denoiser | None, optional): EMA version of the model for evaluation.
                If provided and self.use_ema is True, validation will use the EMA model.
                Defaults to None.
        """
        model_inputs: ModelInput = val_batch["model_inputs"]
        timesteps = diffuser.draw_timesteps(model_inputs["x"].shape[0]).to(self.accelerator.device)
        model_inputs: ModelInput = self.move_dict_to_device(model_inputs)  # type: ignore
        extra_args = val_batch.get("extra", {})
        extra_args = self.move_dict_to_device(extra_args)

        if self.use_ema and ema_eval is not None:
            # Temporarily swap the model in diffuser to use EMA for validation
            original_model = diffuser.denoiser
            diffuser.denoiser = ema_eval
            for loss in diffuser.extra_losses:
                if hasattr(loss, 'set_model'):
                    loss.set_model(ema_eval) # type: ignore
            val_losses = diffuser.compute_loss(model_inputs=model_inputs, timesteps=timesteps, extra_args=extra_args)

            # Restore original model and eventual hooks
            diffuser.denoiser = original_model
            for loss in diffuser.extra_losses:
                if hasattr(loss, 'set_model'):
                    loss.set_model(original_model) # type: ignore
        else:
            val_losses = diffuser.compute_loss(model_inputs=model_inputs, timesteps=timesteps, extra_args=extra_args)
            
        for key, val_loss in val_losses.items():
            tracker.update(val_loss.item(), key=f"val/{key}")

    def save_model(
        self,
        optimizer: Optimizer,
        diffuser: "Diffuser",
        ema_denoiser: Denoiser | None = None,
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
            - denoiser.pth: Main model state dict
            - optimizer.pth: Optimizer state dict
            - ema.pth: EMA model state dict (if EMA is used)
            - scheduler.pth: Scheduler state dict (if scheduler is used)
        """
        unwrapped_denoiser: Denoiser = self.accelerator.unwrap_model(diffuser.denoiser)  # type: ignore
        self.accelerator.save(unwrapped_denoiser.state_dict(), self.save_path / "denoiser.pth")  # type: ignore
        self.accelerator.save(optimizer.optimizer.state_dict(), self.save_path / "optimizer.pth")  # type: ignore
        if ema_denoiser is not None:
            unwrapped_ema: Denoiser = self.accelerator.unwrap_model(ema_denoiser)  # type: ignore
            self.accelerator.save(unwrapped_ema.ema_model.state_dict(), self.save_path / "ema.pth")  # type: ignore
        if scheduler is not None:
            self.accelerator.save(scheduler.scheduler.state_dict(), self.save_path / "scheduler.pth")  # type: ignore

    @torch.no_grad()  # type: ignore
    def log_images(
        self,
        diffuser: "Diffuser",
        val_dataloader: Iterable[BatchData],
        epoch: int,
        ema_eval: Denoiser | None = None,
        val_steps: int = 50,
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
            ema_eval (Diffuser | None, optional): EMA version of the model for evaluation.
                If provided and self.use_ema is True, generation will use the EMA model.
                Defaults to None.
            val_steps (int, optional): Number of diffusion steps to use for validation
                image generation. Using fewer steps speeds up generation. Defaults to 50.
        Note:
            - The method temporarily changes the number of diffusion steps to val_steps
              for faster generation, then restores the original number of steps.
            - Generated images are normalized from [-1, 1] to [0, 1] range before logging.
            - Images are logged to wandb with the key 'val/images'.
        """
        batch: ModelInput = next(iter(val_dataloader))["model_inputs"]  # type: ignore
        x: Tensor = batch.pop("x")  # type: ignore
        original_steps = diffuser.n_steps
        diffuser.set_steps(val_steps)

        if self.use_ema and ema_eval is not None:
            original_model = diffuser.denoiser
            diffuser.denoiser = ema_eval
            images = diffuser.generate(data_shape=x.shape, model_inputs=batch)
            diffuser.denoiser = original_model
        else:
            images = diffuser.generate(data_shape=x.shape, model_inputs=batch)

        images = (images * 0.5 + 0.5).clamp(0, 1).cpu()
        images = wandb.Image(images, caption="Validation Images")
        self.accelerator.log({"val/images": images}, step=epoch + 1, log_kwargs={"wandb": {"commit": True}})  # type: ignore
        diffuser.set_steps(original_steps)

    def train(
        self,
        diffuser: "Diffuser",
        optimizer: Optimizer,
        train_dataloader: Iterable[BatchData],
        val_dataloader: Iterable[BatchData] | None = None,
        scheduler: LRScheduler | None = None,
        per_batch_scheduler: bool = False,
        log_validation_images: bool = False,
        train_embedder: bool = False,
        p_classifier_free_guidance: float = 0,
        val_steps: int = 50,
        ema_ckpt: str | None = None,
        epoch_start: int = 0,
    ):
        """
        Main training loop for diffusion models.
        This method orchestrates the entire training process, including validation, logging,
        and model checkpointing. It supports distributed training, mixed precision, gradient
        accumulation, and EMA model averaging.
        Args:
            diffuser (Diffuser): The diffusion model wrapper containing both model and diffusion process.
            optimizer (Optimizer): Optimizer for model parameter updates.
            train_dataloader (Iterable[ModelInput]): Iterator yielding training batches.
            val_dataloader (Iterable[ModelInput] | None, optional): Iterator yielding validation batches.
                If None, no validation is performed. Defaults to None.
            scheduler (LRScheduler | None, optional): Learning rate scheduler.
                If None, no learning rate scheduling is performed. Defaults to None.
            per_batch_scheduler (bool, optional): Whether to step scheduler after each batch
                instead of each epoch. Defaults to False.
            log_validation_images (bool, optional): Whether to generate and log sample images
                during validation. Defaults to False.
            train_embedder (bool, optional): Whether to train the context embedder if present.
                Defaults to False.
            p_classifier_free_guidance (float, optional): Probability of using classifier-free
                guidance during training. Defaults to 0.
            val_steps (int, optional): Number of steps to use for validation image generation.
                Defaults to 50.
            ema_ckpt (str | None, optional): Path to EMA model checkpoint for loading pretrained
                weights. Defaults to None.
            epoch_start (int, optional): Starting epoch number, useful for resuming training.
                Defaults to 0.
        Note:
            - The method automatically handles device placement through the accelerator.
            - Validation (if enabled) includes loss computation and optionally image generation.
            - Training metrics are logged using the configured accelerator's logging mechanism.
            - Model checkpoints are saved when validation loss improves.
            - EMA model is used for validation if enabled.
        """
        if self.use_ema:
            ema_denoiser = EMA(
                diffuser.denoiser,
                beta=self.ema_rate,
                update_after_step=self.ema_update_after_step,
                update_every=self.ema_update_every,
            ).to(self.accelerator.device)
            if ema_ckpt:
                ema_denoiser.ema_model.load_state_dict(torch.load(ema_ckpt, weights_only=True))  # type: ignore
            ema_denoiser = self.accelerator.prepare(ema_denoiser)  # type: ignore
        else:
            ema_denoiser = None

        if diffuser.vision_tower:
            diffuser.vision_tower = self.accelerator.prepare(diffuser.vision_tower)  # type: ignore

        diffuser.denoiser, train_dataloader, val_dataloader, optimizer = self.accelerator.prepare(  # type: ignore
            diffuser.denoiser, train_dataloader, val_dataloader, optimizer
        )
        for loss_idx in range(len(diffuser.extra_losses)):
            diffuser.extra_losses[loss_idx] = self.accelerator.prepare(diffuser.extra_losses[loss_idx])  # type: ignore

        if scheduler is not None:
            scheduler = self.accelerator.prepare_scheduler(scheduler)  # type: ignore
        best_val_loss = float("inf")

        if diffuser.denoiser.context_embedder is not None and not train_embedder:  # type: ignore
            for param in diffuser.denoiser.context_embedder.parameters():  # type: ignore
                param.requires_grad = False

        tracker = AverageMeter()
        tq_epoch = tqdm(
            range(epoch_start, self.n_epoch), disable=not self.accelerator.is_main_process, leave=False, position=0
        )
        logging.info("Begin training")
        for epoch in tq_epoch:
            diffuser.train()
            tq_epoch.set_description(f"Epoch {epoch + 1}/{self.n_epoch}")

            tq_batch = tqdm(train_dataloader, disable=not self.accelerator.is_main_process, leave=False)  # type: ignore
            for batch in tq_batch:
                with self.accelerator.accumulate(diffuser.denoiser):  # type: ignore
                    with self.accelerator.autocast():
                        self.training_step(
                            diffuser=diffuser,
                            optimizer=optimizer,  # type: ignore
                            batch=batch,
                            tracker=tracker,
                            p_classifier_free_guidance=p_classifier_free_guidance,
                            scheduler=scheduler,  # type: ignore
                            per_batch_scheduler=per_batch_scheduler,
                            ema_denoiser=ema_denoiser,  # type: ignore
                        )
                        tq_batch.set_description(
                            f"Loss: {sum(v for k, v in tracker.avg.items() if k.startswith('train/')):.4f}"
                        )

            for key, value in tracker.avg.items():
                if key.startswith("train/"):
                    gathered_loss: Tensor = self.accelerator.gather(  # type: ignore
                        torch.tensor(value, device=self.accelerator.device)
                    )
                    self.accelerator.log(  # type: ignore
                        {key: gathered_loss.mean().item()}, step=epoch + 1
                    )
            tracker.reset()

            if val_dataloader is not None:
                diffuser.eval()  # type: ignore
                if ema_denoiser is not None:
                    ema_eval = ema_denoiser.eval()  # type: ignore
                else:
                    ema_eval = None
                tq_val_batch = tqdm(
                    val_dataloader,  # type: ignore
                    disable=not self.accelerator.is_main_process,
                    leave=False,
                    position=1,
                )
                for val_batch in tq_val_batch:
                    with self.accelerator.autocast():
                        self.validation_step(
                            diffuser=diffuser,
                            val_batch=val_batch,
                            tracker=tracker,
                            ema_eval=ema_eval,  # type: ignore
                        )
                    tq_val_batch.set_description(
                        f"Val Loss: {sum(v for k, v in tracker.avg.items() if k.startswith('val/')):.4f}"
                    )

                total_loss = 0
                for key, value in tracker.avg.items():
                    if key.startswith("val/"):
                        gathered_loss: Tensor = self.accelerator.gather(  # type: ignore
                            torch.tensor(value, device=self.accelerator.device)
                        )
                        self.accelerator.log(  # type: ignore
                            {key: gathered_loss.mean().item()}, step=epoch + 1
                        )
                        total_loss += gathered_loss.mean().item()

                if total_loss < best_val_loss:  # type: ignore
                    best_val_loss = total_loss
                    self.save_model(optimizer, diffuser, ema_denoiser, scheduler)  # type: ignore
                tracker.reset()

                if log_validation_images:
                    logging.info("creating validation images")
                    if self.accelerator.is_main_process:
                        with self.accelerator.autocast():
                            self.log_images(diffuser, val_dataloader, epoch, ema_eval, val_steps)  # type: ignore

            self.accelerator.wait_for_everyone()

        self.accelerator.end_training()
        logging.info("Training complete")

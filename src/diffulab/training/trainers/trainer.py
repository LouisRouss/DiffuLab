import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

import torch
from ema_pytorch import EMA  # type: ignore [stub file not found]
from torch import Tensor
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from tqdm import tqdm

from diffulab.datasets.base import BatchData
from diffulab.networks.denoisers.common import Denoiser, ModelInput
from diffulab.training.trainers import Trainer
from diffulab.training.utils import AverageMeter

if TYPE_CHECKING:
    from diffulab.diffuse import Diffuser


class BaseTrainer(Trainer):
    """
    A training class that handles the supervised training loop for diffusion models with support
    for distributed training, mixed precision, gradient accumulation, and EMA model averaging.
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
        super().__init__(
            n_epoch=n_epoch,
            gradient_accumulation_step=gradient_accumulation_step,
            precision_type=precision_type,
            save_path=save_path,
            project_name=project_name,
            run_config=run_config,
            init_kwargs=init_kwargs,
            use_ema=use_ema,
            ema_rate=ema_rate,
            ema_update_after_step=ema_update_after_step,
            ema_update_every=ema_update_every,
            compile=compile,
            dynamo_plugin_kwargs=dynamo_plugin_kwargs,
        )

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
        val_losses = diffuser.compute_loss(model_inputs=model_inputs, timesteps=timesteps, extra_args=extra_args)
        for key, val_loss in val_losses.items():
            tracker.update(val_loss.item(), key=f"val/{key}")

    def train(
        self,
        diffuser: "Diffuser",
        optimizer: Optimizer,
        train_dataloader: Iterable[BatchData],
        val_dataloader: Iterable[BatchData] | None = None,
        scheduler: LRScheduler | None = None,
        per_batch_scheduler: bool = False,
        log_validation_images: bool = True,
        train_embedder: bool = False,
        p_classifier_free_guidance: float = 0.2,
        val_steps: int = 50,
        optimizer_ckpt: str | None = None,
        denoiser_ckpt: str | None = None,
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
                during validation. Defaults to True.
            train_embedder (bool, optional): Whether to train the context embedder if present.
                Defaults to False.
            p_classifier_free_guidance (float, optional): Probability of using classifier-free
                guidance during training. Defaults to 0.2
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
        if not diffuser.denoiser.classifier_free:
            p_classifier_free_guidance = 0

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

        for i, loss in enumerate(diffuser.extra_losses):
            loss.set_model(diffuser.denoiser)
            diffuser.extra_losses[i] = self.accelerator.prepare(loss)  # type: ignore

        if denoiser_ckpt:
            diffuser.denoiser.load_state_dict(
                torch.load(denoiser_ckpt),  # type: ignore
            )

        if optimizer_ckpt:
            optimizer.load_state_dict(
                torch.load(optimizer_ckpt, weights_only=False),  # type: ignore
            )

        if diffuser.vision_tower:
            diffuser.vision_tower = self.accelerator.prepare_model(diffuser.vision_tower)  # type: ignore

        diffuser.denoiser, train_dataloader, val_dataloader, optimizer = self.accelerator.prepare(  # type: ignore
            diffuser.denoiser, train_dataloader, val_dataloader, optimizer
        )

        if optimizer_ckpt:
            device = self.accelerator.device
            for state in optimizer.state.values():  # type: ignore
                for k, v in state.items():  # type: ignore
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)

        if scheduler is not None:
            scheduler = self.accelerator.prepare_scheduler(scheduler)  # type: ignore

        if diffuser.denoiser.context_embedder is not None and not train_embedder:  # type: ignore
            for param in diffuser.denoiser.context_embedder.parameters():  # type: ignore
                param.requires_grad = False

        best_val_loss = float("inf")

        tracker = AverageMeter()
        tq_epoch = tqdm(
            range(epoch_start, self.n_epoch), disable=not self.accelerator.is_main_process, leave=False, position=0
        )
        logging.info("Begin training")
        for epoch in tq_epoch:
            diffuser.train()
            tq_epoch.set_description(f"Epoch {epoch + 1}/{self.n_epoch}")

            tq_batch = tqdm(train_dataloader, disable=not self.accelerator.is_main_process, leave=False)  # type: ignore
            for i, batch in enumerate(tq_batch):
                with self.accelerator.accumulate(diffuser.denoiser, *diffuser.extra_losses):  # type: ignore
                    with self.accelerator.autocast():
                        self.training_step(
                            diffuser=diffuser,
                            optimizer=optimizer,  # type: ignore
                            batch=batch,
                            tracker=tracker,
                            p_classifier_free_guidance=p_classifier_free_guidance,
                            scheduler=scheduler,
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
                diffuser.eval()
                original_model: Denoiser = diffuser.denoiser  # type: ignore
                if ema_denoiser is not None:
                    diffuser.denoiser = ema_denoiser.ema_model.eval()  # type: ignore
                    for loss in diffuser.extra_losses:
                        loss.set_model(ema_denoiser.ema_model)  # type: ignore

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

                if log_validation_images:
                    logging.info("creating validation images")
                    if self.accelerator.is_main_process:
                        with self.accelerator.autocast():
                            self.log_images(
                                diffuser,
                                val_dataloader,  # type: ignore
                                epoch,
                                val_steps,
                                guidance_scale=4 if original_model.classifier_free else 0,  # type: ignore
                            )

                if ema_denoiser is not None:
                    # Restore original model and eventual hooks
                    diffuser.denoiser = original_model
                    for loss in diffuser.extra_losses:
                        loss.set_model(original_model)  # type: ignore

                if total_loss < best_val_loss:  # type: ignore
                    best_val_loss = total_loss
                    self.save_model(optimizer, diffuser, ema_denoiser, scheduler)  # type: ignore
                tracker.reset()

            self.accelerator.wait_for_everyone()

        self.accelerator.end_training()
        logging.info("Training complete")

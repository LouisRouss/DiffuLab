import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, cast

import torch
from ema_pytorch import EMA  # type: ignore[reportMissingTypeStubs]
from torch import Tensor
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from tqdm import tqdm

from diffulab.datasets.base import BatchData, BatchDataGRPO
from diffulab.diffuse.utils import SamplingOutput
from diffulab.networks.denoisers.common import ModelInput
from diffulab.networks.rewards import RewardModel
from diffulab.training.trainers.common import Trainer
from diffulab.training.utils import AverageMeter

if TYPE_CHECKING:
    from diffulab.diffuse import Diffuser
    from diffulab.networks.denoisers import Denoiser


class GRPOTrainer(Trainer):
    """
    A training class that handles the GRPO alignment training loop for diffusion/flow models with support
    for distributed training, mixed precision, gradient accumulation, and EMA model averaging.
    This class provides a complete training pipeline including validation, logging, and model checkpointing.
    It uses the Hugging Face Accelerate library for distributed training.

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
        eta (float, optional): The eta parameter for GRPO sampling. Defaults to 0.7. Controls the amount of
            stochasticity during the sampling process.
        timestep_fraction (float, optional): Fraction of the total timesteps to consider for GRPO loss
            computation. Defaults to 0.6.
        kl_beta (float, optional): Weight of the KL divergence term in the GRPO loss. Defaults to 0.0.
        eps (float, optional): Small constant for numerical stability in GRPO loss computation. Defaults to 1e-4.

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
        timestep_fraction: float = 0.6,
        kl_beta: float = 0.0,
        eps: float = 1e-4,
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
        self.timestep_fraction = timestep_fraction
        self.kl_beta = kl_beta
        self.eps = eps

    def repeat_batch(self, batch: BatchDataGRPO, n_repeat: int) -> BatchData:
        """
        Repeat a GRPO batch n_repeat times along the batch dimension.

        This is used to expand each prompt into multiple samples (e.g., n_image_per_prompt)
        while keeping per-prompt metadata aligned. Supports Tensor and list fields in
        model_inputs and extra. Floats in model_inputs are kept unchanged, and None in extra
        are preserved.

        Args:
            batch (BatchDataGRPO): Input batch with keys:
                - model_inputs: dict of tensors/lists used by the denoiser.
                - extra: dict containing auxiliary fields (e.g., 'captions' for rewards).
            n_repeat (int): Number of repetitions per original item. Must be > 0.

        Returns:
            BatchData: A batch where each original item is repeated n_repeat times.

        Raises:
            AssertionError: If n_repeat <= 0 or 'extra' is missing from the batch.
            ValueError: If an unsupported type is encountered in model_inputs or extra.
        """
        assert n_repeat > 0, "n_repeat must be a positive integer."
        assert "extra" in batch, "extra field must be present in the batch for GRPO (captions for the)."

        repeated_inputs = {}
        for k, v in batch["model_inputs"].items():
            if isinstance(v, Tensor):
                repeated_inputs[k] = v.repeat_interleave(n_repeat, dim=0)
            elif isinstance(v, list):
                repeated_inputs[k] = [item for item in v for _ in range(n_repeat)]  # type: ignore[reportUnknownVariableType]
            elif isinstance(v, float):
                repeated_inputs[k] = v
            else:
                raise ValueError(f"Unsupported type {type(v)} for key {k} in model_inputs.")

        repeated_extra = {}
        for k, v in batch["extra"].items():
            if isinstance(v, Tensor):
                repeated_extra[k] = v.repeat_interleave(n_repeat, dim=0)
            elif isinstance(v, list):
                repeated_extra[k] = [item for item in v for _ in range(n_repeat)]  # type: ignore[reportUnknownVariableType]
            elif v is None:
                repeated_extra[k] = v
            else:
                raise ValueError(f"Unsupported type {type(v)} for key {k} in extra.")

        return {"model_inputs": cast(ModelInput, repeated_inputs), "extra": repeated_extra}

    def sample_model(
        self,
        diffuser: "Diffuser",
        batch: BatchDataGRPO,
        n_image_per_prompt: int,
        image_resolution: tuple[int, int],
        guidance_scale: float = 0,
    ) -> tuple[BatchData, SamplingOutput]:
        """
        Generate samples for GRPO by repeating the batch and calling the diffuser.

        The method:
        - Ensures an input noise tensor is present (or samples one deterministically per prompt).
        - Repeats the batch n_image_per_prompt times.
        - Calls diffuser.generate in mini-batches of the original batch size.
        - Concatenates SamplingOutput fields across repeats.

        Args:
            diffuser (Diffuser): Diffusion/flow wrapper providing generate().
            batch (BatchDataGRPO): Original batch with model_inputs and extra (e.g., captions).
            n_image_per_prompt (int): Number of samples to produce per prompt.
            image_resolution (tuple[int, int]): Target (H, W) for sampling.
            guidance_scale (float, optional): Guidance scale for generation. Defaults to 0.

        Returns:
            tuple[BatchData, SamplingOutput]:
                - repeated_batch: The batch repeated n_image_per_prompt times.
                - samples: SamplingOutput with concatenated results for all repeats.

        Raises:
            AssertionError: If no samples are produced.
        """
        original_batch_size = batch["model_inputs"]["context"].shape[0]

        if diffuser.vision_tower:
            data_shape = (
                original_batch_size,
                diffuser.vision_tower.latent_channels,
                image_resolution[0] // diffuser.vision_tower.compression_factor,
                image_resolution[1] // diffuser.vision_tower.compression_factor,
            )
        else:
            data_shape = (
                original_batch_size,
                3,
                image_resolution[0],
                image_resolution[1],
            )  # only RGB images supported for now

        # We sample noise for each element of the batch if not provided
        # same noises will be used for the sampling of same prompts
        if "x" not in batch["model_inputs"]:
            batch["model_inputs"]["x"] = torch.randn(data_shape, device=self.accelerator.device)

        repeated_batch = self.repeat_batch(batch, n_image_per_prompt)

        grpo_sampling: SamplingOutput | None = None
        for mini_batch_idx in range(0, original_batch_size * n_image_per_prompt, original_batch_size):
            model_inputs = cast(
                ModelInput,
                {
                    k: v[mini_batch_idx : mini_batch_idx + original_batch_size]
                    if isinstance(v, Tensor) or isinstance(v, list)
                    else v
                    for k, v in repeated_batch["model_inputs"].items()
                },
            )

            group_grpo_sampling = diffuser.generate(
                model_inputs=model_inputs,
                guidance_scale=guidance_scale,
                return_intermediates=True,
                return_latents=False,
            )

            if grpo_sampling is None:
                grpo_sampling = group_grpo_sampling
            else:
                for k, v in group_grpo_sampling.items():
                    grpo_sampling[k] = torch.cat((cast(Tensor, grpo_sampling[k]), cast(Tensor, v)), dim=0)

        assert grpo_sampling is not None, "No samples generated during GRPO sampling."
        return repeated_batch, grpo_sampling

    def training_step(
        self,
        diffuser: "Diffuser",
        optimizer: Optimizer,
        batch: BatchDataGRPO,
        tracker: AverageMeter,
        reward_model: RewardModel,
        n_image_per_prompt: int,
        image_resolution: tuple[int, int],
        scheduler: LRScheduler | None = None,
        per_batch_scheduler: bool = False,
        ema_denoiser: EMA | None = None,
        guidance_scale: float = 0,
    ):
        """
        Perform a single GRPO training step.

        The method:
        - Samples n_image_per_prompt images per prompt with diffuser.generate.
        - Computes per-sample advantages via the reward_model.
        - Computes GRPO loss via diffuser.compute_loss(grpo=True, grpo_args=...).
        - Backpropagates and steps the optimizer (and scheduler if per-batch).
        - Optionally updates EMA.

        Args:
            diffuser (Diffuser): Diffusion/flow wrapper used for sampling and loss computation.
            optimizer (Optimizer): Optimizer for model updates.
            batch (BatchDataGRPO): Training batch with model_inputs and extra['captions'].
            tracker (AverageMeter): Metric tracker updated with train/* losses.
            reward_model (RewardModel): Callable that maps images and context to advantages.
            n_image_per_prompt (int): Number of generated images per prompt.
            image_resolution (tuple[int, int]): Sampling resolution (H, W).
            scheduler (LRScheduler | None, optional): Learning rate scheduler. Defaults to None.
            per_batch_scheduler (bool, optional): Step scheduler after each batch. Defaults to False.
            ema_denoiser (EMA | None, optional): EMA wrapper for the denoiser. Defaults to None.
            guidance_scale (float, optional): Guidance scale used during sampling. Defaults to 0.

        Returns:
            None
        """
        optimizer.zero_grad()
        original_batch_size = batch["model_inputs"]["context"].shape[0]
        repeated_batch, samples = self.sample_model(
            diffuser,
            batch,
            n_image_per_prompt=n_image_per_prompt,
            image_resolution=image_resolution,
            guidance_scale=guidance_scale,
        )

        assert "extra" in repeated_batch and "captions" in repeated_batch["extra"], (
            "Captions are required in the extra field of the batch."
        )
        advantages: Tensor = reward_model(images=samples["x"], context=repeated_batch["extra"]["captions"])
        for batch_idx in range(0, original_batch_size * n_image_per_prompt, original_batch_size):
            batch_inputs = ModelInput(
                **{
                    k: cast(Tensor, v[batch_idx : batch_idx + original_batch_size])  # type: ignore
                    for k, v in repeated_batch["model_inputs"].items()
                }
            )
            batch_samples = SamplingOutput(
                **{
                    k: cast(Tensor, v[batch_idx : batch_idx + original_batch_size])  # type: ignore
                    for k, v in samples.items()
                }
            )
            batch_advantages = advantages[batch_idx : batch_idx + original_batch_size]

            losses = diffuser.compute_loss(
                model_inputs=batch_inputs,
                grpo=True,
                grpo_args={
                    "sampling": batch_samples,
                    "advantages": batch_advantages,
                    "kl_beta": self.kl_beta,
                    "eps": self.eps,
                    "timestep_fraction": self.timestep_fraction,
                    "guidance_scale": guidance_scale,
                },
            )
            for key, loss in losses.items():
                tracker.update(loss.item(), key=f"train/{key}")

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
        batch: BatchDataGRPO,
        tracker: AverageMeter,
        reward_model: RewardModel,
        n_image_per_prompt: int,
        image_resolution: tuple[int, int],
        guidance_scale: float = 4.0,
    ):
        """
        Perform a validation step for GRPO.

        The method:
        - Generates n_image_per_prompt samples per prompt.
        - Computes advantages with the reward model.
        - Computes GRPO losses for logging (no grads).

        Args:
            diffuser (Diffuser): Diffusion/flow wrapper used for sampling and loss computation.
            batch (BatchDataGRPO): Validation batch with model_inputs and extra['captions'].
            tracker (AverageMeter): Metric tracker updated with val/* losses.
            reward_model (RewardModel): Callable that maps images and context to advantages.
            n_image_per_prompt (int): Number of generated images per prompt.
            image_resolution (tuple[int, int]): Sampling resolution (H, W).
            guidance_scale (float, optional): Guidance scale used during sampling. Defaults to 4.0.

        Returns:
            None
        """
        original_batch_size = batch["model_inputs"]["context"].shape[0]
        repeated_batch, samples = self.sample_model(
            diffuser,
            batch,
            n_image_per_prompt=n_image_per_prompt,
            image_resolution=image_resolution,
            guidance_scale=guidance_scale,
        )
        advantages: Tensor = reward_model(images=samples["x"], context=batch["extra"]["captions"])
        for batch_idx in range(0, original_batch_size * n_image_per_prompt, original_batch_size):
            batch_inputs = ModelInput(
                **{
                    k: cast(Tensor, v[batch_idx : batch_idx + original_batch_size])  # type: ignore
                    for k, v in repeated_batch["model_inputs"].items()
                }
            )
            batch_samples = SamplingOutput(
                **{
                    k: cast(Tensor, v[batch_idx : batch_idx + original_batch_size])  # type: ignore
                    for k, v in samples.items()
                }
            )
            batch_advantages = advantages[batch_idx : batch_idx + original_batch_size]

            losses = diffuser.compute_loss(
                model_inputs=batch_inputs,
                grpo=True,
                grpo_args={
                    "sampling": batch_samples,
                    "advantages": batch_advantages,
                    "kl_beta": self.kl_beta,
                    "eps": self.eps,
                    "timestep_fraction": self.timestep_fraction,
                    "guidance_scale": guidance_scale,
                },
            )

            for key, loss in losses.items():
                tracker.update(loss.item(), key=f"val/{key}")

    def train(
        self,
        diffuser: "Diffuser",
        reward_model: RewardModel,
        optimizer: Optimizer,
        train_dataloader: Iterable[BatchDataGRPO],
        val_dataloader: Iterable[BatchDataGRPO] | None = None,
        scheduler: LRScheduler | None = None,
        per_batch_scheduler: bool = False,
        log_validation_images: bool = True,
        val_steps: int = 25,
        optimizer_ckpt: str | None = None,
        denoiser_ckpt: str | None = None,
        ema_ckpt: str | None = None,
        epoch_start: int = 0,
        n_image_per_prompt: int = 16,
        guidance_scale: float = 4.0,
        image_resolution: tuple[int, int] = (512, 512),
    ):
        """
        Main training loop for GRPO alignment training.

        This orchestrates GRPO training with distributed/mixed precision support, gradient
        accumulation, optional EMA, periodic validation, image logging, and checkpointing.

        Args:
            diffuser (Diffuser): Diffusion/flow wrapper. Must have a context embedder in the denoiser.
            reward_model (RewardModel): Callable producing per-sample advantages from images and context.
            optimizer (Optimizer): Optimizer for parameter updates.
            train_dataloader (Iterable[BatchDataGRPO]): Training data iterator yielding GRPO batches.
            val_dataloader (Iterable[BatchDataGRPO] | None, optional): Validation iterator. Defaults to None.
            scheduler (LRScheduler | None, optional): LR scheduler. Prepared with accelerator if provided.
            per_batch_scheduler (bool, optional): Step scheduler after each batch instead of each epoch.
                Defaults to False.
            log_validation_images (bool, optional): Generate and log images during validation. Defaults to True.
            val_steps (int, optional): Number of sampler steps for validation image generation. Defaults to 25.
            optimizer_ckpt (str | None, optional): Path to optimizer state to resume. Defaults to None.
            denoiser_ckpt (str | None, optional): Path to denoiser weights to load. Defaults to None.
            ema_ckpt (str | None, optional): Path to EMA model weights to load. Defaults to None.
            epoch_start (int, optional): Starting epoch index for resuming. Defaults to 0.
            n_image_per_prompt (int, optional): Number of generated images per prompt for GRPO. Defaults to 16.
            guidance_scale (float, optional): Guidance scale during sampling (train/val). Defaults to 4.0.
            image_resolution (tuple[int, int], optional): Sampling resolution (H, W). Defaults to (512, 512).

        Returns:
            None

        Notes:
            - Freezes the denoiser's context_embedder parameters during GRPO (required for alignment).
            - If EMA is enabled, validation can run on the EMA model and best checkpoints are saved
              when validation loss improves.
            - Uses Hugging Face Accelerate for preparation, device placement, mixed precision, and logging.
        """
        assert diffuser.denoiser.context_embedder is not None, (
            "Alignment training requires a context embedder in the denoiser model."
        )

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

        reward_model.set_n_image_per_prompt(n_image_per_prompt)

        diffuser.denoiser, train_dataloader, val_dataloader, optimizer, reward_model = self.accelerator.prepare(  # type: ignore
            diffuser.denoiser, train_dataloader, val_dataloader, optimizer, reward_model
        )

        if optimizer_ckpt:
            device = self.accelerator.device
            for state in optimizer.state.values():  # type: ignore
                for k, v in state.items():  # type: ignore
                    if isinstance(v, Tensor):
                        state[k] = v.to(device)

        if scheduler is not None:
            scheduler = self.accelerator.prepare_scheduler(scheduler)  # type: ignore

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
            for batch in tq_batch:
                with self.accelerator.accumulate(diffuser.denoiser):  # type: ignore
                    with self.accelerator.autocast():
                        self.training_step(
                            diffuser=diffuser,
                            optimizer=optimizer,  # type: ignore
                            batch=batch,
                            tracker=tracker,
                            reward_model=reward_model,  # type: ignore
                            n_image_per_prompt=n_image_per_prompt,
                            image_resolution=image_resolution,
                            scheduler=scheduler,
                            per_batch_scheduler=per_batch_scheduler,
                            ema_denoiser=ema_denoiser,  # type: ignore
                            guidance_scale=guidance_scale,
                        )
                        tq_batch.set_description(
                            f"Loss: {sum(v for k, v in tracker.avg.items() if k.startswith('train/')):.4f}"
                        )

            if scheduler is not None and not per_batch_scheduler:
                scheduler.step()

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
                            batch=val_batch,
                            tracker=tracker,
                            reward_model=reward_model,  # type: ignore
                            n_image_per_prompt=n_image_per_prompt,
                            image_resolution=image_resolution,
                            guidance_scale=guidance_scale,
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
                                guidance_scale=4,
                            )

                if ema_denoiser is not None:
                    diffuser.denoiser = original_model

                if total_loss < best_val_loss:
                    best_val_loss = total_loss
                    self.save_model(optimizer, diffuser, ema_denoiser, scheduler)  # type: ignore
                tracker.reset()

            self.accelerator.wait_for_everyone()

        self.accelerator.end_training()
        logging.info("Training complete")

import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, cast

import torch
from ema_pytorch import EMA  # type: ignore [stub file not found]
from torch import Tensor
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from tqdm import tqdm

from diffulab.datasets.base import BatchDataGRPO
from diffulab.diffuse.modelizations.utils import GRPOSamplingOutput
from diffulab.networks.denoisers.common import ExtraInputGRPO, ModelInputGRPO
from diffulab.networks.rewards import RewardModel
from diffulab.training.trainers import Trainer
from diffulab.training.utils import AverageMeter

if TYPE_CHECKING:
    from diffulab.diffuse import Diffuser


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

    def repeat_batch(self, batch: BatchDataGRPO, n_repeat: int) -> BatchDataGRPO:
        assert n_repeat > 0, "n_repeat must be a positive integer."

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

        return {"model_inputs": cast(ModelInputGRPO, repeated_inputs), "extra": cast(ExtraInputGRPO, repeated_extra)}

    def sample_model(
        self,
        diffuser: "Diffuser",
        batch: BatchDataGRPO,
        n_image_per_prompt: int,
        image_resolution: tuple[int, int],
        guidance_scale: float = 0,
        eta: float = 0.7,
    ) -> tuple[BatchDataGRPO, GRPOSamplingOutput]:
        original_batch_size = batch["model_inputs"]["context"].shape[0]
        repeated_batch = self.repeat_batch(batch, n_image_per_prompt)
        groups_per_minibatch = max(1, original_batch_size // n_image_per_prompt)

        grpo_sampling: GRPOSamplingOutput | None = None
        for group in range(0, original_batch_size, groups_per_minibatch):
            group_idx_min = group * n_image_per_prompt
            group_idx_max = min((group + groups_per_minibatch, original_batch_size)) * n_image_per_prompt

            model_inputs = cast(
                ModelInputGRPO,
                {
                    k: v[group_idx_min:group_idx_max] if isinstance(v, Tensor) or isinstance(v, list) else v
                    for k, v in repeated_batch["model_inputs"].items()
                },
            )

            group_grpo_sampling = diffuser.generate_GRPO(
                model_inputs=model_inputs,
                data_shape=(n_image_per_prompt, 3, image_resolution[0], image_resolution[1]),
                guidance_scale=guidance_scale,
                eta=eta,
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
        eta: float = 0.7,
        timestep_fraction: float = 0.6,
        kl_beta: float = 0.0,
        eps: float = 1e-4,
    ):
        optimizer.zero_grad()
        original_batch_size = batch["model_inputs"]["context"].shape[0]
        repeated_batch, samples = self.sample_model(
            diffuser,
            batch,
            n_image_per_prompt=n_image_per_prompt,
            image_resolution=image_resolution,
            guidance_scale=guidance_scale,
            eta=eta,
        )
        advantages: Tensor = reward_model(images=samples["x"], context=batch["extra"]["captions"])
        for batch_idx in range(0, original_batch_size * n_image_per_prompt, original_batch_size):
            batch_inputs: ModelInputGRPO = {
                k: v[batch_idx : batch_idx + original_batch_size] for k, v in repeated_batch["model_inputs"].items()
            }  # type: ignore
            batch_samples: GRPOSamplingOutput = {
                k: v[batch_idx : batch_idx + original_batch_size] for k, v in samples.items()
            }  # type: ignore
            batch_advantages = advantages[batch_idx : batch_idx + original_batch_size]

            losses = diffuser.compute_grpo_loss(
                model_inputs=batch_inputs,
                grpo_sampling_output=batch_samples,
                advantages=batch_advantages,
                kl_beta=kl_beta,
                eps=eps,
                timestep_fraction=timestep_fraction,
                guidance_scale=guidance_scale,
                eta=eta,
            )
            for key, loss in losses.items():
                tracker.update(loss.item(), key=f"train/{key}")

            self.accelerator.backward(loss)  # type: ignore
            optimizer.step()
            if scheduler is not None and per_batch_scheduler:
                scheduler.step()
            if ema_denoiser is not None:
                ema_denoiser.update()

    # def validation_step(self):
    #     pass

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
        eta: float = 0.7,
        n_image_per_prompt: int = 16,
        guidance_scale: float = 4.0,
        image_resolution: tuple[int, int] = (512, 512),
        timestep_fraction: float = 0.6,
        kl_beta: float = 0.0,
        eps: float = 1e-4,
    ):
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
                            eta=eta,
                            timestep_fraction=timestep_fraction,
                            kl_beta=kl_beta,
                            eps=eps,
                        )
                        tq_batch.set_description(
                            f"Loss: {sum(v for k, v in tracker.avg.items() if k.startswith('train/')):.4f}"
                        )

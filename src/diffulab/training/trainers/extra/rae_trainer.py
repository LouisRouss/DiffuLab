import logging
import os
import random
from datetime import datetime
from math import ceil, sqrt
from pathlib import Path
from typing import Any, Iterable, cast

import numpy as np
import torch
import torchvision.transforms.v2 as v2  # type: ignore[reportMissingTypeStubs]
import wandb
from accelerate import Accelerator  # type: ignore [stub file not found]
from accelerate.utils import TorchDynamoPlugin  # type: ignore [stub file not found]
from ema_pytorch import EMA  # type: ignore[reportMissingTypeStubs]
from numpy.typing import NDArray
from torch import Tensor
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torchvision.utils import make_grid  # type: ignore
from tqdm import tqdm

from diffulab.datasets.base import BatchData
from diffulab.networks.disc.rae import RAEDiscriminator
from diffulab.networks.vision_towers.rae import RAE, RAEDecoder
from diffulab.training.losses.rae import LPIPS, GANLoss
from diffulab.training.utils import AverageMeter

HOME_PATH = Path.home()


class RAETrainer:
    """
    A training class that handles the supervised training loop for the RAE rae with support
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

    def disc_augment(
        self, x: Tensor, prob: float = 1.0, cutout_ratio: float = 0.2, translate_ratio: float = 0.125
    ) -> Tensor:
        """Apply data augmentation to the input tensor for the discriminator.

        Args:
            x (Tensor): Input tensor of shape (B, C, H, W) in the range [-1, 1].
        Returns:
            Tensor: Augmented tensor of shape (B, C, H, W) in the range [-1, 1].
        """
        x = x.float()
        x = (x + 1.0) * 0.5  # rescale to [0, 1]
        _, _, H, W = x.shape
        dtype = x.dtype
        # crop
        if random.random() < prob:
            delta = int(round(translate_ratio * min(H, W)))
            translate = v2.RandomCrop(size=(H, W), padding=(delta, delta, delta, delta), pad_if_needed=False)
            x = translate(x)
        # color
        if random.random() < prob:
            bias = AddPerChannelBias(0.5)
            color = v2.ColorJitter(brightness=0.5, contrast=(0.5, 1.5))
            x = bias(x)
            x = color(x)
            x = x.clamp(0.0, 1.0).to(dtype)
        # cutout
        if random.random() < prob:
            area = cutout_ratio**2
            cutout = v2.RandomErasing(p=1.0, scale=(area, area), ratio=(W / H, W / H), value=0.0, inplace=False)
            x = cutout(x)
        x = x * 2.0 - 1.0  # rescale to [-1, 1]
        x = x.to(dtype)
        return x

    @torch.no_grad()  # type: ignore
    def log_images(
        self,
        rae: RAE,
        val_dataloader: Iterable[BatchData],
        epoch: int,
    ) -> None:
        batch = next(iter(val_dataloader))
        real = cast(torch.Tensor | None, batch.get("extra", {}).get("x0")).cpu()  # type: ignore
        assert real is not None, "Validation batch must contain 'x0' in extra"
        reconstruct = rae.decode(batch["model_inputs"]["x"])
        reconstruct = (reconstruct * 0.5 + 0.5).clamp(0, 1).cpu()
        paired = torch.cat([real, reconstruct], dim=-1)
        grid = make_grid(paired, nrow=int(ceil(sqrt(real.shape[0]))), padding=2)
        np_grid: NDArray[np.uint8] = (grid * 255).round().byte().permute(1, 2, 0).numpy()  # type: ignore
        to_log = wandb.Image(np_grid, caption="Validation: Real (left) vs Reconstruction (right)")
        self.accelerator.log({"val/images_comparison": to_log}, step=epoch + 1, log_kwargs={"wandb": {"commit": True}})  # type: ignore

    def save_model(
        self,
        rae: RAE,
        disc: RAEDiscriminator,
        rae_optimizer: Optimizer,
        disc_optimizer: Optimizer,
        ema: EMA | None = None,
        rae_scheduler: LRScheduler | None = None,
        disc_scheduler: LRScheduler | None = None,
    ) -> None:
        unwrapped_rae: RAE = self.accelerator.unwrap_model(rae)  # type: ignore
        state_dict = cast(dict[str, Tensor], unwrapped_rae.decoder.state_dict())  # type: ignore
        if self.compile:
            state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        self.accelerator.save(state_dict, self.save_path / "rae_decoder.pt")  # type: ignore

        unwrapped_disc: RAEDiscriminator = self.accelerator.unwrap_model(disc)  # type: ignore
        state_dict = cast(dict[str, Tensor], unwrapped_disc.state_dict())  # type: ignore
        if self.compile:
            state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        self.accelerator.save(state_dict, self.save_path / "rae_discriminator.pt")  # type: ignore

        self.accelerator.save(rae_optimizer.optimizer.state_dict(), self.save_path / "rae_optimizer.pt")  # type: ignore
        self.accelerator.save(disc_optimizer.optimizer.state_dict(), self.save_path / "disc_optimizer.pt")  # type: ignore

        if ema is not None:
            unwrapped_ema = self.accelerator.unwrap_model(ema)  # type: ignore
            self.accelerator.save(unwrapped_ema.ema_model.state_dict(), self.save_path / "ema.pt")  # type: ignore

        if rae_scheduler is not None:
            self.accelerator.save(rae_scheduler.scheduler.state_dict(), self.save_path / "rae_scheduler.pt")  # type: ignore
        if disc_scheduler is not None:
            self.accelerator.save(disc_scheduler.scheduler.state_dict(), self.save_path / "disc_scheduler.pt")  # type: ignore

    def training_step(
        self,
        rae: RAE,
        disc: RAEDiscriminator,
        rae_optimizer: Optimizer,
        disc_optimizer: Optimizer,
        batch: BatchData,
        lpips_loss: LPIPS,
        gan_loss: GANLoss,
        train_gan: bool,
        train_disc: bool,
        tracker: AverageMeter,
        rae_scheduler: LRScheduler | None = None,
        disc_scheduler: LRScheduler | None = None,
        per_batch_scheduler: bool = False,
        ema: EMA | None = None,
        lambda_lpips: float = 1.0,
        lambda_gan: float = 0.75,
        use_adaptive_weight_loss: bool = True,
        use_lpips: bool = True,
    ) -> None:
        # Train generator (RAE)
        for p in disc.heads.parameters():
            p.requires_grad = False
        rae_optimizer.zero_grad(set_to_none=True)
        real = cast(torch.Tensor | None, batch.get("extra", {}).get("x0"))
        assert real is not None, "Training batch must contain 'x0' in extra"
        real_rescaled = real * 2.0 - 1.0  # rescale to [-1, 1]
        # rae input is assumed to be in [0, 1] and output in [-1, 1] for consistency in our framework
        fake = rae.decode(batch["model_inputs"]["x"])
        loss_rae = torch.nn.functional.l1_loss(fake, real_rescaled)
        tracker.update(key="train/loss_l1", val=loss_rae.item())
        if use_lpips:
            loss_lpips: Tensor = lpips_loss(fake, real_rescaled)
            tracker.update(key="train/loss_lpips", val=loss_lpips.item())
            loss_rae = loss_rae + lambda_lpips * loss_lpips
        if train_gan:
            # Compute adversarial loss for RAE
            disc.eval()
            fake_aug = self.disc_augment(fake)
            fake_pred = disc(fake_aug * 0.5 + 0.5)
            loss_gan_rae: Tensor = gan_loss(logits_fake=fake_pred, is_disc=False)
            if use_adaptive_weight_loss:
                # Compute adaptive weight for GAN loss
                loss_rae_grads = torch.autograd.grad(
                    loss_rae, cast(torch.nn.Linear, rae.decoder.last_layer[1]).weight, retain_graph=True
                )[0]
                loss_gan_grads = torch.autograd.grad(
                    loss_gan_rae, cast(torch.nn.Linear, rae.decoder.last_layer[1]).weight, retain_graph=True
                )[0]
                adaptive_weight = cast(Tensor, torch.norm(loss_rae_grads) / (torch.norm(loss_gan_grads) + 1e-6))  # type: ignore
                adaptive_weight = torch.clamp(adaptive_weight, 0.0, 1e4).detach()
                loss_rae = loss_rae + adaptive_weight * lambda_gan * loss_gan_rae
            else:
                loss_rae = loss_rae + lambda_gan * loss_gan_rae
            tracker.update(key="train/loss_gan_rae", val=loss_gan_rae.item())
        self.accelerator.backward(loss_rae)  # type: ignore
        rae_optimizer.step()
        if rae_scheduler is not None and per_batch_scheduler:
            rae_scheduler.step()
        if ema is not None:
            ema.update()

        # train discriminator
        for p in disc.heads.parameters():
            p.requires_grad = True
        if train_disc:
            disc.train()
            disc_optimizer.zero_grad(set_to_none=True)
            fake = fake.detach()
            fake_aug = self.disc_augment(fake)
            real_aug = self.disc_augment(real_rescaled)
            fake_pred = disc(fake_aug * 0.5 + 0.5)  # rescale to [0, 1]
            real_pred = disc(real_aug * 0.5 + 0.5)  # rescale to [0, 1]
            loss_disc: Tensor = gan_loss(logits_real=real_pred, logits_fake=fake_pred, is_disc=True)
            tracker.update(key="train_disc/loss_disc", val=loss_disc.item())
            self.accelerator.backward(loss_disc)  # type: ignore
            disc_optimizer.step()
            if disc_scheduler is not None and per_batch_scheduler:
                disc_scheduler.step()

    def validation_step(
        self,
        rae: RAE,
        disc: RAEDiscriminator,
        batch: BatchData,
        lpips_loss: LPIPS,
        gan_loss: GANLoss,
        train_gan: bool,
        train_disc: bool,
        tracker: AverageMeter,
        use_lpips: bool = True,
    ) -> None:
        real = cast(torch.Tensor | None, batch.get("extra", {}).get("x0"))
        assert real is not None, "Validation batch must contain 'x0' in extra"
        real_rescaled = real * 2.0 - 1.0  # rescale to [-1, 1]
        fake = rae.decode(batch["model_inputs"]["x"])
        loss_l1 = torch.nn.functional.l1_loss(fake, real_rescaled)
        tracker.update(key="val/loss_l1", val=loss_l1.item())
        if use_lpips:
            loss_lpips: Tensor = lpips_loss(fake, real_rescaled)
            tracker.update(key="val/loss_lpips", val=loss_lpips.item())
        if train_gan:
            fake_pred = disc(fake * 0.5 + 0.5)  # rescale to [0, 1]
            loss_gan_rae: Tensor = gan_loss(logits_fake=fake_pred, is_disc=False)
            tracker.update(key="val/loss_gan_rae", val=loss_gan_rae.item())

        # train discriminator
        if train_disc:
            fake_pred = disc(fake * 0.5 + 0.5)  # rescale to [0, 1]
            real_pred = disc(real)
            loss_disc: Tensor = gan_loss(logits_real=real_pred, logits_fake=fake_pred, is_disc=True)
            tracker.update(key="val_disc/loss_disc", val=loss_disc.item())

    def train(
        self,
        rae: RAE,
        disc: RAEDiscriminator,
        rae_optimizer: Optimizer,
        disc_optimizer: Optimizer,
        train_dataloader: Iterable[BatchData],
        val_dataloader: Iterable[BatchData] | None = None,
        rae_scheduler: LRScheduler | None = None,
        disc_scheduler: LRScheduler | None = None,
        per_batch_scheduler: bool = False,
        log_validation_images: bool = True,
        rae_optimizer_ckpt: str | None = None,
        disc_optimizer_ckpt: str | None = None,
        rae_scheduler_ckpt: str | None = None,
        disc_scheduler_ckpt: str | None = None,
        rae_ckpt: str | None = None,
        disc_ckpt: str | None = None,
        ema_ckpt: str | None = None,
        epoch_start: int = 0,
        disc_epoch_start: int = 6,
        gan_epoch_start: int = 8,
        lpips_epoch_start: int = 1,
        lambda_lpips: float = 1.0,
        lambda_gan: float = 0.75,
        use_adaptive_weight_loss: bool = True,
    ):
        assert not (self.compile and use_adaptive_weight_loss), (
            "Adaptive weight loss is not supported with compiled mode (double backward)."
        )
        if self.use_ema:
            ema = EMA(
                rae.decoder,
                beta=self.ema_rate,
                update_after_step=self.ema_update_after_step,
                update_every=self.ema_update_every,
            ).to(self.accelerator.device)
            if ema_ckpt:
                ema.ema_model.load_state_dict(torch.load(ema_ckpt, weights_only=True))  # type: ignore
            ema = self.accelerator.prepare(ema)  # type: ignore
        else:
            ema = None

        if rae_ckpt:
            rae.decoder.load_state_dict(
                torch.load(rae_ckpt),  # type: ignore
            )

        if disc_ckpt:
            disc.load_state_dict(
                torch.load(disc_ckpt),  # type: ignore
            )

        if rae_optimizer_ckpt:
            rae_optimizer.load_state_dict(
                torch.load(rae_optimizer_ckpt, weights_only=False),  # type: ignore
            )

        if disc_optimizer_ckpt:
            disc_optimizer.load_state_dict(
                torch.load(disc_optimizer_ckpt, weights_only=False),  # type: ignore
            )

        if rae_scheduler_ckpt and rae_scheduler is not None:
            rae_scheduler.load_state_dict(
                torch.load(rae_scheduler_ckpt, weights_only=False),  # type: ignore
            )
        if disc_scheduler_ckpt and disc_scheduler is not None:
            disc_scheduler.load_state_dict(
                torch.load(disc_scheduler_ckpt, weights_only=False),  # type: ignore
            )

        gan_loss = GANLoss()
        lpips_loss = LPIPS()

        rae, disc, train_dataloader, val_dataloader, rae_optimizer, disc_optimizer, lpips_loss, gan_loss = (  # type: ignore
            self.accelerator.prepare(  # type: ignore
                rae, disc, train_dataloader, val_dataloader, rae_optimizer, disc_optimizer, lpips_loss, gan_loss
            )
        )

        if rae_optimizer_ckpt:
            device = self.accelerator.device
            for state in rae_optimizer.state.values():  # type: ignore
                for k, v in state.items():  # type: ignore
                    if isinstance(v, Tensor):
                        state[k] = v.to(device)

        if disc_optimizer_ckpt:
            device = self.accelerator.device
            for state in disc_optimizer.state.values():  # type: ignore
                for k, v in state.items():  # type: ignore
                    if isinstance(v, Tensor):
                        state[k] = v.to(device)

        if rae_scheduler is not None:
            rae_scheduler = self.accelerator.prepare_scheduler(rae_scheduler)
        if disc_scheduler is not None:
            disc_scheduler = self.accelerator.prepare_scheduler(disc_scheduler)

        tracker = AverageMeter()
        tq_epoch = tqdm(
            range(epoch_start, self.n_epoch), disable=not self.accelerator.is_main_process, leave=False, position=0
        )
        logging.info("Begin training")
        for epoch in tq_epoch:
            rae.train()  # type: ignore
            disc.train()  # type: ignore
            tq_epoch.set_description(f"Epoch {epoch + 1}/{self.n_epoch}")

            tq_batch = tqdm(train_dataloader, disable=not self.accelerator.is_main_process, leave=False)  # type: ignore
            for batch in tq_batch:
                with self.accelerator.accumulate(rae, disc):  # type: ignore
                    with self.accelerator.autocast():
                        self.training_step(
                            rae=rae,  # type: ignore
                            disc=disc,  # type: ignore
                            rae_optimizer=rae_optimizer,  # type: ignore
                            disc_optimizer=disc_optimizer,  # type: ignore
                            batch=batch,
                            lpips_loss=lpips_loss,  # type: ignore
                            gan_loss=gan_loss,  # type: ignore
                            tracker=tracker,
                            rae_scheduler=rae_scheduler,
                            disc_scheduler=disc_scheduler,
                            per_batch_scheduler=per_batch_scheduler,
                            ema=ema,  # type: ignore
                            lambda_lpips=lambda_lpips,
                            lambda_gan=lambda_gan,
                            use_adaptive_weight_loss=use_adaptive_weight_loss,
                            train_gan=(epoch + 1 >= gan_epoch_start),
                            train_disc=(epoch + 1 >= disc_epoch_start),
                            use_lpips=(epoch + 1 >= lpips_epoch_start),
                        )
                        tq_batch.set_description(
                            f"Loss: {sum(v for k, v in tracker.avg.items() if k.startswith('train/')):.4f}"
                        )
            if rae_scheduler is not None and not per_batch_scheduler:
                rae_scheduler.step()
            if disc_scheduler is not None and not per_batch_scheduler and epoch + 1 >= disc_epoch_start:
                disc_scheduler.step()

            for key, value in tracker.avg.items():
                if key.startswith("train"):
                    gathered_loss: Tensor = self.accelerator.gather(  # type: ignore
                        torch.tensor(value, device=self.accelerator.device)
                    )
                    self.accelerator.log(  # type: ignore
                        {key: gathered_loss.mean().item()}, step=epoch + 1
                    )
            tracker.reset()

            if val_dataloader is not None:
                rae.eval()  # type: ignore
                disc.eval()  # type: ignore
                original_model: RAEDecoder = rae.decoder  # type: ignore
                if ema is not None:
                    rae.decoder = ema.ema_model.eval()  # type: ignore

                tq_val_batch = tqdm(
                    val_dataloader,  # type: ignore
                    disable=not self.accelerator.is_main_process,
                    leave=False,
                    position=1,
                )
                for val_batch in tq_val_batch:
                    with self.accelerator.autocast():
                        self.validation_step(
                            rae=rae,  # type: ignore
                            disc=disc,  # type: ignore
                            batch=val_batch,
                            lpips_loss=lpips_loss,  # type: ignore
                            gan_loss=gan_loss,  # type: ignore
                            train_gan=(epoch + 1 >= gan_epoch_start),
                            train_disc=(epoch + 1 >= disc_epoch_start),
                            tracker=tracker,
                            use_lpips=(epoch + 1 >= lpips_epoch_start),
                        )
                    tq_val_batch.set_description(
                        f"Val Loss: {sum(v for k, v in tracker.avg.items() if k.startswith('val/')):.4f}"
                    )

                for key, value in tracker.avg.items():
                    if key.startswith("val"):
                        gathered_loss: Tensor = self.accelerator.gather(  # type: ignore
                            torch.tensor(value, device=self.accelerator.device)
                        )
                        self.accelerator.log(  # type: ignore
                            {key: gathered_loss.mean().item()}, step=epoch + 1
                        )

                if log_validation_images:
                    logging.info("creating validation images")
                    if self.accelerator.is_main_process:
                        with self.accelerator.autocast():
                            self.log_images(
                                rae=rae,  # type: ignore
                                val_dataloader=val_dataloader,  # type: ignore
                                epoch=epoch,
                            )

                if ema is not None:
                    # Restore original model
                    rae.decoder = original_model

                self.save_model(
                    rae=rae,  # type: ignore
                    disc=disc,  # type: ignore
                    rae_optimizer=rae_optimizer,  # type: ignore
                    disc_optimizer=disc_optimizer,  # type: ignore
                    ema=ema,  # type: ignore
                    rae_scheduler=rae_scheduler,
                    disc_scheduler=disc_scheduler,
                )
                tracker.reset()

            self.accelerator.wait_for_everyone()

        self.accelerator.end_training()
        logging.info("Training complete")


class AddPerChannelBias(torch.nn.Module):
    def __init__(self, max_abs: float = 0.5):
        super().__init__()  # type: ignore
        self.max_abs = max_abs

    def forward(self, x: Tensor) -> Tensor:
        B, C, _, _ = x.shape
        bias = (torch.rand(B, C, 1, 1, device=x.device, dtype=x.dtype) - 0.5) * 2 * self.max_abs
        return x + bias

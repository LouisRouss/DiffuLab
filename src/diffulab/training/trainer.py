import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import torch
from accelerate import Accelerator  # type: ignore
from ema_pytorch import EMA  # type: ignore
from torch import Tensor
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from tqdm import tqdm

from diffulab.diffuse.flow import Diffuser
from diffulab.networks.denoisers.common import Denoiser
from diffulab.training.utils import AverageMeter

HOME_PATH = Path.home()


class Trainer:
    def __init__(
        self,
        n_epoch: int,
        batch_size: int,
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
        assert (
            HOME_PATH / ".cache" / "huggingface" / "accelerate" / "default_config.yaml"
        ).exists(), "please run `accelerate config` first in the CLI and save the config at the default location"
        self.n_epoch = n_epoch
        self.batch_size = batch_size
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
        self.accelerator.init_trackers(project_name=project_name, config=run_config, init_kwargs=init_kwargs)  # type: ignore

    def training_step(
        self,
        diffuser: Diffuser,
        optimizer: Optimizer,
        batch: dict[str, Any],
        tracker: AverageMeter,
        p_classifier_free_guidance: float = 0,
        scheduler: LRScheduler | None = None,
        per_batch_scheduler: bool = False,
        ema_denoiser: EMA | None = None,
    ) -> None:
        optimizer.zero_grad()  # type: ignore
        timesteps = diffuser.draw_timesteps(self.batch_size).to(self.accelerator.device)
        batch.update({"p": p_classifier_free_guidance})
        loss = diffuser.compute_loss(model_inputs=batch, timesteps=timesteps)
        tracker.update(loss.item(), key="loss")
        self.accelerator.backward(loss)  # type: ignore
        optimizer.step()  # type: ignore
        if scheduler is not None and per_batch_scheduler:
            scheduler.step()  # type: ignore
        if ema_denoiser is not None:
            ema_denoiser.update()  # type: ignore

    @torch.no_grad()  # type: ignore
    def validation_step(
        self,
        diffuser: Diffuser,
        val_batch: dict[str, Any],
        tracker: AverageMeter,
        ema_eval: Denoiser | None = None,
    ) -> None:
        timesteps = diffuser.draw_timesteps(self.batch_size).to(self.accelerator.device)
        val_loss = (
            diffuser.compute_loss(model_inputs=val_batch, timesteps=timesteps)
            if not self.use_ema
            else diffuser.flow.compute_loss(model=ema_eval, model_inputs=val_batch, timesteps=timesteps)  # type: ignore
        )
        tracker.update(val_loss.item(), key="val_loss")

    def save_model(
        self,
        optimizer: Optimizer,
        ema_denoiser: Denoiser | None = None,
        scheduler: LRScheduler | None = None,
    ) -> None:
        unwrapped_denoiser: Denoiser = self.accelerator.unwrap_model(diffuser.denoiser)  # type: ignore
        self.accelerator.save(unwrapped_denoiser.state_dict(), self.save_path / "denoiser.pth")  # type: ignore
        self.accelerator.save(optimizer.optimizer.state_dict(), self.save_path / "optimizer.pth")  # type: ignore
        if ema_denoiser is not None:
            self.accelerator.save(ema_denoiser.ema_model.state_dict(), self.save_path / "ema.pth")  # type: ignore
        if scheduler is not None:
            self.accelerator.save(scheduler.scheduler.state_dict(), self.save_path / "scheduler.pth")  # type: ignore

    @torch.no_grad()  # type: ignore
    def log_images(
        self,
        diffuser: Diffuser,
        val_dataloader: Iterable[dict[str, Any]],
        epoch: int,
        ema_eval: Diffuser | None = None,
        val_steps: int = 25,
    ) -> None:
        diffuser.eval()
        original_steps = diffuser.n_steps
        diffuser.set_steps(val_steps)
        (Path(self.save_path) / "images").mkdir(exist_ok=True)
        wandb_tracker = self.accelerator.get_tracker("wandb")  # type: ignore
        batch: dict[str, Any] = next(iter(val_dataloader))  # type: ignore
        images = (
            diffuser.generate(data_shape=batch["x"].shape, model_inputs=batch)  # type: ignore
            if not self.use_ema
            else diffuser.flow.denoise(model=ema_eval, data_shape=batch["x"].shape, model_inputs=batch)  # type: ignore
        )
        images = wandb_tracker.Image(images, caption="Validation Images")  # type: ignore
        self.accelerator.log({"val/images": images}, step=epoch)  # type: ignore
        diffuser.set_steps(original_steps)

    def train(
        self,
        diffuser: Diffuser,
        optimizer: Optimizer,
        train_dataloader: Iterable[dict[str, Any]],
        val_dataloader: Iterable[dict[str, Any]] | None = None,
        scheduler: LRScheduler | None = None,
        per_batch_scheduler: bool = False,
        log_validation_images: bool = False,
        train_embedder: bool = False,
        p_classifier_free_guidance: float = 0,
        val_steps: int = 25,
    ):
        if self.use_ema:
            ema_denoiser = EMA(
                diffuser.denoiser,
                beta=self.ema_rate,
                update_after_step=self.ema_update_after_step,
                update_every=self.ema_update_every,
            )
        else:
            ema_denoiser = None
        diffuser.denoiser, train_dataloader, val_dataloader, optimizer = self.accelerator.prepare(  # type: ignore
            diffuser.denoiser, train_dataloader, val_dataloader, optimizer
        )
        if scheduler is not None:
            scheduler = self.accelerator.prepare_scheduler(scheduler)  # type: ignore
        best_val_loss = float("inf")

        if diffuser.denoiser.context_embedder is not None and not train_embedder:  # type: ignore
            for param in diffuser.denoiser.context_embedder.parameters():  # type: ignore
                param.requires_grad = False

        tracker = AverageMeter()
        tq_epoch = tqdm(range(self.n_epoch), disable=not self.accelerator.is_main_process, leave=False, position=0)
        logging.info("Begin training")
        for epoch in tq_epoch:
            diffuser.train()
            tq_epoch.set_description(f"Epoch {epoch + 1}/{self.n_epoch}")

            tq_batch = tqdm(train_dataloader, disable=not self.accelerator.is_main_process)  # type: ignore
            for batch in tq_batch:
                with self.accelerator.accumulate(diffuser.denoiser):  # type: ignore
                    self.training_step(
                        diffuser=diffuser,
                        optimizer=optimizer,  # type: ignore
                        batch=batch,
                        tracker=tracker,
                        p_classifier_free_guidance=p_classifier_free_guidance,
                        scheduler=scheduler,  # type: ignore
                        per_batch_scheduler=per_batch_scheduler,
                        ema_denoiser=ema_denoiser,
                    )
                    tq_batch.set_description(f"Loss: {tracker.avg['loss'] :.4f}")

            gathered_loss: list[Tensor] = self.accelerator.gather(  # type: ignore
                torch.tensor(tracker.avg["loss"], device=self.accelerator.device)
            )
            self.accelerator.log({"train/loss": gathered_loss.mean().item()}, step=epoch)  # type: ignore
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
                    self.validation_step(
                        diffuser=diffuser,
                        val_batch=val_batch,
                        tracker=tracker,
                        ema_eval=ema_eval,  # type: ignore
                    )
                    tq_val_batch.set_description(f"Val Loss: {tracker.avg['val_loss'] :.4f}")
                gathered_val_loss: Tensor = self.accelerator.gather(  # type: ignore
                    torch.tensor(tracker.avg["val_loss"], device=self.accelerator.device)  # type: ignore
                )
                self.accelerator.log({"val/loss": gathered_val_loss.mean().item()}, step=epoch)  # type: ignore
                if gathered_val_loss.mean().item() < best_val_loss:  # type: ignore
                    best_val_loss = gathered_val_loss
                    self.save_model(optimizer, ema_denoiser, scheduler)  # type: ignore
                tracker.reset()

                if log_validation_images:
                    logging.info("creating validation images")
                    if self.accelerator.is_main_process:
                        self.log_images(diffuser, val_dataloader, epoch, ema_eval, val_steps)  # type: ignore

            self.accelerator.wait_for_everyone()

        self.accelerator.end_training()
        logging.info("Training complete")

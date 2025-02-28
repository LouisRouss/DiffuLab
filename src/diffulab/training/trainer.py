import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import torch
import wandb
from accelerate import Accelerator  # type: ignore [stub file not found]
from ema_pytorch import EMA  # type: ignore [stub file not found]
from torch import Tensor
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from tqdm import tqdm

from diffulab.diffuse.diffuser import Diffuser
from diffulab.networks.denoisers.common import Denoiser, ModelInput
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
        assert (HOME_PATH / ".cache" / "huggingface" / "accelerate" / "default_config.yaml").exists(), (
            "please run `accelerate config` first in the CLI and save the config at the default location"
        )
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
        os.environ["WANDB_DIR"] = str(self.save_path / "wandb")
        Path(os.environ["WANDB_DIR"]).mkdir(parents=True, exist_ok=True)
        self.accelerator.init_trackers(project_name=project_name, config=run_config, init_kwargs=init_kwargs)  # type: ignore

    def training_step(
        self,
        diffuser: Diffuser,
        optimizer: Optimizer,
        batch: ModelInput,
        tracker: AverageMeter,
        p_classifier_free_guidance: float = 0,
        scheduler: LRScheduler | None = None,
        per_batch_scheduler: bool = False,
        ema_denoiser: EMA | None = None,
    ) -> None:
        optimizer.zero_grad()
        timesteps = diffuser.draw_timesteps(batch["x"].shape[0]).to(self.accelerator.device)
        batch.update({"p": p_classifier_free_guidance})
        loss = diffuser.compute_loss(model_inputs=batch, timesteps=timesteps)
        tracker.update(loss.item(), key="loss")
        self.accelerator.backward(loss)  # type: ignore
        optimizer.step()
        if scheduler is not None and per_batch_scheduler:
            scheduler.step()
        if ema_denoiser is not None:
            ema_denoiser.update()

    def move_dict_to_device(self, batch: ModelInput) -> ModelInput:
        return ModelInput(**{
            k: v.to(self.accelerator.device) if isinstance(v, Tensor) else v
            for k, v in batch.items()  # type: ignore
        })

    @torch.no_grad()  # type: ignore
    def validation_step(
        self,
        diffuser: Diffuser,
        val_batch: ModelInput,
        tracker: AverageMeter,
        ema_eval: Denoiser | None = None,
    ) -> None:
        timesteps = diffuser.draw_timesteps(val_batch["x"].shape[0]).to(self.accelerator.device)
        val_batch = self.move_dict_to_device(val_batch)
        val_loss = (
            diffuser.compute_loss(model_inputs=val_batch, timesteps=timesteps)
            if not self.use_ema
            else diffuser.diffusion.compute_loss(model=ema_eval, model_inputs=val_batch, timesteps=timesteps)  # type: ignore
        )
        tracker.update(val_loss.item(), key="val_loss")

    def save_model(
        self,
        optimizer: Optimizer,
        diffuser: Diffuser,
        ema_denoiser: Denoiser | None = None,
        scheduler: LRScheduler | None = None,
    ) -> None:
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
        diffuser: Diffuser,
        val_dataloader: Iterable[dict[str, Any]],
        epoch: int,
        ema_eval: Diffuser | None = None,
        val_steps: int = 25,
    ) -> None:
        diffuser.eval()
        original_steps = diffuser.n_steps
        diffuser.set_steps(val_steps)
        batch: dict[str, Any] = next(iter(val_dataloader))  # type: ignore
        images = (
            diffuser.generate(data_shape=batch["x"].shape, model_inputs=batch)  # type: ignore
            if not self.use_ema
            else diffuser.diffusion.denoise(model=ema_eval, data_shape=batch["x"].shape, model_inputs=batch)  # type: ignore
        )
        images = wandb.Image(images, caption="Validation Images")
        self.accelerator.log({"val/images": images}, step=epoch)  # type: ignore
        diffuser.set_steps(original_steps)

    def train(
        self,
        diffuser: Diffuser,
        optimizer: Optimizer,
        train_dataloader: Iterable[ModelInput],
        val_dataloader: Iterable[ModelInput] | None = None,
        scheduler: LRScheduler | None = None,
        per_batch_scheduler: bool = False,
        log_validation_images: bool = False,
        train_embedder: bool = False,
        p_classifier_free_guidance: float = 0,
        val_steps: int = 25,
        ema_ckpt: str | None = None,
        epoch_start: int = 0,
    ):
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
                        tq_batch.set_description(f"Loss: {tracker.avg['loss']:.4f}")

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
                    with self.accelerator.autocast():
                        self.validation_step(
                            diffuser=diffuser,
                            val_batch=val_batch,
                            tracker=tracker,
                            ema_eval=ema_eval,  # type: ignore
                        )
                    tq_val_batch.set_description(f"Val Loss: {tracker.avg['val_loss']:.4f}")
                gathered_val_loss: Tensor = self.accelerator.gather(  # type: ignore
                    torch.tensor(tracker.avg["val_loss"], device=self.accelerator.device)  # type: ignore
                )
                self.accelerator.log({"val/loss": gathered_val_loss.mean().item()}, step=epoch)  # type: ignore
                if gathered_val_loss.mean().item() < best_val_loss:  # type: ignore
                    best_val_loss = gathered_val_loss
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

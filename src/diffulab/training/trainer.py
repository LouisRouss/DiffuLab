import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import torch
from accelerate import Accelerator  # type: ignore
from torch import Tensor
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from tqdm import tqdm

from diffulab.diffuse.flow import Diffuser
from diffulab.networks.common import Denoiser
from diffulab.training.utils import AverageMeter

HOME_PATH = Path.home()


class Trainer:
    def __init__(
        self,
        n_epoch: int,
        batch_size: int,
        learning_rate: float,
        gradient_accumulation_step: int = 1,
        precision_type: str = "no",
        project_name: str = "my_project",
        run_config: dict[str, Any] | None = None,
        init_kwargs: dict[str, Any] | None = None,
    ):
        assert (
            HOME_PATH / ".cache" / "huggingface" / "accelerate" / "default_config.yaml"
        ).exists(), "please run `accelerate config` first in the CLI and save the config at the default location"
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.accelerator = Accelerator(
            split_batches=True,
            mixed_precision=precision_type,
            gradient_accumulation_steps=gradient_accumulation_step,
            log_with="wandb",
        )
        self.accelerator.init_trackers(project_name=project_name, config=run_config, init_kwargs=init_kwargs)  # type: ignore

    def train(
        self,
        diffuser: Diffuser,
        train_dataloader: Iterable[dict[str, Any]],
        optimizer: Optimizer,
        save_path: str | Path = Path.home() / "experiments" / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        val_dataloader: Iterable[dict[str, Any]] | None = None,
        scheduler: LRScheduler | None = None,
        per_batch_scheduler: bool = False,
        log_validation_images: bool = False,
        train_embedder: bool = False,
        p_classifier_free_guidance: float = 0,
    ):
        Path(save_path).mkdir(parents=True, exist_ok=True)
        diffuser.denoiser, train_dataloader, val_dataloader, optimizer = self.accelerator.prepare(  # type: ignore
            (diffuser.denoiser, train_dataloader, val_dataloader, optimizer)
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
                    optimizer.zero_grad()  # type: ignore
                    timesteps = diffuser.draw_timesteps(self.batch_size)
                    batch.update({"p": p_classifier_free_guidance})
                    loss = diffuser.compute_loss(model_inputs=batch, timesteps=timesteps)
                    tracker.update(loss.item(), key="loss")
                    self.accelerator.backward(loss)  # type: ignore
                    optimizer.step()  # type: ignore
                    if scheduler is not None and per_batch_scheduler:
                        scheduler.step()  # type: ignore
                    tq_batch.set_description(f"Loss: {tracker.avg['loss'] :.4f}")

            gathered_loss: list[Tensor] = self.accelerator.gather(  # type: ignore
                torch.tensor(tracker.avg["loss"], device=self.accelerator.device)
            )
            self.accelerator.log({"train/loss": gathered_loss.mean().item()}, step=epoch)  # type: ignore
            tracker.reset()

            if val_dataloader is not None:
                diffuser.eval()  # type: ignore
                tq_val_batch = tqdm(
                    val_dataloader,  # type: ignore
                    disable=not self.accelerator.is_main_process,
                    leave=False,
                    position=1,
                )
                for val_batch in tq_val_batch:
                    with torch.no_grad():  # type: ignore
                        timesteps = diffuser.draw_timesteps(self.batch_size)
                        val_loss = diffuser.compute_loss(model_inputs=val_batch, timesteps=timesteps)
                        tracker.update(val_loss.item(), key="val_loss")
                        tq_val_batch.set_description(f"Val Loss: {tracker.avg['val_loss'] :.4f}")
                gathered_val_loss: Tensor = self.accelerator.gather(  # type: ignore
                    torch.tensor(tracker.avg["val_loss"], device=self.accelerator.device)  # type: ignore
                )
                self.accelerator.log({"val/loss": gathered_val_loss.mean().item()}, step=epoch)  # type: ignore
                if gathered_val_loss.mean().item() < best_val_loss:
                    best_val_loss = gathered_val_loss
                    unwrapped_denoiser: Denoiser = self.accelerator.unwrap_model(diffuser.denoiser)  # type: ignore
                    self.accelerator.save(unwrapped_denoiser.state_dict(), save_path / "denoiser.pth")  # type: ignore
                    self.accelerator.save(optimizer.optimizer.state_dict(), save_path / "optimizer.pth")  # type: ignore
                    if scheduler is not None:
                        self.accelerator.save(scheduler.scheduler.state_dict(), save_path / "scheduler.pth")  # type: ignore

                tracker.reset()

            self.accelerator.wait_for_everyone()

            if log_validation_images:
                logging.info("creating validation images")
                if self.accelerator.is_main_process:
                    diffuser.eval()
                    (Path(save_path) / "images").mkdir(exist_ok=True)
                    wandb_tracker = self.accelerator.get_tracker("wandb")  # type: ignore
                    batch: dict[str, Any] = next(iter(val_dataloader))  # type: ignore
                    images = diffuser.generate(data_shape=batch["x"], model_inputs=batch)  # type: ignore
                    images = wandb_tracker.Image(images, caption="Validation Images")  # type: ignore
                    self.accelerator.log({"val/images": images}, step=epoch)  # type: ignore
        logging.info("Training complete")

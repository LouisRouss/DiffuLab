import math

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from diffulab.datasets.imagenet import ImageNetLatent
from diffulab.networks.disc import RAEDiscriminator
from diffulab.networks.vision_towers.rae import RAE
from diffulab.training.trainers.extra.rae_trainer import RAETrainer


def cosine_with_warmup_and_min_lr_lambda(
    current_step: int, num_warmup_steps: int, num_training_steps: int, min_lr_factor: float = 0.1
) -> float:
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = (current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    # interpolate between 1 and min_lr_factor
    return min_lr_factor + (1 - min_lr_factor) * cosine


def get_cosine_schedule_with_warmup_and_min_lr(
    optimizer: torch.optim.Optimizer, num_warmup_steps: int, num_training_steps: int, min_lr_factor: float = 0.1
) -> LambdaLR:
    lr_lambda = lambda step: cosine_with_warmup_and_min_lr_lambda(  # type: ignore
        step,
        num_warmup_steps,
        num_training_steps,
        min_lr_factor,  # type: ignore
    )
    return LambdaLR(optimizer, lr_lambda)  # type: ignore


@hydra.main(version_base=None, config_path="../configs", config_name="train_imagenet_rae_decoder")
def train(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    train_dataset: ImageNetLatent = instantiate(cfg.dataset.train)
    val_dataset: ImageNetLatent = instantiate(cfg.dataset.val)

    train_dataset.set_latent_scale(1)
    val_dataset.set_latent_scale(1)

    dl_cfg = cfg.get("dataloader", {})
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=dl_cfg.get("batch_size", 32),
        shuffle=dl_cfg.get("shuffle", True),
        num_workers=dl_cfg.get("num_workers", 0),
        pin_memory=dl_cfg.get("pin_memory", False),
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=dl_cfg.get("batch_size", 32),
        shuffle=dl_cfg.get("shuffle", False),
        num_workers=dl_cfg.get("num_workers", 0),
        pin_memory=dl_cfg.get("pin_memory", False),
    )

    rae: RAE = instantiate(cfg.vision_tower)
    discriminator: RAEDiscriminator = instantiate(cfg.discriminator)

    def count_parameters(model: torch.nn.Module) -> int:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Number of trainable parameters in the rae decoder: {count_parameters(rae.decoder):,}")
    print(f"Number of trainable parameters in the discriminator: {count_parameters(discriminator):,}")

    rae_optimizer = instantiate(cfg.optimizer.rae_decoder, params=rae.decoder.parameters())
    disc_optimizer = instantiate(cfg.optimizer.discriminator, params=discriminator.parameters())

    rae_scheduler = get_cosine_schedule_with_warmup_and_min_lr(
        optimizer=rae_optimizer,
        num_warmup_steps=len(train_loader),
        num_training_steps=cfg.trainer.n_epoch * len(train_loader),
    )
    disc_scheduler = get_cosine_schedule_with_warmup_and_min_lr(
        optimizer=disc_optimizer,
        num_warmup_steps=len(train_loader),
        num_training_steps=(cfg.trainer.n_epoch - cfg.trainer.disc_epoch_start) * len(train_loader),
    )

    rae_trainer = RAETrainer(
        n_epoch=cfg.trainer.n_epoch,
        gradient_accumulation_step=cfg.trainer.gradient_accumulation_step,
        precision_type=cfg.trainer.precision_type,
        project_name=cfg.trainer.project_name,
        use_ema=cfg.trainer.use_ema,
        ema_update_after_step=cfg.trainer.get("ema_update_after_step", 0),
        ema_update_every=cfg.trainer.get("ema_update_every", 10),
        run_config=OmegaConf.to_container(cfg, resolve=True),  # type: ignore[reportArgumentType]
        compile=cfg.trainer.get("compile", False),
        init_kwargs={
            "wandb": cfg.trainer.get("wandb", {}),
        },
    )

    rae_trainer.train(
        rae=rae,
        disc=discriminator,
        rae_optimizer=rae_optimizer,
        disc_optimizer=disc_optimizer,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        rae_scheduler=rae_scheduler,
        disc_scheduler=disc_scheduler,
        per_batch_scheduler=cfg.trainer.get("per_batch_scheduler", True),
        log_validation_images=cfg.trainer.get("log_validation_images", True),
        disc_epoch_start=cfg.trainer.get("disc_epoch_start", 6),
        gan_epoch_start=cfg.trainer.get("gan_epoch_start", 8),
        lpips_epoch_start=cfg.trainer.get("lpips_epoch_start", 1),
        lambda_lpips=cfg.trainer.get("lambda_lpips", 1.0),
        lambda_gan=cfg.trainer.get("lambda_gan", 0.75),
        use_adaptive_weight_loss=cfg.trainer.get("use_adaptive_weight_loss", True),
    )


if __name__ == "__main__":
    train()

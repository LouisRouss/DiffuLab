import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader

from diffulab.diffuse import Diffuser
from diffulab.training import Trainer


@hydra.main(version_base=None, config_path="../configs", config_name="train_mnist_flow_matching")
def train(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    train_dataset = instantiate(cfg.dataset.train)
    val_dataset = instantiate(cfg.dataset.val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.dataloader.batch_size,
        shuffle=True,
        num_workers=cfg.dataloader.num_workers
        if hasattr(cfg, "dataloader") and hasattr(cfg.dataloader, "num_workers")
        else 0,
        pin_memory=cfg.dataloader.pin_memory
        if hasattr(cfg, "dataloader") and hasattr(cfg.dataloader, "pin_memory")
        else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.dataloader.batch_size,
        shuffle=False,
        num_workers=cfg.dataloader.num_workers
        if hasattr(cfg, "dataloader") and hasattr(cfg.dataloader, "num_workers")
        else 0,
        pin_memory=cfg.dataloader.pin_memory
        if hasattr(cfg, "dataloader") and hasattr(cfg.dataloader, "pin_memory")
        else False,
    )

    denoiser = instantiate(cfg.model)

    def count_parameters(model: torch.nn.Module) -> int:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Number of trainable parameters: {count_parameters(denoiser):,}")

    diffuser = Diffuser(
        denoiser=denoiser,
        model_type=cfg.diffuser.model_type,
        n_steps=cfg.diffuser.n_steps,
        sampling_method=cfg.diffuser.sampling_method,
        extra_args=cfg.diffuser.extra_args if hasattr(cfg.diffuser, "extra_args") else {},
    )

    optimizer = instantiate(
        cfg.optimizer,
        params=denoiser.parameters(),
    )

    # TODO: add a run name for wandb
    trainer = Trainer(
        n_epoch=cfg.trainer.n_epoch,
        gradient_accumulation_step=cfg.trainer.gradient_accumulation_step,
        precision_type=cfg.trainer.precision_type,
        project_name=cfg.trainer.project_name,
        use_ema=cfg.trainer.use_ema,
        ema_update_after_step=cfg.trainer.ema_update_after_step if hasattr(cfg.trainer, "ema_update_after_step") else 0,
        ema_update_every=cfg.trainer.ema_update_every if hasattr(cfg.trainer, "ema_update_every") else 10,
    )

    trainer.train(
        diffuser=diffuser,
        optimizer=optimizer,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        log_validation_images=cfg.trainer.log_validation_images,
        val_steps=cfg.trainer.val_steps if hasattr(cfg.trainer, "val_steps") else 50,
    )


if __name__ == "__main__":
    train()

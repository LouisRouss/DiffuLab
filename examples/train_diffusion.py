import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from diffulab.diffuse import Diffuser
from diffulab.training import Trainer


@hydra.main(version_base=None, config_path="../configs", config_name="train_mnist_flow_matching")
def train(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    train_dataset = instantiate(cfg.dataset.train)
    val_dataset = instantiate(cfg.dataset.val)

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

    denoiser = instantiate(cfg.model)

    def count_parameters(model: torch.nn.Module) -> int:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Number of trainable parameters: {count_parameters(denoiser):,}")

    diffuser = Diffuser(
        denoiser=denoiser,
        model_type=cfg.diffuser.model_type,
        n_steps=cfg.diffuser.n_steps,
        sampling_method=cfg.diffuser.sampling_method,
        extra_args=cfg.diffuser.get("extra_args", {}),
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
        ema_update_after_step=cfg.trainer.get("ema_update_after_step", 0),
        ema_update_every=cfg.trainer.get("ema_update_every", 10),
        run_config=OmegaConf.to_container(cfg, resolve=True),  # type: ignore[reportArgumentType]
    )

    trainer.train(
        diffuser=diffuser,
        optimizer=optimizer,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        log_validation_images=cfg.trainer.log_validation_images,
        val_steps=cfg.trainer.get("val_steps", 50),
    )


if __name__ == "__main__":
    train()

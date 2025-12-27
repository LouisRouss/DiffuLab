import hydra
import torch

torch._dynamo.config.cache_size_limit = 64  # type: ignore
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from diffulab.datasets.imagenet import MultiARBatchSampler, collate_fn
from diffulab.diffuse import Diffuser
from diffulab.training import BaseTrainer
from diffulab.training.losses.repa import RepaLoss


@hydra.main(version_base=None, config_path="../configs", config_name="train_imagenet_repa_txt_to_img")
def train(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # Dataset
    train_dataset = instantiate(cfg.dataset.train)
    val_dataset = instantiate(cfg.dataset.val)

    # Model
    embedder = instantiate(cfg.embedder)
    denoiser = instantiate(cfg.model, context_embedder=embedder)

    def count_parameters(model: torch.nn.Module) -> int:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Number of trainable parameters: {count_parameters(denoiser):,}")

    # Repa Specific parameters
    repa_loss = RepaLoss(
        denoiser_dimension=cfg.model.get("input_dim"),
        embedding_dim=384,
        load_dino=False,
        use_resampler=cfg.perceiver_resampler.get("use_resampler", False),
        resampler_params=cfg.perceiver_resampler.get("parameters", {}),
        coeff=0.5,
    )
    vision_tower = instantiate(cfg.vision_tower)

    train_dataset.set_latent_scale(vision_tower.latent_scale)
    train_dataset.set_latent_bias(vision_tower.latent_bias)
    val_dataset.set_latent_scale(vision_tower.latent_scale)
    val_dataset.set_latent_bias(vision_tower.latent_bias)

    dl_cfg = cfg.get("dataloader", {})
    train_sampler = MultiARBatchSampler(
        dataset=train_dataset, batch_size=dl_cfg.get("batch_size", 32), shuffle=True, drop_last=True
    )
    val_sampler = MultiARBatchSampler(
        dataset=val_dataset, batch_size=dl_cfg.get("batch_size", 32), shuffle=False, drop_last=False
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_sampler=val_sampler,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # Diffuser
    diffuser = Diffuser(
        denoiser=denoiser,
        model_type=cfg.diffuser.model_type,
        n_steps=cfg.diffuser.n_steps,
        sampling_method=cfg.diffuser.sampling_method,
        vision_tower=vision_tower,
        extra_args=cfg.diffuser.get("extra_args", {}),
        extra_losses=[repa_loss],
    )

    optimizer = instantiate(
        cfg.optimizer,
        params=list(denoiser.parameters()) + list(repa_loss.proj.parameters()),  # type: ignore
    )

    trainer = BaseTrainer(
        n_epoch=cfg.trainer.n_epoch,
        gradient_accumulation_step=cfg.trainer.gradient_accumulation_step,
        precision_type=cfg.trainer.precision_type,
        project_name=cfg.trainer.project_name,
        use_ema=cfg.trainer.use_ema,
        ema_rate=cfg.trainer.get("ema_rate", 0.9999),
        ema_update_after_step=cfg.trainer.get("ema_update_after_step", 0),
        ema_update_every=cfg.trainer.get("ema_update_every", 10),
        run_config=OmegaConf.to_container(cfg, resolve=True),  # type: ignore[reportArgumentType]
        compile=cfg.trainer.get("compile", False),
        init_kwargs={
            "wandb": cfg.trainer.get("wandb", {}),
        },
    )

    trainer.train(
        diffuser=diffuser,
        optimizer=optimizer,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        log_validation_images=cfg.trainer.log_validation_images,
        val_steps=cfg.trainer.get("val_steps", 50),
        val_step_shift=cfg.trainer.get("val_step_shift", None),
        p_classifier_free_guidance=cfg.trainer.get("p_classifier_free_guidance", 0),
    )


if __name__ == "__main__":
    train()

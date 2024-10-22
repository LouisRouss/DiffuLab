from pathlib import Path

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import pil_to_tensor  # type: ignore

from diffulab.diffuse.flow import Diffuser
from diffulab.networks.denoisers.unet import UNetModel
from diffulab.training.trainer import Trainer

BATCH_SIZE = 16
EPOCHS = 1000
LR = 1e-4


class ImageNet64(Dataset[dict[str, Tensor]]):
    def __init__(self, data_path: str, wnids_path: str):
        super().__init__()
        self.images = list(Path(data_path).rglob("*.JPEG"))
        with Path(wnids_path).open("r") as f:
            wnids = f.readlines()
        self.class_matching = {wnid.strip(): idx for idx, wnid in enumerate(wnids)}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int):
        image_path = self.images[idx]
        image = pil_to_tensor(Image.open(image_path).convert("RGB"))
        image = image / 255.0
        label = self.class_matching[image_path.parts[-3]]
        label = torch.tensor([label]).long()
        return {"x": image, "y": label}


def train():
    path_train = "/home/louis/datasets/imagenet64/train"
    path_val = "/home/louis/datasets/imagenet64/val"
    wnids_path = "/home/louis/datasets/imagenet64/wnids.txt"
    train_dataset = ImageNet64(path_train, wnids_path)
    val_dataset = ImageNet64(path_val, wnids_path)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    denoiser = UNetModel(
        image_size=[64, 64],
        in_channels=3,
        model_channels=192,
        out_channels=3,
        num_res_blocks=2,
        attention_resolutions=[32, 16, 8],
        num_heads=3,
        resblock_updown=True,
        n_classes=200,
        classifier_free=False,
    )

    diffuser = Diffuser(denoiser, model_type="rectified_flow", n_steps=1000, sampling_method="euler")
    optimizer = torch.optim.AdamW(denoiser.parameters(), lr=LR)  # type: ignore

    trainer = Trainer(
        n_epoch=EPOCHS,
        batch_size=BATCH_SIZE,
        gradient_accumulation_step=1,
        precision_type="no",
        project_name="imagenet64",
        use_ema=True,
        ema_update_after_step=len(train_loader),
        ema_update_every=1,
    )

    trainer.train(
        diffuser,
        optimizer,  # type: ignore
        train_loader,
        val_loader,
        log_validation_images=True,
    )


if __name__ == "__main__":
    train()

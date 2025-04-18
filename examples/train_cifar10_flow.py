import pickle
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from diffulab.diffuse import Diffuser
from diffulab.networks import Denoiser, MMDiT
from diffulab.training import Trainer

BATCH_SIZE = 64
EPOCHS = 25
LR = 1e-4


class Cifar10(Dataset[dict[str, Tensor]]):
    def __init__(
        self,
        data_path: str,
        batches_to_load: list[str] = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"],
    ):
        super().__init__()
        images: list[NDArray[np.uint8]] = []
        labels: list[NDArray[np.int64]] = []
        for batch in batches_to_load:
            images_batch, labels_batch = self.load_cifar10_data(Path(data_path) / batch)
            images.append(images_batch)
            labels.append(labels_batch)
        self.images: NDArray[np.uint8] = np.concatenate(images, axis=0)
        self.labels: NDArray[np.int64] = np.concatenate(labels, axis=0)

    def load_cifar10_data(self, file: Path) -> tuple[NDArray[np.uint8], NDArray[np.int64]]:
        with open(file, "rb") as f:
            batch = pickle.load(f, encoding="latin1")
        features = batch["data"]
        r = features[:, :1024].reshape(-1, 32, 32)
        g = features[:, 1024:2048].reshape(-1, 32, 32)
        b = features[:, 2048:].reshape(-1, 32, 32)
        images = np.stack([r, g, b], axis=-1, dtype=np.uint8)
        labels = np.array(batch["labels"], dtype=np.int64)
        return images, labels

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, idx: int):
        image = self.images[idx].astype(np.float32) / 255.0
        image = image.transpose(2, 0, 1)
        image = (image - 0.5) / 0.5
        label = self.labels[idx]
        return {"x": image, "y": label}


def train():
    data_path = "/home/louis/datasets/cifar-10-batches-py"
    batches_to_load_train = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4"]
    batches_to_load_val = ["data_batch_5"]

    train_dataset = Cifar10(data_path, batches_to_load_train)
    val_dataset = Cifar10(data_path, batches_to_load_val)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    denoiser = MMDiT(
        simple_dit=True,
        input_channels=3,
        output_channels=3,
        input_dim=512,
        hidden_dim=1024,
        embedding_dim=512,
        num_heads=4,
        mlp_ratio=2,
        patch_size=4,
        depth=6,
        n_classes=10,
        classifier_free=True,
    )

    # Print number of trainable parameters
    def count_parameters(model: Denoiser) -> int:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Number of trainable parameters: {count_parameters(denoiser):,}")

    diffuser = Diffuser(denoiser, model_type="rectified_flow", n_steps=50, sampling_method="euler")
    optimizer = torch.optim.AdamW(denoiser.parameters(), lr=LR)

    trainer = Trainer(
        n_epoch=EPOCHS,
        batch_size=BATCH_SIZE,
        gradient_accumulation_step=1,
        precision_type="no",
        project_name="cifar10",
        use_ema=True,
        ema_update_after_step=len(train_loader),
        ema_update_every=1,
    )

    trainer.train(diffuser, optimizer, train_loader, val_loader, log_validation_images=True, val_steps=50)


if __name__ == "__main__":
    train()

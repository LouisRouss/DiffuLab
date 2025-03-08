import struct
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from diffulab.diffuse.diffuser import Diffuser
from diffulab.networks.denoisers.common import Denoiser
from diffulab.networks.denoisers.unet import UNetModel
from diffulab.training.trainer import Trainer

BATCH_SIZE = 256
EPOCHS = 100
LR = 1e-4


class MNISTDataset(Dataset[dict[str, Tensor]]):
    def __init__(self, data_path: str, train: bool = True):
        super().__init__()
        self.data_path = Path(data_path)
        if train:
            images_file = self.data_path / "train-images-idx3-ubyte"
            labels_file = self.data_path / "train-labels-idx1-ubyte"
        else:
            images_file = self.data_path / "t10k-images-idx3-ubyte"
            labels_file = self.data_path / "t10k-labels-idx1-ubyte"

        self.images = self._load_images(images_file)
        self.labels = self._load_labels(labels_file)

    def _load_images(self, file: Path) -> NDArray[np.float32]:
        with open(file, "rb") as f:
            _, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
            images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, 1, rows, cols)

        # Resize images to 32x32 while preserving the channel dimension
        resized_images = np.zeros((num_images, 1, 32, 32), dtype=np.float32)
        for i in range(num_images):
            # Center the 28x28 image in the 32x32 frame with padding
            resized_images[i, 0, 2:30, 2:30] = images[i, 0]

        return ((resized_images.astype(np.float32) / 255.0) - 0.5) / 0.5

    def _load_labels(self, file: Path) -> NDArray[np.int64]:
        with open(file, "rb") as f:
            _, _ = struct.unpack(">II", f.read(8))
            labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels.astype(np.int64)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        image = torch.tensor(self.images[idx])
        label = torch.tensor(self.labels[idx])
        return {"x": image, "y": label}


def train():
    data_path = "/home/louis/datasets/mnist"

    train_dataset = MNISTDataset(data_path, True)
    val_dataset = MNISTDataset(data_path, False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    denoiser = UNetModel(
        image_size=[32, 32],
        in_channels=1,
        model_channels=128,
        out_channels=1,
        num_res_blocks=2,
        attention_resolutions=[4, 8, 16],
        num_heads=2,
        resblock_updown=True,
        n_classes=10,
        use_scale_shift_norm=True,
        classifier_free=False,
    )

    # Print number of trainable parameters
    def count_parameters(model: Denoiser) -> int:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Number of trainable parameters: {count_parameters(denoiser):,}")

    diffuser = Diffuser(
        denoiser, model_type="rectified_flow", n_steps=50, sampling_method="euler", extra_args={"logits_normal": True}
    )
    optimizer = torch.optim.AdamW(denoiser.parameters(), lr=LR)

    trainer = Trainer(
        n_epoch=EPOCHS,
        batch_size=BATCH_SIZE,
        gradient_accumulation_step=1,
        precision_type="no",
        project_name="mnist",
        use_ema=False,
    )

    trainer.train(
        diffuser,
        optimizer,
        train_loader,
        val_loader,
        log_validation_images=True,
    )


if __name__ == "__main__":
    train()

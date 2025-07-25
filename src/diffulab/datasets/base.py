from abc import ABC, abstractmethod
from typing import NotRequired, Required, TypedDict

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data import Dataset

from diffulab.networks.denoisers.common import ModelInput


class BatchData(TypedDict, total=False):
    model_inputs: Required[ModelInput]
    extra: NotRequired[dict[str, Tensor | None]]


class DiffusionDataset(Dataset[BatchData], ABC):
    """Base class for datasets used in diffusion models.

    This abstract class defines the common interface that all diffusion datasets
    should implement, ensuring consistency across different data sources.
    """

    def __init__(self):
        super().__init__()
        self.images = None
        self.labels = None

    @abstractmethod
    def load_data(self) -> tuple[NDArray[np.uint8 | np.float32], NDArray[np.int64]]:
        """Load and preprocess the dataset images and labels.

        Returns:
            Tuple containing (images, labels) arrays
        """
        pass

    @abstractmethod
    def preprocess_image(self, image: NDArray[np.uint8 | np.float32]) -> NDArray[np.float32]:
        """Preprocess a single image according to the dataset's requirements.

        Args:
            image: The raw image data

        Returns:
            Preprocessed image data
        """
        pass

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        if self.images is None:
            raise ValueError("Dataset has not been initialized properly. Images are None.")
        return len(self.images)

    def __getitem__(self, idx: int) -> BatchData:
        """Get a sample from the dataset.

        Args:
            idx: Index of the sample to get

        Returns:
            Dictionary containing 'x' (image tensor) and 'y' (label tensor)
        """
        if self.images is None or self.labels is None:
            raise ValueError("Dataset has not been initialized properly. Images or labels are None.")

        image = self.preprocess_image(self.images[idx])
        label = self.labels[idx]

        return {"model_inputs": {"x": torch.tensor(image), "y": torch.tensor(label, dtype=torch.long)}}

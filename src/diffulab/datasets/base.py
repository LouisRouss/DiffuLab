from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import torch
from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data import Dataset


class DiffusionDataset(Dataset[Dict[str, Tensor]], ABC):
    """Base class for datasets used in diffusion models.

    This abstract class defines the common interface that all diffusion datasets
    should implement, ensuring consistency across different data sources.
    """

    def __init__(self):
        super().__init__()
        self.images = None
        self.labels = None

    @abstractmethod
    def load_data(self) -> Tuple[NDArray[Any], NDArray[Any]]:
        """Load and preprocess the dataset images and labels.

        Returns:
            Tuple containing (images, labels) arrays
        """
        pass

    @abstractmethod
    def preprocess_image(self, image: NDArray[Any]) -> NDArray[Any]:
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

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
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

        return {"x": torch.tensor(image), "y": torch.tensor(label, dtype=torch.long)}

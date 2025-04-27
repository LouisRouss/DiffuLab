import struct
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .base import DiffusionDataset


class MNISTDataset(DiffusionDataset):
    """MNIST dataset for diffusion models."""
    
    def __init__(self, data_path: str, train: bool = True):
        """Initialize the MNIST dataset.
        
        Args:
            data_path: Path to the MNIST data directory
            train: Whether to load the training set (True) or test set (False)
        """
        super().__init__()
        self.data_path = Path(data_path)
        self.train = train
        self.images, self.labels = self.load_data()
    
    def load_data(self) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
        """Load MNIST data from files."""
        if self.train:
            images_file = self.data_path / "train-images-idx3-ubyte"
            labels_file = self.data_path / "train-labels-idx1-ubyte"
        else:
            images_file = self.data_path / "t10k-images-idx3-ubyte"
            labels_file = self.data_path / "t10k-labels-idx1-ubyte"
            
        images = self._load_images(images_file)
        labels = self._load_labels(labels_file)
        
        return images, labels
    
    def _load_images(self, file: Path) -> NDArray[np.float32]:
        """Load and preprocess MNIST images."""
        with open(file, "rb") as f:
            _, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
            images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, 1, rows, cols)

        # Resize images to 32x32 while preserving the channel dimension
        resized_images = np.zeros((num_images, 1, 32, 32), dtype=np.float32)
        for i in range(num_images):
            # Center the 28x28 image in the 32x32 frame with padding
            resized_images[i, 0, 2:30, 2:30] = images[i, 0]

        return resized_images
    
    def _load_labels(self, file: Path) -> NDArray[np.int64]:
        """Load MNIST labels."""
        with open(file, "rb") as f:
            _, _ = struct.unpack(">II", f.read(8))
            labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels.astype(np.int64)
    
    def preprocess_image(self, image: NDArray) -> NDArray:
        """Normalize the image to [-1, 1] range."""
        return ((image.astype(np.float32) / 255.0) - 0.5) / 0.5 
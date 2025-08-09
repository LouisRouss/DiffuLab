"""Network-specific fixtures for testing."""

from typing import Any

import pytest
import torch

from diffulab.networks.embedders.common import ContextEmbedder


class MockContextEmbedder(ContextEmbedder):
    """Mock context embedder for testing."""

    def __init__(self, pooled_dim: int = 256, sequence_dim: int = 512, sequence_len: int = 32):
        super().__init__()
        self.pooled_dim = pooled_dim
        self.sequence_dim = sequence_dim
        self.sequence_len = sequence_len

    @property
    def n_output(self) -> int:
        return 2

    @property
    def output_size(self) -> tuple[int, int]:
        return (self.pooled_dim, self.sequence_dim)

    def drop_conditions(self, context: Any, p: float) -> Any:
        return context

    def forward(self, context: Any = None, p: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = 4 if context is None else (context.shape[0] if hasattr(context, "shape") else 4)
        pooled = torch.randn(batch_size, self.pooled_dim)
        sequence = torch.randn(batch_size, self.sequence_len, self.sequence_dim)
        return pooled, sequence


@pytest.fixture(scope="module")
def mock_context_embedder():
    """Create a mock context embedder for testing."""
    return MockContextEmbedder()


@pytest.fixture(scope="module")
def input_dim():
    """Standard input dimension for testing."""
    return 512


@pytest.fixture(scope="module")
def hidden_dim():
    """Standard hidden dimension for testing."""
    return 512


@pytest.fixture(scope="module")
def embedding_dim():
    """Standard embedding dimension for testing."""
    return 256


@pytest.fixture(scope="module")
def context_dim():
    """Standard context dimension for testing."""
    return 512


@pytest.fixture(scope="module")
def num_heads():
    """Standard number of attention heads for testing."""
    return 8


@pytest.fixture(scope="module")
def mlp_ratio():
    """Standard MLP ratio for testing."""
    return 4


@pytest.fixture(scope="module")
def patch_size():
    """Standard patch size for testing."""
    return 16


@pytest.fixture(scope="module")
def input_channels():
    """Standard number of input channels for testing."""
    return 3


@pytest.fixture(scope="module")
def image_size():
    """Standard image size for testing."""
    return 256


@pytest.fixture(scope="module")
def sample_image(batch_size: int, input_channels: int, image_size: int) -> torch.Tensor:
    """Create a sample image tensor for testing."""
    return torch.randn(batch_size, input_channels, image_size, image_size)


@pytest.fixture(scope="module")
def sample_timesteps(batch_size: int) -> torch.Tensor:
    """Create sample timesteps for testing."""
    return torch.randn(batch_size)


@pytest.fixture
def sample_labels(batch_size: int) -> torch.Tensor:
    """Create sample class labels for testing."""
    return torch.randint(0, 10, (batch_size,))


@pytest.fixture(scope="module")
def device():
    """Return the device to use for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="module")
def batch_size():
    """Standard batch size for testing."""
    return 4


@pytest.fixture(scope="module")
def seq_len():
    """Standard sequence length for testing."""
    return 25


@pytest.fixture(scope="module")
def context_seq_len():
    """Standard context sequence length for testing."""
    return 32

"""Pytest configuration and fixtures for the entire test suite."""

import pytest
import torch


@pytest.fixture(autouse=True)
def set_random_seed():
    """Set random seed for reproducible testing."""
    torch.manual_seed(42)  # type: ignore
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    yield
    pass

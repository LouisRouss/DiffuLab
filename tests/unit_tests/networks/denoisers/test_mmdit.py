"""
Comprehensive unit tests for MMDiT module components.

This module provides exhaustive test coverage for all classes and functions in the MMDiT (Multimodal Diffusion Transformer)
implementation, including:
- DiTAttention: Self-attention mechanism with rotary positional embeddings
- MMDiTAttention: Multi-modal attention for input and context tensors
- DiTBlock: Complete transformer block with modulation for simple DiT
- MMDiTBlock: Multi-modal transformer block with separate input/context processing
- ModulatedLastLayer: Final output layer with adaptive layer normalization
- MMDiT: Main diffusion transformer model supporting both simple and multi-modal modes

The tests cover initialization, forward pass shapes, gradient flow, edge cases, and error conditions
to ensure robust and reliable behavior across different configurations and inputs.

Classes:
    TestDiTAttention: Tests for the DiTAttention self-attention mechanism.
    TestMMDiTAttention: Tests for the MMDiTAttention multi-modal attention mechanism.
    TestDiTBlock: Tests for the DiTBlock transformer block.
    TestMMDiTBlock: Tests for the MMDiTBlock multi-modal transformer block.
    TestModulatedLastLayer: Tests for the ModulatedLastLayer output layer.
    TestMMDiT: Tests for the main MMDiT model.
    TestMMDiTEdgeCases: Tests for edge cases and error conditions.
"""

from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from diffulab.networks.denoisers.common import ModelOutput
from diffulab.networks.denoisers.mmdit import (
    DiTAttention,
    DiTBlock,
    MMDiT,
    MMDiTAttention,
    MMDiTBlock,
    ModulatedLastLayer,
)
from diffulab.networks.embedders.common import ContextEmbedder


class MockContextEmbedder(ContextEmbedder):
    """Mock context embedder for testing."""

    def __init__(self, pooled_dim: int = 64, sequence_dim: int = 128, sequence_len: int = 32):
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
        pooled = torch.randn(context, self.pooled_dim)
        sequence = torch.randn(context, self.sequence_len, self.sequence_dim)
        return pooled, sequence


@pytest.fixture(scope="module")
def mock_context_embedder():
    """Create a mock context embedder for testing."""
    return MockContextEmbedder()


class TestDiTAttention:
    """
    Test class for DiTAttention module.

    This class provides comprehensive testing for the DiTAttention self-attention mechanism,
    which implements multi-head attention with rotary positional embeddings for diffusion transformers.

    Attributes:
        None

    Methods:
        dit_attention: Fixture that creates a DiTAttention instance for testing.
        test_init: Tests proper initialization of DiTAttention components.
        test_forward_shape: Tests output shape consistency in forward pass.
        test_forward_different_batch_sizes: Tests behavior with various batch sizes.
        test_forward_different_seq_lengths: Tests behavior with various sequence lengths.
        test_partial_rotary_factor: Tests rotary embedding dimension calculation.
        test_gradient_flow: Tests gradient propagation through the attention mechanism.
    """

    @pytest.fixture(scope="class")
    def input_dim(self):
        """Standard input dimension for testing."""
        return 128

    @pytest.fixture(scope="class")
    def hidden_dim(self):
        """Standard hidden dimension for testing."""
        return 128

    @pytest.fixture(scope="class")
    def num_heads(self):
        """Standard number of attention heads for testing."""
        return 4

    @pytest.fixture(scope="class")
    def batch_size(self):
        """Standard batch size for testing."""
        return 2

    @pytest.fixture(scope="class")
    def seq_len(self):
        """Standard sequence length for testing."""
        return 32

    @pytest.fixture
    def dit_attention(self, input_dim: int, hidden_dim: int, num_heads: int):
        """
        Create DiTAttention instance for testing.

        Args:
            input_dim (int): Dimension of the input features.
            hidden_dim (int): Hidden dimension for attention computation.
            num_heads (int): Number of attention heads.

        Returns:
            DiTAttention: Configured attention module for testing.
        """
        return DiTAttention(
            input_dim=input_dim,
            dim=hidden_dim,
            num_heads=num_heads,
            partial_rotary_factor=1.0,
            base=10000,
        )

    def test_init(self, dit_attention: DiTAttention, input_dim: int, hidden_dim: int, num_heads: int):
        """
        Test DiTAttention initialization.

        Verifies that all components are properly initialized with correct dimensions
        and that attributes match the expected values based on input parameters.

        Args:
            dit_attention (DiTAttention): The attention module to test.
            input_dim (int): Expected input dimension.
            hidden_dim (int): Expected hidden dimension.
            num_heads (int): Expected number of attention heads.

        Raises:
            AssertionError: If any component is not initialized correctly.
        """
        assert dit_attention.num_heads == num_heads
        assert dit_attention.head_dim == hidden_dim // num_heads
        assert dit_attention.scale == (hidden_dim // num_heads) ** -0.5
        assert dit_attention.partial_rotary_factor == 1.0
        assert dit_attention.rotary_dim == dit_attention.head_dim

        # Check layer dimensions
        assert dit_attention.qkv.in_features == input_dim
        assert dit_attention.qkv.out_features == 3 * hidden_dim
        assert dit_attention.proj_out.in_features == hidden_dim
        assert dit_attention.proj_out.out_features == input_dim

    def test_forward_shape(self, dit_attention: DiTAttention, batch_size: int, seq_len: int, input_dim: int):
        """
        Test DiTAttention forward pass output shape.

        Verifies that the attention mechanism produces output tensors with the correct
        shape and data type, maintaining input dimensions while processing attention.

        Args:
            dit_attention (DiTAttention): The attention module to test.
            batch_size (int): Size of the input batch.
            seq_len (int): Length of the input sequence.
            input_dim (int): Dimension of input features.

        Raises:
            AssertionError: If output shape or dtype doesn't match expectations.
        """
        input_tensor = torch.randn(batch_size, seq_len, input_dim)
        output: torch.Tensor = dit_attention(input_tensor)

        assert output.shape == (batch_size, seq_len, input_dim)
        assert output.dtype == input_tensor.dtype

    def test_forward_different_batch_sizes(self, dit_attention: DiTAttention, seq_len: int, input_dim: int):
        """
        Test DiTAttention with different batch sizes.

        Ensures the attention mechanism works correctly across various batch sizes

        Args:
            dit_attention (DiTAttention): The attention module to test.
            seq_len (int): Length of the input sequence.
            input_dim (int): Dimension of input features.

        Raises:
            AssertionError: If output shapes are incorrect for any batch size.
        """
        for batch_size in [1, 2, 8, 16]:
            input_tensor = torch.randn(batch_size, seq_len, input_dim)
            output: torch.Tensor = dit_attention(input_tensor)
            assert output.shape == (batch_size, seq_len, input_dim)

    def test_forward_different_seq_lengths(self, dit_attention: DiTAttention, batch_size: int, input_dim: int):
        """
        Test DiTAttention with different sequence lengths.

        Verifies that the attention mechanism handles variable sequence lengths correctly.
        Args:
            dit_attention (DiTAttention): The attention module to test.
            batch_size (int): Size of the input batch.
            input_dim (int): Dimension of input features.

        Raises:
            AssertionError: If output shapes are incorrect for any sequence length.
        """
        for seq_len in [10, 25, 50, 100]:
            input_tensor = torch.randn(batch_size, seq_len, input_dim)
            output: torch.Tensor = dit_attention(input_tensor)
            assert output.shape == (batch_size, seq_len, input_dim)

    def test_partial_rotary_factor(self, input_dim: int, hidden_dim: int, num_heads: int):
        """
        Test DiTAttention with different partial rotary factors.

        Validates that the rotary embedding dimension is calculated correctly
        based on the partial rotary factor.

        Args:
            input_dim (int): Dimension of the input features.
            hidden_dim (int): Hidden dimension for attention computation.
            num_heads (int): Number of attention heads.

        Raises:
            AssertionError: If rotary dimension calculation is incorrect.
        """
        partial_rotary_factor = 0.5
        attention = DiTAttention(
            input_dim=input_dim,
            dim=hidden_dim,
            num_heads=num_heads,
            partial_rotary_factor=partial_rotary_factor,
        )

        expected_rotary_dim = int((hidden_dim // num_heads) * partial_rotary_factor)
        assert attention.rotary_dim == expected_rotary_dim

    def test_gradient_flow(self, dit_attention: DiTAttention, batch_size: int, seq_len: int, input_dim: int):
        """
        Test gradient flow through DiTAttention.

        Ensures that gradients can properly flow backward through the attention mechanism.

        Args:
            dit_attention (DiTAttention): The attention module to test.
            batch_size (int): Size of the input batch.
            seq_len (int): Length of the input sequence.
            input_dim (int): Dimension of input features.

        Raises:
            AssertionError: If gradients are not computed or have incorrect shapes.
        """
        input_tensor = torch.randn(batch_size, seq_len, input_dim, requires_grad=True)
        output: torch.Tensor = dit_attention(input_tensor)
        loss = output.sum()
        loss.backward()  # type: ignore[reportUnknownMemberType]

        assert input_tensor.grad is not None
        assert input_tensor.grad.shape == input_tensor.shape


class TestMMDiTAttention:
    """
    Test class for MMDiTAttention module.

    This class provides comprehensive testing for the MMDiTAttention multi-modal attention mechanism.

    Attributes:
        None

    Methods:
        mmdit_attention: Fixture that creates an MMDiTAttention instance for testing.
        test_init: Tests proper initialization of MMDiTAttention components.
        test_forward_shape: Tests output shapes for both input and context tensors.
        test_forward_different_dimensions: Tests behavior with different input/context dimensions.
        test_gradient_flow: Tests gradient propagation through the multi-modal attention.
    """

    @pytest.fixture(scope="class")
    def context_dim(self):
        """Standard context dimension for testing."""
        return 128

    @pytest.fixture(scope="class")
    def input_dim(self):
        """Standard input dimension for testing."""
        return 256

    @pytest.fixture(scope="class")
    def hidden_dim(self):
        """Standard hidden dimension for testing."""
        return 384

    @pytest.fixture(scope="class")
    def num_heads(self):
        """Standard number of attention heads for testing."""
        return 4

    @pytest.fixture(scope="class")
    def batch_size(self):
        """Standard batch size for testing."""
        return 2

    @pytest.fixture(scope="class")
    def seq_len(self):
        """Standard sequence length for testing."""
        return 28

    @pytest.fixture(scope="class")
    def context_seq_len(self):
        """Standard context sequence length for testing."""
        return 32

    @pytest.fixture(scope="class")
    def mmdit_attention(self, context_dim: int, input_dim: int, hidden_dim: int, num_heads: int):
        """
        Create MMDiTAttention instance for testing.

        Args:
            context_dim (int): Dimension of the context features.
            input_dim (int): Dimension of the input features.
            hidden_dim (int): Hidden dimension for attention computation.
            num_heads (int): Number of attention heads.

        Returns:
            MMDiTAttention: Configured multi-modal attention module for testing.
        """
        return MMDiTAttention(
            context_dim=context_dim,
            input_dim=input_dim,
            dim=hidden_dim,
            num_heads=num_heads,
            partial_rotary_factor=1.0,
            base=10000,
        )

    def test_init(
        self, mmdit_attention: MMDiTAttention, context_dim: int, input_dim: int, hidden_dim: int, num_heads: int
    ):
        """
        Test MMDiTAttention initialization.

        Verifies that all components for multi-modal attention are properly initialized
        with correct dimensions for both input and context processing paths.

        Args:
            mmdit_attention (MMDiTAttention): The multi-modal attention module to test.
            context_dim (int): Expected context dimension.
            input_dim (int): Expected input dimension.
            hidden_dim (int): Expected hidden dimension.
            num_heads (int): Expected number of attention heads.

        Raises:
            AssertionError: If any component is not initialized correctly.
        """
        assert mmdit_attention.num_heads == num_heads
        assert mmdit_attention.head_dim == hidden_dim // num_heads
        assert mmdit_attention.scale == (hidden_dim // num_heads) ** -0.5

        # Check layer dimensions
        assert mmdit_attention.qkv_input.in_features == input_dim
        assert mmdit_attention.qkv_input.out_features == 3 * hidden_dim
        assert mmdit_attention.qkv_context.in_features == context_dim
        assert mmdit_attention.qkv_context.out_features == 3 * hidden_dim
        assert mmdit_attention.input_proj_out.out_features == input_dim
        assert mmdit_attention.context_proj_out.out_features == context_dim

    def test_forward_shape(
        self,
        mmdit_attention: MMDiTAttention,
        batch_size: int,
        seq_len: int,
        context_seq_len: int,
        input_dim: int,
        context_dim: int,
    ):
        """
        Test MMDiTAttention forward pass output shapes.

        Verifies that the multi-modal attention produces output tensors with correct
        shapes for both input and context, maintaining their respective dimensions.

        Args:
            mmdit_attention (MMDiTAttention): The multi-modal attention module to test.
            batch_size (int): Size of the input batch.
            seq_len (int): Length of the input sequence.
            context_seq_len (int): Length of the context sequence.
            input_dim (int): Dimension of input features.
            context_dim (int): Dimension of context features.

        Raises:
            AssertionError: If output shapes don't match expected dimensions.
        """
        input_tensor = torch.randn(batch_size, seq_len, input_dim)
        context_tensor = torch.randn(batch_size, context_seq_len, context_dim)

        input_output, context_output = cast(
            tuple[torch.Tensor, torch.Tensor], mmdit_attention(input_tensor, context_tensor)
        )

        assert input_output.shape == (batch_size, seq_len, input_dim)
        assert context_output.shape == (batch_size, context_seq_len, context_dim)

    def test_forward_different_dimensions(self, batch_size: int, seq_len: int):
        """
        Test MMDiTAttention with different input and context dimensions.

        Validates that the multi-modal attention can handle different dimensional inputs
        for context and input tensors.

        Args:
            batch_size (int): Size of the input batch.
            seq_len (int): Length of the input sequence.

        Raises:
            AssertionError: If output shapes don't match input tensor shapes.
        """
        input_dim, context_dim = 256, 512
        hidden_dim = 384
        num_heads = 6

        attention = MMDiTAttention(
            context_dim=context_dim,
            input_dim=input_dim,
            dim=hidden_dim,
            num_heads=num_heads,
        )

        input_tensor = torch.randn(batch_size, seq_len, input_dim)
        context_tensor = torch.randn(batch_size, seq_len + 5, context_dim)

        input_output, context_output = cast(tuple[torch.Tensor, torch.Tensor], attention(input_tensor, context_tensor))

        assert input_output.shape == input_tensor.shape
        assert context_output.shape == context_tensor.shape

    def test_gradient_flow(
        self,
        mmdit_attention: MMDiTAttention,
        batch_size: int,
        seq_len: int,
        context_seq_len: int,
        input_dim: int,
        context_dim: int,
    ):
        """
        Test gradient flow through MMDiTAttention.

        Ensures that gradients can properly flow backward through the multi-modal attention
        mechanism for both input and context paths.

        Args:
            mmdit_attention (MMDiTAttention): The multi-modal attention module to test.
            batch_size (int): Size of the input batch.
            seq_len (int): Length of the input sequence.
            context_seq_len (int): Length of the context sequence.
            input_dim (int): Dimension of input features.
            context_dim (int): Dimension of context features.

        Raises:
            AssertionError: If gradients are not computed for input or context tensors.
        """
        input_tensor = torch.randn(batch_size, seq_len, input_dim, requires_grad=True)
        context_tensor = torch.randn(batch_size, context_seq_len, context_dim, requires_grad=True)

        input_output, context_output = cast(
            tuple[torch.Tensor, torch.Tensor], mmdit_attention(input_tensor, context_tensor)
        )
        loss = input_output.sum() + context_output.sum()
        loss.backward()  # type: ignore[reportUnknownMemberType]

        assert input_tensor.grad is not None
        assert context_tensor.grad is not None


class TestDiTBlock:
    """
    Test class for DiTBlock module.

    This class provides comprehensive testing for the DiTBlock transformer block,
    which combines modulation, normalization, self-attention, and MLP operations
    for simple diffusion transformers (without multi-modal capabilities).

    Attributes:
        None

    Methods:
        dit_block: Fixture that creates a DiTBlock instance for testing.
        test_init: Tests proper initialization of all DiTBlock components.
        test_forward_shape: Tests output shape consistency in forward pass.
        test_modulation_calls: Tests that modulation is applied correctly.
        test_gradient_flow: Tests gradient propagation through the transformer block.
    """

    @pytest.fixture(scope="class")
    def input_dim(self):
        """Standard input dimension for testing."""
        return 256

    @pytest.fixture(scope="class")
    def hidden_dim(self):
        """Standard hidden dimension for testing."""
        return 384

    @pytest.fixture(scope="class")
    def num_heads(self):
        """Standard number of attention heads for testing."""
        return 4

    @pytest.fixture(scope="class")
    def batch_size(self):
        """Standard batch size for testing."""
        return 2

    @pytest.fixture(scope="class")
    def embedding_dim(self):
        """Standard embedding dimension for testing."""
        return 64

    @pytest.fixture(scope="class")
    def mlp_ratio(self):
        """Standard MLP ratio for testing."""
        return 4

    @pytest.fixture(scope="class")
    def seq_len(self):
        """Standard sequence length for testing."""
        return 16

    @pytest.fixture(scope="class")
    def dit_block(self, input_dim: int, hidden_dim: int, embedding_dim: int, num_heads: int, mlp_ratio: int):
        """
        Create DiTBlock instance for testing.

        Args:
            input_dim (int): Dimension of the input features.
            hidden_dim (int): Hidden dimension for attention computation.
            embedding_dim (int): Dimension of the conditioning embeddings.
            num_heads (int): Number of attention heads.
            mlp_ratio (int): Ratio for MLP hidden dimension expansion.

        Returns:
            DiTBlock: Configured transformer block for testing.
        """
        return DiTBlock(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            partial_rotary_factor=1.0,
        )

    def test_init(self, dit_block: DiTBlock, input_dim: int, mlp_ratio: int):
        """
        Test DiTBlock initialization.

        Verifies that all components of the transformer block are properly initialized,
        including modulation layers, normalization layers, attention mechanism, and MLP.
        Validates that MLP dimensions are correctly computed based on the mlp_ratio.

        Args:
            dit_block (DiTBlock): The transformer block to test.
            input_dim (int): Expected input dimension.
            mlp_ratio (int): Expected MLP expansion ratio.

        Raises:
            AssertionError: If any component is not initialized correctly or has wrong dimensions.
        """
        assert isinstance(dit_block.modulation, nn.Module)
        assert isinstance(dit_block.norm_1, nn.RMSNorm)
        assert isinstance(dit_block.attention, DiTAttention)
        assert isinstance(dit_block.norm_2, nn.RMSNorm)
        assert len(dit_block.mlp_input) == 3  # Linear, GELU, Linear

        # Check MLP dimensions
        assert dit_block.mlp_input[0].in_features == input_dim
        assert dit_block.mlp_input[0].out_features == mlp_ratio * input_dim
        assert dit_block.mlp_input[2].in_features == mlp_ratio * input_dim
        assert dit_block.mlp_input[2].out_features == input_dim

    def test_forward_shape(
        self, dit_block: DiTBlock, batch_size: int, seq_len: int, input_dim: int, embedding_dim: int
    ):
        """
        Test DiTBlock forward pass output shape.

        Verifies that the transformer block maintains input tensor shape and data type
        throughout the forward pass, ensuring proper integration with the overall model.

        Args:
            dit_block (DiTBlock): The transformer block to test.
            batch_size (int): Size of the input batch.
            seq_len (int): Length of the input sequence.
            input_dim (int): Dimension of input features.
            embedding_dim (int): Dimension of the conditioning embeddings.

        Raises:
            AssertionError: If output shape or data type doesn't match input.
        """
        input_tensor = torch.randn(batch_size, seq_len, input_dim)
        y = torch.randn(batch_size, embedding_dim)

        output: torch.Tensor = dit_block(input_tensor, y)

        assert output.shape == input_tensor.shape
        assert output.dtype == input_tensor.dtype

    @patch("diffulab.networks.denoisers.mmdit.modulate")
    def test_modulation_calls(
        self,
        mock_modulate: MagicMock,
        dit_block: DiTBlock,
        batch_size: int,
        seq_len: int,
        input_dim: int,
        embedding_dim: int,
    ):
        """
        Test that modulation is called correctly.

        Ensures that the modulate function is called the expected number of times
        Uses mocking to verify the modulation behavior without relying on actual computations.

        Args:
            mock_modulate (MagicMock): Mock object for the modulate function.
            dit_block (DiTBlock): The transformer block to test.
            batch_size (int): Size of the input batch.
            seq_len (int): Length of the input sequence.
            input_dim (int): Dimension of input features.
            embedding_dim (int): Dimension of the conditioning embeddings.

        Raises:
            AssertionError: If modulate is not called the expected number of times (twice).
        """
        input_tensor = torch.randn(batch_size, seq_len, input_dim)
        y = torch.randn(batch_size, embedding_dim)

        # Mock modulate to return input unchanged
        mock_modulate.return_value = input_tensor

        dit_block(input_tensor, y)

        # Should be called twice (for attention and MLP)
        assert mock_modulate.call_count == 2

    def test_gradient_flow(
        self, dit_block: DiTBlock, batch_size: int, seq_len: int, input_dim: int, embedding_dim: int
    ):
        """
        Test gradient flow through DiTBlock.

        Ensures that gradients can properly flow backward through the transformer block
        for both input tensors and conditioning embeddings.

        Args:
            dit_block (DiTBlock): The transformer block to test.
            batch_size (int): Size of the input batch.
            seq_len (int): Length of the input sequence.
            input_dim (int): Dimension of input features.
            embedding_dim (int): Dimension of the conditioning embeddings.

        Raises:
            AssertionError: If gradients are not computed for input tensor or embeddings.
        """
        input_tensor = torch.randn(batch_size, seq_len, input_dim, requires_grad=True)
        y = torch.randn(batch_size, embedding_dim, requires_grad=True)

        output: torch.Tensor = dit_block(input_tensor, y)
        loss = output.sum()
        loss.backward()  # type: ignore[reportUnknownMemberType]

        assert input_tensor.grad is not None
        assert y.grad is not None


class TestMMDiTBlock:
    """
    Test class for MMDiTBlock module.

    This class provides comprehensive testing for the MMDiTBlock multi-modal transformer block,
    which combines modulation, normalization, multi-modal attention, and MLP operations
    for processing both input and context tensors in multi-modal diffusion transformers.

    Attributes:
        None

    Methods:
        mmdit_block: Fixture that creates an MMDiTBlock instance for testing.
        test_init: Tests proper initialization of all MMDiTBlock components.
        test_forward_shape: Tests output shapes for both input and context tensors.
        test_gradient_flow: Tests gradient propagation through the multi-modal block.
    """

    @pytest.fixture(scope="class")
    def input_dim(self):
        """Standard input dimension for testing."""
        return 256

    @pytest.fixture(scope="class")
    def context_dim(self):
        """Standard context dimension for testing."""
        return 128

    @pytest.fixture(scope="class")
    def hidden_dim(self):
        """Standard hidden dimension for testing."""
        return 384

    @pytest.fixture(scope="class")
    def num_heads(self):
        """Standard number of attention heads for testing."""
        return 4

    @pytest.fixture(scope="class")
    def batch_size(self):
        """Standard batch size for testing."""
        return 2

    @pytest.fixture(scope="class")
    def embedding_dim(self):
        """Standard embedding dimension for testing."""
        return 64

    @pytest.fixture(scope="class")
    def mlp_ratio(self):
        """Standard MLP ratio for testing."""
        return 4

    @pytest.fixture(scope="class")
    def seq_len(self):
        """Standard sequence length for testing."""
        return 16

    @pytest.fixture(scope="class")
    def context_seq_len(self):
        """Standard context sequence length for testing."""
        return 32

    @pytest.fixture(scope="class")
    def mmdit_block(
        self, context_dim: int, input_dim: int, hidden_dim: int, embedding_dim: int, num_heads: int, mlp_ratio: int
    ):
        """
        Create MMDiTBlock instance for testing.

        Args:
            context_dim (int): Dimension of the context features.
            input_dim (int): Dimension of the input features.
            hidden_dim (int): Hidden dimension for attention computation.
            embedding_dim (int): Dimension of the conditioning embeddings.
            num_heads (int): Number of attention heads.
            mlp_ratio (int): Ratio for MLP hidden dimension expansion.

        Returns:
            MMDiTBlock: Configured multi-modal transformer block for testing.
        """
        return MMDiTBlock(
            context_dim=context_dim,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            partial_rotary_factor=1.0,
        )

    def test_init(self, mmdit_block: MMDiTBlock, context_dim: int, input_dim: int, mlp_ratio: int):
        """
        Test MMDiTBlock initialization.

        Verifies that all components for multi-modal processing are properly initialized,
        including separate modulation paths for input and context, multi-modal attention,
        and correctly sized MLP layers for both input and context processing.

        Args:
            mmdit_block (MMDiTBlock): The multi-modal transformer block to test.
            context_dim (int): Expected context dimension.
            input_dim (int): Expected input dimension.
            mlp_ratio (int): Expected MLP expansion ratio.

        Raises:
            AssertionError: If any component is not initialized correctly or has wrong dimensions.
        """
        assert isinstance(mmdit_block.modulation_context, nn.Module)
        assert isinstance(mmdit_block.modulation_input, nn.Module)
        assert isinstance(mmdit_block.attention, MMDiTAttention)

        # Check MLP dimensions
        assert mmdit_block.mlp_context[0].in_features == context_dim
        assert mmdit_block.mlp_context[0].out_features == mlp_ratio * context_dim
        assert mmdit_block.mlp_input[0].in_features == input_dim
        assert mmdit_block.mlp_input[0].out_features == mlp_ratio * input_dim

    def test_forward_shape(
        self,
        mmdit_block: MMDiTBlock,
        batch_size: int,
        seq_len: int,
        context_seq_len: int,
        input_dim: int,
        context_dim: int,
        embedding_dim: int,
    ):
        """
        Test MMDiTBlock forward pass output shapes.

        Verifies that the multi-modal transformer block maintains correct shapes for both
        input and context tensors throughout the forward pass, ensuring proper integration
        with the overall multi-modal architecture.

        Args:
            mmdit_block (MMDiTBlock): The multi-modal transformer block to test.
            batch_size (int): Size of the input batch.
            seq_len (int): Length of the input sequence.
            context_seq_len (int): Length of the context sequence.
            input_dim (int): Dimension of input features.
            context_dim (int): Dimension of context features.
            embedding_dim (int): Dimension of conditioning embeddings.

        Raises:
            AssertionError: If output shapes don't match input tensor shapes.
        """
        input_tensor = torch.randn(batch_size, seq_len, input_dim)
        context_tensor = torch.randn(batch_size, context_seq_len, context_dim)
        y = torch.randn(batch_size, embedding_dim)

        input_output, context_output = cast(
            tuple[torch.Tensor, torch.Tensor], mmdit_block(input_tensor, y, context_tensor)
        )
        assert input_output.shape == input_tensor.shape
        assert context_output.shape == context_tensor.shape

    def test_gradient_flow(
        self,
        mmdit_block: MMDiTBlock,
        batch_size: int,
        seq_len: int,
        context_seq_len: int,
        input_dim: int,
        context_dim: int,
        embedding_dim: int,
    ):
        """
        Test gradient flow through MMDiTBlock.

        Ensures that gradients can properly flow backward through the multi-modal transformer
        block for input tensors, context tensors, and conditioning embeddings.

        Args:
            mmdit_block (MMDiTBlock): The multi-modal transformer block to test.
            batch_size (int): Size of the input batch.
            seq_len (int): Length of the input sequence.
            context_seq_len (int): Length of the context sequence.
            input_dim (int): Dimension of input features.
            context_dim (int): Dimension of context features.
            embedding_dim (int): Dimension of conditioning embeddings.

        Raises:
            AssertionError: If gradients are not computed for any of the input tensors.
        """
        input_tensor = torch.randn(batch_size, seq_len, input_dim, requires_grad=True)
        context_tensor = torch.randn(batch_size, context_seq_len, context_dim, requires_grad=True)
        y = torch.randn(batch_size, embedding_dim, requires_grad=True)

        input_output, context_output = cast(
            tuple[torch.Tensor, torch.Tensor], mmdit_block(input_tensor, y, context_tensor)
        )
        loss = input_output.sum() + context_output.sum()
        loss.backward()  # type: ignore[reportUnknownMemberType]

        assert input_tensor.grad is not None
        assert context_tensor.grad is not None
        assert y.grad is not None


class TestModulatedLastLayer:
    """
    Test class for ModulatedLastLayer module.

    This class provides comprehensive testing for the ModulatedLastLayer, which is the final
    output layer of the diffusion transformer that converts hidden features back to pixel space
    using adaptive layer normalization based on conditioning embeddings.

    Attributes:
        None

    Methods:
        last_layer: Fixture that creates a ModulatedLastLayer instance for testing.
        test_init: Tests proper initialization of the output layer components.
        test_forward_shape: Tests output shape matches expected patch dimensions.
        test_gradient_flow: Tests gradient propagation through the output layer.
    """

    @pytest.fixture(scope="class")
    def embedding_dim(self):
        """Standard embedding dimension for testing."""
        return 64

    @pytest.fixture(scope="class")
    def input_dim(self):
        """Standard input dimension for testing."""
        return 128

    @pytest.fixture(scope="class")
    def patch_size(self):
        """Standard patch size for testing."""
        return 16

    @pytest.fixture(scope="class")
    def input_channels(self):
        """Standard number of output channels for testing."""
        return 3

    @pytest.fixture(scope="class")
    def batch_size(self):
        """Standard batch size for testing."""
        return 2

    @pytest.fixture(scope="class")
    def seq_len(self):
        """Standard sequence length for testing."""
        return 32

    @pytest.fixture(scope="class")
    def last_layer(self, embedding_dim: int, input_dim: int, patch_size: int, input_channels: int):
        """
        Create ModulatedLastLayer instance for testing.

        Args:
            embedding_dim (int): Dimension of the conditioning embeddings.
            input_dim (int): Dimension of the hidden features.
            patch_size (int): Size of image patches.
            input_channels (int): Number of output channels.

        Returns:
            ModulatedLastLayer: Configured output layer for testing.
        """
        return ModulatedLastLayer(
            embedding_dim=embedding_dim,
            hidden_size=input_dim,
            patch_size=patch_size,
            out_channels=input_channels,
        )

    def test_init(
        self,
        last_layer: ModulatedLastLayer,
        input_dim: int,
        patch_size: int,
        input_channels: int,
    ):
        """
        Test ModulatedLastLayer initialization.

        Verifies that the output layer is properly initialized with correct normalization,
        linear projection, and adaptive layer normalization components. Validates that
        output dimensions match the expected patch and channel configuration.

        Args:
            last_layer (ModulatedLastLayer): The output layer to test.
            input_dim (int): Expected hidden feature dimension.
            patch_size (int): Expected patch size.
            input_channels (int): Expected number of output channels.

        Raises:
            AssertionError: If any component is not initialized correctly or has wrong dimensions.
        """
        assert isinstance(last_layer.norm_final, nn.LayerNorm)
        assert last_layer.norm_final.normalized_shape == (input_dim,)
        assert not last_layer.norm_final.elementwise_affine

        expected_out_features = patch_size * patch_size * input_channels
        assert last_layer.linear.out_features == expected_out_features
        assert last_layer.linear.in_features == input_dim

        # Check adaLN_modulation
        assert len(last_layer.adaLN_modulation) == 2
        assert last_layer.adaLN_modulation[1].out_features == 2 * input_dim

    def test_forward_shape(
        self,
        last_layer: ModulatedLastLayer,
        batch_size: int,
        seq_len: int,
        input_dim: int,
        embedding_dim: int,
        patch_size: int,
        input_channels: int,
    ):
        """
        Test ModulatedLastLayer forward pass output shape.

        Verifies that the output layer produces tensors with the correct shape for
        reconstruction into image patches, ensuring proper dimensionality for the
        unpatchify operation in the main model.

        Args:
            last_layer (ModulatedLastLayer): The output layer to test.
            batch_size (int): Size of the input batch.
            seq_len (int): Length of the input sequence.
            input_dim (int): Dimension of hidden features.
            embedding_dim (int): Dimension of conditioning embeddings.
            patch_size (int): Size of image patches.
            input_channels (int): Number of output channels.

        Raises:
            AssertionError: If output shape doesn't match expected patch dimensions.
        """
        x = torch.randn(batch_size, seq_len, input_dim)
        vec = torch.randn(batch_size, embedding_dim)

        output: torch.Tensor = last_layer(x, vec)

        expected_features = patch_size * patch_size * input_channels
        assert output.shape == (batch_size, seq_len, expected_features)

    def test_gradient_flow(
        self, last_layer: ModulatedLastLayer, batch_size: int, seq_len: int, input_dim: int, embedding_dim: int
    ):
        """
        Test gradient flow through ModulatedLastLayer.

        Ensures that gradients can properly flow backward through the output layer
        for both hidden features and conditioning embeddings, which is essential
        for end-to-end training of the diffusion model.

        Args:
            last_layer (ModulatedLastLayer): The output layer to test.
            batch_size (int): Size of the input batch.
            seq_len (int): Length of the input sequence.
            input_dim (int): Dimension of hidden features.
            embedding_dim (int): Dimension of conditioning embeddings.

        Raises:
            AssertionError: If gradients are not computed for input features or embeddings.
        """
        x = torch.randn(batch_size, seq_len, input_dim, requires_grad=True)
        vec = torch.randn(batch_size, embedding_dim, requires_grad=True)

        output: torch.Tensor = last_layer(x, vec)
        loss = output.sum()
        loss.backward()  # type: ignore[reportUnknownMemberType]

        assert x.grad is not None
        assert vec.grad is not None


class TestMMDiT:
    """
    Test class for MMDiT module.

    This class provides comprehensive testing for the main MMDiT (Multimodal Diffusion Transformer)
    model, which supports both simple DiT and multi-modal configurations. Tests cover initialization,
    forward passes, patchification/unpatchification, different configurations, and error conditions.

    Attributes:
        None

    Methods:
        simple_dit: Fixture that creates a simple DiT configuration for testing.
        mmdit_with_context: Fixture that creates a multi-modal DiT configuration for testing.
        test_simple_dit_init: Tests initialization of simple DiT configuration.
        test_mmdit_with_context_init: Tests initialization of multi-modal configuration.
        test_invalid_init_both_classes_and_context: Tests error when conflicting configs provided.
        test_invalid_init_mmdit_without_context: Tests error when context embedder missing.
        test_patchify_unpatchify_roundtrip: Tests image-to-patches-to-image conversion.
        test_simple_dit_forward: Tests forward pass for simple DiT mode.
        test_mmdit_forward_with_context: Tests forward pass for multi-modal mode.
        test_forward_with_intermediate_features: Tests intermediate feature extraction.
        test_forward_with_x_context: Tests input concatenation with context.
        test_forward_classifier_free_guidance: Tests classifier-free guidance probability.
        test_forward_invalid_both_context_and_labels: Tests error with conflicting inputs.
        test_forward_invalid_p_without_classifier_free: Tests error with invalid guidance config.
        test_gradient_flow_simple_dit: Tests gradient flow in simple DiT mode.
        test_gradient_flow_mmdit_with_context: Tests gradient flow in multi-modal mode.
        test_different_output_channels: Tests model with different output channel counts.
        test_different_depths: Tests model with various transformer depths.
        test_different_patch_sizes: Tests model with various patch sizes.
        test_weight_initialization: Tests proper weight initialization.
    """

    # Hyperparameter fixtures (mirroring other test classes for consistency)
    @pytest.fixture(scope="class")
    def input_channels(self):
        return 3

    @pytest.fixture(scope="class")
    def input_dim(self):
        return 256

    @pytest.fixture(scope="class")
    def hidden_dim(self):
        return 384

    @pytest.fixture(scope="class")
    def embedding_dim(self):
        return 64

    @pytest.fixture(scope="class")
    def num_heads(self):
        return 4

    @pytest.fixture(scope="class")
    def mlp_ratio(self):
        return 4

    @pytest.fixture(scope="class")
    def patch_size(self):
        return 16

    @pytest.fixture(scope="class")
    def context_dim(self):
        return 128

    # Sample data fixtures
    @pytest.fixture(scope="class")
    def sample_image(self, input_channels: int):
        return torch.randn(2, input_channels, 64, 64)

    @pytest.fixture(scope="class")
    def sample_timesteps(self):
        return torch.randn(2)

    @pytest.fixture(scope="class")
    def sample_labels(self):
        return torch.randint(0, 10, (2,))

    @pytest.fixture(scope="class")
    def simple_dit(
        self,
        input_channels: int,
        input_dim: int,
        hidden_dim: int,
        embedding_dim: int,
        num_heads: int,
        mlp_ratio: int,
        patch_size: int,
    ):
        """
        Create simple MMDiT instance for testing.

        Args:
            input_channels (int): Number of input image channels.
            input_dim (int): Dimension of the input features.
            hidden_dim (int): Hidden dimension for attention computation.
            embedding_dim (int): Dimension of the conditioning embeddings.
            num_heads (int): Number of attention heads.
            mlp_ratio (int): Ratio for MLP hidden dimension expansion.
            patch_size (int): Size of image patches.

        Returns:
            MMDiT: Configured simple DiT model for testing.
        """
        return MMDiT(
            simple_dit=True,
            input_channels=input_channels,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            patch_size=patch_size,
            depth=2,  # Small depth for testing
            n_classes=10,
            classifier_free=True,
        )

    @pytest.fixture(scope="class")
    def mmdit_with_context(
        self,
        input_channels: int,
        input_dim: int,
        hidden_dim: int,
        embedding_dim: int,
        num_heads: int,
        mlp_ratio: int,
        patch_size: int,
        context_dim: int,
        mock_context_embedder: ContextEmbedder,
    ):
        """
        Create MMDiT with context embedder for testing.

        Args:
            input_channels (int): Number of input image channels.
            input_dim (int): Dimension of the input features.
            hidden_dim (int): Hidden dimension for attention computation.
            embedding_dim (int): Dimension of the conditioning embeddings.
            num_heads (int): Number of attention heads.
            mlp_ratio (int): Ratio for MLP hidden dimension expansion.
            patch_size (int): Size of image patches.
            context_dim (int): Dimension of context features.
            mock_context_embedder (ContextEmbedder): Mock context embedder for testing.

        Returns:
            MMDiT: Configured multi-modal DiT model for testing.
        """
        return MMDiT(
            simple_dit=False,
            input_channels=input_channels,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            patch_size=patch_size,
            depth=2,
            context_dim=context_dim,
            context_embedder=mock_context_embedder,
        )

    def test_simple_dit_init(self, simple_dit: MMDiT, input_channels: int, patch_size: int):
        """
        Test simple MMDiT initialization.

        Verifies that the simple DiT configuration is properly initialized with correct
        parameters, including class conditioning setup and absence of context embedder.

        Args:
            simple_dit (MMDiT): Simple DiT model instance to test.
            input_channels (int): Expected number of input channels.
            patch_size (int): Expected patch size.

        Raises:
            AssertionError: If any initialization parameter is incorrect.
        """
        assert simple_dit.simple_dit is True
        assert simple_dit.input_channels == input_channels
        assert simple_dit.output_channels == input_channels  # Should default to input_channels
        assert simple_dit.patch_size == patch_size
        assert simple_dit.n_classes == 10
        assert simple_dit.classifier_free is True
        assert simple_dit.context_embedder is None
        assert simple_dit.label_embed is not None

    def test_mmdit_with_context_init(self, mmdit_with_context: MMDiT, mock_context_embedder: ContextEmbedder):
        """
        Test MMDiT with context embedder initialization.

        Verifies that the multi-modal DiT configuration is properly initialized with
        context embedder setup and absence of class conditioning components.

        Args:
            mmdit_with_context (MMDiT): Multi-modal DiT model instance to test.
            input_channels (int): Expected number of input channels.
            patch_size (int): Expected patch size.
            mock_context_embedder (ContextEmbedder): Expected context embedder instance.

        Raises:
            AssertionError: If any initialization parameter is incorrect or missing components.
        """
        assert mmdit_with_context.simple_dit is False
        assert mmdit_with_context.context_embedder is mock_context_embedder
        assert mmdit_with_context.label_embed is None
        assert hasattr(mmdit_with_context, "mlp_pooled_context")
        assert hasattr(mmdit_with_context, "context_embed")

    def test_invalid_init_both_classes_and_context(self, mock_context_embedder: ContextEmbedder):
        """
        Test that MMDiT raises error when both n_classes and context_embedder are provided.

        Ensures that the model correctly validates initialization parameters and prevents
        conflicting configurations that would be ambiguous during training.

        Args:
            mock_context_embedder (ContextEmbedder): Mock context embedder for testing.

        Raises:
            AssertionError: When both n_classes and context_embedder are specified, which is invalid.
        """
        with pytest.raises(AssertionError, match="n_classes and context_embedder cannot both be specified"):
            MMDiT(
                n_classes=10,
                context_embedder=mock_context_embedder,
            )

    def test_invalid_init_mmdit_without_context(self):
        """
        Test that MMDiT raises error when not simple_dit but no context_embedder.

        Ensures that multi-modal DiT configurations require a context embedder,
        preventing incomplete initialization that would cause runtime errors.

        Raises:
            AssertionError: When simple_dit=False but no context_embedder is provided.
        """
        with pytest.raises(AssertionError, match="for MMDiT context embedder must be provided"):
            MMDiT(simple_dit=False)

    def test_patchify_unpatchify_roundtrip(self, simple_dit: MMDiT, sample_image: torch.Tensor):
        """
        Test that patchify and unpatchify are inverse operations.

        Verifies that the image-to-patches conversion and patches-to-image reconstruction
        maintain consistent shapes and allow proper processing through the transformer.
        This is crucial for the diffusion model's ability to process images as sequences.

        Args:
            simple_dit (MMDiT): Simple DiT model instance for testing.
            sample_image (torch.Tensor): Sample input image tensor.

        Raises:
            AssertionError: If patch dimensions or reconstruction shapes are incorrect.
        """
        # Patchify
        patches = simple_dit.patchify(sample_image)

        # Check patchify output shape
        batch_size, _, height, width = sample_image.shape
        expected_num_patches = (height // simple_dit.patch_size) * (width // simple_dit.patch_size)
        assert patches.shape == (batch_size, expected_num_patches, simple_dit.conv_proj.out_channels)

        # Create dummy output with correct channel dimension
        dummy_output = torch.randn(
            batch_size,
            expected_num_patches,
            simple_dit.patch_size * simple_dit.patch_size * simple_dit.output_channels,
        )

        # Unpatchify
        reconstructed = simple_dit.unpatchify(dummy_output)

        # Check reconstructed shape
        assert reconstructed.shape == sample_image.shape

    def test_simple_dit_forward(
        self,
        simple_dit: MMDiT,
        sample_image: torch.Tensor,
        sample_timesteps: torch.Tensor,
        sample_labels: torch.Tensor,
    ):
        """
        Test simple DiT forward pass.

        Verifies that the simple DiT configuration can successfully process images with
        class labels and timesteps, producing outputs with correct shapes and data types.

        Args:
            simple_dit (MMDiT): Simple DiT model instance to test.
            sample_image (torch.Tensor): Sample input image tensor.
            sample_timesteps (torch.Tensor): Sample timestep tensor.
            sample_labels (torch.Tensor): Sample class label tensor.

        Raises:
            AssertionError: If output shape, keys, or data type are incorrect.
        """
        output: ModelOutput = simple_dit(sample_image, sample_timesteps, y=sample_labels)

        assert "x" in output
        assert output["x"].shape == sample_image.shape
        assert output["x"].dtype == sample_image.dtype

    def test_mmdit_forward_with_context(
        self, mmdit_with_context: MMDiT, sample_image: torch.Tensor, sample_timesteps: torch.Tensor
    ):
        """
        Test MMDiT forward pass with context.

        Verifies that the multi-modal DiT configuration can successfully process images
        with context information, producing outputs with correct shapes.

        Args:
            mmdit_with_context (MMDiT): Multi-modal DiT model instance to test.
            sample_image (torch.Tensor): Sample input image tensor.
            sample_timesteps (torch.Tensor): Sample timestep tensor.

        Raises:
            AssertionError: If output shape or keys are incorrect.
        """
        output: ModelOutput = mmdit_with_context(sample_image, sample_timesteps, initial_context=sample_image.shape[0])

        assert "x" in output
        assert output["x"].shape == sample_image.shape

    def test_forward_with_intermediate_features(
        self,
        simple_dit: MMDiT,
        sample_image: torch.Tensor,
        sample_timesteps: torch.Tensor,
        sample_labels: torch.Tensor,
    ):
        """
        Test forward pass with intermediate features enabled.

        Verifies that the model can output intermediate layer features in addition to
        the final output, which is useful for analysis, visualization, and feature
        extraction applications.

        Args:
            simple_dit (MMDiT): Simple DiT model instance to test.
            sample_image (torch.Tensor): Sample input image tensor.
            sample_timesteps (torch.Tensor): Sample timestep tensor.
            sample_labels (torch.Tensor): Sample class label tensor.

        Raises:
            AssertionError: If intermediate features are not returned or have incorrect format.
        """
        # Need to access the internal forward methods
        patches = simple_dit.patchify(sample_image)
        output = simple_dit.simple_dit_forward(patches, sample_timesteps, y=sample_labels, intermediate_features=True)

        assert "x" in output
        assert "features" in output
        assert isinstance(output["features"], list)
        assert len(output["features"]) == simple_dit.layers.__len__() + 1  # +1 for last layer

    def test_forward_with_x_context(
        self,
        sample_image: torch.Tensor,
        sample_timesteps: torch.Tensor,
        sample_labels: torch.Tensor,
        input_channels: int,
        input_dim: int,
        hidden_dim: int,
        embedding_dim: int,
        num_heads: int,
        mlp_ratio: int,
        patch_size: int,
    ):
        """
        Test forward pass with x_context concatenation.

        Verifies that the model can handle additional context images concatenated
        to the input, which is useful for conditional generation scenarios.

        Args:
            sample_image (torch.Tensor): Sample input image tensor.
            sample_timesteps (torch.Tensor): Sample timestep tensor.
            sample_labels (torch.Tensor): Sample class label tensor.

        Raises:
            AssertionError: If output shape is incorrect when using x_context.
        """
        # Create a simple DiT model with twice the input channels for x_context
        simple_dit = MMDiT(
            simple_dit=True,
            input_channels=input_channels * 2,
            output_channels=input_channels,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            patch_size=patch_size,
            depth=2,  # Small depth for testing
            n_classes=10,
            classifier_free=True,
        )

        x_context = torch.randn_like(sample_image)
        output: ModelOutput = simple_dit(sample_image, sample_timesteps, y=sample_labels, x_context=x_context)

        assert "x" in output
        assert output["x"].shape == sample_image.shape

    def test_forward_classifier_free_guidance(
        self,
        simple_dit: MMDiT,
        sample_image: torch.Tensor,
        sample_timesteps: torch.Tensor,
        sample_labels: torch.Tensor,
    ):
        """
        Test forward pass with classifier-free guidance probability.

        Verifies that the model can handle classifier-free guidance dropout probability,
        which is used during training to enable guidance-free generation at inference.

        Args:
            simple_dit (MMDiT): Simple DiT model instance to test.
            sample_image (torch.Tensor): Sample input image tensor.
            sample_timesteps (torch.Tensor): Sample timestep tensor.
            sample_labels (torch.Tensor): Sample class label tensor.

        Raises:
            AssertionError: If output shape is incorrect when using guidance probability.
        """
        output: ModelOutput = simple_dit(sample_image, sample_timesteps, y=sample_labels, p=0.1)

        assert "x" in output
        assert output["x"].shape == sample_image.shape

    def test_forward_invalid_both_context_and_labels(
        self,
        mmdit_with_context: MMDiT,
        sample_image: torch.Tensor,
        sample_timesteps: torch.Tensor,
        sample_labels: torch.Tensor,
    ):
        """
        Test that forward raises error when both initial_context and y are provided.

        Ensures that the model prevents ambiguous conditioning scenarios where both
        context embeddings and class labels are provided simultaneously.

        Args:
            mmdit_with_context (MMDiT): Multi-modal DiT model instance.
            sample_image (torch.Tensor): Sample input image tensor.
            sample_timesteps (torch.Tensor): Sample timestep tensor.
            sample_labels (torch.Tensor): Sample class label tensor.

        Raises:
            AssertionError: When both initial_context and y parameters are specified.
        """
        with pytest.raises(AssertionError, match="initial_context and y cannot both be specified"):
            mmdit_with_context(sample_image, sample_timesteps, initial_context="test", y=sample_labels)

    def test_forward_invalid_p_without_classifier_free(self, input_channels: int):
        """
        Test that forward raises error when p > 0 but classifier_free is False.

        Ensures that classifier-free guidance dropout probability can only be used
        when the model is properly configured for classifier-free guidance training.

        Args:
            input_channels (int): Number of input image channels.

        Raises:
            AssertionError: When p > 0 but classifier_free=False in model configuration.
        """
        model = MMDiT(
            simple_dit=True,
            input_channels=input_channels,
            n_classes=10,
            classifier_free=False,
            depth=1,
        )

        sample_image = torch.randn(2, input_channels, 64, 64)
        sample_timesteps = torch.randn(2)
        sample_labels = torch.randint(0, 10, (2,))

        with pytest.raises(AssertionError, match="probability of dropping for classifier free guidance"):
            model(sample_image, sample_timesteps, y=sample_labels, p=0.1)

    def test_gradient_flow_simple_dit(
        self,
        simple_dit: MMDiT,
        sample_image: torch.Tensor,
        sample_timesteps: torch.Tensor,
        sample_labels: torch.Tensor,
    ):
        """
        Test gradient flow through simple DiT.

        Ensures that gradients can properly flow backward through the simple DiT
        configuration for both image inputs and timesteps.
        Args:
            simple_dit (MMDiT): Simple DiT model instance to test.
            sample_image (torch.Tensor): Sample input image tensor.
            sample_timesteps (torch.Tensor): Sample timestep tensor.
            sample_labels (torch.Tensor): Sample class label tensor.

        Raises:
            AssertionError: If gradients are not computed for inputs or timesteps.
        """
        sample_image.requires_grad_(True)
        sample_timesteps.requires_grad_(True)

        output: ModelOutput = simple_dit(sample_image, sample_timesteps, y=sample_labels)
        loss = output["x"].sum()
        loss.backward()  # type: ignore[reportUnknownMemberType]

        assert sample_image.grad is not None
        assert sample_timesteps.grad is not None

    def test_gradient_flow_mmdit_with_context(
        self, mmdit_with_context: MMDiT, sample_image: torch.Tensor, sample_timesteps: torch.Tensor
    ):
        """
        Test gradient flow through MMDiT with context.

        Ensures that gradients can properly flow backward through the multi-modal DiT
        configuration for both image inputs and timesteps.

        Args:
            mmdit_with_context (MMDiT): Multi-modal DiT model instance to test.
            sample_image (torch.Tensor): Sample input image tensor.
            sample_timesteps (torch.Tensor): Sample timestep tensor.

        Raises:
            AssertionError: If gradients are not computed for inputs or timesteps.
        """
        sample_image.requires_grad_(True)
        sample_timesteps.requires_grad_(True)

        output: ModelOutput = mmdit_with_context(sample_image, sample_timesteps, initial_context=sample_image.shape[0])
        loss = output["x"].sum()
        loss.backward()  # type: ignore[reportUnknownMemberType]

        assert sample_image.grad is not None
        assert sample_timesteps.grad is not None

    def test_different_output_channels(self, input_channels: int, patch_size: int):
        """
        Test MMDiT with different output channels.

        Validates that the model can be configured with a different number of output
        channels than input channels, which is useful for various generation tasks.

        Args:
            input_channels (int): Number of input image channels.
            patch_size (int): Size of image patches.

        Raises:
            AssertionError: If model doesn't handle different output channels correctly
                          or if output shape is wrong.
        """
        output_channels = input_channels * 2
        model = MMDiT(
            simple_dit=True,
            input_channels=input_channels,
            output_channels=output_channels,
            n_classes=10,
            patch_size=patch_size,
            depth=1,
        )

        assert model.output_channels == output_channels

        sample_image = torch.randn(2, input_channels, 64, 64)
        sample_timesteps = torch.randn(2)
        sample_labels = torch.randint(0, 10, (2,))

        output = model(sample_image, sample_timesteps, y=sample_labels)
        expected_shape = (2, output_channels, 64, 64)
        assert output["x"].shape == expected_shape

    @pytest.mark.parametrize("depth", [1, 2, 4, 8])
    def test_different_depths(self, depth: int, input_channels: int):
        """
        Test MMDiT with different depths.

        Validates that the model can be configured with various numbers of transformer
        layers and that each configuration produces valid outputs. This tests the
        model's scalability and ensures deep configurations work correctly.

        Args:
            depth (int): Number of transformer layers to test (1, 2, 4, or 8).
            input_channels (int): Number of input image channels.

        Raises:
            AssertionError: If the model doesn't have the expected number of layers
                          or if forward pass fails with the given depth.
        """
        model = MMDiT(
            simple_dit=True,
            input_channels=input_channels,
            n_classes=10,
            depth=depth,
        )

        assert len(model.layers) == depth

        sample_image = torch.randn(2, input_channels, 64, 64)
        sample_timesteps = torch.randn(2)
        sample_labels = torch.randint(0, 10, (2,))

        output: ModelOutput = model(sample_image, sample_timesteps, y=sample_labels)
        assert output["x"].shape == sample_image.shape

    @pytest.mark.parametrize("patch_size", [8, 16, 32])
    def test_different_patch_sizes(self, patch_size: int, input_channels: int):
        """
        Test MMDiT with different patch sizes.

        Validates that the model can handle various patch sizes for image tokenization,
        which affects the sequence length and computational requirements of the model.

        Args:
            patch_size (int): Size of image patches to test (8, 16, or 32).
            input_channels (int): Number of input image channels.

        Raises:
            AssertionError: If the model doesn't work correctly with the given patch size
                          or if output shape is incorrect.
        """
        model = MMDiT(
            simple_dit=True,
            input_channels=input_channels,
            patch_size=patch_size,
            n_classes=10,
            depth=1,
        )

        # Image size must be divisible by patch size
        image_size = patch_size * 4
        sample_image = torch.randn(2, input_channels, image_size, image_size)
        sample_timesteps = torch.randn(2)
        sample_labels = torch.randint(0, 10, (2,))

        output = model(sample_image, sample_timesteps, y=sample_labels)
        assert output["x"].shape == sample_image.shape

    def test_weight_initialization(self, simple_dit: MMDiT):
        """
        Test that weights are properly initialized.

        Ensures that all Linear and Conv2d layers have been properly initialized using
        Xavier uniform initialization for weights and zero initialization for biases,
        which is important for stable training convergence.

        Args:
            simple_dit (MMDiT): Simple DiT model instance to check.

        Raises:
            AssertionError: If any weights are improperly initialized (all zeros) or
                          if biases are not initialized to zero.
        """
        for module in simple_dit.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                # Check that weights are not all zeros (indicating initialization occurred)
                assert not torch.allclose(module.weight, torch.zeros_like(module.weight))
                if module.bias is not None:
                    # Biases should be initialized to zero
                    assert torch.allclose(module.bias, torch.zeros_like(module.bias))


class TestMMDiTEdgeCases:
    """
    Test edge cases and error conditions for MMDiT.

    This class focuses on testing boundary conditions, edge cases, and error scenarios
    that might occur in real-world usage of the MMDiT model. These tests ensure
    robustness and proper error handling across various input conditions.

    Attributes:
        None

    Methods:
        test_zero_batch_size: Tests behavior with empty batch tensors.
        test_non_square_images: Tests processing of non-square image inputs.
        test_single_head_attention: Tests model with single attention head configuration.
        test_very_small_dimensions: Tests model with minimal dimension settings.
    """

    # Fixtures (mirroring style used in other test classes for consistency)
    @pytest.fixture(scope="class")
    def input_channels(self):
        return 3

    @pytest.fixture(scope="class")
    def input_dim(self):
        return 64

    @pytest.fixture(scope="class")
    def hidden_dim(self):
        return 128

    @pytest.fixture(scope="class")
    def embedding_dim(self):
        return 128

    @pytest.fixture(scope="class")
    def num_heads(self):
        return 2  # small for tests

    @pytest.fixture(scope="class")
    def mlp_ratio(self):
        return 4

    @pytest.fixture(scope="class")
    def patch_size(self):
        return 16

    @pytest.fixture(scope="class")
    def small_image(self, input_channels: int):
        return torch.randn(2, input_channels, 32, 32)

    @pytest.fixture(scope="class")
    def empty_image(self, input_channels: int):
        return torch.empty((0, input_channels, 64, 64))

    @pytest.fixture(scope="class")
    def sample_timesteps(self):
        return torch.randn(2)

    @pytest.fixture(scope="class")
    def sample_labels(self):
        return torch.randint(0, 10, (2,))

    def test_zero_batch_size(
        self,
        input_channels: int,
        input_dim: int,
        hidden_dim: int,
        embedding_dim: int,
        num_heads: int,
        mlp_ratio: int,
        patch_size: int,
    ):
        """
        Test behavior with zero batch size.

        Verifies that the model can handle empty batches gracefully without crashing,
        which is important for robust deployment and edge case handling.

        Args:
            input_channels (int): Number of input image channels.
            input_dim (int): Dimension of the input features.
            hidden_dim (int): Hidden dimension for attention computation.
            embedding_dim (int): Dimension of the conditioning embeddings.
            num_heads (int): Number of attention heads.
            mlp_ratio (int): Ratio for MLP hidden dimension expansion.
            patch_size (int): Size of image patches.

        Raises:
            AssertionError: If the model doesn't handle empty batches correctly or
                          if output batch dimension is not zero.
        """
        model = MMDiT(
            simple_dit=True,
            input_channels=input_channels,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            patch_size=patch_size,
            n_classes=10,
            depth=1,
        )
        sample_image = torch.empty((0, input_channels, 64, 64))
        sample_timesteps = torch.empty((0,))
        sample_labels = torch.empty((0,), dtype=torch.long)
        output: ModelOutput = model(sample_image, sample_timesteps, y=sample_labels)
        assert output["x"].shape[0] == 0

    def test_non_square_images(
        self,
        input_channels: int,
        input_dim: int,
        hidden_dim: int,
        embedding_dim: int,
        num_heads: int,
        mlp_ratio: int,
        patch_size: int,
    ):
        """
        Test model behavior with non-square image inputs.

        Validates that the MMDiT model can correctly process rectangular images
        with different width and height dimensions, which is important for handling
        diverse image aspect ratios in real-world applications.

        Args:
            input_channels (int): Number of input image channels.
            input_dim (int): Dimension of the input features.
            hidden_dim (int): Hidden dimension for attention computation.
            embedding_dim (int): Dimension of the conditioning embeddings.
            num_heads (int): Number of attention heads.
            mlp_ratio (int): Ratio for MLP hidden dimension expansion.
            patch_size (int): Size of image patches.

        Raises:
            AssertionError: If the model fails to process non-square images correctly
                          or if output dimensions don't match expected values.
            ValueError: If the patch size doesn't divide evenly into the image dimensions.
        """
        model = MMDiT(
            simple_dit=True,
            input_channels=input_channels,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            patch_size=patch_size,
            n_classes=10,
            depth=1,
        )
        sample_image = torch.randn(2, input_channels, 64, 128)
        sample_timesteps = torch.randn(2)
        sample_labels = torch.randint(0, 10, (2,))
        output: ModelOutput = model(sample_image, sample_timesteps, y=sample_labels)
        assert output["x"].shape == sample_image.shape

    def test_single_head_attention(
        self,
        input_channels: int,
        input_dim: int,
        hidden_dim: int,
        embedding_dim: int,
        mlp_ratio: int,
        patch_size: int,
    ):
        """
        Test model with single attention head configuration.

        Verifies that the MMDiT model works correctly with the minimal attention
        head configuration of one head, which can be useful for smaller models
        or resource-constrained environments.

        Args:
            input_channels (int): Number of input image channels.
            input_dim (int): Dimension of the input features.
            hidden_dim (int): Hidden dimension for attention computation.
            embedding_dim (int): Dimension of the conditioning embeddings.
            mlp_ratio (int): Ratio for MLP hidden dimension expansion.
            patch_size (int): Size of image patches.

        Raises:
            AssertionError: If the model fails with single head attention or
                          if output shape is incorrect.
        """
        model = MMDiT(
            simple_dit=True,
            input_channels=input_channels,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            num_heads=1,
            mlp_ratio=mlp_ratio,
            patch_size=patch_size,
            n_classes=10,
            depth=1,
        )
        sample_image = torch.randn(2, input_channels, 64, 64)
        sample_timesteps = torch.randn(2)
        sample_labels = torch.randint(0, 10, (2,))
        output: ModelOutput = model(sample_image, sample_timesteps, y=sample_labels)
        assert output["x"].shape == sample_image.shape

    def test_very_small_dimensions(self, input_channels: int):
        """
        Test model with very small dimension configurations.

        Validates that the MMDiT model can function with minimal dimension settings.

        Args:
            input_channels (int): Number of input image channels.

        Raises:
            AssertionError: If the model fails with small dimensions or
                          if output shape is incorrect.
            ValueError: If the dimension configuration is invalid or incompatible.
        """
        model = MMDiT(
            simple_dit=True,
            input_channels=input_channels,
            input_dim=64,
            hidden_dim=64,
            embedding_dim=64,
            num_heads=2,
            n_classes=10,
            depth=1,
        )

        sample_image = torch.randn(2, input_channels, 32, 32)
        sample_timesteps = torch.randn(2)
        sample_labels = torch.randint(0, 10, (2,))

        output: ModelOutput = model(sample_image, sample_timesteps, y=sample_labels)
        assert output["x"].shape == sample_image.shape

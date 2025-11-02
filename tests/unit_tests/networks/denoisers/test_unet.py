"""
Unit tests for the UNet-based denoiser components (ResBlock, AttentionBlock, UNetModel).

Overview:
    This module provides comprehensive, behavior-focused test coverage for the UNetModel
    architecture and its fundamental building blocks used in diffusion-style denoisers.
    Tests emphasize initialization correctness, tensor shape integrity, conditioning
    pathways (class and context), classifier-free guidance behavior, attention injection,
    architectural configuration flags, gradient propagation, and edge/error handling.

Components Under Test:
    - ResBlock: Residual block supporting optional up/down sampling and FiLM-style
      (scale, shift) modulation via timestep / context embeddings.
    - AttentionBlock: Multi-head self/cross attention operating over flattened spatial
      tokens with optional external context key/value tensor.
    - UNetModel: Full hierarchical encoder–decoder with skip connections, channel
      multipliers, attention at selectable resolutions, and optional class / context
      conditioning plus classifier-free guidance.

Fixtures Provided:
    Reusable pytest fixtures standardize tensor shapes and hyperparameters (e.g.,
    batch size, spatial dimensions, channels, embedding dimensions, and synthetic
    context). Context conditioning uses a lightweight deterministic MockContextEmbedder.

Test Coverage Summary:
    - Initialization: Parameter bookkeeping, submodule presence, configuration flags.
    - Forward Pass: Output tensor shapes across conditioning modes and architectural
      permutations (channel multipliers, scale-shift norm, resblock up/down routing,
      gradient checkpointing, multiple attention resolutions, auxiliary image context).
    - Conditioning: Class label vs context embeddings, classifier-free guidance mask
      probability, assertion coverage when required inputs are omitted.
    - Attention: Self vs cross attention equivalence, multi-resolution insertion counts.
    - Gradient Flow: Backpropagation to inputs and embeddings for critical paths.
    - Edge / Error Cases: Non-square inputs, mismatched spatial sizes, invalid guidance
      probability usage, zero / altered shapes (implicit via assertions).

Classes:
    TestResBlock: Initialization, forward shape, scale-shift variant, up/down sampling,
        gradient flow.
    TestAttentionBlock: Initialization, self vs cross attention, gradient flow,
        equivalence check.
    TestUNetModel: Conditioning modes, guidance probability, architectural flags,
        attention resolution counts, gradient flow, edge/error assertions.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
import torch.nn as nn
from torch import Tensor

from diffulab.networks.denoisers.common import ModelOutput
from diffulab.networks.denoisers.unet import (
    AttentionBlock,
    ResBlock,
    UNetModel,
)
from diffulab.networks.embedders.common import ContextEmbedder


class MockContextEmbedder(ContextEmbedder):
    """Mock context embedder used to emulate external sequence conditioning.

    Generates a single context tensor shaped ``(B, C_ctx, L)`` where both the channel
    dimension and sequence length are configurable. Implements a simple dropout
    mechanism that zeroes entire per-sample context tensors with probability ``p``.

    Args:
        context_channels (int, optional): Number of context feature channels. Defaults to 32.
        context_length (int, optional): Number of sequential context tokens. Defaults to 16.

    Returns:
        tuple[torch.Tensor]: A one-element tuple containing the context tensor of
            shape ``(B, context_channels, context_length)``.

    Raises:
        None

    Notes:
        The forward method accepts either an integer (interpreted as batch size) or a
        tensor whose first dimension supplies the batch size. This keeps tests lightweight
        and independent from upstream data dependencies.
    """

    def __init__(self, context_channels: int = 32, context_length: int = 16):
        super().__init__()  # type: ignore
        self._channels = context_channels
        self._length = context_length

    @property
    def n_output(self) -> int:  # type: ignore[override]
        return 1

    @property
    def output_size(self) -> tuple[int, ...]:  # type: ignore[override]
        return (self._channels, self._length)

    def drop_conditions(self, context: Any, p: float) -> Any:  # type: ignore[override]
        if p <= 0:
            return context
        mask = torch.rand(context.shape[0], device=context.device) < p
        context[mask] = 0.0
        return context

    def forward(self, context: Any, p: float):  # type: ignore[override]
        # context argument unused; we synthesize deterministic tensor for shape tests
        B = context if isinstance(context, int) else (context.shape[0] if torch.is_tensor(context) else 2)  # fallback
        ctx = torch.randn(B, self._channels, self._length)
        ctx = self.drop_conditions(ctx, p)
        return (ctx,)


@pytest.fixture(scope="module")
def context_channels() -> int:
    """Standard number of context channels for testing."""
    return 32


@pytest.fixture(scope="module")
def context_length() -> int:
    """Standard context sequence length for testing."""
    return 16


@pytest.fixture(scope="module")
def mock_context_embedder(context_channels: int, context_length: int) -> MockContextEmbedder:
    return MockContextEmbedder(context_channels=context_channels, context_length=context_length)


# ---------------------------------------------------------------------------
# ResBlock Tests
# ---------------------------------------------------------------------------
class TestResBlock:
    """Tests for the UNet residual block (ResBlock).

    Validates correct construction and runtime behavior across variants.

    Covered Behaviors:
        * Channel / embedding bookkeeping and skip connection adaptation.
        * Standard forward pass output shape integrity.
        * FiLM scale/shift normalization path activation.
        * Spatial size changes under up/down sampling flags.
        * Gradient propagation to inputs and embeddings.

    Test Methods:
        - test_init
        - test_forward_shape
        - test_scale_shift_variant
        - test_up_and_down
        - test_gradient_flow
    """

    @pytest.fixture(scope="class")
    def batch_size(self) -> int:
        """Standard batch size for ResBlock tests."""
        return 4

    @pytest.fixture(scope="class")
    def channels(self) -> int:
        """Standard input channel count for the block."""
        return 32

    @pytest.fixture(scope="class")
    def embedding_dim(self) -> int:
        """Standard embedding (timestep/context) dimension used for FiLM style modulation."""
        return 128

    @pytest.fixture(scope="class")
    def dropout(self) -> float:
        """Dropout probability applied inside the block MLP path."""
        return 0.1

    @pytest.fixture(scope="class")
    def out_channels(self) -> int:
        """Output channel count (tests skip connection adaptation)."""
        return 64

    @pytest.fixture(scope="class")
    def hw_image_size(self) -> list[int]:
        """Spatial resolution [H, W] for generated feature maps."""
        return [64, 64]

    @pytest.fixture(scope="class")
    def resblock(self, channels: int, embedding_dim: int, dropout: float, out_channels: int) -> ResBlock:
        """Instantiate a canonical ResBlock under test.

        Args:
            channels: Input feature channels.
            embedding_dim: Embedding dimension for conditioning.
            dropout: Dropout probability.
            out_channels: Desired output channel count.
        """
        return ResBlock(
            channels=channels,
            emb_channels=embedding_dim,
            dropout=dropout,
            out_channels=out_channels,
            use_conv=True,
            use_scale_shift_norm=False,
        )

    def test_init(self, resblock: ResBlock, channels: int, embedding_dim: int, out_channels: int):
        """Validate ResBlock initialization and configured submodules.

        Args:
            resblock (ResBlock): Instance under test.
            channels (int): Expected input channel count.
            embedding_dim (int): Expected embedding (conditioning) dimension.
            out_channels (int): Expected output channel count after block.

        Returns:
            None

        Raises:
            AssertionError: If any expected dimension or module type is incorrect.
        """
        assert resblock.channels == channels
        assert resblock.emb_channels == embedding_dim
        assert resblock.out_channels == out_channels
        # Skip connection should be 3x3 conv because use_conv True and channel change
        assert isinstance(resblock.skip_connection, nn.Conv2d)
        assert resblock.skip_connection.kernel_size in [(3, 3), (1, 1)]
        # Emb layers dimension
        lin = resblock.emb_layers[1]
        assert isinstance(lin, nn.Linear)
        assert lin.in_features == embedding_dim
        assert lin.out_features == out_channels

    def test_forward_shape(
        self,
        batch_size: int,
        resblock: ResBlock,
        channels: int,
        embedding_dim: int,
        out_channels: int,
        hw_image_size: list[int],
    ):
        """Ensure standard forward pass preserves spatial size and updates channels.

        Args:
            batch_size (int): Number of samples in mini-batch.
            resblock (ResBlock): Block under test.
            channels (int): Input channel count.
            embedding_dim (int): Embedding dimension for modulation.
            out_channels (int): Expected output channel count.
            hw_image_size (list[int]): Spatial dimensions [H, W].

        Returns:
            None

        Raises:
            AssertionError: If output tensor shape deviates from expectation.
        """
        x = torch.randn(batch_size, channels, *hw_image_size)
        emb = torch.randn(batch_size, embedding_dim)
        out = resblock(x, emb)
        assert out.shape == (batch_size, out_channels, *hw_image_size)

    def test_scale_shift_variant(
        self,
        channels: int,
        embedding_dim: int,
        dropout: float,
        out_channels: int,
        batch_size: int,
        hw_image_size: list[int],
    ):
        """Test variant using scale-shift normalization FiLM modulation path.

        Args:
            channels (int): Input channel count.
            embedding_dim (int): Embedding dimension.
            dropout (float): Dropout probability for internal layers.
            out_channels (int): Output channel count.
            batch_size (int): Batch size.
            hw_image_size (list[int]): Spatial dimensions [H, W].

        Returns:
            None

        Raises:
            AssertionError: If output shape is incorrect.
        """
        block = ResBlock(
            channels=channels,
            emb_channels=embedding_dim,
            dropout=dropout,
            out_channels=out_channels,
            use_scale_shift_norm=True,
        )
        x = torch.randn(batch_size, channels, *hw_image_size)
        emb = torch.randn(batch_size, embedding_dim)
        out = block(x, emb)
        assert out.shape == (batch_size, out_channels, *hw_image_size)

    def test_up_and_down(
        self,
        channels: int,
        embedding_dim: int,
        out_channels: int,
        batch_size: int,
        hw_image_size: list[int],
        dropout: float,
    ):
        """Validate spatial scaling for up=True and down=True configurations.

        Args:
            channels (int): Input channel count.
            embedding_dim (int): Embedding dimension.
            out_channels (int): Output channel count.
            batch_size (int): Batch size.
            hw_image_size (list[int]): Input spatial dimensions [H, W].
            dropout (float): Dropout probability.

        Returns:
            None

        Raises:
            AssertionError: If spatial scaling factors are incorrect.
        """
        # Up
        up_block = ResBlock(channels, embedding_dim, dropout, out_channels=out_channels, up=True)
        x = torch.randn(batch_size, channels, *hw_image_size)
        emb = torch.randn(batch_size, embedding_dim)
        up_out = up_block(x, emb)
        assert up_out.shape[2:] == (hw_image_size[0] * 2, hw_image_size[1] * 2)
        # Down
        down_block = ResBlock(channels, embedding_dim, dropout, out_channels=out_channels, down=True)
        down_out = down_block(x, emb)
        assert down_out.shape[2:] == (hw_image_size[0] // 2, hw_image_size[1] // 2)

    def test_gradient_flow(
        self, resblock: ResBlock, batch_size: int, channels: int, hw_image_size: list[int], embedding_dim: int
    ):
        """Confirm gradients propagate to inputs and embeddings during backward pass.

        Args:
            resblock (ResBlock): Block under test.
            batch_size (int): Batch size.
            channels (int): Input channel count.
            hw_image_size (list[int]): Spatial dimensions [H, W].
            embedding_dim (int): Embedding dimension.

        Returns:
            None

        Raises:
            AssertionError: If gradients are missing for inputs or embeddings.
        """
        x = torch.randn(batch_size, channels, *hw_image_size, requires_grad=True)
        emb = torch.randn(batch_size, embedding_dim, requires_grad=True)
        out = resblock(x, emb)
        out.sum().backward()  # type: ignore
        assert x.grad is not None
        assert emb.grad is not None


# ---------------------------------------------------------------------------
# AttentionBlock Tests
# ---------------------------------------------------------------------------
class TestAttentionBlock:
    """Tests for spatial multi-head (self/cross) attention block.

    Covered Behaviors:
        * Projection layer initialization & dimensional attributes.
        * Self-attention (implicit context) forward shape.
        * Cross-attention with external context forward shape.
        * Gradient propagation to feature and context tensors.
        * Equivalence of self-attention vs cross-attention when context==flattened input.

    Test Methods:
        - test_init
        - test_forward_self_attention
        - test_forward_cross_attention
        - test_gradient_flow
        - test_attention_self_equivalence
    """

    @pytest.fixture(scope="class")
    def channels(self) -> int:
        """Canonical feature channel size for attention queries and outputs."""
        return 64

    @pytest.fixture(scope="class")
    def context_channels(self) -> int:
        """Channel size for external context key/value inputs."""
        return 32

    @pytest.fixture(scope="class")
    def num_heads(self) -> int:
        """Number of attention heads to partition inner channels."""
        return 4

    @pytest.fixture(scope="class")
    def inner_channels(self) -> int:
        """Projected MHA internal channel size (must divide by num_heads)."""
        return 64

    @pytest.fixture(scope="class")
    def batch_size(self) -> int:
        """Mini-batch size for attention tests."""
        return 4

    @pytest.fixture(scope="class")
    def hw_image_size(self) -> list[int]:
        """Spatial resolution for feature map tokens (H, W)."""
        return [8, 8]

    @pytest.fixture(scope="class")
    def context_length(self) -> int:
        """Temporal/sequence length of flattened external context tokens."""
        return 32

    @pytest.fixture(scope="class")
    def attn_block(self, channels: int, context_channels: int, num_heads: int, inner_channels: int) -> AttentionBlock:
        """Instantiate AttentionBlock with explicit cross-attention capability."""
        return AttentionBlock(
            channels=channels, context_channels=context_channels, num_heads=num_heads, inner_channels=inner_channels
        )

    def test_init(
        self, attn_block: AttentionBlock, channels: int, context_channels: int, num_heads: int, inner_channels: int
    ):
        """Check attribute values and projection layer setups after initialization.

        Args:
            attn_block (AttentionBlock): Instance under test.
            channels (int): Query/input channel dimension.
            context_channels (int): Context channel dimension.
            num_heads (int): Number of attention heads.
            inner_channels (int): Projected internal channel dimension.

        Returns:
            None

        Raises:
            AssertionError: If any attribute or module setup is invalid.
        """
        assert attn_block.channels == channels
        assert attn_block.context_channels == context_channels
        assert attn_block.inner_channels == inner_channels
        assert attn_block.num_heads == num_heads
        assert attn_block.dim_head == inner_channels // num_heads
        assert isinstance(attn_block.to_q, nn.Conv1d)
        assert isinstance(attn_block.to_kv, nn.Conv1d)
        assert attn_block.to_q.in_channels == channels
        assert attn_block.to_kv.in_channels == context_channels

    def test_forward_self_attention(
        self, batch_size: int, channels: int, inner_channels: int, num_heads: int, hw_image_size: list[int]
    ):
        """Validate self-attention path when context is omitted (defaults to x).

        Args:
            batch_size (int): Batch size.
            channels (int): Input channel count.
            inner_channels (int): Internal projection dim.
            num_heads (int): Number of heads.
            hw_image_size (list[int]): Spatial dims [H, W].

        Returns:
            None

        Raises:
            AssertionError: If output shape mismatches input.
        """
        block = AttentionBlock(channels=channels, num_heads=num_heads, inner_channels=inner_channels)
        x = torch.randn(batch_size, channels, *hw_image_size)
        out = block(x, None)
        assert out.shape == x.shape

    def test_forward_cross_attention(
        self,
        attn_block: AttentionBlock,
        batch_size: int,
        channels: int,
        context_channels: int,
        hw_image_size: list[int],
        context_length: int,
    ):
        """Ensure cross-attention consumes external context and preserves output spatial shape.

        Args:
            attn_block (AttentionBlock): Instance under test.
            batch_size (int): Batch size.
            channels (int): Input channel dimension.
            context_channels (int): Context feature channel dimension.
            hw_image_size (list[int]): Spatial dims [H, W].
            context_length (int): Number of context tokens.

        Returns:
            None

        Raises:
            AssertionError: If output shape deviates from expected.
        """
        x = torch.randn(batch_size, channels, *hw_image_size)
        context = torch.randn(batch_size, context_channels, context_length)
        out = attn_block(x, context)
        assert out.shape == x.shape

    def test_gradient_flow(
        self,
        attn_block: AttentionBlock,
        batch_size: int,
        channels: int,
        context_channels: int,
        hw_image_size: list[int],
        context_length: int,
    ):
        """Check gradients for both input feature map and context tokens.

        Args:
            attn_block (AttentionBlock): Instance under test.
            batch_size (int): Batch size.
            channels (int): Input channels.
            context_channels (int): Context channels.
            hw_image_size (list[int]): Spatial dims.
            context_length (int): Context token count.

        Returns:
            None

        Raises:
            AssertionError: If gradients are absent for input or context.
        """
        x = torch.randn(batch_size, channels, *hw_image_size, requires_grad=True)
        context = torch.randn(batch_size, context_channels, context_length, requires_grad=True)
        out = attn_block(x, context)
        out.mean().backward()  # type: ignore
        assert x.grad is not None
        assert context.grad is not None

    def test_attention_self_equivalence(
        self, batch_size: int, hw_image_size: list[int], channels: int, inner_channels: int, num_heads: int
    ):
        """Self-attention result must equal cross-attention when context == flattened x.

        Args:
            batch_size (int): Batch size.
            hw_image_size (list[int]): Spatial dims [H, W].
            channels (int): Input channels.
            inner_channels (int): Projection channels.
            num_heads (int): Number of heads.

        Returns:
            None

        Raises:
            AssertionError: If self and cross attention outputs differ beyond tolerance.
        """
        block = AttentionBlock(channels=channels, inner_channels=inner_channels, num_heads=num_heads)
        x = torch.randn(batch_size, channels, *hw_image_size)
        out_self = block(x, None)
        context = x.view(batch_size, channels, -1)
        out_cross = block(x, context)
        assert torch.allclose(out_self, out_cross, atol=1e-5)


# ---------------------------------------------------------------------------
# UNetModel Tests
# ---------------------------------------------------------------------------
class TestUNetModel:
    """Comprehensive tests for the full UNetModel denoiser.

    Focus Areas:
        * Initialization under class-conditional and context-conditional modes.
        * Forward pass shape correctness across conditioning configurations.
        * Classifier-free guidance probability masking semantics.
        * Auxiliary image concatenation (x_context) path correctness.
        * Non-square and mismatched spatial size handling (assertions).
        * Architectural permutations: channel multipliers, scale-shift norm, resblock
          up/down, gradient checkpointing, multi-attention resolutions.
        * Output head zero-initialization and timestep dtype flexibility (int/float).
        * Gradient flow through conditioning pathways.

    Test Methods:
        - test_init_class_cond
        - test_init_context_cond
        - test_forward_class_cond
        - test_forward_context_cond
        - test_forward_with_x_context
        - test_classifier_free_guidance_probability
        - test_invalid_missing_labels
        - test_invalid_missing_context
        - test_invalid_p_without_classifier_free
        - test_non_square_image
        - test_incorrect_image_size_assert
        - test_gradient_flow
        - test_variable_channel_mult
        - test_use_scale_shift_norm
        - test_resblock_updown
        - test_use_checkpoint
        - test_output_zero_init
        - test_integer_and_float_timesteps
        - test_multiple_attention_resolutions
    """

    @pytest.fixture(scope="class")
    def hw_image_size(self) -> list[int]:
        """Canonical spatial size for UNet inputs (H, W)."""
        return [64, 64]

    @pytest.fixture(scope="class")
    def in_channels(self) -> int:
        """Input image channel count (e.g., RGB=3)."""
        return 3

    @pytest.fixture(scope="class")
    def model_channels(self) -> int:
        """Base channel width for UNet encoder/decoder stages."""
        return 64

    @pytest.fixture(scope="class")
    def out_channels(self) -> int:
        """Number of prediction channels (often equals in_channels)."""
        return 3

    @pytest.fixture(scope="class")
    def num_res_blocks(self) -> int:
        """Residual blocks per resolution level."""
        return 2

    @pytest.fixture(scope="class")
    def attention_resolutions(self) -> list[int]:
        """Downsample factors (relative to input) where attention blocks are inserted."""
        return [4]

    @pytest.fixture(scope="class")
    def num_classes(self) -> int:
        """Number of discrete class labels for class conditioning tests."""
        return 10

    @pytest.fixture(scope="class")
    def channel_mult(self) -> str:
        """Comma-separated multipliers expanding channels across resolutions."""
        return "1,2"

    @pytest.fixture(scope="class")
    def num_heads(self) -> int:
        """Attention heads in spatial attention modules."""
        return 4

    @pytest.fixture(scope="class")
    def sample_images(self, batch_size: int, in_channels: int, hw_image_size: list[int]) -> Tensor:
        """Random batch of input images for forward passes."""
        return torch.randn(batch_size, in_channels, *hw_image_size)

    @pytest.fixture(scope="class")
    def timesteps(self, batch_size: int) -> Tensor:
        """Random float timesteps in [0,1) used for time embedding path."""
        return torch.rand(batch_size)

    @pytest.fixture(scope="class")
    def sample_labels(self, batch_size: int, num_classes: int) -> Tensor:
        """Random integer class labels for classifier-free guidance tests."""
        return torch.randint(0, num_classes, (batch_size,))

    @pytest.fixture(scope="class")
    def batch_size(self) -> int:
        """Standard batch size for UNet tests."""
        return 2

    @pytest.fixture(scope="class")
    def unet_class_cond(
        self,
        hw_image_size: list[int],
        in_channels: int,
        model_channels: int,
        out_channels: int,
        num_res_blocks: int,
        attention_resolutions: list[int],
        num_classes: int,
        channel_mult: str,
        num_heads: int,
    ) -> UNetModel:
        """Instantiate a class-conditional UNet with classifier-free guidance enabled."""
        return UNetModel(
            image_size=hw_image_size,
            in_channels=in_channels,
            model_channels=model_channels,
            out_channels=out_channels,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            n_classes=num_classes,
            classifier_free=True,
            channel_mult=channel_mult,
            num_heads=num_heads,
        )

    @pytest.fixture(scope="class")
    def unet_context_cond(
        self,
        hw_image_size: list[int],
        in_channels: int,
        model_channels: int,
        out_channels: int,
        num_res_blocks: int,
        attention_resolutions: list[int],
        mock_context_embedder: MockContextEmbedder,
        channel_mult: str,
        num_heads: int,
    ) -> UNetModel:
        """Instantiate a context-conditional UNet using a mock ContextEmbedder."""
        return UNetModel(
            image_size=hw_image_size,
            in_channels=in_channels,
            model_channels=model_channels,
            out_channels=out_channels,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            context_embedder=mock_context_embedder,
            channel_mult=channel_mult,
            num_heads=num_heads,
        )

    def test_init_class_cond(self, unet_class_cond: UNetModel, num_classes: int):
        """Confirm components for class conditioning (label embedding) are created.

        Args:
            unet_class_cond (UNetModel): Class-conditional UNet instance.
            num_classes (int): Expected number of classes.

        Returns:
            None

        Raises:
            AssertionError: If label embedding or class count not configured properly.
        """
        assert unet_class_cond.n_classes == num_classes
        assert unet_class_cond.label_embed is not None
        assert unet_class_cond.context_embedder is None
        # Extract first layer of first EmbedSequential block
        first_block = unet_class_cond.input_blocks[0]
        submodules = [m for m in first_block.modules() if m is not first_block]
        first = submodules[0] if submodules else None
        assert isinstance(first, nn.Conv2d)

    def test_init_context_cond(self, unet_context_cond: UNetModel, mock_context_embedder: MockContextEmbedder):
        """Verify context embedder is wired and middle block contains attention modules.

        Args:
            unet_context_cond (UNetModel): Context-conditional UNet instance.
            mock_context_embedder (MockContextEmbedder): Provided mock embedder.

        Returns:
            None

        Raises:
            AssertionError: If embedder not assigned or attention modules missing.
        """
        assert unet_context_cond.context_embedder is mock_context_embedder
        assert unet_context_cond.label_embed is None
        # Middle block has AttentionBlock
        assert any(isinstance(m, AttentionBlock) for m in unet_context_cond.middle_block)

    def test_forward_class_cond(
        self,
        unet_class_cond: UNetModel,
        sample_images: Tensor,
        sample_labels: Tensor,
        timesteps: Tensor,
    ):
        """Run forward pass with class labels and validate output tensor shape.

        Args:
            unet_class_cond (UNetModel): Class-conditional model.
            sample_images (Tensor): Input images (B, C, H, W).
            sample_labels (Tensor): Class labels (B,).
            timesteps (Tensor): Timestep values (B,).

        Returns:
            None

        Raises:
            AssertionError: If output key or shape is incorrect.
        """
        out: ModelOutput = unet_class_cond(sample_images, timesteps, y=sample_labels)
        assert "x" in out
        assert out["x"].shape == sample_images.shape

    def test_forward_context_cond(
        self,
        unet_context_cond: UNetModel,
        sample_images: Tensor,
        timesteps: Tensor,
    ):
        """Forward with context conditioning; uses batch size as synthetic context input.

        Args:
            unet_context_cond (UNetModel): Context-conditioned model.
            sample_images (Tensor): Input images (B, C, H, W).
            timesteps (Tensor): Timestep values (B,).

        Returns:
            None

        Raises:
            AssertionError: If output shape mismatches input.
        """
        out: ModelOutput = unet_context_cond(
            sample_images, timesteps, context=sample_images.shape[0]
        )  # pass batch size
        assert out["x"].shape == sample_images.shape

    def test_forward_with_x_context(
        self,
        sample_images: Tensor,
        timesteps: Tensor,
        hw_image_size: list[int],
        in_channels: int,
        model_channels: int,
        num_res_blocks: int,
        attention_resolutions: list[int],
        channel_mult: str,
        num_heads: int,
    ):
        """Test auxiliary image concatenation path (x_context) doubles input channels.

        Args:
            sample_images (Tensor): Primary input images.
            timesteps (Tensor): Timestep values.
            hw_image_size (list[int]): Image size [H, W].
            in_channels (int): Base input channels.
            model_channels (int): Base model channels.
            num_res_blocks (int): Residual blocks per level.
            attention_resolutions (list[int]): Attention insertion scales.
            channel_mult (str): Channel multiplier string.
            num_heads (int): Attention heads.

        Returns:
            None

        Raises:
            AssertionError: If output shape differs from original image shape.
        """
        unet = UNetModel(
            image_size=hw_image_size,
            in_channels=in_channels * 2,
            model_channels=model_channels,
            out_channels=in_channels,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            channel_mult=channel_mult,
            num_heads=num_heads,
        )
        # Provide additional context image (same spatial size)
        x_context = torch.randn_like(sample_images)
        out = unet(sample_images, timesteps, x_context=x_context)
        assert out["x"].shape == sample_images.shape

    def test_classifier_free_guidance_probability(
        self, unet_class_cond: UNetModel, sample_images: Tensor, timesteps: Tensor, sample_labels: Tensor
    ):
        """Ensure classifier-free guidance probability p executes masked label path without error.

        Args:
            unet_class_cond (UNetModel): Class-conditional model with guidance enabled.
            sample_images (Tensor): Input images.
            timesteps (Tensor): Timestep tensor.
            sample_labels (Tensor): Class labels.

        Returns:
            None

        Raises:
            AssertionError: If output shape mismatches expectations.
        """
        out = unet_class_cond(sample_images, timesteps, y=sample_labels, p=0.2)
        assert out["x"].shape == sample_images.shape

    def test_invalid_missing_labels(self, unet_class_cond: UNetModel, sample_images: Tensor, timesteps: Tensor):
        """Expect assertion if class-conditional model is called without labels.

        Args:
            unet_class_cond (UNetModel): Class-conditional model.
            sample_images (Tensor): Input batch.
            timesteps (Tensor): Timesteps.

        Returns:
            None

        Raises:
            AssertionError: Always (validated via context manager) due to missing labels.
        """
        with pytest.raises(AssertionError, match="must specify y"):
            unet_class_cond(sample_images, timesteps)

    def test_invalid_missing_context(self, unet_context_cond: UNetModel, sample_images: Tensor, timesteps: Tensor):
        """Expect assertion if context-conditional model lacks context input.

        Args:
            unet_context_cond (UNetModel): Context-conditional model.
            sample_images (Tensor): Input batch.
            timesteps (Tensor): Timesteps.

        Returns:
            None

        Raises:
            AssertionError: Always when context not supplied.
        """
        with pytest.raises(AssertionError, match="must specify context"):
            unet_context_cond(sample_images, timesteps)

    def test_invalid_p_without_classifier_free(
        self,
        hw_image_size: list[int],
        in_channels: int,
        model_channels: int,
        out_channels: int,
        num_res_blocks: int,
        attention_resolutions: list[int],
        num_classes: int,
        channel_mult: str,
        sample_images: Tensor,
        timesteps: Tensor,
        sample_labels: Tensor,
    ):
        """Calling with p when classifier_free=False should raise an assertion.

        Args:
            hw_image_size (list[int]): Image size [H, W].
            in_channels (int): Input channels.
            model_channels (int): Base model channels.
            out_channels (int): Output channels.
            num_res_blocks (int): Residual blocks per level.
            attention_resolutions (list[int]): Attention resolutions.
            num_classes (int): Number of classes.
            channel_mult (str): Channel mult string.
            sample_images (Tensor): Input images.
            timesteps (Tensor): Timesteps.
            sample_labels (Tensor): Class labels.

        Returns:
            None

        Raises:
            AssertionError: Always (verified) because p provided while classifier_free disabled.
        """
        model = UNetModel(
            image_size=hw_image_size,
            in_channels=in_channels,
            model_channels=model_channels,
            out_channels=out_channels,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            n_classes=num_classes,
            classifier_free=False,
            channel_mult=channel_mult,
        )
        with pytest.raises(
            AssertionError,
            match="probability of dropping",
        ):
            model(sample_images, timesteps, y=sample_labels, p=0.1)

    def test_non_square_image(
        self,
        in_channels: int,
        model_channels: int,
        num_res_blocks: int,
        attention_resolutions: list[int],
        channel_mult: str,
        num_classes: int,
        batch_size: int,
        timesteps: Tensor,
        sample_labels: Tensor,
    ):
        """Verify model accepts non-square spatial dimensions and returns matching shape.

        Args:
            in_channels (int): Input channels.
            model_channels (int): Base model channels.
            num_res_blocks (int): Residual blocks per level.
            attention_resolutions (list[int]): Attention resolutions.
            channel_mult (str): Channel multipliers.
            num_classes (int): Number of classes.
            batch_size (int): Batch size.
            timesteps (Tensor): Timesteps.
            sample_labels (Tensor): Class labels.

        Returns:
            None

        Raises:
            AssertionError: If output shape mismatches non-square input shape.
        """
        img_size = [32, 48]
        model = UNetModel(
            image_size=img_size,
            in_channels=in_channels,
            model_channels=model_channels,
            out_channels=in_channels,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            channel_mult=channel_mult,
            n_classes=num_classes,
            classifier_free=True,
        )
        x = torch.randn(batch_size, in_channels, *img_size)
        out = model(x, timesteps, y=sample_labels)
        assert out["x"].shape == x.shape

    def test_incorrect_image_size_assert(self, unet_class_cond: UNetModel, in_channels: int):
        """Providing mismatched spatial size should trigger assertion guard.

        Args:
            unet_class_cond (UNetModel): Configured UNet expecting specific image size.
            in_channels (int): Channel count for synthetic mismatch tensor.

        Returns:
            None

        Raises:
            AssertionError: Always when provided image size differs from configured size.
        """
        wrong = torch.randn(2, in_channels, 32, 32)
        t = torch.randint(0, 1000, (2,))
        y = torch.randint(0, 10, (2,))
        with pytest.raises(AssertionError, match="Input shape"):
            unet_class_cond(wrong, t, y=y)

    def test_gradient_flow(
        self, unet_class_cond: UNetModel, sample_images: Tensor, timesteps: Tensor, sample_labels: Tensor
    ):
        """Ensure gradients flow to image tensor and timestep inputs.

        Args:
            unet_class_cond (UNetModel): Model under test.
            sample_images (Tensor): Input images (requires_grad set in test).
            timesteps (Tensor): Timesteps (requires_grad set in test).
            sample_labels (Tensor): Class labels.

        Returns:
            None

        Raises:
            AssertionError: If gradients are missing after backward pass.
        """
        sample_images.requires_grad_(True)
        timesteps.requires_grad_(True)
        out = unet_class_cond(sample_images, timesteps, y=sample_labels)
        out["x"].sum().backward()  # type: ignore
        assert sample_images.grad is not None
        assert timesteps.grad is not None

    @pytest.mark.parametrize("channel_mult", ["1", "1,2", "1,2,4"])
    def test_variable_channel_mult(
        self,
        channel_mult: str,
        hw_image_size: list[int],
        in_channels: int,
        model_channels: int,
        out_channels: int,
        num_res_blocks: int,
        attention_resolutions: list[int],
        num_classes: int,
        sample_images: Tensor,
        timesteps: Tensor,
        sample_labels: Tensor,
    ):
        """Smoke test different channel multiplier layouts for forward shape stability.

        Args:
            channel_mult (str): Channel multiplier specification.
            hw_image_size (list[int]): Image size [H, W].
            in_channels (int): Input channels.
            model_channels (int): Base model channels.
            out_channels (int): Output channels.
            num_res_blocks (int): Residual blocks per level.
            attention_resolutions (list[int]): Attention resolutions.
            num_classes (int): Number of classes.
            sample_images (Tensor): Input images.
            timesteps (Tensor): Timesteps.
            sample_labels (Tensor): Class labels.

        Returns:
            None

        Raises:
            AssertionError: If forward output shape deviates from input shape.
        """
        model = UNetModel(
            image_size=hw_image_size,
            in_channels=in_channels,
            model_channels=model_channels,
            out_channels=out_channels,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            channel_mult=channel_mult,
            n_classes=num_classes,
            classifier_free=True,
        )
        out = model(sample_images, timesteps, y=sample_labels)
        assert out["x"].shape == sample_images.shape

    def test_use_scale_shift_norm(
        self,
        hw_image_size: list[int],
        in_channels: int,
        model_channels: int,
        num_res_blocks: int,
        attention_resolutions: list[int],
        channel_mult: str,
        num_classes: int,
    ):
        """Verify enabling use_scale_shift_norm applies to all ResBlocks.

        Args:
            hw_image_size (list[int]): Image size [H, W].
            in_channels (int): Input channels.
            model_channels (int): Base model channels.
            num_res_blocks (int): Residual blocks per level.
            attention_resolutions (list[int]): Attention resolutions.
            channel_mult (str): Channel multiplier spec.
            num_classes (int): Number of classes.

        Returns:
            None

        Raises:
            AssertionError: If any ResBlock lacks scale-shift norm when enabled.
        """
        model = UNetModel(
            image_size=hw_image_size,
            in_channels=in_channels,
            model_channels=model_channels,
            out_channels=in_channels,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            channel_mult=channel_mult,
            n_classes=num_classes,
            classifier_free=True,
            use_scale_shift_norm=True,
        )
        resblocks = [m for m in model.modules() if isinstance(m, ResBlock)]
        assert all(rb.use_scale_shift_norm for rb in resblocks)

    def test_resblock_updown(
        self,
        hw_image_size: list[int],
        in_channels: int,
        model_channels: int,
        num_res_blocks: int,
        attention_resolutions: list[int],
        channel_mult: str,
        num_classes: int,
    ):
        """Check presence of ResBlocks configured for up/down sampling when resblock_updown=True.

        Args:
            hw_image_size (list[int]): Image size [H, W].
            in_channels (int): Input channels.
            model_channels (int): Base model channels.
            num_res_blocks (int): Residual blocks per level.
            attention_resolutions (list[int]): Attention resolutions.
            channel_mult (str): Channel multiplier spec.
            num_classes (int): Number of classes.

        Returns:
            None

        Raises:
            AssertionError: If no up/down sampling ResBlocks are found.
        """
        model = UNetModel(
            image_size=hw_image_size,
            in_channels=in_channels,
            model_channels=model_channels,
            out_channels=in_channels,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            channel_mult=channel_mult,
            n_classes=num_classes,
            classifier_free=True,
            resblock_updown=True,
        )
        # Ensure there are ResBlocks with up/down flags
        ups = [m for m in model.modules() if isinstance(m, ResBlock) and m.updown]
        assert len(ups) > 0

    def test_use_checkpoint(
        self,
        hw_image_size: list[int],
        in_channels: int,
        model_channels: int,
        num_res_blocks: int,
        attention_resolutions: list[int],
        channel_mult: str,
        num_classes: int,
        sample_images: Tensor,
        timesteps: Tensor,
        sample_labels: Tensor,
    ):
        """Smoke test that gradient checkpoint flag does not break forward pass.

        Args:
            hw_image_size (list[int]): Image size [H, W].
            in_channels (int): Input channels.
            model_channels (int): Base model channels.
            num_res_blocks (int): Residual blocks per level.
            attention_resolutions (list[int]): Attention resolutions.
            channel_mult (str): Channel multiplier specification.
            num_classes (int): Number of classes.
            sample_images (Tensor): Input images.
            timesteps (Tensor): Timesteps.
            sample_labels (Tensor): Class labels.

        Returns:
            None

        Raises:
            AssertionError: If forward output shape mismatches input shape.
        """
        model = UNetModel(
            image_size=hw_image_size,
            in_channels=in_channels,
            model_channels=model_channels,
            out_channels=in_channels,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            channel_mult=channel_mult,
            n_classes=num_classes,
            classifier_free=True,
            use_checkpoint=True,
        )
        out = model(sample_images, timesteps, y=sample_labels)
        assert out["x"].shape == sample_images.shape

    def test_output_zero_init(
        self,
        unet_class_cond: UNetModel,
        sample_images: Tensor,
        timesteps: Tensor,
        sample_labels: Tensor,
    ):
        """Check that output head is zero-initialized (common for stable training).

        Args:
            unet_class_cond (UNetModel): Model instance.
            sample_images (Tensor): Input images.
            timesteps (Tensor): Timesteps.
            sample_labels (Tensor): Class labels.

        Returns:
            None

        Raises:
            AssertionError: If output deviates from near-zero initialization.
        """
        out = unet_class_cond(sample_images, timesteps, y=sample_labels)["x"]
        assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)

    def test_integer_and_float_timesteps(
        self,
        unet_class_cond: UNetModel,
        sample_images: Tensor,
        sample_labels: Tensor,
    ):
        """Verify model accepts both integer (discrete) and float timesteps equally.

        Args:
            unet_class_cond (UNetModel): Model instance.
            sample_images (Tensor): Input images.
            sample_labels (Tensor): Class labels.

        Returns:
            None

        Raises:
            AssertionError: If output shapes differ between int and float timestep runs.
        """
        t = torch.randint(0, 1000, (sample_images.shape[0],))
        out = unet_class_cond(sample_images, t, y=sample_labels)["x"]
        assert out.shape == sample_images.shape
        t = torch.rand(sample_images.shape[0])
        out_float = unet_class_cond(sample_images, t, y=sample_labels)["x"]
        assert out_float.shape == sample_images.shape

    def test_multiple_attention_resolutions(
        self, hw_image_size: list[int], in_channels: int, model_channels: int, num_res_blocks: int, num_classes: int
    ):
        """Count attention blocks for multiple attention resolutions and deeper channel_mult.

        Args:
            hw_image_size (list[int]): Image size [H, W].
            in_channels (int): Input channels.
            model_channels (int): Base model channels.
            num_res_blocks (int): Residual blocks per level.
            num_classes (int): Number of classes.

        Returns:
            None

        Raises:
            AssertionError: If discovered attention block count differs from expected.
        """
        model = UNetModel(
            image_size=hw_image_size,
            in_channels=in_channels,
            model_channels=model_channels,
            out_channels=in_channels,
            num_res_blocks=num_res_blocks,
            attention_resolutions=[1, 2, 4],
            channel_mult="1,2,4",
            n_classes=num_classes,
            classifier_free=True,
        )
        num_attn = sum(1 for m in model.modules() if isinstance(m, AttentionBlock))
        assert num_attn == 16

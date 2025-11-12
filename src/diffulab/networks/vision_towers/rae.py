from typing import TYPE_CHECKING, cast

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from jaxtyping import Float
from numpy.typing import NDArray
from PIL import Image
from torch import Tensor
from torch.utils.checkpoint import checkpoint  # type: ignore
from transformers import AutoImageProcessor, AutoModel

from diffulab.networks.utils.nn import QKNorm, RotaryPositionalEmbedding
from diffulab.networks.vision_towers.common import VisionTower

if TYPE_CHECKING:
    from transformers.feature_extraction_utils import BatchFeature
    from transformers.modeling_outputs import BaseModelOutputWithPooling
    from transformers.models.dinov3_vit import (
        DINOv3ViTImageProcessorFast,
        DINOv3ViTModel,
    )


class Attention(nn.Module):
    """
    Multi-head self-attention mechanism with rotary positional embeddings.

    Args:
        input_dim (int): Dimension of the input.
        dim (int): Inner Attention dimension.
        num_heads (int): Number of attention heads.
        partial_rotary_factor (float, optional): Factor for partial rotary embeddings. Default is 0.5.
        base (int, optional): Base for rotary positional embeddings. Default is 10000.

    Attributes:
        num_heads (int): Number of attention heads.
        head_dim (int): Dimension of each attention head.
        scale (float): Scaling factor for attention scores.
        qkv (nn.Linear): Linear layer for query, key, and value projection of input.
        qk_norm (QKNorm): Normalization layer for input query and key.
        partial_rotary_factor (float): Factor for partial rotary embeddings.
        rotary_dim (int): Dimension for rotary embeddings.
        rope (RotaryPositionalEmbedding): Rotary positional embedding layer.
        input_proj (nn.Linear): Linear layer for projecting the output of the input.

    Methods:
        forward(input: Tensor, context: Tensor) -> tuple[Tensor, Tensor]:
            Forward pass of the attention mechanism.

            Args:
                input (Tensor): Input tensor of shape (batch_size, seq_len, input_dim).

            Returns:
                Tensor: output tensor.
    """

    def __init__(
        self,
        input_dim: int,
        dim: int,
        num_heads: int,
        partial_rotary_factor: float = 1,
        base: int = 10000,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()  # type: ignore
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(input_dim, 3 * dim)
        self.qk_norm = QKNorm(dim)

        self.partial_rotary_factor = partial_rotary_factor
        self.rotary_dim = int(self.head_dim * self.partial_rotary_factor)
        self.rope = RotaryPositionalEmbedding(dim=self.rotary_dim, base=base)
        self.dropout = dropout

        self.proj_out = nn.Linear(dim, input_dim)

    def forward(self, input: Float[Tensor, "batch_size seq_len dim"]) -> Float[Tensor, "batch_size seq_len dim"]:
        input_q, input_k, input_v = self.qkv(input).chunk(3, dim=-1)
        input_q, input_k = self.qk_norm(input_q, input_k, input_v)

        q, k, v = (
            rearrange(input_q, "b n (h d) -> b n h d", h=self.num_heads),
            rearrange(input_k, "b n (h d) -> b n h d", h=self.num_heads),
            rearrange(input_v, "b n (h d) -> b n h d", h=self.num_heads),
        )
        # Remove CLS token for rotary embeddings
        q[:, 1:], k[:, 1:], v = self.rope(q=q[:, 1:], k=k[:, 1:], v=v)
        q, k, v = map(lambda x: rearrange(x, "b n h d -> b h n d"), [q, k, v])

        attn_output = nn.functional.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            scale=self.scale,
            dropout_p=self.dropout,
        )

        attn_output = rearrange(attn_output, "b h n d -> b n (h d)")

        output: Tensor = self.proj_out(attn_output)

        return output


class TransformerBlock(nn.Module):
    """
    Transformer block with attention mechanisms.

    Args:
        input_dim (int): Dimension of the input tensor.
        hidden_dim (int): Dimension of the hidden layer in the attention mechanism.
        num_heads (int): Number of attention heads.
        mlp_ratio (int): Ratio used to determine the size of the MLP layers.

    Methods:
        forward(input: Tensor, y: Tensor, context: Tensor) -> Tuple[Tensor, Tensor]:
            Performs the forward pass of the MMDiTBlock.
            Args:
                input (Tensor): The input tensor.
                context (Tensor): The context tensor.
            Returns:
                Tuple[Tensor, Tensor]: The processed input tensor.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: int,
        partial_rotary_factor: float = 1,
        use_checkpoint: bool = False,
        dropout_attn: float = 0,
        dropout_mlp: float = 0,
    ):
        super().__init__()  # type: ignore
        self.norm_1 = nn.RMSNorm(input_dim)
        self.attention = Attention(
            input_dim, hidden_dim, num_heads, partial_rotary_factor=partial_rotary_factor, dropout=dropout_attn
        )
        self.norm_2 = nn.RMSNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, mlp_ratio * input_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_ratio * input_dim, input_dim),
        )
        self.use_checkpoint = use_checkpoint
        self.dropout = nn.Dropout(dropout_mlp)

    def forward(
        self,
        input: Float[Tensor, "batch_size seq_len embedding_dim"],
    ) -> Float[Tensor, "batch_size seq_len input_dim"]:
        """
        Forward pass of the DiTBlock applying modulation and attention mechanisms.
        Args:
            - input (Tensor): The input tensor to be processed
            - y (Tensor): The conditioning tensor used for modulation
        Returns:
            Tensor: The processed input tensor with residual connection
        """
        return checkpoint(self._forward, *(input), use_reentrant=False) if self.use_checkpoint else self._forward(input)  # type: ignore

    def _forward(
        self,
        input: Float[Tensor, "batch_size seq_len embedding_dim"],
    ) -> Float[Tensor, "batch_size seq_len embedding_dim"]:
        input = input + self.attention(self.norm_1(input))
        input = input + self.dropout(self.mlp(self.norm_2(input)))
        return input


class RAEDecoder(nn.Module):
    """
    Transformer-based image decoder for ViT-style self-supervised encoders.

    This module reconstructs an image from a sequence of patch tokens produced by a
    frozen vision encoder (e.g., DINOv3 ViT). It first linearly projects encoder
    embeddings to the decoder width, prepends a learned [CLS] token, applies a stack
    of Transformer blocks with rotary positional embeddings, and finally maps each
    token to a patch vector that is unpatchified into the output image.

    Notes:
        The spatial size of the reconstructed image must be divisible by `patch_size`.

    Args:
        out_size (tuple[int, int]): Spatial size (H, W) of the reconstructed image.
        out_channels (int): Number of output channels (e.g., 3 for RGB).
        encoder_dim (int): Channel dimension of encoder tokens (input to the decoder).
        input_dim (int): Internal decoder width (embedding size after projection).
        hidden_dim (int): Attention inner dimension for multi-head attention.
        num_heads (int): Number of attention heads.
        mlp_ratio (int): Expansion ratio for the MLP inside each Transformer block.
        patch_size (int): Patch size used by the encoder; also drives unpatchify.
        depth (int): Number of Transformer blocks in the decoder.
        partial_rotary_factor (float): Fraction of head dimension using rotary embeddings.
        use_checkpoint (bool): Enable gradient checkpointing inside blocks.
        dropout_attn (float): Dropout applied in attention.
        dropout_mlp (float): Dropout applied in MLP.

    Attributes:
        input_proj (nn.Linear): Projects encoder_dim -> input_dim.
        layers (nn.ModuleList[TransformerBlock]): Transformer stack.
        last_layer (nn.Sequential): LayerNorm + Linear to patch vectors (p*p*C).
        cls_token (nn.Parameter): Learned [CLS] token prepended to the sequence.
        patch_size (int): Patch size used for unpatchify.
        out_size (tuple[int, int]): Target reconstruction size (H, W).
        out_channels (int): Number of output channels.

    Typical usage:
        z = encoder_tokens  # (B, N, encoder_dim)
        x_recon = decoder(z)  # (B, C, H, W)
    """

    def __init__(
        self,
        out_size: tuple[int, int] = (256, 256),
        out_channels: int = 3,
        encoder_dim: int = 768,
        input_dim: int = 1152,
        hidden_dim: int = 1152,
        num_heads: int = 16,
        mlp_ratio: int = 4,
        patch_size: int = 16,
        depth: int = 28,
        partial_rotary_factor: float = 1,
        use_checkpoint: bool = False,
        dropout_attn: float = 0,
        dropout_mlp: float = 0,
    ):
        super().__init__()  # type: ignore
        self.input_proj = nn.Linear(encoder_dim, input_dim)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    partial_rotary_factor=partial_rotary_factor,
                    use_checkpoint=use_checkpoint,
                    dropout_attn=dropout_attn,
                    dropout_mlp=dropout_mlp,
                )
                for _ in range(depth)
            ]
        )

        self.last_layer = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, patch_size * patch_size * out_channels),
        )
        self.apply(self._init_weights)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, input_dim))
        self.patch_size = patch_size
        self.out_size = out_size
        self.out_channels = out_channels
        self.encoder_dim = encoder_dim

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:  # type: ignore
                nn.init.constant_(module.bias, 0)

    def unpatchify(
        self, x: Float[Tensor, "batch_size seq_len patch_dim"]
    ) -> Float[Tensor, "batch_size channels height width"]:
        """
        Convert patches back into the original image tensor.
        Args:
            x (Tensor): Patchified tensor of shape (B, num_patches, patch_dim)
        Returns:
            Tensor: Reconstructed image tensor of shape (B, C, H, W)
        """
        H, W = self.out_size
        patch_size = self.patch_size
        p = self.out_channels

        # Calculate number of patches in height and width dimensions
        h = H // patch_size
        w = W // patch_size

        # Reshape the tensor to the original image dimensions
        x = rearrange(x, "b (h w) (p1 p2 c) -> b c (h p1) (w p2)", h=h, w=w, p1=patch_size, p2=patch_size, c=p)
        return x

    def forward(
        self,
        x: Float[Tensor, "batch_size seq_len encoder_dim"],
    ) -> Float[Tensor, "batch_size channels height width"]:
        x = self.input_proj(x)
        x = torch.cat([self.cls_token.repeat(x.size(0), 1, 1), x], dim=1)

        for layer in self.layers:
            x = layer(x)

        x = self.last_layer(x)
        x = self.unpatchify(x[:, 1:])  # remove cls token
        return x


class RAE(VisionTower):
    """
    Reconstruction Auto-Encoder built from a frozen DINOv3 ViT encoder and a
    Transformer decoder.

    The encoder is loaded from Hugging Face Transformers and kept frozen in eval
    mode. Inputs are preprocessed with the corresponding `AutoImageProcessor` and
    encoded into a sequence of patch embeddings. The decoder then reconstructs the
    image from those tokens.

    Pipeline:
        1) preprocess(x): Accepts a tensor in [0,1] or [0,255], converts to PIL,
           runs the DINOv3 image processor (no resize), and moves data to the
           encoder device.
        2) encode(x): Runs the ViT encoder and returns patch tokens, excluding the
           CLS token and any register tokens.
        3) decode(z): Uses `RAEDecoder` to reconstruct the image from tokens.
        4) forward(x): Convenience wrapper calling encode -> decode.

    Args:
        decoder (RAEDecoder | None): Optional decoder instance. If None,
            `decoder_config` must be provided to construct one.
        decoder_config (dict[str, Any] | None): Keyword args for `RAEDecoder`
            when `decoder` is not provided.
        dinov3_id (str): Model identifier for the DINOv3 ViT on Hugging Face
            (e.g., "facebook/dinov3-vitb16-pretrain-lvd1689m").

    Properties:
        latent_channels (int): Encoder hidden size (channel dimension of tokens).
        patch_size (int): Patch size used by the encoder.

    Methods:
        preprocess(x): Prepares inputs for the encoder. Raises ValueError if the
            value range is not [0,1] or [0,255].
        encode(x): Returns token embeddings with CLS and register tokens removed.
        decode(z): Reconstructs an image from token embeddings.
        forward(x): End-to-end reconstruction.

    Notes:
        - The encoder parameters are frozen and set to eval mode.
        - Device placement uses `device_map="auto"` when loading the encoder.
        - The decoder output spatial size is controlled by `decoder.out_size`.
    """

    def __init__(
        self,
        decoder: RAEDecoder,
        dinov3_id: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
        load_encoder: bool = True,
        encoder_patch_size: int | None = None,
    ) -> None:
        super().__init__()  # type: ignore
        assert load_encoder or encoder_patch_size, "If not loading the encoder, must provide encoder_patch_size"
        self.load_encoder = load_encoder
        self._patch_size = encoder_patch_size
        self._latent_channels = decoder.encoder_dim
        if load_encoder:
            self.processor = cast("DINOv3ViTImageProcessorFast", AutoImageProcessor.from_pretrained(dinov3_id))  # type: ignore[reportUnknownMemberType]
            self.encoder = cast(
                "DINOv3ViTModel",
                AutoModel.from_pretrained(  # type: ignore[reportUnknownMemberType]
                    dinov3_id,
                    device_map="auto",
                ),
            )
            norm = self.encoder.norm
            norm.register_parameter("weight", None)
            norm.register_parameter("bias", None)
            norm.elementwise_affine = False
            self.encoder.eval()
            for param in self.encoder.parameters():
                param.requires_grad = False

            assert self._latent_channels == self.encoder.config.hidden_size
            self._patch_size = self.encoder.config.patch_size

        self.decoder = decoder

    @property
    def latent_channels(self) -> int:
        """
        Number of channels in the latent space.
        This should be implemented in subclasses to return the specific number of latent channels.
        """
        assert self._latent_channels is not None, "latent_channels is not set"
        return self._latent_channels

    @property
    def patch_size(self) -> int:
        """
        Patch size of the encoder.
        """
        assert self._patch_size is not None, "patch_size is not set"
        return self._patch_size

    def preprocess(self, x: Float[Tensor, "batch_size channels height width"]) -> "BatchFeature":
        """
        Preprocess the input tensor before encoding.

        Args:
            x (Tensor): Input tensor to be preprocessed.
            Assumed to be an image tensor with shape [N, C, H, W].

        Returns:
            BatchFeature: Preprocessed input tensor.
        """
        # ensure float
        x = x.float()

        # detect range once (cheaper than calling .min()/.max() repeatedly)
        x_min = x.min().item()
        x_max = x.max().item()

        # convert to [0, 255]
        if x_min >= 0.0 and x_max <= 1.0:
            x = x * 255.0
        elif x_min >= 0.0 and x_max <= 255.0 and x_max > 1.0:
            pass
        else:
            raise ValueError("Input tensor range is not supported. Expected 0–255 or 0–1")

        x = x.clamp(0.0, 255.0)

        # convert to PIL Images
        x_cpu = x.detach().to("cpu").round().byte()
        imgs: list[Image.Image] = []
        for xi in x_cpu:
            arr = cast(NDArray[np.uint8], xi.permute(1, 2, 0).contiguous().numpy())  # type: ignore[reportUnknownMemberType]
            imgs.append(Image.fromarray(arr))

        processed_imgs = self.processor(images=imgs, return_tensors="pt", do_resize=False).to(self.encoder.device)  # type: ignore[reportUnknownMemberType]
        return processed_imgs

    @torch.inference_mode()
    def encode(self, x: Float[Tensor, "batch_size channels height width"]) -> Float[Tensor, "batch_size seq_len dim"]:
        """
        Forward pass of the encoder.
        Args:
            x (Tensor): Input tensor to be encoded.
        Returns:
            Tensor: Encoded representation of the input tensor.
        """
        assert self.load_encoder, "Encoder must be loaded to use encode()"
        x_processed = self.preprocess(x)
        with torch.no_grad():
            outputs: "BaseModelOutputWithPooling" = self.encoder(**x_processed)
            last_hidden_states = cast(Tensor, outputs.last_hidden_state)
            z = last_hidden_states[:, 1 + self.encoder.config.num_register_tokens :, :]
        return z

    def decode(
        self, z: Float[Tensor, "batch_size seq_len dim"]
    ) -> Float[Tensor, "batch_size channels height_d width_d"]:
        """
        Forward pass of the decoder.
        Args:
            z (Tensor): Latent tensor to be decoded.
        Returns:
            Tensor: Decoded representation of the latent tensor.
        """
        x_recon = self.decoder(z)
        return x_recon

    def forward(
        self,
        x: Float[Tensor, "batch_size channels height width"],
    ) -> Float[Tensor, "batch_size channels height_d width_d"]:
        """
        Forward pass of the RAE.

        Args:
            x (Tensor): Input tensor to be encoded and decoded.

        Returns:
            Tensor: Output tensor after encoding and decoding.
        """
        assert self.load_encoder, "Encoder must be loaded to use forward(), use decode() directly instead."
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon

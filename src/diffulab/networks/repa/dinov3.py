from typing import TYPE_CHECKING, cast

import numpy as np
import torch
from jaxtyping import Float
from numpy.typing import NDArray
from PIL import Image
from torch import Tensor
from transformers import AutoImageProcessor, AutoModel

from diffulab.networks.repa.common import REPA

if TYPE_CHECKING:
    from transformers.feature_extraction_utils import BatchFeature
    from transformers.modeling_outputs import BaseModelOutputWithPooling
    from transformers.models.dinov3_vit import (
        DINOv3ViTImageProcessorFast,
        DINOv3ViTModel,
    )


class DinoV3(REPA):
    def __init__(
        self,
        dino_model: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
        cancel_affine: bool = False,
        size: tuple[int, int] = (256, 256),
    ) -> None:
        super().__init__()
        self.processor = cast("DINOv3ViTImageProcessorFast", AutoImageProcessor.from_pretrained(dino_model))  # type: ignore[reportUnknownMemberType]
        self._encoder = cast(
            "DINOv3ViTModel",
            AutoModel.from_pretrained(  # type: ignore[reportUnknownMemberType]
                dino_model,
                device_map="auto",
            ),
        )
        self.size = size
        if cancel_affine:
            norm = self.encoder.norm
            norm.register_parameter("weight", None)
            norm.register_parameter("bias", None)
            norm.elementwise_affine = False
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False

    @property
    def encoder(self) -> "DINOv3ViTModel":
        """
        The encoder module that processes the input tensor.
        Returns:
            DINOv3ViTModel: The DINO V3 model.
        """
        return self._encoder

    @property
    def embedding_dim(self) -> int:
        """
        The dimension of the encoded representation.
        Returns:
            int: The embedding dimension of the DINO V2 encoder.
        """
        return self.encoder.config.hidden_size

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

        processed_imgs = self.processor(images=imgs, return_tensors="pt", size=self.size).to(self.encoder.device)  # type: ignore[reportUnknownMemberType]
        return processed_imgs

    @torch.inference_mode()
    def forward(self, x: Float[Tensor, "batch_size channels height width"]) -> Float[Tensor, "batch_size seq_len dim"]:
        """
        Forward pass of the encoder.
        Args:
            x (Tensor): Input tensor to be encoded.
        Returns:
            Tensor: Encoded representation of the input tensor.
        """
        x_processed = self.preprocess(x)
        with torch.no_grad():
            outputs: "BaseModelOutputWithPooling" = self.encoder(**x_processed)
            last_hidden_states = cast(Tensor, outputs.last_hidden_state)
            z = last_hidden_states[:, 1 + self.encoder.config.num_register_tokens :, :]
        return z

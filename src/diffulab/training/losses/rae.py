# loss borrowed directly from the official RAE implementation:
# https://github.com/bytetriper/RAE/blob/main/src/disc/lpips.py
# MiT license as of the 2 of November 2025

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal, NamedTuple

import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models  # type: ignore
from tqdm import tqdm

from diffulab.training.losses.common import LossFunction

# -------------------------------------------------
# LPIPS loss
# -------------------------------------------------


class ScalingLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()  # type: ignore
        self.register_buffer(name="shift", tensor=torch.tensor([-0.030, -0.088, -0.188])[None, :, None, None])
        self.register_buffer(name="scale", tensor=torch.tensor([0.458, 0.448, 0.450])[None, :, None, None])

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return (tensor - self.shift) / self.scale  # type: ignore


class NetLinLayer(nn.Module):
    def __init__(self, chn_in: int, chn_out: int = 1, use_dropout: bool = False) -> None:
        super().__init__()  # type: ignore
        layers: list[nn.Module] = [nn.Dropout()] if use_dropout else []
        layers.append(nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False))
        self.model = nn.Sequential(*layers)


class VGG16FeatureExtractor(nn.Module):
    def __init__(self, requires_grad: bool = False, pretrained: bool = True) -> None:
        super().__init__()  # type: ignore
        features = models.vgg16(pretrained=pretrained).features
        self.slice1 = nn.Sequential(*[features[x] for x in range(4)])  # type: ignore
        self.slice2 = nn.Sequential(*[features[x] for x in range(4, 9)])  # type: ignore
        self.slice3 = nn.Sequential(*[features[x] for x in range(9, 16)])  # type: ignore
        self.slice4 = nn.Sequential(*[features[x] for x in range(16, 23)])  # type: ignore
        self.slice5 = nn.Sequential(*[features[x] for x in range(23, 30)])  # type: ignore
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, tensor: torch.Tensor):
        h = self.slice1(tensor)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        VggOutputs = NamedTuple(
            "VggOutputs",
            [
                ("relu1_2", torch.Tensor),
                ("relu2_2", torch.Tensor),
                ("relu3_3", torch.Tensor),
                ("relu4_3", torch.Tensor),
                ("relu5_3", torch.Tensor),
            ],
        )
        return VggOutputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)


def _normalize(tensor: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    norm_factor = torch.sqrt(torch.sum(tensor**2, dim=1, keepdim=True))
    return tensor / (norm_factor + eps)


def _spatial_average(tensor: torch.Tensor, keepdim: bool = True) -> torch.Tensor:
    return tensor.mean([2, 3], keepdim=keepdim)


class LPIPS(LossFunction):
    """Learned perceptual metric used by VQGAN."""

    URL_MAP: dict[str, str] = {"vgg_lpips": "https://heibox.uni-heidelberg.de/f/607503859c864bc1b30b/?dl=1"}

    CKPT_MAP: dict[str, str] = {"vgg_lpips": "vgg.pth"}

    def __init__(self, use_dropout: bool = True, root: str = Path.home().as_posix()) -> None:
        super().__init__()
        self.scaling_layer = ScalingLayer()
        self.chns = [64, 128, 256, 512, 512]
        self.net = VGG16FeatureExtractor(pretrained=True, requires_grad=False)
        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
        self._load_pretrained_weights(root=root)
        for param in self.parameters():
            param.requires_grad = False

    def _download(self, url: str, local_path: str | Path, chunk_size: int = 1024):
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        with requests.get(url, stream=True) as r:
            total_size = int(r.headers.get("content-length", 0))
            with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
                with open(local_path, "wb") as f:
                    for data in r.iter_content(chunk_size=chunk_size):
                        if data:
                            f.write(data)
                            pbar.update(chunk_size)

    def _get_ckpt_path(self, root: str, name: str, check: bool = False):
        assert name in self.URL_MAP
        path = Path(root) / ".cache/diffulab" / self.CKPT_MAP[name]
        if not path.exists():
            print("Downloading {} model from {} to {}".format(name, self.URL_MAP[name], path))
            self._download(self.URL_MAP[name], path)
            with open(path, "rb") as f:
                content = f.read()
                assert hashlib.md5(content).hexdigest() == "d507d7349b931f0638a25a48a722f98a"
        return path

    def _load_pretrained_weights(self, root: str, name: str = "vgg_lpips") -> None:
        ckpt = self._get_ckpt_path(root, name)
        state = torch.load(ckpt, map_location=torch.device("cpu"))
        self.load_state_dict(state, strict=False)
        print(f"[LPIPS] Loaded pretrained weights from {ckpt}")

    def forward(self, input: torch.Tensor, target: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
        input_scaled, target_scaled = self.scaling_layer(input), self.scaling_layer(target)
        feats_input = self.net(input_scaled)
        feats_target = self.net(target_scaled)
        diffs: list[torch.Tensor] = []
        lin_layers = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        for idx, (feat_in, feat_tgt) in enumerate(zip(feats_input, feats_target)):
            feat_in = _normalize(feat_in)
            feat_tgt = _normalize(feat_tgt)
            diff = (feat_in - feat_tgt) ** 2
            diffs.append(_spatial_average(lin_layers[idx].model(diff), keepdim=True))

        value: torch.Tensor = sum(diffs)  # type: ignore

        if reduction == "none":
            return value
        if reduction == "sum":
            return value.sum()
        if reduction == "mean":
            return value.mean()
        raise ValueError(f"Unsupported reduction '{reduction}'")


# -------------------------------------------------
# GAN loss
# -------------------------------------------------


class GANLoss(LossFunction):
    def __init__(self, gan_type: Literal["vanilla", "hinge"] = "vanilla") -> None:
        super().__init__()
        assert gan_type in ["vanilla", "hinge"], f"Unsupported GAN loss type '{gan_type}'"
        self.gan_type = gan_type

    def compute_d_loss(
        self, logits_real: torch.Tensor, logits_fake: torch.Tensor, reduction: str = "mean"
    ) -> torch.Tensor:
        reduce = torch.mean if reduction == "mean" else torch.sum
        if self.gan_type == "hinge":
            loss_real = reduce(F.relu(torch.ones_like(logits_real) - logits_real))
            loss_fake = reduce(F.relu(torch.ones_like(logits_fake) + logits_fake))
            return 0.5 * (loss_real + loss_fake)
        return 0.5 * (reduce(F.softplus(-logits_real)) + reduce(F.softplus(logits_fake)))

    def compute_g_loss(self, logits_fake: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
        reduce = torch.mean if reduction == "mean" else torch.sum
        if self.gan_type == "hinge":
            return -reduce(logits_fake)
        return reduce(F.softplus(-logits_fake))

    def forward(
        self,
        logits_fake: torch.Tensor,
        logits_real: torch.Tensor | None = None,
        is_disc: bool = False,
        reduction: str = "mean",
    ) -> torch.Tensor:
        if is_disc:
            assert logits_real is not None, "logits_real must be provided when computing discriminator loss"
            return self.compute_d_loss(logits_real, logits_fake, reduction=reduction)
        return self.compute_g_loss(logits_fake, reduction=reduction)

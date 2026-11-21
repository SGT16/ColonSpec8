from __future__ import annotations

import hashlib
from collections.abc import Mapping

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from fewlabel.models import _groups
from fewlabel_v21.models import _double_conv
from fewlabel_v21.deterministic_ops import resize_bilinear

ARMS = ("learned", "random", "pca8")
PROJECTION = "local_projection.0.weight"
ARCHITECTURE = "borkovits814_local_unet_image_bag_v1"
EXPECTED_PARAMETERS = 83010
EXPECTED_FIXED_TRAINABLE = 76498


def state_digest(state: Mapping[str, Tensor], *, shared: bool = False) -> str:
    digest = hashlib.sha256()
    for name, value in state.items():
        if shared and name.startswith("local_projection.0."):
            continue
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class FixedPCA(nn.Module):


    def __init__(self, components, mean):
        super().__init__()
        components = np.asarray(components, dtype=np.float32)
        mean = np.asarray(mean, dtype=np.float32)
        if components.shape != (8, 814) or mean.shape != (814,):
            raise ValueError("PCA requires components[8,814] and mean[814]")
        if not np.isfinite(components).all() or not np.isfinite(mean).all():
            raise ValueError("PCA contains nonfinite values")
        if not np.allclose(components @ components.T, np.eye(8), atol=2e-5, rtol=2e-5):
            raise ValueError("PCA components must be orthonormal without whitening")
        self.weight = nn.Parameter(torch.from_numpy(components.copy())[:, :, None, None],
                                   requires_grad=False)
        self.register_buffer("pca_bias", torch.from_numpy((-components @ mean).copy()))

    def forward(self, inputs: Tensor) -> Tensor:
        return F.conv2d(inputs, self.weight, self.pca_bias)


class LocalImageClassifier(nn.Module):


    def __init__(self):
        super().__init__()
        self.bands, self.patch_size, self.classes = 814, 64, 2
        self.architecture = ARCHITECTURE
        self.local_projection = nn.Sequential(
            nn.Conv2d(814, 8, 1, bias=False), nn.GroupNorm(_groups(8), 8), nn.GELU())
        self.local_encoder = _double_conv(8, 8)
        self.local_down = nn.ModuleList([_double_conv(8, 16), _double_conv(16, 24),
                                         _double_conv(24, 32)])
        self.bottleneck_fusion = _double_conv(32, 32)
        self.local_up = nn.ModuleList([_double_conv(56, 24), _double_conv(40, 16),
                                       _double_conv(24, 8)])
        self.feature_projection = nn.Sequential(
            nn.Conv2d(8, 32, 1, bias=False), nn.GroupNorm(_groups(32), 32), nn.GELU())
        self.classifier = nn.Conv2d(32, 2, 1)

    def patch_logits(self, inputs: Tensor) -> Tensor:

        if inputs.ndim != 4 or tuple(inputs.shape[1:]) != (814, 64, 64):
            raise ValueError("Expected patch tensor [N,814,64,64]")
        skips = [self.local_encoder(self.local_projection(inputs))]
        for block in self.local_down:
            skips.append(block(F.max_pool2d(skips[-1], 2)))
        features = self.bottleneck_fusion(skips[-1])
        for block, skip in zip(self.local_up, reversed(skips[:-1])):
            features = block(torch.cat((resize_bilinear(features, tuple(skip.shape[-2:])), skip), dim=1))
        spatial_logits = self.classifier(self.feature_projection(features))
        return spatial_logits.mean(dim=(-2, -1))

    def forward(self, bags: Tensor) -> Tensor:

        if bags.ndim != 5 or bags.shape[0] < 1 or bags.shape[1] < 1:
            raise ValueError("Expected nonempty image bags [B,K,814,64,64]")
        batch, patches = bags.shape[:2]
        logits = self.patch_logits(bags.flatten(0, 1))
        return logits.reshape(batch, patches, 2).mean(dim=1)


def build_model(arm: str, init_seed: int, pca=None) -> LocalImageClassifier:
    if arm not in ARMS:
        raise ValueError("Unknown Borkovits arm")
    if isinstance(init_seed, bool) or not isinstance(init_seed, int) or not 100 <= init_seed <= 999:
        raise ValueError("A persisted three-digit initialization seed is required")

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(init_seed)
        model = LocalImageClassifier()
    model.parent_initial_sha256 = state_digest(model.state_dict())
    model.shared_initial_sha256 = state_digest(model.state_dict(), shared=True)
    if arm == "random":
        model.local_projection[0].weight.requires_grad_(False)
    elif arm == "pca8":
        if pca is None:
            raise ValueError("TRAIN-only PCA basis is required")
        model.local_projection[0] = FixedPCA(pca["components"], pca["mean"])
    model.arm, model.init_seed = arm, init_seed
    model.initial_sha256 = state_digest(model.state_dict())
    if state_digest(model.state_dict(), shared=True) != model.shared_initial_sha256:
        raise RuntimeError("Projection replacement changed shared initialization")
    if sum(parameter.numel() for parameter in model.parameters()) != EXPECTED_PARAMETERS:
        raise RuntimeError("Borkovits local classifier parameter count changed")
    expected_trainable = EXPECTED_PARAMETERS if arm == "learned" else EXPECTED_FIXED_TRAINABLE
    if sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad) != expected_trainable:
        raise RuntimeError("Borkovits trainable parameter count changed")
    return model

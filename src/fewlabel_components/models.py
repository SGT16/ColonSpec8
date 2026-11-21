from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from fewlabel_multiscale.models import LOCAL_WIDTHS
from fewlabel_nomamba.models import MultiscaleNoMambaSegmenter


LOCAL_ARCHITECTURE = "local_only_v1"
PARENT_ARCHITECTURE = "multiscale_no_mamba_v1"
EXPECTED_PARAMETERS = {"local_only_mt": 79306, "nomamba_supervised": 99250}
VARIANTS = {"local_only_mt": "local_only_mt",
            "nomamba_supervised": "multiscale_no_mamba_supervised"}
REMOVED_BRANCHES = ("stem", "spectral_projection", "spectral_readout",
                    "spatial_projection", "fusion", "skip_projection")
SLICED_INPUTS = {"bottleneck_fusion.0.weight": 32,
                 "local_up.1.0.weight": 40}


def _retain_input_prefix(layer: nn.Conv2d, channels: int) -> nn.Conv2d:

    if layer.groups != 1 or not 0 < channels < layer.in_channels:
        raise ValueError("Only the authorized ungrouped input-channel removal is supported")
    result = copy.deepcopy(layer)
    result.in_channels = channels
    result.weight = nn.Parameter(layer.weight[:, :channels].detach().clone(),
                                 requires_grad=layer.weight.requires_grad)
    return result


class LocalOnlySegmenter(MultiscaleNoMambaSegmenter):


    def __init__(self, cfg: Mapping[str, Any]):
        super().__init__(cfg)
        for name in REMOVED_BRANCHES:
            delattr(self, name)


        self.bottleneck_fusion[0] = _retain_input_prefix(self.bottleneck_fusion[0], 32)
        self.local_up[1][0] = _retain_input_prefix(self.local_up[1][0], 40)
        self.architecture = LOCAL_ARCHITECTURE

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        if x.device.type != "cuda":
            raise RuntimeError("The explicit local-only control is configured for CUDA training")
        expected = (self.bands, self.patch_size, self.patch_size)
        if x.ndim != 4 or tuple(x.shape[1:]) != expected:
            raise ValueError(f"Expected [B,{expected[0]},{expected[1]},{expected[2]}], "
                             f"received {tuple(x.shape)}")
        local_skips = [self.local_encoder(self.local_projection(x))]
        for block in self.local_down:
            local_skips.append(block(F.max_pool2d(local_skips[-1], 2)))
        features = self.bottleneck_fusion(local_skips[-1])
        for block, skip in zip(self.local_up, reversed(local_skips[:-1])):
            upsampled = F.interpolate(features, size=skip.shape[-2:],
                                      mode="bilinear", align_corners=False)
            features = block(torch.cat((upsampled, skip), dim=1))
        features = self.feature_projection(features)
        return {"logits": self.classifier(features), "features": features}


def build_model(name: str, cfg: Mapping[str, Any]) -> nn.Module:

    if name not in EXPECTED_PARAMETERS:
        raise ValueError("Only local_only_mt and nomamba_supervised are authorized here")
    architecture = LOCAL_ARCHITECTURE if name == "local_only_mt" else PARENT_ARCHITECTURE
    values: dict[str, Any] = {
        "input_bands": 351, "patch_size": 64, "stem_channels": 8,
        "spatial_grid": 8, "d_model": 32, "d_state": 16, "d_conv": 4,
        "expand": 2, "checkpoint_spectral": True,
        "architecture": architecture, "multiscale_widths": list(LOCAL_WIDTHS),
    }
    if "model" in cfg:
        values["input_bands"] = cfg.get("data", {}).get("expected_bands", 351)
        values.update(cfg["model"])
        expected_lambda_u = 1.0 if name == "local_only_mt" else 0.0
        if cfg.get("ssl", {}).get("lambda_u_max") != expected_lambda_u:
            raise ValueError("Control identity and consistency-loss coefficient disagree")
    else:
        if name == "nomamba_supervised":
            raise ValueError("The supervised control requires the full config proving lambda_u_max=0")
        values.update(cfg)
    expected = {
        "input_bands": 351, "patch_size": 64, "stem_channels": 8,
        "spatial_grid": 8, "d_model": 32, "d_state": 16, "d_conv": 4,
        "expand": 2, "checkpoint_spectral": True, "architecture": architecture,
    }
    if any(values.get(key) != required for key, required in expected.items()):
        raise ValueError("The authorized component-control architecture changed")
    if tuple(values["multiscale_widths"]) != LOCAL_WIDTHS:
        raise ValueError("The authorized local widths changed")
    if values.get("allow_fallback", False):
        raise ValueError("Automatic model fallback is forbidden")
    if values.get("mamba_class", "mamba_ssm.Mamba") != "mamba_ssm.Mamba":
        raise ValueError("The frozen initialization constructor may not be substituted")
    model = LocalOnlySegmenter(values) if name == "local_only_mt" else MultiscaleNoMambaSegmenter(values)
    if sum(parameter.numel() for parameter in model.parameters()) != EXPECTED_PARAMETERS[name]:
        raise RuntimeError("Component-control parameter count changed")
    return model

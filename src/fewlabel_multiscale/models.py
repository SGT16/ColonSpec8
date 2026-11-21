from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from fewlabel.models import _groups, restore_spatial_sequences, spatial_sequences
from fewlabel_v21.models import HighResolutionMambaSegmenter, _double_conv


ARCHITECTURE = "multiscale_mamba_v1"
LOCAL_WIDTHS = (8, 16, 24, 32)


class MultiscaleMambaSegmenter(HighResolutionMambaSegmenter):


    def __init__(self, cfg: Mapping[str, Any]):
        super().__init__(cfg)
        self.architecture = ARCHITECTURE
        self.local_widths = tuple(int(value) for value in cfg["multiscale_widths"])
        if self.local_widths != LOCAL_WIDTHS:
            raise ValueError("Only the authorized local widths 8/16/24/32 are supported")


        del self.decoder
        del self.skip_fusion
        width0, width1, width2, width3 = self.local_widths
        self.local_projection = nn.Sequential(
            nn.Conv2d(self.bands, width0, 1, bias=False),
            nn.GroupNorm(_groups(width0), width0),
            nn.GELU(),
        )
        self.local_encoder = _double_conv(width0, width0)
        self.local_down = nn.ModuleList(
            _double_conv(previous, current)
            for previous, current in zip(self.local_widths[:-1], self.local_widths[1:])
        )
        self.bottleneck_fusion = _double_conv(width3 + self.channels, width3)
        self.local_up = nn.ModuleList([
            _double_conv(width3 + width2, width2),

            _double_conv(width2 + width1 + self.channels, width1),
            _double_conv(width1 + width0, width0),
        ])
        self.feature_projection = nn.Sequential(
            nn.Conv2d(width0, self.channels, 1, bias=False),
            nn.GroupNorm(_groups(self.channels), self.channels),
            nn.GELU(),
        )


    def forward(self, x: Tensor) -> dict[str, Tensor]:
        if x.device.type != "cuda":
            raise RuntimeError("Multiscale Mamba requires real CUDA selective-scan kernels")
        sequence, spatial, stem_skip = self._stem_sequences_and_skip(x)
        if self.checkpoint_spectral and self.training and torch.is_grad_enabled():
            sequence = checkpoint(self._spectral_branch, sequence, use_reentrant=False)
        else:
            sequence = self._spectral_branch(sequence)
        scanned = self.spatial_mamba(torch.cat(spatial_sequences(spatial), dim=0))
        restored = restore_spatial_sequences(
            tuple(scanned.chunk(4, dim=0)), self.grid, self.grid
        )
        mamba_context = self.fusion(torch.cat(
            (self._spectral_map(sequence, x.shape[0]), restored), dim=1
        ))

        local_skips = [self.local_encoder(self.local_projection(x))]
        for block in self.local_down:
            local_skips.append(block(F.max_pool2d(local_skips[-1], 2)))
        features = self.bottleneck_fusion(torch.cat(
            (local_skips[-1], mamba_context), dim=1
        ))
        for index, (block, skip) in enumerate(zip(self.local_up, reversed(local_skips[:-1]))):
            parts = [F.interpolate(features, size=skip.shape[-2:],
                                   mode="bilinear", align_corners=False), skip]
            if index == 1:
                if stem_skip.shape[-2:] != skip.shape[-2:]:
                    raise RuntimeError("The retained stem skip must align with the 32x32 local scale")
                parts.append(stem_skip)
            features = block(torch.cat(parts, dim=1))
        features = self.feature_projection(features)
        return {"logits": self.classifier(features), "features": features}


def build_model(name: str, cfg: Mapping[str, Any]) -> nn.Module:

    if name != "mamba_mt":
        raise ValueError("This pilot supports only internal model name mamba_mt")
    values: dict[str, Any] = {
        "input_bands": 351, "patch_size": 64, "stem_channels": 8,
        "spatial_grid": 8, "d_model": 32, "d_state": 16, "d_conv": 4,
        "expand": 2, "checkpoint_spectral": True,
        "architecture": ARCHITECTURE, "multiscale_widths": list(LOCAL_WIDTHS),
    }
    if "model" in cfg:
        values["input_bands"] = cfg.get("data", {}).get("expected_bands", 351)
        values.update(cfg["model"])
    else:
        values.update(cfg)
    expected = {
        "input_bands": 351, "patch_size": 64, "stem_channels": 8,
        "spatial_grid": 8, "d_model": 32, "d_state": 16, "d_conv": 4,
        "expand": 2, "checkpoint_spectral": True, "architecture": ARCHITECTURE,
    }
    if any(values.get(key) != required for key, required in expected.items()):
        raise ValueError("The authorized multiscale Mamba architecture changed")
    if tuple(values["multiscale_widths"]) != LOCAL_WIDTHS:
        raise ValueError("The authorized multiscale Mamba widths changed")
    if values.get("allow_fallback", False):
        raise ValueError("Fallback is forbidden by the protocol")
    if values.get("mamba_class", "mamba_ssm.Mamba") != "mamba_ssm.Mamba":
        raise ValueError("Only the real mamba_ssm.Mamba implementation is authorized")
    return MultiscaleMambaSegmenter(values)

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from fewlabel_multiscale.models import LOCAL_WIDTHS, MultiscaleMambaSegmenter


ARCHITECTURE = "multiscale_no_mamba_v1"
EXPECTED_PARAMETERS = 99250


class MultiscaleNoMambaSegmenter(MultiscaleMambaSegmenter):


    def __init__(self, cfg: Mapping[str, Any]):


        super().__init__(cfg)
        del self.spectral_mamba
        del self.spatial_mamba
        self.architecture = ARCHITECTURE
        self.mamba_enabled = False

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        if x.device.type != "cuda":
            raise RuntimeError("The explicit no-Mamba control is configured for CUDA training")
        sequence, spatial, stem_skip = self._stem_sequences_and_skip(x)


        context = self.fusion(torch.cat(
            (self._spectral_map(sequence, x.shape[0]), spatial), dim=1
        ))
        local_skips = [self.local_encoder(self.local_projection(x))]
        for block in self.local_down:
            local_skips.append(block(F.max_pool2d(local_skips[-1], 2)))
        features = self.bottleneck_fusion(torch.cat(
            (local_skips[-1], context), dim=1
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

    if name != "cnn_mt":
        raise ValueError("The explicit no-Mamba pilot supports only internal model name cnn_mt")
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
        raise ValueError("The authorized no-Mamba architecture changed")
    if tuple(values["multiscale_widths"]) != LOCAL_WIDTHS:
        raise ValueError("The authorized multiscale widths changed")
    if values.get("allow_fallback", False):
        raise ValueError("Automatic fallback is forbidden; this control requires its own explicit identity")


    if values.get("mamba_class", "mamba_ssm.Mamba") != "mamba_ssm.Mamba":
        raise ValueError("The frozen initialization constructor may not be substituted")
    return MultiscaleNoMambaSegmenter(values)

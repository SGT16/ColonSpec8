from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from fewlabel.models import (
    CNNSegmenter,
    MambaSegmenter,
    _groups,
    restore_spatial_sequences,
    spatial_sequences,
)


class _HighResolutionDecoder:


    def _init_high_resolution_decoder(self, cfg: Mapping[str, Any]) -> None:
        stem_channels = int(cfg["stem_channels"])
        self.skip_projection = nn.Sequential(
            nn.Conv2d(2 * stem_channels, self.channels, 1, bias=False),
            nn.GroupNorm(_groups(self.channels), self.channels),
            nn.GELU(),
        )
        self.skip_fusion = nn.Sequential(
            nn.Conv2d(2 * self.channels, self.channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(self.channels), self.channels),
            nn.GELU(),
        )

    def _stem_sequences_and_skip(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        expected = (self.bands, self.patch_size, self.patch_size)
        if x.ndim != 4 or tuple(x.shape[1:]) != expected:
            raise ValueError(f"Expected [B,{expected[0]},{expected[1]},{expected[2]}], "
                             f"received {tuple(x.shape)}")
        encoded = self.stem(x.unsqueeze(1))


        skip = self.skip_projection(torch.cat(
            (encoded.mean(dim=2), encoded.amax(dim=2)), dim=1
        ))
        pooled = F.adaptive_avg_pool3d(encoded, (self.bands, self.grid, self.grid))
        b, stem_channels, bands, height, width = pooled.shape
        sequence = pooled.permute(0, 3, 4, 2, 1).reshape(
            b * height * width, bands, stem_channels
        )
        sequence = self.spectral_projection(sequence)
        spatial = self.spatial_projection(torch.cat(
            (pooled.mean(dim=2), pooled.amax(dim=2)), dim=1
        ))
        return sequence, spatial, skip

    def _decode_high_resolution(
        self, spectral: Tensor, spatial: Tensor, skip: Tensor
    ) -> dict[str, Tensor]:
        merged = self.fusion(torch.cat((spectral, spatial), dim=1))
        context = F.interpolate(merged, size=skip.shape[-2:],
                                mode="bilinear", align_corners=False)
        detail = self.skip_fusion(torch.cat((context, skip), dim=1))
        features = self.decoder(F.interpolate(
            detail, size=(self.patch_size, self.patch_size),
            mode="bilinear", align_corners=False,
        ))
        return {"logits": self.classifier(features), "features": features}


class HighResolutionCNNSegmenter(_HighResolutionDecoder, CNNSegmenter):
    def __init__(self, cfg: Mapping[str, Any]):
        super().__init__(cfg)
        self._init_high_resolution_decoder(cfg)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        sequence, spatial, skip = self._stem_sequences_and_skip(x)
        sequence = sequence + self.spectral_convs(sequence.transpose(1, 2)).transpose(1, 2)
        spatial = spatial + self.spatial_convs(spatial)
        return self._decode_high_resolution(
            self._spectral_map(sequence, x.shape[0]), spatial, skip
        )


class HighResolutionMambaSegmenter(_HighResolutionDecoder, MambaSegmenter):
    def __init__(self, cfg: Mapping[str, Any]):
        super().__init__(cfg)
        self._init_high_resolution_decoder(cfg)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        if x.device.type != "cuda":
            raise RuntimeError("Mamba v2.1 requires CUDA with the real selective-scan kernel")
        sequence, spatial, skip = self._stem_sequences_and_skip(x)
        if self.checkpoint_spectral and self.training and torch.is_grad_enabled():
            sequence = checkpoint(self._spectral_branch, sequence, use_reentrant=False)
        else:
            sequence = self._spectral_branch(sequence)
        scanned = self.spatial_mamba(torch.cat(spatial_sequences(spatial), dim=0))
        restored = restore_spatial_sequences(
            tuple(scanned.chunk(4, dim=0)), self.grid, self.grid
        )
        return self._decode_high_resolution(
            self._spectral_map(sequence, x.shape[0]), restored, skip
        )


def _double_conv(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
        nn.GroupNorm(_groups(out_channels), out_channels),
        nn.GELU(),
        nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
        nn.GroupNorm(_groups(out_channels), out_channels),
        nn.GELU(),
    )


class UNetSegmenter(nn.Module):


    def __init__(self, cfg: Mapping[str, Any]):
        super().__init__()
        self.bands = int(cfg["input_bands"])
        self.patch_size = int(cfg["patch_size"])
        self.channels = int(cfg["d_model"])
        widths = tuple(int(value) for value in cfg.get(
            "unet_widths", [self.channels * factor for factor in (1, 2, 4, 8)]
        ))
        if len(widths) != 4 or min(widths) < 1 or widths[0] != self.channels:
            raise ValueError("U-Net requires four positive widths, first equal to d_model")
        if self.patch_size < 8:
            raise ValueError("U-Net patch size must support three pooling stages")
        self.widths = widths

        self.stem = _double_conv(self.bands, widths[0])
        self.down_blocks = nn.ModuleList(
            _double_conv(previous, current)
            for previous, current in zip(widths[:-1], widths[1:])
        )
        self.up_blocks = nn.ModuleList(
            _double_conv(widths[index + 1] + widths[index], widths[index])
            for index in (2, 1, 0)
        )
        self.classifier = nn.Conv2d(widths[0], 2, 1)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        expected = (self.bands, self.patch_size, self.patch_size)
        if x.ndim != 4 or tuple(x.shape[1:]) != expected:
            raise ValueError(f"Expected [B,{expected[0]},{expected[1]},{expected[2]}], "
                             f"received {tuple(x.shape)}")
        skip_features = [self.stem(x)]
        for block in self.down_blocks:
            skip_features.append(block(F.max_pool2d(skip_features[-1], 2)))
        features = skip_features[-1]
        for block, skip in zip(self.up_blocks, reversed(skip_features[:-1])):
            upsampled = F.interpolate(features, size=skip.shape[-2:],
                                     mode="bilinear", align_corners=False)
            features = block(torch.cat((upsampled, skip), dim=1))
        return {"logits": self.classifier(features), "features": features}


def build_model(name: str, cfg: Mapping[str, Any]) -> nn.Module:

    defaults: dict[str, Any] = {
        "input_bands": 351, "patch_size": 64, "stem_channels": 8,
        "spatial_grid": 8, "d_model": 32, "d_state": 16, "d_conv": 4,
        "expand": 2, "checkpoint_spectral": True,
    }
    if "model" in cfg:
        defaults["input_bands"] = cfg.get("data", {}).get("expected_bands", 351)
        defaults.update(cfg["model"])
    else:
        defaults.update(cfg)
    if defaults.get("allow_fallback", False):
        raise ValueError("Fallback is forbidden by the protocol")
    if name == "cnn_mt":
        return HighResolutionCNNSegmenter(defaults)
    if name in {"mamba_mt", "fewlabel_mamba", "mamba_supervised"}:
        return HighResolutionMambaSegmenter(defaults)
    if name == "unet_mt":
        return UNetSegmenter(defaults)
    raise ValueError(f"Model is not implemented for v2.1: {name}")

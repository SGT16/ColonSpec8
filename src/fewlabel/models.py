from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def _groups(channels: int) -> int:
    return next(g for g in (8, 4, 2, 1) if channels % g == 0)


def spatial_sequences(x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:

    rows = x.flatten(2).transpose(1, 2)
    columns = x.transpose(2, 3).flatten(2).transpose(1, 2)
    return rows, rows.flip(1), columns, columns.flip(1)


def restore_spatial_sequences(
    scans: tuple[Tensor, Tensor, Tensor, Tensor], height: int, width: int
) -> Tensor:

    rows, reversed_rows, columns, reversed_columns = scans
    b, _, channels = rows.shape
    row_map = rows.transpose(1, 2).reshape(b, channels, height, width)
    reverse_row_map = reversed_rows.flip(1).transpose(1, 2).reshape(
        b, channels, height, width
    )
    column_map = columns.transpose(1, 2).reshape(
        b, channels, width, height
    ).transpose(2, 3)
    reverse_column_map = reversed_columns.flip(1).transpose(1, 2).reshape(
        b, channels, width, height
    ).transpose(2, 3)
    return (row_map + reverse_row_map + column_map + reverse_column_map) * 0.25


class _ResidualMamba(nn.Module):
    def __init__(self, cfg: Mapping[str, Any]):
        super().__init__()

        from mamba_ssm import Mamba

        channels = int(cfg["d_model"])
        self.norm = nn.LayerNorm(channels)
        self.mamba = Mamba(
            d_model=channels,
            d_state=int(cfg["d_state"]),
            d_conv=int(cfg["d_conv"]),
            expand=int(cfg["expand"]),
        )

    def forward(self, sequence: Tensor) -> Tensor:
        return sequence + self.mamba(self.norm(sequence).contiguous())


class _BaseSegmenter(nn.Module):
    def __init__(self, cfg: Mapping[str, Any]):
        super().__init__()
        self.bands = int(cfg["input_bands"])
        self.patch_size = int(cfg["patch_size"])
        self.grid = int(cfg["spatial_grid"])
        self.channels = int(cfg["d_model"])
        stem_channels = int(cfg["stem_channels"])
        self.stem = nn.Sequential(
            nn.Conv3d(1, stem_channels, (7, 3, 3), stride=(1, 2, 2),
                      padding=(3, 1, 1), bias=False),
            nn.GroupNorm(_groups(stem_channels), stem_channels),
            nn.GELU(),
        )
        self.spectral_projection = nn.Linear(stem_channels, self.channels)
        self.spectral_readout = nn.Linear(2 * self.channels, self.channels)
        self.spatial_projection = nn.Conv2d(
            2 * stem_channels, self.channels, 1
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(2 * self.channels, self.channels, 1, bias=False),
            nn.GroupNorm(_groups(self.channels), self.channels),
            nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(self.channels), self.channels),
            nn.GELU(),
            nn.Conv2d(self.channels, self.channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(self.channels), self.channels),
            nn.GELU(),
        )
        self.classifier = nn.Conv2d(self.channels, 2, 1)

    def _stem_and_sequences(self, x: Tensor) -> tuple[Tensor, Tensor]:
        expected = (self.bands, self.patch_size, self.patch_size)
        if x.ndim != 4 or tuple(x.shape[1:]) != expected:
            raise ValueError(f"Expected [B,{expected[0]},{expected[1]},{expected[2]}], "
                             f"received {tuple(x.shape)}")
        encoded = self.stem(x.unsqueeze(1))

        encoded = F.adaptive_avg_pool3d(
            encoded, (self.bands, self.grid, self.grid)
        )
        b, stem_channels, bands, height, width = encoded.shape
        sequence = encoded.permute(0, 3, 4, 2, 1).reshape(
            b * height * width, bands, stem_channels
        )
        sequence = self.spectral_projection(sequence)
        spatial = self.spatial_projection(torch.cat(
            (encoded.mean(dim=2), encoded.amax(dim=2)), dim=1
        ))
        return sequence, spatial

    def _spectral_map(self, sequence: Tensor, batch_size: int) -> Tensor:
        pooled = torch.cat((sequence.mean(1), sequence.amax(1)), dim=-1)
        pooled = self.spectral_readout(pooled)
        return pooled.reshape(batch_size, self.grid, self.grid, self.channels).permute(
            0, 3, 1, 2
        ).contiguous()

    def _decode(self, spectral: Tensor, spatial: Tensor) -> dict[str, Tensor]:
        merged = self.fusion(torch.cat((spectral, spatial), dim=1))
        features = self.decoder(F.interpolate(
            merged, size=(self.patch_size, self.patch_size),
            mode="bilinear", align_corners=False
        ))
        return {"logits": self.classifier(features), "features": features}


class MambaSegmenter(_BaseSegmenter):
    def __init__(self, cfg: Mapping[str, Any]):
        super().__init__(cfg)
        self.spectral_mamba = _ResidualMamba(cfg)
        self.spatial_mamba = _ResidualMamba(cfg)
        self.checkpoint_spectral = bool(cfg.get("checkpoint_spectral", True))

    def _spectral_branch(self, sequence: Tensor) -> Tensor:
        forward = self.spectral_mamba(sequence)
        backward = self.spectral_mamba(sequence.flip(1)).flip(1)
        return (forward + backward) * 0.5

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        if x.device.type != "cuda":
            raise RuntimeError("Mamba pilot requires CUDA with the real selective-scan kernel")
        sequence, spatial = self._stem_and_sequences(x)
        if self.checkpoint_spectral and self.training and torch.is_grad_enabled():
            sequence = checkpoint(self._spectral_branch, sequence, use_reentrant=False)
        else:
            sequence = self._spectral_branch(sequence)
        scans = spatial_sequences(spatial)
        scanned = self.spatial_mamba(torch.cat(scans, dim=0))
        restored = restore_spatial_sequences(
            tuple(scanned.chunk(4, dim=0)), self.grid, self.grid
        )
        return self._decode(self._spectral_map(sequence, x.shape[0]), restored)


class CNNSegmenter(_BaseSegmenter):


    def __init__(self, cfg: Mapping[str, Any]):
        super().__init__(cfg)
        d = self.channels
        self.spectral_convs = nn.Sequential(
            nn.Conv1d(d, d, 7, padding=3, bias=False),
            nn.GroupNorm(_groups(d), d), nn.GELU(),
            nn.Conv1d(d, d, 7, padding=3, bias=False),
            nn.GroupNorm(_groups(d), d), nn.GELU(),
        )
        self.spatial_convs = nn.Sequential(
            nn.Conv2d(d, d, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(d), d), nn.GELU(),
            nn.Conv2d(d, d, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(d), d), nn.GELU(),
        )

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        sequence, spatial = self._stem_and_sequences(x)
        sequence = sequence + self.spectral_convs(sequence.transpose(1, 2)).transpose(1, 2)
        spatial = spatial + self.spatial_convs(spatial)
        return self._decode(self._spectral_map(sequence, x.shape[0]), spatial)


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
        return CNNSegmenter(defaults)
    if name in {"mamba_mt", "fewlabel_mamba", "mamba_supervised"}:
        return MambaSegmenter(defaults)
    raise ValueError(f"Model is not implemented: {name}")

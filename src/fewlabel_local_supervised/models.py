from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

from torch import nn

from fewlabel_components.models import (
    LOCAL_ARCHITECTURE, LocalOnlySegmenter, build_model as frozen_local_builder,
)

MODEL = "local_only_supervised"
VARIANT = MODEL
EXPECTED_PARAMETERS = 79306


def build_model(name: str, cfg: Mapping[str, Any]) -> nn.Module:

    if name != MODEL:
        raise ValueError("Only local_only_supervised is authorized here")
    if "model" not in cfg or cfg.get("ssl", {}).get("lambda_u_max") != 0:
        raise ValueError("The full training config must prove lambda_u_max=0")
    if cfg["model"].get("architecture") != LOCAL_ARCHITECTURE:
        raise ValueError("The frozen local_only_v1 architecture is required")
    constructor_cfg = copy.deepcopy(cfg)
    constructor_cfg["ssl"]["lambda_u_max"] = 1.0
    model = frozen_local_builder("local_only_mt", constructor_cfg)
    if not isinstance(model, LocalOnlySegmenter):
        raise RuntimeError("Frozen local-only constructor identity changed")
    if sum(parameter.numel() for parameter in model.parameters()) != EXPECTED_PARAMETERS:
        raise RuntimeError("The frozen 79,306-parameter local body changed")
    return model

from collections.abc import Mapping
from typing import Any

from torch import nn

from fewlabel_local_supervised.models import (
    LOCAL_ARCHITECTURE, LocalOnlySegmenter, build_model as parent_builder,
)

MODEL = "local_frozen_projection"
VARIANT = MODEL
EXPECTED_PARAMETERS = 79306
EXPECTED_TRAINABLE_PARAMETERS = 76498
FROZEN_PARAMETER_NAMES = ("local_projection.0.weight",)


def build_model(name: str, cfg: Mapping[str, Any]) -> nn.Module:
    if name != MODEL:
        raise ValueError("Only the local_frozen_projection control is supported")
    model = parent_builder("local_only_supervised", cfg)
    parameters = dict(model.named_parameters())
    layer = model.local_projection[0]
    if (not isinstance(layer, nn.Conv2d) or layer.bias is not None
            or layer.in_channels != 351 or layer.out_channels != 8
            or layer.kernel_size != (1, 1) or layer.groups != 1
            or tuple(parameters[FROZEN_PARAMETER_NAMES[0]].shape) != (8, 351, 1, 1)):
        raise RuntimeError("The inherited 351-to-8 pointwise projection changed")
    if any(not parameter.requires_grad for parameter in parameters.values()):
        raise RuntimeError("Parent has an unexpected frozen parameter")
    parameters[FROZEN_PARAMETER_NAMES[0]].requires_grad_(False)
    if sum(parameter.numel() for parameter in parameters.values()) != EXPECTED_PARAMETERS:
        raise RuntimeError("Total parameter count changed")
    if sum(parameter.numel() for parameter in parameters.values()
           if parameter.requires_grad) != EXPECTED_TRAINABLE_PARAMETERS:
        raise RuntimeError("Trainable parameter count changed")
    if tuple(key for key, parameter in parameters.items()
             if not parameter.requires_grad) != FROZEN_PARAMETER_NAMES:
        raise RuntimeError("Only the projection weight may be frozen")
    return model

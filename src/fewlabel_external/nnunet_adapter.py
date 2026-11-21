from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from ._vendor_nnunet.dynamic_network_architectures.architectures.unet import PlainConvUNet
from ._vendor_nnunet.network_topology import get_pool_and_conv_props

MODEL = "nnunet2d_supervised"
ARCHITECTURE = "nnunetv2_plainconv_2d_common_v1"
DISPLAY_NAME = "nnU-Net v2 2D backbone (common-protocol adaptation)"
UPSTREAM_COMMIT = "7bf2053613e2ec3e76ccac8a5fdc51dd72668287"
NNUNET_COMMIT = "066918fe8ddf1cafa731af87e63ccc56770bfc1f"


def allowed_unused_parameter_names() -> frozenset[str]:


    return frozenset(f"network.decoder.seg_layers.{stage}.{kind}"
                     for stage in range(3) for kind in ("weight", "bias"))


def architecture_plan() -> dict[str, Any]:


    pools, strides, kernels, patch, divisible = get_pool_and_conv_props(
        spacing=(1.0, 1.0), patch_size=(64, 64),
        min_feature_map_size=4, max_numpool=999999,
    )
    stages = len(strides)
    return {
        "display_name": DISPLAY_NAME,
        "architecture": ARCHITECTURE,
        "upstream_network_class": "dynamic_network_architectures.architectures.unet.PlainConvUNet",
        "dna_commit": UPSTREAM_COMMIT,
        "nnunet_commit": NNUNET_COMMIT,
        "planning_scope": "official topology helper on fixed common-protocol 2D patch; not full self-configuration",
        "planning_data": "none; protocol constants only",
        "spectral_axis": "351 input channels, not a spatial dimension",
        "patch_size": list(map(int, patch)),
        "pixel_spacing": [1.0, 1.0],
        "num_pool_per_axis": list(map(int, pools)),
        "shape_must_be_divisible_by": list(map(int, divisible)),
        "n_stages": stages,
        "features_per_stage": [min(32 * 2 ** stage, 512) for stage in range(stages)],
        "kernel_sizes": [list(map(int, kernel)) for kernel in kernels],
        "strides": [list(map(int, stride)) for stride in strides],
        "n_conv_per_stage": [2] * stages,
        "n_conv_per_stage_decoder": [2] * (stages - 1),
        "input_channels": 351,
        "num_classes": 2,
        "conv_bias": True,
        "norm": "InstanceNorm2d(eps=1e-5, affine=True)",
        "nonlinearity": "LeakyReLU(negative_slope=0.01, inplace=True)",
        "deep_supervision": False,
        "initialization": "upstream PlainConvUNet.initialize / InitWeights_He(1e-2)",
        "test_status": "LOCKED",
    }


def upstream_constructor_kwargs() -> dict[str, Any]:

    plan = architecture_plan()
    return {
        "input_channels": 351, "n_stages": plan["n_stages"],
        "features_per_stage": plan["features_per_stage"], "conv_op": nn.Conv2d,
        "kernel_sizes": plan["kernel_sizes"], "strides": plan["strides"],
        "n_conv_per_stage": plan["n_conv_per_stage"], "num_classes": 2,
        "n_conv_per_stage_decoder": plan["n_conv_per_stage_decoder"],
        "conv_bias": True, "norm_op": nn.InstanceNorm2d,
        "norm_op_kwargs": {"eps": 1e-5, "affine": True},
        "dropout_op": None, "dropout_op_kwargs": None,
        "nonlin": nn.LeakyReLU, "nonlin_kwargs": {"inplace": True},
        "deep_supervision": False,
    }


class ColonNNUNet2D(nn.Module):


    def __init__(self):
        super().__init__()
        self.bands, self.patch_size, self.channels = 351, 64, 32
        self.network = PlainConvUNet(**upstream_constructor_kwargs())
        self.network.apply(self.network.initialize)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        if x.ndim != 4 or tuple(x.shape[1:]) != (self.bands, self.patch_size, self.patch_size):
            raise ValueError("nnU-Net Colon adapter expects [B,351,64,64]")


        skips = self.network.encoder(x)
        decoder = self.network.decoder
        features = skips[-1]
        for stage in range(len(decoder.stages)):
            features = decoder.transpconvs[stage](features)
            features = torch.cat((features, skips[-(stage + 2)]), dim=1)
            features = decoder.stages[stage](features)
        return {"logits": decoder.seg_layers[-1](features), "features": features}


def build_model(name: str, cfg: Mapping[str, Any]) -> nn.Module:
    if name != MODEL:
        raise ValueError("Only nnunet2d_supervised is supported by this adapter")
    model = cfg.get("model", {})
    if model.get("architecture") != ARCHITECTURE:
        raise ValueError("The pinned nnU-Net 2D common-protocol architecture is required")
    if (model.get("bands"), model.get("patch_size"), model.get("d_model")) != (351, 64, 32):
        raise ValueError("Keep all 351 bands, 64-pixel patches, and 32 decoder feature channels")
    ssl = cfg.get("ssl", {})
    if ssl.get("lambda_u_max") != 0 or ssl.get("lambda_proto_max") != 0:
        raise ValueError("The external comparator uses the same supervised sparse objective")
    return ColonNNUNet2D()

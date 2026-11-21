from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


UPSTREAM_COMMIT = "8e9fd37cdcbff5f9717433cddb83f5215f9ec54e"
ARCHITECTURE = "hybridsn_pca30_sliding25_v1"
MODEL = "hybridsn_supervised"
INPUT_BANDS = 351
PCA_COMPONENTS = 30
PATCH_WINDOW = 25
HALO = PATCH_WINDOW // 2
FEATURE_CHANNELS = 128


class FixedWhitenedPCA(nn.Module):


    def __init__(self, weight: Tensor, bias: Tensor):
        super().__init__()
        weight = torch.as_tensor(weight).detach().to(device="cpu", dtype=torch.float32).clone()
        bias = torch.as_tensor(bias).detach().to(device="cpu", dtype=torch.float32).clone()
        if tuple(weight.shape) != (PCA_COMPONENTS, INPUT_BANDS) or tuple(bias.shape) != (PCA_COMPONENTS,):
            raise ValueError("HybridSN requires fixed PCA weight[30,351] and bias[30]")
        if not bool(torch.isfinite(weight).all() and torch.isfinite(bias).all()):
            raise ValueError("PCA parameters must be finite")
        self.register_buffer("weight", weight[:, :, None, None])
        self.register_buffer("bias", bias)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4 or x.shape[1] != INPUT_BANDS or min(x.shape[-2:]) < 1:
            raise ValueError("Expected standardized input [B,351,H,W], H,W >= 1")
        return F.conv2d(x, self.weight, self.bias)


class HybridSNDense(nn.Module):


    architecture = ARCHITECTURE
    feature_channels = FEATURE_CHANNELS
    receptive_field = PATCH_WINDOW

    def __init__(self, pca_weight: Tensor, pca_bias: Tensor):
        super().__init__()
        self.pca = FixedWhitenedPCA(pca_weight, pca_bias)
        self.conv3d_1 = nn.Conv3d(1, 8, kernel_size=(7, 3, 3))
        self.conv3d_2 = nn.Conv3d(8, 16, kernel_size=(5, 3, 3))
        self.conv3d_3 = nn.Conv3d(16, 32, kernel_size=(3, 3, 3))

        self.conv2d = nn.Conv2d(18 * 32, 64, kernel_size=3)

        self.dense256 = nn.Conv2d(64, 256, kernel_size=17)
        self.dropout256 = nn.Dropout(p=0.4)
        self.dense128 = nn.Conv2d(256, FEATURE_CHANNELS, kernel_size=1)
        self.dropout128 = nn.Dropout(p=0.4)
        self.classifier = nn.Conv2d(FEATURE_CHANNELS, 2, kernel_size=1)
        self.reset_parameters()

    def reset_parameters(self) -> None:


        for module in (self.conv3d_1, self.conv3d_2, self.conv3d_3, self.conv2d,
                       self.dense128, self.classifier):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.dense256.weight.view(256, -1))
        nn.init.zeros_(self.dense256.bias)

    def _spatial_features(self, projected: Tensor) -> Tensor:
        x = F.relu(self.conv3d_1(projected.unsqueeze(1)))
        x = F.relu(self.conv3d_2(x))
        x = F.relu(self.conv3d_3(x))


        b, channels, spectral, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(b, spectral * channels, h, w)
        return F.relu(self.conv2d(x))

    def _valid_projected(self, projected: Tensor) -> dict[str, Tensor]:
        if min(projected.shape[-2:]) < PATCH_WINDOW:
            raise ValueError("Unpadded classifier input must be at least 25x25")
        x = self._spatial_features(projected)
        x = self.dropout256(F.relu(self.dense256(x)))
        features = self.dropout128(F.relu(self.dense128(x)))
        return {"logits": self.classifier(features), "features": features}

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        projected = self.pca(x)


        projected = F.pad(projected, (HALO, HALO, HALO, HALO), mode="constant", value=0.0)
        return self._valid_projected(projected)

    def patch_logits(self, x: Tensor) -> Tensor:
        if x.ndim != 4 or tuple(x.shape[-2:]) != (PATCH_WINDOW, PATCH_WINDOW):
            raise ValueError("patch_logits requires one or more unpadded 25x25 patches")
        return self._valid_projected(self.pca(x))["logits"][:, :, 0, 0]


def build_model(pca_weight: Tensor, pca_bias: Tensor) -> HybridSNDense:

    return HybridSNDense(pca_weight=pca_weight, pca_bias=pca_bias)

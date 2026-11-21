import hashlib

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from fewlabel_local_supervised.models import LOCAL_ARCHITECTURE, LocalOnlySegmenter, build_model as parent_builder
from .pca import load_basis

MODEL = VARIANT = 'local_pca8_supervised'
EXPECTED_PARAMETERS = 79306
EXPECTED_TRAINABLE_PARAMETERS = 76498
FROZEN_PARAMETER_NAMES = ('local_projection.0.weight',)
FROZEN_BUFFER_NAMES = ('local_projection.0.pca_bias',)
IDENTITY_FIELDS = ('parent_initial_weights_sha256', 'shared_initial_weights_sha256',
                   'pca_basis_sha256', 'pca_manifest_sha256')


def state_digest(model, shared=False):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        if shared and name in FROZEN_PARAMETER_NAMES + FROZEN_BUFFER_NAMES:
            continue
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class FixedPCAPointwise(nn.Module):
    def __init__(self, components, mean):
        super().__init__()

        w = np.array(components, dtype=np.float32, copy=True)
        self.weight = nn.Parameter(torch.from_numpy(w[:, :, None, None]), requires_grad=False)
        self.register_buffer('pca_bias', torch.from_numpy(np.asarray(-w @ mean.astype(np.float32), dtype=np.float32)))

    def forward(self, x):
        return F.conv2d(x, self.weight, self.pca_bias)


def build_model(name, cfg):
    if name != MODEL:
        raise ValueError('Only local_pca8_supervised is authorized')
    model = parent_builder('local_only_supervised', cfg)
    layer = model.local_projection[0]
    if not isinstance(layer, nn.Conv2d) or layer.bias is not None or tuple(layer.weight.shape) != (8, 351, 1, 1):
        raise RuntimeError('Inherited input projection changed')
    parent_digest, shared_digest = state_digest(model), state_digest(model, shared=True)
    arrays, metadata = load_basis(cfg)
    model.local_projection[0] = FixedPCAPointwise(arrays['components'], arrays['mean'])
    model.parent_initial_weights_sha256 = parent_digest
    model.shared_initial_weights_sha256 = shared_digest
    model.pca_basis_sha256 = metadata['basis_sha256']
    model.pca_manifest_sha256 = metadata['manifest_sha256']
    if (sum(p.numel() for p in model.parameters()) != EXPECTED_PARAMETERS or
            sum(p.numel() for p in model.parameters() if p.requires_grad) != EXPECTED_TRAINABLE_PARAMETERS or
            tuple(k for k, p in model.named_parameters() if not p.requires_grad) != FROZEN_PARAMETER_NAMES or
            state_digest(model, shared=True) != shared_digest):
        raise RuntimeError('PCA parameter/shared initialization contract changed')
    return model

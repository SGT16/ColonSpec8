from __future__ import annotations
import hashlib
import os
from pathlib import Path
from types import MethodType
import torch
from torch.nn import functional as F
from fewlabel_local_supervised.models import build_model as historical_builder
from fewlabel_v21.deterministic_ops import resize_bilinear

MODELS = ('local_only_supervised', 'local_frozen_projection')
PROJECTION = 'local_projection.0.weight'


def state_digest(state):
    digest = hashlib.sha256()
    for name, tensor in state.items():
        digest.update(name.encode()); digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def create_initial_state(cfg, init_seed, path):
    path = Path(path)
    if path.exists():
        raise FileExistsError('Initial state must not be overwritten')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(init_seed))
        model = historical_builder('local_only_supervised', cfg)
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    payload = {'init_seed': int(init_seed), 'state_dict': state,
               'initial_weights_sha256': state_digest(state), 'architecture': 'local_only_v1'}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as handle:
        torch.save(payload, handle); handle.flush(); os.fsync(handle.fileno())
    return {'init_seed': int(init_seed), 'initial_weights_sha256': payload['initial_weights_sha256'],
            'path': str(path.resolve()), 'parameters': sum(p.numel() for p in model.parameters())}


def deterministic_forward(self, x):
    if x.ndim != 4 or tuple(x.shape[1:]) != (self.bands, self.patch_size, self.patch_size):
        raise ValueError('Mask-study input shape differs from frozen local body')
    local_skips = [self.local_encoder(self.local_projection(x))]
    for block in self.local_down:
        local_skips.append(block(F.max_pool2d(local_skips[-1], 2)))
    features = self.bottleneck_fusion(local_skips[-1])
    for block, skip in zip(self.local_up, reversed(local_skips[:-1])):
        upsampled = resize_bilinear(features, tuple(skip.shape[-2:]))
        features = block(torch.cat((upsampled, skip), dim=1))
    features = self.feature_projection(features)
    return {'logits': self.classifier(features), 'features': features}


def build_model(model_name, cfg, initial_path, init_seed):
    if model_name not in MODELS:
        raise ValueError('Unauthorized mask-study model')
    initial_path = Path(initial_path)
    if not initial_path.is_file():
        raise FileNotFoundError('Precreated common initial weights are required')
    payload = torch.load(initial_path, map_location='cpu', weights_only=True)
    if payload.get('init_seed') != int(init_seed):
        raise RuntimeError('Initial checkpoint seed mismatch')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(init_seed))
        model = historical_builder('local_only_supervised', cfg)
    model.load_state_dict(payload['state_dict'], strict=True)
    digest = state_digest(model.state_dict())
    if digest != payload.get('initial_weights_sha256', digest):
        raise RuntimeError('Initial tensor digest mismatch')
    if model_name == 'local_frozen_projection':
        dict(model.named_parameters())[PROJECTION].requires_grad_(False)
    if sum(p.numel() for p in model.parameters()) != 79306:
        raise RuntimeError('Local architecture parameter count changed')

    model.forward = MethodType(deterministic_forward, model)
    model.maskstudy_initial_sha256 = digest
    model.maskstudy_variant = model_name
    return model

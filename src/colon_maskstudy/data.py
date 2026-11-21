from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from fewlabel.data import load_prepared
from fewlabel.training import PatchPlan
from fewlabel.runtime import derived_seed, sha


def load_data(root, mask_seed, val_seed=938, fraction=.005, cfg=None):
    if fraction != .005:
        raise ValueError('Mask study requires the existing 0.5% masks')


    frozen_data_cfg = json.loads((Path(root)/'configs/project.json').read_text(encoding='utf-8-sig'))
    prepared = load_prepared(root, frozen_data_cfg, int(mask_seed), fraction)
    validation = load_prepared(root, frozen_data_cfg, int(val_seed), fraction)
    if (prepared.train_ids != validation.train_ids or prepared.val_ids != validation.val_ids
            or not np.array_equal(prepared.mean, validation.mean)
            or not np.array_equal(prepared.std, validation.std)):
        raise RuntimeError('Split or TRAIN normalization mismatch')

    prepared.masks = dict(prepared.masks)
    prepared.mask_sha256 = dict(prepared.mask_sha256)
    for image in prepared.val_ids:
        prepared.masks[image] = validation.masks[image]
        prepared.mask_sha256[image] = validation.mask_sha256[image]
    prepared.train_mask_seed = int(mask_seed)
    prepared.validation_mask_seed = int(val_seed)
    return prepared


def uniform_coordinates(prepared, cfg, data_seed, epoch, n_samples):
    size = int(cfg['model']['patch_size'])
    ids = list(prepared.train_ids)
    if not ids or n_samples <= 0:
        raise ValueError('Nonempty TRAIN and positive sample count required')
    rng = np.random.default_rng(derived_seed(data_seed, 'uniform_maskstudy_sampling', epoch))
    plan = np.empty((n_samples, 3), dtype=np.int32)
    for index in range(n_samples):
        image_index = int(rng.integers(len(ids)))
        height, width = prepared.image_shape(ids[image_index])
        plan[index] = (image_index, int(rng.integers(max(1, height-size+1))),
                       int(rng.integers(max(1, width-size+1))))
    return plan


def create_shared_plan(prepared, cfg, data_seed, path, epochs=100, steps=128, effective=32):
    path = Path(path)
    if path.exists():
        raise FileExistsError('Shared coordinate plan is immutable once created')
    coordinates = np.stack([uniform_coordinates(prepared, cfg, data_seed, epoch, steps*effective)
                            for epoch in range(1, epochs+1)])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temporary.open('xb') as handle:
        np.savez_compressed(handle, coordinates=coordinates,
            train_ids=np.asarray(prepared.train_ids), data_seed=np.int64(data_seed),
            patch_size=np.int64(cfg['model']['patch_size']))
        handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)
    return {'path': str(path.resolve()), 'sha256': sha(path),
            'coordinate_payload_sha256': hashlib.sha256(coordinates.tobytes()).hexdigest(),
            'shape': list(coordinates.shape), 'data_seed': int(data_seed),
            'mask_independent': True, 'train_ids': list(prepared.train_ids)}


class UniformPatchPlan(PatchPlan):

    def __init__(self, prepared, cfg, seed, epoch, n_samples):
        self.prepared, self.cfg, self.seed, self.epoch = prepared, cfg, int(seed), int(epoch)
        self.size = int(cfg['model']['patch_size'])
        self.ids = list(prepared.train_ids)
        spec = cfg.get('maskstudy', {})
        path = spec.get('plan_path')
        if path:
            if spec.get('plan_sha256') and sha(path) != spec['plan_sha256']:
                raise RuntimeError('Common plan hash changed')
            with np.load(path, allow_pickle=False) as plan:
                if (int(plan['data_seed']) != self.seed or int(plan['patch_size']) != self.size
                        or plan['train_ids'].tolist() != self.ids):
                    raise RuntimeError('Common plan identity mismatch')
                coordinates = plan['coordinates'][epoch-1].copy()
            if coordinates.shape != (n_samples, 3):
                raise RuntimeError('Common plan sample count mismatch')
        elif spec.get('require_plan', True):
            raise RuntimeError('Production training requires a frozen shared plan')
        else:
            coordinates = uniform_coordinates(prepared, cfg, seed, epoch, n_samples)
        self.plan = []
        for image_index, y, x in coordinates:
            if not 0 <= image_index < len(self.ids):
                raise RuntimeError('Unknown TRAIN image in common plan')
            height, width = prepared.image_shape(self.ids[int(image_index)])
            if not (0 <= y <= max(0, height-self.size) and 0 <= x <= max(0, width-self.size)):
                raise RuntimeError('Invalid common patch origin')
            self.plan.append((int(image_index), int(y), int(x), False))
        self.coordinate_sha256 = hashlib.sha256(coordinates.astype(np.int32).tobytes()).hexdigest()

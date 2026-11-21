from pathlib import Path
import os
import re

import numpy as np

from fewlabel.data import digest_json
from fewlabel.runtime import fingerprint, now, read_json, sha, write_json
from fewlabel_unet_compare.common import safe_path

ROOT = Path(__file__).resolve().parents[2]
BASIS = 'experiments/pca8_v1/artifacts/pca8/basis.npz'
MANIFEST = 'experiments/pca8_v1/artifacts/pca8/manifest.json'
TRAIN = [f'HSI_train_{i}' for i in (1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17)]
SPLIT_SHA = '5ec1b1326042921ccdca62a560f3a7b08566afe9be56b11cde5a375afa4e4ced'
SAMPLING = '64x64_spatial_cell_centers_per_TRAIN_image_v1'


def grid_indices(height, width):
    if min(height, width) < 64:
        raise ValueError('PCA sampling requires at least 64 pixels on each axis')
    y = np.floor((np.arange(64) + .5) * height / 64).astype(np.int64)
    x = np.floor((np.arange(64) + .5) * width / 64).astype(np.int64)
    return y, x


def fit_arrays(chunks, components=8):

    n, mean, scatter = 0, None, None
    for chunk in chunks:
        a = np.asarray(chunk, dtype=np.float64)
        if a.ndim != 2 or len(a) < 2 or not np.isfinite(a).all():
            raise ValueError('Invalid/nonfinite PCA sample chunk')
        m = a.mean(0)
        centered = a - m
        s = centered.T @ centered
        if mean is None:
            n, mean, scatter = len(a), m, s
        else:
            if m.shape != mean.shape:
                raise ValueError('PCA band count changed')
            delta = m - mean
            merged = n + len(a)
            scatter += s + np.outer(delta, delta) * (n * len(a) / merged)
            mean += delta * (len(a) / merged)
            n = merged
    if mean is None or n <= components or len(mean) < components:
        raise ValueError('Insufficient PCA samples/bands')
    covariance = scatter / (n - 1)
    eigenvalues, vectors = np.linalg.eigh((covariance + covariance.T) * .5)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    if eigenvalues[components - 1] <= max(eigenvalues[0] * 1e-12, 1e-15):
        raise ValueError('PCA samples have insufficient numerical rank')
    w = vectors[:, order[:components]].T.copy()
    signs = np.sign(w[np.arange(components), np.abs(w).argmax(1)])
    w *= signs[:, None]
    return dict(mean=mean, components=w, eigenvalues=eigenvalues[:components],
                explained_variance_ratio=eigenvalues[:components] / np.trace(covariance),
                n_samples=np.array(n, dtype=np.int64))


def provenance(root):

    root = Path(root)
    split_path = root / 'state/fixed_split.json'
    report_path = root / 'data/processed/pilot_v2/prepared.json'
    split, report = read_json(split_path), read_json(report_path)
    if (split.get('train_ids') != TRAIN or set(split.get('val_ids', [])) !=
            {'HSI_train_3', 'HSI_train_4', 'HSI_train_16'} or
            split.get('split_sha256') != SPLIT_SHA or report.get('split_sha256') != SPLIT_SHA or
            report.get('status') != 'READY' or split.get('final_test_status') != 'LOCKED'):
        raise RuntimeError('Fixed TRAIN/VAL/locked TEST split changed')
    normal_path = safe_path(root, report['normalization_path'])
    normal = read_json(normal_path)
    if sha(normal_path) != report['normalization_sha256'] or normal['fit_image_ids'] != TRAIN:
        raise RuntimeError('PCA requires the verified TRAIN-only scaler')
    mean, std = np.array(normal['mean'], dtype=np.float32), np.array(normal['std'], dtype=np.float32)
    if mean.shape != (351,) or std.shape != (351,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise RuntimeError('Invalid TRAIN scaler')
    evidence = dict(split_file_sha256=sha(split_path), prepared_file_sha256=sha(report_path),
                    normalization_sha256=sha(normal_path), normalization_path=report['normalization_path'])
    return report, mean, std, evidence


def train_cache_entry(root, report, image_id):
    if image_id not in TRAIN or not re.fullmatch(r'HSI_train_[1-9][0-9]*', image_id):
        raise ValueError('PCA payload access restricted to fixed TRAIN IDs')
    root = Path(root)
    image = report['images'][image_id]
    source = safe_path(root, image['path'])
    meta_path = safe_path(root, image['metadata_path'])
    if source.name != image_id + '.npy' or meta_path.name != image_id + '.json':
        raise RuntimeError('TRAIN cube basename mismatch')
    if sha(meta_path) != image['metadata_sha256']:
        raise RuntimeError('TRAIN cube metadata hash changed')
    metadata, stat = read_json(meta_path), source.stat()
    if metadata['size_bytes'] != stat.st_size or metadata['mtime_ns'] != stat.st_mtime_ns or metadata['shape'] != image['shape'] or image['shape'][0] != 351:
        raise RuntimeError('TRAIN source cube identity changed')
    identity = dict(image_id=image_id, source_metadata_sha256=image['metadata_sha256'],
                    source_payload_sha256=metadata['cache_payload_sha256'], source_size_bytes=stat.st_size,
                    source_mtime_ns=stat.st_mtime_ns, source_shape_chw=image['shape'],
                    local_cache_format='float32_HWC_npy_v1')
    cache = Path.home() / '.cache/fewlabel_mamba' / fingerprint(str(root.resolve()))[:16] / 'hwc_v1'
    target = cache / f'{image_id}.hwc.npy'
    index = cache / f'{image_id}.hwc.json'
    if target.resolve().parent != cache.resolve() or index.resolve().parent != cache.resolve():
        raise RuntimeError('TRAIN cache path escapes cache directory')
    envelope = read_json(index)
    entry, stat = envelope['metadata'], target.stat()
    shape = [image['shape'][1], image['shape'][2], 351]
    if (envelope.get('metadata_sha256') != digest_json(entry) or
            entry.get('source_fingerprint') != digest_json(identity) or
            entry.get('size_bytes') != stat.st_size or entry.get('mtime_ns') != stat.st_mtime_ns or
            entry.get('shape_hwc') != shape or entry.get('dtype') != 'float32'):
        raise RuntimeError('Verified TRAIN HWC cache identity changed')
    return target, dict(image_id=image_id, source_identity=identity, cache_metadata_sha256=digest_json(entry),
                        cache_payload_sha256=entry['hwc_payload_sha256'], cache_size_bytes=stat.st_size,
                        cache_mtime_ns=stat.st_mtime_ns, shape_hwc=shape)


def _load_pair(basis_path, manifest_path, scope):
    manifest = read_json(manifest_path)
    if (manifest.get('status') != 'READY' or manifest.get('scope') != scope or
            manifest.get('n_components') != 8 or manifest.get('input_dim') != 351 or
            manifest.get('whiten') is not False or manifest.get('basis_sha256') != sha(basis_path)):
        raise RuntimeError('PCA artifact identity/hash/scope mismatch')
    with np.load(basis_path, allow_pickle=False) as data:
        arrays = {key: data[key].copy() for key in data.files}
    for key, shape in (('mean', (351,)), ('components', (8, 351)), ('eigenvalues', (8,)), ('explained_variance_ratio', (8,))):
        if key not in arrays or arrays[key].shape != shape or not np.isfinite(arrays[key]).all():
            raise RuntimeError('PCA array shape/finite check failed')
    w, e, ratio = arrays['components'], arrays['eigenvalues'], arrays['explained_variance_ratio']
    if (not np.allclose(w @ w.T, np.eye(8), atol=1e-9, rtol=0) or
            np.any(e <= 0) or np.any(np.diff(e) > 0) or np.any(ratio <= 0) or ratio.sum() > 1 + 1e-9 or
            int(arrays['n_samples']) != manifest['n_samples']):
        raise RuntimeError('PCA orthogonality/variance/sample check failed')
    info = dict(manifest, basis_sha256=sha(basis_path), manifest_sha256=sha(manifest_path))
    return arrays, info


def verify_pca(root=ROOT):
    root = Path(root)
    report, _, _, source = provenance(root)
    _, info = _load_pair(root / BASIS, root / MANIFEST, 'TRAIN_ONLY_PCA')
    if (info.get('fit_image_ids') != TRAIN or info.get('sampling') != SAMPLING or
            info.get('n_samples') != 57344 or info.get('pixels_per_image') != 4096 or
            info.get('source_metadata') != source):
        raise RuntimeError('PCA TRAIN provenance changed')
    current = [train_cache_entry(root, report, image_id)[1] for image_id in TRAIN]
    if info.get('train_sources') != current:
        raise RuntimeError('PCA source cache identity changed')
    info.update(basis_path=BASIS, manifest_path=MANIFEST,
                data_transform_hashes={BASIS: info['basis_sha256'], MANIFEST: info['manifest_sha256']})
    return info


def load_basis(cfg, root=ROOT):
    config = cfg.get('pca', {})
    if config.get('components') != 8 or config.get('whiten') is not False:
        raise ValueError('Only centered, non-whitened PCA-8 is authorized')
    if cfg.get('runtime', {}).get('synthetic_fixture'):
        return _load_pair(Path(config['basis_path']), Path(config['manifest_path']), 'SENTETIK_TEST')
    if config['basis_path'] != BASIS or config['manifest_path'] != MANIFEST:
        raise ValueError('Production PCA paths changed')
    info = verify_pca(root)
    arrays, _ = _load_pair(Path(root) / BASIS, Path(root) / MANIFEST, 'TRAIN_ONLY_PCA')
    return arrays, info


def ensure_pca(root=ROOT, status=None):
    root = Path(root)
    basis, manifest = root / BASIS, root / MANIFEST
    if basis.exists() or manifest.exists():
        if not basis.is_file() or not manifest.is_file():
            raise RuntimeError('Incomplete PCA artifacts; explicit review required')
        return verify_pca(root)
    report, mean, std, source = provenance(root)
    sources = []

    def chunks():
        for index, image_id in enumerate(TRAIN):
            target, evidence = train_cache_entry(root, report, image_id)
            if status is not None:
                status.update(status='PREPARING_PCA', stage='pca_fit', pca_image=image_id,
                              pca_images_completed=index, pca_images_total=len(TRAIN))
            cube = np.load(target, mmap_mode='r', allow_pickle=False)
            try:
                if cube.dtype != np.float32 or list(cube.shape) != evidence['shape_hwc']:
                    raise RuntimeError('PCA TRAIN cache dtype/shape mismatch')
                y, x = grid_indices(*cube.shape[:2])
                sample = np.array(cube[y[:, None], x[None, :], :], dtype=np.float32).reshape(-1, 351)
            finally:
                cube._mmap.close()
            sample -= mean
            sample /= std

            if train_cache_entry(root, report, image_id)[1] != evidence:
                raise RuntimeError('PCA input changed during sampling')
            sources.append(evidence)
            yield sample

    arrays = fit_arrays(chunks())
    if int(arrays['n_samples']) != 57344:
        raise RuntimeError('PCA sample budget changed')
    basis.parent.mkdir(parents=True, exist_ok=True)
    pending = basis.with_name(basis.name + '.pending')
    with pending.open('wb') as handle:
        np.savez(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(pending, basis)
    write_json(manifest, dict(schema_version=1, status='READY', scope='TRAIN_ONLY_PCA',
        dataset='ColonCancerHSI', created_at=now(), fit_image_ids=TRAIN, n_components=8,
        input_dim=351, whiten=False, sampling=SAMPLING, pixels_per_image=4096, n_samples=57344,
        basis_sha256=sha(basis), source_metadata=source, train_sources=sources,
        explained_variance_ratio=arrays['explained_variance_ratio'].tolist(),
        explained_variance_ratio_sum=float(arrays['explained_variance_ratio'].sum()),
        centering='sample_mean_after_existing_TRAIN_scaler', sign_rule='largest_absolute_loading_positive',
        input_policy='TRAIN spectra only; no labels/masks, no VAL/TEST payloads',
        cache_verification='Previously hashed cache; metadata hashes, sizes and mtimes rechecked; no full payload rehash',
        inference='frozen float32 components and bias=-components@mean; no whitening'))
    return verify_pca(root)

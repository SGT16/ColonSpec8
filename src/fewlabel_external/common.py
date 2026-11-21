import os
from pathlib import Path
from fewlabel.runtime import read_json, now, sha, write_json
ROOT=Path(__file__).resolve().parents[2]
PCA_BASIS="experiments/external_comparators_v1/artifacts/pca30/basis.npz"
PCA_MANIFEST="experiments/external_comparators_v1/artifacts/pca30/manifest.json"

class IntegrityError(RuntimeError):
    pass


def ensure_pca30():

    import numpy as np
    from fewlabel_pca8 import pca
    basis, manifest = ROOT / PCA_BASIS, ROOT / PCA_MANIFEST
    if basis.exists() or manifest.exists():
        return verify_pca30()[1]
    report, mean, std, source = pca.provenance(ROOT)
    sources = []
    def chunks():
        for image_id in pca.TRAIN:
            path, evidence = pca.train_cache_entry(ROOT, report, image_id)
            cube = np.load(path, mmap_mode='r', allow_pickle=False)
            try:
                if cube.dtype != np.float32 or list(cube.shape) != evidence['shape_hwc']:
                    raise IntegrityError('PCA30_TRAIN_CACHE_CHANGED')
                y, x = pca.grid_indices(*cube.shape[:2])
                sample = np.array(cube[y[:, None], x[None, :], :], dtype=np.float32).reshape(-1, 351)
            finally:
                cube._mmap.close()
            sample -= mean
            sample /= std
            if pca.train_cache_entry(ROOT, report, image_id)[1] != evidence:
                raise IntegrityError('PCA30_SOURCE_CHANGED_DURING_READ')
            sources.append(evidence)
            yield sample
    arrays = pca.fit_arrays(chunks(), components=30)
    basis.parent.mkdir(parents=True, exist_ok=True)
    pending = basis.with_suffix('.pending')
    with pending.open('wb') as handle:
        np.savez(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(pending, basis)
    write_json(manifest, dict(status='READY', scope='TRAIN_ONLY_PCA30_WHITENED',
        fit_image_ids=pca.TRAIN, n_samples=57344, sampling=pca.SAMPLING,
        n_components=30, input_dim=351, whiten=True, source_metadata=source,
        train_sources=sources, basis_sha256=sha(basis), created_at=now(),
        whitening='components / sqrt(sample_covariance_eigenvalues)',
        fit_uses_labels=False, val_or_test_used=False, test_status='LOCKED'))
    return verify_pca30()[1]


def verify_pca30():
    import numpy as np
    from fewlabel_pca8 import pca
    manifest = read_json(ROOT / PCA_MANIFEST)
    report, _, _, source = pca.provenance(ROOT)
    expected = dict(status='READY', scope='TRAIN_ONLY_PCA30_WHITENED', fit_image_ids=pca.TRAIN,
        n_samples=57344, sampling=pca.SAMPLING, n_components=30, input_dim=351, whiten=True,
        source_metadata=source, train_sources=[pca.train_cache_entry(ROOT, report, image)[1] for image in pca.TRAIN],
        basis_sha256=sha(ROOT / PCA_BASIS), fit_uses_labels=False, val_or_test_used=False, test_status='LOCKED')
    if any(manifest.get(k) != v for k, v in expected.items()):
        raise IntegrityError('PCA30_PROVENANCE_CHANGED')
    with np.load(ROOT / PCA_BASIS, allow_pickle=False) as data:
        arrays = {key: data[key].copy() for key in data.files}
    for key, shape in [('mean', (351,)), ('components', (30, 351)), ('eigenvalues', (30,))]:
        if arrays[key].shape != shape or not np.isfinite(arrays[key]).all():
            raise IntegrityError('PCA30_ARRAY_INVALID')
    if (int(arrays['n_samples']) != 57344 or np.any(arrays['eigenvalues'] <= 0)
            or not np.allclose(arrays['components'] @ arrays['components'].T, np.eye(30), atol=1e-9, rtol=0)):
        raise IntegrityError('PCA30_COVARIANCE_INVALID')
    return arrays, dict(manifest, manifest_sha256=sha(ROOT / PCA_MANIFEST))

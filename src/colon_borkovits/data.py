from __future__ import annotations
from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from .acquisition import ROOT, RUN, DEST, META, SCOPE, now, sha, write_json, metadata_records, checksums

CLASSES = ('NC', 'CRC')
WAVENUMBERS = np.arange(4000., 744., -4., dtype=np.float64)


def configuration(root=ROOT):
    return json.loads((Path(root) / 'configs/borkovits_v1.json').read_text(encoding='utf8'))


def make_splits(records, seed=741):

    identities = [r['image_id'] for r in records]
    if len(set(identities)) != len(identities):
        raise ValueError('Duplicate core identity')
    by_patient = defaultdict(list)
    for row in records:
        if not row['patient_id']:
            raise ValueError('Patient grouping requires a nonempty authoritative ID')
        by_patient[row['patient_id']].append(row)
    strata = defaultdict(list)
    for patient, members in by_patient.items():
        strata[tuple(sorted({r['class_index'] for r in members}))].append(patient)
    rng = np.random.default_rng(seed)
    held_patients = [[], [], []]
    offset = 0
    for stratum in sorted(strata):
        members = sorted(strata[stratum])
        for j in rng.permutation(len(members)):
            held_patients[offset % 3].append(members[int(j)])
            offset += 1
    all_patients = set(by_patient)
    folds = []
    for k, patients in enumerate(held_patients):
        hp = set(patients)
        tp = all_patients - hp
        train = sorted(r['image_id'] for r in records if r['patient_id'] in tp)
        held = sorted(r['image_id'] for r in records if r['patient_id'] in hp)
        if hp & tp or set(train) & set(held):
            raise ValueError('Patient or core leakage')
        for partition in (set(train), set(held)):
            if {r['class_index'] for r in records if r['image_id'] in partition} != {0, 1}:
                raise ValueError('A class is absent from a fold')
        folds.append(dict(fold=k, train_ids=train, heldout_ids=held,
            train_patient_ids=sorted(tp), heldout_patient_ids=sorted(hp),
            train_class_counts=dict(Counter(r['class_code'] for r in records if r['image_id'] in set(train))),
            heldout_class_counts=dict(Counter(r['class_code'] for r in records if r['image_id'] in set(held)))))
    if Counter(i for f in folds for i in f['heldout_ids']) != Counter(identities):
        raise ValueError('Each core must occur exactly once as held-out')
    return folds


def parse_cube(path):

    with open(path, encoding='utf8', newline='') as handle:
        header = next(csv.reader(handle))
    if header[:3] != ['', 'X', 'Y'] or len(header) != 817:
        raise ValueError('Unexpected spectral column header')
    wavenumbers = np.asarray(header[3:], dtype=np.float64)
    if not np.array_equal(wavenumbers, WAVENUMBERS):
        raise ValueError('Wavenumber identity/order mismatch')
    values = np.loadtxt(path, delimiter=',', skiprows=1, dtype=np.float64)
    if values.shape != (7744, 817) or not np.isfinite(values).all():
        raise ValueError('Unexpected shape or nonfinite raw values')
    if len(np.unique(values[:, 0])) != 7744:
        raise ValueError('Duplicate source row index')
    xs, xi = np.unique(values[:, 1], return_inverse=True)
    ys, yi = np.unique(values[:, 2], return_inverse=True)
    if len(xs) != 88 or len(ys) != 88 or len(np.unique(yi * 88 + xi)) != 7744:
        raise ValueError('Invalid or repeated XY spatial lattice')
    if not np.allclose(np.diff(xs), np.diff(xs)[0]) or not np.allclose(np.diff(ys), np.diff(ys)[0]):
        raise ValueError('Irregular XY grid')
    cube = np.empty((88, 88, 814), dtype=np.float32)
    cube[yi, xi] = values[:, 3:].astype(np.float32)
    if not np.isfinite(cube).all():
        raise ValueError('Nonfinite float32 conversion')
    return cube, dict(shape=list(cube.shape), dtype=str(cube.dtype),
        x_values=xs.tolist(), y_values=ys.tolist(), spatial_order='ascending Y, ascending X, original descending wavenumber',
        raw_min=float(values[:, 3:].min()), raw_max=float(values[:, 3:].max()),
        source_nonfinite=0, float32_nonfinite=0,
        float32_max_abs_rounding=float(np.abs(cube[yi,xi].astype(np.float64)-values[:,3:]).max()),
        content_sha256=hashlib.sha256(cube.tobytes()).hexdigest())


def core_moments(cube):
    grid = np.linspace(0, 87, 64).round().astype(np.int64)
    if len(np.unique(grid)) != 64:
        raise ValueError('Nonunique transform grid')
    x = np.asarray(cube[np.ix_(grid, grid, np.arange(814))], dtype=np.float64).reshape(-1, 814)
    mean = x.mean(axis=0)
    centred = x - mean
    return dict(mean=mean, m2=centred.T @ centred, count=np.array(len(x)), grid=grid)


def merge_moments(items):
    count = 0
    mean = np.zeros(814, dtype=np.float64)
    m2 = np.zeros((814, 814), dtype=np.float64)
    for item in items:
        n = int(item['count'])
        delta = item['mean'] - mean
        total = count + n
        m2 += item['m2'] + np.outer(delta, delta) * (count * n / total)
        mean += delta * n / total
        count = total
    if count < 2:
        raise ValueError('Insufficient TRAIN spectra')
    std = np.maximum(np.sqrt(np.maximum(np.diag(m2) / count, 0)), 1e-6)
    covariance = (m2 / (count - 1)) / np.outer(std, std)
    covariance = (covariance + covariance.T) * .5
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    components = eigenvectors[:, order[:8]].T.copy()
    for component in components:
        if component[np.argmax(np.abs(component))] < 0:
            component *= -1
    if not np.isfinite(components).all() or np.linalg.matrix_rank(components) != 8:
        raise ValueError('Invalid PCA transform')
    return dict(mean=mean.astype(np.float32), std=std.astype(np.float32),
        pca_mean=np.zeros(814, dtype=np.float32), components=components.astype(np.float32),
        eigenvalues=eigenvalues[order], fitted_spectra=np.array(count))


def make_plan(records, split, cfg):
    rng = np.random.default_rng(cfg['data_seed'] + split['fold'])
    shape = (cfg['epochs'], cfg['steps_per_epoch'] * cfg['effective_batch'])
    membership = set(split['train_ids'])
    buckets = [defaultdict(list), defaultdict(list)]
    for index, row in enumerate(records):
        if row['image_id'] in membership:
            buckets[row['class_index']][row['patient_id']].append(index)
    classes = rng.integers(0, 2, size=shape)
    images = np.empty(shape, dtype=np.int16)
    for label in range(2):
        patients = sorted(buckets[label])
        if not patients:
            raise ValueError('Empty TRAIN class')
        positions = np.argwhere(classes == label)
        sampled_patients = rng.integers(0, len(patients), size=len(positions))
        for pos, patient_index in zip(positions, sampled_patients):
            members = buckets[label][patients[int(patient_index)]]
            images[tuple(pos)] = members[int(rng.integers(len(members)))]
    coords = rng.integers(0, 88 - cfg['patch_size'] + 1,
        size=(*shape, cfg['bag_patches'], 2), dtype=np.int16)
    augment = rng.integers(0, 2, size=(*shape, cfg['bag_patches'], 3), dtype=np.uint8)
    augment[..., 0] = rng.integers(0, 4, size=augment.shape[:-1], dtype=np.uint8)
    return dict(image_indices=images, coordinates=coords, augment=augment)


def _save_npz(path, **values):
    path = Path(path)
    with open(path.with_suffix('.npz.partial'), 'wb') as out:
        np.savez(out, **values)
    os.replace(path.with_suffix('.npz.partial'), path)


def freeze_patient_split(records, cfg, base):
    folds=make_splits(records,cfg['split_seed'])
    frozen=dict(scope=SCOPE, frozen_at=now(), seed=cfg['split_seed'],
        algorithm='label-presence strata, seeded permutation, cyclic patient allocation',
        records=[{k:r[k] for k in ('image_id','patient_id','class_code','class_index')} for r in records],
        folds=folds, performance_used=False)
    path=Path(base)/'state/patient_split_frozen.json'
    if path.exists():
        previous=json.loads(path.read_text(encoding='utf8'))
        if any(previous.get(k)!=v for k,v in frozen.items() if k!='frozen_at'):
            raise ValueError('Existing frozen patient split differs; never reassign')
    else:
        write_json(path,frozen)
    return folds


def prepare(root=ROOT):
    root = Path(root)
    cfg = configuration(root)
    base = root / RUN
    artifacts = base / 'artifacts'
    artifacts.mkdir(parents=True, exist_ok=True)
    dataset_path = artifacts / 'dataset.json'
    if dataset_path.exists():
        raise FileExistsError('Prepared dataset is immutable; never overwrite')
    manifest_path = root / DEST / 'metadata/download_manifest.json'
    download = json.loads(manifest_path.read_text(encoding='utf8'))
    if download['status'] != 'COMPLETED_VERIFIED_SUBSET_DOWNLOAD' or len(download['files']) != 238:
        raise ValueError('Complete verified download required')
    authoritative={r['image_id']:r for r in metadata_records(root)}
    if len({r['image_id'] for r in download['files']})!=238:
        raise ValueError('Repeated download identity')
    official_md5=checksums(root)
    for source in download['files']:
        expected=authoritative.get(source['image_id'])
        if expected is None or any(source.get(k)!=value for k,value in expected.items()):
            raise ValueError('Download patient/class identity differs from authoritative CSV')
        expected_path=(DEST/'raw'/(source['image_id']+'.txt')).as_posix()
        if source['path']!=expected_path or source['published_md5']!=official_md5.get(source['archive_entry']):
            raise ValueError('Download path/checksum differs from selected published source')
        if source.get('zip_crc_pass') is not True or source.get('published_md5_pass') is not True:
            raise ValueError('Download integrity flags missing')
    cube_dir = artifacts / 'cubes'
    moments_dir = artifacts / 'core_moments'
    cube_dir.mkdir(exist_ok=True)
    moments_dir.mkdir(exist_ok=True)
    state_path = base / 'state/acquisition_status.json'
    state = dict(scope=SCOPE, status='PREPARING', stage='converting_cubes', updated_at=now(),
        completed_files=238, total_files=238, downloaded_bytes=sum(x['compressed_bytes'] for x in download['files']),
        total_bytes=sum(x['compressed_bytes'] for x in download['files']), converted_cores=0, total_cores=238, error=None)
    write_json(state_path, state)
    records, qcs, hashes = [], [], {}
    try:
        for source in download['files']:
            path = root / source['path']
            if sha(path) != source['sha256'] or sha(path, 'md5') != source['published_md5']:
                raise ValueError('Raw spectral file hash mismatch')
            cube, qc = parse_cube(path)
            identity = source['image_id']
            target = cube_dir / (identity + '.npy')
            if target.exists():
                prior = np.load(target, mmap_mode='r', allow_pickle=False)
                if prior.shape != cube.shape or not np.array_equal(prior, cube):
                    raise ValueError('Existing derived cube differs')
            else:
                with open(target.with_suffix('.npy.partial'), 'wb') as out:
                    np.save(out, cube, allow_pickle=False)
                os.replace(target.with_suffix('.npy.partial'), target)
            moments = core_moments(cube)
            moment_path = moments_dir / (identity + '.npz')
            if moment_path.exists():
                with np.load(moment_path, allow_pickle=False) as old:
                    if any(not np.array_equal(old[k], value) for k, value in moments.items()):
                        raise ValueError('Existing core moments differ')
            else:
                _save_npz(moment_path, **moments)
            records.append(dict(image_id=identity, patient_id=source['patient_id'],
                class_index=source['class_index'], class_code=source['class_code'], tma=source['tma'], core=source['core'],
                source=source['path'], source_sha256=source['sha256'], source_md5=source['published_md5'],
                cube=target.relative_to(root).as_posix(), cube_sha256=sha(target), shape=[88,88,814], dtype='float32',
                moments=moment_path.relative_to(root).as_posix(), moments_sha256=sha(moment_path)))
            qcs.append(dict(image_id=identity, **qc))
            hashes[identity] = qc['content_sha256']
            state.update(updated_at=now(), converted_cores=len(records), last_file=identity)
            write_json(state_path, state)
            print(f'PREPARE {len(records)}/238 {identity}', flush=True)
        by_id = {r['image_id']: r for r in records}
        duplicate_pairs = [dict(first=a, second=b, patient_a=by_id[a]['patient_id'], patient_b=by_id[b]['patient_id'])
            for j,a in enumerate(hashes) for b in list(hashes)[j+1:] if hashes[a] == hashes[b]]
        write_json(base / 'reports/duplicate_audit.json', dict(content_sha256=hashes,
            exact_duplicate_pairs=duplicate_pairs, near_duplicate_screen_performed=False,
            interpretation='Exact content equality only; patient identity is supplied by authoritative CSVs.'))
        if duplicate_pairs:
            raise ValueError('Exact duplicate cube content requires review before split')

        folds=freeze_patient_split(records,cfg,base)
        transform_reviews=[]
        for fold in folds:
            state.update(stage='train_only_transforms', updated_at=now(), transform_fold=fold['fold'])
            write_json(state_path, state)
            def items():
                for identity in fold['train_ids']:
                    with np.load(root / by_id[identity]['moments'], allow_pickle=False) as source:
                        yield {key:source[key].copy() for key in ('mean','m2','count')}
            values = merge_moments(items())
            values['train_ids'] = np.array(fold['train_ids'])
            values['train_patient_ids'] = np.array(fold['train_patient_ids'])
            _save_npz(artifacts / f'fold{fold["fold"]}_transform.npz', **values)
            _save_npz(artifacts / f'fold{fold["fold"]}_plan.npz', **make_plan(records, fold, cfg))
            transform_reviews.append(dict(fold=fold['fold'], fitted_spectra=int(values['fitted_spectra']),
                pca8_explained_variance_fraction=float(values['eigenvalues'][:8].sum()/values['eigenvalues'].sum()),
                std_min=float(values['std'].min()), train_ids=fold['train_ids'],
                train_patient_ids=fold['train_patient_ids'], heldout_statistics_used=False))
        original_hashes={r['source']:r['source_sha256'] for r in records}
        for name in ['checksums.md5','README.txt']+[f'TMA{i}_core_metadata.csv' for i in range(1,10)]:
            for folder in (root / DEST / 'metadata', root / META):
                source=folder/name
                original_hashes[source.relative_to(root).as_posix()]=sha(source)
        for name in ('download_manifest.json','selected_archive_entries.json'):
            source=root/DEST/'metadata'/name
            original_hashes[source.relative_to(root).as_posix()]=sha(source)
        dataset = dict(scope=SCOPE, prepared_at=now(), samples=records, folds=folds,
            classes=list(CLASSES), source_manifest=manifest_path.relative_to(root).as_posix(),
            source_manifest_sha256=sha(manifest_path), split_unit='authoritative patient ID',
            original_source_hashes=original_hashes,
            patient_count=75, core_count=238, independent_subject_ids_available=True,
            heldout_cube_count_per_initialization=238, transform_fitted_on='TRAIN cores only; fixed64x64 spread grid=4096spectra/core',
            spectral_bands=814, wavenumbers_cm_inverse=WAVENUMBERS.tolist(),
            class_sampling='uniformclass thenuniformTRAINpatientwithclass thencore',
            heldout_used_for_training_or_selection=False,
            interpretation='Independent FTIR core classification replication; not fixed ColonCancerHSI checkpoint external segmentation test')
        write_json(base / 'reports/cube_quality_review.json', dict(status='PASS', samples=qcs,
            source_crc_and_md5_pass=True, finite=True, expected_shape_all=True, no_pixel_labels_invented=True))
        write_json(base / 'reports/transforms_review.json', dict(status='PASS', folds=transform_reviews,
            covariance='float64 centred per-core moments combined by Welford formula; populationSD; PCA no whitening',
            pca_mean='Zero by definition after centering with the TRAIN mean; storedfloat32',
            grid=np.linspace(0,87,64).round().astype(int).tolist()))
        write_json(dataset_path, dataset)
        state.update(status='PREPARED', stage='patient_split_and_transforms_ready', updated_at=now())
        write_json(state_path, state)
        write_json(base / 'reports/data_preparation_review.json', dict(status='PASS', completed_at=now(),
            cores=238, patients=75, folds=3, class_counts={'NC':117,'CRC':121},
            dataset_sha256=sha(dataset_path), patient_split_sha256=sha(base / 'state/patient_split_frozen.json'),
            transform_reviews=transform_reviews, new_fit_or_inference=False))
        return dataset
    except BaseException as exc:
        state.update(status='FAILED', stage='data_preparation_failed', updated_at=now(), error=repr(exc))
        write_json(state_path, state)
        raise


class FoldData:
    def __init__(self, root, fold):
        self.root=Path(root)
        self.cfg=configuration(root)
        self.base=self.root/RUN/'artifacts'
        self.manifest=json.loads((self.base/'dataset.json').read_text(encoding='utf8'))
        self.records=self.manifest['samples']
        split=self.manifest['folds'][fold]
        if split['fold'] != fold: raise ValueError('Fold identity mismatch')
        self.train_ids=split['train_ids']
        self.heldout_ids=split['heldout_ids']
        self.train_patient_ids=split['train_patient_ids']
        self.heldout_patient_ids=split['heldout_patient_ids']
        if set(self.train_patient_ids)&set(self.heldout_patient_ids): raise ValueError('Patient leakage')
        self._by_id={r['image_id']:r for r in self.records}
        self._cubes={}
        with np.load(self.base/f'fold{fold}_transform.npz', allow_pickle=False) as t:
            self.mean,self.std=t['mean'].copy(),t['std'].copy()
            self.pca={'mean':t['pca_mean'].copy(),'components':t['components'].copy()}
            if t['train_ids'].tolist()!=self.train_ids or t['train_patient_ids'].tolist()!=self.train_patient_ids:
                raise ValueError('Transform belongs to other TRAIN patients/cores')
        plan=self.base/f'fold{fold}_plan.npz'
        self.input_plan_identity=sha(plan)
        with np.load(plan,allow_pickle=False) as p:
            self.plan={key:p[key].copy() for key in p.files}
        allowed={j for j,r in enumerate(self.records) if r['image_id'] in self.train_ids}
        if not set(np.unique(self.plan['image_indices'])).issubset(allowed):
            raise ValueError('Heldout core appears in training plan')

    def label(self,image_id): return self._by_id[image_id]['class_index']
    def patient(self,image_id): return self._by_id[image_id]['patient_id']
    def _cube(self,image_id):
        if image_id not in self._cubes:
            self._cubes[image_id]=np.load(self.root/self._by_id[image_id]['cube'],mmap_mode='r',allow_pickle=False)
        return self._cubes[image_id]
    def _normalise(self,patch):
        return ((np.asarray(patch,dtype=np.float32)-self.mean)/self.std).transpose(2,0,1)
    def training_batches(self,epoch):
        if not 1<=epoch<=self.cfg['epochs']: raise ValueError('Epoch outside plan')
        images=self.plan['image_indices'][epoch-1]
        coords=self.plan['coordinates'][epoch-1]
        augment=self.plan['augment'][epoch-1]
        size=self.cfg['patch_size']
        batch=self.cfg['physical_batch']
        for first in range(0,len(images),batch):
            ids,bags,labels=[],[],[]
            for pos in range(first,first+batch):
                image_id=self.records[int(images[pos])]['image_id']
                if image_id not in self.train_ids or self.patient(image_id) not in self.train_patient_ids:
                    raise ValueError('Heldout training access')
                cube=self._cube(image_id)
                patches=[]
                for (y,x),(rot,flip_y,flip_x) in zip(coords[pos],augment[pos]):
                    patch=cube[int(y):int(y)+size,int(x):int(x)+size,:]
                    patch=np.rot90(patch,int(rot))
                    if flip_y: patch=patch[::-1,:,:]
                    if flip_x: patch=patch[:,::-1,:]
                    patches.append(self._normalise(patch))
                ids.append(image_id)
                labels.append(self.label(image_id))
                bags.append(patches)
            yield dict(inputs=np.ascontiguousarray(bags,dtype=np.float32),
                       labels=np.asarray(labels,dtype=np.int64),image_ids=ids)
    def prediction_tiles(self,image_id):
        if image_id not in self.heldout_ids or self.patient(image_id) not in self.heldout_patient_ids:
            raise ValueError('Inference requested outside heldout patient set')
        cube=self._cube(image_id)
        size=self.cfg['patch_size']
        if size!=64: raise ValueError('Protocol fixed64patches')
        pending=[]
        for y,x in [(0,0),(0,24),(24,0),(24,24)]:
            pending.append(self._normalise(cube[y:y+size,x:x+size,:]))
            if len(pending)==self.cfg['inference_tile_batch']:
                yield np.ascontiguousarray(pending,dtype=np.float32)
                pending=[]
        if pending: yield np.ascontiguousarray(pending,dtype=np.float32)

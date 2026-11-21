from __future__ import annotations
import csv
import hashlib
from pathlib import Path
from types import FunctionType
import numpy as np
from colon_borkovits import data as original

EXTRA_CORE = 'BORKOVITS2026_TMA5_B1'
EXTRA_SHA256 = '2d96f4649a9e6283531ef2e8cb7c6cd6389aaf98edba5af30765d99d658aea56'
EXTRA_AXIS = np.arange(4000., 644., -4., dtype=np.float64)


def selected_columns(header, identity):

    if header[:3] != ['', 'X', 'Y']:
        raise ValueError('Unexpected spatial/index column header')
    try:
        axis = np.asarray(header[3:], dtype=np.float64)
    except ValueError as exc:
        raise ValueError('Nonnumeric spectral axis') from exc
    if np.array_equal(axis, original.WAVENUMBERS):
        return np.arange(3, 817, dtype=np.int64)
    if identity != EXTRA_CORE or not np.array_equal(axis, EXTRA_AXIS):
        raise ValueError('Unexpected spectral axis; only the published TMA5_B1 extension is allowed')
    indices = np.searchsorted(-axis, -original.WAVENUMBERS)
    if not np.array_equal(axis[indices], original.WAVENUMBERS):
        raise ValueError('Frozen 814-band axis is not an exact subset')
    return indices + 3


def parse_cube(path):
    path = Path(path)
    with path.open(encoding='utf8', newline='') as handle:
        header = next(csv.reader(handle))
    indices = selected_columns(header, path.stem)
    if len(header) == 817:
        return original.parse_cube(path)
    if original.sha(path) != EXTRA_SHA256:
        raise ValueError('Published extra-band core SHA256 differs')
    values = np.loadtxt(path, delimiter=',', skiprows=1, dtype=np.float64)
    if values.shape != (7744, 842) or not np.isfinite(values).all():
        raise ValueError('Unexpected full extra-band shape or nonfinite source values')
    if len(np.unique(values[:, 0])) != 7744:
        raise ValueError('Duplicate source row index')
    xs, xi = np.unique(values[:, 1], return_inverse=True)
    ys, yi = np.unique(values[:, 2], return_inverse=True)
    if len(xs) != 88 or len(ys) != 88 or len(np.unique(yi * 88 + xi)) != 7744:
        raise ValueError('Invalid or repeated XY spatial lattice')
    if not np.allclose(np.diff(xs), np.diff(xs)[0]) or not np.allclose(np.diff(ys), np.diff(ys)[0]):
        raise ValueError('Irregular XY grid')
    spectra = values[:, indices]
    cube = np.empty((88, 88, 814), dtype=np.float32)
    cube[yi, xi] = spectra.astype(np.float32)
    if not np.isfinite(cube).all():
        raise ValueError('Nonfinite float32 conversion')
    return cube, dict(shape=list(cube.shape), dtype=str(cube.dtype),
        x_values=xs.tolist(), y_values=ys.tolist(),
        spatial_order='ascending Y, ascending X, original descending wavenumber',
        raw_min=float(spectra.min()), raw_max=float(spectra.max()),
        source_nonfinite=0, float32_nonfinite=0,
        float32_max_abs_rounding=float(np.abs(cube[yi,xi].astype(np.float64)-spectra).max()),
        content_sha256=hashlib.sha256(cube.tobytes()).hexdigest(),
        source_spectral_bands=839, retained_spectral_bands=814,
        retained_wavenumbers_cm_inverse=original.WAVENUMBERS.tolist(),
        unused_source_wavenumbers_cm_inverse=EXTRA_AXIS[814:].tolist(),
        selection='Exact frozen 4000..748 cm^-1 band identity; no interpolation',
        source_sha256=EXTRA_SHA256, all_839_source_bands_finite=True)


_prepare_globals = dict(original.prepare.__globals__)
_prepare_globals['parse_cube'] = parse_cube
prepare = FunctionType(original.prepare.__code__, _prepare_globals,
                       original.prepare.__name__, original.prepare.__defaults__,
                       original.prepare.__closure__)
prepare.__kwdefaults__ = original.prepare.__kwdefaults__

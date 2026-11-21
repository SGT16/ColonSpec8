from __future__ import annotations

from collections.abc import Mapping
import time

import numpy as np

from fewlabel_gmtssa.tensor_ssa import iter_tiles, transform_tile

TILE_SHAPE = (230, 258)
TENSOR_WINDOW = 5
TENSOR_NEIGHBORS = 60
TENSOR_CHUNK = 256
CLASS_CHUNK = 2048
BANDS = 351
CLASS_RANK = 20


def _validate_models(models):
    if not isinstance(models, Mapping) or not models:
        raise ValueError("NONEMPTY_SEED_TO_FROZEN_MODEL_MAPPING_REQUIRED")
    for seed, model in models.items():
        if not isinstance(seed, int) or not 100 <= seed <= 999:
            raise ValueError("PRESPECIFIED_THREE_DIGIT_SEED_REQUIRED")
        if (model.n_bands != BANDS or model.rank != CLASS_RANK
                or model.means is None or model.bases is None):
            raise ValueError("FROZEN_351_BAND_RANK20_CLASSIFIER_REQUIRED")


def _score_features(features, models):

    values = np.asarray(features, dtype=np.float64)
    if (values.ndim != 3 or values.shape[2] != BANDS
            or not np.isfinite(values).all()):
        raise ValueError("FINITE_HWC_351_FEATURE_TILE_REQUIRED")
    height, width, bands = values.shape
    flat = values.reshape(-1, bands)
    output = {}
    for seed, model in models.items():
        scores = np.empty((len(flat), 2), dtype=np.float64)
        for start in range(0, len(flat), CLASS_CHUNK):
            block = model.decision_function(flat[start:start + CLASS_CHUNK])
            if block.shape != (min(CLASS_CHUNK, len(flat) - start), 2) or not np.isfinite(block).all():
                raise FloatingPointError("INVALID_FROZEN_GM_SCORES")
            scores[start:start + CLASS_CHUNK] = block
        output[seed] = scores.reshape(height, width, 2)
    return output


def infer_gm_tile(normalized_chw_numpy, models):


    _validate_models(models)
    value = np.asarray(normalized_chw_numpy)
    if (value.ndim != 3 or value.shape[0] != BANDS or value.dtype != np.float64
            or min(value.shape[1:]) < 1 or not np.isfinite(value).all()):
        raise ValueError("FINITE_NORMALIZED_FLOAT64_CHW_351_TILE_REQUIRED")
    features = transform_tile(value.transpose(1, 2, 0), u=TENSOR_WINDOW,
                              neighbors=TENSOR_NEIGHBORS, chunk_pixels=TENSOR_CHUNK)
    return _score_features(features, models)


def infer_gm_image(provider, image_id, models, progress=None):


    _validate_models(models)
    height, width = tuple(provider.image_shape(image_id))
    if min(height, width) < 1:
        raise ValueError("POSITIVE_IMAGE_GEOMETRY_REQUIRED")
    result = {seed: np.empty((height, width, 2), dtype=np.float64) for seed in models}
    locations = list(iter_tiles(height, width, TILE_SHAPE))
    started = time.monotonic()
    for index, bounds in enumerate(locations):
        y0, y1, x0, x1 = bounds
        raw = np.asarray(provider.get_gm_tile(image_id, bounds))
        if raw.shape != (y1-y0, x1-x0, BANDS) or raw.dtype != np.float64:
            raise ValueError("PROVIDER_GM_TILE_CONTRACT_MISMATCH")
        scores = infer_gm_tile(raw.transpose(2, 0, 1), models)
        for seed, values in scores.items():
            result[seed][y0:y1, x0:x1] = values
        if progress is not None:
            progress(image_id=image_id, tile=index+1, tiles=len(locations),
                     elapsed_s=time.monotonic()-started)
    return result

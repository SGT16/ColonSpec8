from __future__ import annotations

from pathlib import Path
import numpy as np

from .grassmann import ClassStatistics, squared_residuals

MODEL_VARIANT = "gm_tensorssa_train_only_inductive_v1"
MEAN_CONVENTION = "released_matlab_mean_normalized_then_center_raw"


class TrainOnlyGMTSSA:
    def __init__(self, n_bands: int = 351, rank: int = 20):
        self.n_bands, self.rank = n_bands, rank
        self.statistics = ClassStatistics(n_bands)
        self.means = self.bases = None

    def update(self, features, labels, *, split: str = "TRAIN"):
        if self.means is not None:
            raise RuntimeError("FINALIZED_MODEL_IS_IMMUTABLE")
        self.statistics.update(features, labels, split=split)
        return self

    def finalize(self):
        if self.means is not None:
            raise RuntimeError("MODEL_ALREADY_FINALIZED")
        self.means, self.bases = self.statistics.finalize(self.rank)
        self.means.setflags(write=False)
        self.bases.setflags(write=False)
        return self

    def fit(self, features, labels, *, split: str = "TRAIN"):
        if self.means is not None or self.statistics.counts.sum():
            raise RuntimeError("FITTING_REQUIRES_FRESH_MODEL")
        return self.update(features, labels, split=split).finalize()

    def decision_function(self, features):
        if self.means is None:
            raise RuntimeError("MODEL_NOT_FITTED")
        return -squared_residuals(features, self.means, self.bases)

    def predict_features(self, features):

        return self.decision_function(features).argmax(axis=1).astype(np.int8)

    def save(self, path):
        if self.means is None:
            raise RuntimeError("MODEL_NOT_FITTED")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("wb") as handle:
            np.savez_compressed(handle, variant=np.array(MODEL_VARIANT),
                mean_convention=np.array(MEAN_CONVENTION), n_bands=np.array(self.n_bands),
                rank=np.array(self.rank), counts=self.statistics.counts,
                means=self.means, bases=self.bases)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as archive:
            if str(archive["variant"]) != MODEL_VARIANT or str(archive["mean_convention"]) != MEAN_CONVENTION:
                raise ValueError("MODEL_VARIANT_OR_MEAN_CONVENTION_MISMATCH")
            obj = cls(int(archive["n_bands"]), int(archive["rank"]))
            obj.statistics.counts = archive["counts"].astype(np.int64)
            obj.means, obj.bases = archive["means"].copy(), archive["bases"].copy()
        if (obj.means.shape != (2, obj.n_bands) or obj.bases.shape != (2, obj.n_bands, obj.rank)
                or obj.statistics.counts.shape != (2,) or (obj.statistics.counts < obj.rank + 1).any()
                or not np.isfinite(obj.means).all() or not np.isfinite(obj.bases).all()):
            raise ValueError("INVALID_MODEL_ARTIFACT")
        for basis in obj.bases:
            if not np.allclose(basis.T @ basis, np.eye(obj.rank), atol=1e-10, rtol=1e-10):
                raise ValueError("MODEL_BASIS_NOT_ORTHONORMAL")
        obj.means.setflags(write=False)
        obj.bases.setflags(write=False)
        return obj

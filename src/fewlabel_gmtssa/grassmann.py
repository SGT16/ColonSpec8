from __future__ import annotations

import numpy as np


def checked_features(features, n_bands: int) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != n_bands or not np.isfinite(x).all():
        raise ValueError("FINITE_N_BY_B_FEATURES_REQUIRED")
    return x


class ClassStatistics:


    def __init__(self, n_bands: int):
        if not isinstance(n_bands, int) or n_bands < 1:
            raise ValueError("POSITIVE_BAND_COUNT_REQUIRED")
        self.n_bands = n_bands
        self.counts = np.zeros(2, dtype=np.int64)
        self.raw_sum = np.zeros((2, n_bands), dtype=np.float64)
        self.unit_sum = np.zeros_like(self.raw_sum)
        self.cross_sum = np.zeros((2, n_bands, n_bands), dtype=np.float64)

    def update(self, features, labels, *, split: str = "TRAIN"):
        if split != "TRAIN":
            raise ValueError("ONLY_TRAIN_LABELS_MAY_FIT_SUBSPACES")
        x = checked_features(features, self.n_bands)
        y = np.asarray(labels)
        if y.shape != (len(x),) or not np.isin(y, (0, 1)).all():
            raise ValueError("REVEALED_BINARY_TRAIN_LABELS_ONLY")
        for c in range(2):
            rows = x[y == c]
            if not len(rows):
                continue
            norms = np.linalg.norm(rows, axis=1, keepdims=True)
            unit = np.divide(rows, norms, out=np.zeros_like(rows), where=norms > 0)
            self.counts[c] += len(rows)
            self.raw_sum[c] += rows.sum(axis=0)
            self.unit_sum[c] += unit.sum(axis=0)
            self.cross_sum[c] += rows.T @ rows
        return self

    def finalize(self, rank: int = 20):
        if not isinstance(rank, int) or not 1 <= rank <= self.n_bands:
            raise ValueError("INVALID_GRASSMANN_RANK")

        if (self.counts < rank + 1).any():
            raise ValueError("INSUFFICIENT_REVEALED_TRAIN_SUPPORT_FOR_FIXED_RANK")
        means = self.unit_sum / self.counts[:, None]
        bases = np.empty((2, self.n_bands, rank), dtype=np.float64)
        for c in range(2):
            mu, sums, count = means[c], self.raw_sum[c], self.counts[c]
            gram = (self.cross_sum[c] - np.outer(sums, mu)
                    - np.outer(mu, sums) + count * np.outer(mu, mu))
            gram = (gram + gram.T) * .5
            eigval, eigvec = np.linalg.eigh(gram)
            tolerance = np.finfo(np.float64).eps * max(1., float(eigval[-1])) * self.n_bands * 20
            if eigval[0] < -tolerance:
                raise FloatingPointError("NON_PSD_TRAIN_SCATTER")
            if eigval[-rank] <= tolerance:
                raise ValueError("RANK_DEFICIENT_TRAIN_SUBSPACE_NO_FALLBACK")
            bases[c] = eigvec[:, -rank:][:, ::-1]
        return means, bases


def squared_residuals(features, means, bases):
    means = np.asarray(means, dtype=np.float64)
    bases = np.asarray(bases, dtype=np.float64)
    if means.ndim != 2 or means.shape[0] != 2 or bases.shape[:2] != means.shape:
        raise ValueError("INVALID_CLASS_SUBSPACE_SHAPE")
    x = checked_features(features, means.shape[1])
    distances = np.empty((len(x), 2), dtype=np.float64)
    for c in range(2):
        centered = x - means[c]

        residual = centered - (centered @ bases[c]) @ bases[c].T
        distances[:, c] = np.einsum("nb,nb->n", residual, residual)
    return distances

from __future__ import annotations

from collections.abc import Iterator
import numpy as np


def iter_tiles(height: int, width: int,
               tile_shape: tuple[int, int] = (230, 258)) -> Iterator[tuple[int, int, int, int]]:

    if min(height, width, *tile_shape) < 1:
        raise ValueError("POSITIVE_TILE_AND_IMAGE_DIMENSIONS_REQUIRED")
    th, tw = tile_shape
    for y in range(0, height, th):
        for x in range(0, width, tw):
            yield y, min(y + th, height), x, min(x + tw, width)


def _validate(cube: np.ndarray, u: int, neighbors: int, chunk_pixels: int):
    x = np.asarray(cube, dtype=np.float64)
    if x.ndim != 3 or min(x.shape) < 1 or not np.isfinite(x).all():
        raise ValueError("FINITE_NONEMPTY_HWC_CUBE_REQUIRED")
    if not isinstance(u, int) or u < 0:
        raise ValueError("NONNEGATIVE_INTEGER_RADIUS_REQUIRED")
    if not isinstance(neighbors, int) or not 1 <= neighbors <= (2 * u + 1) ** 2:
        raise ValueError("INVALID_ADAPTIVE_NEIGHBOR_COUNT")
    if not isinstance(chunk_pixels, int) or chunk_pixels < 1:
        raise ValueError("POSITIVE_CHUNK_REQUIRED")
    return x


def _offsets(u: int):

    side = 2 * u + 1
    return np.tile(np.arange(side), side), np.repeat(np.arange(side), side)


def _embedding(padded, height, width, u, neighbors, chunk_pixels):
    dy, dx = _offsets(u)
    for start in range(0, height * width, chunk_pixels):
        pos = np.arange(start, min(start + chunk_pixels, height * width))
        yy, xx = pos // width, pos % width
        patches = padded[yy[:, None] + dy[None], xx[:, None] + dx[None], :]
        norms = np.linalg.norm(patches, axis=2, keepdims=True)
        normalized = np.divide(patches, norms, out=np.zeros_like(patches), where=norms > 0)
        center = normalized[:, ((2 * u + 1) ** 2) // 2, :]


        distance = np.linalg.norm(normalized - center[:, None, :], axis=2)
        selected = np.argsort(distance, axis=1, kind="stable")[:, :neighbors]
        selected_values = np.take_along_axis(patches, selected[:, :, None], axis=1)
        yield yy, xx, selected, selected_values


def transform_tile(cube: np.ndarray, *, u: int = 5, neighbors: int = 60,
                   chunk_pixels: int = 256) -> np.ndarray:


    x = _validate(cube, u, neighbors, chunk_pixels)
    height, width, bands = x.shape
    padded = np.pad(x, ((u, u), (u, u), (0, 0)), mode="symmetric")
    frequencies = bands // 2 + 1
    gram = np.zeros((frequencies, neighbors, neighbors), dtype=np.complex128)
    for _, _, _, values in _embedding(padded, height, width, u, neighbors, chunk_pixels):

        fourier = np.fft.rfft(values, axis=2).transpose(2, 1, 0)
        gram += fourier @ fourier.conj().transpose(0, 2, 1)
    gram = (gram + gram.conj().transpose(0, 2, 1)) * .5
    _, eigvec = np.linalg.eigh(gram)
    leading = eigvec[:, :, -1]
    reconstructed = np.zeros(padded.shape, dtype=np.float64)
    repeats = np.zeros(padded.shape[:2], dtype=np.int64)
    dy, dx = _offsets(u)
    for yy, xx, selected, values in _embedding(padded, height, width, u, neighbors, chunk_pixels):
        fourier = np.fft.rfft(values, axis=2).transpose(2, 1, 0)
        projected = np.einsum("fl,fln->fn", leading.conj(), fourier, optimize=True)
        recovered_fft = leading[:, :, None] * projected[:, None, :]
        recovered = np.fft.irfft(recovered_fft.transpose(2, 1, 0), n=bands, axis=2)
        ry, rx = yy[:, None] + dy[selected], xx[:, None] + dx[selected]


        np.add.at(reconstructed, (ry.ravel(), rx.ravel()), recovered.reshape(-1, bands))
        np.add.at(repeats, (ry.ravel(), rx.ravel()), 1)
    np.divide(reconstructed, repeats[:, :, None], out=reconstructed,
              where=repeats[:, :, None] > 0)
    result = reconstructed[u:u + height, u:u + width]
    if not np.isfinite(result).all():
        raise FloatingPointError("NONFINITE_TENSORSSA_FEATURES")
    return result

from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


CLASS_SOURCE = "https://github.com/ikopriva/ColonCancerHSI/blob/main/Grassman_HSI.m"
CLASS_EVIDENCE = (
    "Source uses patch_labels==(i-1), describes X0 as Noncancer and X1 as Cancer, "
    "and computes TP from logical(TestLabels). Canonical 0=noncancer, 1=cancer."
)
PREPARATION_VERSION = "2.0.0-data1"
REAL_SHAPE = (351, 1035, 1384)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".pending")
    pending.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(pending, path)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _guard_development_path(raw: Path, name: str) -> Path:

    import re
    allowed = name == "GT_train.mat" or bool(re.fullmatch(r"(?:HSI_train_[1-9][0-9]*\.h5|psdRGB_train_[1-9][0-9]*\.tiff)", name))
    if not allowed:
        raise ValueError(f"Forbidden development input basename: {name}")
    path = raw / name
    if path.resolve().parent != raw.resolve():
        raise ValueError("Raw input escapes the development data directory")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def derive_seed(seed: int, namespace: str, image_id: str = "") -> int:
    value = f"{int(seed)}|{namespace}|{image_id}".encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big")


def fixed_split(image_ids: list[str], validation_seed: int, n_validation: int) -> dict:
    if len(set(image_ids)) != len(image_ids) or not 0 < n_validation < len(image_ids):
        raise ValueError("Invalid development identifiers or validation count")
    order = sorted(image_ids, key=lambda x: hashlib.sha256(f"{validation_seed}|{x}".encode()).hexdigest())
    validation_ids = order[:n_validation]
    train_ids = [image_id for image_id in image_ids if image_id not in validation_ids]
    payload = {
        "schema_version": 1,
        "protocol_version": "2.0.0",
        "policy": "SHA256(validation_seed|image_id)_ascending_first_n_validation",
        "validation_seed": validation_seed,
        "development_ids": image_ids,
        "train_ids": train_ids,
        "val_ids": validation_ids,
        "patient_ids": {image_id: None for image_id in image_ids},
        "validation_patient_disjointness": "NOT_VERIFIED",
        "final_test_status": "LOCKED",
    }
    return {**payload, "split_sha256": digest_json(payload)}


def _save_immutable_json(path: Path, expected: dict) -> None:
    if path.exists():
        if _read_json(path) != expected:
            raise RuntimeError(f"Refusing to replace a frozen artifact: {path}")
    else:
        atomic_json(path, expected)


def nested_coordinates(n_pixels: int, fractions: list[float], seed: int, image_id: str) -> dict[str, np.ndarray]:

    if n_pixels <= 0 or not fractions or any(not 0 < fraction <= 1 for fraction in fractions):
        raise ValueError("Invalid pixel count or label fractions")
    counts = [math.floor(n_pixels * float(fraction)) for fraction in fractions]
    rng = np.random.default_rng(derive_seed(seed, "label_coordinates", image_id))
    indices = rng.choice(n_pixels, size=max(counts), replace=False).astype(np.int64)
    return {fraction_key(fraction): indices[:count].copy() for fraction, count in zip(fractions, counts)}


def fraction_key(fraction: float) -> str:
    return format(float(fraction), ".8g")


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    x = a.astype(np.float64, copy=False).ravel()
    y = b.astype(np.float64, copy=False).ravel()
    x = x - x.mean()
    y = y - y.mean()
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(np.dot(x, y) / denominator) if denominator > 0 else 0.0


def validate_alignment(hsi_dataset: Any, rgb_path: Path, expected_shape: tuple[int, int, int]) -> dict:


    from PIL import Image
    if tuple(hsi_dataset.shape) != expected_shape:
        raise ValueError(f"Unexpected HDF geometry: {hsi_dataset.shape}")
    bands, width, height = expected_shape
    with Image.open(rgb_path) as image:
        if image.size != (width, height):
            raise ValueError("Pseudo-RGB geometry does not match MATLAB HSI geometry")
        rgb = np.asarray(image.convert("RGB"), dtype=np.float64)
    stride = max(1, min(height, width) // 64)
    gray = rgb[::stride, ::stride].mean(axis=-1)
    del rgb
    band_ids = sorted(set([bands // 7, bands // 2, 5 * bands // 7]))
    scores: list[list[float]] = []
    for band in band_ids:
        plane = np.asarray(hsi_dataset[band, ::stride, ::stride], dtype=np.float64).T
        if plane.shape != gray.shape or not np.isfinite(plane).all():
            raise ValueError("Invalid sampled HSI geometry/values")
        scores.append([_correlation(candidate, gray) for candidate in (
            plane, plane[::-1], plane[:, ::-1], plane[::-1, ::-1],
        )])

    chosen = max(range(len(scores)), key=lambda index: scores[index][0])
    selected = scores[chosen]
    margin = selected[0] - max(selected[1:])
    passed = selected[0] >= 0.50 and margin >= 0.10
    evidence = {
        "status": "PASS" if passed else "FAIL",
        "hdf_to_chw_axes": [0, 2, 1],
        "hdf_to_hwc_axes": [2, 1, 0],
        "selected_band_index": band_ids[chosen],
        "identity_luminance_correlation": selected[0],
        "reflection_correlations": selected[1:],
        "identity_margin": margin,
        "minimum_identity_correlation": 0.50,
        "minimum_margin": 0.10,
        "gt_alignment_basis": "Official Grassman_HSI.m pairs GT_train(:,:,i) with HSI_train_i; no GT-dependent orientation selection",
    }
    if not passed:
        raise ValueError(f"HSI/pseudoRGB registration gate failed: {json.dumps(evidence)}")
    return evidence


def _contract(root: Path, cfg: dict) -> tuple[dict, dict]:
    seeds = _read_json(root / cfg["split"]["persistent_seeds"])
    all_seeds = seeds["main_seeds"] + seeds["pilot_seeds"] + [seeds["validation_seed"]]
    if len(all_seeds) != 13 or len(set(all_seeds)) != 13 or any(not isinstance(seed, int) or not 100 <= seed <= 999 for seed in all_seeds):
        raise ValueError("Expected the frozen 13 distinct three-digit seeds")
    if len(seeds["main_seeds"]) != 10 or len(seeds["pilot_seeds"]) != 2:
        raise ValueError("Protocol requires ten main and two separate pilot seeds")
    if cfg["protocol_version"] != "2.0.0" or cfg["split"]["development_images"] != 17 or cfg["split"]["validation_images"] != 3:
        raise ValueError("This preparation implements the fixed 17-image development/3-image validation protocol only")
    if cfg["data"]["expected_bands"] != 351 or cfg["labels"]["ignore_index"] != -1:
        raise ValueError("Unexpected bands or sparse-label ignore value")
    payload = {
        "preparation_version": PREPARATION_VERSION,
        "protocol_version": cfg["protocol_version"],
        "seed_manifest_sha256": file_sha256(root / cfg["split"]["persistent_seeds"]),
        "seeds": seeds,
        "fractions": cfg["labels"]["fractions"],
        "class_map": {"0": "noncancer", "1": "cancer"},
        "raw_shape": list(REAL_SHAPE),
        "cache_layout": "float32_CHW_per_image_npy_mmap",
        "normalization": "training_images_only_per_band_population_mean_std_all_pixels_label_blind",
        "normalization_variance_floor": 1e-12,
        "validation_images": 3,
        "validation_seed": seeds["validation_seed"],
        "raw_path": cfg["paths"]["raw"],
    }
    return payload, seeds


def _prepared_directory(root: Path, cfg: dict) -> Path:
    return root / cfg["paths"]["processed"] / "pilot_v2"


def _write_cube(hsi_path: Path, rgb_path: Path, target: Path, source_sha: str) -> dict:
    import h5py
    metadata_path = target.with_suffix(".json")
    if target.exists() and metadata_path.exists():
        metadata = _read_json(metadata_path)
        stat = target.stat()
        if (metadata.get("source_sha256") == source_sha and
                metadata.get("size_bytes") == stat.st_size and
                metadata.get("mtime_ns") == stat.st_mtime_ns and
                metadata.get("preparation_version") == PREPARATION_VERSION):
            return metadata
        raise RuntimeError(f"Prepared cube identity changed; explicit repair required: {target.name}")
    target.parent.mkdir(parents=True, exist_ok=True)
    pending = target.with_name(target.name + ".pending")
    with h5py.File(hsi_path, "r") as source:
        if "/img" not in source:
            raise ValueError("Missing required /img dataset")
        dataset = source["/img"]
        alignment = validate_alignment(dataset, rgb_path, REAL_SHAPE)
        bands, width, height = REAL_SHAPE
        output = np.lib.format.open_memmap(pending, mode="w+", dtype=np.float32, shape=(bands, height, width))
        payload_hash = hashlib.sha256()
        means, squares = np.empty(bands, dtype=np.float64), np.empty(bands, dtype=np.float64)
        try:
            for start in range(0, bands, 8):
                stop = min(start + 8, bands)
                raw = np.asarray(dataset[start:stop], dtype=np.float64)
                if not np.isfinite(raw).all():
                    raise ValueError(f"Nonfinite development input in {hsi_path.name}")
                chunk = np.ascontiguousarray(raw.transpose(0, 2, 1), dtype=np.float32)
                if not np.isfinite(chunk).all():
                    raise ValueError("Float32 conversion overflow")
                output[start:stop] = chunk
                payload_hash.update(chunk.tobytes(order="C"))

                means[start:stop] = np.mean(chunk, axis=(1, 2), dtype=np.float64)
                squares[start:stop] = np.mean(np.square(chunk, dtype=np.float64), axis=(1, 2), dtype=np.float64)
                output.flush()
                if hasattr(output._mmap, "madvise"):
                    import mmap
                    output._mmap.madvise(mmap.MADV_DONTNEED)
                del raw, chunk
        finally:
            output.flush()
            output._mmap.close()
    os.replace(pending, target)
    stat = target.stat()
    metadata = {
        "preparation_version": PREPARATION_VERSION,
        "source_sha256": source_sha,
        "cache_payload_sha256": payload_hash.hexdigest(),
        "cache_payload_hash_scope": "little/native-endian float32 CHW C-order payload, excluding npy header",
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "shape": [bands, height, width],
        "dtype": "float32",
        "finite_check": "PASS_ALL_DEVELOPMENT_PIXELS",
        "alignment": alignment,
        "band_mean": means.tolist(),
        "band_mean_square": squares.tolist(),
        "pixels": height * width,
    }
    atomic_json(metadata_path, metadata)
    return metadata


def prepare(root: str | Path, cfg: dict) -> dict:

    from scipy.io import loadmat
    root = Path(root).resolve()
    status_path = root / "state" / "data_preparation.json"
    status = {"status": "PREPARING", "started_at_utc": utc_now(), "final_test_status": "LOCKED", "final_test_decoded": False}
    atomic_json(status_path, status)
    try:
        contract, seeds = _contract(root, cfg)
        destination = _prepared_directory(root, cfg)
        destination.mkdir(parents=True, exist_ok=True)
        _save_immutable_json(destination / "contract.json", contract)
        completion = destination / "prepared.json"
        if completion.exists():
            report = _read_json(completion)
            if report.get("contract_sha256") != digest_json(contract):
                raise RuntimeError("Completed preparation has a different frozen contract")

            load_prepared(root, cfg, seeds["pilot_seeds"][0])
            atomic_json(status_path, {**status, "status": "READY_REUSED", "finished_at_utc": utc_now(), "report": str(completion.relative_to(root))})
            return report

        source_integrity = _read_json(root / "data" / "metadata" / "download_integrity.json")
        if source_integrity.get("completed_files") != 57 or not source_integrity.get("download_metadata_prepared"):
            raise RuntimeError("Completed download/hash/header audit is required")
        source_by_name = {entry["name"]: entry for entry in source_integrity["files"]}
        image_ids = [f"HSI_train_{index}" for index in range(1, 18)]
        split = fixed_split(image_ids, seeds["validation_seed"], 3)
        _save_immutable_json(root / cfg["split"]["validation_manifest"], split)
        raw = root / cfg["paths"]["raw"]
        gt_path = _guard_development_path(raw, "GT_train.mat")
        gt = loadmat(gt_path, variable_names=["GT_train"])["GT_train"]
        if tuple(gt.shape) != (REAL_SHAPE[2], REAL_SHAPE[1], 17):
            raise ValueError(f"Unexpected development GT shape: {gt.shape}")
        if not np.isin(gt, [0, 1]).all():
            raise ValueError("Development GT has unknown/nonfinite class codes")

        gt = gt.astype(np.int8, copy=False)
        images: dict[str, dict] = {}
        mask_index: dict[str, dict] = {}
        all_seeds = seeds["main_seeds"] + seeds["pilot_seeds"]
        for image_index, image_id in enumerate(image_ids):
            status.update({"phase": "DEVELOPMENT_FINITE_SCAN_CACHE_AND_MASKS", "image_id": image_id, "completed_images": image_index, "total_images": 17})
            atomic_json(status_path, status)
            hsi_name = f"{image_id}.h5"
            rgb_name = f"psdRGB_train_{image_index + 1}.tiff"
            hsi_path = _guard_development_path(raw, hsi_name)
            rgb_path = _guard_development_path(raw, rgb_name)
            for name, path in [(hsi_name, hsi_path), (rgb_name, rgb_path), ("GT_train.mat", gt_path)]:
                entry = source_by_name.get(name)
                if not entry or entry.get("status") != "COMPLETED" or path.stat().st_size != entry["actual_size_bytes"]:
                    raise RuntimeError(f"Downloaded source no longer matches completed audit: {name}")
            target = destination / "cubes" / f"{image_id}.npy"
            cube = _write_cube(hsi_path, rgb_path, target, source_by_name[hsi_name]["sha256"])
            images[image_id] = {"path": str(target.relative_to(root)), "metadata_path": str(target.with_suffix(".json").relative_to(root)),
                                "metadata_sha256": file_sha256(target.with_suffix(".json")), "shape": cube["shape"], "patient_id": None}
            dense = gt[:, :, image_index].reshape(-1)
            for seed in all_seeds:
                coordinates = nested_coordinates(dense.size, cfg["labels"]["fractions"], seed, image_id)
                for key, indices in coordinates.items():
                    mask_path = destination / "masks" / str(seed) / key / f"{image_id}.npz"
                    mask_path.parent.mkdir(parents=True, exist_ok=True)
                    values = dense[indices]
                    if mask_path.exists():
                        with np.load(mask_path, allow_pickle=False) as prior:
                            if not np.array_equal(prior["indices"], indices) or not np.array_equal(prior["labels"], values):
                                raise RuntimeError("Existing mask differs from frozen seed/source; refusing reselection")
                    else:
                        pending = mask_path.with_name(mask_path.name + ".pending")
                        with pending.open("wb") as stream:
                            np.savez_compressed(stream, indices=indices, labels=values, shape=np.array(gt.shape[:2], dtype=np.int64))
                        os.replace(pending, mask_path)
                    mask_index.setdefault(str(seed), {}).setdefault(key, {})[image_id] = {
                        "path": str(mask_path.relative_to(root)), "sha256": file_sha256(mask_path), "selected_pixels": int(indices.size),
                    }
        del gt, dense

        train_cubes = [_read_json(root / images[image_id]["metadata_path"]) for image_id in split["train_ids"]]
        train_pixels = sum(item["pixels"] for item in train_cubes)
        mean = sum(np.array(item["band_mean"]) * item["pixels"] for item in train_cubes) / train_pixels
        mean_square = sum(np.array(item["band_mean_square"]) * item["pixels"] for item in train_cubes) / train_pixels
        variance = np.maximum(mean_square - mean * mean, 1e-12)
        std = np.sqrt(variance)
        normalization = {"mean": mean.tolist(), "std": std.tolist(), "fit_image_ids": split["train_ids"], "fit_pixels": train_pixels,
                         "ddof": 0, "variance_floor": 1e-12, "uses_gt": False, "validation_fit": False, "test_fit": False}
        atomic_json(destination / "normalization.json", normalization)
        label_budget = {}
        for seed in all_seeds:
            label_budget[str(seed)] = {}
            for key, masks in mask_index[str(seed)].items():
                label_budget[str(seed)][key] = {
                    "train": sum(masks[image_id]["selected_pixels"] for image_id in split["train_ids"]),
                    "validation": sum(masks[image_id]["selected_pixels"] for image_id in split["val_ids"]),
                    "development": sum(item["selected_pixels"] for item in masks.values()),
                    "development_pixels": 17 * REAL_SHAPE[1] * REAL_SHAPE[2],
                    "dense_hidden_labels_available_to_training": False,
                }

        union = {fraction_key(fraction): 0 for fraction in cfg["labels"]["fractions"]}
        for image_id in image_ids:
            for key in union:
                selected = np.zeros(REAL_SHAPE[1] * REAL_SHAPE[2], dtype=bool)
                for seed in seeds["main_seeds"]:
                    with np.load(root / mask_index[str(seed)][key][image_id]["path"], allow_pickle=False) as mask:
                        selected[mask["indices"]] = True
                union[key] += int(selected.sum())
        report = {
            "status": "READY", "preparation_version": PREPARATION_VERSION, "finished_at_utc": utc_now(),
            "contract_sha256": digest_json(contract), "split_sha256": split["split_sha256"],
            "images": images, "masks": mask_index, "label_budget": label_budget,
            "main_10_seed_union_selected_pixels_by_fraction": union,
            "normalization_path": str((destination / "normalization.json").relative_to(root)),
            "normalization_sha256": file_sha256(destination / "normalization.json"),
            "class_map": {"0": "noncancer", "1": "cancer"}, "class_source": CLASS_SOURCE, "class_evidence": CLASS_EVIDENCE,
            "final_test_status": "LOCKED", "final_test_decoded": False, "patient_metrics_available": False,
            "memory_policy": "No persistent per-worker cube cache; each patch opens a read-only mmap, copies only its patch, then closes the mapping. OS file cache is managed by OS.",
            "source_audit_sha256": file_sha256(root / "data" / "metadata" / "download_integrity.json"),
        }
        atomic_json(completion, report)
        atomic_json(status_path, {**status, "status": "READY", "completed_images": 17, "finished_at_utc": utc_now(), "report": str(completion.relative_to(root))})
        return report
    except Exception as exc:
        atomic_json(status_path, {**status, "status": "FAILED", "failed_at_utc": utc_now(), "error_type": type(exc).__name__, "error": str(exc)})
        raise


class PreparedData:

    def __init__(self, root: Path, report: dict, split: dict, normalization: dict, seed: int, fraction: float):
        self.root = Path(root)
        self.train_ids = list(split["train_ids"])
        self.val_ids = list(split["val_ids"])
        self.seed = int(seed)
        self.fraction = float(fraction)
        self.images = report["images"]


        self.local_cubes: dict[str, Path] = {}
        self.mean = np.asarray(normalization["mean"], dtype=np.float32)[:, None, None]
        self.std = np.asarray(normalization["std"], dtype=np.float32)[:, None, None]
        self.masks: dict[str, np.ndarray] = {}
        self.mask_sha256: dict[str, str] = {}
        if np.any(self.std <= 0) or not np.isfinite(self.mean).all() or not np.isfinite(self.std).all():
            raise ValueError("Invalid frozen normalization")
        if normalization["fit_image_ids"] != self.train_ids:
            raise ValueError("Normalization was not fitted on exactly the frozen TRAIN images")
        selected = report["masks"][str(seed)][fraction_key(fraction)]
        for image_id in self.train_ids + self.val_ids:
            if image_id not in self.images or not image_id.startswith("HSI_train_"):
                raise ValueError("Prepared loader is development-only")
            height, width = self.image_shape(image_id)
            item = selected[image_id]
            path = self.root / item["path"]
            actual_hash = file_sha256(path)
            if actual_hash != item["sha256"]:
                raise RuntimeError("Frozen sparse mask hash mismatch")
            with np.load(path, allow_pickle=False) as mask:
                indices, labels = mask["indices"], mask["labels"]
                if tuple(mask["shape"]) != (height, width):
                    raise ValueError("Sparse mask shape mismatch")
                if (indices.ndim != 1 or labels.shape != indices.shape or
                        indices.size != math.floor(height * width * fraction) or
                        not np.issubdtype(indices.dtype, np.integer) or
                        np.unique(indices).size != indices.size or
                        (indices.size and (indices.min() < 0 or indices.max() >= height * width)) or
                        not np.isin(labels, [0, 1]).all()):
                    raise ValueError("Invalid sparse mask contents or budget")
                sparse = np.full((height, width), -1, dtype=np.int8)
                sparse.reshape(-1)[indices] = labels
            sparse.setflags(write=False)
            self.masks[image_id] = sparse
            self.mask_sha256[image_id] = actual_hash

    def image_shape(self, image_id: str) -> tuple[int, int]:
        if image_id not in self.images or not image_id.startswith("HSI_train_"):
            raise ValueError("Unknown or non-development image")
        return tuple(self.images[image_id]["shape"][-2:])

    def enable_local_io_cache(self, cache_root: Path, on_progress=None) -> dict:


        import mmap

        self.local_cubes = {}
        cache_root = Path(cache_root).expanduser().resolve()
        cache_root.mkdir(parents=True, exist_ok=True)
        ids = self.train_ids + self.val_ids
        if len(ids) != len(set(ids)) or set(ids) != set(self.images):
            raise ValueError("Local I/O cache must contain exactly the development images")
        resolved: dict[str, Path] = {}
        cache_index: dict[str, dict] = {}
        total_bytes = created = reused = 0
        for image_id in ids:
            if not image_id.startswith("HSI_train_"):
                raise ValueError("Local I/O cache is development-only")
            image = self.images[image_id]
            source_path = self.root / image["path"]
            metadata_path = self.root / image["metadata_path"]
            if file_sha256(metadata_path) != image["metadata_sha256"]:
                raise RuntimeError("Source cube metadata hash changed; refusing local I/O cache")
            source_metadata = _read_json(metadata_path)
            source_stat = source_path.stat()
            if (source_metadata["size_bytes"] != source_stat.st_size or
                    source_metadata["mtime_ns"] != source_stat.st_mtime_ns or
                    source_metadata["shape"] != image["shape"]):
                raise RuntimeError("Source prepared cube changed; refusing local I/O cache")
            source_identity = {
                "image_id": image_id,
                "source_metadata_sha256": image["metadata_sha256"],
                "source_payload_sha256": source_metadata["cache_payload_sha256"],
                "source_size_bytes": source_stat.st_size,
                "source_mtime_ns": source_stat.st_mtime_ns,
                "source_shape_chw": image["shape"],
                "local_cache_format": "float32_HWC_npy_v1",
            }
            source_fingerprint = digest_json(source_identity)
            target = cache_root / f"{image_id}.hwc.npy"
            entry_path = cache_root / f"{image_id}.hwc.json"
            if target.resolve() == source_path.resolve():
                raise ValueError("Local I/O cache cannot overwrite its prepared source")
            if target.exists() or entry_path.exists():
                if not target.is_file() or not entry_path.is_file():
                    raise RuntimeError(f"Incomplete local I/O cache for {image_id}; explicitly rebuild in an empty cache directory")
                envelope = _read_json(entry_path)
                entry = envelope.get("metadata", {})
                if envelope.get("metadata_sha256") != digest_json(entry):
                    raise RuntimeError(f"Local I/O cache metadata hash mismatch: {image_id}")
                stat = target.stat()
                if (entry.get("source_fingerprint") != source_fingerprint or
                        entry.get("size_bytes") != stat.st_size or
                        entry.get("mtime_ns") != stat.st_mtime_ns or
                        entry.get("shape_hwc") != [image["shape"][1], image["shape"][2], image["shape"][0]]):
                    raise RuntimeError(f"Local I/O cache changed: {image_id}; explicitly rebuild in an empty cache directory")
                reused += 1
            else:


                source = np.load(source_path, allow_pickle=False)
                if (source.dtype != np.float32 or tuple(source.shape) != tuple(image["shape"]) or
                        not source.flags.c_contiguous):
                    raise ValueError("Source prepared cube must be C-contiguous float32 CHW")
                source_payload = hashlib.sha256(memoryview(source).cast("B")).hexdigest()
                if source_payload != source_metadata["cache_payload_sha256"]:
                    raise RuntimeError(f"Source prepared cube payload hash mismatch: {image_id}")
                after_read = source_path.stat()
                if (source_stat.st_size, source_stat.st_mtime_ns) != (after_read.st_size, after_read.st_mtime_ns):
                    raise RuntimeError("Source prepared cube changed during sequential cache read")
                channels, height, width = source.shape
                pending = target.with_name(target.name + ".pending")
                output = np.lib.format.open_memmap(pending, mode="w+", dtype=np.float32,
                                                  shape=(height, width, channels))
                payload_hash = hashlib.sha256()
                try:
                    for y in range(0, height, 64):
                        rows = np.ascontiguousarray(source[:, y:y + 64, :].transpose(1, 2, 0))
                        output[y:y + rows.shape[0]] = rows
                        payload_hash.update(memoryview(rows).cast("B"))
                        output.flush()
                        if hasattr(output._mmap, "madvise"):
                            output._mmap.madvise(mmap.MADV_DONTNEED)
                        del rows
                finally:
                    output.flush()
                    output._mmap.close()
                    del source
                os.replace(pending, target)
                stat = target.stat()
                entry = {
                    "schema_version": 1,
                    "source_fingerprint": source_fingerprint,
                    "source_identity": source_identity,
                    "shape_hwc": [height, width, channels],
                    "dtype": "float32",
                    "size_bytes": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                    "hwc_payload_sha256": payload_hash.hexdigest(),
                    "payload_hash_scope": "native-endian float32 HWC C-order payload excluding npy header",
                    "created_at_utc": utc_now(),
                }
                atomic_json(entry_path, {"metadata": entry, "metadata_sha256": digest_json(entry)})
                created += 1
            resolved[image_id] = target
            cache_index[image_id] = {"path": str(target), "index_path": str(entry_path),
                                     "metadata_sha256": digest_json(entry), "size_bytes": entry["size_bytes"],
                                     "hwc_payload_sha256": entry["hwc_payload_sha256"],
                                     "source_fingerprint": source_fingerprint}
            total_bytes += entry["size_bytes"]
            if on_progress is not None:
                on_progress(image_id, len(resolved), len(ids), total_bytes)
        self.local_cubes = resolved
        return {
            "status": "READY", "cache_root": str(cache_root), "layout": "float32_HWC",
            "images": cache_index, "created_images": created, "reused_images": reused,
            "cache_bytes": total_bytes, "resident_full_cube_cache_bytes": 0,
            "construction_memory_policy": "one source CHW cube plus <=64 HWC rows; output pages flushed and released on Linux",
            "reuse_validation": "source/index metadata SHA256 plus source/cache size and mtime; payloads hashed on creation",
            "seed_independent": True, "final_test_access": False, "gt_access": False,
            "source_prepared_artifacts_modified": False,
        }

    def get_patch(self, image_id: str, y: int, x: int, size: int = 64, *, use_local_cache: bool = True):
        import torch
        height, width = self.image_shape(image_id)
        if size <= 0 or not 0 <= y < height or not 0 <= x < width:
            raise ValueError("Patch origin/size outside development image")
        crop_h, crop_w = min(size, height - y), min(size, width - x)
        result = np.zeros((self.mean.shape[0], size, size), dtype=np.float32)
        local = use_local_cache and image_id in self.local_cubes
        cube_path = self.local_cubes[image_id] if local else self.root / self.images[image_id]["path"]
        cube = np.load(cube_path, mmap_mode="r", allow_pickle=False)
        try:
            chw = tuple(self.images[image_id]["shape"])
            expected_shape = (chw[1], chw[2], chw[0]) if local else chw
            if cube.shape != expected_shape or cube.dtype != np.float32:
                raise ValueError("Prepared cube changed shape or dtype")
            patch_view = cube[y:y + crop_h, x:x + crop_w, :].transpose(2, 0, 1) if local else cube[:, y:y + crop_h, x:x + crop_w]
            patch = np.array(patch_view, dtype=np.float32, copy=True)
            del patch_view
        finally:
            cube._mmap.close()
        patch -= self.mean
        patch /= self.std
        result[:, :crop_h, :crop_w] = patch
        return torch.from_numpy(result)

    def get_sparse_patch(self, image_id: str, y: int, x: int, size: int = 64):
        import torch
        height, width = self.image_shape(image_id)
        if size <= 0 or not 0 <= y < height or not 0 <= x < width:
            raise ValueError("Patch origin/size outside development image")
        crop_h, crop_w = min(size, height - y), min(size, width - x)
        result = np.full((size, size), -1, dtype=np.int64)
        result[:crop_h, :crop_w] = self.masks[image_id][y:y + crop_h, x:x + crop_w]
        return torch.from_numpy(result)


def load_prepared(root: str | Path, cfg: dict, seed: int, fraction: float | None = None) -> PreparedData:
    root = Path(root).resolve()
    contract, seeds = _contract(root, cfg)
    if seed not in seeds["main_seeds"] + seeds["pilot_seeds"]:
        raise ValueError("Seed is not in the frozen seed manifest")
    destination = _prepared_directory(root, cfg)
    if _read_json(destination / "contract.json") != contract:
        raise RuntimeError("Frozen data preparation contract differs")
    report = _read_json(destination / "prepared.json")
    if report["status"] != "READY" or report["contract_sha256"] != digest_json(contract):
        raise RuntimeError("Preparation is incomplete or belongs to another protocol")
    split = _read_json(root / cfg["split"]["validation_manifest"])
    expected_split = fixed_split([f"HSI_train_{index}" for index in range(1, 18)], seeds["validation_seed"], 3)
    if split != expected_split or report["split_sha256"] != split["split_sha256"]:
        raise RuntimeError("Fixed split changed")
    normalization_path = root / report["normalization_path"]
    if file_sha256(normalization_path) != report["normalization_sha256"]:
        raise RuntimeError("Normalization hash changed")
    for image in report["images"].values():
        metadata_path = root / image["metadata_path"]
        if file_sha256(metadata_path) != image["metadata_sha256"]:
            raise RuntimeError("Cube metadata hash changed")
        metadata = _read_json(metadata_path)
        stat = (root / image["path"]).stat()
        if metadata["size_bytes"] != stat.st_size or metadata["mtime_ns"] != stat.st_mtime_ns:
            raise RuntimeError("Prepared cube was modified after preparation")
    return PreparedData(root, report, split, _read_json(normalization_path), int(seed),
                        cfg["pilot"]["fraction"] if fraction is None else fraction)

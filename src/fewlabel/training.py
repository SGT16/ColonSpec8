from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .metrics import image_summary, segmentation_metrics
from .models import build_model
from .objectives import dice_from_statistics, ema_update, masked_supervised, prototype_loss
from .runtime import clean, derived_seed, now, sha, write_json


def dice_gradient_coefficients(intersection, prediction_sum, target_sum, epsilon=1e-6):

    denominator = prediction_sum + target_sum + epsilon
    numerator = 2 * intersection + epsilon
    classes = intersection.numel()
    return (-2 / (classes * denominator)).detach(), (numerator / (classes * denominator.square())).detach()


def dice_gradient_surrogate(stats, coefficients):

    intersection_coefficient, prediction_coefficient = coefficients
    return (stats["dice_intersection"] * intersection_coefficient).sum() + (
        stats["dice_prediction_sum"] * prediction_coefficient).sum()


def _tensor(value, dtype=None):
    if isinstance(value, torch.Tensor):
        result = value.detach().cpu().clone()
        return result.to(dtype=dtype) if dtype is not None else result
    result = torch.from_numpy(np.array(value, copy=True))
    return result.to(dtype=dtype) if dtype is not None else result


class PatchPlan(Dataset):


    def __init__(self, prepared, cfg, seed, epoch, n_samples):
        self.prepared, self.cfg, self.seed, self.epoch = prepared, cfg, int(seed), int(epoch)
        self.size = int(cfg["model"]["patch_size"])
        self.ids = list(prepared.train_ids)
        if not self.ids:
            raise ValueError("No TRAIN images")
        self.anchors = {image: np.argwhere(np.asarray(prepared.masks[image]) >= 0) for image in self.ids}
        self.plan = []
        rng = np.random.default_rng(derived_seed(seed, "sampling", epoch))
        for index in range(n_samples):
            image_index = int(rng.integers(len(self.ids)))
            image = self.ids[image_index]
            height, width = prepared.image_shape(image)
            anchored = index % 2 == 0
            if anchored:
                anchors = self.anchors[image]
                if not len(anchors):
                    raise ValueError(f"TRAIN image has no revealed anchor: {image}")
                center_y, center_x = anchors[int(rng.integers(len(anchors)))]
                y = int(np.clip(center_y - rng.integers(self.size), 0, max(0, height - self.size)))
                x = int(np.clip(center_x - rng.integers(self.size), 0, max(0, width - self.size)))
            else:
                y = int(rng.integers(max(1, height - self.size + 1)))
                x = int(rng.integers(max(1, width - self.size + 1)))
            self.plan.append((image_index, y, x, anchored))

    def __len__(self):
        return len(self.plan)

    def __getitem__(self, index):
        image_index, y, x, anchored = self.plan[index]
        image = self.ids[image_index]
        patch = _tensor(self.prepared.get_patch(image, y, x, self.size), torch.float32).numpy()
        labels = _tensor(self.prepared.get_sparse_patch(image, y, x, self.size), torch.int64).numpy()
        height, width = self.prepared.image_shape(image)
        yy, xx = np.mgrid[y:y + self.size, x:x + self.size]
        coordinate = yy * width + xx
        coordinate[(yy >= height) | (xx >= width)] = -1
        if patch.shape[1:] != (self.size, self.size) or labels.shape != (self.size, self.size):
            raise ValueError("Prepared patch/padding shape does not match protocol")
        rng = np.random.default_rng(derived_seed(self.seed, "augmentation", self.epoch, index))
        aug = self.cfg["augment"]
        rotations = int(rng.integers(4)) if aug.get("rotate90", True) else 0
        patch, labels, coordinate = (np.rot90(a, rotations, axes=(-2, -1)) for a in (patch, labels, coordinate))
        if aug.get("flip", True):
            for axis in (-2, -1):
                if rng.integers(2):
                    patch, labels, coordinate = (np.flip(a, axis=axis) for a in (patch, labels, coordinate))
        weak = np.array(patch, dtype=np.float32, copy=True)
        strong = weak.copy()
        noise = float(aug.get("noise_std_normalized", 0.01))
        if noise:
            weak += rng.normal(0, noise, weak.shape).astype(np.float32)
            strong += rng.normal(0, noise, strong.shape).astype(np.float32)
        limits = aug.get("illumination_scale", [0.95, 1.05])
        strong *= np.float32(rng.uniform(*limits))
        bands = strong.shape[0]
        n_drop = int(rng.integers(int(math.floor(bands * aug.get("band_dropout_max_fraction", 0.05))) + 1))
        if n_drop:
            strong[rng.choice(bands, n_drop, replace=False)] = 0
        n_mask = int(rng.integers(int(math.floor(bands * aug.get("spectral_mask_max_fraction", 0.05))) + 1))
        if n_mask:
            start = int(rng.integers(bands - n_mask + 1))
            strong[start:start + n_mask] = 0
        if not np.isfinite(weak).all() or not np.isfinite(strong).all():
            raise FloatingPointError("Nonfinite normalized/augmented TRAIN patch")
        return {"weak": torch.from_numpy(weak), "strong": torch.from_numpy(strong),
                "labels": torch.from_numpy(np.array(labels, copy=True)),
                "coordinates": torch.from_numpy(np.array(coordinate, dtype=np.int64, copy=True)),
                "image_index": image_index, "anchored": anchored}


class PrototypeBank:


    def __init__(self, image_ids, channels, device, momentum):
        self.ids = list(image_ids)
        self.prototypes = torch.zeros(2, channels, device=device)
        self.valid = torch.zeros(2, dtype=torch.bool, device=device)
        self.momentum = float(momentum)
        self.begin_epoch()

    def begin_epoch(self):
        self.points = {image: {} for image in self.ids}

    @torch.no_grad()
    def observe(self, teacher_features, labels, coordinates, image_indices):
        normalized = F.normalize(teacher_features.detach().float(), dim=1)
        for batch_index, image_index in enumerate(image_indices.tolist()):
            selected = (labels[batch_index] >= 0) & (coordinates[batch_index].to(labels.device) >= 0)
            ids = coordinates[batch_index][selected.cpu()].tolist()
            truth = labels[batch_index][selected].cpu().tolist()
            vectors = normalized[batch_index].permute(1, 2, 0)[selected].cpu().numpy()
            bank = self.points[self.ids[image_index]]
            for point, label, vector in zip(ids, truth, vectors):
                if point not in bank:
                    bank[point] = (int(label), vector.copy())

    @torch.no_grad()
    def finish_epoch(self):
        counts, contributors, drift = [], [], []
        for cls in (0, 1):
            per_image, n_points = [], 0
            for bank in self.points.values():
                vectors = [vector for label, vector in bank.values() if label == cls]
                n_points += len(vectors)
                if vectors:
                    per_image.append(np.mean(vectors, axis=0))
            counts.append(n_points)
            contributors.append(len(per_image))
            if per_image:
                estimate = torch.as_tensor(np.mean(per_image, axis=0), device=self.prototypes.device)
                estimate = F.normalize(estimate, dim=0)
                old = self.prototypes[cls].clone()
                updated = self.momentum * old + (1 - self.momentum) * estimate if self.valid[cls] else estimate
                self.prototypes[cls] = F.normalize(updated, dim=0)
                drift.append(float((self.prototypes[cls] - old).norm().item()))
                self.valid[cls] = True
            else:
                drift.append(None)
        return {"unique_points_per_class": counts, "contributing_images_per_class": contributors,
                "valid": self.valid.cpu().tolist(), "drift_l2": drift,
                "source": "unique_revealed_train_teacher_features_equal_image_means",
                "status": "ACTIVE" if bool(self.valid.all()) else "MISSING_REVEALED_CLASS"}


def _tiles(length, size, stride):
    positions = list(range(0, max(1, length - size + 1), stride))
    last = max(0, length - size)
    if positions[-1] != last:
        positions.append(last)
    return positions


@torch.no_grad()
def validate(student, prepared, cfg, physical_batch, device, progress=None):

    student.eval()
    size = int(cfg["model"]["patch_size"])
    stride = int(cfg["evaluation"]["tile_stride"])
    image_metrics, sparse_predictions = {}, {}
    loss_sum, count = 0.0, 0
    for image in prepared.val_ids:
        height, width = prepared.image_shape(image)
        summed = np.zeros((height, width, 2), dtype=np.float32)
        coverage = np.zeros((height, width), dtype=np.uint16)
        locations = [(y, x) for y in _tiles(height, size, stride) for x in _tiles(width, size, stride)]
        for offset in range(0, len(locations), physical_batch):
            batch_locations = locations[offset:offset + physical_batch]
            patches = torch.stack([_tensor(prepared.get_patch(image, y, x, size), torch.float32) for y, x in batch_locations])
            probabilities = student(patches.to(device))["logits"].float().softmax(1).permute(0, 2, 3, 1).cpu().numpy()
            if not np.isfinite(probabilities).all():
                raise FloatingPointError("Nonfinite sparse-validation probabilities")
            for (y, x), patch in zip(batch_locations, probabilities):
                dh, dw = min(size, height - y), min(size, width - x)
                summed[y:y + dh, x:x + dw] += patch[:dh, :dw]
                coverage[y:y + dh, x:x + dw] += 1
            if progress is not None:
                progress()
        if not np.all(coverage > 0):
            raise RuntimeError("Validation tiling left uncovered image pixels")
        ys, xs = np.nonzero(np.asarray(prepared.masks[image]) >= 0)
        truth = np.asarray(prepared.masks[image])[ys, xs].astype(np.int64)
        probabilities = summed[ys, xs] / coverage[ys, xs, None]

        probabilities /= probabilities.sum(axis=1, keepdims=True)
        image_metrics[image] = segmentation_metrics(truth, probabilities)
        if len(truth):
            loss_sum -= float(np.log(np.clip(probabilities[np.arange(len(truth)), truth], 1e-12, 1)).sum())
        count += len(truth)
        sparse_predictions[image] = {"y": ys, "x": xs, "y_true": truth, "probabilities": probabilities}
    summary = image_summary(image_metrics)
    if summary["status"] != "COMPLETE":
        raise RuntimeError(f"Validation primary metric is not fully defined: {summary}")
    pooled = segmentation_metrics(np.concatenate([v["y_true"] for v in sparse_predictions.values()]),
                                  np.concatenate([v["probabilities"] for v in sparse_predictions.values()]))
    student.train()
    return {"summary": summary, "image_metrics": image_metrics, "pooled_metrics": pooled,
            "sparse_ce": loss_sum / count if count else None,
            "n_revealed": count}, sparse_predictions


def _atomic_weights(path, value):
    temporary = Path(str(path) + f".{os.getpid()}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _finite_gradients(model):
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    if not gradients or not bool(torch.stack([torch.isfinite(g).all() for g in gradients]).all()):
        raise FloatingPointError("Absent or nonfinite student gradients")
    return float(torch.stack([g.detach().float().square().sum() for g in gradients]).sum().sqrt().item())


def train_job(root, cfg, model_name, seed, prepared, out_dir, status, max_epochs=None,
              profile=False, progress_callback=None):

    root, out_dir = Path(root), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "result.json").exists():
        raise RuntimeError("A finished job must be verified/skipped by the queue, not overwritten")
    if not torch.cuda.is_available():
        raise RuntimeError("Real pilot training requires CUDA")
    if set(prepared.train_ids) & set(prepared.val_ids):
        raise RuntimeError("TRAIN and VAL image identities overlap")
    start = time.monotonic()
    device = torch.device("cuda")
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    np.random.seed(int(seed))
    torch.backends.cudnn.benchmark = False

    student = build_model(model_name, cfg).to(device)
    teacher = copy.deepcopy(student).eval().requires_grad_(False)
    initial_weight_hash = None
    import hashlib
    digest = hashlib.sha256()
    for name, value in student.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    initial_weight_hash = digest.hexdigest()
    optimizer = torch.optim.AdamW(student.parameters(), lr=float(cfg["train"]["lr"]),
                                  weight_decay=float(cfg["train"]["weight_decay"]))
    effective = int(cfg["train"]["effective_batch_size"])
    physical = cfg["train"].get("physical_batch_size", 2)
    physical = int(physical) if isinstance(physical, (int, float)) else 2
    if physical <= 0 or effective % physical:
        raise ValueError("Physical batch must divide the effective batch exactly")
    steps = int(cfg["pilot"].get("steps_per_epoch", 32))
    epochs = int(max_epochs if max_epochs is not None else cfg["pilot"]["max_epochs"])
    if profile:
        epochs = 2
    warmup, ramp = int(cfg["ssl"]["warmup_epochs"]), int(cfg["ssl"]["ramp_epochs"])
    bank = PrototypeBank(prepared.train_ids, int(cfg["model"]["d_model"]), device,
                         cfg["ssl"]["prototype_momentum"])
    history, best_epoch, best_score, best_ce, stale = [], 0, -math.inf, math.inf, 0
    torch.cuda.reset_peak_memory_stats()
    write_json(out_dir / "effective_config.json", {"config": cfg, "model": model_name, "seed": int(seed),
               "profile": bool(profile), "effective_batch": effective, "physical_batch": physical,
               "steps_per_epoch": steps, "precision": "float32", "initial_weights_sha256": initial_weight_hash,
               "restart_policy": "epoch_1_same_seed_masks_split", "test_status": "LOCKED"})

    def progress():
        if progress_callback is not None:
            progress_callback(time.monotonic() - start)

    for epoch in range(1, epochs + 1):
        epoch_start = time.monotonic()
        bank.begin_epoch()
        student.train()
        plan = PatchPlan(prepared, cfg, seed, epoch, effective * steps)
        workers = int(cfg["train"].get("num_workers", 4))
        loader_options = {"batch_size": physical, "shuffle": False, "num_workers": workers,
                          "pin_memory": True, "drop_last": False,
                          "generator": torch.Generator().manual_seed(derived_seed(seed, "loader", epoch))}
        if workers:
            loader_options.update(prefetch_factor=int(cfg["train"].get("prefetch_factor", 4)), persistent_workers=False)
        loader = iter(DataLoader(plan, **loader_options))
        ramp_weight = 0.0 if epoch <= warmup else min(1.0, (epoch - warmup) / max(1, ramp))
        lambda_u = ramp_weight * float(cfg["ssl"]["lambda_u_max"])
        lambda_p = ramp_weight * float(cfg["ssl"]["lambda_proto_max"]) if model_name == "fewlabel_mamba" else 0.0
        sums = {key: 0.0 for key in ("ce", "dice", "consistency", "prototype", "total", "gradient_norm")}
        pseudo_selected, unlabeled_total, supervised_count = 0, 0, 0
        pseudo_classes = np.zeros(2, dtype=np.int64)
        prototype_gradient_norm, prototype_measurement_done = None, False
        for step in range(1, steps + 1):
            cached, totals = [], {key: torch.zeros(2, device=device) for key in (
                "dice_intersection", "dice_prediction_sum", "dice_target_sum")}
            total_labels = total_pseudo = 0
            ce_value = consistency_value = prototype_value = 0.0
            with torch.no_grad():
                for _ in range(effective // physical):
                    batch = next(loader)
                    labels = batch["labels"].to(device, non_blocking=True)
                    weak = batch["weak"].to(device, non_blocking=True)
                    strong = batch["strong"].to(device, non_blocking=True)
                    teacher_output, output = teacher(weak), student(strong)
                    if not bool(torch.isfinite(output["logits"]).all() & torch.isfinite(teacher_output["logits"]).all()):
                        raise FloatingPointError("Nonfinite teacher/student logits")
                    stats = masked_supervised(output["logits"], labels)
                    confidence, pseudo = teacher_output["logits"].float().softmax(1).max(1)
                    accepted = labels.eq(-1) & confidence.ge(float(cfg["ssl"]["confidence_threshold"]))
                    n_labels, n_pseudo = int(stats["labeled_count"].item()), int(accepted.sum().item())
                    total_labels += n_labels
                    total_pseudo += n_pseudo
                    unlabeled_total += int(labels.eq(-1).sum().item())
                    if n_pseudo:
                        pseudo_classes += torch.bincount(pseudo[accepted], minlength=2).cpu().numpy()
                    ce_value += float(stats["ce_sum"].item())
                    consistency_value += float((F.cross_entropy(output["logits"].float(), pseudo, reduction="none") * accepted).sum().item())
                    if bool(bank.valid.all()):
                        prototype_value += float(prototype_loss(output["features"], bank.prototypes, bank.valid,
                            accepted, pseudo, float(cfg["ssl"]["prototype_temperature"])).item()) * n_pseudo
                    for key in totals:
                        totals[key] += stats[key]
                    bank.observe(teacher_output["features"], labels, batch["coordinates"], batch["image_index"])
                    cached.append((batch["strong"], batch["labels"], accepted.cpu(), pseudo.cpu(), n_pseudo))
                    del weak, strong, teacher_output, output, stats
            coefficients = dice_gradient_coefficients(totals["dice_intersection"], totals["dice_prediction_sum"], totals["dice_target_sum"])
            optimizer.zero_grad(set_to_none=True)
            for strong_cpu, labels_cpu, accepted_cpu, pseudo_cpu, n_pseudo in cached:
                output = student(strong_cpu.to(device, non_blocking=True))
                labels = labels_cpu.to(device, non_blocking=True)
                accepted, pseudo = accepted_cpu.to(device), pseudo_cpu.to(device)
                stats = masked_supervised(output["logits"], labels)
                supervised = stats["ce_sum"] / max(1, total_labels) + dice_gradient_surrogate(stats, coefficients)
                consistency = (F.cross_entropy(output["logits"].float(), pseudo, reduction="none") * accepted).sum() / max(1, total_pseudo)
                proto = prototype_loss(output["features"], bank.prototypes, bank.valid, accepted, pseudo,
                                       float(cfg["ssl"]["prototype_temperature"])) * n_pseudo / max(1, total_pseudo)
                if lambda_p > 0 and n_pseudo and bool(bank.valid.all()) and not prototype_measurement_done:
                    gradient = torch.autograd.grad(lambda_p * proto, student.stem[0].weight, retain_graph=True)[0]
                    prototype_gradient_norm = float(gradient.float().norm().item())
                    prototype_measurement_done = True
                loss = supervised + lambda_u * consistency + lambda_p * proto
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Nonfinite training objective")
                loss.backward()
                del output, stats, loss
            gradient_norm = _finite_gradients(student)
            optimizer.step()
            ema_update(teacher, student, float(cfg["ssl"]["ema_teacher"]))
            ce = ce_value / max(1, total_labels)
            dice = float(dice_from_statistics(totals["dice_intersection"], totals["dice_prediction_sum"], totals["dice_target_sum"]).item())
            consistency = consistency_value / max(1, total_pseudo)
            proto = prototype_value / max(1, total_pseudo)
            values = {"ce": ce, "dice": dice, "consistency": consistency, "prototype": proto,
                      "total": ce + dice + lambda_u * consistency + lambda_p * proto, "gradient_norm": gradient_norm}
            for key, value in values.items():
                sums[key] += value
            supervised_count += total_labels
            pseudo_selected += total_pseudo
            if status is not None:
                status.update(status="RUNNING", stage="profile" if profile else "pilot", model=model_name,
                              seed=int(seed), epoch=epoch, max_epochs=epochs, step=step, steps_per_epoch=steps,
                              last_metrics={"loss": values["total"], **values}, peak_allocated_bytes=torch.cuda.max_memory_allocated())
            progress()
            del cached
        del loader
        torch.cuda.synchronize()
        train_seconds = time.monotonic() - epoch_start
        prototype_status = bank.finish_epoch()
        validation_start = time.monotonic()
        last_validation_status = [0.0]
        def validation_progress():
            if status is not None and time.monotonic()-last_validation_status[0] >= 2:
                status.update(stage="profile_validation" if profile else "pilot_validation",
                              validation_elapsed_s=time.monotonic()-validation_start)
                last_validation_status[0] = time.monotonic()
            progress()
        validation, predictions = validate(student, prepared, cfg, physical, device, validation_progress)
        validation_seconds = time.monotonic() - validation_start
        score, val_ce = validation["summary"]["image_mean_macro_f1"], validation["sparse_ce"]
        improved = score > best_score or (score == best_score and val_ce < best_ce)
        if improved:
            best_score, best_ce, best_epoch, stale = score, val_ce, epoch, 0
            _atomic_weights(out_dir / "best.pt", {"state_dict": student.state_dict(), "model": model_name,
                            "seed": int(seed), "epoch": epoch, "prediction_model": "student"})
            for image, arrays in predictions.items():
                path = out_dir / f"best_sparsevalpred_{image}.npz"
                temporary = path.with_name(path.name + ".tmp")
                with temporary.open("wb") as handle:
                    np.savez_compressed(handle, **arrays)
                os.replace(temporary, path)
            write_json(out_dir / "best_validation.json", validation)
        else:
            stale += 1
        record = {"timestamp": now(), "epoch": epoch, "optimizer_steps": steps,
                  "sampled_patches": steps * effective, "losses": {key: value / steps for key, value in sums.items()},
                  "lambda_u": lambda_u, "lambda_p": lambda_p,
                  "pseudo_selected": pseudo_selected, "unlabeled_presentations": unlabeled_total,
                  "pseudo_acceptance": pseudo_selected / unlabeled_total if unlabeled_total else None,
                  "pseudo_class_counts": pseudo_classes.tolist(), "revealed_label_presentations": supervised_count,
                  "prototype": prototype_status, "prototype_encoder_gradient_norm": prototype_gradient_norm,
                  "prototype_gradient_measurement": "MEASURED" if prototype_measurement_done else "NO_ACTIVE_ACCEPTED_PROTOTYPE_TERM",
                  "gradient_finite": True, "teacher_has_gradient": any(p.grad is not None for p in teacher.parameters()),
                  "validation": validation, "val_image_macro_f1": score,
                  "train_s": train_seconds, "validation_s": validation_seconds,
                  "epoch_s": time.monotonic() - epoch_start, "elapsed_s": time.monotonic() - start,
                  "best_epoch": best_epoch, "best_val_image_macro_f1": best_score, "early_stopping_bad_epochs": stale,
                  "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                  "peak_reserved_bytes": int(torch.cuda.max_memory_reserved())}
        history.append(record)
        with (out_dir / "epochs.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(clean(record), ensure_ascii=False, allow_nan=False) + "\n")
        if status is not None:
            status.update(history=[{"epoch": r["epoch"], "loss": r["losses"]["total"], "val_f1": r["val_image_macro_f1"]} for r in history],
                          last_metrics={"loss": record["losses"]["total"], "val_f1": score,
                                        "pseudo_acceptance": record["pseudo_acceptance"], "epoch_s": record["epoch_s"]},
                          epoch=epoch, best=best_score, early_stopping_bad_epochs=stale)
            status.event("epoch_complete", model=model_name, seed=int(seed), epoch=epoch,
                         val_image_macro_f1=score, elapsed_s=record["elapsed_s"])
        print(f"{epoch:03d}/{epochs:03d} | {model_name} | seed={seed} | sparse_val_f1={score:.4f} | "
              f"loss={sums['total']/steps:.4f} | epoch={record['epoch_s']:.1f}s | esb={stale} | best={best_score:.4f}", flush=True)
        progress()
        if not profile and stale >= int(cfg["pilot"]["patience"]):
            break
    result = {"status": "COMPLETED", "scope": "SENTETIK_TEST" if cfg.get("runtime", {}).get("synthetic_fixture") else "REAL_DATA_PROFILE" if profile else "REAL_PILOT",
              "profile": bool(profile), "model": model_name, "seed": int(seed), "best_epoch": best_epoch,
              "best_val_image_macro_f1": best_score, "best_val_sparse_ce": best_ce,
              "epochs_completed": len(history), "epoch_history": history, "elapsed_s": time.monotonic() - start,
              "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
              "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
              "initial_weights_sha256": initial_weight_hash, "checkpoint_sha256": sha(out_dir / "best.pt"),
              "prediction_model": "student", "test_status": "LOCKED", "error_counts": {"nonfinite": 0},
              "effective_batch": effective, "physical_batch": physical, "steps_per_epoch": steps,
              "precision": "float32", "checkpoint_selection": "sparse_val_image_macro_f1_then_lower_sparse_ce_then_earlier_epoch"}
    write_json(out_dir / "result.json", result)
    del student, teacher, optimizer, bank
    torch.cuda.empty_cache()
    return result

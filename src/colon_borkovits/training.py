from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import time
import uuid

import numpy as np
import torch
from torch.nn import functional as F

from colon_nnunet_recovery.runtime import replace_with_backoff, write_json
from fewlabel.runtime import now, sha
from .models import ARMS, ARCHITECTURE, PROJECTION, build_model, state_digest


def tensor_digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def configure_strict(seed):
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG=:4096:8 is required before CUDA initialization")
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    random.seed(int(seed))
    if not torch.cuda.is_available():
        raise RuntimeError("Production Borkovits training/inference requires CUDA; no fallback")
    torch.cuda.manual_seed_all(int(seed))


def atomic_torch(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
    with temporary.open("xb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    replace_with_backoff(temporary, path)


def image_cross_entropy(logits, labels, *, reduction="mean"):

    if logits.ndim != 2 or logits.shape[1] != 2:
        raise ValueError("Image objective requires logits [images,2]")
    if labels.ndim != 1 or labels.shape[0] != logits.shape[0] or labels.dtype != torch.long:
        raise ValueError("Image objective requires exactly one int64 label per image")
    if logits.shape[0] < 1 or not bool(((labels >= 0) & (labels < 2)).all()):
        raise ValueError("Image labels must be 0,1")
    if reduction not in ("mean", "sum", "none"):
        raise ValueError("Invalid image loss reduction")

    losses = -F.log_softmax(logits, dim=1).gather(1, labels[:, None]).squeeze(1)
    return losses if reduction == "none" else (losses.mean() if reduction == "mean" else losses.sum())


def effective_batch_backward(model, batches, device, *, total_images=None):

    if total_images is None:
        if not isinstance(batches, (list, tuple)) or not batches:
            raise ValueError("Streaming batches require an explicit total_images")
        count = sum(int(labels.shape[0]) for _, labels in batches)
    else:
        count = int(total_images)
    if count < 1:
        raise ValueError("Effective batch has no image labels")
    total_loss, observed = 0.0, 0
    for bags, labels in batches:
        if bags.ndim != 5 or bags.shape[0] != labels.shape[0]:
            raise ValueError("Inputs must be image bags [B,K,C,H,W]")
        logits = model(bags.to(device, non_blocking=True))
        loss_sum = image_cross_entropy(logits, labels.to(device, non_blocking=True), reduction="sum")
        if not bool(torch.isfinite(loss_sum)):
            raise FloatingPointError("Nonfinite image-level cross entropy")
        (loss_sum / count).backward()
        total_loss += float(loss_sum.detach())
        observed += int(labels.shape[0])
        if observed > count:
            raise ValueError("Streaming batch exceeded fixed effective image count")
    if observed != count:
        raise ValueError("Streaming effective batch image count is incomplete")
    return {"loss_ce": total_loss / count, "image_presentations": count}


def validate_protocol(cfg, job):
    required = {"epochs": 100, "steps_per_epoch": 16, "physical_batch": 2,
                "bag_patches": 4, "gradient_accumulation": 8,
                "effective_batch": 16, "patch_size": 64}
    if any(cfg.get(key) != value for key, value in required.items()):
        raise ValueError("Borkovits fixed full-budget training protocol changed")
    if float(cfg.get("lr", -1)) != 0.001 or float(cfg.get("weight_decay", -1)) != 0.0001:
        raise ValueError("Fixed AdamW learning rate or weight decay changed")
    if job.get("arm") not in ARMS or not isinstance(job.get("job_id"), str):
        raise ValueError("Invalid Borkovits job identity")
    if job.get("init_seed") not in (938, 476, 993):
        raise ValueError("Initialization seed is not in the fixed study")


def _cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_cpu_tree(item) for item in value)
    return value


def _rng_state():
    numpy_state = np.random.get_state()
    return {"torch_cpu": torch.get_rng_state(), "torch_cuda": torch.cuda.get_rng_state_all(),
            "python": random.getstate(),
            "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]]}


def _notify(callback, **values):
    if callback is not None:
        callback(values)


def _check_gradients(model):
    parameters = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    missing = [name for name, p in parameters if p.grad is None]
    if missing:
        raise FloatingPointError("Missing gradients: " + ", ".join(missing))

    if not bool(torch.stack([torch.isfinite(p.grad).all() for _, p in parameters]).all()):
        raise FloatingPointError("Nonfinite trainable gradient")


def train_job(cfg, job, data, out_dir, status_callback=None):

    validate_protocol(cfg, job)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    forbidden = ("fit_started.json", "result.json", "epochs.jsonl", "last.pt", "final.pt")
    if any((out_dir / name).exists() for name in forbidden):
        raise RuntimeError("Refuse automatic fit retry or overwriting an existing attempt")
    if len(data.train_ids) != len(set(data.train_ids)) or not data.train_ids:
        raise RuntimeError("Invalid training image identity set")
    if set(data.train_ids) & set(data.heldout_ids):
        raise RuntimeError("Training and held-out image identity sets overlap")
    if not data.train_patient_ids or set(data.train_patient_ids) & set(data.heldout_patient_ids):
        raise RuntimeError("Training and held-out patient identity sets overlap")

    with (out_dir / "fit_started.json").open("x", encoding="utf-8") as handle:
        json.dump({"job": job, "started_at": now(), "pid": os.getpid(),
                   "policy": "NO_AUTOMATIC_RETRY_OR_RESUME"}, handle, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
    configure_strict(job["init_seed"])
    device = torch.device("cuda")
    model = build_model(job["arm"], job["init_seed"], data.pca).to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                 lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
    projection_reference = model.local_projection[0].weight.detach().clone()
    initial_projection = tensor_digest(projection_reference)
    metadata = {"job": dict(job), "config": cfg, "architecture": ARCHITECTURE,
                "train_ids": list(data.train_ids), "heldout_ids": list(data.heldout_ids),
                "train_patient_ids": list(data.train_patient_ids),
                "heldout_patient_ids": list(data.heldout_patient_ids),
                "frozen_sampling_plan_sha256": data.input_plan_identity,
                "initial_weights_sha256": model.initial_sha256,
                "initial_backbone_sha256": model.shared_initial_sha256,
                "parent_initial_sha256": model.parent_initial_sha256,
                "initial_projection_sha256": initial_projection,
                "loss": "mean_image_bag_cross_entropy", "label_unit": "tissue_core",
                "sampling": "uniform_class_then_patient_with_class_then_core",
                "supervision": "all_training_core_class_labels_no_pixel_annotations",
                "primary_endpoint": "fixed_epoch100_step1600", "selection_data": None,
                "precision": "float32", "heldout_access_during_fit": False,
                "source_dataset_test_access": False, "retries": 0,
                "numerics": "STRICT_DETERMINISTIC_CE_GATHER_BILINEAR_MATRIX"}
    write_json(out_dir / "effective_config.json", metadata)
    started = time.monotonic()
    history = []
    total_steps = 0
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, 101):
        epoch_started = time.monotonic()
        model.train()
        batches = iter(data.training_batches(epoch))
        losses, input_digest = [], hashlib.sha256()
        def micro_batches():
            for _ in range(8):
                try:
                    batch = next(batches)
                except StopIteration as exc:
                    raise RuntimeError("Shared epoch plan ended before 128 physical batches") from exc
                inputs = torch.as_tensor(batch["inputs"], dtype=torch.float32)
                raw_labels = torch.as_tensor(batch["labels"])
                if raw_labels.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                    raise RuntimeError("Training cube labels must be integers, never coerced floats")
                labels = raw_labels.to(dtype=torch.long)
                image_ids = list(batch["image_ids"])
                if tuple(inputs.shape) != (2, 4, 814, 64, 64) or tuple(labels.shape) != (2,):
                    raise RuntimeError("Shared batch does not match physical2 x four patches")
                if len(image_ids) != 2 or any(image not in data.train_ids for image in image_ids):
                    raise RuntimeError("Non-TRAIN image entered a training bag")
                if any(data.patient(image) not in data.train_patient_ids for image in image_ids):
                    raise RuntimeError("Non-TRAIN patient entered a training bag")
                if any(int(label) != int(data.label(image)) for image, label in zip(image_ids, labels)):
                    raise RuntimeError("Training bag label differs from its cube class")
                if not bool(torch.isfinite(inputs).all()):
                    raise FloatingPointError("Nonfinite normalized training input")
                input_digest.update(memoryview(inputs.contiguous().numpy()).cast("B"))
                input_digest.update(labels.contiguous().numpy().tobytes())
                input_digest.update(json.dumps(image_ids, ensure_ascii=False).encode("utf-8"))
                yield inputs, labels
        for step in range(1, 17):
            optimizer.zero_grad(set_to_none=True)
            values = effective_batch_backward(model, micro_batches(), device, total_images=16)
            if values["image_presentations"] != 16:
                raise RuntimeError("Effective batch contains a different image count")
            _check_gradients(model)
            if job["arm"] != "learned":
                projection = model.local_projection[0].weight
                if projection.grad is not None or not torch.equal(projection, projection_reference):
                    raise RuntimeError("Fixed projection changed or received a gradient")
            optimizer.step()
            if not bool(torch.stack([torch.isfinite(parameter).all() for parameter in model.parameters()]).all()):
                raise FloatingPointError("Nonfinite model after optimizer update")
            total_steps += 1
            losses.append(values["loss_ce"])
            _notify(status_callback, phase="training", job_id=job["job_id"], arm=job["arm"],
                    fold=job["fold"], init_seed=job["init_seed"], epoch=epoch, step=step,
                    optimizer_steps=total_steps, loss_ce=values["loss_ce"],
                    elapsed_s=time.monotonic() - started)
        try:
            next(batches)
        except StopIteration:
            pass
        else:
            raise RuntimeError("Shared epoch plan exceeded 128 physical batches")
        epoch_row = {"epoch": epoch, "optimizer_steps": total_steps,
                     "mean_train_loss_ce": float(np.mean(losses)),
                     "image_presentations": 256, "input_plan_sha256": input_digest.hexdigest(),
                     "train_compute_s": time.monotonic() - epoch_started}
        payload = {"job": dict(job), "architecture": ARCHITECTURE, "epoch": epoch,
                   "optimizer_steps": total_steps, "state_dict": _cpu_tree(model.state_dict()),
                   "optimizer_state_dict": _cpu_tree(optimizer.state_dict()),
                   "rng_state": _rng_state(), "metadata": metadata,
                   "heldout_access_during_fit": False}
        atomic_torch(out_dir / "last.pt", payload)
        epoch_row["elapsed_s"] = time.monotonic() - epoch_started
        with (out_dir / "epochs.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(epoch_row, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        history.append(epoch_row)
    if total_steps != 1600 or len(history) != 100:
        raise RuntimeError("Fixed endpoint budget was not completed")
    if job["arm"] != "learned" and not torch.equal(model.local_projection[0].weight, projection_reference):
        raise RuntimeError("Fixed projection changed at final endpoint")
    atomic_torch(out_dir / "final.pt", payload)
    result = {"status": "COMPLETED", "job": dict(job), **dict(job),
              "epochs_completed": 100, "optimizer_steps": 1600,
              "primary_endpoint": "fixed_epoch100_step1600", "label_unit": "tissue_core",
              "checkpoint_path": str(out_dir / "final.pt"), "checkpoint_sha256": sha(out_dir / "final.pt"),
              "initial_weights_sha256": model.initial_sha256,
              "initial_backbone_sha256": model.shared_initial_sha256,
              "initial_projection_sha256": initial_projection,
              "final_projection_sha256": tensor_digest(model.local_projection[0].weight),
              "final_weights_sha256": state_digest(model.state_dict()),
              "train_final_loss_ce": history[-1]["mean_train_loss_ce"],
              "image_presentations": 25600, "patch_presentations": 102400,
              "epoch_history": history, "mean_epoch_s": float(np.mean([h["elapsed_s"] for h in history])),
              "elapsed_s": time.monotonic() - started,
              "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
              "heldout_access_during_fit": False, "source_dataset_test_access": False,
              "completed_at": now()}
    write_json(out_dir / "result.json", result)
    return result


def _binary_curves(truth, scores):

    truth, scores = np.asarray(truth, dtype=np.int64), np.asarray(scores, dtype=np.float64)
    positives, negatives = int(truth.sum()), int((1 - truth).sum())
    if positives == 0 or negatives == 0:
        return None, None
    order = np.argsort(-scores, kind="stable")
    ordered, sorted_scores = truth[order], scores[order]
    ends = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(sorted_scores) - 1]
    tp, fp = np.cumsum(ordered)[ends], np.cumsum(1 - ordered)[ends]
    tpr, fpr = np.r_[0.0, tp / positives], np.r_[0.0, fp / negatives]
    auc = float(np.sum((tpr[1:] + tpr[:-1]) * np.diff(fpr) / 2))
    precision, recall = tp / (tp + fp), tp / positives
    average_precision = float(np.sum(np.diff(np.r_[0.0, recall]) * precision))
    return auc, average_precision


def classification_metrics(labels, probabilities):

    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if labels.ndim != 1 or len(labels) < 1 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Metrics require one integer class per image")
    if probabilities.shape != (len(labels), 2) or not np.isfinite(probabilities).all():
        raise ValueError("Metrics require finite probabilities[images,2]")
    if (probabilities < 0).any() or (probabilities > 1).any() or not np.allclose(probabilities.sum(1), 1, atol=1e-6):
        raise ValueError("Probabilities must be normalized")
    if ((labels < 0) | (labels > 1)).any():
        raise ValueError("Labels outside two classes")
    predictions = probabilities.argmax(axis=1)
    confusion = np.zeros((2, 2), dtype=np.int64)
    np.add.at(confusion, (labels, predictions), 1)
    support, predicted = confusion.sum(1), confusion.sum(0)
    tp = np.diag(confusion)
    precision = np.divide(tp, predicted, out=np.zeros(2, dtype=float), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros(2, dtype=float), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(2, dtype=float), where=(precision + recall) > 0)
    tn = len(labels) - support - predicted + tp
    specificity = np.divide(tn, len(labels) - support, out=np.zeros(2, dtype=float), where=(len(labels) - support) > 0)
    accuracy = float(tp.sum() / len(labels))
    expected = float(np.dot(support, predicted) / len(labels) ** 2)
    denominator = float(np.sqrt((len(labels) ** 2 - np.dot(predicted, predicted)) *
                               (len(labels) ** 2 - np.dot(support, support))))
    confidence = probabilities.max(1)
    correct = predictions == labels
    ece = 0.0
    for index in range(10):
        chosen = (confidence >= index / 10) & ((confidence < (index + 1) / 10) if index < 9 else (confidence <= 1))
        if chosen.any():
            ece += float(chosen.mean() * abs(correct[chosen].mean() - confidence[chosen].mean()))
    curves = [_binary_curves(labels == index, probabilities[:, index]) for index in range(2)]
    return {"unit": "tissue_core", "n_images": int(len(labels)), "class_order": [0, 1],
            "confusion_matrix": confusion.tolist(), "accuracy": accuracy,
            "balanced_accuracy": float(recall.mean()), "macro_f1": float(f1.mean()),
            "sensitivity_crc": float(recall[1]), "specificity_nc": float(recall[0]),
            "auroc_crc": curves[1][0], "average_precision_crc": curves[1][1],
            "weighted_f1": float(np.dot(f1, support) / len(labels)),
            "mcc": float((tp.sum() * len(labels) - np.dot(support, predicted)) / denominator) if denominator else 0.0,
            "cohen_kappa": float((accuracy - expected) / (1 - expected)) if expected < 1 else 0.0,
            "negative_log_likelihood": float(-np.log(np.clip(probabilities[np.arange(len(labels)), labels], 1e-15, 1)).mean()),
            "brier_multiclass_sum": float(np.square(probabilities - np.eye(2)[labels]).sum(1).mean()),
            "ece10": ece, "zero_division_policy": 0,
            "macro_ovr_auroc": float(np.mean([x[0] for x in curves])) if all(x[0] is not None for x in curves) else None,
            "macro_ovr_average_precision": float(np.mean([x[1] for x in curves])) if all(x[1] is not None for x in curves) else None,
            "per_class": [{"class_id": i, "support": int(support[i]), "precision": float(precision[i]),
                           "recall": float(recall[i]), "f1": float(f1[i]), "specificity": float(specificity[i]),
                           "ovr_auroc": curves[i][0], "ovr_average_precision": curves[i][1]} for i in range(2)]}


def predict_job(cfg, job, data, checkpoint_path, out_dir, status_callback=None):

    validate_protocol(cfg, job)
    checkpoint_path, out_dir = Path(checkpoint_path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if any((out_dir / name).exists() for name in ("predictions.json", "inference_started.json")):
        raise RuntimeError("Refuse repeated held-out evaluation or overwrite")
    with (out_dir / "inference_started.json").open("x", encoding="utf-8") as handle:
        json.dump({"job": job, "checkpoint_sha256": sha(checkpoint_path),
                   "started_at": now(), "pid": os.getpid(), "policy": "NO_AUTOMATIC_RETRY"}, handle)
        handle.flush()
        os.fsync(handle.fileno())
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (payload.get("job") != job or payload.get("epoch") != 100 or
            payload.get("optimizer_steps") != 1600 or payload.get("architecture") != ARCHITECTURE or
            payload.get("heldout_access_during_fit") is not False):
        raise RuntimeError("Checkpoint is not the declared fixed endpoint")
    configure_strict(job["init_seed"])
    model = build_model(job["arm"], job["init_seed"], data.pca)
    model.load_state_dict(payload["state_dict"], strict=True)
    model = model.to("cuda").eval()
    started, rows = time.monotonic(), []
    if len(data.heldout_ids) != len(set(data.heldout_ids)) or set(data.train_ids) & set(data.heldout_ids):
        raise RuntimeError("Held-out cube identity set is invalid")
    if set(data.train_patient_ids) & set(data.heldout_patient_ids):
        raise RuntimeError("Patient separation failed at inference")
    with torch.no_grad():
        for image_id in data.heldout_ids:
            pieces = []
            for inputs in data.prediction_tiles(image_id):
                inputs = torch.as_tensor(inputs, dtype=torch.float32)
                if inputs.ndim != 4 or tuple(inputs.shape[1:]) != (814, 64, 64) or not 1 <= inputs.shape[0] <= 4:
                    raise RuntimeError("Invalid fixed-grid inference tile batch")
                if not bool(torch.isfinite(inputs).all()):
                    raise FloatingPointError("Nonfinite held-out input")
                patch_logits = model.patch_logits(inputs.to("cuda"))
                if not bool(torch.isfinite(patch_logits).all()):
                    raise FloatingPointError("Nonfinite held-out output")
                pieces.append(patch_logits.cpu().numpy().astype(np.float64))
            all_logits = np.concatenate(pieces, axis=0) if pieces else np.empty((0, 2))
            if all_logits.shape != (4, 2):
                raise RuntimeError("Inference requires all four corner 64x64 tiles of an 88x88 core")
            logits = all_logits.mean(axis=0)
            exponential = np.exp(logits - logits.max())
            probabilities = exponential / exponential.sum()
            rows.append({"image_id": image_id, "patient_id": data.patient(image_id),
                         "label": int(data.label(image_id)),
                         "logits": logits.tolist(), "probabilities": probabilities.tolist(),
                         "predicted_class": int(probabilities.argmax()), "tiles": 4})
            _notify(status_callback, phase="heldout_inference", job_id=job["job_id"],
                    images_completed=len(rows), images_total=len(data.heldout_ids),
                    elapsed_s=time.monotonic() - started)
    result = {"status": "COMPLETED", "job": dict(job), "checkpoint_sha256": sha(checkpoint_path),
              "label_unit": "tissue_core", "aggregation": "mean_spatial_then_mean_4_corner_patch_logits_then_softmax",
              "predictions": rows, "metrics": classification_metrics([r["label"] for r in rows],
                                                                         [r["probabilities"] for r in rows]),
              "elapsed_s": time.monotonic() - started, "completed_at": now(),
              "source_dataset_test_access": False}
    write_json(out_dir / "predictions.json", result)
    return result

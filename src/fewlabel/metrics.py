from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def segmentation_metrics(ytrue, yprob) -> dict:


    truth = np.asarray(ytrue)
    prob = np.asarray(yprob, dtype=np.float64)
    if truth.ndim != 1 or prob.shape != (truth.size, 2):
        raise ValueError("Expected labels [N] and probabilities [N,2].")
    if not np.isin(truth, [0, 1]).all():
        raise ValueError("Only revealed binary labels 0/1 are accepted; filter ignore pixels before scoring.")
    if not np.isfinite(prob).all() or np.any(prob < 0) or np.any(prob > 1):
        raise ValueError("Probabilities must be finite and inside [0,1].")
    if not np.allclose(prob.sum(axis=1), 1.0, rtol=0, atol=1e-6):
        raise ValueError("Each probability row must sum to one.")
    truth = truth.astype(np.int64, copy=False)
    prediction = (prob[:, 1] >= 0.5).astype(np.int64)
    confusion = np.bincount(2 * truth + prediction, minlength=4).reshape(2, 2)
    tn, fp, fn, tp = (int(v) for v in confusion.ravel())
    count = int(truth.size)
    undefined: dict[str, str] = {}

    def ratio(name: str, numerator, denominator, reason="ZERO_DENOMINATOR"):
        if denominator == 0:
            undefined[name] = "EMPTY_EVALUATION" if count == 0 else reason
            return None
        return float(numerator / denominator)

    def fixed_mean(name: str, values):
        if any(value is None for value in values):
            undefined[name] = "EMPTY_EVALUATION" if count == 0 else "UNDEFINED_FIXED_CLASS_TERM"
            return None
        return float(sum(values) / len(values))

    classwise = {}
    for cls in (0, 1):
        true_positive = int(confusion[cls, cls])
        true_support = int(confusion[cls, :].sum())
        predicted_support = int(confusion[:, cls].sum())
        false_positive = predicted_support - true_positive
        false_negative = true_support - true_positive
        prefix = f"classwise.{cls}."
        classwise[str(cls)] = {
            "support": true_support, "predicted_support": predicted_support,
            "tp": true_positive, "fp": false_positive, "fn": false_negative,
            "precision": ratio(prefix + "precision", true_positive, predicted_support, "NO_PREDICTED_CLASS"),
            "recall": ratio(prefix + "recall", true_positive, true_support, "NO_TRUE_CLASS"),
            "f1": ratio(prefix + "f1", 2 * true_positive, 2 * true_positive + false_positive + false_negative, "NO_TRUE_OR_PREDICTED_CLASS"),
            "iou": ratio(prefix + "iou", true_positive, true_positive + false_positive + false_negative, "NO_TRUE_OR_PREDICTED_CLASS"),
        }
    negative, positive = classwise["0"], classwise["1"]
    result = {
        "status": "MEASURED" if count else "UNDEFINED",
        "n_pixels": count, "threshold": 0.5, "tie_prediction": 1,
        "confusion": confusion.tolist(), "confusion_order": [["TN", "FP"], ["FN", "TP"]],
        "confusion_counts": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "support": {"0": negative["support"], "1": positive["support"]},
        "classwise": classwise,
        "accuracy": ratio("accuracy", tp + tn, count),
        "balanced_accuracy": fixed_mean("balanced_accuracy", [negative["recall"], positive["recall"]]),
        "macro_f1": fixed_mean("macro_f1", [negative["f1"], positive["f1"]]),
        "weighted_f1": ratio("weighted_f1", sum(c["support"] * c["f1"] for c in classwise.values() if c["support"]), count),
        "macro_iou": fixed_mean("macro_iou", [negative["iou"], positive["iou"]]),
        "mcc": ratio("mcc", tp * tn - fp * fn, math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))),
    }
    for name, cls, source in [
        ("tumor_dice", "1", "f1"), ("tumor_f1", "1", "f1"),
        ("iou", "1", "iou"), ("tumor_iou", "1", "iou"),
        ("sensitivity", "1", "recall"), ("specificity", "0", "recall"),
        ("precision", "1", "precision"),
    ]:
        result[name] = classwise[cls][source]
        if result[name] is None:
            undefined[name] = undefined[f"classwise.{cls}.{source}"]

    both_classes = negative["support"] > 0 and positive["support"] > 0
    for name, scorer in [("auroc", roc_auc_score), ("average_precision", average_precision_score)]:
        if both_classes:
            result[name] = float(scorer(truth, prob[:, 1]))
        else:
            result[name] = None
            undefined[name] = "EMPTY_EVALUATION" if not count else "SINGLE_TRUE_CLASS"
    result["brier"] = float(np.mean((prob[:, 1] - truth) ** 2)) if count else None
    result["brier_score"] = result["brier"]
    if not count:
        undefined["brier"] = undefined["brier_score"] = "EMPTY_EVALUATION"

    confidence = np.max(prob, axis=1)
    correctness = prediction == truth
    edges = np.linspace(0.0, 1.0, 16)
    indices = np.clip(np.searchsorted(edges, confidence, side="left") - 1, 0, 14)
    bins = []
    for index in range(15):
        selected = indices == index
        n_bin = int(selected.sum())
        bin_accuracy = float(correctness[selected].mean()) if n_bin else None
        bin_confidence = float(confidence[selected].mean()) if n_bin else None
        contribution = n_bin / count * abs(bin_accuracy - bin_confidence) if n_bin else 0.0
        bins.append({"lower": float(edges[index]), "upper": float(edges[index + 1]),
                     "lower_inclusive": index == 0, "upper_inclusive": True,
                     "count": n_bin, "accuracy": bin_accuracy, "confidence": bin_confidence,
                     "ece_contribution": float(contribution)})
    result["ece15"] = float(sum(b["ece_contribution"] for b in bins)) if count else None
    result["ece"] = result["ece15"]
    result["calibration_bins"] = bins
    if not count:
        undefined["ece15"] = undefined["ece"] = "EMPTY_EVALUATION"
    result["boundary_f1"] = result["hd95"] = None
    result["not_applicable"] = {"boundary_f1": "SPARSE_OR_UNSTRUCTURED_EVALUATION",
                                "hd95": "SPARSE_OR_UNSTRUCTURED_EVALUATION"}
    result["undefined"] = undefined
    excluded = {"threshold", "tie_prediction", "n_pixels", "status"}
    result["metric_status"] = {name: "UNDEFINED" if value is None else "MEASURED"
                               for name, value in result.items()
                               if name not in excluded and (value is None or isinstance(value, (int, float)))}
    result["metric_status"].update({name: "NOT_APPLICABLE" for name in result["not_applicable"]})
    return result


def image_summary(image_metrics: Mapping[str, dict]) -> dict:


    if not isinstance(image_metrics, Mapping):
        raise TypeError("Expected a mapping from image_id to segmentation metrics.")
    values, undefined_images = [], {}
    for image_id, metrics in image_metrics.items():
        value = metrics.get("macro_f1")
        if value is None:
            undefined_images[str(image_id)] = metrics.get("undefined", {}).get("macro_f1", "MISSING_MACRO_F1")
        elif not isinstance(value, (int, float, np.number)) or not math.isfinite(float(value)) or not 0 <= value <= 1:
            raise ValueError("Image macro_f1 must be null or a finite number in [0,1].")
        else:
            values.append(float(value))
    status = "COMPLETE" if values and not undefined_images else "INCOMPLETE" if values else "UNDEFINED"
    return {"image_mean_macro_f1": float(np.mean(values)) if values else None,
            "status": status, "aggregation": "equal_weight_per_valid_image",
            "total_images": len(image_metrics), "valid_images": len(values),
            "undefined_images": len(undefined_images), "undefined_image_reasons": undefined_images,
            "reason_code": None if status == "COMPLETE" else "INCOMPLETE_IMAGE_SUPPORT" if values else "NO_VALID_IMAGE_METRICS"}

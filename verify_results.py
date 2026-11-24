import csv
from hashlib import sha256
from itertools import product
import json
from math import isclose
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parent / "results"
SEEDS = {938, 476, 993, 518, 663, 628, 828, 353, 621, 306}


def read(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def equal(actual, expected):
    if not isclose(actual, expected, rel_tol=1e-11, abs_tol=1e-12):
        raise ValueError(f"Value mismatch: {actual} vs {expected}")


def verify_r4():
    study = read("r4_exploratory_test.json")
    models = {"ColonSpec8_R4_PARENT", "R4_STATIC", "R4_ZERO", "R4_ADDITIVE", "R4_FULL"}
    images = {f"HSI_test_{i}" for i in range(1, 11)}
    if set(study["models"]) != models or set(study["images"]) != images or len(study["images"]) != 10:
        raise ValueError("Incomplete R4 comparison")
    if set(study["seeds"]) != SEEDS or len(study["seeds"]) != len(SEEDS):
        raise ValueError("Unexpected R4 seed grid")
    for model, result in study["models"].items():
        if set(result["by_seed"]) != {str(seed) for seed in SEEDS}:
            raise ValueError(f"Incomplete R4 seed results for {model}")
        seed_means = []
        for seed, row in result["by_seed"].items():
            values = row["per_image"]
            if set(values) != images or not all(0 <= x <= 1 for x in values.values()):
                raise ValueError(f"Invalid R4 image results for {model}, seed {seed}")
            score = mean(values.values())
            equal(score, row["macro_f1"])
            seed_means.append(score)
        equal(mean(seed_means), result["mean_macro_f1"])
        equal(stdev(seed_means), result["sample_sd"])


def main():
    for name, provenance in read("provenance.json").items():
        if sha256((ROOT / name).read_bytes()).hexdigest() != provenance["released_sha256"]:
            raise ValueError(f"Release digest mismatch: {name}")
    with (ROOT / "segmentation_per_seed.csv").open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    summary = read("segmentation_summary.json")
    models = {r["model"] for r in summary["models"]}
    pairs = {(r["model"], int(r["seed"])) for r in rows}
    if len(models) != 8 or len(rows) != 80 or pairs != set(product(models, SEEDS)):
        raise ValueError("Incomplete segmentation run grid")
    checkpoints = read("segmentation_checkpoints.json")
    if len(checkpoints) != 80 or pairs != {(r["model"],r["seed"]) for r in checkpoints}:
        raise ValueError("Incomplete checkpoint index")
    for model in summary["models"]:
        scores = [float(r["image_mean_macro_f1"]) for r in rows if r["model"] == model["model"]]
        metric = model["groups"]["all_ten"]["image_mean"]["macro_f1"]
        equal(mean(scores), metric["mean"])
        equal(stdev(scores), metric["sd"])
    mask = read("mask_initialization.json")
    pairs = mask["pairs"]
    three = {938, 476, 993}
    if len(pairs) != 9 or {(r["mask_seed"],r["init_seed"]) for r in pairs} != set(product(three, three)):
        raise ValueError("Incomplete initialization control grid")
    deltas = [r["fixed_end_f1_difference_pp"] for r in pairs]
    equal(mean(deltas),mask["primary_mean_difference_pp"])
    equal(sum(d > 0 for d in deltas),mask["primary_positive_pairs"])
    ftir = read("ftir_classification.json")
    if (ftir["tissue_core_count"],ftir["independent_patient_count"]) != (238,75):
        raise ValueError("FTIR cohort mismatch")
    for arm in ("learned","random","pca8"):
        groups = [g for g in ftir["groups"] if g["arm"] == arm]
        if {g["init_seed"] for g in groups} != three or len(groups) != 3:
            raise ValueError("Incomplete FTIR out-of-fold summary")
        scores = []
        for g in groups:
            (tn, fp), (fn, tp) = g["metrics"]["confusion_matrix"]
            if tn+fp+fn+tp != 238:
                raise ValueError("Incomplete FTIR out-of-fold counts")
            score = (2*tp/(2*tp+fp+fn) + 2*tn/(2*tn+fp+fn))/2
            equal(score,g["metrics"]["macro_f1"])
            scores.append(score)
        equal(mean(scores),ftir["oof_by_arm"][arm]["macro_f1_mean"])
        equal(stdev(scores),ftir["oof_by_arm"][arm]["macro_f1_sd"])
    verify_r4()
    print("Verified 80 primary segmentation summaries, 9 paired controls, 9 FTIR out-of-fold "
          "summaries and 500 exploratory R4 comparison scores (5 models, 10 seeds, 10 images).")


if __name__ == "__main__":
    main()

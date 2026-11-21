import statistics
import numpy as np
from pathlib import Path
from fewlabel.runtime import read_json as read, now
from .acquisition import RUN, SCOPE, write_json
ARMS=("learned","random","pca8")
SEEDS=(938,476,993)

def analysis_report(root, cfg, predictions):
    from .training import classification_metrics
    root = Path(root)
    data = read(root / RUN / "artifacts/dataset.json")
    sample_ids = {r["image_id"] for r in data["samples"]}
    label_map = {r["image_id"]: r["class_index"] for r in data["samples"]}
    patient_map = {r["image_id"]: str(r["patient_id"]) for r in data["samples"]}
    groups, oof_rows = [], []
    if len(predictions) != 27:
        raise ValueError("Expected 27 held-out evaluations")
    for arm in ARMS:
        for seed in SEEDS:
            rows = []
            for job in cfg["jobs"]:
                if (job["arm"], job["init_seed"]) != (arm, seed):
                    continue
                value = predictions[job["job_id"]]
                if value["job"] != job or value["status"] != "COMPLETED":
                    raise ValueError("Prediction job identity mismatch")
                fold_ids = set(data["folds"][job["fold"]]["heldout_ids"])
                found = [r["image_id"] for r in value["predictions"]]
                if set(found) != fold_ids or len(found) != len(fold_ids):
                    raise ValueError("Held-out fold cube identities incomplete")
                for row in value["predictions"]:
                    if row["label"] != label_map[row["image_id"]]:
                        raise ValueError("Held-out label differs from class file")
                    if str(row["patient_id"]) != patient_map[row["image_id"]]:
                        raise ValueError("Held-out patient differs from published map")
                    rows.append(row)
                    oof_rows.append(dict(arm=arm, init_seed=seed, fold=job["fold"], **row))
            if len(rows) != 238 or {r["image_id"] for r in rows} != sample_ids:
                raise ValueError("Each cube must appear once per method/initialization")
            metric = classification_metrics([r["label"] for r in rows], [r["probabilities"] for r in rows])
            groups.append(dict(arm=arm, init_seed=seed, metrics=metric))
    summary = {}
    for arm in ARMS:
        rows = [r["metrics"] for r in groups if r["arm"] == arm]
        summary[arm] = dict(initialization_count=3, sample_count_per_initialization=238,
                           patient_count_per_initialization=75)
        for metric in ("macro_f1", "accuracy", "balanced_accuracy", "mcc",
                       "negative_log_likelihood", "brier_multiclass_sum", "ece10",
                       "macro_ovr_auroc", "macro_ovr_average_precision",
                       "sensitivity_crc", "specificity_nc", "auroc_crc", "average_precision_crc"):
            values = [r[metric] for r in rows]
            summary[arm][metric + "_mean"] = statistics.mean(values)
            summary[arm][metric + "_sd"] = statistics.stdev(values)
    paired = []
    for baseline in ("random", "pca8"):
        differences = []
        for seed in SEEDS:
            lookup = {r["arm"]: r["metrics"]["macro_f1"] for r in groups if r["init_seed"] == seed}
            differences.append(dict(init_seed=seed, difference_pp=100 * (lookup["learned"] - lookup[baseline])))
        paired.append(dict(contrast="learned-" + baseline, rows=differences,
                           mean_difference_pp=statistics.mean(r["difference_pp"] for r in differences)))
    cluster = patient_cluster_bootstrap(oof_rows, seed=741, draws=2000)
    report = dict(scope=SCOPE, status="COMPLETED", created_at=now(), groups=groups,
                  oof_by_arm=summary, paired_contrasts=paired, oof_rows=oof_rows,
                  tissue_core_count=238, independent_patient_count=75, patient_mapping_verified=True,
                  patient_cluster_bootstrap=cluster,
                  statistical_significance_claim=False, original_colon_test_access=False,
                  interpretation="Three fixed reinitializations of patient-grouped OOF core classification on 75 patients; 238 correlated cores, not 2142 independent samples. FTIR method replication, not original-checkpoint external segmentation or clinical validation. Cluster confidence intervals condition on this fixed partition and fitted models.")
    write_json(root / RUN / "reports/analysis_report.json", report)
    return report


def patient_cluster_bootstrap(rows, *, seed=741, draws=2000):


    import numpy as np
    patients = sorted({str(row["patient_id"]) for row in rows})
    identities = sorted({str(row["image_id"]) for row in rows})
    if not patients or draws < 2:
        raise ValueError("No patient clusters or insufficient bootstrap draws")
    pindex = {patient: index for index, patient in enumerate(patients)}
    matrices = np.zeros((len(ARMS), len(SEEDS), len(patients), 2, 2), np.int64)
    references = None
    for ai, arm in enumerate(ARMS):
        for si, init in enumerate(SEEDS):
            selected = [r for r in rows if r["arm"] == arm and r["init_seed"] == init]
            found = [str(r["image_id"]) for r in selected]
            if len(found) != len(identities) or sorted(found) != identities:
                raise ValueError("Unpaired or duplicate core predictions in patient bootstrap")
            mapping = {str(r["image_id"]): (str(r["patient_id"]), int(r["label"])) for r in selected}
            if references is not None and mapping != references:
                raise ValueError("Labels/patients changed between paired methods")
            references = mapping
            for row in selected:
                pred = int(np.asarray(row["probabilities"]).argmax())
                matrices[ai, si, pindex[str(row["patient_id"])], int(row["label"]), pred] += 1

    def scores(counts):
        tp = np.diagonal(counts, axis1=-2, axis2=-1)
        denom = counts.sum(-1) + counts.sum(-2)
        f1 = np.divide(2 * tp, denom, out=np.zeros_like(denom, dtype=float), where=denom > 0)
        return f1.mean(-1).mean(-1)

    point = scores(matrices.sum(2))
    rng = np.random.default_rng(seed)
    values = np.empty((draws, len(ARMS)), dtype=float)
    for index in range(draws):
        weights = np.bincount(rng.integers(0, len(patients), size=len(patients)), minlength=len(patients))
        values[index] = scores(np.einsum("aspij,p->asij", matrices, weights))
    arms = {arm: dict(mean_seed_macro_f1=float(point[i]),
                     ci95_percentile=np.quantile(values[:, i], [.025, .975]).tolist())
            for i, arm in enumerate(ARMS)}
    contrasts = {}
    for i, baseline in enumerate(ARMS[1:], 1):
        differences = 100 * (values[:, 0] - values[:, i])
        contrasts["learned-" + baseline] = dict(difference_pp=float(100 * (point[0] - point[i])),
            ci95_percentile_pp=np.quantile(differences, [.025, .975]).tolist())
    return dict(resampling_unit="patient", core_metric="macro_f1", patients=len(patients),
                draws=draws, seed=seed, confidence=.95, arms=arms, paired_contrasts=contrasts,
                all_cores_retained_per_sampled_patient=True, initialization_replicates_fixed=True,
                inference_models_fixed=True, independent_test_cohort_replicates=1,
                scope="Conditional uncertainty of fixed patient-grouped OOF predictions; no refitting")

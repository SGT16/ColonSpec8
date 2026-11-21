import argparse
import importlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
SEEDS = (938, 476, 993, 518, 663, 628, 828, 353, 621, 306)
FACTORIES = {
    "local_only_supervised": "fewlabel_local_supervised.models",
    "local_frozen_projection": "fewlabel_frozen_projection.models",
    "local_pca8_supervised": "fewlabel_pca8.models",
    "mamba_mt": "fewlabel_multiscale.models",
    "unet_mt": "fewlabel_v21.models",
    "nnunet2d_supervised": "study_comparators",
    "hybridsn_supervised": "study_comparators",
}
GM = "gmtssa_train_only"
MODELS = tuple(FACTORIES) + (GM,)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def config(name):
    return read(ROOT / "configs" / (name + ".json"))


def prepared(seed):
    from fewlabel.data import load_prepared
    from fewlabel.runtime import fingerprint
    data = load_prepared(ROOT, config("project"), seed, .005)
    cache = Path.home() / ".cache/fewlabel_mamba" / fingerprint(str(ROOT))[:16] / "hwc_v1"
    data.enable_local_io_cache(cache)
    return data


def prepare():
    from fewlabel.data import prepare as prepare_cubes
    from fewlabel.runtime import sha, write_json
    import shutil
    files = []
    for name, (size, expected) in config("data_checksums").items():
        path = ROOT / "data/raw" / name
        if path.stat().st_size != size or sha(path) != expected:
            raise ValueError(f"Source file differs from the published study input: {name}")
        files.append(dict(name=name, actual_size_bytes=size, sha256=expected, status="COMPLETED"))
    write_json(ROOT / "data/metadata/download_integrity.json",
               dict(completed_files=len(files), download_metadata_prepared=True, files=files))
    src, dest = ROOT / "configs/preparation_seeds.json", ROOT / "state/seeds.json"
    dest.parent.mkdir(exist_ok=True)
    if dest.exists() and read(src) != read(dest):
        raise ValueError("Seed manifest changed")
    shutil.copyfile(src, dest)
    prepare_cubes(ROOT, config("project"))
    prepared(SEEDS[0])
    from fewlabel_pca8.pca import ensure_pca
    from fewlabel_external.common import ensure_pca30
    ensure_pca(ROOT)
    ensure_pca30()


def train(model, seed):
    from study_comparators import fit_gm
    data = prepared(seed)
    out = ROOT / "outputs/segmentation" / model / str(seed)
    out.mkdir(parents=True, exist_ok=False)
    if model == GM:
        fit_gm(data, out)
        return
    import torch
    from fewlabel_v21.training import with_model_factory
    torch.set_num_threads(4)
    cfg = config(model)
    if model == "local_pca8_supervised":
        from fewlabel_pca8.training import train_job as trainer
    else:
        trainer = with_model_factory(importlib.import_module(FACTORIES[model]).build_model)
    if model == "hybridsn_supervised":
        original = trainer.__globals__["ema_update"]
        def update(teacher, student, momentum):
            original(teacher, student, momentum)
            with torch.no_grad():
                teacher.pca.weight.copy_(student.pca.weight)
                teacher.pca.bias.copy_(student.pca.bias)
        trainer.__globals__["ema_update"] = update
    trainer(ROOT, cfg, model, seed, data, out, None)


def mask_control(mask, init, model):
    import torch
    from colon_maskstudy.models import create_initial_state
    from colon_maskstudy.data import create_shared_plan, load_data
    from colon_maskstudy.training import train_job
    from fewlabel.runtime import sha
    cfg = config("mask_initialization")
    common = ROOT / "experiments/maskstudy_v1"
    plan = ROOT / cfg["maskstudy"]["plan_path"]
    initial = common / f"init_{init}.pt"
    data = load_data(ROOT, mask)
    data.local_cubes = prepared(mask).local_cubes
    if not plan.exists():
        create_shared_plan(data, cfg, 741, plan)
    cfg["maskstudy"].update(plan_sha256=sha(plan), plan_path=str(plan))
    if not initial.exists():
        create_initial_state(cfg, init, initial)
    torch.set_num_threads(4)
    out = ROOT / "outputs/maskstudy" / f"{model}_mask{mask}_init{init}"
    out.mkdir(parents=True, exist_ok=False)
    train_job(ROOT, cfg, model, mask, init, data, out, None, initial)


def freeze():
    from fewlabel.runtime import sha, write_json
    rows = []
    for model in MODELS:
        for seed in SEEDS:
            out = ROOT / "outputs/segmentation" / model / str(seed)
            result = read(out / "result.json")
            checkpoint = out / ("model.npz" if model == GM else "best.pt")
            identity = sha(checkpoint)
            if result["status"] != "COMPLETED" or result["checkpoint_sha256"] != identity:
                raise ValueError(f"Incomplete or changed checkpoint: {model}/{seed}")
            if (result["model"], result["seed"]) != (model, seed):
                raise ValueError("Checkpoint and training result disagree")
            rows.append(dict(model=model, seed=seed, file=checkpoint.relative_to(ROOT).as_posix(),
                             sha256=identity))
    path = ROOT / "outputs/test_registry.json"
    if path.exists() and read(path) != rows:
        raise ValueError("A different freeze already exists")
    write_json(path, rows)


def evaluate(model, seed):
    import numpy as np
    from fewlabel.runtime import sha, write_json
    from colon_final_test.inference import infer_image
    from colon_final_test.metrics import dense_neural_metrics, dense_gm_metrics
    from test_images import TestImages
    rows = read(ROOT / "outputs/test_registry.json")
    if {(r["model"],r["seed"]) for r in rows} != {(m,s) for m in MODELS for s in SEEDS} or len(rows) != 80:
        raise ValueError("Expected 80 frozen checkpoints")
    row = next(r for r in rows if (r["model"], r["seed"]) == (model, seed))
    path = ROOT / row["file"]
    if sha(path) != row["sha256"]:
        raise ValueError("Frozen checkpoint changed")
    out = ROOT / "outputs/test" / model / str(seed)
    out.mkdir(parents=True, exist_ok=False)
    if model == GM:
        from colon_final_test.gm_inference import infer_gm_image
        from fewlabel_gmtssa.model import TrainOnlyGMTSSA
        student = TrainOnlyGMTSSA.load(path)
    else:
        import torch
        torch.set_num_threads(4)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        cfg = config(model)
        torch.manual_seed(seed)
        student = importlib.import_module(FACTORIES[model]).build_model(model, cfg)
        weights = torch.load(path, map_location="cpu", weights_only=True)
        if (weights["model"], weights["seed"], weights["prediction_model"]) != (model, seed, "student"):
            raise ValueError("Unexpected model or prediction head")
        student.load_state_dict(weights["state_dict"], strict=True)
        student.requires_grad_(False).eval().to("cuda")
    provider = TestImages(ROOT)
    results = {}
    try:
        for image in provider.ids:
            provider.prepare_image(image)
            provider.open_image(image)
            if model == GM:
                scores = infer_gm_image(provider, image, {seed: student})[seed]
                metrics, _, _ = dense_gm_metrics(provider.labels(image), scores)
            else:
                scores = infer_image(student, provider, image, cfg)
                metrics = dense_neural_metrics(provider.labels(image), scores)
            results[image] = metrics
    finally:
        provider.close()
    write_json(out / "metrics.json", dict(images=results, model=model, seed=seed,
               image_mean_macro_f1=float(np.mean([r["macro_f1"] for r in results.values()]))))


def main():
    parser = argparse.ArgumentParser(description="ColonSpec8 research experiments")
    subs = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "freeze", "ftir-evaluate"):
        subs.add_parser(name)
    for name in ("train", "evaluate"):
        p = subs.add_parser(name)
        p.add_argument("--model", choices=MODELS, required=True)
        p.add_argument("--seed", type=int, choices=SEEDS, required=True)
    p = subs.add_parser("mask-control")
    p.add_argument("--model", required=True, choices=MODELS[:2])
    p.add_argument("--mask", type=int, choices=SEEDS[:3], required=True)
    p.add_argument("--init", type=int, choices=SEEDS[:3], required=True)
    subs.add_parser("ftir-prepare").add_argument("zip", type=Path)
    p = subs.add_parser("ftir-train")
    p.add_argument("--arm", choices=("learned","random","pca8"), required=True)
    p.add_argument("--fold", type=int, choices=(0,1,2), required=True)
    p.add_argument("--init", type=int, choices=SEEDS[:3], required=True)
    args = parser.parse_args()
    if args.command.startswith("ftir-"):
        from ftir_experiment import execute
        execute(ROOT, args)
    elif args.command == "prepare":
        prepare()
    elif args.command == "freeze":
        freeze()
    elif args.command == "mask-control":
        mask_control(args.mask, args.init, args.model)
    elif args.command == "train":
        train(args.model, args.seed)
    else:
        evaluate(args.model, args.seed)


if __name__ == "__main__":
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    main()

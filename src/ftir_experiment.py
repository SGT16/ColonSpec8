import shutil
from zipfile import ZipFile

from colon_borkovits.acquisition import DEST, META, RUN, checksums, metadata_records, sha, write_json
from colon_borkovits.data import configuration, FoldData


def prepare_archive(root, archive):
    target = root / DEST
    meta = root / META
    meta.mkdir(parents=True, exist_ok=False)
    raw = target / "raw"
    raw.mkdir(parents=True, exist_ok=False)
    directory = target / "metadata"
    directory.mkdir(exist_ok=False)
    with ZipFile(archive) as source:
        entries = source.infolist()
        def export(entry, dest):
            with source.open(entry) as inp, dest.open("xb") as out:
                shutil.copyfileobj(inp, out)
        names = ["checksums.md5", "README.txt"] + [f"TMA{i}_core_metadata.csv" for i in range(1,10)]
        for name in names:
            parent = name.split("_")[0] + "/" if name.startswith("TMA") else ""
            export("multimodal_CRC_dataset/" + parent + name, meta / name)
            shutil.copyfile(meta / name, directory / name)
        digests = checksums(root)
        records = []
        for core in metadata_records(root):
            tma = core["tma"]
            prefix = f"multimodal_CRC_dataset/TMA{tma}/MIR/spectra/tma{tma}_{core['core']}_"
            entries_for_core = [e for e in entries if e.filename.startswith(prefix) and e.filename.endswith(".txt")]
            if len(entries_for_core) != 1:
                raise ValueError("Ambiguous core spectrum")
            entry, = entries_for_core
            dest = raw / (core["image_id"] + ".txt")
            export(entry, dest)
            if sha(dest, "md5") != digests[entry.filename]:
                raise ValueError("Published spectral checksum mismatch")
            records.append(dict(**core, archive_entry=entry.filename, path=dest.relative_to(root).as_posix(),
                           sha256=sha(dest), published_md5=digests[entry.filename], published_md5_pass=True,
                           zip_crc_pass=True, compressed_bytes=entry.compress_size))
    write_json(directory/"selected_archive_entries.json", dict(entries=[r["archive_entry"] for r in records]))
    write_json(directory/"download_manifest.json", dict(status="COMPLETED_VERIFIED_SUBSET_DOWNLOAD", files=records))
    from colon_borkovits_headerfix_v1.data import prepare
    prepare(root)


def execute(root, args):
    if args.command == "ftir-prepare":
        prepare_archive(root, args.zip)
        return
    from colon_borkovits.training import train_job, predict_job
    from fewlabel.runtime import read_json
    cfg = configuration(root)
    outputs = root / "outputs/ftir"
    if args.command == "ftir-train":
        job = next(j for j in cfg["jobs"] if (j["arm"], j["fold"], j["init_seed"]) ==
                   (args.arm, args.fold, args.init))
        out = outputs / job["job_id"]
        out.mkdir(parents=True, exist_ok=False)
        train_job(cfg, job, FoldData(root, job["fold"]), out)
        return
    checkpoints = {}
    for job in cfg["jobs"]:
        job_id = job["job_id"]
        result = read_json(outputs / job_id / "result.json")
        path = outputs / job_id / "final.pt"
        if (result["job"] != job or result["status"] != "COMPLETED" or
                result["checkpoint_sha256"] != sha(path)):
            raise ValueError("Incomplete or changed FTIR checkpoint")
        checkpoints[job_id] = path
    from colon_borkovits.queue import analysis_report
    predictions = {}
    for job in cfg["jobs"]:
        job_id = job["job_id"]
        predictions[job_id] = predict_job(cfg, job, FoldData(root,job["fold"]), checkpoints[job_id],
                                          outputs / job_id / "evaluation")
    analysis_report(root, cfg, predictions)

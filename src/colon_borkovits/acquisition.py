import csv
import hashlib
import json
import os
import re
import time
from pathlib import Path
from datetime import datetime, timezone
ROOT=Path(__file__).resolve().parents[2]
META=Path("data/external/borkovits_metadata")
DEST=Path("data/external/borkovits_v1")
RUN=Path("experiments/borkovits_v1")
SCOPE="BORKOVITS_MIR814_PATIENT_GROUPED_COLON_CLASSIFICATION_V1"

def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf8')
    deadline = time.monotonic() + 10
    while True:
        try:
            os.replace(temp, path)
            return
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(.1)


def checksums(root):
    result = {}
    for line in (Path(root) / META / 'checksums.md5').read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        result['multimodal_CRC_dataset/' + name.removeprefix('./')] = digest
    return result


def metadata_records(root=ROOT):
    root = Path(root)
    digests = checksums(root)
    records = []
    for tma in range(1, 10):
        source = root / META / f'TMA{tma}_core_metadata.csv'
        archive_name = f'multimodal_CRC_dataset/TMA{tma}/TMA{tma}_core_metadata.csv'
        if sha(source, 'md5') != digests[archive_name]:
            raise ValueError('Authoritative patient/class CSV failed MD5')
        with source.open(encoding='utf8', newline='') as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            if row['has_MIR'] != 'yes' or row['cancer'] not in ('NC', 'CRC'):
                continue
            patient = row['patient'].strip()
            core = row['cores'].strip()
            if not patient or not re.fullmatch(r'[B-I][1-6]', core):
                raise ValueError('Missing patient or invalid core identity')
            records.append(dict(image_id=f'BORKOVITS2026_TMA{tma}_{core}', tma=tma,
                core=core, patient_id=patient, class_code=row['cancer'],
                class_index=('NC', 'CRC').index(row['cancer']),
                label=row['cancer'], metadata_source=source.relative_to(root).as_posix(),
                metadata_sha256=sha(source)))
    if len(records) != 238 or len({r['patient_id'] for r in records}) != 75:
        raise ValueError('Unexpected NC/CRC subset size')
    if [sum(r['class_index'] == c for r in records) for c in range(2)] != [117, 121]:
        raise ValueError('Unexpected NC/CRC class counts')
    return sorted(records, key=lambda r: (r['tma'], r['core']))

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def clean(value):
    import math
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [clean(v) for v in value]
    if hasattr(value, "item") and not hasattr(value, "__len__"): return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value): return None
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(clean(value), handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""): digest.update(block)
    return digest.hexdigest()


def derived_seed(seed, *parts):
    return int.from_bytes(hashlib.sha256("|".join(map(str, (seed,) + parts)).encode()).digest()[:8], "little") % (2**63 - 1)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

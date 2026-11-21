from pathlib import Path

def safe_path(root, path):
    candidate = (Path(root) / path).resolve()
    if not candidate.is_relative_to(Path(root).resolve()):
        raise RuntimeError('Artifact path escapes its experiment')
    return candidate

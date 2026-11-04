"""Read-only verification of the original package, including restored data."""

import hashlib
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# util/SHA256SUMS 의 `data/…`는 data_dir, `model/…`는 model_dir, 나머지는 패키지 루트 기준
DATA_PREFIX = "data/"
MODEL_PREFIX = "model/"


def verify_original(root, data_dir=None, model_dir=None):
    root = Path(root).resolve()
    relocated = {DATA_PREFIX: data_dir, MODEL_PREFIX: model_dir}
    results = {}
    for line in (root / "util" / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split("  ", 1)
        base, path = root, root / name
        for prefix, directory in relocated.items():
            if directory is not None and name.startswith(prefix):
                base = Path(directory).resolve()
                path = base / name[len(prefix) :]
        if path.is_symlink() or not path.resolve().is_relative_to(base):
            raise ValueError(f"Unsafe source path: {name}")
        if not path.is_file():
            if name.startswith(MODEL_PREFIX):
                raise FileNotFoundError(
                    f"Missing checkpoint {path}. Copy {path.name} into model/deemc "
                    "from the submission archive"
                )
            raise FileNotFoundError(
                f"Missing original file {path}. "
                "Run: python -m MAF.datamodule.deemc.download_data"
            )
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"Original file checksum mismatch: {name}")
        results[name] = actual
    return results

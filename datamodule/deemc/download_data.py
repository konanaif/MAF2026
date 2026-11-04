#!/usr/bin/env python3
"""Acquire MovieLens 1M from GroupLens and verify the exact evaluation data.

Run from any working directory. An existing file is never overwritten; an
existing file with a different checksum is an error. Only three named archive
members are read, and all their checksums are checked before any installation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import urllib.request
import zipfile


MAF_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DESTINATION = MAF_ROOT / "data" / "deemc"
DEFAULT_MANIFEST = DEFAULT_DESTINATION / "split_manifest.json"
DOWNLOAD_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
ARCHIVE_MEMBERS = {
    "movielens_1m_dataset.dat": "ml-1m/ratings.dat",
    "users.dat": "ml-1m/users.dat",
    "movies.dat": "ml-1m/movies.dat",
}
MAX_BYTES = 100 * 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_expected_hashes(manifest: Path) -> dict[str, str]:
    hashes = json.loads(manifest.read_text(encoding="utf-8"))["data_sha256"]
    validate_expected_hashes(hashes)
    return hashes


def validate_expected_hashes(hashes: dict[str, str]) -> None:
    if set(hashes) != set(ARCHIVE_MEMBERS):
        raise ValueError("The split manifest must specify exactly the three MovieLens data files")
    for name, digest in hashes.items():
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"Invalid SHA-256 in split manifest: {name}")


def check_existing(destination: Path, hashes: dict[str, str]) -> list[str]:
    """Return missing files, raising before any writes if a local file differs."""
    validate_expected_hashes(hashes)
    missing = []
    for name, expected in hashes.items():
        path = destination / name
        if path.is_symlink():
            raise ValueError(f"Refusing a symlink at the data destination: {path}")
        if not path.exists():
            missing.append(name)
        elif not path.is_file() or sha256_file(path) != expected:
            raise ValueError(
                f"Existing file does not match the saved split: {path}. "
                "It was not overwritten; move it aside explicitly before retrying."
            )
    return missing


def read_verified_archive(archive: Path, hashes: dict[str, str]) -> dict[str, bytes]:
    """Read only whitelisted members; verify every payload before returning."""
    validate_expected_hashes(hashes)
    payloads = {}
    with zipfile.ZipFile(archive) as source:
        names = source.namelist()
        for target, member in ARCHIVE_MEMBERS.items():
            if names.count(member) != 1:
                raise ValueError(f"Expected exactly one archive member: {member}")
            info = source.getinfo(member)
            if info.is_dir() or info.file_size > MAX_BYTES:
                raise ValueError(f"Invalid or oversized archive member: {member}")
            with source.open(info) as stream:
                data = stream.read(MAX_BYTES + 1)
            if len(data) > MAX_BYTES or hashlib.sha256(data).hexdigest() != hashes[target]:
                raise ValueError(f"SHA-256 mismatch for {member}; no data files were installed")
            payloads[target] = data
    return payloads


def _write_missing_atomically(path: Path, data: bytes, expected_hash: str) -> None:
    """Publish a complete file without replacing a concurrently created file."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            # An atomic hard link creates the final filename only if it is absent.
            # Unlike replace(), this cannot silently overwrite an existing file.
            os.link(temporary, path)
        except FileExistsError:
            if path.is_symlink() or not path.is_file() or sha256_file(path) != expected_hash:
                raise ValueError(f"Data destination changed during installation: {path}") from None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def install_archive(archive: Path, destination: Path, hashes: dict[str, str]) -> list[str]:
    """Install only missing files from a fully verified archive."""
    missing = check_existing(destination, hashes)
    if not missing:
        return []
    payloads = read_verified_archive(archive, hashes)
    destination.mkdir(parents=True, exist_ok=True)
    for name in missing:
        _write_missing_atomically(destination / name, payloads[name], hashes[name])
    check_existing(destination, hashes)
    return missing


def download_archive(destination: Path) -> None:
    request = urllib.request.Request(DOWNLOAD_URL, headers={"User-Agent": "DEE-MC-MAF/1.0"})
    total = 0
    with urllib.request.urlopen(request, timeout=60) as response, destination.open("wb") as stream:
        while chunk := response.read(1024 * 1024):
            total += len(chunk)
            if total > MAX_BYTES:
                raise ValueError("Downloaded archive exceeds the 100 MiB limit")
            stream.write(chunk)


def ensure_data(
    destination: Path = DEFAULT_DESTINATION,
    manifest: Path = DEFAULT_MANIFEST,
) -> list[str]:
    """Download missing MovieLens files into destination; return installed names."""
    hashes = load_expected_hashes(manifest)
    if not check_existing(destination, hashes):
        return []
    print(f"Downloading MovieLens 1M from {DOWNLOAD_URL}", flush=True)
    with tempfile.TemporaryDirectory(prefix="dee_mc_movielens_") as temporary:
        downloaded = Path(temporary) / "ml-1m.zip"
        try:
            download_archive(downloaded)
        except OSError as error:
            raise OSError(
                f"Download from {DOWNLOAD_URL} failed ({error}). Without network access, "
                f"place {', '.join(ARCHIVE_MEMBERS)} from ml-1m.zip into {destination}."
            ) from error
        return install_archive(downloaded, destination, hashes)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    try:
        installed = ensure_data()
        print(f"Verified all three SHA-256 values; installed {len(installed)} file(s) in {DEFAULT_DESTINATION}")
        return 0
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
        parser.exit(1, f"MovieLens 1M setup failed: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())

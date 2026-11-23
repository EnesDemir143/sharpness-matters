"""Freeze-once manifests: write an artifact, its SHA-256 and its creation record.

# B (protocol Section 4.5 / Section 4.7): manifests were produced ad hoc, with no
#   recorded hash, so "same list" could not be proven after the fact. New: every
#   manifest csv is written once next to `<name>.sha256` and `<name>.record.json`;
#   readers only load when the hash matches and otherwise raise (never silently
#   rebuild, never overwrite frozen data).
#   check: tests/corrected/test_data.py::test_frozen_* / test_e5_clean_*
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict

import pandas as pd

from sharpness_matters.config.load_config import cfg

_CHUNK_SIZE = 1024 * 1024
_HASH_SUFFIX = ".sha256"
_RECORD_SUFFIX = ".record.json"


def sha256_file(path: Path) -> str:
    """SHA-256 of a file, streamed in 1 MiB chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_root() -> Path:
    """Repository root (the parent of the ``sharpness_matters`` package)."""
    return Path(__file__).resolve().parents[3]


def manifest_dir() -> Path:
    """Manifest directory: ``repo_root() / cfg.ptx.corrected.manifest_dir``."""
    return repo_root() / str(cfg.ptx.corrected.manifest_dir)


def hash_path(path: Path) -> Path:
    """Sibling path holding the SHA-256 of ``path``."""
    return Path(str(path) + _HASH_SUFFIX)


def record_path(path: Path) -> Path:
    """Sibling path holding the creation record of ``path``."""
    return Path(str(path) + _RECORD_SUFFIX)


def _id_dtypes(path: Path) -> Dict[str, type]:
    """dtype map that keeps ID columns as strings (``image_id``, ``*_id``, ``id``)."""
    columns = pd.read_csv(path, nrows=0).columns
    return {
        column: str
        for column in columns
        if column == "id" or column.endswith("_id")
    }


def _environment_record() -> Dict[str, object]:
    record: Dict[str, object] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
    }
    try:
        import numpy
        import pandas

        record["numpy_version"] = numpy.__version__
        record["pandas_version"] = pandas.__version__
    except ImportError:  # pragma: no cover - both are hard dependencies
        pass
    try:
        import torch

        record["torch_version"] = torch.__version__
    except ImportError:  # pragma: no cover - torch is a hard dependency
        pass
    return record


def _display_path(path: Path) -> str:
    root = repo_root()
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def load_or_create_frozen(
    path: Path,
    build_fn: Callable[[], pd.DataFrame],
    record: dict,
) -> pd.DataFrame:
    """Load a frozen csv, or build and freeze it once.

    States (contract module 1):
      * csv + ``<path>.sha256`` exist and the hash matches -> load the csv, no rebuild;
      * neither exists -> ``build_fn()``, then write csv, hash and creation record;
      * anything else (csv without hash, hash without csv, hash mismatch, record
        without csv) -> ``RuntimeError``. Frozen artifacts are never rebuilt or
        overwritten silently.

    ID columns (``image_id``/``*_id``/``id``) are read back as strings.
    """
    path = Path(path)
    hash_file = hash_path(path)
    record_file = record_path(path)
    csv_exists = path.exists()
    hash_exists = hash_file.exists()

    if csv_exists and hash_exists:
        expected = hash_file.read_text(encoding="utf-8").split()[0]
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(
                f"Frozen manifest hash mismatch for {path}: recorded {expected}, "
                f"file is {actual}. Refusing to rebuild or overwrite a frozen manifest."
            )
        return pd.read_csv(path, dtype=_id_dtypes(path))

    if csv_exists or hash_exists or record_file.exists():
        present = [str(p.name) for p in (path, hash_file, record_file) if p.exists()]
        raise RuntimeError(
            f"Inconsistent frozen manifest state for {path}: present {present}. "
            "A frozen manifest needs the csv, its .sha256 and its .record.json "
            "together; refusing to rebuild silently."
        )

    frame = build_fn()
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    digest = sha256_file(path)
    hash_file.write_text(digest + "\n", encoding="utf-8")
    full_record = dict(record)
    full_record.update(_environment_record())
    full_record["csv_sha256"] = digest
    full_record["csv"] = _display_path(path)
    record_file.write_text(
        json.dumps(full_record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return frame

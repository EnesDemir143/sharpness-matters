"""Per-run CV record for the Stage-2 (B) baseline (contract 5).

B (§5.5, §5.7, §5.8): current -> the author's trainer only logs mean `val/loss`/`val/acc`
and saves `last-checkpoint`; problem -> the reported score cannot be traced to a
checkpoint, a fold or an image; new behaviour -> every fold records its best epoch,
checkpoint path and SHA-256, the run keeps the pooled OOF probabilities by ImageId, and
a reader verifies the recorded hashes before using a checkpoint.
Check: tests/corrected/test_train.py::test_cv_record_*.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Union

import pandas as pd

from sharpness_matters.config.load_config import cfg

CV_RECORD_NAME = "cv_record.json"
OOF_CSV_NAME = "oof_predictions.csv"
OOF_COLUMNS = ["image_id", "label", "fold", "prob"]
# B (§4.1 / PTX-16): version 1 = the trainer proved the in-memory fold frame equals
# the hashed manifest file before writing this record. Older records hash a file but
# may have trained on a different frame, so their hash alone is not provenance.
FOLD_MANIFEST_PROVENANCE_VERSION = 1
FOLD_KEYS = [
    "fold",
    "best_epoch",
    "stopped_epoch",
    "checkpoint_path",
    "sha256",
    "val_auroc",
    "n_train",
    "n_val",
    "pos_weight",
]
RECORD_KEYS = [
    "model_name",
    "img_size",
    "k_folds",
    "fold_manifest_sha256",
    "fold_manifest_provenance_version",
    "folds",
    "S_h",
    "pooled_oof_auroc",
    "tau",
    "target_sensitivity",
    "oof_csv",
]


def file_sha256(path: Union[str, Path]) -> str:
    """SHA-256 of a checkpoint or manifest file (contract 1, `frozen.sha256_file`).

    Imported lazily because `ptx/corrected/frozen.py` is owned by the data slice and
    this module must stay importable on its own.
    """
    from sharpness_matters.ptx.corrected.frozen import sha256_file

    return sha256_file(Path(path))


def run_dir(model_name: str, img_size: int) -> Path:
    """Directory that holds the CV record, the OOF csv and this run's checkpoints."""
    return (
        Path(cfg.ptx.model.checkpoint_dir)
        / model_name
        / f"{model_name}_{img_size}_logs"
    )


def write_cv_record(record: Dict[str, Any], oof_df: pd.DataFrame) -> Path:
    """Write `cv_record.json` and `oof_predictions.csv` into `run_dir`; return json path."""
    # `oof_csv` is written by this function; every other key must come from the trainer.
    missing = [key for key in RECORD_KEYS if key != "oof_csv" and key not in record]
    if missing:
        raise ValueError(f"cv record is missing keys: {missing}")
    _check_folds(record, context="write")
    _check_oof(oof_df)
    _check_provenance(record, context="write")

    directory = run_dir(record["model_name"], record["img_size"])
    directory.mkdir(parents=True, exist_ok=True)

    record = dict(record)
    record["oof_csv"] = OOF_CSV_NAME
    oof_df.to_csv(directory / OOF_CSV_NAME, index=False)

    json_path = directory / CV_RECORD_NAME
    with open(json_path, "w") as handle:
        json.dump(record, handle, indent=2, default=_json_default)
    return json_path


def load_cv_record(model_name: str, img_size: int) -> Dict[str, Any]:
    """Read `cv_record.json` and verify fold coverage, checkpoint files and hashes.

    Raises:
        RuntimeError: record missing, a fold index outside 0..k_folds-1 missing or
            duplicated, a recorded checkpoint file absent, or a recorded SHA-256 that
            does not match the file on disk. A record is never silently repaired.
    """
    json_path = run_dir(model_name, img_size) / CV_RECORD_NAME
    if not json_path.is_file():
        raise RuntimeError(f"cv record not found: {json_path}")
    with open(json_path) as handle:
        record = json.load(handle)

    _check_folds(record, context=f"load ({json_path})")
    for entry in sorted(record["folds"], key=lambda e: int(e["fold"])):
        checkpoint = Path(entry["checkpoint_path"])
        if not checkpoint.is_file():
            raise RuntimeError(
                f"fold {entry['fold']}: checkpoint listed in the cv record is missing: "
                f"{checkpoint}"
            )
        actual = file_sha256(checkpoint)
        if actual != entry["sha256"]:
            raise RuntimeError(
                f"fold {entry['fold']}: sha256 mismatch for {checkpoint}: "
                f"recorded {entry['sha256']}, file has {actual}"
            )
    return record


def development_manifest_from_record(
    record: Dict[str, Any], manifest_path: Union[str, Path]
) -> pd.DataFrame:
    """Read the development manifest a run's checkpoints were trained on.

    The record's ``fold_manifest_sha256`` is the only accepted provenance: a record
    written before that binding existed (missing/empty value) or a manifest file whose
    bytes no longer match the recorded hash raises. Nothing is assumed or repaired, so
    a holdout check can rely on the returned frame being the exact list training used.
    """
    recorded = record.get("fold_manifest_sha256")
    if not isinstance(recorded, str) or not recorded.strip():
        raise RuntimeError(
            "cv record has no fold_manifest_sha256; the development manifest actually "
            "used by these checkpoints cannot be established. Retrain with the current "
            "trainer (it binds the frozen fold manifest) before evaluating the holdout."
        )
    version = record.get("fold_manifest_provenance_version")
    if (
        not isinstance(version, int)
        or isinstance(version, bool)
        or version != FOLD_MANIFEST_PROVENANCE_VERSION
    ):
        raise RuntimeError(
            "cv record fold-manifest provenance is missing or outdated "
            f"(found {version!r}, need {FOLD_MANIFEST_PROVENANCE_VERSION}). Older "
            "trainers wrote a manifest hash without proving the frame trained on was "
            "that file, so the hash alone cannot establish the development split. "
            "Retrain with the current trainer before evaluating the holdout."
        )
    path = Path(manifest_path)
    if not path.is_file():
        raise RuntimeError(
            f"development manifest bound by the cv record is missing: {path}"
        )
    actual = file_sha256(path)
    if actual != recorded.strip().lower():
        raise RuntimeError(
            f"development manifest provenance mismatch: cv record was trained on a "
            f"manifest with sha256 {recorded}, but {path} has {actual}. Refusing to "
            "treat this file as the split behind the checkpoints."
        )
    frame = pd.read_csv(path, dtype={"image_id": str})
    missing = {"image_id", "label", "fold"} - set(frame.columns)
    if missing:
        raise RuntimeError(
            f"development manifest {path} is missing columns {sorted(missing)}; "
            "expected 'image_id,label,fold'"
        )
    return frame


def fold_checkpoints(record: Dict[str, Any]) -> List[str]:
    """Checkpoint paths of a cv record, ordered by fold."""
    return [
        entry["checkpoint_path"]
        for entry in sorted(record["folds"], key=lambda e: int(e["fold"]))
    ]


def _check_folds(record: Dict[str, Any], context: str) -> None:
    folds = record.get("folds")
    if not isinstance(folds, list) or not folds:
        raise RuntimeError(f"cv record has no folds ({context})")
    k_folds = int(record["k_folds"])
    indices = []
    for entry in folds:
        missing = [key for key in FOLD_KEYS if key not in entry]
        if missing:
            raise RuntimeError(
                f"fold entry {entry.get('fold')} is missing keys {missing} ({context})"
            )
        indices.append(int(entry["fold"]))
    if sorted(indices) != list(range(k_folds)):
        raise RuntimeError(
            f"cv record fold indices {sorted(indices)} are not exactly 0..{k_folds - 1} "
            f"({context})"
        )


def _check_provenance(record: Dict[str, Any], context: str) -> None:
    """A new record must carry the 64-hex sha256 of the development manifest used."""
    value = record.get("fold_manifest_sha256")
    if not isinstance(value, str) or len(value.strip()) != 64:
        raise RuntimeError(
            f"cv record has no valid fold_manifest_sha256 ({context}); future records "
            "must bind the sha256 of the frozen fold manifest they trained on"
        )
    try:
        int(value.strip(), 16)
    except ValueError as error:
        raise RuntimeError(
            f"cv record fold_manifest_sha256 is not hexadecimal ({context}): {value!r}"
        ) from error
    version = record.get("fold_manifest_provenance_version")
    if (
        not isinstance(version, int)
        or isinstance(version, bool)
        or version != FOLD_MANIFEST_PROVENANCE_VERSION
    ):
        raise RuntimeError(
            "cv record fold_manifest_provenance_version must be "
            f"{FOLD_MANIFEST_PROVENANCE_VERSION} ({context}), got {version!r}; only "
            "the trainer that verified the manifest frame against the hashed file "
            "may write a record"
        )


def _check_oof(oof_df: pd.DataFrame) -> None:
    if list(oof_df.columns) != OOF_COLUMNS:
        raise ValueError(
            f"OOF columns must be {OOF_COLUMNS}, got {list(oof_df.columns)}"
        )
    if oof_df["image_id"].astype(str).duplicated().any():
        raise ValueError("OOF predictions contain a duplicated image_id")
    if oof_df.empty:
        raise ValueError("OOF predictions are empty")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):  # numpy scalar
        return value.item()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")

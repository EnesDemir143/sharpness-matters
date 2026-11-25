"""Frozen-manifest verification shared by the B evaluation entry points.

# B (Section 4.7 / 6.1): evaluation may only read an artifact that was frozen before
#   the run. The author's evaluators read whatever path the caller passed, so a
#   renamed E5 manifest (E5-clean instead of reference-like, or the reverse) produced
#   a full metrics CSV under the other protocol row. New: every evaluation cohort is
#   loaded through `frozen.load_frozen` (csv + .sha256 + .record.json, hash checked)
#   and the loaded dataset is compared row-for-row with that frozen frame. The E5
#   mode additionally maps to exactly one canonical manifest: the supplied path must
#   have the canonical file's content (byte hash), and the canonical record must name
#   the mode's own artifact, so a renamed copy is refused instead of scored.
#   check: tests/corrected/test_manifest_eval_integrity.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from sharpness_matters.ptx.corrected.frozen import (
    load_frozen,
    manifest_dir,
    record_path,
    sha256_file,
)
from sharpness_matters.ptx.corrected.make_e5 import CLEAN_NAME, REFERENCE_COMBINED_NAME

# E5 mode -> (canonical file name, identity marker in the artifact's creation record).
# The two cohorts are different ID sets; scoring one under the other's label is the
# error this table prevents.
E5_MANIFESTS: Dict[str, Tuple[str, str]] = {
    "ood_reference_like": (REFERENCE_COMBINED_NAME, "E5 reference-like"),
    "ood_clean": (CLEAN_NAME, "E5-clean"),
}


def load_e5_manifest(
    validation_mode: str,
    e5_manifest: Optional[Any] = None,
    manifest_directory: Optional[Any] = None,
) -> Tuple[Path, pd.DataFrame]:
    """Return the verified canonical E5 manifest path and frame for ``validation_mode``.

    ``validation_mode`` must be ``ood_reference_like`` or ``ood_clean``. The canonical
    manifest for that mode is loaded and hash-checked first. A supplied ``e5_manifest``
    is accepted only when it is frozen itself and byte-identical to the canonical file
    for the requested mode, so the other mode's manifest (or a renamed copy of it) is
    rejected before any dataset or checkpoint is touched.

    Args:
        validation_mode: ``ood_reference_like`` or ``ood_clean``.
        e5_manifest: optional path to the frozen manifest named on the command line.
        manifest_directory: directory holding the canonical artifacts; defaults to
            ``frozen.manifest_dir()`` (evaluation passes its own resolved directory so
            a test/entry-point override applies here too).

    Returns:
        ``(canonical_path, canonical_frame)``; the path is the artifact whose content
        was verified and is what the loader should read.
    """
    try:
        canonical_name, identity = E5_MANIFESTS[validation_mode]
    except KeyError as exc:
        raise ValueError(
            f"{validation_mode!r} is not an E5 mode; expected one of "
            f"{sorted(E5_MANIFESTS)}"
        ) from exc
    root = (
        Path(manifest_directory) if manifest_directory is not None else manifest_dir()
    )
    canonical_path = root / canonical_name
    canonical_frame = load_frozen(canonical_path)

    missing_columns = {"image_id", "label"} - set(canonical_frame.columns)
    if missing_columns:
        raise KeyError(
            f"{canonical_path} is missing columns {sorted(missing_columns)}; an E5 "
            "manifest must carry image_id,label"
        )

    if e5_manifest is not None:
        supplied_path = Path(e5_manifest)
        if not supplied_path.is_file():
            raise RuntimeError(
                f"E5 manifest {supplied_path} does not exist; {validation_mode} needs "
                f"the frozen {canonical_name}"
            )
        if supplied_path.resolve() != canonical_path.resolve():
            supplied_frame = load_frozen(supplied_path)
            supplied_sha = sha256_file(supplied_path)
            canonical_sha = sha256_file(canonical_path)
            if supplied_sha != canonical_sha:
                raise RuntimeError(
                    f"E5 manifest {supplied_path} is not the frozen artifact for "
                    f"{validation_mode}: its content differs from {canonical_path}. "
                    "Reference-like and E5-clean are different ID sets; pass the "
                    "matching manifests/reference_like_combined.csv or "
                    "manifests/reference_like_clean.csv."
                )
            if set(supplied_frame.columns) != set(canonical_frame.columns):
                raise RuntimeError(
                    f"E5 manifest {supplied_path} has different columns from "
                    f"{canonical_path}"
                )

    record = json.loads(record_path(canonical_path).read_text(encoding="utf-8"))
    recorded_item = str(record.get("item", ""))
    if identity not in recorded_item:
        raise RuntimeError(
            f"Frozen manifest {canonical_path} records item {recorded_item!r}, which "
            f"is not {identity!r}. {validation_mode} cannot be scored from another "
            "mode's cohort; restore the canonical frozen artifact."
        )
    return canonical_path, canonical_frame


def verify_dataset_against_frozen(
    dataset: Any,
    frame: pd.DataFrame,
    *,
    what: str,
    id_attribute: str = "image_ids",
    labels_attribute: str = "labels",
) -> None:
    """Fail unless ``dataset`` exposes exactly the frozen frame's ids, order and labels.

    Evaluation datasets re-read the manifest path themselves, so this is the second
    half of the hash check: the rows the model will actually see must be the rows that
    were verified. A reordered, shortened or relabelled cohort raises instead of
    producing a prediction CSV under the frozen manifest's name.
    """
    dataset_ids = [str(value) for value in getattr(dataset, id_attribute)]
    frozen_ids = [str(value) for value in frame["image_id"]]
    if dataset_ids != frozen_ids:
        raise RuntimeError(
            f"{what}: loader order differs from the frozen manifest "
            f"({len(dataset_ids)} dataset ids vs {len(frozen_ids)} manifest ids)"
        )
    dataset_labels = getattr(dataset, labels_attribute)
    frozen_labels = {
        str(image_id): int(label)
        for image_id, label in zip(frame["image_id"], frame["label"])
    }
    mismatched = [
        image_id
        for image_id in dataset_ids
        if int(dataset_labels[image_id]) != frozen_labels[image_id]
    ]
    if mismatched:
        raise RuntimeError(
            f"{what}: loader labels differ from the frozen manifest for "
            f"{len(mismatched)} ids, e.g. {mismatched[:5]}"
        )

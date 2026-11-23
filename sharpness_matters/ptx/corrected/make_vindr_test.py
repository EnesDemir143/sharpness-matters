"""E2 VinDr-CXR v1.0.0 official-test manifest (Section 4.4, row E2).

# B (§4.4 E2): the author's only VinDr cohort is the `.txt` ID selection of E5, with no
#   source, split or label file of its own; the official test split was never used and
#   its consensus labels were never tied to the image files. New (E2): one row per
#   image of the PhysioNet VinDr-CXR v1.0.0 test split, `label` = the 5-radiologist
#   consensus `Pneumothorax` column of `image_labels_test.csv`; duplicate image ids,
#   labels outside {0,1} and ids without `<test_root>/<image_id>.dicom` raise instead
#   of being dropped, and the frozen manifest records n, n_pos and n_neg. The Kaggle
#   competition test labels are hidden, so Kaggle is not a valid E2 source.
#   check: tests/corrected/test_e2_e4_eval.py::test_vindr_test_manifest_happy_path,
#   ::test_vindr_test_manifest_duplicate_image_id_raises,
#   ::test_vindr_test_manifest_non_binary_label_raises,
#   ::test_vindr_test_manifest_missing_dicom_raises
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Union

import click
import pandas as pd

from sharpness_matters.config.load_config import cfg
from sharpness_matters.ptx.corrected.frozen import (
    hash_path,
    load_or_create_frozen,
    manifest_dir,
    record_path,
)

MANIFEST_NAME = "vindr_test.csv"
LABEL_COLUMN = "Pneumothorax"


def build_vindr_test_manifest(
    labels_csv: Union[str, Path], test_root: Union[str, Path]
) -> pd.DataFrame:
    """Return ``image_id,label`` for the VinDr-CXR official test split, in file order.

    Rules (contract item 1):
      * ``labels_csv`` holds one consensus row per image; a duplicate ``image_id``
        raises (it would silently double-weight an image in every metric);
      * ``Pneumothorax`` must be binary 0/1; any other value raises;
      * every id must have ``<test_root>/<image_id>.dicom``; missing files raise with
        the list of ids, so a partially downloaded split is never frozen.

    The label is the 5-radiologist consensus (`image_labels_test.csv`), one row per
    image, no radiologist id column.
    """
    labels_csv = Path(labels_csv)
    test_root = Path(test_root)
    if not labels_csv.exists():
        raise FileNotFoundError(f"VinDr test label csv not found: {labels_csv}")
    frame = pd.read_csv(labels_csv, dtype={"image_id": str})
    missing_columns = {"image_id", LABEL_COLUMN} - set(frame.columns)
    if missing_columns:
        raise KeyError(f"{labels_csv} is missing columns {sorted(missing_columns)}")
    if frame.empty:
        raise ValueError(f"{labels_csv} has no rows; refusing to freeze an empty E2 split")

    image_ids = frame["image_id"].astype(str).tolist()
    duplicated = frame.loc[frame["image_id"].astype(str).duplicated(), "image_id"]
    if len(duplicated):
        raise ValueError(
            f"{labels_csv} labels image_id(s) more than once: "
            f"{sorted(set(duplicated.astype(str)))}; E2 needs exactly one consensus "
            "row per image"
        )

    raw_labels = frame[LABEL_COLUMN]
    numeric_labels = pd.to_numeric(raw_labels, errors="coerce")
    invalid = numeric_labels.isna() | ~numeric_labels.isin([0, 1])
    if bool(invalid.any()):
        offenders = [
            f"{image_id}={value}"
            for image_id, value in zip(
                frame.loc[invalid, "image_id"], raw_labels.loc[invalid]
            )
        ]
        raise ValueError(
            f"{labels_csv} has non-binary {LABEL_COLUMN} values: {offenders}; E2 "
            "labels must be 0/1 consensus"
        )
    labels = numeric_labels.astype(int).tolist()

    missing_files = [
        image_id
        for image_id in image_ids
        if not (test_root / f"{image_id}.dicom").exists()
    ]
    if missing_files:
        raise FileNotFoundError(
            f"{len(missing_files)} image(s) have no <test_root>/<image_id>.dicom under "
            f"{test_root}: {missing_files}"
        )

    return pd.DataFrame({"image_id": image_ids, "label": labels})


def manifest_counts(frame: pd.DataFrame) -> Dict[str, int]:
    """Image, positive and negative counts of an ``image_id,label`` manifest."""
    labels = frame["label"].astype(int)
    return {
        "n": int(len(frame)),
        "n_pos": int((labels == 1).sum()),
        "n_neg": int((labels == 0).sum()),
    }


def load_vindr_test_manifest() -> pd.DataFrame:
    """Load ``manifests/vindr_test.csv``; build and freeze it once if absent.

    Frozen (csv + .sha256 + .record.json) is loaded and hash-verified without touching
    the label csv or the DICOM root; a partial or mismatching frozen state raises
    instead of rebuilding (contract module 1, Section 4.7).
    """
    path = manifest_dir() / MANIFEST_NAME
    frozen_state = any(p.exists() for p in (path, hash_path(path), record_path(path)))

    if frozen_state:

        def _build():  # pragma: no cover - only reached for a broken frozen state
            raise RuntimeError(f"Frozen VinDr test manifest {path} is incomplete")

        record: Dict[str, object] = {}
        built = None
    else:
        labels_csv = Path(str(cfg.ptx.vindr.test_labels_csv))
        test_root = Path(str(cfg.ptx.vindr.test_root_dir))
        built = build_vindr_test_manifest(labels_csv, test_root)
        record = {
            "item": "E2 (Section 4.4) VinDr-CXR v1.0.0 official test manifest",
            "source": "PhysioNet VinDr-CXR v1.0.0 official test split",
            "labels_csv": str(labels_csv),
            "test_root": str(test_root),
            "label_column": LABEL_COLUMN,
            **manifest_counts(built),
        }

        def _build():
            return built

    frame = load_or_create_frozen(path, _build, record)
    frame["label"] = frame["label"].astype(int)
    return frame


@click.command()
def main() -> None:
    """Build-and-freeze (first run) or load-and-verify manifests/vindr_test.csv."""
    frame = load_vindr_test_manifest()
    click.echo(
        json.dumps(
            {"csv": str(manifest_dir() / MANIFEST_NAME), **manifest_counts(frame)},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

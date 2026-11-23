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
# B (§4.4 E2-wide, secondary row next to E2): current -> only the 3,000-image official
#   test split was turned into a frozen cohort, leaving the 45,000-row train label file
#   (3 readers per image) unused and the pre-registered vote rules unimplemented. New:
#   `manifests/vindr_wide.csv` covers train + test with `label` (train >= 2 of 3 reader
#   votes = MAIN rule; test = consensus) and `label_any` (train >= 1 vote = author's E5
#   positive definition; test = consensus), so one prediction set yields both the main
#   and the sensitivity metric. The record carries both rules, per-split/total counts
#   for both labels and the SHA-256 of both label CSVs. Neither split is ever frozen
#   partially: a reader count other than 3, an id in both splits, or a missing DICOM
#   raises.
#   check: tests/corrected/test_e2_e4_eval.py::test_vindr_wide_manifest_rules_and_counts,
#   ::test_vindr_wide_reader_count_raises,
#   ::test_vindr_wide_split_overlap_raises,
#   ::test_vindr_wide_missing_dicom_raises
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Sequence, Union

import click
import pandas as pd

from sharpness_matters.config.load_config import cfg
from sharpness_matters.ptx.corrected.frozen import (
    hash_path,
    load_or_create_frozen,
    manifest_dir,
    record_path,
    sha256_file,
)

MANIFEST_NAME = "vindr_test.csv"
WIDE_MANIFEST_NAME = "vindr_wide.csv"
LABEL_COLUMN = "Pneumothorax"

TEST_RULE = (
    "label = the 5-radiologist consensus Pneumothorax of image_labels_test.csv "
    "(one row per image, no rad_id)"
)
WIDE_MAIN_RULE = (
    "label: train = 1 if Pneumothorax reader votes >= 2 of 3 (pre-registered "
    "majority); test = the consensus label"
)
WIDE_SENSITIVITY_RULE = (
    "label_any: train = 1 if Pneumothorax reader votes >= 1 of 3 (pre-registered "
    "sensitivity rule = author's E5 positive definition); test = the consensus label"
)

TRAIN_READERS = 3
TEST_CONSENSUS_READERS = 5
MAIN_VOTE_THRESHOLD = 2
ANY_VOTE_THRESHOLD = 1
WIDE_SPLITS = ("train", "test")
WIDE_COLUMNS = [
    "image_id",
    "split",
    "label",
    "label_any",
    "n_readers",
    "n_pos_votes",
]
# Source-reported VinDr-CXR v1.0.0 counts; the frozen record keeps the measured counts
# and reports whether they match these (binding values are the measured ones).
EXPECTED_WIDE_COUNTS = {
    "total": {"n": 18000, "label_pos": 76, "label_any_pos": 114},
    "train": {"n": 15000, "label_pos": 58, "label_any_pos": 96},
    "test": {"n": 3000, "label_pos": 18, "label_any_pos": 18},
}


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

    _require_dicoms(image_ids, test_root, "test")

    return pd.DataFrame({"image_id": image_ids, "label": labels})


def _binary_column(frame: pd.DataFrame, csv_path: Path) -> pd.Series:
    """``frame[Pneumothorax]`` as int 0/1, or raise listing the offending rows."""
    raw = frame[LABEL_COLUMN]
    numeric = pd.to_numeric(raw, errors="coerce")
    invalid = numeric.isna() | ~numeric.isin([0, 1])
    if bool(invalid.any()):
        offenders = [
            f"{image_id}={value}"
            for image_id, value in zip(frame.loc[invalid, "image_id"], raw.loc[invalid])
        ]
        raise ValueError(
            f"{csv_path} has non-binary {LABEL_COLUMN} values: {offenders}; every "
            "row must be a 0/1 vote"
        )
    return numeric.astype(int)


def _require_dicoms(image_ids: Sequence[str], root: Path, split: str) -> None:
    """Raise unless every id has ``<root>/<image_id>.dicom`` (never freeze partially)."""
    missing = [
        str(image_id)
        for image_id in image_ids
        if not (Path(root) / f"{image_id}.dicom").exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} VinDr {split} image(s) have no <root>/<image_id>.dicom "
            f"under {root}: {missing}"
        )


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
            "labels_csv_sha256": sha256_file(labels_csv),
            "test_root": str(test_root),
            "label_column": LABEL_COLUMN,
            "rule": TEST_RULE,
            **manifest_counts(built),
        }

        def _build():
            return built

    frame = load_or_create_frozen(path, _build, record)
    frame["label"] = frame["label"].astype(int)
    return frame


# ------------------------------------------------------------------------- E2-wide


def _wide_train_rows(train_labels_csv: Path) -> pd.DataFrame:
    """Train rows: 3 reader votes per image -> majority `label` + any-vote `label_any`."""
    frame = pd.read_csv(train_labels_csv, dtype={"image_id": str, "rad_id": str})
    missing_columns = {"image_id", "rad_id", LABEL_COLUMN} - set(frame.columns)
    if missing_columns:
        raise KeyError(
            f"{train_labels_csv} is missing columns {sorted(missing_columns)}"
        )
    if frame.empty:
        raise ValueError(
            f"{train_labels_csv} has no rows; refusing to freeze an empty split"
        )
    votes = _binary_column(frame, train_labels_csv)
    grouped = frame.assign(_vote=votes).groupby("image_id", sort=False)["_vote"]
    n_readers = grouped.size()
    n_pos_votes = grouped.sum()
    wrong = n_readers[n_readers != TRAIN_READERS]
    if len(wrong):
        detail = [f"{image_id}: {int(count)}" for image_id, count in wrong.items()]
        raise ValueError(
            f"{train_labels_csv} needs exactly {TRAIN_READERS} reader rows (rad_id) per "
            f"image; got {detail}"
        )
    positives = n_pos_votes.to_numpy()
    return pd.DataFrame(
        {
            "image_id": [str(image_id) for image_id in n_readers.index],
            "split": "train",
            "label": (positives >= MAIN_VOTE_THRESHOLD).astype(int),
            "label_any": (positives >= ANY_VOTE_THRESHOLD).astype(int),
            "n_readers": n_readers.to_numpy().astype(int),
            "n_pos_votes": positives.astype(int),
        }
    )


def _wide_test_rows(test_labels_csv: Path) -> pd.DataFrame:
    """Test rows: consensus label, 5 consensus readers, votes = the consensus value."""
    frame = pd.read_csv(test_labels_csv, dtype={"image_id": str})
    missing_columns = {"image_id", LABEL_COLUMN} - set(frame.columns)
    if missing_columns:
        raise KeyError(f"{test_labels_csv} is missing columns {sorted(missing_columns)}")
    if frame.empty:
        raise ValueError(
            f"{test_labels_csv} has no rows; refusing to freeze an empty split"
        )
    duplicated = frame.loc[frame["image_id"].astype(str).duplicated(), "image_id"]
    if len(duplicated):
        raise ValueError(
            f"{test_labels_csv} labels image_id(s) more than once: "
            f"{sorted(set(duplicated.astype(str)))}; the test split has one consensus "
            "row per image"
        )
    consensus = _binary_column(frame, test_labels_csv).to_numpy().astype(int)
    return pd.DataFrame(
        {
            "image_id": frame["image_id"].astype(str).tolist(),
            "split": "test",
            "label": consensus,
            "label_any": consensus,
            "n_readers": TEST_CONSENSUS_READERS,
            "n_pos_votes": consensus,
        }
    )


def build_vindr_wide_manifest(
    train_labels_csv: Union[str, Path],
    test_labels_csv: Union[str, Path],
    train_root: Union[str, Path],
    test_root: Union[str, Path],
) -> pd.DataFrame:
    """Return the E2-wide manifest (train then test) with both pre-registered labels.

    ``label`` is the MAIN rule (train: >= 2 of 3 reader votes; test: consensus) and
    ``label_any`` the SENSITIVITY rule (train: >= 1 vote, the author's E5 positive
    definition; test: consensus). The splits must be disjoint and every id must have
    its DICOM under the matching root; nothing is dropped to make a count look right.
    """
    train_labels_csv = Path(train_labels_csv)
    test_labels_csv = Path(test_labels_csv)
    train_root = Path(train_root)
    test_root = Path(test_root)
    for path in (train_labels_csv, test_labels_csv):
        if not path.exists():
            raise FileNotFoundError(f"VinDr label csv not found: {path}")
    train = _wide_train_rows(train_labels_csv)
    test = _wide_test_rows(test_labels_csv)
    overlap = sorted(set(train["image_id"]) & set(test["image_id"]))
    if overlap:
        raise ValueError(
            f"VinDr train and test splits share {len(overlap)} image_id(s): {overlap}; "
            "the wide cohort needs disjoint splits"
        )
    _require_dicoms(train["image_id"].tolist(), train_root, "train")
    _require_dicoms(test["image_id"].tolist(), test_root, "test")
    return pd.concat([train, test], ignore_index=True)[WIDE_COLUMNS]


def wide_manifest_counts(frame: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    """Per-split and total image counts for both labels of a wide manifest."""
    counts: Dict[str, Dict[str, int]] = {}
    total = {"n": 0, "label_pos": 0, "label_any_pos": 0}
    for split in WIDE_SPLITS:
        part = frame.loc[frame["split"].astype(str) == split]
        split_counts = {
            "n": int(len(part)),
            "label_pos": int((part["label"].astype(int) == 1).sum()),
            "label_any_pos": int((part["label_any"].astype(int) == 1).sum()),
        }
        counts[split] = split_counts
        for key, value in split_counts.items():
            total[key] += value
    counts["total"] = total
    return counts


def load_vindr_wide_manifest() -> pd.DataFrame:
    """Load ``manifests/vindr_wide.csv``; build and freeze it once if absent.

    The train label file is the sibling of ``cfg.ptx.vindr.test_labels_csv``
    (``image_labels_train.csv``), so no extra config key is needed; its path is
    recorded together with both label CSVs' SHA-256.
    """
    path = manifest_dir() / WIDE_MANIFEST_NAME
    frozen_state = any(p.exists() for p in (path, hash_path(path), record_path(path)))

    if frozen_state:

        def _build():  # pragma: no cover - only reached for a broken frozen state
            raise RuntimeError(f"Frozen VinDr wide manifest {path} is incomplete")

        record: Dict[str, object] = {}
        built = None
    else:
        test_labels_csv = Path(str(cfg.ptx.vindr.test_labels_csv))
        train_labels_csv = test_labels_csv.with_name("image_labels_train.csv")
        train_root = Path(str(cfg.ptx.vindr.root_dir))
        test_root = Path(str(cfg.ptx.vindr.test_root_dir))
        built = build_vindr_wide_manifest(
            train_labels_csv, test_labels_csv, train_root, test_root
        )
        counts = wide_manifest_counts(built)
        record = {
            "item": "E2-wide (Section 4.4 E2) VinDr-CXR v1.0.0 train+test manifest",
            "source": "PhysioNet VinDr-CXR v1.0.0 official train and test splits",
            "train_labels_csv": str(train_labels_csv),
            "train_labels_csv_sha256": sha256_file(train_labels_csv),
            "test_labels_csv": str(test_labels_csv),
            "test_labels_csv_sha256": sha256_file(test_labels_csv),
            "train_root": str(train_root),
            "test_root": str(test_root),
            "label_column": LABEL_COLUMN,
            "main_rule": WIDE_MAIN_RULE,
            "sensitivity_rule": WIDE_SENSITIVITY_RULE,
            "counts": counts,
            "expected_counts": EXPECTED_WIDE_COUNTS,
            "counts_match_expected": counts == EXPECTED_WIDE_COUNTS,
        }

        def _build():
            return built

    frame = load_or_create_frozen(path, _build, record)
    for column in ("label", "label_any", "n_readers", "n_pos_votes"):
        frame[column] = frame[column].astype(int)
    return frame


@click.command()
@click.option(
    "--wide",
    is_flag=True,
    help="Build/load the E2-wide train+test manifest (manifests/vindr_wide.csv)",
)
def main(wide: bool) -> None:
    """Build-and-freeze (first run) or load-and-verify an E2 manifest.

    Default: manifests/vindr_test.csv (official test split, row E2). With ``--wide``:
    manifests/vindr_wide.csv (train + test, E2-wide main and sensitivity labels).
    """
    if wide:
        frame = load_vindr_wide_manifest()
        click.echo(
            json.dumps(
                {
                    "csv": str(manifest_dir() / WIDE_MANIFEST_NAME),
                    "counts": wide_manifest_counts(frame),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
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

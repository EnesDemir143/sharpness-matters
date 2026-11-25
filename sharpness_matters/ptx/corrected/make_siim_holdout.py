"""Frozen SIIM internal holdout, verified against the development manifest.

# B (Section 4.1 / 6.x): the author's internal test split was re-created at every
#   evaluation from ``train-rle.csv`` + ``train_test_split``. Its membership therefore
#   depended on CSV row order and on the drifts the split never recorded, and nothing
#   proved the evaluated images were disjoint from the development images behind the
#   checkpoints. New behaviour: ``manifests/siim_holdout.csv`` (image_id,label) is
#   frozen once with ``.sha256`` and ``.record.json``; its IDs are the complement of
#   the frozen development manifest (``manifests/siim_folds.csv``) inside
#   ``train-rle.csv``. Evaluation loads the frozen file by membership, and refuses to
#   run if the holdout and the development manifest actually used by the cv record
#   overlap, if provenance is absent or mismatched, or if the holdout is not exactly
#   the complement of that development list.
#   check: tests/corrected/test_siim_holdout.py

Usage:
    python -m sharpness_matters.ptx.corrected.make_siim_holdout
    python -m sharpness_matters.ptx.corrected.make_siim_holdout --root-dir /data/siim-acr
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import click
import numpy as np
import pandas as pd

from sharpness_matters.config.load_config import cfg
from sharpness_matters.ptx.corrected.cv_record import development_manifest_from_record
from sharpness_matters.ptx.corrected.frozen import (
    hash_path,
    load_frozen,
    load_or_create_frozen,
    manifest_dir,
    record_path,
    repo_root,
    sha256_file,
)
from sharpness_matters.ptx.corrected.make_fold_manifest import FOLD_MANIFEST_NAME

HOLDOUT_MANIFEST_NAME = "siim_holdout.csv"
TEST_SIZE = 0.1
SPLIT_SEED = 42


def holdout_manifest_path() -> Path:
    """Frozen internal-holdout manifest location."""
    return Path(manifest_dir()) / HOLDOUT_MANIFEST_NAME


def _display_path(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(repo_root()))
    except ValueError:
        return str(path)


def _read_label_csv(label_csv: Union[str, Path]) -> Tuple[List[str], Dict[str, int]]:
    """Unique ImageIds in CSV order and their RLE-rule labels (order-independent map)."""
    from sharpness_matters.ptx.datasets.siim_acr_classification_dataset import (
        siim_labels_from_rle,
    )

    frame = pd.read_csv(label_csv)
    missing = [c for c in ("ImageId", " EncodedPixels") if c not in frame.columns]
    if missing:
        raise RuntimeError(
            f"{label_csv} is missing columns {missing}; expected the SIIM "
            "'ImageId, EncodedPixels' layout"
        )
    image_ids = [str(value) for value in frame["ImageId"].unique()]
    return image_ids, siim_labels_from_rle(frame)


def _manifest_labels(frame: pd.DataFrame, source: str) -> Dict[str, int]:
    """Validate an ``image_id,label`` frame and return its unique ID -> label map."""
    missing = [c for c in ("image_id", "label") if c not in frame.columns]
    if missing:
        raise RuntimeError(
            f"{source} is missing columns {missing}; expected 'image_id,label'"
        )
    image_ids = [str(value) for value in frame["image_id"].tolist()]
    duplicated = sorted(set(pd.Series(image_ids)[pd.Series(image_ids).duplicated()]))
    if duplicated:
        raise RuntimeError(f"{source} lists an image_id more than once: {duplicated[:5]}")
    labels: Dict[str, int] = {}
    invalid = []
    for image_id, value in zip(image_ids, frame["label"].tolist()):
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = float("nan")
        if numeric not in (0.0, 1.0):
            invalid.append((image_id, value))
        else:
            labels[image_id] = int(numeric)
    if invalid:
        raise RuntimeError(f"{source} has non-binary labels, e.g. {invalid[:5]}")
    return labels


def build_holdout_frame(
    image_ids: List[str],
    labels: Dict[str, int],
    development_ids: List[str],
) -> pd.DataFrame:
    """Complement of the development IDs inside the labelled cohort, sorted by ID.

    Pure set arithmetic on the ID+label inputs, so the frozen membership does not
    depend on the row order of ``train-rle.csv`` or of the development manifest.
    """
    all_ids = [str(value) for value in image_ids]
    if len(set(all_ids)) != len(all_ids):
        raise RuntimeError("labelled cohort contains a duplicated ImageId")
    dev_ids = [str(value) for value in development_ids]
    if len(set(dev_ids)) != len(dev_ids):
        raise RuntimeError("development manifest contains a duplicated image_id")
    unknown = sorted(set(dev_ids) - set(all_ids))
    if unknown:
        raise RuntimeError(
            f"{len(unknown)} development image_id(s) have no label row, e.g. {unknown[:5]}"
        )
    holdout_ids = sorted(set(all_ids) - set(dev_ids))
    if not holdout_ids:
        raise RuntimeError(
            "the development manifest covers every labelled image; the holdout is empty"
        )
    frame = pd.DataFrame(
        {"image_id": holdout_ids, "label": [int(labels[i]) for i in holdout_ids]}
    )
    return frame


def _author_holdout_ids(image_ids: List[str], labels: List[int]) -> List[str]:
    """Reproduce the author's internal test split from the same CSV order."""
    from sklearn.model_selection import train_test_split

    indices = np.arange(len(image_ids))
    _, test_idx = train_test_split(
        indices,
        stratify=labels,
        test_size=TEST_SIZE,
        random_state=SPLIT_SEED,
    )
    return [image_ids[int(index)] for index in test_idx]


def verify_holdout_membership(
    holdout: pd.DataFrame,
    development: pd.DataFrame,
    label_csv: Union[str, Path],
) -> Dict[str, Any]:
    """Prove the holdout is exactly the labelled cohort minus the development manifest.

    Checks, in order: duplicated IDs, any overlap with the development IDs (the
    fail-closed condition for checkpoints trained on that list), missing or extra
    holdout IDs, and label agreement of both manifests with the RLE rule. Raises
    ``RuntimeError`` on the first violation; returns counts otherwise.
    """
    all_ids, all_labels = _read_label_csv(label_csv)
    holdout_labels = _manifest_labels(holdout, "SIIM holdout manifest")
    development_labels = _manifest_labels(development, "development manifest")
    holdout_ids = list(holdout_labels)
    development_ids = list(development_labels)

    overlap = sorted(set(holdout_ids) & set(development_ids))
    if overlap:
        raise RuntimeError(
            f"SIIM holdout overlaps the development manifest on {len(overlap)} "
            f"ImageId(s), e.g. {overlap[:5]}; evaluation would leak training images"
        )

    expected = set(all_ids) - set(development_ids)
    missing = sorted(expected - set(holdout_ids))
    extra = sorted(set(holdout_ids) - expected)
    if missing or extra:
        raise RuntimeError(
            "SIIM holdout is not the complement of the development manifest: "
            f"{len(missing)} labelled image(s) are missing from the holdout, "
            f"{len(extra)} holdout image(s) are not in the labelled cohort or are "
            f"development images (e.g. missing {missing[:5]}, extra {extra[:5]})"
        )

    for name, frame_labels in (
        ("SIIM holdout manifest", holdout_labels),
        ("development manifest", development_labels),
    ):
        mismatched = sorted(
            image_id
            for image_id, label in frame_labels.items()
            if all_labels.get(image_id) != label
        )
        if mismatched:
            raise RuntimeError(
                f"{name} disagrees with the RLE labels for {len(mismatched)} "
                f"ImageId(s), e.g. {mismatched[:5]}"
            )

    positives = int(sum(holdout_labels.values()))
    return {
        "n_images": len(holdout_labels),
        "n_positive": positives,
        "n_negative": len(holdout_labels) - positives,
        "n_development": len(development_labels),
        "n_labelled": len(all_ids),
    }


def _read_frozen_record(path: Path) -> Dict[str, Any]:
    return json.loads(record_path(path).read_text(encoding="utf-8"))


def _load_holdout_artifact(
    path: Path,
    expected_dev_sha256: Optional[str] = None,
    expected_source: str = "the provided development manifest",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load the frozen holdout and enforce its creation-record bindings.

    Unlike generic frozen manifests, a holdout record must carry ``csv_sha256`` and
    ``dev_manifest_sha256``: a matching ``.sha256`` sidecar alone is not enough, and a
    holdout frozen against a different development manifest (even one with the same
    IDs but different folds) is rejected.
    """
    frame = load_frozen(path)
    frozen_record = _read_frozen_record(path)

    actual_csv = sha256_file(path)
    recorded_csv = frozen_record.get("csv_sha256")
    if not isinstance(recorded_csv, str) or not recorded_csv.strip():
        raise RuntimeError(
            f"frozen SIIM holdout record {record_path(path)} has no csv_sha256; the "
            "csv is not bound to its creation record"
        )
    if recorded_csv.strip().lower() != actual_csv:
        raise RuntimeError(
            f"frozen SIIM holdout record {record_path(path)} names csv_sha256 "
            f"{recorded_csv}, but the csv is {actual_csv}"
        )

    frozen_dev = frozen_record.get("dev_manifest_sha256")
    if not isinstance(frozen_dev, str) or not frozen_dev.strip():
        raise RuntimeError(
            f"frozen SIIM holdout record {record_path(path)} has no "
            "dev_manifest_sha256; it cannot be tied to a development split"
        )
    if expected_dev_sha256 is not None:
        if frozen_dev.strip().lower() != str(expected_dev_sha256).strip().lower():
            raise RuntimeError(
                "SIIM holdout provenance mismatch: it was frozen against development "
                f"manifest sha256 {frozen_dev}, but {expected_source} is "
                f"{expected_dev_sha256}. Even with identical image IDs, a different "
                "fold manifest is not the holdout's source."
            )
    return frame, frozen_record


def _prepare_frozen_record(
    holdout: pd.DataFrame,
    label_csv: Union[str, Path],
    development_path: Path,
    development_sha256: str,
    info: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "item": "SIIM internal holdout (author test split), frozen by ImageId+label",
        "derivation": (
            "complement of manifests/siim_folds.csv inside train-rle.csv; "
            "author train_test_split reproduced at build"
        ),
        "order": "image_id ascending",
        "dev_manifest": _display_path(development_path),
        "dev_manifest_sha256": development_sha256,
        "label_csv": _display_path(Path(label_csv)),
        "label_csv_sha256": sha256_file(Path(label_csv)),
        "test_size": TEST_SIZE,
        "seed": SPLIT_SEED,
        "n_images": int(len(holdout)),
        "n_positive": int(holdout["label"].sum()),
        "n_negative": int(len(holdout) - holdout["label"].sum()),
        "n_development": info["n_development"],
        "n_labelled": info["n_labelled"],
    }


def prepare_holdout(
    label_csv: Union[str, Path],
    development_manifest: Union[str, Path],
    out_path: Optional[Union[str, Path]] = None,
    verify_author_split: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load-and-verify the frozen holdout, or build and freeze it once.

    A frozen holdout is never overwritten: any partial state or hash mismatch raises
    through :func:`load_frozen`. On first creation, membership is the development
    complement, the author's ``train_test_split`` result is re-derived as a
    cross-check, and the development manifest is required to be frozen (csv, hash and
    record together) before the holdout record binds its SHA-256.
    """
    path = Path(out_path) if out_path is not None else holdout_manifest_path()
    development_path = Path(development_manifest)
    frozen_state = any(
        item.exists() for item in (path, hash_path(path), record_path(path))
    )
    if frozen_state:
        development = load_frozen(development_path)
        development_sha256 = sha256_file(development_path)
        holdout, _ = _load_holdout_artifact(
            path,
            expected_dev_sha256=development_sha256,
            expected_source="the provided development manifest",
        )
        info = verify_holdout_membership(holdout, development, label_csv)
        return holdout, info

    development = load_frozen(development_path)
    development_sha256 = sha256_file(development_path)
    image_ids, labels = _read_label_csv(label_csv)
    development_labels = _manifest_labels(development, "development manifest")
    holdout = build_holdout_frame(image_ids, labels, list(development_labels))
    info = verify_holdout_membership(holdout, development, label_csv)

    if verify_author_split:
        reproduced = _author_holdout_ids(
            image_ids, [int(labels[image_id]) for image_id in image_ids]
        )
        if set(reproduced) != set(holdout["image_id"]):
            raise RuntimeError(
                "the local train-rle.csv no longer reproduces the author's internal "
                "test split; refusing to freeze a holdout from ambiguous data"
            )

    record = _prepare_frozen_record(
        holdout, label_csv, development_path, development_sha256, info
    )
    frame = load_or_create_frozen(path, lambda: holdout, record)
    return frame, info


def load_and_verify_holdout(
    record: Dict[str, Any],
    label_csv: Union[str, Path],
    holdout_path: Optional[Union[str, Path]] = None,
    development_manifest: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Load the frozen holdout and fail closed against the checkpoint's dev manifest.

    Reads only frozen artifacts (never rebuilds), requires the cv record to carry
    ``fold_manifest_sha256``, requires that hash to match the development manifest
    file, and requires the holdout record's ``dev_manifest_sha256`` to be the same
    hash. Only then is the complement/overlap/label verification run.
    """
    path = Path(holdout_path) if holdout_path is not None else holdout_manifest_path()
    development_path = (
        Path(development_manifest)
        if development_manifest is not None
        else Path(manifest_dir()) / FOLD_MANIFEST_NAME
    )
    development = development_manifest_from_record(record, development_path)
    recorded_dev_sha = str(record["fold_manifest_sha256"]).strip().lower()
    holdout, _ = _load_holdout_artifact(
        path,
        expected_dev_sha256=recorded_dev_sha,
        expected_source="the cv record's checkpoints",
    )

    verify_holdout_membership(holdout, development, label_csv)
    return holdout


@click.command()
@click.option(
    "--root-dir",
    type=click.Path(file_okay=False),
    default=None,
    help="SIIM-ACR root (default: ptx.siim_acr.root_dir); must hold train-rle.csv",
)
@click.option(
    "--label-csv",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="train-rle.csv path (default: <root-dir>/train-rle.csv)",
)
@click.option(
    "--dev-manifest",
    type=click.Path(dir_okay=False),
    default=None,
    help="frozen development manifest (default: manifests/siim_folds.csv)",
)
@click.option(
    "--manifest-dir",
    "manifest_directory",
    type=click.Path(file_okay=False),
    default=None,
    help="directory that holds siim_holdout.csv (default: configured manifests/)",
)
def main(
    root_dir: Optional[str],
    label_csv: Optional[str],
    dev_manifest: Optional[str],
    manifest_directory: Optional[str],
) -> None:
    """Freeze (first run) or verify (later runs) manifests/siim_holdout.csv."""
    out_dir = (
        Path(manifest_directory) if manifest_directory is not None else manifest_dir()
    )
    root = Path(root_dir) if root_dir is not None else Path(str(cfg.ptx.siim_acr.root_dir))
    label_path = Path(label_csv) if label_csv is not None else root / "train-rle.csv"
    development_path = (
        Path(dev_manifest) if dev_manifest is not None else Path(manifest_dir()) / FOLD_MANIFEST_NAME
    )
    if not label_path.is_file():
        raise click.ClickException(f"label CSV not found: {label_path}")
    frame, info = prepare_holdout(
        label_path, development_path, out_path=out_dir / HOLDOUT_MANIFEST_NAME
    )
    print(
        f"{HOLDOUT_MANIFEST_NAME}: {info['n_images']} images "
        f"({info['n_positive']} positive, {info['n_negative']} negative); "
        f"development {info['n_development']} / labelled {info['n_labelled']}; "
        f"sha256={sha256_file(out_dir / HOLDOUT_MANIFEST_NAME)}"
    )
    print(f"frozen frame: {len(frame)} rows")


if __name__ == "__main__":
    main()

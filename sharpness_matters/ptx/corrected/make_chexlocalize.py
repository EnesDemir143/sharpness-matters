"""Build the frozen CheXlocalize E4 manifests (classification and XAI cohorts).

# B (protocol Section 4.4 E4): CheXlocalize had no cohort definition. New:
#   `manifests/chexlocalize_cls.csv` keeps frontal views with a verified CheXpert
#   `Pneumothorax` 0/1 label (laterals and non-0/1 values are excluded and counted);
#   `manifests/chexlocalize_xai.csv` keeps the label-1 images whose decoded
#   Pneumothorax GT is non-empty and matches the image grid. val, test and combined
#   images/patients/classes/masks are counted in the frozen record.
# B (protocol Section 4.7): both manifests are freeze-once via
#   `load_or_create_frozen`; readers never rebuild from a partial data directory.
#   check: tests/corrected/test_chexlocalize.py::test_*

Usage:
    python -m sharpness_matters.ptx.corrected.make_chexlocalize
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
from PIL import Image

from sharpness_matters.config.load_config import cfg
from sharpness_matters.ptx.corrected.frozen import (
    hash_path,
    load_or_create_frozen,
    manifest_dir,
    record_path,
    sha256_file,
)
from sharpness_matters.ptx.datasets.chexlocalize_dataset import (
    FINDING,
    SEGMENTATION_FILES,
    cxr_id_from_path,
    patient_id_from_path,
    rle_foreground_pixels,
    view_from_path,
)

CLS_MANIFEST_NAME = "chexlocalize_cls.csv"
XAI_MANIFEST_NAME = "chexlocalize_xai.csv"
MANIFEST_COLUMNS = ["image_id", "split", "patient_id", "path", "label"]
SPLITS = ("val", "test")
LABEL_COLUMN = FINDING

_SCALAR_COUNT_KEYS = (
    "rows_seen",
    "laterals_excluded",
    "uncertain_excluded",
    "positives_without_mask",
    "masks_without_positive_label",
    "nonempty_masks",
)


def _normalise_column(name: Any) -> str:
    return str(name).replace("\ufeff", "").strip().lower().replace(" ", "")


def find_column(columns: Sequence[Any], wanted: str) -> str:
    """Find one CSV column by exact/normalised name; ambiguity raises."""
    for column in columns:
        if str(column) == wanted:
            return str(column)
    matches = [
        str(column) for column in columns if _normalise_column(column) == _normalise_column(wanted)
    ]
    if len(matches) != 1:
        raise KeyError(
            f"Expected exactly one {wanted!r} column in {list(columns)}, "
            f"found {matches}"
        )
    return matches[0]


def find_pneumothorax_column(columns: Sequence[Any]) -> str:
    """The CheXpert label column by name (never by position)."""
    return find_column(columns, LABEL_COLUMN)


def _label_value(value: Any) -> Optional[int]:
    """0/1 label, or None for an excluded/uncertain value (NaN, -1, blank)."""
    if pd.isna(value):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric == 1.0:
        return 1
    if numeric == 0.0:
        return 0
    return None


def _candidate_image_paths(raw_path: Any, split: str, chexpert_dir: Path) -> List[Path]:
    """Possible on-disk locations of a label-CSV ``Path`` value, best first.

    CheXpert label CSVs use ``CheXpert/v1.0/valid/patient/study/view.jpg`` while the
    CheXlocalize download stores ``<chexpert_dir>/{val,test}/patient/study/view.jpg``;
    the last three components are the same in both layouts.
    """
    posix = str(raw_path).replace("\\", "/").strip()
    parts = [part for part in posix.split("/") if part]
    candidates: List[Path] = []
    if len(parts) >= 3:
        candidates.append(chexpert_dir / split / Path(*parts[-3:]))
        if split == "val":
            # CheXpert's official directory is `valid/`; the CheXlocalize download
            # may keep it while the config/split name is `val`.
            candidates.append(chexpert_dir / "valid" / Path(*parts[-3:]))
    direct = Path(posix)
    if direct.is_absolute():
        candidates.append(direct)
    else:
        candidates.append(chexpert_dir / posix)
        candidates.append(chexpert_dir / posix.replace("/valid/", "/val/"))
    unique: List[Path] = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return unique


def _resolve_image(raw_path: Any, split: str, chexpert_dir: Path) -> Optional[Path]:
    for candidate in _candidate_image_paths(raw_path, split, chexpert_dir):
        if candidate.is_file():
            return candidate
    return None


def _image_size(path: Path) -> Tuple[int, int]:
    """Native ``(height, width)`` of an image file (numpy/mask order)."""
    with Image.open(path) as image:
        width, height = image.size
    return int(height), int(width)


def _relative_path(path: Path, chexpert_dir: Path) -> str:
    try:
        return path.relative_to(chexpert_dir).as_posix()
    except ValueError:
        return str(path)


def _empty_counts() -> Dict[str, int]:
    return {
        "rows_seen": 0,
        "images": 0,
        "patients": 0,
        "positives": 0,
        "negatives": 0,
        "laterals_excluded": 0,
        "uncertain_excluded": 0,
        "positives_without_mask": 0,
        "masks_without_positive_label": 0,
        "nonempty_masks": 0,
        "xai_images": 0,
        "xai_patients": 0,
        "nonempty_masks_outside_xai": 0,
    }


def _load_segmentations(annotations_dir: Path, split: str) -> Dict[str, Any]:
    path = annotations_dir / SEGMENTATION_FILES[split]
    if not path.is_file():
        raise FileNotFoundError(f"CheXlocalize segmentations not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise RuntimeError(f"{path} must map image_id -> finding -> RLE")
    return data


def _build_split(
    labels_csv: Path,
    split: str,
    annotations_dir: Path,
    chexpert_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """One split: classification rows, XAI rows and the reported counts."""
    frame = pd.read_csv(labels_csv)
    path_column = find_column(frame.columns, "Path")
    label_column = find_pneumothorax_column(frame.columns)
    segmentations = _load_segmentations(annotations_dir, split)

    counts = _empty_counts()
    counts["labels_csv"] = str(labels_csv)
    counts["annotations_json"] = str(annotations_dir / SEGMENTATION_FILES[split])
    counts["label_column"] = label_column
    cls_rows: List[Dict[str, Any]] = []
    xai_rows: List[Dict[str, Any]] = []
    missing_images: List[Tuple[str, List[str]]] = []

    for raw_path, raw_label in zip(frame[path_column], frame[label_column]):
        counts["rows_seen"] += 1
        image_id = cxr_id_from_path(raw_path)
        view = view_from_path(raw_path)
        patient_id = patient_id_from_path(raw_path)
        label = _label_value(raw_label)
        entry = segmentations.get(image_id) or {}
        rle = entry.get(FINDING)
        foreground = int(rle_foreground_pixels(rle)) if rle is not None else 0
        if foreground:
            counts["nonempty_masks"] += 1

        if view == "lateral":
            counts["laterals_excluded"] += 1
            if foreground and label != 1:
                counts["masks_without_positive_label"] += 1
            continue
        if label is None:
            counts["uncertain_excluded"] += 1
            if foreground:
                counts["masks_without_positive_label"] += 1
            continue

        image_path = _resolve_image(raw_path, split, chexpert_dir)
        if image_path is None:
            missing_images.append((image_id, [str(p) for p in _candidate_image_paths(raw_path, split, chexpert_dir)]))
            continue
        image_shape = _image_size(image_path)
        row = {
            "image_id": image_id,
            "split": split,
            "patient_id": patient_id,
            "path": _relative_path(image_path, chexpert_dir),
            "label": int(label),
        }
        cls_rows.append(row)
        counts["images"] += 1
        if label == 1:
            counts["positives"] += 1
        else:
            counts["negatives"] += 1

        if foreground:
            rle_size = tuple(int(value) for value in rle["size"])
            if rle_size != image_shape:
                raise RuntimeError(
                    f"{image_id}: {FINDING} GT grid {rle_size} does not match the "
                    f"image grid {image_shape}; the XAI mask is never resized"
                )
            if label == 1:
                xai_rows.append(dict(row))
            else:
                counts["masks_without_positive_label"] += 1
        elif label == 1:
            counts["positives_without_mask"] += 1

    if missing_images:
        examples = "; ".join(
            f"{image_id} -> {candidates[0]}" for image_id, candidates in missing_images[:5]
        )
        raise RuntimeError(
            f"{len(missing_images)} labelled frontal CheXlocalize image(s) from "
            f"{labels_csv} are missing on disk, e.g. {examples}. Place the CheXpert "
            "images before freezing the E4 cohort."
        )

    counts["patients"] = len({row["patient_id"] for row in cls_rows})
    counts["xai_images"] = len(xai_rows)
    counts["xai_patients"] = len({row["patient_id"] for row in xai_rows})
    counts["nonempty_masks_outside_xai"] = counts["nonempty_masks"] - counts["xai_images"]

    cls_frame = pd.DataFrame(cls_rows, columns=MANIFEST_COLUMNS)
    xai_frame = pd.DataFrame(xai_rows, columns=MANIFEST_COLUMNS)
    return cls_frame, xai_frame, counts


def _combined_counts(
    per_split: Dict[str, Dict[str, Any]],
    cls_frame: pd.DataFrame,
    xai_frame: pd.DataFrame,
) -> Dict[str, Any]:
    combined: Dict[str, Any] = {
        key: int(sum(counts[key] for counts in per_split.values()))
        for key in _SCALAR_COUNT_KEYS
    }
    combined["images"] = int(len(cls_frame))
    combined["patients"] = int(cls_frame["patient_id"].nunique())
    combined["positives"] = int((cls_frame["label"] == 1).sum())
    combined["negatives"] = int((cls_frame["label"] == 0).sum())
    combined["xai_images"] = int(len(xai_frame))
    combined["xai_patients"] = int(xai_frame["patient_id"].nunique())
    combined["nonempty_masks_outside_xai"] = int(
        combined["nonempty_masks"] - combined["xai_images"]
    )
    return combined


def _require_unique(frame: pd.DataFrame, name: str) -> None:
    duplicated = frame["image_id"][frame["image_id"].duplicated()].tolist()
    if duplicated:
        raise RuntimeError(
            f"{name} lists duplicated image_id(s), e.g. {sorted(set(duplicated))[:5]}; "
            "CheXlocalize val and test must not share an image"
        )


def build_chexlocalize_manifests(
    chexpert_dir: Optional[str] = None,
    annotations_dir: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Build ``(cls, xai, counts)`` from ``{val,test}_labels.csv``.

    ``counts`` has one entry per split and a combined entry: images, patients,
    positives, negatives, laterals excluded, uncertain excluded, positives without
    mask, masks without a positive label, non-empty masks and XAI images/patients.
    Labels other than 0/1 (including NaN and -1) are excluded as uncertain and
    counted; lateral views are excluded and counted.
    """
    chexpert_root = Path(
        chexpert_dir
        if chexpert_dir is not None
        else cfg.ptx.chexlocalize.chexpert_dir
    )
    annotations_root = Path(
        annotations_dir
        if annotations_dir is not None
        else cfg.ptx.chexlocalize.annotations_dir
    )
    per_split: Dict[str, Dict[str, Any]] = {}
    cls_frames: List[pd.DataFrame] = []
    xai_frames: List[pd.DataFrame] = []
    for split in SPLITS:
        labels_csv = chexpert_root / f"{split}_labels.csv"
        if not labels_csv.is_file():
            raise FileNotFoundError(
                f"CheXlocalize label CSV not found: {labels_csv}"
            )
        cls_frame, xai_frame, counts = _build_split(
            labels_csv, split, annotations_root, chexpert_root
        )
        per_split[split] = counts
        cls_frames.append(cls_frame)
        xai_frames.append(xai_frame)

    cls_all = pd.concat(cls_frames, ignore_index=True)
    xai_all = pd.concat(xai_frames, ignore_index=True)
    _require_unique(cls_all, "CheXlocalize classification cohort")
    _require_unique(xai_all, "CheXlocalize XAI cohort")
    if len(cls_all) == 0:
        raise RuntimeError(
            "CheXlocalize classification cohort is empty; check the label column "
            f"{LABEL_COLUMN!r} and chexpert_dir={chexpert_root}"
        )
    counts = dict(per_split)
    counts["combined"] = _combined_counts(per_split, cls_all, xai_all)
    return cls_all, xai_all, counts


def _load_manifest(name: str, select: str) -> pd.DataFrame:
    """Load one frozen manifest first; build both once only when absent."""
    path = manifest_dir() / name
    frozen_state = any(
        candidate.exists() for candidate in (path, hash_path(path), record_path(path))
    )
    record: Dict[str, Any] = {}
    if frozen_state:
        def _build():  # pragma: no cover - only reached for a broken frozen state
            raise RuntimeError(f"Frozen CheXlocalize manifest {path} is incomplete")
    else:
        record = {
            "item": f"E4 CheXlocalize {select} manifest (protocol Section 4.4)",
            "rule": (
                "frontal views with CheXpert Pneumothorax in {0,1}"
                if select == "cls"
                else "frontal views with label 1 and a non-empty decoded "
                "Pneumothorax GT whose grid equals the image grid"
            ),
            "label_column": LABEL_COLUMN,
            "splits": list(SPLITS),
        }

        def _build():
            cls_frame, xai_frame, counts = build_chexlocalize_manifests()
            record["counts"] = counts
            record["source_sha256"] = _source_hashes()
            return cls_frame if select == "cls" else xai_frame

    frame = load_or_create_frozen(path, _build, record)
    frame["label"] = frame["label"].astype(int)
    return frame


def _source_hashes() -> Dict[str, str]:
    """SHA-256 of the label CSVs and segmentation JSONs in the build record."""
    chexpert_root = Path(cfg.ptx.chexlocalize.chexpert_dir)
    annotations_root = Path(cfg.ptx.chexlocalize.annotations_dir)
    sources: List[Path] = []
    for split in SPLITS:
        sources.append(chexpert_root / f"{split}_labels.csv")
        sources.append(annotations_root / SEGMENTATION_FILES[split])
    return {
        str(path): sha256_file(path) for path in sources if path.is_file()
    }


def load_chexlocalize_cls_manifest() -> pd.DataFrame:
    """Frozen ``manifests/chexlocalize_cls.csv`` (classification cohort)."""
    return _load_manifest(CLS_MANIFEST_NAME, "cls")


def load_chexlocalize_xai_manifest() -> pd.DataFrame:
    """Frozen ``manifests/chexlocalize_xai.csv`` (XAI cohort)."""
    return _load_manifest(XAI_MANIFEST_NAME, "xai")


def _record_counts(path: Path) -> Optional[Dict[str, Any]]:
    record_file = record_path(path)
    if not record_file.is_file():
        return None
    with record_file.open("r", encoding="utf-8") as handle:
        return json.load(handle).get("counts")


if __name__ == "__main__":
    cls_manifest = load_chexlocalize_cls_manifest()
    xai_manifest = load_chexlocalize_xai_manifest()
    missing_xai = sorted(set(xai_manifest["image_id"]) - set(cls_manifest["image_id"]))
    if missing_xai:
        raise RuntimeError(
            f"XAI cohort is not a subset of the classification cohort: {missing_xai[:5]}"
        )
    counts = _record_counts(manifest_dir() / CLS_MANIFEST_NAME)
    print(
        f"{CLS_MANIFEST_NAME}: {len(cls_manifest)} images, "
        f"{int(cls_manifest['label'].sum())} positives; "
        f"{XAI_MANIFEST_NAME}: {len(xai_manifest)} images"
    )
    if counts is not None:
        print(json.dumps(counts, indent=2, sort_keys=True))

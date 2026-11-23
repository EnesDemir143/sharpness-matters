"""Pre-run data checks (baseline_correction.md Section 9) — read-only.

# B (Section 9.4, approved 23 Nov 2026): the reference baseline assumed the datasets
#   were correct (PTX-498 mask value range, SIIM DICOM polarity, VinDr bit-depth
#   handling, SIIM ImageId -> file mapping, PTX-498 orientation) and stage A never
#   checked any of them. New: one script that inspects the data ONCE, after the
#   datasets are mounted and before stage A, and writes a dated report
#   (`reports/data_checks_<YYYY-MM-DD>.md` + `.json`).
#   The script only reads: it never writes into a dataset root, never moves or
#   renames data, never trains and never loads a model. Every check closes with
#   exactly one status:
#     "ISSUE -> B item <no>" | "NO CHANGE NEEDED (evidence: ...)" |
#     "OPEN (reason: ...)"   | "NOT RUN (path missing)"
#   An ambiguous result is OPEN, never "no change needed".
#   check: tests/corrected/test_data_checks.py

Checks and their B items (Section 9 rows):

* 9.2  PTX-498 mask unique values over ``Site*/<n>.5.mask.nii.gz`` -> B item 9.2
* 9.3a PhotometricInterpretation per dataset (MONOCHROME1 count + IDs) -> B item 9.3a
* 9.3b VinDr metadata distribution + unmodified vs ``BitsStored=16`` decode -> B 9.3b
* 9.4  SIIM ImageId -> file map, labelled-ID count and the author's 90/10 split
* GT-EMPTY positives whose GT mask has no foreground pixel (PTX-498 masks and
  SIIM positives) -> B item 7.4, because K1 saliency scoring raises on them
* 9.1  PTX-498 orientation vs SIIM overlays (human decision, always OPEN)

Known limit, recorded instead of hidden: with the installed decoders (pydicom
2.4.0, numpy handler and python-gdcm 3.0.25) ``BitsStored`` is not used when
decoding *uncompressed* pixel data, so forcing it cannot change those values. The
9.3b comparison still runs and reports pixel-level equality/difference; whether a
difference found on real files is meaningful stays a human decision (OPEN).
"""

from __future__ import annotations

import json
import math
import platform
import warnings
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import click
import nibabel as nib
import numpy as np
import pandas as pd
import pydicom
import sklearn
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from sharpness_matters.config.load_config import cfg
from sharpness_matters.ptx.corrected.frozen import repo_root

# --------------------------------------------------------------------------- paths

REPORT_PREFIX = "data_checks"
OVERLAY_SUFFIX = "_overlays"
SIIM_TRAIN_DIRNAME = "dicom-images-train"
SIIM_RLE_NAME = "train-rle.csv"
# The author globs ``dicom-images-train/*/*/*.dcm``; the flatter layouts are listed
# so a differently unpacked copy of the same dataset still resolves. The pattern
# that matched is recorded in the report, never swapped silently.
SIIM_DICOM_PATTERNS: Tuple[str, ...] = (
    f"{SIIM_TRAIN_DIRNAME}/*/*/*.dcm",
    f"{SIIM_TRAIN_DIRNAME}/*/*.dcm",
    f"{SIIM_TRAIN_DIRNAME}/*.dcm",
    "stage_2_train_images/*.dcm",
    "*.dcm",
)
VINDR_DICOM_PATTERNS: Tuple[str, ...] = ("*.dicom",)

# The author's SIIM split (siim_acr_classification_dataset.py): stratified 10%
# hold-out over ``label_df["ImageId"].unique()`` order with seed 42.
SIIM_TEST_SIZE = 0.1
SIIM_SPLIT_SEED = 42
# Protocol Section 4.7 / Section 9.4 table.
EXPECTED_SIIM: Dict[str, int] = {
    "labelled_ids": 10675,
    "dev_ids": 9607,
    "test_ids": 1068,
    "test_pos": 238,
    "test_neg": 830,
}

# ------------------------------------------------------------------------ statuses

ST_ISSUE = "ISSUE"
ST_NO_CHANGE = "NO CHANGE NEEDED"
ST_OPEN = "OPEN"
ST_NOT_RUN = "NOT RUN"
STATUS_CODES: Tuple[str, ...] = (ST_ISSUE, ST_NO_CHANGE, ST_OPEN, ST_NOT_RUN)

B_ITEM_CONSTANTS_NOTE = (
    "B items are the Section 9 rows of baseline_correction.md; '9.4-SIIM' is the "
    "SIIM ImageId -> file mapping row of the Section 9.4 table."
)


def status_issue(item: str) -> str:
    """``ISSUE -> B item <no>`` (the problem enters B as that item)."""
    return f"{ST_ISSUE} → B item {item}"


def status_no_change(evidence: str) -> str:
    """``NO CHANGE NEEDED (evidence: ...)`` — only with concrete evidence."""
    return f"{ST_NO_CHANGE} (evidence: {evidence})"


def status_open(reason: str) -> str:
    """``OPEN (reason)`` — ambiguous or undecidable results stay open."""
    return f"{ST_OPEN} ({reason})"


def status_not_run(reason: str = "path missing") -> str:
    """``NOT RUN (path missing)`` — a missing root is never reported as passing."""
    return f"{ST_NOT_RUN} ({reason})"


def status_code(status: str) -> str:
    """Status code of a status string (used for the report counts)."""
    for code in STATUS_CODES:
        if status.startswith(code):
            return code
    return "UNKNOWN"


# ------------------------------------------------------------------------- helpers


def _jsonable(value: Any) -> Any:
    """JSON-safe copy (numpy scalars, Paths, NaN)."""
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and math.isnan(value):
        return "NaN"
    if isinstance(value, bytes):
        return value.decode("latin-1", errors="replace")
    return value


def _result(
    check_id: str,
    title: str,
    status: str,
    summary: Sequence[str],
    data: Dict[str, Any],
) -> Dict[str, Any]:
    """One check record: id, title, single status, human summary, structured data."""
    return {
        "id": check_id,
        "title": title,
        "status": status,
        "code": status_code(status),
        "summary": [str(line) for line in summary],
        "data": _jsonable(data),
    }


def find_files(root: Path, patterns: Sequence[str]) -> Tuple[List[Path], str]:
    """Files matching the first non-empty pattern, sorted; also returns the pattern."""
    root = Path(root)
    for pattern in patterns:
        if not root.is_dir():
            return [], ""
        matches = sorted(path for path in root.glob(pattern) if path.is_file())
        if matches:
            return matches, pattern
    return [], ""


def _as_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _evenly_spaced(items: Sequence[Any], count: int) -> List[Any]:
    """Deterministic, evenly spread sample of ``count`` items (``count >= 1``)."""
    if count <= 0 or not items:
        return []
    if count >= len(items):
        return list(items)
    stride = len(items) / float(count)
    return [items[int(index * stride)] for index in range(count)]


def _value_label(value: float) -> str:
    if isinstance(value, float) and math.isnan(value):
        return "NaN"
    return f"{float(value):.10g}"


def _warning_summary(messages: Sequence[str]) -> Dict[str, Any]:
    """Warnings text + count, unique messages capped so the report stays readable."""
    unique = sorted(set(messages))
    return {
        "n": len(messages),
        "messages": unique[:5],
        "n_distinct": len(unique),
    }


def _array_stats(values: np.ndarray) -> Dict[str, Any]:
    """min/max in pixel units and the bit count actually used (max magnitude)."""
    if values.size == 0:
        return {"min": None, "max": None, "bits_used": 0}
    low, high = int(values.min()), int(values.max())
    peak = max(abs(low), abs(high))
    return {"min": low, "max": high, "bits_used": int(peak).bit_length()}


def _nifti_array(path: Path) -> np.ndarray:
    """Load a NIfTI volume as a 2D float array (native grid, no resize)."""
    data = np.squeeze(np.asanyarray(nib.load(str(path)).get_fdata()))
    if data.ndim != 2:
        raise RuntimeError(
            f"{path} is not a 2D grid after squeeze: shape {data.shape}"
        )
    return data


# ------------------------------------------------------------- 9.2 PTX-498 masks

PTX498_MASK_TITLE = "PTX-498 mask values (row 9.2)"


def check_ptx498_masks(root: Path) -> Dict[str, Any]:
    """Unique mask values and counts over every PTX-498 mask (Section 9.2 / PTX-02).

    The stage-2 loader binarises with ``mask > 0`` (contract module 6), so a
    {0, 255} or multi-label mask still yields a {0, 1} GT; this check records the
    actual value range because the scorer's old ``* 255 -> uint8`` path assumed
    {0, 1} (Y7).
    """
    root = Path(root)
    files = sorted(path for path in root.glob("Site*/*.5.mask.nii.gz") if path.is_file())
    base: Dict[str, Any] = {
        "root": str(root),
        "n_mask_files": len(files),
        "binarisation": "stage-2 loaders use `mask > 0` (contract module 6)",
    }
    if not files:
        return _result(
            "9.2",
            PTX498_MASK_TITLE,
            status_open("reason: no Site*/<n>.5.mask.nii.gz under the PTX-498 root"),
            ["no PTX-498 mask file found"],
            base,
        )

    totals: Counter = Counter()
    non_binary: List[Dict[str, Any]] = []
    unreadable: List[Dict[str, Any]] = []
    for path in tqdm(files, desc="9.2 PTX-498 masks", leave=False):
        try:
            data = _nifti_array(path)
        except Exception as exc:  # unreadable mask -> the check cannot pass
            unreadable.append({"file": path.name, "error": f"{type(exc).__name__}: {exc}"})
            continue
        values, counts = np.unique(data, return_counts=True)
        extra = []
        for value, count in zip(values, counts):
            label = _value_label(float(value))
            totals[label] += int(count)
            if label not in ("0", "1"):
                extra.append({"value": label, "pixels": int(count)})
        if extra:
            non_binary.append({"file": f"{path.parent.name}/{path.name}", "values": extra})

    top_values = [
        {"value": label, "pixels": count}
        for label, count in sorted(totals.items(), key=lambda item: -item[1])[:20]
    ]
    data = dict(base)
    data.update(
        {
            "n_non_binary_files": len(non_binary),
            "non_binary_files": non_binary[:20],
            "n_unreadable_files": len(unreadable),
            "unreadable_files": unreadable[:10],
            "value_counts": top_values,
            "total_pixels": int(sum(totals.values())),
        }
    )
    summary = [
        f"mask files: {len(files)}; distinct values: {len(totals)}",
        "top values: "
        + ", ".join(f"{item['value']} x {item['pixels']}" for item in top_values[:5]),
    ]
    if unreadable:
        summary.append(f"unreadable masks: {len(unreadable)}")
    if non_binary:
        examples = ", ".join(
            f"{item['file']} -> {[v['value'] for v in item['values']]}"
            for item in non_binary[:3]
        )
        summary.append(
            f"values outside {{0, 1}} in {len(non_binary)} mask file(s): {examples}"
        )
        return _result("9.2", PTX498_MASK_TITLE, status_issue("9.2"), summary, data)
    if unreadable:
        return _result(
            "9.2",
            PTX498_MASK_TITLE,
            status_open(f"reason: {len(unreadable)} mask file(s) could not be read"),
            summary,
            data,
        )
    summary.append("every mask value is 0 or 1")
    return _result(
        "9.2",
        PTX498_MASK_TITLE,
        status_no_change(f"all {len(files)} mask files hold only values 0 and 1"),
        summary,
        data,
    )


# ------------------------------------------------------- 9.3a PhotometricInterpretation

PHOTOMETRIC_TITLE = "{dataset} PhotometricInterpretation (row 9.3a)"


def check_photometric(
    root: Path, dataset: str, patterns: Sequence[str]
) -> Dict[str, Any]:
    """PhotometricInterpretation distribution + MONOCHROME1 IDs (Section 9.3a / Y8a).

    The author's SIIM loader reads ``pixel_array`` without looking at the tag, so a
    MONOCHROME1 image would enter the pipeline with inverted polarity.
    """
    root = Path(root)
    title = PHOTOMETRIC_TITLE.format(dataset=dataset)
    files, pattern = find_files(root, patterns)
    base: Dict[str, Any] = {
        "root": str(root),
        "dataset": dataset,
        "pattern": pattern,
        "n_dicom_files": len(files),
    }
    if not files:
        return _result(
            "9.3a",
            title,
            status_open("reason: no DICOM file under the dataset root"),
            ["no DICOM file found"],
            base,
        )

    counts: Counter = Counter()
    monochrome1: List[str] = []
    missing: List[str] = []
    unreadable: List[Dict[str, Any]] = []
    for path in tqdm(files, desc=f"9.3a {dataset} headers", leave=False):
        try:
            header = pydicom.dcmread(str(path), stop_before_pixels=True)
        except Exception as exc:
            unreadable.append({"file": path.name, "error": f"{type(exc).__name__}: {exc}"})
            continue
        value = header.get("PhotometricInterpretation", None)
        label = str(value) if value is not None else "MISSING"
        counts[label] += 1
        if label == "MONOCHROME1":
            monochrome1.append(path.stem)
        elif label == "MISSING":
            missing.append(path.stem)

    data = dict(base)
    data.update(
        {
            "distribution": [
                {"photometric_interpretation": label, "n_files": count}
                for label, count in sorted(counts.items(), key=lambda item: -item[1])
            ],
            "monochrome1_count": len(monochrome1),
            "monochrome1_ids": sorted(monochrome1)[:500],
            "monochrome1_ids_truncated": max(0, len(monochrome1) - 500),
            "missing_tag_count": len(missing),
            "missing_tag_ids": sorted(missing)[:50],
            "n_unreadable": len(unreadable),
            "unreadable_files": unreadable[:10],
        }
    )
    summary = [
        "distribution: "
        + ", ".join(f"{label} x {count}" for label, count in counts.most_common())
    ]
    if monochrome1:
        summary.append(
            f"MONOCHROME1: {len(monochrome1)} file(s), e.g. {sorted(monochrome1)[:5]}"
        )
        return _result("9.3a", title, status_issue("9.3a"), summary, data)
    if missing:
        summary.append(f"tag missing on {len(missing)} file(s), e.g. {sorted(missing)[:5]}")
        return _result(
            "9.3a",
            title,
            status_open(
                f"reason: PhotometricInterpretation missing on {len(missing)} of "
                f"{len(files)} file(s)"
            ),
            summary,
            data,
        )
    if unreadable:
        summary.append(f"unreadable headers: {len(unreadable)}")
        return _result(
            "9.3a",
            title,
            status_open(f"reason: {len(unreadable)} header(s) could not be read"),
            summary,
            data,
        )
    summary.append("no MONOCHROME1 image")
    return _result(
        "9.3a",
        title,
        status_no_change(f"0 of {len(files)} file(s) are MONOCHROME1"),
        summary,
        data,
    )


# ------------------------------------------------------------ 9.3b VinDr bit depth

VINDR_BITS_TITLE = "VinDr bit depth: unmodified vs forced BitsStored=16 (row 9.3b)"
VINDR_FORCED_BITS_STORED = 16


def _pixel_meta(header: Any) -> Dict[str, Any]:
    """The metadata tuple the 9.3b grouping uses."""
    meta = getattr(header, "file_meta", None)
    transfer_syntax = getattr(meta, "TransferSyntaxUID", None) if meta is not None else None
    return {
        "BitsStored": _as_int(header.get("BitsStored")),
        "BitsAllocated": _as_int(header.get("BitsAllocated")),
        "HighBit": _as_int(header.get("HighBit")),
        "PixelRepresentation": _as_int(header.get("PixelRepresentation")),
        "TransferSyntax": str(transfer_syntax) if transfer_syntax is not None else "MISSING",
    }


def _meta_key(meta: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        meta["BitsStored"],
        meta["BitsAllocated"],
        meta["HighBit"],
        meta["PixelRepresentation"],
        meta["TransferSyntax"],
    )


def decode_unmodified(path: Path) -> Tuple[np.ndarray, List[str]]:
    """Decode with the file's own metadata (``pixel_array``) and capture warnings."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        dataset = pydicom.dcmread(str(path))
        array = np.asarray(dataset.pixel_array)
    return array, [f"{type(item.message).__name__}: {item.message}" for item in caught]


def decode_forced_bits_stored(
    path: Path, bits: int = VINDR_FORCED_BITS_STORED
) -> Tuple[np.ndarray, List[str]]:
    """Decode the author's way: set ``BitsStored`` before touching ``pixel_array``.

    Replicates ``VinDataset.load_dicom`` (vin_dataset.py L137-145): ``dcmread``,
    ``dicom.BitsStored = 16``, then ``pixel_array``.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        dataset = pydicom.dcmread(str(path))
        dataset.BitsStored = bits
        array = np.asarray(dataset.pixel_array)
    return array, [f"{type(item.message).__name__}: {item.message}" for item in caught]


def compare_decodes(
    original: np.ndarray, forced: np.ndarray, bits_stored: Optional[int]
) -> Dict[str, Any]:
    """Pixel-level comparison of the two decodes (Section 9.3b evidence)."""
    row: Dict[str, Any] = {
        "shape_original": list(np.shape(original)),
        "shape_forced": list(np.shape(forced)),
        "comparable": bool(np.shape(original) == np.shape(forced)),
        "exact_equal": bool(
            np.shape(original) == np.shape(forced) and np.array_equal(original, forced)
        ),
    }
    if not row["comparable"]:
        return row
    left = np.asarray(original).astype(np.int64, copy=False)
    right = np.asarray(forced).astype(np.int64, copy=False)
    difference = np.abs(left - right)
    total = int(difference.size)
    changed = int(np.count_nonzero(difference))
    limit = 2 ** int(bits_stored) - 1 if bits_stored is not None else None
    row.update(
        {
            "n_pixels": total,
            "n_changed": changed,
            "fraction_changed": (changed / total) if total else 0.0,
            "max_abs_diff": int(difference.max()) if total else 0,
            "original": _array_stats(left),
            "forced": _array_stats(right),
            "bits_stored_limit": limit,
            "original_exceeds_bits_stored": (
                None if limit is None else bool(total and int(left.max()) > limit)
            ),
            "forced_exceeds_bits_stored": (
                None if limit is None else bool(total and int(right.max()) > limit)
            ),
        }
    )
    return row


def _is_compressed(transfer_syntax: str) -> Optional[bool]:
    if transfer_syntax in ("MISSING", "", "None"):
        return None
    try:
        return bool(pydicom.uid.UID(transfer_syntax).is_compressed)
    except Exception:
        return None


def check_vindr_bit_depth(root: Path, samples_per_group: int = 5) -> Dict[str, Any]:
    """VinDr metadata distribution + decode comparison (Section 9.3b / Y8b).

    For every metadata group, up to ``samples_per_group`` files are decoded twice
    (unmodified, then with ``BitsStored = 16``). Equality is evidence that the
    author's intervention changes nothing; a difference only shows that it changes
    the result, never which decode is right, so a difference stays OPEN.
    """
    root = Path(root)
    files, pattern = find_files(root, VINDR_DICOM_PATTERNS)
    base: Dict[str, Any] = {
        "root": str(root),
        "pattern": pattern,
        "n_dicom_files": len(files),
        "samples_per_group": samples_per_group,
        "forced_bits_stored": VINDR_FORCED_BITS_STORED,
    }
    if not files:
        return _result(
            "9.3b",
            VINDR_BITS_TITLE,
            status_open("reason: no *.dicom under the VinDr root"),
            ["no VinDr DICOM file found"],
            base,
        )

    groups: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    unreadable: List[Dict[str, Any]] = []
    for path in tqdm(files, desc="9.3b VinDr metadata", leave=False):
        try:
            header = pydicom.dcmread(str(path), stop_before_pixels=True)
        except Exception as exc:
            unreadable.append({"file": path.name, "error": f"{type(exc).__name__}: {exc}"})
            continue
        meta = _pixel_meta(header)
        key = _meta_key(meta)
        slot = groups.setdefault(key, {"meta": meta, "files": []})
        slot["files"].append(path)

    distribution = [
        dict(slot["meta"], n_files=len(slot["files"]))
        for _, slot in sorted(groups.items(), key=lambda item: -len(item[1]["files"]))
    ]

    samples: List[Dict[str, Any]] = []
    details: List[Dict[str, Any]] = []
    n_equal = 0
    n_differing = 0
    n_failed = 0
    for key, slot in sorted(groups.items(), key=lambda item: -len(item[1]["files"])):
        for path in _evenly_spaced(slot["files"], samples_per_group):
            try:
                original, warnings_original = decode_unmodified(path)
                forced, warnings_forced = decode_forced_bits_stored(
                    path, VINDR_FORCED_BITS_STORED
                )
            except Exception as exc:
                n_failed += 1
                samples.append(
                    {"file": path.name, "error": f"{type(exc).__name__}: {exc}"}
                )
                details.append(
                    {
                        "file": path.name,
                        "group": slot["meta"],
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
            comparison = compare_decodes(original, forced, slot["meta"]["BitsStored"])
            if comparison["exact_equal"]:
                n_equal += 1
            else:
                n_differing += 1
            warnings_unmodified = _warning_summary(warnings_original)
            warnings_forced_summary = _warning_summary(warnings_forced)
            row: Dict[str, Any] = {
                "file": path.name,
                "BitsStored": slot["meta"]["BitsStored"],
                "BitsAllocated": slot["meta"]["BitsAllocated"],
                "HighBit": slot["meta"]["HighBit"],
                "PixelRepresentation": slot["meta"]["PixelRepresentation"],
                "TransferSyntax": slot["meta"]["TransferSyntax"],
                "exact_equal": comparison["exact_equal"],
                "n_changed": comparison.get("n_changed"),
                "fraction_changed": comparison.get("fraction_changed"),
                "max_abs_diff": comparison.get("max_abs_diff"),
                "warnings_unmodified_n": warnings_unmodified["n"],
                "warnings_forced_n": warnings_forced_summary["n"],
                "warnings_unmodified": warnings_unmodified["messages"],
                "warnings_forced": warnings_forced_summary["messages"],
            }
            for side, stats in (
                ("original", comparison.get("original", {})),
                ("forced", comparison.get("forced", {})),
            ):
                row[f"{side}_min"] = stats.get("min")
                row[f"{side}_max"] = stats.get("max")
                row[f"{side}_bits_used"] = stats.get("bits_used")
                row[f"{side}_exceeds_bits_stored"] = comparison.get(
                    f"{side}_exceeds_bits_stored"
                )
            samples.append(row)
            details.append(
                {
                    "file": path.name,
                    "group": slot["meta"],
                    "warnings_unmodified": warnings_unmodified,
                    "warnings_forced": warnings_forced_summary,
                    "comparison": comparison,
                }
            )

    compressed = {_is_compressed(slot["meta"]["TransferSyntax"]) for slot in groups.values()}
    data = dict(base)
    data.update(
        {
            "n_groups": len(groups),
            "distribution": distribution,
            "n_unreadable": len(unreadable),
            "unreadable_files": unreadable[:10],
            "n_sampled": n_equal + n_differing + n_failed,
            "n_equal": n_equal,
            "n_differing": n_differing,
            "n_failed": n_failed,
            "samples": samples,
            "samples_detail": details,
            "any_compressed": compressed,
        }
    )
    summary = [
        f"metadata groups: {len(groups)}; sampled files: {n_equal + n_differing + n_failed}",
        f"pixel-identical: {n_equal}; differing: {n_differing}; failed: {n_failed}",
        "groups: "
        + "; ".join(
            f"(BS={row['BitsStored']}, BA={row['BitsAllocated']}, HB={row['HighBit']}, "
            f"PR={row['PixelRepresentation']}, {row['TransferSyntax']}) x {row['n_files']}"
            for row in distribution[:5]
        ),
    ]
    if unreadable:
        summary.append(f"unreadable headers: {len(unreadable)}")
    if n_failed:
        summary.append(f"{n_failed} sampled file(s) could not be decoded")
        return _result(
            "9.3b",
            VINDR_BITS_TITLE,
            status_open(f"reason: {n_failed} sampled file(s) could not be decoded"),
            summary,
            data,
        )
    if n_differing:
        summary.append(
            "same min-max range is not proof; which decode is correct is not decided "
            "here (human review of the warning cause and the real bit encoding)"
        )
        return _result(
            "9.3b",
            VINDR_BITS_TITLE,
            status_open(
                f"reason: forced and unmodified decode differ on {n_differing} of "
                f"{n_equal + n_differing} sampled file(s); which decode is correct "
                "requires human review"
            ),
            summary,
            data,
        )
    if compressed == {False}:
        summary.append(
            "all sampled files are uncompressed, so BitsStored does not enter the "
            "decode; equality shows the intervention is a no-op for these files"
        )
    summary.append("no pixel value changed between the two decodes")
    return _result(
        "9.3b",
        VINDR_BITS_TITLE,
        status_no_change(
            f"pixel-identical on {n_equal} of {n_equal} sampled file(s) across "
            f"{len(groups)} metadata group(s)"
        ),
        summary,
        data,
    )


# ----------------------------------------------------- SIIM ImageId -> file mapping

SIIM_MAPPING_TITLE = "SIIM-ACR ImageId -> DICOM mapping and author split (row 9.4)"


def _label_columns(label_df: pd.DataFrame) -> Tuple[str, str]:
    """The ImageId / EncodedPixels columns (the author's CSV keeps a leading space)."""
    stripped = {str(column).strip(): column for column in label_df.columns}
    missing = [name for name in ("ImageId", "EncodedPixels") if name not in stripped]
    if missing:
        raise KeyError(f"train-rle.csv is missing columns {missing}")
    return stripped["ImageId"], stripped["EncodedPixels"]


def author_label_frame(
    label_df: pd.DataFrame,
) -> Tuple[List[Any], Dict[Any, int], int]:
    """Replicate the author's label rule without importing the dataset class.

    ``process_labels`` sets label 1 when *any* row of an ImageId has
    ``row[" EncodedPixels"] != " -1"`` and iterates ``label_df["ImageId"].unique()``.
    The literal rule is kept next to a whitespace-normalised one
    (``strip() != "-1"``); any ImageId where the two disagree is counted and
    reported, because the literal rule would label a bare ``-1`` as positive.
    """
    id_col, encoded_col = _label_columns(label_df)
    encoded = label_df[encoded_col].astype(str)
    normalised = encoded.str.strip() != "-1"
    literal = encoded != " -1"
    grouped = normalised.groupby(label_df[id_col], sort=False).any()
    grouped_literal = literal.groupby(label_df[id_col], sort=False).any()
    image_ids = list(label_df[id_col].unique())
    labels = {image_id: int(bool(grouped[image_id])) for image_id in image_ids}
    labels_literal = {image_id: int(bool(grouped_literal[image_id])) for image_id in image_ids}
    disagreements = sum(1 for image_id in image_ids if labels[image_id] != labels_literal[image_id])
    return image_ids, labels, disagreements


def author_split(
    image_ids: Sequence[Any], labels: Dict[Any, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """The author's ``train_test_split(stratify, test_size=0.1, random_state=42)``."""
    split_labels = [labels[image_id] for image_id in image_ids]
    indices = np.arange(len(image_ids))
    return train_test_split(
        indices,
        stratify=split_labels,
        test_size=SIIM_TEST_SIZE,
        random_state=SIIM_SPLIT_SEED,
    )


def check_siim_mapping(
    root: Path, expected: Optional[Dict[str, int]] = None
) -> Dict[str, Any]:
    """ImageId -> file uniqueness, labelled-ID count and the author's 90/10 split."""
    root = Path(root)
    rle_path = root / SIIM_RLE_NAME
    files, pattern = find_files(root, SIIM_DICOM_PATTERNS)
    file_map: Dict[str, List[Path]] = {}
    for path in files:
        file_map.setdefault(path.stem, []).append(path)
    expected_counts = dict(EXPECTED_SIIM if expected is None else expected)
    base: Dict[str, Any] = {
        "root": str(root),
        "dicom_pattern": pattern,
        "n_dicom_files": len(files),
        "rle_csv": str(rle_path),
        "expected": expected_counts,
    }
    if not rle_path.is_file():
        return _result(
            "9.4-SIIM",
            SIIM_MAPPING_TITLE,
            status_open(f"reason: {SIIM_RLE_NAME} not found under the SIIM root"),
            [f"{SIIM_RLE_NAME} not found"],
            base,
        )

    label_df = pd.read_csv(str(rle_path))
    image_ids, labels, disagreements = author_label_frame(label_df)
    try:
        train_idx, test_idx = author_split(image_ids, labels)
    except ValueError as exc:
        # The author's stratified split is undefined for a tiny label set (fewer than
        # two members in a class). Reported, never swallowed: no split, no pass.
        error_data = dict(base)
        error_data.update(
            {
                "n_labelled_ids": len(image_ids),
                "n_positive_ids": int(sum(labels.values())),
                "split_error": f"{type(exc).__name__}: {exc}",
            }
        )
        return _result(
            "9.4-SIIM",
            SIIM_MAPPING_TITLE,
            status_open(
                "reason: the author's stratified split is undefined for this label "
                f"set ({type(exc).__name__})"
            ),
            [f"{len(image_ids)} labelled ImageId(s) in the csv", str(exc)],
            error_data,
        )
    test_pos = int(sum(labels[image_ids[index]] for index in test_idx))
    observed = {
        "labelled_ids": len(image_ids),
        "dev_ids": int(len(train_idx)),
        "test_ids": int(len(test_idx)),
        "test_pos": test_pos,
        "test_neg": int(len(test_idx)) - test_pos,
    }

    labelled_ids = {str(image_id) for image_id in image_ids}
    missing = sorted(str(image_id) for image_id in image_ids if str(image_id) not in file_map)
    duplicates = sorted(
        stem
        for stem, paths in file_map.items()
        if len(paths) > 1 and stem in labelled_ids
    )
    mismatches = [
        {"field": field, "observed": observed[field], "expected": expected_counts[field]}
        for field in expected_counts
        if field in observed and observed[field] != expected_counts[field]
    ]

    data = dict(base)
    data.update(
        {
            "observed": observed,
            "mismatches": mismatches,
            "n_missing_files": len(missing),
            "missing_examples": missing[:5],
            "n_duplicate_image_ids": len(duplicates),
            "duplicate_examples": duplicates[:5],
            "n_unique_file_stems": len(file_map),
            "label_rule_disagreements": disagreements,
            "split": (
                f"train_test_split(test_size={SIIM_TEST_SIZE}, "
                f"random_state={SIIM_SPLIT_SEED})"
            ),
        }
    )
    summary = [
        "observed: "
        + ", ".join(f"{field}={value}" for field, value in observed.items()),
        "expected: "
        + ", ".join(f"{field}={value}" for field, value in expected_counts.items()),
        f"DICOM files: {len(files)} (pattern {pattern!r}); unique stems: {len(file_map)}",
    ]
    if disagreements:
        summary.append(
            f"{disagreements} ImageId(s) where the literal rule `!= \" -1\"` disagrees "
            "with `strip() != \"-1\"`"
        )
        return _result(
            "9.4-SIIM",
            SIIM_MAPPING_TITLE,
            status_open(
                f"reason: {disagreements} ImageId(s) depend on the whitespace of "
                "the EncodedPixels sentinel"
            ),
            summary,
            data,
        )
    if duplicates:
        summary.append(
            f"{len(duplicates)} ImageId(s) map to more than one file: "
            f"{duplicates[:5]}"
        )
        return _result(
            "9.4-SIIM", SIIM_MAPPING_TITLE, status_issue("9.4-SIIM"), summary, data
        )
    if missing:
        summary.append(f"{len(missing)} ImageId(s) have no DICOM file: {missing[:5]}")
        return _result(
            "9.4-SIIM",
            SIIM_MAPPING_TITLE,
            status_open(
                f"reason: {len(missing)} of {len(image_ids)} labelled ImageId(s) have "
                "no DICOM file with that filename stem"
            ),
            summary,
            data,
        )
    if mismatches:
        summary.append(
            "mismatches: "
            + ", ".join(
                f"{item['field']} {item['observed']} != {item['expected']}"
                for item in mismatches
            )
        )
        return _result(
            "9.4-SIIM",
            SIIM_MAPPING_TITLE,
            status_open(
                f"reason: {len(mismatches)} observed count(s) differ from the protocol "
                "expectation"
            ),
            summary,
            data,
        )
    summary.append("every labelled ImageId maps to exactly one file; counts match")
    return _result(
        "9.4-SIIM",
        SIIM_MAPPING_TITLE,
        status_no_change(
            f"{observed['labelled_ids']} labelled ImageIds, one file each; split "
            f"{observed['dev_ids']}/{observed['test_ids']} and test "
            f"{observed['test_pos']}/{observed['test_neg']} match"
        ),
        summary,
        data,
    )


# ---------------------------------------------------- 9.1 orientation overlays

ORIENTATION_TITLE = "PTX-498 orientation vs SIIM (row 9.1, human decision)"


def _ptx498_pairs(root: Path, count: int) -> List[Tuple[Path, Path]]:
    """First ``count`` PTX-498 image/mask pairs (image ``.4.img``, mask ``.5.mask``)."""
    pairs: List[Tuple[Path, Path]] = []
    if count <= 0:
        return pairs
    for image_path in sorted(Path(root).glob("Site*/*.4.img.nii.gz")):
        mask_path = image_path.with_name(
            image_path.name.replace(".4.img.nii.gz", ".5.mask.nii.gz")
        )
        if mask_path.is_file():
            pairs.append((image_path, mask_path))
        if len(pairs) >= count:
            break
    return pairs


def _rle_decode(encoded: Any, size: int = 1024) -> np.ndarray:
    """Author's RLE decode: 1024x1024 grid, ``rle2mask(...).T`` (siim_acr_dataset.py)."""
    flat = np.zeros(size * size, dtype=np.uint8)
    tokens = np.asarray([int(token) for token in str(encoded).split()], dtype=np.int64)
    if tokens.size:
        starts = tokens[0::2]
        lengths = tokens[1::2]
        position = 0
        for start, length in zip(starts, lengths):
            position += int(start)
            flat[position:position + int(length)] = 1
            position += int(length)
    return flat.reshape(size, size).T


def _siim_positive_pairs(
    root: Path, count: int
) -> List[Tuple[str, Path, np.ndarray]]:
    """First ``count`` SIIM positive images that exist, with path and GT mask."""
    root = Path(root)
    pairs: List[Tuple[str, Path, np.ndarray]] = []
    rle_path = root / SIIM_RLE_NAME
    if count <= 0 or not rle_path.is_file():
        return pairs
    label_df = pd.read_csv(str(rle_path))
    image_ids, labels, _ = author_label_frame(label_df)
    files, _ = find_files(root, SIIM_DICOM_PATTERNS)
    file_map: Dict[str, List[Path]] = {}
    for path in files:
        file_map.setdefault(path.stem, []).append(path)
    id_col, encoded_col = _label_columns(label_df)
    for image_id in image_ids:
        if len(pairs) >= count:
            break
        matches = file_map.get(str(image_id), [])
        if labels[image_id] != 1 or len(matches) != 1:
            continue
        rows = label_df[label_df[id_col] == image_id]
        mask = np.zeros((1024, 1024), dtype=np.uint8)
        for encoded in rows[encoded_col].astype(str):
            if str(encoded).strip() != "-1":
                mask[_rle_decode(encoded) > 0] = 1
        pairs.append((str(image_id), matches[0], mask))
    return pairs


def _overlay_rgb(image: np.ndarray, mask: Optional[np.ndarray]) -> Tuple[np.ndarray, bool]:
    """Grey image in [0, 1] with mask foreground in red; returns (rgb, mask_resized)."""
    image = np.asarray(image, dtype=np.float64)
    low, high = float(image.min()), float(image.max())
    span = high - low
    scaled = (image - low) / span if span > 0 else np.zeros_like(image)
    rgb = np.stack([scaled] * 3, axis=-1)
    resized = False
    if mask is not None:
        mask = np.asarray(mask)
        if mask.shape != image.shape:
            from PIL import Image

            mask = np.asarray(
                Image.fromarray((mask > 0).astype(np.uint8) * 255).resize(
                    (image.shape[1], image.shape[0]), resample=Image.NEAREST
                )
            )
            resized = True
        fore = mask > 0
        rgb[fore] = 0.4 * rgb[fore] + 0.6 * np.array([1.0, 0.0, 0.0])
    return rgb, resized


def write_orientation_overlays(
    siim_root: Path, ptx498_root: Path, out_dir: Path, samples: int = 5
) -> Dict[str, Any]:
    """Side-by-side PTX-498 / SIIM overlays; the human decides, the script does not."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ptx_pairs = _ptx498_pairs(Path(ptx498_root), samples)
    siim_pairs = _siim_positive_pairs(Path(siim_root), samples)

    items: List[Dict[str, Any]] = []
    files: List[str] = []
    if ptx_pairs and siim_pairs:
        total = min(len(ptx_pairs), len(siim_pairs))
    else:
        total = max(len(ptx_pairs), len(siim_pairs))
    for index in range(total):
        figure, axes = plt.subplots(1, 2, figsize=(10.0, 5.0), squeeze=False)
        left, right = axes[0][0], axes[0][1]
        item: Dict[str, Any] = {
            "file": str(out_dir / f"overlay_{index + 1:02d}.png"),
            "ptx498_id": None,
            "ptx498_image": None,
            "ptx498_mask": None,
            "ptx_mask_resized": None,
            "siim_id": None,
            "siim_image": None,
            "siim_mask_resized": None,
            "error": None,
        }
        if index < len(ptx_pairs):
            image_path, mask_path = ptx_pairs[index]
            image = np.rot90(_nifti_array(image_path), k=1)
            mask = np.rot90(_nifti_array(mask_path), k=1)
            rgb, resized = _overlay_rgb(image, mask)
            left.imshow(rgb)
            left.set_title(
                f"PTX-498 {image_path.parent.name}/{image_path.name.split('.')[0]} "
                "(author rot90 k=1)",
                fontsize=9,
            )
            item["ptx498_id"] = f"{image_path.parent.name}/{image_path.name.split('.')[0]}"
            item["ptx498_image"] = str(image_path)
            item["ptx498_mask"] = str(mask_path)
            item["ptx_mask_resized"] = resized
        else:
            left.text(0.5, 0.5, "PTX-498 not available", ha="center", va="center")
        left.axis("off")
        if index < len(siim_pairs):
            image_id, image_path, mask = siim_pairs[index]
            item["siim_id"] = image_id
            item["siim_image"] = str(image_path)
            try:
                image = np.asarray(pydicom.dcmread(str(image_path)).pixel_array)
            except Exception as exc:
                item["error"] = f"{type(exc).__name__}: {exc}"
                right.text(
                    0.5,
                    0.5,
                    f"SIIM {image_id} unreadable\n{type(exc).__name__}",
                    ha="center",
                    va="center",
                )
                right.axis("off")
                figure.savefig(item["file"], dpi=110, bbox_inches="tight")
                plt.close(figure)
                items.append(item)
                files.append(item["file"])
                continue
            rgb, resized = _overlay_rgb(image, mask)
            right.imshow(rgb)
            right.set_title(f"SIIM {image_id} (positive GT)", fontsize=9)
            item["siim_mask_resized"] = resized
        else:
            right.text(0.5, 0.5, "SIIM not available", ha="center", va="center")
        right.axis("off")
        figure.suptitle(
            "Row 9.1 — PTX-498 vs SIIM orientation (human visual decision)", fontsize=10
        )
        figure.tight_layout()
        figure.savefig(item["file"], dpi=110, bbox_inches="tight")
        plt.close(figure)
        items.append(item)
        files.append(item["file"])

    return {
        "dir": str(out_dir),
        "n_ptx498": len(ptx_pairs),
        "n_siim": len(siim_pairs),
        "files": files,
        "items": items,
    }


def _orientation_check(overlay: Dict[str, Any]) -> Dict[str, Any]:
    summary = [
        f"overlay files: {len(overlay['files'])} "
        f"({overlay['n_ptx498']} PTX-498, {overlay['n_siim']} SIIM)",
        "the human compares the two columns and records the decision in the report",
        "no automatic orientation/mirroring decision is made",
    ]
    if not overlay["files"]:
        return _result(
            "9.1",
            ORIENTATION_TITLE,
            status_open(
                "reason: no PTX-498 image/mask pair and no SIIM positive image found"
            ),
            summary,
            overlay,
        )
    return _result(
        "9.1",
        ORIENTATION_TITLE,
        status_open("human visual decision required"),
        summary,
        overlay,
    )


# ------------------------------------ positives with an empty GT mask (K1 guard)

EMPTY_GT_TITLE = "Positive images with an empty GT mask (saliency K1 guard)"
EMPTY_GT_B_ITEM = "7.4"


def _empty_ptx498_masks(root: Path) -> Dict[str, Any]:
    """PTX-498 masks without a single foreground pixel."""
    files = sorted(path for path in Path(root).glob("Site*/*.5.mask.nii.gz") if path.is_file())
    empty: List[str] = []
    unreadable: List[Dict[str, Any]] = []
    for path in tqdm(files, desc="GT-EMPTY PTX-498", leave=False):
        try:
            data = _nifti_array(path)
        except Exception as exc:
            unreadable.append({"file": path.name, "error": f"{type(exc).__name__}: {exc}"})
            continue
        if not bool(np.any(data > 0)):
            empty.append(f"{path.parent.name}/{path.name.split('.')[0]}")
    return {
        "root": str(root),
        "available": True,
        "n_masks": len(files),
        "n_empty": len(empty),
        "empty_ids": empty[:500],
        "empty_ids_truncated": max(0, len(empty) - 500),
        "n_unreadable": len(unreadable),
        "unreadable_files": unreadable[:10],
    }


def _empty_siim_positives(root: Path) -> Dict[str, Any]:
    """SIIM ImageIds labelled positive whose decoded RLE mask has no foreground."""
    root = Path(root)
    rle_path = root / SIIM_RLE_NAME
    if not rle_path.is_file():
        return {
            "root": str(root),
            "available": False,
            "reason": f"{SIIM_RLE_NAME} not found",
        }
    label_df = pd.read_csv(str(rle_path))
    id_col, encoded_col = _label_columns(label_df)
    image_ids, labels, _ = author_label_frame(label_df)
    encodings: Dict[Any, List[str]] = {}
    for image_id, encoded in zip(label_df[id_col], label_df[encoded_col]):
        encodings.setdefault(image_id, []).append(str(encoded))
    positives = [image_id for image_id in image_ids if labels[image_id] == 1]
    empty: List[str] = []
    for image_id in tqdm(positives, desc="GT-EMPTY SIIM", leave=False):
        has_foreground = False
        for encoded in encodings.get(image_id, []):
            if encoded.strip() == "-1":
                continue
            if bool(np.any(_rle_decode(encoded) > 0)):
                has_foreground = True
                break
        if not has_foreground:
            empty.append(str(image_id))
    return {
        "root": str(root),
        "available": True,
        "rle_csv": str(rle_path),
        "n_positive_ids": len(positives),
        "n_empty": len(empty),
        "empty_ids": empty[:500],
        "empty_ids_truncated": max(0, len(empty) - 500),
    }


def check_empty_gt_masks(
    siim_root: Optional[Path] = None, ptx498_root: Optional[Path] = None
) -> Dict[str, Any]:
    """Count positives whose GT mask is empty (B item 7.4 / K1 saliency scoring).

    ``score_saliency`` raises ``RuntimeError`` listing the ImageId(s) of a GT-positive
    image whose reference-grid mask is empty (saliency_eval.py, PTX-04 / Section 7.4),
    so such an image cannot be scored or dropped silently. A dataset root that is
    missing is not inspected; the check never reports zero from missing data.
    """
    runners = (
        ("ptx498", ptx498_root, _empty_ptx498_masks),
        ("siim", siim_root, _empty_siim_positives),
    )
    blocks: Dict[str, Any] = {}
    for name, root, runner in runners:
        if root is not None and Path(root).is_dir():
            blocks[name] = runner(Path(root))
        else:
            blocks[name] = {
                "root": None if root is None else str(root),
                "available": False,
                "reason": "root missing",
            }

    inspected = [name for name, block in blocks.items() if block.get("available")]
    empty_total = sum(int(block.get("n_empty", 0)) for block in blocks.values())
    data: Dict[str, Any] = {
        "datasets": blocks,
        "n_empty_total": empty_total,
        "b_item": EMPTY_GT_B_ITEM,
        "scoring_effect": (
            "score_saliency raises RuntimeError when a GT-positive image has an empty "
            "reference-grid mask (saliency_eval.py, PTX-04 / Section 7.4)"
        ),
    }
    summary = [
        f"inspected datasets: {', '.join(inspected) if inspected else 'none'}",
    ]
    for name, block in blocks.items():
        if block.get("available"):
            summary.append(
                f"{name}: {block['n_empty']} empty GT mask(s) of "
                + (
                    f"{block['n_masks']} mask(s)"
                    if "n_masks" in block
                    else f"{block['n_positive_ids']} positive ImageId(s)"
                )
            )
        else:
            summary.append(f"{name}: not inspected ({block.get('reason')})")
    if blocks["ptx498"].get("available"):
        summary.append(
            f"PTX-498 example ids: {blocks['ptx498']['empty_ids'][:5] or 'none'}"
        )
    if blocks["siim"].get("available"):
        summary.append(
            f"SIIM example ids: {blocks['siim']['empty_ids'][:5] or 'none'}"
        )
    unreadable = int(blocks["ptx498"].get("n_unreadable", 0))
    if unreadable:
        summary.append(f"unreadable PTX-498 masks: {unreadable}")

    if empty_total:
        return _result(
            "GT-EMPTY", EMPTY_GT_TITLE, status_issue(EMPTY_GT_B_ITEM), summary, data
        )
    if not inspected:
        return _result(
            "GT-EMPTY", EMPTY_GT_TITLE, status_not_run("path missing"), summary, data
        )
    if len(inspected) != len(blocks):
        missing = [name for name, block in blocks.items() if not block.get("available")]
        reason = "; ".join(
            f"{name}: {blocks[name].get('reason')}" for name in missing
        )
        return _result(
            "GT-EMPTY",
            EMPTY_GT_TITLE,
            status_open(f"reason: {reason}, completeness unproven"),
            summary,
            data,
        )
    if unreadable:
        return _result(
            "GT-EMPTY",
            EMPTY_GT_TITLE,
            status_open(f"reason: {unreadable} PTX-498 mask(s) could not be read"),
            summary,
            data,
        )
    evidence = (
        f"0 empty GT masks in {blocks['ptx498']['n_masks']} PTX-498 mask(s) and "
        f"{blocks['siim']['n_positive_ids']} SIIM positive ImageId(s)"
    )
    summary.append("no positive image has an empty GT mask")
    return _result(
        "GT-EMPTY", EMPTY_GT_TITLE, status_no_change(evidence), summary, data
    )


# ------------------------------------------------------------------- orchestration

ROOT_NAMES: Tuple[str, ...] = ("siim", "vindr", "ptx498")


def _resolve_roots(
    siim_root: Optional[str], vindr_root: Optional[str], ptx498_root: Optional[str]
) -> Dict[str, Path]:
    """Explicit roots win; otherwise the configured ones (never a hard-coded path)."""
    return {
        "siim": Path(siim_root) if siim_root else Path(str(cfg.ptx.siim_acr.root_dir)),
        "vindr": Path(vindr_root) if vindr_root else Path(str(cfg.ptx.vindr.root_dir)),
        "ptx498": Path(ptx498_root) if ptx498_root else Path(str(cfg.ptx.ptx498.root_dir)),
    }


def _not_run(check_id: str, title: str, root: Path, detail: str = "") -> Dict[str, Any]:
    summary = [f"root not found: {root}"]
    if detail:
        summary.append(detail)
    summary.append("nothing was inspected for this check; this is not a pass")
    return _result(check_id, title, status_not_run("path missing"), summary, {"root": str(root)})


def run_checks(
    out: Any = "reports",
    siim_root: Optional[str] = None,
    vindr_root: Optional[str] = None,
    ptx498_root: Optional[str] = None,
    samples_per_group: int = 5,
    as_of: Optional[str] = None,
) -> Dict[str, Any]:
    """Run every Section 9 check and write the dated markdown + JSON report."""
    repo = repo_root()
    out_dir = Path(out) if Path(out).is_absolute() else repo / Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_date = as_of or date.today().isoformat()
    roots = _resolve_roots(siim_root, vindr_root, ptx498_root)
    present = {name: path.is_dir() for name, path in roots.items()}
    overlay_dir = out_dir / f"{REPORT_PREFIX}_{report_date}{OVERLAY_SUFFIX}"

    checks: List[Dict[str, Any]] = []

    if present["ptx498"]:
        checks.append(check_ptx498_masks(roots["ptx498"]))
    else:
        checks.append(_not_run("9.2", PTX498_MASK_TITLE, roots["ptx498"]))

    if present["siim"]:
        checks.append(
            check_photometric(roots["siim"], "SIIM-ACR", SIIM_DICOM_PATTERNS)
        )
    else:
        checks.append(
            _not_run("9.3a", PHOTOMETRIC_TITLE.format(dataset="SIIM-ACR"), roots["siim"])
        )

    if present["vindr"]:
        checks.append(
            check_photometric(roots["vindr"], "VinDr", VINDR_DICOM_PATTERNS)
        )
    else:
        checks.append(
            _not_run("9.3a", PHOTOMETRIC_TITLE.format(dataset="VinDr"), roots["vindr"])
        )

    if present["vindr"]:
        checks.append(check_vindr_bit_depth(roots["vindr"], samples_per_group))
    else:
        checks.append(_not_run("9.3b", VINDR_BITS_TITLE, roots["vindr"]))

    if present["siim"]:
        checks.append(check_siim_mapping(roots["siim"]))
    else:
        checks.append(_not_run("9.4-SIIM", SIIM_MAPPING_TITLE, roots["siim"]))

    checks.append(
        check_empty_gt_masks(
            siim_root=roots["siim"] if present["siim"] else None,
            ptx498_root=roots["ptx498"] if present["ptx498"] else None,
        )
    )

    if present["siim"] and present["ptx498"]:
        overlay = write_orientation_overlays(
            roots["siim"], roots["ptx498"], overlay_dir, samples_per_group
        )
        checks.append(_orientation_check(overlay))
    else:
        missing = [name for name in ("siim", "ptx498") if not present[name]]
        checks.append(
            _not_run(
                "9.1",
                ORIENTATION_TITLE,
                roots["siim"] if not present["siim"] else roots["ptx498"],
                f"overlays need both roots; missing: {', '.join(missing)}",
            )
        )

    counts = Counter(check["code"] for check in checks)
    payload: Dict[str, Any] = {
        "date": report_date,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "repo_root": str(repo),
        "script": "sharpness_matters/ptx/corrected/data_checks.py",
        "samples_per_group": samples_per_group,
        "roots": {name: str(path) for name, path in roots.items()},
        "roots_present": present,
        "checks": checks,
        "status_counts": {code: counts.get(code, 0) for code in STATUS_CODES},
        "b_item_note": B_ITEM_CONSTANTS_NOTE,
        "environment": {
            "python": platform.python_version(),
            "pydicom": pydicom.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "nibabel": nib.__version__,
        },
        "overlay_dir": str(overlay_dir),
    }
    return write_report(payload, out_dir)


# ------------------------------------------------------------------------- report


def _md_cell(value: Any) -> str:
    text = _md_value(value)
    return text.replace("|", "\\|").replace("\n", " ")


def _md_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (list, tuple)):
        if not value:
            return "—"
        items = [_md_value(item) for item in value]
        if len(items) > 12:
            items = items[:12] + [f"… (+{len(value) - 12} more)"]
        return ", ".join(items)
    if isinstance(value, dict):
        return json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True) if value else "—"
    if value is None:
        return "—"
    if value == "":
        return "—"
    return str(value)


def _md_table(rows: Sequence[Dict[str, Any]], indent: str = "") -> List[str]:
    columns: List[str] = []
    for row in rows:
        for column in row:
            if column not in columns:
                columns.append(column)
    lines = [
        f"{indent}| " + " | ".join(str(column) for column in columns) + " |",
        f"{indent}|" + "|".join("---" for _ in columns) + "|",
    ]
    shown = list(rows)[:200]
    for row in shown:
        lines.append(
            f"{indent}| "
            + " | ".join(_md_cell(row.get(column)) for column in columns)
            + " |"
        )
    if len(rows) > len(shown):
        lines.append(f"{indent}| … | {len(rows) - len(shown)} more row(s) in the JSON |")
    return lines


# Keys that are kept in the JSON only: they repeat a table already rendered in the
# markdown report (per-sample deep comparison) and would only bloat it.
JSON_ONLY_KEYS: Tuple[str, ...] = ("samples_detail",)


def _md_block(
    data: Dict[str, Any], indent: str = "", skip: Sequence[str] = ()
) -> List[str]:
    lines: List[str] = []
    for key, value in data.items():
        if str(key) in skip:
            continue
        label = str(key).replace("_", " ")
        if isinstance(value, dict):
            lines.append(f"{indent}- **{label}**")
            lines.extend(_md_block(value, indent + "  ", skip))
        elif isinstance(value, list) and value and all(isinstance(item, dict) for item in value):
            lines.append(f"{indent}- **{label}** ({len(value)} row(s))")
            lines.append("")
            lines.extend(_md_table(value, indent + "  "))
            lines.append("")
        else:
            lines.append(f"{indent}- **{label}**: {_md_value(value)}")
    return lines


def write_report(payload: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
    """Write ``data_checks_<date>.md`` and ``data_checks_<date>.json`` under out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / f"{REPORT_PREFIX}_{payload['date']}.md"
    json_path = out_dir / f"{REPORT_PREFIX}_{payload['date']}.json"
    payload["outputs"] = {
        "markdown": str(md_path),
        "json": str(json_path),
        "overlay_dir": payload.get("overlay_dir"),
    }

    lines: List[str] = [
        "# Pre-run data checks — SIIM-ACR / VinDr / PTX-498",
        "",
        f"- Date: {payload['date']}",
        f"- Generated (UTC): {payload['generated_at_utc']}",
        f"- Script: `{payload['script']}`",
        f"- Repo: `{payload['repo_root']}`",
        "- Read-only: no training, no model load, no write/move inside a dataset root",
        f"- Report: `{md_path}` / `{json_path}`",
        "",
        "## Roots",
        "",
        "| Dataset | Root | Present |",
        "|---|---|---|",
    ]
    for name in ROOT_NAMES:
        lines.append(
            f"| {name} | `{payload['roots'][name]}` | "
            f"{'yes' if payload['roots_present'][name] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Status summary",
            "",
            "| Check | Title | Status |",
            "|---|---|---|",
        ]
    )
    for check in payload["checks"]:
        lines.append(
            f"| {check['id']} | {_md_cell(check['title'])} | {_md_cell(check['status'])} |"
        )
    lines.extend(
        [
            "",
            "Counts: "
            + ", ".join(
                f"{code}={count}" for code, count in payload["status_counts"].items()
            ),
            "",
            "Status vocabulary: `ISSUE → B item <no>` / `NO CHANGE NEEDED (evidence: …)` / "
            "`OPEN (reason: …)` / `NOT RUN (path missing)`. An ambiguous result is OPEN.",
            "",
        ]
    )
    for check in payload["checks"]:
        lines.append(f"## {check['id']} — {check['title']}")
        lines.append("")
        lines.append(f"**Status:** {check['status']}")
        lines.append("")
        for item in check["summary"]:
            lines.append(f"- {item}")
        lines.append("")
        lines.append("### Evidence")
        lines.append("")
        lines.extend(_md_block(check["data"], skip=JSON_ONLY_KEYS))
        hidden = [key for key in JSON_ONLY_KEYS if key in check["data"]]
        if hidden:
            lines.append("")
            lines.append(
                "- deeper per-sample detail: "
                + ", ".join(f"`{key}`" for key in hidden)
                + " in the JSON report"
            )
        lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(f"- {payload['b_item_note']}")
    lines.append("- 9.1 is a human decision: the PNG overlays are inputs, the status stays OPEN.")
    lines.append("- This report never asserts that a decode is correct; it records equality.")
    lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=False),
        encoding="utf-8",
    )
    return payload


# ---------------------------------------------------------------------------- CLI


@click.command()
@click.option(
    "--out",
    default="reports",
    show_default=True,
    help="Report directory; relative paths resolve against the repository root.",
)
@click.option(
    "--siim-root",
    default=None,
    type=click.Path(file_okay=False),
    help="SIIM-ACR root (default: cfg.ptx.siim_acr.root_dir).",
)
@click.option(
    "--vindr-root",
    default=None,
    type=click.Path(file_okay=False),
    help="VinDr root (default: cfg.ptx.vindr.root_dir).",
)
@click.option(
    "--ptx498-root",
    default=None,
    type=click.Path(file_okay=False),
    help="PTX-498 root (default: cfg.ptx.ptx498.root_dir).",
)
@click.option(
    "--samples-per-group",
    default=5,
    show_default=True,
    type=click.IntRange(min=1),
    help="Files decoded per VinDr metadata group, and overlays per dataset.",
)
def cli(
    out: str,
    siim_root: Optional[str],
    vindr_root: Optional[str],
    ptx498_root: Optional[str],
    samples_per_group: int,
) -> None:
    """Run the Section 9 pre-run data checks and write a dated report."""
    payload = run_checks(
        out=out,
        siim_root=siim_root,
        vindr_root=vindr_root,
        ptx498_root=ptx498_root,
        samples_per_group=samples_per_group,
    )
    for check in payload["checks"]:
        click.echo(f"{check['id']}: {check['status']}")
    click.echo(f"markdown: {payload['outputs']['markdown']}")
    click.echo(f"json:     {payload['outputs']['json']}")


if __name__ == "__main__":
    cli()

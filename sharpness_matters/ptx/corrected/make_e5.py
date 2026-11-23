"""E5 (reference-like) and E5-clean VinDr manifests — PTX-03 / PTX-08.

# B (PTX-03 / PTX-08, Section 4.2): the author's selector keeps every
#   `class_name != "Pneumothorax"` row as a negative candidate and removes no positive
#   ID from it; VinDataset then merges the two .txt lists without deduplication and
#   labels by membership in POS, so an overlapping ID counts as a positive twice.
#   New, three separate steps:
#   1. run the UNMODIFIED author script once (subprocess, recorded PYTHONHASHSEED),
#      freeze vin500neg.txt + vin500pos.txt with SHA-256, count N_dup = |NEG ∩ POS|,
#      and write manifests/reference_like_combined.csv in author loader order
#      (negative list then positive list, duplicates kept, label = 1 if id in POS).
#   2. derive E5-clean from the frozen lists: drop POS ids from NEG, and if N_dup > 0
#      fill exactly N_dup ids drawn from sorted(candidates - POS - NEG_frozen) with the
#      recorded seed; a pool that is too small is reported as a shortfall, never
#      padded to look complete. E5-clean is a separate frozen manifest.
#   3. multi-draw sensitivity is out of scope here (separate script/records).
#   check: tests/corrected/test_data.py::test_e5_clean_* / test_e5_shim_*
"""

from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import click
import pandas as pd

from sharpness_matters.config.load_config import cfg
from sharpness_matters.ptx.corrected.frozen import (
    hash_path,
    load_or_create_frozen,
    manifest_dir,
    record_path,
    repo_root,
    sha256_file,
)

AUTHOR_SCRIPT = (
    Path(__file__).resolve().parents[1] / "datasets" / "create_vin_dr_subset.py"
)
AUTHOR_OUTPUT_DIR = AUTHOR_SCRIPT.parents[1] / "output"
AUTHOR_NEG_LIST = "vin500neg.txt"
AUTHOR_POS_LIST = "vin500pos.txt"
AUTHOR_CSV_BASENAME = "train_vin.csv"

REFERENCE_DIRNAME = "reference_like"
REFERENCE_COMBINED_NAME = "reference_like_combined.csv"
CLEAN_NAME = "reference_like_clean.csv"
REFERENCE_REPORT_NAME = "reference.json"
CLEAN_REPORT_NAME = "reference_like_clean.report.json"

EXPECTED_NEG_COUNT = 594
EXPECTED_POS_COUNT = 96

# Rule recorded in cfg.ptx.corrected.e5.negative_rule and decided in
# baseline_correction.md Section 11. The code implements it literally; a config edit
# must be followed by a code edit, so the two are compared instead of evaluated.
NEGATIVE_RULE = "class_name != 'Pneumothorax'"


# --------------------------------------------------------------------------- paths


def reference_dir() -> Path:
    return manifest_dir() / REFERENCE_DIRNAME


def author_output_files() -> Tuple[Path, Path]:
    """Where the unmodified author script writes its two lists."""
    return (
        AUTHOR_OUTPUT_DIR / AUTHOR_NEG_LIST,
        AUTHOR_OUTPUT_DIR / AUTHOR_POS_LIST,
    )


# ------------------------------------------------------------------------- parsing


def parse_txt_list(path: Path) -> List[str]:
    """Parse a list exactly like the author's loader (``vin_dataset.py``).

    The author does ``text.replace("\\n", " ").split(" ")[:-1]``, so the last empty
    token produced by the trailing newline is dropped. That quirk is reproduced here;
    a file without the trailing newline (or with CRLF) would make the author's loader
    silently drop/alter an ID, so it is rejected instead.
    """
    text = Path(path).read_text(encoding="utf-8")
    if "\r" in text:
        raise RuntimeError(
            f"{path}: CRLF line endings; the author loader would keep '\\r' inside IDs"
        )
    tokens = text.replace("\n", " ").split(" ")
    if not tokens or tokens[-1] != "":
        raise RuntimeError(
            f"{path}: expected a trailing newline, as written by the author script "
            "(its loader drops the last token)"
        )
    return tokens[:-1]


def _write_sha256(path: Path) -> Path:
    target = hash_path(path)
    target.write_text(sha256_file(path) + "\n", encoding="utf-8")
    return target


def _verify_sha256(path: Path) -> None:
    target = hash_path(path)
    if not target.exists():
        raise RuntimeError(f"Frozen artifact {path} has no {target.name}")
    recorded = target.read_text(encoding="utf-8").split()[0]
    actual = sha256_file(path)
    if recorded != actual:
        raise RuntimeError(
            f"Frozen artifact hash mismatch for {path}: recorded {recorded}, "
            f"file is {actual}. Refusing to overwrite frozen data."
        )


def _copy_or_verify(source: Path, target: Path) -> None:
    """Copy ``source`` to the frozen location once; later runs only verify it."""
    source = Path(source)
    target = Path(target)
    target_hash = hash_path(target)
    state_exists = target.exists() or target_hash.exists()
    if state_exists:
        if not (target.exists() and target_hash.exists()):
            raise RuntimeError(
                f"Inconsistent frozen state for {target}: it and its .sha256 must "
                "both exist"
            )
        _verify_sha256(target)
        if sha256_file(target) != sha256_file(source):
            raise RuntimeError(
                f"Frozen copy {target} differs from the freshly produced {source}. "
                "Frozen lists are never replaced; inspect both files."
            )
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.resolve() != source.resolve():
        shutil.copyfile(source, target)
    _write_sha256(target)


def _verify_frozen_txt_copies(ref_dir: Path) -> bool:
    """True when both frozen lists and their hashes exist; raise on a partial state."""
    neg = ref_dir / AUTHOR_NEG_LIST
    pos = ref_dir / AUTHOR_POS_LIST
    arts = [neg, hash_path(neg), pos, hash_path(pos)]
    present = [p.exists() for p in arts]
    if not any(present):
        return False
    if not all(present):
        names = [p.name for p, ok in zip(arts, present) if not ok]
        raise RuntimeError(
            f"Incomplete frozen E5 reference state in {ref_dir}: missing {names}"
        )
    _verify_sha256(neg)
    _verify_sha256(pos)
    return True


# ---------------------------------------------------------------- reference-like


def build_shim(script_path: Path, annotations_csv: Path) -> str:
    """Build the ``python -c`` source that runs the author script unmodified.

    The shim patches ``pandas.read_csv`` so the author's hardcoded
    ``.../train_vin.csv`` path resolves to ``annotations_csv``, then executes the
    script with ``runpy.run_path(..., run_name="__main__")``. Only a request for a
    file named ``train_vin.csv`` is redirected; any other CSV read fails loudly
    instead of silently using the wrong annotation file. The shim also fails if the
    author script never read that CSV, so a silent no-op cannot be frozen.
    """
    script_literal = repr(str(Path(script_path).resolve()))
    csv_literal = repr(str(Path(annotations_csv).resolve()))
    basename_literal = repr(AUTHOR_CSV_BASENAME)
    return "\n".join(
        [
            "import os",
            "import runpy",
            "import pandas as pd",
            "",
            f"_AUTHOR_CSV_BASENAME = {basename_literal}",
            f"_TARGET_CSV = {csv_literal}",
            f"_SCRIPT = {script_literal}",
            "_original_read_csv = pd.read_csv",
            "_redirected = [0]",
            "",
            "def _read_csv_for_author(filepath_or_buffer, *args, **kwargs):",
            "    if os.path.basename(str(filepath_or_buffer)) == _AUTHOR_CSV_BASENAME:",
            "        _redirected[0] += 1",
            "        return _original_read_csv(_TARGET_CSV, *args, **kwargs)",
            "    return _original_read_csv(filepath_or_buffer, *args, **kwargs)",
            "",
            "pd.read_csv = _read_csv_for_author",
            'runpy.run_path(_SCRIPT, run_name="__main__")',
            "if _redirected[0] == 0:",
            "    raise SystemExit(",
            '        "shim: the author script never read train_vin.csv; refusing to freeze"',
            "    )",
            "",
        ]
    )


def _reference_combined_frame(neg: Sequence[str], pos: Sequence[str]) -> pd.DataFrame:
    """Author loader order and labelling: NEG rows then POS rows, label by POS membership."""
    pos_set = set(pos)
    rows = [(image_id, 1 if image_id in pos_set else 0, "neg") for image_id in neg]
    rows += [(image_id, 1, "pos") for image_id in pos]
    return pd.DataFrame(rows, columns=["image_id", "label", "source_list"])


def _verify_frame(path: Path, frame: pd.DataFrame, expected: pd.DataFrame) -> None:
    """Fail if a frozen csv does not hold the rows the current inputs derive."""
    got = list(zip(frame["image_id"].astype(str), frame["label"].astype(int)))
    want = list(zip(expected["image_id"].astype(str), expected["label"].astype(int)))
    if got != want:
        raise RuntimeError(
            f"Frozen manifest {path} does not match the current derivation "
            f"({len(got)} rows vs {len(want)} expected). Frozen artifacts are never "
            "rewritten; inspect the inputs or remove the freeze deliberately."
        )


def reference_combined_csv() -> Path:
    return manifest_dir() / REFERENCE_COMBINED_NAME


def clean_manifest_csv() -> Path:
    return manifest_dir() / CLEAN_NAME


def _frozen_artifacts_exist(path: Path) -> bool:
    """True when any of csv/.sha256/.record.json exists (load-first trigger)."""
    return any(p.exists() for p in (path, hash_path(path), record_path(path)))


def _load_frozen_only(path: Path) -> pd.DataFrame:
    """Load an existing frozen manifest; never build one (Section 4.2 load-first)."""

    def _refuse_build():  # pragma: no cover - only reached for a broken frozen state
        raise RuntimeError(
            f"{path} is not frozen (csv/.sha256/.record.json missing); this path only "
            "loads an existing freeze"
        )

    return load_or_create_frozen(path, _refuse_build, {})


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _reference_report_from_frame(
    frame: pd.DataFrame, ref_dir: Path, combined_csv: Path
) -> Dict[str, object]:
    """Report the frozen reference manifest without re-deriving it from the lists."""
    required = {"image_id", "label", "source_list"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(
            f"Frozen reference manifest {combined_csv} lacks columns {sorted(missing)}"
        )
    image_ids = frame["image_id"].astype(str)
    source = frame["source_list"].astype(str)
    n_neg = int((source == "neg").sum())
    n_pos = int((source == "pos").sum())
    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "loaded_from_frozen": True,
        "reference_dir": str(ref_dir),
        "combined_csv": str(combined_csv),
        "combined_csv_sha256": sha256_file(combined_csv),
        "n_neg": n_neg,
        "n_pos": n_pos,
        "n_combined_rows": int(len(frame)),
        "n_unique_ids": int(image_ids.nunique()),
        "n_dup": int(len(frame) - image_ids.nunique()),
        "expected_neg_count": EXPECTED_NEG_COUNT,
        "expected_pos_count": EXPECTED_POS_COUNT,
        "counts_match_expected": (
            n_neg == EXPECTED_NEG_COUNT and n_pos == EXPECTED_POS_COUNT
        ),
    }
    for name, txt in (
        ("neg_list_sha256", ref_dir / AUTHOR_NEG_LIST),
        ("pos_list_sha256", ref_dir / AUTHOR_POS_LIST),
    ):
        recorded = hash_path(txt)
        report[name] = (
            recorded.read_text(encoding="utf-8").split()[0]
            if recorded.exists()
            else None
        )
    return report


def freeze_reference_lists(
    neg_txt: Path,
    pos_txt: Path,
    ref_dir: Path = None,
    combined_csv: Path = None,
) -> Dict[str, object]:
    """Load the frozen reference manifest if present, else freeze the two author lists.

    Section 4.2 (load-if-exists): a frozen ``reference_like_combined.csv`` is read
    back before anything is recomputed; the author lists are only parsed when the
    manifest does not exist yet.
    """
    ref_dir = Path(ref_dir) if ref_dir is not None else reference_dir()
    combined_csv = (
        Path(combined_csv) if combined_csv is not None else reference_combined_csv()
    )

    if _frozen_artifacts_exist(combined_csv):
        # Integrity of the raw evidence only; the report is derived from the frozen
        # csv, never from re-parsing the lists or re-running the author script.
        _verify_frozen_txt_copies(ref_dir)
        report = _reference_report_from_frame(
            _load_frozen_only(combined_csv), ref_dir, combined_csv
        )
        write_json(ref_dir / REFERENCE_REPORT_NAME, report)
        return report

    frozen_neg = ref_dir / AUTHOR_NEG_LIST
    frozen_pos = ref_dir / AUTHOR_POS_LIST
    _copy_or_verify(Path(neg_txt), frozen_neg)
    _copy_or_verify(Path(pos_txt), frozen_pos)

    neg = parse_txt_list(frozen_neg)
    pos = parse_txt_list(frozen_pos)
    n_dup = len(set(neg) & set(pos))
    expected = _reference_combined_frame(neg, pos)
    record = {
        "item": "PTX-03/PTX-08 E5 reference-like (unmodified author script)",
        "author_script": str(AUTHOR_SCRIPT.name),
        "author_script_sha256": sha256_file(AUTHOR_SCRIPT),
        "author_script_run": "subprocess, unmodified, runpy.run_path(run_name='__main__')",
        "pythonhashseed": str(cfg.ptx.corrected.e5.pythonhashseed),
        "annotations_csv": str(cfg.ptx.vindr_annotations_csv),
        "n_neg": len(neg),
        "n_pos": len(pos),
        "n_dup": n_dup,
        "n_combined_rows": int(len(expected)),
        "n_unique_ids": len(set(neg) | set(pos)),
        "loader_order": "negative list then positive list; duplicates kept",
        "label_rule": "label = 1 if image_id in POS",
    }
    frame = load_or_create_frozen(combined_csv, lambda: expected, record)
    _verify_frame(combined_csv, frame, expected)

    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "loaded_from_frozen": False,
        "reference_dir": str(ref_dir),
        "combined_csv": str(combined_csv),
        "neg_list_sha256": sha256_file(frozen_neg),
        "pos_list_sha256": sha256_file(frozen_pos),
        "combined_csv_sha256": sha256_file(combined_csv),
        "n_neg": len(neg),
        "n_pos": len(pos),
        "n_combined_rows": int(len(expected)),
        "n_unique_ids": len(set(neg) | set(pos)),
        "n_dup": n_dup,
        "expected_neg_count": EXPECTED_NEG_COUNT,
        "expected_pos_count": EXPECTED_POS_COUNT,
        "counts_match_expected": (
            len(neg) == EXPECTED_NEG_COUNT and len(pos) == EXPECTED_POS_COUNT
        ),
    }
    write_json(ref_dir / REFERENCE_REPORT_NAME, report)
    return report


def run_reference_e5() -> Dict[str, object]:
    """Load the frozen reference manifest, or freeze it; run the author once only."""
    ref_dir = reference_dir()
    if _frozen_artifacts_exist(reference_combined_csv()):
        return freeze_reference_lists(
            ref_dir / AUTHOR_NEG_LIST, ref_dir / AUTHOR_POS_LIST
        )
    if _verify_frozen_txt_copies(ref_dir):
        return freeze_reference_lists(
            ref_dir / AUTHOR_NEG_LIST, ref_dir / AUTHOR_POS_LIST
        )

    out_neg, out_pos = author_output_files()
    existing = [str(p) for p in (out_neg, out_pos) if p.exists()]
    if existing:
        raise RuntimeError(
            f"Refusing to run the author script: {existing} already exist and the "
            "script would prompt for confirmation (input() in a non-interactive "
            "process). Move them away deliberately or freeze an existing pair."
        )

    annotations_csv = Path(str(cfg.ptx.vindr_annotations_csv))
    if not annotations_csv.exists():
        raise FileNotFoundError(
            f"cfg.ptx.vindr_annotations_csv does not exist: {annotations_csv}"
        )
    if not AUTHOR_SCRIPT.exists():  # pragma: no cover - ships with the repo
        raise FileNotFoundError(f"Author script not found: {AUTHOR_SCRIPT}")

    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(cfg.ptx.corrected.e5.pythonhashseed)
    try:
        subprocess.run(
            [sys.executable, "-c", build_shim(AUTHOR_SCRIPT, annotations_csv)],
            cwd=str(repo_root()),
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip().splitlines()[-5:]
        raise RuntimeError(
            "The unmodified author selector failed; nothing was frozen. "
            f"Exit code {exc.returncode}. Last stderr lines: {stderr}"
        ) from exc
    for produced in (out_neg, out_pos):
        if not produced.exists():
            raise RuntimeError(
                f"Author script finished without writing {produced}; nothing to freeze"
            )
    return freeze_reference_lists(out_neg, out_pos)


# --------------------------------------------------------------------- E5-clean


def _normalized_rule(text: str) -> str:
    return "".join(str(text).split()).replace('"', "'")


def verify_negative_rule() -> str:
    """Raise unless cfg.ptx.corrected.e5.negative_rule is the rule implemented here."""
    rule = str(cfg.ptx.corrected.e5.negative_rule)
    if _normalized_rule(rule) != _normalized_rule(NEGATIVE_RULE):
        raise RuntimeError(
            f"cfg.ptx.corrected.e5.negative_rule is {rule!r} but this module implements "
            f"{NEGATIVE_RULE!r}; change the code together with the config"
        )
    return NEGATIVE_RULE


def candidate_negative_ids(annotations_csv: Path) -> List[str]:
    """Negative candidate pool: image IDs of rows with ``class_name != Pneumothorax``."""
    verify_negative_rule()
    frame = pd.read_csv(annotations_csv)
    missing = {"image_id", "class_name"} - set(frame.columns)
    if missing:
        raise KeyError(f"{annotations_csv} is missing columns {sorted(missing)}")
    mask = frame["class_name"].astype(str) != "Pneumothorax"
    return frame.loc[mask, "image_id"].astype(str).tolist()


def build_e5_clean(
    neg_frozen: Sequence[str],
    pos: Sequence[str],
    candidate_neg_ids: Sequence[str],
    seed: int,
) -> Tuple[List[str], Dict[str, object]]:
    """Derive the E5-clean negative list from the frozen reference-like list.

    Rules (baseline_correction.md Section 4.2, item 2):
      * ``NEG_remaining = NEG_frozen - POS`` keeps the frozen order and membership;
      * ``N_dup = 0`` -> nothing is removed and nothing is added (identity);
      * ``N_dup > 0`` -> exactly ``N_dup`` IDs are drawn from
        ``sorted(candidates - POS - NEG_frozen)`` with ``random.Random(seed)``; a pool
        that is too small leaves a reported shortfall instead of a padded list;
      * the result is unique and disjoint from POS, and contains ``NEG_frozen - POS``.
    """
    neg_frozen = list(neg_frozen)
    pos = list(pos)
    candidates = list(candidate_neg_ids)
    if len(set(pos)) != len(pos):
        raise RuntimeError("Frozen POS list contains duplicate IDs; refusing to label")
    pos_set = set(pos)
    neg_frozen_set = set(neg_frozen)
    if len(neg_frozen_set) != len(neg_frozen):
        raise RuntimeError("Frozen NEG list contains duplicate IDs; refusing to derive")
    n_dup = len(neg_frozen_set & pos_set)

    neg_remaining = [image_id for image_id in neg_frozen if image_id not in pos_set]
    filled: List[str] = []
    pool: List[str] = []
    shortfall = 0
    if n_dup:
        pool = sorted(set(candidates) - pos_set - neg_frozen_set)
        take = min(n_dup, len(pool))
        filled = random.Random(int(seed)).sample(pool, take)
        shortfall = n_dup - take

    neg_clean = neg_remaining + filled
    checks = {
        "disjoint_from_pos": not (set(neg_clean) & pos_set),
        "unique": len(set(neg_clean)) == len(neg_clean),
        "kept_neg_frozen_minus_pos": set(neg_remaining).issubset(set(neg_clean)),
        "count_complete": len(neg_clean) == len(neg_frozen),
    }
    hard_failures = [name for name in ("disjoint_from_pos", "unique", "kept_neg_frozen_minus_pos") if not checks[name]]
    if hard_failures:
        raise RuntimeError(
            f"E5-clean invariants violated: {hard_failures}; neg_clean={len(neg_clean)} "
            f"pos={len(pos)} n_dup={n_dup}"
        )
    report = {
        "seed": int(seed),
        "n_dup": n_dup,
        "neg_frozen": len(neg_frozen),
        "neg_after_removal": len(neg_remaining),
        "filled": len(filled),
        "pool_size": len(pool),
        "shortfall": shortfall,
        "complete": shortfall == 0,
        "target_neg": len(neg_frozen),
        "neg_clean": len(neg_clean),
        "pos": len(pos),
        "checks": checks,
    }
    return neg_clean, report


def _clean_manifest_frame(neg_clean: Sequence[str], pos: Sequence[str]) -> pd.DataFrame:
    rows = [(image_id, 0) for image_id in neg_clean]
    rows += [(image_id, 1) for image_id in pos]
    return pd.DataFrame(rows, columns=["image_id", "label"])


def _split_reference_lists(
    combined: pd.DataFrame,
) -> Tuple[List[str], List[str]]:
    """Recover the frozen NEG and POS lists (author order) from the combined manifest."""
    required = {"image_id", "label", "source_list"}
    missing = required - set(combined.columns)
    if missing:
        raise RuntimeError(
            f"Frozen reference manifest lacks columns {sorted(missing)}; run the "
            "`reference` subcommand to rebuild it from the author lists"
        )
    source = combined["source_list"].astype(str)
    image_ids = combined["image_id"].astype(str)
    neg = image_ids[source == "neg"].tolist()
    pos = image_ids[source == "pos"].tolist()
    return neg, pos


def _clean_report_from_frame(frame: pd.DataFrame, path: Path) -> Dict[str, object]:
    """Report the frozen E5-clean manifest without re-deriving or sampling it."""
    missing = {"image_id", "label"} - set(frame.columns)
    if missing:
        raise RuntimeError(f"Frozen clean manifest {path} lacks columns {sorted(missing)}")
    labels = frame["label"].astype(int)
    image_ids = frame["image_id"].astype(str)
    neg_ids = image_ids[labels == 0].tolist()
    pos_ids = image_ids[labels == 1].tolist()
    report: Dict[str, object] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "loaded_from_frozen": True,
        "manifest": str(path),
        "manifest_sha256": sha256_file(path),
        "neg_list": str(reference_dir() / AUTHOR_NEG_LIST),
        "pos_list": str(reference_dir() / AUTHOR_POS_LIST),
        "n_neg": len(neg_ids),
        "neg_clean": len(neg_ids),
        "n_pos": len(pos_ids),
        "pos": len(pos_ids),
        "checks": {
            "disjoint_from_pos": not (set(neg_ids) & set(pos_ids)),
            "unique": len(set(neg_ids)) == len(neg_ids),
        },
    }
    recorded = record_path(path)
    if recorded.exists():
        data = json.loads(recorded.read_text(encoding="utf-8"))
        for key in (
            "seed",
            "n_dup",
            "pool_size",
            "filled",
            "shortfall",
            "complete",
            "target_neg",
            "annotations_csv",
            "reference_counts_match_expected",
        ):
            if key in data:
                report[key] = data[key]
        if "target_neg" in data:
            report["checks"]["count_complete"] = len(neg_ids) == int(data["target_neg"])
    return report


def run_clean_e5(seed: int = None) -> Dict[str, object]:
    """Load the frozen clean manifest if present, else derive and freeze it.

    Section 4.2 (load-if-exists): ``reference_like_clean.csv`` + ``.sha256`` are
    checked first; when they match, the manifest is returned without reading the
    annotation CSV and without drawing anything. Only the absent-manifest path
    reads the candidate pool and calls ``build_e5_clean``.
    """
    path = clean_manifest_csv()
    if _frozen_artifacts_exist(path):
        report = _clean_report_from_frame(_load_frozen_only(path), path)
        write_json(path.parent / CLEAN_REPORT_NAME, report)
        return report

    seed = (
        int(cfg.ptx.corrected.e5.clean_fill_seed) if seed is None else int(seed)
    )
    combined_csv = reference_combined_csv()
    if not _frozen_artifacts_exist(combined_csv):
        raise RuntimeError(
            f"Frozen reference manifest {combined_csv} is missing; run the "
            "`reference` subcommand first (E5-clean is derived from it)"
        )
    neg_frozen, pos = _split_reference_lists(_load_frozen_only(combined_csv))

    annotations_csv = Path(str(cfg.ptx.vindr_annotations_csv))
    candidates = candidate_negative_ids(annotations_csv)
    neg_clean, report = build_e5_clean(neg_frozen, pos, candidates, seed)
    expected = _clean_manifest_frame(neg_clean, pos)

    record = {
        "item": "PTX-03/PTX-08 E5-clean negative manifest",
        "derived_from": str(combined_csv),
        "derived_from_sha256": sha256_file(combined_csv),
        "annotations_csv": str(annotations_csv),
        "annotations_csv_sha256": sha256_file(annotations_csv),
        "negative_rule": NEGATIVE_RULE,
        "fill_rule": "N_dup ids from sorted(candidates - POS - NEG_frozen) via random.Random(seed).sample",
        "seed": seed,
        "n_dup": report["n_dup"],
        "pool_size": report["pool_size"],
        "filled": report["filled"],
        "shortfall": report["shortfall"],
        "complete": report["complete"],
        "target_neg": report["target_neg"],
        "n_neg": len(neg_clean),
        "n_pos": len(pos),
        "reference_counts_match_expected": (
            len(neg_frozen) == EXPECTED_NEG_COUNT and len(pos) == EXPECTED_POS_COUNT
        ),
    }
    frame = load_or_create_frozen(path, lambda: expected, record)
    _verify_frame(path, frame, expected)

    report = dict(report)
    report.update(
        {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "loaded_from_frozen": False,
            "manifest": str(path),
            "manifest_sha256": sha256_file(path),
            "n_neg": len(neg_clean),
            "n_pos": len(pos),
            "neg_list": str(reference_dir() / AUTHOR_NEG_LIST),
            "pos_list": str(reference_dir() / AUTHOR_POS_LIST),
            "annotations_csv": str(annotations_csv),
            "reference_counts_match_expected": (
                len(neg_frozen) == EXPECTED_NEG_COUNT
                and len(pos) == EXPECTED_POS_COUNT
            ),
        }
    )
    write_json(path.parent / CLEAN_REPORT_NAME, report)
    return report


# -------------------------------------------------------------------------- CLI


@click.group()
def cli() -> None:
    """Build the E5 (reference-like) and E5-clean VinDr manifests."""


@cli.command("reference")
def reference_command() -> None:
    """Run the unmodified author selector once and freeze both lists."""
    click.echo(json.dumps(run_reference_e5(), indent=2, sort_keys=True, default=str))


@cli.command("clean")
@click.option(
    "--seed",
    type=int,
    default=None,
    help="Fill seed for N_dup replacements (default: e5.clean_fill_seed from config).",
)
def clean_command(seed: int) -> None:
    """Derive the E5-clean manifest from the frozen reference-like lists."""
    click.echo(json.dumps(run_clean_e5(seed), indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    cli()

"""Stage-2 (B) pre-registered paired statistics: plan, lock, run.

# B (PTX-05 / Section 8.1): current -> the author's `p_value_saliency.py` compared the
#   `values()` lists of two dicts: equal length was the only pairing evidence, the test
#   (t-test / Wilcoxon / Mann-Whitney) was chosen from the sample, and image identity was
#   discarded. Problem -> a paired test on positionally aligned lists cannot prove which
#   image pair it scored. New -> every comparison is defined in a plan that names the two
#   ID-keyed CSVs; rows are aligned by `image_id` with `align_by_id` (duplicate IDs, a
#   one-sided ID and a classification label disagreement raise), the paired difference is
#   the registered `b - a`, the zero rule is the registered `zero_method`, and the test is
#   chosen by `kind`, never by length or normality. Uncertainty is bootstrapped over
#   images (registered `n`/`seed`).
#   check: tests/corrected/test_analysis.py::test_align_by_id_rejects_duplicate_ids,
#   ::test_align_by_id_rejects_different_id_sets,
#   ::test_row_order_does_not_change_results_and_hash_is_stamped,
#   ::test_registered_zero_method_is_used
# B (PTX-22 / Section 8.2): current -> the classification code applied BH inside each
#   model x dataset block of 15 tests while the saliency code pooled both models into 30
#   tests, so the family boundary was an accident of the call site. Problem -> `p_bh`
#   depended on which comparisons happened to share a loop, and a missing input shrank
#   the family silently. New -> the family is declared in the plan (`families[].name`,
#   its exact comparison list, `correction: "bh"`, `alpha`); `multipletests(method=
#   "fdr_bh")` runs once per family over exactly those comparisons, and any comparison
#   that cannot be computed raises before anything is written.
#   check: ::test_bh_matches_hand_computation,
#   ::test_missing_input_raises_and_writes_nothing
# B (PTX-11 / Section 8.3): current -> precision/coverage/IoU were pooled as "explanation
#   quality", mixing localization with claims about the model. Problem -> the three XAI
#   questions are separate endpoints. New -> every comparison carries `endpoint_class` in
#   {"classification", "localization", "sanity", "faithfulness"}; a family must be
#   endpoint-homogeneous (validation error otherwise), so correction and reporting never
#   merge endpoints. `classification_auroc` is a performance endpoint and must use
#   `classification`; `saliency_paired` must use one of the three XAI endpoints, so XAI
#   sanity is never conflated with the classifier's AUROC. The XAI stage locks the plan
#   before scoring.
#   check: ::test_mixed_endpoint_classes_rejected,
#   ::test_classification_auroc_requires_classification_endpoint

Plan schema (JSON, all keys validated; unknown keys are errors):
    {
      "plan_version": 1,
      "description": str,                     # optional
      "created_utc": str,                     # optional
      "families": [
        {"name": str, "correction": "bh", "alpha": float,
         "endpoint_class": str,              # optional; must match the comparisons
         "description": str,                 # optional
         "comparisons": [
           {"id": str,
            "kind": "saliency_paired" | "classification_auroc",
            "endpoint_class": "classification" | "localization" | "sanity" | "faithfulness",
            "description": str,              # optional
            "inputs": {"a": path, "b": path,
                       # saliency_paired only, required:
                       "metric_column": str,
                       "threshold": {"column": str, "value": number},
                       # classification_auroc only, optional:
                       "prob_column": "prob", "label_column": "label"},
            "cohort": {"description": str,   # optional
                       "where": {column: value | [values]}},
            "paired_difference": "b - a",
            "zero_method": "wilcox" | "pratt" | "zsplit",  # saliency only, required
            "uncertainty_unit": "image",
            "bootstrap": {"n": int, "seed": int}}
         ]}
      ]
    }

Relative input paths are resolved against the plan file's directory, so a plan and its
inputs travel together. `lock_plan` writes `<plan>.sha256` (and validates first);
`run_plan` refuses a plan whose lock file is missing or does not match the current bytes,
and stamps the SHA-256 into every output row and JSON document.
"""

from __future__ import annotations

import json
import os
import platform
import re
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy
import sklearn
import statsmodels
from MLstatkit.stats import Delong_test
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests

from sharpness_matters.ptx.corrected.frozen import hash_path, sha256_file

PLAN_VERSION = 1
CLASSIFICATION_ENDPOINT = "classification"
XAI_ENDPOINT_CLASSES = ("localization", "sanity", "faithfulness")
ENDPOINT_CLASSES = (CLASSIFICATION_ENDPOINT,) + XAI_ENDPOINT_CLASSES
SALIENCY_KIND = "saliency_paired"
CLASSIFICATION_KIND = "classification_auroc"
KINDS = (SALIENCY_KIND, CLASSIFICATION_KIND)
ZERO_METHODS = ("wilcox", "pratt", "zsplit")
CORRECTION_METHODS = ("bh",)
PAIRED_DIFFERENCE = "b - a"
UNCERTAINTY_UNITS = ("image",)
CI_LEVEL = 0.95
_AUROC_EQUAL_ATOL = 1e-12
OUTPUT_COLUMNS = [
    "comparison_id",
    "endpoint_class",
    "n",
    "estimate",
    "ci_low",
    "ci_high",
    "statistic",
    "p_raw",
    "p_bh",
    "reject",
    "plan_sha256",
]
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


# --------------------------------------------------------------------------- plan validation


def _fail(where: str, message: str) -> None:
    raise ValueError(f"{where}: {message}")


def _check_keys(
    mapping: Any,
    where: str,
    allowed: Sequence[str],
    required: Sequence[str],
) -> None:
    if not isinstance(mapping, dict):
        _fail(where, f"must be a JSON object, got {type(mapping).__name__}")
    unknown = sorted(set(mapping) - set(allowed))
    if unknown:
        _fail(where, f"unknown key(s) {unknown}; allowed {sorted(allowed)}")
    missing = sorted(set(required) - set(mapping))
    if missing:
        _fail(where, f"missing key(s) {missing}")


def _as_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(where, f"must be an integer, got {value!r}")
    return int(value)


def _as_number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(where, f"must be a number, got {value!r}")
    return float(value)


def _as_text(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(where, f"must be a non-empty string, got {value!r}")
    return value


def _validate_cohort(cohort: Any, where: str) -> Dict[str, Any]:
    if cohort is None:
        return {"description": None, "where": {}}
    _check_keys(cohort, where, allowed=("description", "where"), required=())
    description = cohort.get("description")
    if description is not None:
        description = _as_text(description, f"{where}.description")
    where_map = cohort.get("where", {})
    if not isinstance(where_map, dict):
        _fail(f"{where}.where", f"must be an object, got {type(where_map).__name__}")
    for column, value in where_map.items():
        if not isinstance(column, str) or not column:
            _fail(f"{where}.where", f"column names must be non-empty strings, got {column!r}")
        values = value if isinstance(value, list) else [value]
        if not values:
            _fail(f"{where}.where.{column}", "must not be an empty list")
        for item in values:
            if isinstance(item, (dict, list)) or item is None:
                _fail(
                    f"{where}.where.{column}",
                    f"only scalar or list-of-scalar filters are supported, got {item!r}",
                )
    return {"description": description, "where": dict(where_map)}


def _validate_inputs(inputs: Any, kind: str, where: str) -> Dict[str, Any]:
    saliency = kind == SALIENCY_KIND
    allowed = ["a", "b"]
    required = ["a", "b"]
    if saliency:
        allowed += ["metric_column", "threshold"]
        required += ["metric_column", "threshold"]
    else:
        allowed += ["prob_column", "label_column"]
    _check_keys(inputs, where, allowed=allowed, required=required)
    normalized: Dict[str, Any] = {
        "a": _as_text(inputs["a"], f"{where}.a"),
        "b": _as_text(inputs["b"], f"{where}.b"),
    }
    if saliency:
        normalized["metric_column"] = _as_text(
            inputs["metric_column"], f"{where}.metric_column"
        )
        threshold = inputs["threshold"]
        _check_keys(
            threshold,
            f"{where}.threshold",
            allowed=("column", "value"),
            required=("column", "value"),
        )
        normalized["threshold"] = {
            "column": _as_text(threshold["column"], f"{where}.threshold.column"),
            "value": _as_number(threshold["value"], f"{where}.threshold.value"),
        }
    else:
        normalized["prob_column"] = _as_text(
            inputs.get("prob_column", "prob"), f"{where}.prob_column"
        )
        normalized["label_column"] = _as_text(
            inputs.get("label_column", "label"), f"{where}.label_column"
        )
    return normalized


def _validate_comparison(comparison: Any, where: str) -> Dict[str, Any]:
    allowed = [
        "id",
        "kind",
        "endpoint_class",
        "description",
        "inputs",
        "cohort",
        "paired_difference",
        "zero_method",
        "uncertainty_unit",
        "bootstrap",
    ]
    required = [
        "id",
        "kind",
        "endpoint_class",
        "inputs",
        "paired_difference",
        "uncertainty_unit",
        "bootstrap",
    ]
    _check_keys(comparison, where, allowed=allowed, required=required)

    comparison_id = _as_text(comparison["id"], f"{where}.id")
    if not _NAME_RE.match(comparison_id):
        _fail(
            f"{where}.id",
            f"{comparison_id!r} must match {_NAME_RE.pattern} (used in output files)",
        )
    kind = comparison["kind"]
    if kind not in KINDS:
        _fail(f"{where}.kind", f"must be one of {list(KINDS)}, got {kind!r}")
    endpoint_class = comparison["endpoint_class"]
    if endpoint_class not in ENDPOINT_CLASSES:
        _fail(
            f"{where}.endpoint_class",
            f"must be one of {list(ENDPOINT_CLASSES)}, got {endpoint_class!r}",
        )
    if kind == CLASSIFICATION_KIND and endpoint_class != CLASSIFICATION_ENDPOINT:
        _fail(
            f"{where}.endpoint_class",
            f"classification_auroc is a performance endpoint and must use "
            f"{CLASSIFICATION_ENDPOINT!r}, got {endpoint_class!r}; XAI sanity is a "
            "separate endpoint",
        )
    if kind == SALIENCY_KIND and endpoint_class == CLASSIFICATION_ENDPOINT:
        _fail(
            f"{where}.endpoint_class",
            f"saliency_paired must use one of {list(XAI_ENDPOINT_CLASSES)}, got "
            f"{CLASSIFICATION_ENDPOINT!r}",
        )
    description = comparison.get("description")
    if description is not None:
        description = _as_text(description, f"{where}.description")

    inputs = _validate_inputs(comparison["inputs"], kind, f"{where}.inputs")
    cohort = _validate_cohort(comparison.get("cohort"), f"{where}.cohort")

    paired_difference = comparison["paired_difference"]
    if paired_difference != PAIRED_DIFFERENCE:
        _fail(
            f"{where}.paired_difference",
            f"the registered definition must be {PAIRED_DIFFERENCE!r}, got {paired_difference!r}",
        )
    uncertainty_unit = comparison["uncertainty_unit"]
    if uncertainty_unit not in UNCERTAINTY_UNITS:
        _fail(
            f"{where}.uncertainty_unit",
            f"must be one of {list(UNCERTAINTY_UNITS)}, got {uncertainty_unit!r} "
            "(patient-level units need a patient ID column, absent from the per-image CSVs)",
        )

    if kind == SALIENCY_KIND:
        if "zero_method" not in comparison:
            _fail(f"{where}.zero_method", "is required for saliency_paired")
        zero_method = comparison["zero_method"]
        if zero_method not in ZERO_METHODS:
            _fail(
                f"{where}.zero_method",
                f"must be one of {list(ZERO_METHODS)}, got {zero_method!r}",
            )
    else:
        if "zero_method" in comparison:
            _fail(
                f"{where}.zero_method",
                "is not applicable to classification_auroc (DeLong uses the raw scores)",
            )
        zero_method = None

    bootstrap = comparison["bootstrap"]
    _check_keys(
        bootstrap,
        f"{where}.bootstrap",
        allowed=("n", "seed"),
        required=("n", "seed"),
    )
    n_boot = _as_int(bootstrap["n"], f"{where}.bootstrap.n")
    if n_boot < 1:
        _fail(f"{where}.bootstrap.n", f"must be >= 1, got {n_boot}")
    seed = _as_int(bootstrap["seed"], f"{where}.bootstrap.seed")
    if seed < 0:
        _fail(f"{where}.bootstrap.seed", f"must be >= 0, got {seed}")

    return {
        "id": comparison_id,
        "kind": kind,
        "endpoint_class": endpoint_class,
        "description": description,
        "inputs": inputs,
        "cohort": cohort,
        "paired_difference": paired_difference,
        "zero_method": zero_method,
        "uncertainty_unit": uncertainty_unit,
        "bootstrap": {"n": n_boot, "seed": seed},
    }


def validate_plan(plan: Any) -> Dict[str, Any]:
    """Validate a plan object and return a normalized copy (defaults filled in).

    Unknown keys are rejected at every level so a typo cannot silently drop a registered
    comparison or a registered setting.
    """
    _check_keys(
        plan,
        "plan",
        allowed=("plan_version", "description", "created_utc", "families"),
        required=("plan_version", "families"),
    )
    version = _as_int(plan["plan_version"], "plan.plan_version")
    if version != PLAN_VERSION:
        _fail("plan.plan_version", f"must be {PLAN_VERSION}, got {version}")

    description = plan.get("description")
    if description is not None:
        description = _as_text(description, "plan.description")
    created_utc = plan.get("created_utc")
    if created_utc is not None:
        created_utc = _as_text(created_utc, "plan.created_utc")

    families = plan["families"]
    if not isinstance(families, list) or not families:
        _fail("plan.families", "must be a non-empty list")

    normalized_families: List[Dict[str, Any]] = []
    seen_names = set()
    seen_ids = set()
    for family_index, family in enumerate(families):
        where = f"plan.families[{family_index}]"
        _check_keys(
            family,
            where,
            allowed=("name", "correction", "alpha", "endpoint_class", "description", "comparisons"),
            required=("name", "correction", "alpha", "comparisons"),
        )
        name = _as_text(family["name"], f"{where}.name")
        if not _NAME_RE.match(name):
            _fail(f"{where}.name", f"{name!r} must match {_NAME_RE.pattern}")
        if name in seen_names:
            _fail(f"{where}.name", f"duplicate family name {name!r}")
        seen_names.add(name)

        correction = family["correction"]
        if correction not in CORRECTION_METHODS:
            _fail(
                f"{where}.correction",
                f"must be one of {list(CORRECTION_METHODS)}, got {correction!r}",
            )
        alpha = _as_number(family["alpha"], f"{where}.alpha")
        if not 0.0 < alpha < 1.0:
            _fail(f"{where}.alpha", f"must be in (0, 1), got {alpha}")

        family_class = family.get("endpoint_class")
        if family_class is not None and family_class not in ENDPOINT_CLASSES:
            _fail(
                f"{where}.endpoint_class",
                f"must be one of {list(ENDPOINT_CLASSES)}, got {family_class!r}",
            )
        family_description = family.get("description")
        if family_description is not None:
            family_description = _as_text(family_description, f"{where}.description")

        comparisons = family["comparisons"]
        if not isinstance(comparisons, list) or not comparisons:
            _fail(f"{where}.comparisons", "must be a non-empty list")
        normalized_comparisons = []
        classes = set()
        for comparison_index, comparison in enumerate(comparisons):
            normalized = _validate_comparison(
                comparison, f"{where}.comparisons[{comparison_index}]"
            )
            if normalized["id"] in seen_ids:
                _fail(
                    f"{where}.comparisons[{comparison_index}].id",
                    f"duplicate comparison id {normalized['id']!r} in the plan",
                )
            seen_ids.add(normalized["id"])
            classes.add(normalized["endpoint_class"])
            normalized_comparisons.append(normalized)
        if len(classes) != 1:
            _fail(
                where,
                f"a family must not mix endpoint classes, got {sorted(classes)} "
                "(classification, localization, sanity and faithfulness are separate "
                "endpoints)",
            )
        endpoint_class = classes.pop()
        if family_class is not None and family_class != endpoint_class:
            _fail(
                f"{where}.endpoint_class",
                f"declares {family_class!r} but its comparisons are {endpoint_class!r}",
            )

        normalized_families.append(
            {
                "name": name,
                "correction": correction,
                "alpha": alpha,
                "endpoint_class": endpoint_class,
                "description": family_description,
                "comparisons": normalized_comparisons,
            }
        )

    return {
        "plan_version": PLAN_VERSION,
        "description": description,
        "created_utc": created_utc,
        "families": normalized_families,
    }


# --------------------------------------------------------------------------- plan locking


def load_plan(path: Path) -> Dict[str, Any]:
    """Read a plan JSON file (UTF-8) and return the raw object."""
    plan_path = Path(path)
    if not plan_path.is_file():
        raise FileNotFoundError(f"analysis plan not found: {plan_path}")
    with open(plan_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def plan_sha256(path: Path) -> str:
    """SHA-256 of the plan file's current bytes."""
    return sha256_file(Path(path))


def lock_plan(path: Path) -> str:
    """Validate a plan and write `<plan>.sha256`; return the digest.

    Locking an unchanged plan is idempotent. Re-locking a plan whose bytes changed after
    the first lock raises: a pre-registered family is never silently redefined.
    """
    plan_path = Path(path)
    plan = load_plan(plan_path)
    validate_plan(plan)
    digest = sha256_file(plan_path)
    target = hash_path(plan_path)
    if target.exists():
        recorded = target.read_text(encoding="utf-8").strip().lower()
        if recorded != digest:
            raise RuntimeError(
                f"{plan_path} changed after it was locked: recorded {recorded}, "
                f"current {digest}. A locked plan is immutable; register a new plan file."
            )
        return digest
    target.write_text(digest + "\n", encoding="utf-8")
    return digest


def require_locked(path: Path) -> str:
    """Return the plan digest only if `<plan>.sha256` exists and matches the file."""
    plan_path = Path(path)
    if not plan_path.is_file():
        raise FileNotFoundError(f"analysis plan not found: {plan_path}")
    target = hash_path(plan_path)
    if not target.exists():
        raise RuntimeError(
            f"analysis plan is not locked: {target} is missing. Run "
            f"lock_plan({str(plan_path)!r}) before running the plan."
        )
    recorded = target.read_text(encoding="utf-8").strip().lower()
    current = sha256_file(plan_path)
    if recorded != current:
        raise RuntimeError(
            f"analysis plan hash mismatch: {target} records {recorded!r} but "
            f"{plan_path} hashes to {current!r}. The registered plan was modified."
        )
    return current


# --------------------------------------------------------------------------- ID alignment


def align_by_id(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    key: str = "image_id",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Align two ID-keyed frames by `key`; never by position.

    Raises `ValueError` when either side has duplicate IDs or the ID sets differ, with
    counts and examples. Returned frames are ordered by the sorted ID (identical order on
    both sides), so row order in the input CSVs cannot affect the result.
    """
    for side, frame in (("a", df_a), ("b", df_b)):
        if key not in frame.columns:
            raise ValueError(
                f"side {side}: ID column {key!r} is missing "
                f"(columns: {list(frame.columns)})"
            )
    a = df_a.copy()
    b = df_b.copy()
    a[key] = a[key].astype(str)
    b[key] = b[key].astype(str)

    for side, frame in (("a", a), ("b", b)):
        duplicated = frame[key][frame[key].duplicated(keep=False)]
        if len(duplicated):
            examples = sorted(set(duplicated))[:5]
            raise ValueError(
                f"side {side}: {key!r} is not unique: {duplicated.nunique()} duplicated "
                f"ID(s) over {len(duplicated)} row(s); examples {examples}"
            )

    ids_a = set(a[key])
    ids_b = set(b[key])
    only_a = sorted(ids_a - ids_b)
    only_b = sorted(ids_b - ids_a)
    if only_a or only_b:
        raise ValueError(
            f"{key!r} sets differ: n_a={len(ids_a)}, n_b={len(ids_b)}, "
            f"{len(only_a)} only in a {only_a[:5]}, {len(only_b)} only in b {only_b[:5]}"
        )

    order = sorted(ids_a)
    aligned_a = a.set_index(key).loc[order].reset_index()
    aligned_b = b.set_index(key).loc[order].reset_index()
    return aligned_a, aligned_b


# --------------------------------------------------------------------------- statistics helpers


def _zero_and_tie_counts(differences: np.ndarray) -> Tuple[int, int]:
    """(number of exactly-zero differences, number of observations with tied |diff|).

    Ties are counted among the non-zero differences only, so zeros are reported once in
    `n_zero_diffs` and never again as a tie group.
    """
    diff = np.asarray(differences, dtype=float)
    n_zero = int(np.sum(diff == 0.0))
    nonzero = np.abs(diff[diff != 0.0])
    if nonzero.size == 0:
        return n_zero, 0
    _, counts = np.unique(nonzero, return_counts=True)
    return n_zero, int(counts[counts > 1].sum())


def _percentile_ci(values: np.ndarray) -> Tuple[float, float]:
    tail = (1.0 - CI_LEVEL) / 2.0 * 100.0
    low, high = np.percentile(values, [tail, 100.0 - tail])
    return float(low), float(high)


def bootstrap_mean_median_ci(
    differences: np.ndarray, n_boot: int, seed: int
) -> Dict[str, Any]:
    """Percentile bootstrap over images of the mean and the median paired difference.

    Every draw resamples image differences with replacement (the registered uncertainty
    unit is the image), using the registered `n` and `seed`.
    """
    diff = np.asarray(differences, dtype=float)
    if diff.size == 0:
        raise ValueError("bootstrap needs at least one paired difference")
    rng = np.random.default_rng(seed)
    n = diff.size
    means = np.empty(n_boot, dtype=float)
    medians = np.empty(n_boot, dtype=float)
    for index in range(n_boot):
        sample = diff[rng.integers(0, n, size=n)]
        means[index] = sample.mean()
        medians[index] = np.median(sample)
    mean_low, mean_high = _percentile_ci(means)
    median_low, median_high = _percentile_ci(medians)
    return {
        "n": int(n_boot),
        "n_used": int(n_boot),
        "ci_mean_low": mean_low,
        "ci_mean_high": mean_high,
        "ci_median_low": median_low,
        "ci_median_high": median_high,
    }


def bootstrap_auroc_diff_ci(
    labels: np.ndarray,
    prob_a: np.ndarray,
    prob_b: np.ndarray,
    n_boot: int,
    seed: int,
) -> Dict[str, Any]:
    """Percentile bootstrap over images of AUROC(b) - AUROC(a).

    A resample that contains one class only has no AUROC; it is skipped and counted in
    `n_used` (never hidden). No valid resample raises.
    """
    y = np.asarray(labels, dtype=int)
    pa = np.asarray(prob_a, dtype=float)
    pb = np.asarray(prob_b, dtype=float)
    rng = np.random.default_rng(seed)
    n = y.size
    differences = []
    for _ in range(n_boot):
        index = rng.integers(0, n, size=n)
        sample = y[index]
        if sample.min() == sample.max():
            continue
        differences.append(
            roc_auc_score(sample, pb[index]) - roc_auc_score(sample, pa[index])
        )
    if not differences:
        raise RuntimeError(
            "AUROC bootstrap produced no resample with both classes; "
            "the cohort is too small or too imbalanced for the registered bootstrap"
        )
    array = np.asarray(differences, dtype=float)
    low, high = _percentile_ci(array)
    return {
        "n": int(n_boot),
        "n_used": int(array.size),
        "ci_low": low,
        "ci_high": high,
    }


def delong_paired(
    labels: np.ndarray,
    prob_a: np.ndarray,
    prob_b: np.ndarray,
) -> Tuple[float, float, bool]:
    """Paired DeLong test on aligned rows; returns (|z|, p, degenerate).

    The pinned MLstatkit `Delong_test(true, prob_A, prob_B)` returns `(z, p)` and is the
    only implementation used (PTX-24 records the version in the run JSON). The installed
    build returns the negated absolute z, so the reported statistic is `|z|` (a two-sided
    test has no meaningful sign).

    Non-finite output is split by the observed AUROC difference, never collapsed:
    - AUROC_A == AUROC_B (within 1e-12) -> the zero difference has zero variance; the
      limit is recorded explicitly as `statistic=0.0, p_raw=1.0, degenerate=True`
      (identical predictions or rank-equivalent scores), instead of a NaN reaching BH.
    - AUROC_A != AUROC_B -> a non-finite/zero-variance statistic means the variance
      estimate collapsed while the effect is nonzero (e.g. perfect separation with
      reversed scores); inference is undefined and this raises ValueError. A valid
      p-value is never replaced by 1.
    """
    y = np.asarray(labels)
    pa = np.asarray(prob_a, dtype=float)
    pb = np.asarray(prob_b, dtype=float)
    if y.ndim != 1 or pa.shape != y.shape or pb.shape != y.shape:
        raise ValueError(
            f"shape mismatch: labels {y.shape}, prob_a {pa.shape}, prob_b {pb.shape}"
        )
    if not np.all(np.isin(y, (0, 1))):
        raise ValueError("labels must be binary 0/1")
    if np.unique(y).size != 2:
        raise ValueError("DeLong needs both classes present")
    if not (np.all(np.isfinite(pa)) and np.all(np.isfinite(pb))):
        raise ValueError("probabilities must be finite")

    result = Delong_test(y, pa, pb)
    if not isinstance(result, (tuple, list)) or len(result) != 2:
        raise RuntimeError(
            "MLstatkit.Delong_test must return (z, p); got "
            f"{len(result) if isinstance(result, (tuple, list)) else type(result)} "
            "values. The installed build changed its return arity (PTX-24): pin the "
            "version before running a locked plan."
        )
    z_value, p_value = float(result[0]), float(result[1])
    if not np.isfinite(z_value) or not np.isfinite(p_value):
        auroc_a = float(roc_auc_score(y, pa))
        auroc_b = float(roc_auc_score(y, pb))
        if abs(auroc_a - auroc_b) <= _AUROC_EQUAL_ATOL:
            return 0.0, 1.0, True
        raise ValueError(
            "DeLong variance is zero with a nonzero AUROC difference; inference "
            f"undefined (AUROC_A={auroc_a}, AUROC_B={auroc_b}, z={z_value}, p={p_value})"
        )
    return abs(z_value), p_value, False


def benjamini_hochberg(
    p_values: Sequence[float], alpha: float
) -> Tuple[List[bool], List[float]]:
    """Benjamini-Hochberg FDR over exactly the supplied family; returns (reject, p_bh)."""
    values = np.asarray(p_values, dtype=float)
    if values.size == 0:
        raise ValueError("Benjamini-Hochberg needs at least one p-value")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"p-values must be finite, got {values.tolist()}")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError(f"p-values must lie in [0, 1], got {values.tolist()}")
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    reject, adjusted, _, _ = multipletests(values, alpha=float(alpha), method="fdr_bh")
    return [bool(item) for item in reject], [float(item) for item in adjusted]


# --------------------------------------------------------------------------- comparison inputs


def _resolve_input(path_text: str, plan_dir: Path) -> Path:
    candidate = Path(path_text)
    if candidate.is_absolute():
        return candidate
    return plan_dir / candidate


def _read_input(path: Path, comparison_id: str, side: str) -> pd.DataFrame:
    if not path.is_file():
        raise RuntimeError(
            f"comparison {comparison_id!r}: input {side} not found: {path} "
            "(a registered comparison is never dropped)"
        )
    try:
        frame = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001 - surfaced with comparison identity
        raise RuntimeError(
            f"comparison {comparison_id!r}: input {side} could not be read ({path}): {exc}"
        ) from exc
    if frame.empty:
        raise RuntimeError(f"comparison {comparison_id!r}: input {side} is empty ({path})")
    return frame


def _apply_cohort(
    frame: pd.DataFrame, cohort: Dict[str, Any], comparison_id: str, side: str
) -> pd.DataFrame:
    where = cohort["where"]
    missing = [column for column in where if column not in frame.columns]
    if missing:
        raise RuntimeError(
            f"comparison {comparison_id!r}: cohort column(s) {missing} missing from "
            f"input {side} (columns: {list(frame.columns)})"
        )
    filtered = frame
    for column, value in where.items():
        if isinstance(value, list):
            mask = filtered[column].isin(value)
        else:
            mask = filtered[column] == value
        filtered = filtered[mask]
    if filtered.empty:
        raise RuntimeError(
            f"comparison {comparison_id!r}: the cohort filter left input {side} empty"
        )
    return filtered


def _apply_threshold(
    frame: pd.DataFrame, threshold: Dict[str, Any], comparison_id: str, side: str
) -> pd.DataFrame:
    column = threshold["column"]
    if column not in frame.columns:
        raise RuntimeError(
            f"comparison {comparison_id!r}: threshold column {column!r} missing from "
            f"input {side} (columns: {list(frame.columns)})"
        )
    values = pd.to_numeric(frame[column], errors="coerce")
    if values.isna().any():
        raise RuntimeError(
            f"comparison {comparison_id!r}: input {side} has "
            f"{int(values.isna().sum())} missing/non-numeric {column!r} value(s)"
        )
    target = float(threshold["value"])
    mask = np.isclose(values.to_numpy(dtype=float), target, rtol=0.0, atol=1e-9)
    filtered = frame[mask]
    if filtered.empty:
        available = sorted(set(values.to_numpy(dtype=float).tolist()))
        raise RuntimeError(
            f"comparison {comparison_id!r}: threshold {column}={target} matched no rows "
            f"in input {side} (available: {available})"
        )
    return filtered


def _numeric_column(
    frame: pd.DataFrame, column: str, comparison_id: str, side: str
) -> np.ndarray:
    if column not in frame.columns:
        raise RuntimeError(
            f"comparison {comparison_id!r}: column {column!r} missing from input {side} "
            f"(columns: {list(frame.columns)})"
        )
    values = pd.to_numeric(frame[column], errors="coerce")
    if values.isna().any():
        raise RuntimeError(
            f"comparison {comparison_id!r}: input {side} column {column!r} has "
            f"{int(values.isna().sum())} missing/non-numeric value(s)"
        )
    return values.to_numpy(dtype=float)


def _prepare_saliency_side(
    comparison: Dict[str, Any], side: str, plan_dir: Path
) -> pd.DataFrame:
    inputs = comparison["inputs"]
    path = _resolve_input(inputs[side], plan_dir)
    frame = _read_input(path, comparison["id"], side)
    frame = _apply_cohort(frame, comparison["cohort"], comparison["id"], side)
    return _apply_threshold(frame, inputs["threshold"], comparison["id"], side)


def _prepare_classification_side(
    comparison: Dict[str, Any], side: str, plan_dir: Path
) -> pd.DataFrame:
    inputs = comparison["inputs"]
    path = _resolve_input(inputs[side], plan_dir)
    frame = _read_input(path, comparison["id"], side)
    return _apply_cohort(frame, comparison["cohort"], comparison["id"], side)


# --------------------------------------------------------------------------- comparisons


def saliency_paired_comparison(
    comparison: Dict[str, Any], plan_dir: Path
) -> Dict[str, Any]:
    """Paired saliency comparison: Wilcoxon signed-rank on `b - a` + bootstrap CI.

    `estimate` is the mean paired difference; its percentile CI is written to the CSV
    `ci_low`/`ci_high`. The median and its CI are kept in the family JSON alongside the
    zero/tie counts the registered zero rule depends on.
    """
    if comparison["kind"] != SALIENCY_KIND:
        raise ValueError(f"not a {SALIENCY_KIND} comparison: {comparison['id']!r}")
    cid = comparison["id"]
    inputs = comparison["inputs"]
    frame_a = _prepare_saliency_side(comparison, "a", plan_dir)
    frame_b = _prepare_saliency_side(comparison, "b", plan_dir)
    aligned_a, aligned_b = align_by_id(frame_a, frame_b, key="image_id")
    values_a = _numeric_column(aligned_a, inputs["metric_column"], cid, "a")
    values_b = _numeric_column(aligned_b, inputs["metric_column"], cid, "b")
    differences = values_b - values_a
    n = int(differences.size)
    if n < 1:
        raise RuntimeError(f"comparison {cid!r}: no aligned rows to test")

    n_zero, n_ties = _zero_and_tie_counts(differences)
    degenerate = bool(np.all(differences == 0.0))
    if degenerate:
        statistic, p_value = 0.0, 1.0
    else:
        try:
            result = wilcoxon(
                differences,
                zero_method=comparison["zero_method"],
                alternative="two-sided",
            )
        except ValueError as exc:
            raise RuntimeError(
                f"comparison {cid!r}: Wilcoxon signed-rank failed "
                f"(zero_method={comparison['zero_method']!r}, n={n}): {exc}"
            ) from exc
        statistic, p_value = float(result.statistic), float(result.pvalue)

    bootstrap = bootstrap_mean_median_ci(
        differences,
        comparison["bootstrap"]["n"],
        comparison["bootstrap"]["seed"],
    )
    return {
        "comparison_id": cid,
        "kind": SALIENCY_KIND,
        "endpoint_class": comparison["endpoint_class"],
        "description": comparison["description"],
        "inputs": {
            "a": str(_resolve_input(inputs["a"], plan_dir)),
            "b": str(_resolve_input(inputs["b"], plan_dir)),
            "metric_column": inputs["metric_column"],
            "threshold": inputs["threshold"],
        },
        "cohort": comparison["cohort"],
        "paired_difference": comparison["paired_difference"],
        "zero_method": comparison["zero_method"],
        "uncertainty_unit": comparison["uncertainty_unit"],
        "bootstrap": {
            "n": comparison["bootstrap"]["n"],
            "seed": comparison["bootstrap"]["seed"],
            "n_used": bootstrap["n_used"],
        },
        "n": n,
        "estimate": float(differences.mean()),
        "ci_low": bootstrap["ci_mean_low"],
        "ci_high": bootstrap["ci_mean_high"],
        "statistic": statistic,
        "p_raw": p_value,
        "n_zero_diffs": n_zero,
        "n_ties": n_ties,
        "tie_quantity": "|b - a| (non-zero)",
        "median_diff": float(np.median(differences)),
        "ci_median_low": bootstrap["ci_median_low"],
        "ci_median_high": bootstrap["ci_median_high"],
        "degenerate": degenerate,
    }


def classification_auroc_comparison(
    comparison: Dict[str, Any], plan_dir: Path
) -> Dict[str, Any]:
    """Paired DeLong test on aligned rows + bootstrap CI of the AUROC difference.

    `estimate` is AUROC(b) - AUROC(a). `n_zero_diffs`/`n_ties` describe the paired
    probability differences (the registered `b - a`); DeLong itself uses the raw scores
    and applies its own midrank tie correction.
    """
    if comparison["kind"] != CLASSIFICATION_KIND:
        raise ValueError(f"not a {CLASSIFICATION_KIND} comparison: {comparison['id']!r}")
    cid = comparison["id"]
    inputs = comparison["inputs"]
    label_column = inputs["label_column"]
    prob_column = inputs["prob_column"]
    frame_a = _prepare_classification_side(comparison, "a", plan_dir)
    frame_b = _prepare_classification_side(comparison, "b", plan_dir)
    aligned_a, aligned_b = align_by_id(frame_a, frame_b, key="image_id")

    labels_a = _numeric_column(aligned_a, label_column, cid, "a")
    labels_b = _numeric_column(aligned_b, label_column, cid, "b")
    if not np.array_equal(labels_a, labels_b):
        mismatch = np.flatnonzero(labels_a != labels_b)
        examples = [
            (str(aligned_a["image_id"].iloc[index]), float(labels_a[index]), float(labels_b[index]))
            for index in mismatch[:5]
        ]
        raise RuntimeError(
            f"comparison {cid!r}: classification labels disagree on "
            f"{int(mismatch.size)} of {labels_a.size} aligned image(s); "
            f"first (image_id, a, b) = {examples}"
        )
    if not np.all(np.isin(labels_a, (0.0, 1.0))):
        raise RuntimeError(
            f"comparison {cid!r}: labels must be binary 0/1 for a paired DeLong test"
        )
    labels = labels_a.astype(int)
    if np.unique(labels).size != 2:
        raise RuntimeError(
            f"comparison {cid!r}: cohort has a single class; AUROC is undefined"
        )

    probs_a = _numeric_column(aligned_a, prob_column, cid, "a")
    probs_b = _numeric_column(aligned_b, prob_column, cid, "b")
    for side, probs in (("a", probs_a), ("b", probs_b)):
        if np.any((probs < 0.0) | (probs > 1.0)):
            raise RuntimeError(
                f"comparison {cid!r}: input {side} {prob_column!r} is not in [0, 1]; "
                "the plan compares probabilities, not logits"
            )

    statistic, p_value, degenerate = delong_paired(labels, probs_a, probs_b)
    auroc_a = float(roc_auc_score(labels, probs_a))
    auroc_b = float(roc_auc_score(labels, probs_b))
    bootstrap = bootstrap_auroc_diff_ci(
        labels,
        probs_a,
        probs_b,
        comparison["bootstrap"]["n"],
        comparison["bootstrap"]["seed"],
    )
    n_zero, n_ties = _zero_and_tie_counts(probs_b - probs_a)
    return {
        "comparison_id": cid,
        "kind": CLASSIFICATION_KIND,
        "endpoint_class": comparison["endpoint_class"],
        "description": comparison["description"],
        "inputs": {
            "a": str(_resolve_input(inputs["a"], plan_dir)),
            "b": str(_resolve_input(inputs["b"], plan_dir)),
            "prob_column": prob_column,
            "label_column": label_column,
        },
        "cohort": comparison["cohort"],
        "paired_difference": comparison["paired_difference"],
        "uncertainty_unit": comparison["uncertainty_unit"],
        "bootstrap": {
            "n": comparison["bootstrap"]["n"],
            "seed": comparison["bootstrap"]["seed"],
            "n_used": bootstrap["n_used"],
        },
        "n": int(labels.size),
        "n_positive": int(labels.sum()),
        "n_negative": int((1 - labels).sum()),
        "estimate": auroc_b - auroc_a,
        "ci_low": bootstrap["ci_low"],
        "ci_high": bootstrap["ci_high"],
        "statistic": statistic,
        "p_raw": p_value,
        "auroc_a": auroc_a,
        "auroc_b": auroc_b,
        "n_zero_diffs": n_zero,
        "n_ties": n_ties,
        "tie_quantity": "|b - a| (non-zero)",
        "delong_degenerate": degenerate,
    }


_COMPARATORS = {
    SALIENCY_KIND: saliency_paired_comparison,
    CLASSIFICATION_KIND: classification_auroc_comparison,
}


def compute_comparison(
    comparison: Dict[str, Any], plan_dir: Path
) -> Dict[str, Any]:
    """Dispatch one validated comparison to its registered test."""
    try:
        comparator = _COMPARATORS[comparison["kind"]]
    except KeyError:
        raise ValueError(f"unknown comparison kind {comparison['kind']!r}") from None
    return comparator(comparison, plan_dir)


# --------------------------------------------------------------------------- family / run


def compute_family(family: Dict[str, Any], plan_dir: Path) -> List[Dict[str, Any]]:
    """Compute every registered comparison, then BH once over exactly this family."""
    records = [compute_comparison(comparison, plan_dir) for comparison in family["comparisons"]]
    reject, adjusted = benjamini_hochberg(
        [record["p_raw"] for record in records], family["alpha"]
    )
    for record, p_bh, rejected in zip(records, adjusted, reject):
        record["p_bh"] = p_bh
        record["reject"] = rejected
    return records


def _json_safe(value: Any) -> Any:
    """Convert numpy scalars and non-finite floats into strict-JSON values."""
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _environment_record() -> Dict[str, Any]:
    versions: Dict[str, Any] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "statsmodels": statsmodels.__version__,
        "scikit-learn": sklearn.__version__,
    }
    try:
        versions["MLstatkit"] = metadata.version("MLstatkit")
    except Exception:  # noqa: BLE001 - recorded as unknown, never fabricated
        versions["MLstatkit"] = None
    versions["PYTHONHASHSEED"] = os.environ.get("PYTHONHASHSEED")
    return versions


def run_plan(path: Path, out_dir: Path) -> Dict[str, Any]:
    """Run a locked plan and write one CSV + one JSON per family.

    Refuses to run when `<plan>.sha256` is missing or does not match the plan bytes.
    Computes every family before writing anything, so a missing input cannot leave a
    partial (silently shrunken) family on disk. Existing outputs are never overwritten.
    """
    plan_path = Path(path)
    digest = require_locked(plan_path)
    plan = validate_plan(load_plan(plan_path))
    plan_dir = plan_path.parent

    output_path = Path(out_dir)
    planned_targets = []
    for family in plan["families"]:
        planned_targets.append(output_path / f"{family['name']}.csv")
        planned_targets.append(output_path / f"{family['name']}.json")
    existing = [str(target) for target in planned_targets if target.exists()]
    if existing:
        raise RuntimeError(
            "refusing to overwrite existing analysis output(s): "
            f"{existing}; choose a new out_dir"
        )

    computed = [
        (family, compute_family(family, plan_dir)) for family in plan["families"]
    ]

    output_path.mkdir(parents=True, exist_ok=True)
    generated = datetime.now(timezone.utc).isoformat(timespec="seconds")
    outputs: Dict[str, Dict[str, str]] = {}
    for family, records in computed:
        csv_path = output_path / f"{family['name']}.csv"
        json_path = output_path / f"{family['name']}.json"
        rows = [
            {
                "comparison_id": record["comparison_id"],
                "endpoint_class": record["endpoint_class"],
                "n": record["n"],
                "estimate": record["estimate"],
                "ci_low": record["ci_low"],
                "ci_high": record["ci_high"],
                "statistic": record["statistic"],
                "p_raw": record["p_raw"],
                "p_bh": record["p_bh"],
                "reject": record["reject"],
                "plan_sha256": digest,
            }
            for record in records
        ]
        pd.DataFrame(rows, columns=OUTPUT_COLUMNS).to_csv(csv_path, index=False)
        payload = {
            "family": family["name"],
            "description": family["description"],
            "endpoint_class": family["endpoint_class"],
            "correction": family["correction"],
            "alpha": family["alpha"],
            "ci_level": CI_LEVEL,
            "plan_path": str(plan_path.resolve()),
            "plan_sha256": digest,
            "n_comparisons": len(records),
            "comparisons": records,
            "environment": _environment_record(),
            "generated_utc": generated,
        }
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(_json_safe(payload), handle, indent=2)
        outputs[family["name"]] = {"csv": str(csv_path), "json": str(json_path)}

    return {
        "plan": str(plan_path.resolve()),
        "plan_sha256": digest,
        "outputs": outputs,
    }

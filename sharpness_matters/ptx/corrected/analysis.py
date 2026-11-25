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
# B (PTX-05 / Section 8.1): current -> every saliency p-value came from SciPy's implicit
#   `wilcoxon(..., method="auto")`: exact below 50 pairs unless there are zeros, and the
#   1.10.1 "exact" is not exact with tied |d|, so the same registered comparison flipped
#   between 1.10.1 (tied 12+/4-, n=16: p=0.0833) and 1.15.1 (p=0.0455) and switched to
#   the normal approximation on any zero difference. Problem -> a pre-registered p-value
#   whose method depends on the data and the installed SciPy version is not registered,
#   and a binary 0/1 hit difference is better tested on its discordant pairs. New ->
#   every saliency_paired comparison registers `p_method` in {"permutation", "approx",
#   "mcnemar_exact"}; `approx` calls `wilcoxon(method="approx")` explicitly (the
#   registered normal approximation, ties breaching the exact distribution), `permutation`
#   is the exhaustive 2^n sign-flip test on tie-averaged signed ranks (zeros dropped with
#   `zero_method="wilcox"`, n_nonzero <= 20, deterministic, no seed) and `mcnemar_exact`
#   is the two-sided exact binomial on the discordant binary pairs. The plan validation
#   rejects `auto`, `exact` and a missing `p_method`; the choice is never made from the
#   sample. The SciPy version stays in the run JSON (PTX-24).
#   check: ::test_permutation_matches_exhaustive_sign_flip_reference,
#   ::test_permutation_drops_zero_differences_like_wilcox,
#   ::test_permutation_equals_scipy_exact_without_ties,
#   ::test_permutation_rejects_cohort_above_the_limit,
#   ::test_approx_path_is_the_explicit_normal_approximation,
#   ::test_mcnemar_exact_counts_only_discordant_pairs,
#   ::test_mcnemar_exact_rejects_non_binary_values
# B (Section 11.2): current -> every bootstrap resampled images independently and
#   `uncertainty_unit` accepted only "image", although the CheXlocalize E4 cohorts repeat
#   patients (720 images / 700 patients; 16 images / 15 patients). Problem -> rows that
#   share a patient are not independent, so an image-level interval is too narrow, and no
#   output said which unit was resampled. New -> `uncertainty_unit` is registered per
#   comparison ("image" | "patient"). A patient-unit comparison registers
#   `inputs.patient_manifest`, a frozen `image_id`/`patient_id` manifest loaded with
#   `load_frozen` (csv + .sha256 + .record.json, hash verified); every aligned image must
#   map to a non-empty patient, and `n_patients`/`n_multi_image_patients`/
#   `max_images_per_patient` are recorded. `classification_auroc` keeps the pooled
#   estimate and the image-level paired DeLong p (recorded as `p_clustering_aware:
#   false`, because DeLong cannot model clustering) and takes a cluster percentile
#   bootstrap that resamples patients and keeps all images of each drawn patient.
#   `saliency_paired` collapses to one value per patient (the mean `b - a` over that
#   patient's images) and runs the registered `p_method` and the bootstrap on those
#   patient values; `mcnemar_exact` has no patient-level binary outcome and is rejected.
#   Image-unit rows record no patient fields and `p_clustering_aware: null`.
#   check: ::test_patient_unit_requires_patient_manifest,
#   ::test_image_unit_rejects_patient_manifest,
#   ::test_patient_unit_rejects_mcnemar_exact,
#   ::test_unmapped_aligned_image_raises_and_writes_nothing,
#   ::test_cluster_draw_keeps_whole_patient_groups,
#   ::test_cluster_bootstrap_ci_is_at_least_as_wide_as_image_bootstrap,
#   ::test_classification_patient_unit_reports_cluster_fields,
#   ::test_saliency_patient_collapse_uses_patient_mean_differences

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
                       # one name for both sides:
                       "prob_column": "prob",
                       # or one name per side when the two CSVs differ
                       # (e.g. the stage-1 CSV's `score` vs stage-2's `prob`):
                       "prob_column_a": "score", "prob_column_b": "prob",
                       "label_column": "label",
                       # required if and only if uncertainty_unit is "patient":
                       # frozen image_id/patient_id manifest (csv + .sha256 + record):
                       "patient_manifest": path},
            "cohort": {"description": str,   # optional
                       "where": {column: value | [values]}},
            "paired_difference": "b - a",
            "zero_method": "wilcox" | "pratt" | "zsplit",  # saliency only, required
            "p_method": "permutation" | "approx" | "mcnemar_exact",  # saliency only, required
            "uncertainty_unit": "image" | "patient",
            "bootstrap": {"n": int, "seed": int}}
         ]}
      ]
    }

`mcnemar_exact` needs binary 0/1 values on both sides and tests the discordant pairs; a
zero difference is a concordant pair there, so the registered `zero_method` is recorded
but not applied (it is still required because it defines the kind's zero rule).

`uncertainty_unit` is the resampling unit. With `"patient"` the comparison also registers
`patient_manifest`, every aligned `image_id` must map to a non-empty patient ID, the
bootstrap resamples patients (all images of a drawn patient are kept), and the record
carries `n_patients`, `n_multi_image_patients` and `max_images_per_patient`.
`classification_auroc` keeps the pooled AUROC difference and the image-level paired
DeLong p-value (DeLong cannot model clustering), recorded explicitly as
`p_clustering_aware: false`; `saliency_paired` runs on one value per patient (the mean
`b - a` over the patient's images), so its p-value is clustering-aware
(`p_clustering_aware: true`) and its `n` counts patients. Image-unit rows have no patient
identity and record `p_clustering_aware: null`.

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
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy
import sklearn
import statsmodels
from MLstatkit.stats import Delong_test
from scipy.stats import binomtest, rankdata, wilcoxon
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests

from sharpness_matters.ptx.corrected.frozen import hash_path, load_frozen, sha256_file

PLAN_VERSION = 1
CLASSIFICATION_ENDPOINT = "classification"
XAI_ENDPOINT_CLASSES = ("localization", "sanity", "faithfulness")
ENDPOINT_CLASSES = (CLASSIFICATION_ENDPOINT,) + XAI_ENDPOINT_CLASSES
SALIENCY_KIND = "saliency_paired"
CLASSIFICATION_KIND = "classification_auroc"
KINDS = (SALIENCY_KIND, CLASSIFICATION_KIND)
ZERO_METHODS = ("wilcox", "pratt", "zsplit")
SALIENCY_P_METHODS = ("permutation", "approx", "mcnemar_exact")
PERMUTATION_MAX_N = 20
CORRECTION_METHODS = ("bh",)
PAIRED_DIFFERENCE = "b - a"
UNCERTAINTY_UNITS = ("image", "patient")
PATIENT_UNIT = "patient"
CI_LEVEL = 0.95
_AUROC_EQUAL_ATOL = 1e-12
OUTPUT_COLUMNS = [
    "comparison_id",
    "endpoint_class",
    "p_method",
    "n",
    "estimate",
    "ci_low",
    "ci_high",
    "statistic",
    "p_raw",
    "p_bh",
    "reject",
    "plan_sha256",
    "uncertainty_unit",
    "n_images",
    "n_patients",
    "n_multi_image_patients",
    "patient_manifest_sha256",
    "p_clustering_aware",
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


def _validate_prob_columns(inputs: Dict[str, Any], where: str) -> Tuple[str, str]:
    """Resolve the registered probability column for each side.

    `prob_column` names one column used on both sides (the original contract).
    `prob_column_a`/`prob_column_b` name each side explicitly, so an A-branch `score`
    CSV and a B-branch `prob` CSV are compared without renaming or guessing. Mixing the
    two forms, or registering only one per-side name, is an error.
    """
    single = "prob_column" in inputs
    per_side = [name for name in ("prob_column_a", "prob_column_b") if name in inputs]
    if single and per_side:
        _fail(
            where,
            "set either prob_column (one column for both sides) or "
            "prob_column_a + prob_column_b, not both",
        )
    if per_side and len(per_side) != 2:
        missing = sorted({"prob_column_a", "prob_column_b"} - set(per_side))
        _fail(
            where,
            f"per-side probability columns need both keys; missing {missing} "
            "(one side is never allowed to fall back to the other's column)",
        )
    if single:
        column = _as_text(inputs["prob_column"], f"{where}.prob_column")
        return column, column
    if per_side:
        return (
            _as_text(inputs["prob_column_a"], f"{where}.prob_column_a"),
            _as_text(inputs["prob_column_b"], f"{where}.prob_column_b"),
        )
    return "prob", "prob"


def _validate_inputs(
    inputs: Any, kind: str, where: str, uncertainty_unit: str
) -> Dict[str, Any]:
    saliency = kind == SALIENCY_KIND
    allowed = ["a", "b"]
    required = ["a", "b"]
    if saliency:
        allowed += ["metric_column", "threshold"]
        required += ["metric_column", "threshold"]
    else:
        allowed += ["prob_column", "prob_column_a", "prob_column_b", "label_column"]
    if uncertainty_unit == PATIENT_UNIT:
        allowed.append("patient_manifest")
        required.append("patient_manifest")
    elif isinstance(inputs, dict) and "patient_manifest" in inputs:
        _fail(
            where,
            "patient_manifest is registered only with uncertainty_unit "
            f"{PATIENT_UNIT!r}; an image-unit comparison resamples images and has no "
            "patient identity (set uncertainty_unit and patient_manifest together)",
        )
    _check_keys(inputs, where, allowed=allowed, required=required)
    normalized: Dict[str, Any] = {
        "a": _as_text(inputs["a"], f"{where}.a"),
        "b": _as_text(inputs["b"], f"{where}.b"),
    }
    normalized["patient_manifest"] = (
        _as_text(inputs["patient_manifest"], f"{where}.patient_manifest")
        if uncertainty_unit == PATIENT_UNIT
        else None
    )
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
        normalized["prob_column_a"], normalized["prob_column_b"] = (
            _validate_prob_columns(inputs, where)
        )
        normalized["label_column"] = _as_text(
            inputs.get("label_column", "label"), f"{where}.label_column"
        )
    return normalized


def _validate_zero_method(comparison: Any, where: str) -> str:
    if "zero_method" not in comparison:
        _fail(f"{where}.zero_method", "is required for saliency_paired")
    zero_method = comparison["zero_method"]
    if zero_method not in ZERO_METHODS:
        _fail(
            f"{where}.zero_method",
            f"must be one of {list(ZERO_METHODS)}, got {zero_method!r}",
        )
    return zero_method


def _validate_p_method(comparison: Any, where: str, zero_method: str) -> str:
    """The registered p-value method: never chosen from the sample or the SciPy default."""
    if "p_method" not in comparison:
        _fail(
            f"{where}.p_method",
            "is required for saliency_paired; register one of "
            f"{list(SALIENCY_P_METHODS)}",
        )
    p_method = comparison["p_method"]
    if p_method not in SALIENCY_P_METHODS:
        _fail(
            f"{where}.p_method",
            f"must be one of {list(SALIENCY_P_METHODS)}, got {p_method!r}",
        )
    if p_method == "permutation" and zero_method != "wilcox":
        _fail(
            f"{where}.p_method",
            "permutation needs zero_method 'wilcox' (the exhaustive sign-flip test drops "
            "zero differences and ranks the rest); register zero_method 'wilcox' or "
            "p_method 'approx'",
        )
    return p_method


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
        "p_method",
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
            "(the resampling unit is registered, never chosen from the sample)",
        )
    inputs = _validate_inputs(
        comparison["inputs"], kind, f"{where}.inputs", uncertainty_unit
    )

    if kind == SALIENCY_KIND:
        zero_method = _validate_zero_method(comparison, where)
        p_method = _validate_p_method(comparison, where, zero_method)
        if p_method == "mcnemar_exact" and uncertainty_unit == PATIENT_UNIT:
            _fail(
                f"{where}.p_method",
                "mcnemar_exact needs a binary 0/1 per-image outcome; with "
                f"uncertainty_unit {PATIENT_UNIT!r} the registered values are "
                "patient-mean differences, so register p_method 'permutation' "
                f"(at most {PERMUTATION_MAX_N} non-zero patient differences) or 'approx'",
            )
    else:
        if "zero_method" in comparison:
            _fail(
                f"{where}.zero_method",
                "is not applicable to classification_auroc (DeLong uses the raw scores)",
            )
        if "p_method" in comparison:
            _fail(
                f"{where}.p_method",
                "is not applicable to classification_auroc (the paired DeLong test is "
                "fixed by the plan, not chosen per comparison)",
            )
        zero_method = None
        p_method = None

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
        "p_method": p_method,
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


def _sign_flip_deviations(doubled_ranks: np.ndarray) -> np.ndarray:
    """|sum_i s_i * r_i| for all 2^n sign patterns s, built by doubling (exhaustive).

    `doubled_ranks` holds 2 * (tie-averaged rank), so every value and every deviation is
    an integer: the "at least as extreme" comparison is exact, with no float tolerance.
    Pattern order carries no meaning, so the doubling order (last rank first) is fine.
    """
    deviations = np.zeros(1, dtype=np.int64)
    for rank in doubled_ranks:
        deviations = np.abs(np.concatenate((deviations + rank, deviations - rank)))
    return deviations


def _permutation_signed_rank_test(
    differences: np.ndarray, comparison_id: str
) -> Tuple[float, float]:
    """Exhaustive sign-flip (permutation) test on the tie-averaged signed-rank statistic.

    Zero differences are dropped (plan validation registers `zero_method="wilcox"`), the
    remaining |d| are ranked with `rankdata` (ties averaged), and the two-sided p-value is
    the proportion of the 2^n sign patterns whose |W+ - mean| is at least the observed
    |W+ - mean|. Deterministic, no seed, no normal approximation. `statistic` follows
    SciPy's two-sided convention min(W+, W-).
    """
    nonzero = np.asarray(differences, dtype=float)
    nonzero = nonzero[nonzero != 0.0]
    n = int(nonzero.size)
    if n == 0:
        return 0.0, 1.0
    if n > PERMUTATION_MAX_N:
        raise RuntimeError(
            f"comparison {comparison_id!r}: p_method 'permutation' must enumerate 2^n "
            f"sign patterns over the {n} non-zero differences, above the limit of "
            f"{PERMUTATION_MAX_N}; use p_method 'approx' for large cohorts"
        )
    doubled_ranks = np.rint(2.0 * rankdata(np.abs(nonzero))).astype(np.int64)
    total = int(doubled_ranks.sum())
    observed = int(doubled_ranks[nonzero > 0].sum())
    deviations = _sign_flip_deviations(doubled_ranks)
    extreme = int(np.count_nonzero(deviations >= abs(2 * observed - total)))
    w_plus = observed / 2.0
    w_minus = total / 2.0 - w_plus
    return float(min(w_plus, w_minus)), extreme / float(2**n)


def _approx_signed_rank_test(
    differences: np.ndarray, zero_method: str, comparison_id: str
) -> Tuple[float, float]:
    """SciPy's normal approximation, requested explicitly (`method="approx"`).

    The registered method is passed through; SciPy's `auto` (exact below 50 pairs unless
    there are zeros) is never reachable from a plan. All-zero differences have no ranked
    observation left: they are recorded as `statistic=0.0, p_raw=1.0, degenerate=True`
    instead of reaching SciPy's ValueError.
    """
    if np.all(differences == 0.0):
        return 0.0, 1.0
    try:
        result = wilcoxon(
            differences,
            zero_method=zero_method,
            alternative="two-sided",
            method="approx",
        )
    except ValueError as exc:
        raise RuntimeError(
            f"comparison {comparison_id!r}: Wilcoxon signed-rank failed "
            f"(p_method='approx', zero_method={zero_method!r}, n={differences.size}): "
            f"{exc}"
        ) from exc
    return float(result.statistic), float(result.pvalue)


def _binary_discordant_counts(
    values_a: np.ndarray, values_b: np.ndarray, comparison_id: str
) -> Tuple[int, int]:
    """Count b (a=0 -> b=1 gains) and c (a=1 -> b=0 losses) on binary 0/1 pairs."""
    for side, values in (("a", values_a), ("b", values_b)):
        invalid = values[~np.isin(values, (0.0, 1.0))]
        if invalid.size:
            raise RuntimeError(
                f"comparison {comparison_id!r}: p_method 'mcnemar_exact' needs binary "
                f"0/1 values on both sides; input {side} has {invalid.size} non-binary "
                f"value(s), found {float(invalid[0])!r} in input {side}"
            )
    b = int(np.count_nonzero((values_b == 1.0) & (values_a == 0.0)))
    c = int(np.count_nonzero((values_b == 0.0) & (values_a == 1.0)))
    return b, c


def _mcnemar_exact_test(
    values_a: np.ndarray, values_b: np.ndarray, comparison_id: str
) -> Tuple[float, float]:
    """Exact McNemar on the discordant binary pairs: two-sided exact binomial b vs c."""
    b, c = _binary_discordant_counts(values_a, values_b, comparison_id)
    discordant = b + c
    if discordant == 0:
        return 0.0, 1.0
    smaller = min(b, c)
    return float(smaller), float(binomtest(smaller, discordant, 0.5).pvalue)


def _percentile_ci(values: np.ndarray) -> Tuple[float, float]:
    tail = (1.0 - CI_LEVEL) / 2.0 * 100.0
    low, high = np.percentile(values, [tail, 100.0 - tail])
    return float(low), float(high)


def bootstrap_mean_median_ci(
    differences: np.ndarray, n_boot: int, seed: int
) -> Dict[str, Any]:
    """Percentile bootstrap of the mean and the median paired difference.

    `differences` holds one value per registered uncertainty unit: an image difference
    for `uncertainty_unit: "image"`, or a patient-mean difference for
    `uncertainty_unit: "patient"`, so resampling these values resamples the registered
    unit with replacement (registered `n` and `seed`).
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


def patient_mean_differences(
    differences: np.ndarray, patient_ids: np.ndarray
) -> np.ndarray:
    """Collapse paired image differences to one value per patient: the mean of `b - a`.

    The registered patient-level estimand is the mean of the patient-mean differences,
    so a patient with two images contributes one observation, never two. Returned in
    sorted patient-ID order, so row order in the inputs cannot change the result.
    """
    frame = pd.DataFrame(
        {
            "patient_id": np.asarray(patient_ids, dtype=str),
            "difference": np.asarray(differences, dtype=float),
        }
    )
    means = frame.groupby("patient_id", sort=True)["difference"].mean()
    return means.to_numpy(dtype=float)


def patient_blocks(patient_ids: np.ndarray) -> List[np.ndarray]:
    """Row positions of each patient, ordered by sorted patient ID."""
    ids = np.asarray(patient_ids, dtype=str)
    return [np.flatnonzero(ids == patient) for patient in np.unique(ids)]


def cluster_draw(blocks: Sequence[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    """One cluster-bootstrap draw: patients with replacement, all images of each drawn patient.

    `blocks` comes from `patient_blocks`. A drawn patient contributes every one of its
    rows, so within-patient dependence is preserved instead of being broken into
    independent images.
    """
    if not blocks:
        raise ValueError("cluster bootstrap needs at least one patient block")
    chosen = rng.integers(0, len(blocks), size=len(blocks))
    return np.concatenate([blocks[index] for index in chosen])


def _auroc_diff_percentile_ci(
    labels: np.ndarray,
    prob_a: np.ndarray,
    prob_b: np.ndarray,
    draws: Sequence[np.ndarray],
    n_boot: int,
) -> Dict[str, Any]:
    """Percentile CI of AUROC(b) - AUROC(a) over the supplied resample index draws.

    A resample that contains one class only has no AUROC; it is skipped and counted in
    `n_used` (never hidden). No valid resample raises.
    """
    differences = []
    for index in draws:
        sample = labels[index]
        if sample.min() == sample.max():
            continue
        differences.append(
            roc_auc_score(sample, prob_b[index]) - roc_auc_score(sample, prob_a[index])
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


def bootstrap_auroc_diff_ci(
    labels: np.ndarray,
    prob_a: np.ndarray,
    prob_b: np.ndarray,
    n_boot: int,
    seed: int,
) -> Dict[str, Any]:
    """Percentile bootstrap over images of AUROC(b) - AUROC(a).

    Every draw resamples image rows with replacement (the registered image unit).
    """
    y = np.asarray(labels, dtype=int)
    pa = np.asarray(prob_a, dtype=float)
    pb = np.asarray(prob_b, dtype=float)
    rng = np.random.default_rng(seed)
    draws = (rng.integers(0, y.size, size=y.size) for _ in range(n_boot))
    return _auroc_diff_percentile_ci(y, pa, pb, draws, n_boot)


def bootstrap_auroc_diff_ci_clustered(
    labels: np.ndarray,
    prob_a: np.ndarray,
    prob_b: np.ndarray,
    patient_ids: np.ndarray,
    n_boot: int,
    seed: int,
) -> Dict[str, Any]:
    """Percentile cluster bootstrap of AUROC(b) - AUROC(a).

    Every draw resamples patients with replacement and keeps all images of each drawn
    patient (`cluster_draw`), so the interval reflects repeated patients. The estimate
    itself stays the pooled image-level AUROC difference: only the uncertainty is
    clustered.
    """
    y = np.asarray(labels, dtype=int)
    pa = np.asarray(prob_a, dtype=float)
    pb = np.asarray(prob_b, dtype=float)
    blocks = patient_blocks(patient_ids)
    rng = np.random.default_rng(seed)
    draws = (cluster_draw(blocks, rng) for _ in range(n_boot))
    return _auroc_diff_percentile_ci(y, pa, pb, draws, n_boot)


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


# --------------------------------------------------------------------------- patient identity


def load_patient_map(path: Path, comparison_id: str) -> pd.DataFrame:
    """Load the frozen `image_id` -> `patient_id` manifest (hash and record verified).

    `load_frozen` refuses a manifest whose csv, `.sha256` and `.record.json` do not
    agree, so a patient map is never edited after the plan is locked. The manifest must
    name each image once with a non-empty patient ID.
    """
    try:
        frame = load_frozen(path)
    except Exception as exc:  # noqa: BLE001 - surfaced with comparison identity
        raise RuntimeError(
            f"comparison {comparison_id!r}: patient manifest {path} could not be loaded: "
            f"{exc}"
        ) from exc
    missing = [
        column for column in ("image_id", "patient_id") if column not in frame.columns
    ]
    if missing:
        raise RuntimeError(
            f"comparison {comparison_id!r}: patient manifest {path} is missing "
            f"column(s) {missing} (columns: {list(frame.columns)})"
        )
    image_ids = frame["image_id"].astype(str)
    duplicated = image_ids[image_ids.duplicated(keep=False)]
    if len(duplicated):
        examples = sorted(set(duplicated))[:5]
        raise RuntimeError(
            f"comparison {comparison_id!r}: patient manifest {path} has "
            f"{duplicated.nunique()} duplicated image_id(s) over {len(duplicated)} "
            f"row(s); examples {examples}"
        )
    patients = frame["patient_id"]
    empty = patients.isna() | (patients.astype(str).str.strip() == "")
    if bool(empty.any()):
        raise RuntimeError(
            f"comparison {comparison_id!r}: patient manifest {path} has "
            f"{int(empty.sum())} row(s) with an empty patient_id; examples "
            f"{image_ids[empty].tolist()[:5]}"
        )
    return pd.DataFrame(
        {"image_id": image_ids, "patient_id": patients.astype(str)}
    )


def patient_assignments(
    aligned: pd.DataFrame, patient_map: pd.DataFrame, comparison_id: str
) -> np.ndarray:
    """Patient ID of every aligned row, in row order; an unmapped image raises."""
    rows = pd.DataFrame({"image_id": aligned["image_id"].astype(str)})
    merged = rows.merge(patient_map, on="image_id", how="left")
    unmapped = merged["patient_id"].isna()
    if bool(unmapped.any()):
        examples = merged.loc[unmapped, "image_id"].tolist()[:5]
        raise RuntimeError(
            f"comparison {comparison_id!r}: {int(unmapped.sum())} aligned image(s) have "
            f"no patient_id in the frozen manifest; examples {examples}"
        )
    return merged["patient_id"].to_numpy(dtype=str)


def _patient_context(
    comparison: Dict[str, Any], aligned: pd.DataFrame, plan_dir: Path
) -> Optional[Dict[str, Any]]:
    """Patient identity and summary counts for a patient-unit comparison, else None."""
    if comparison["uncertainty_unit"] != PATIENT_UNIT:
        return None
    manifest_path = _resolve_input(comparison["inputs"]["patient_manifest"], plan_dir)
    patient_map = load_patient_map(manifest_path, comparison["id"])
    patient_ids = patient_assignments(aligned, patient_map, comparison["id"])
    patients, counts = np.unique(patient_ids, return_counts=True)
    return {
        "patient_ids": patient_ids,
        # Absolute on disk, so a report never depends on the directory the plan was
        # invoked from (`_resolve_input` keeps the plan-relative spelling).
        "manifest": str(manifest_path.resolve()),
        "sha256": sha256_file(manifest_path),
        "n_patients": int(patients.size),
        "n_multi_image_patients": int(np.count_nonzero(counts > 1)),
        "max_images_per_patient": int(counts.max()),
    }


def _patient_fields(
    context: Optional[Dict[str, Any]], n_images: int
) -> Dict[str, Any]:
    """Output fields describing the registered unit; image-unit rows record None."""
    if context is None:
        return {
            "n_images": int(n_images),
            "n_patients": None,
            "n_multi_image_patients": None,
            "max_images_per_patient": None,
            "patient_manifest": None,
            "patient_manifest_sha256": None,
        }
    return {
        "n_images": int(n_images),
        "n_patients": context["n_patients"],
        "n_multi_image_patients": context["n_multi_image_patients"],
        "max_images_per_patient": context["max_images_per_patient"],
        "patient_manifest": context["manifest"],
        "patient_manifest_sha256": context["sha256"],
    }


def _p_clustering_aware(kind: str, uncertainty_unit: str) -> Optional[bool]:
    """Does the registered p-value model the within-patient dependence of the rows?

    `classification_auroc` always uses the image-level paired DeLong test, which cannot
    model clustering: on patient-unit rows that limit is recorded as `False` rather than
    claimed away (a cluster-robust p-value is not invented). `saliency_paired` on patient
    units tests the patient-mean differences themselves, one value per patient, so it is
    clustering-aware. Image-unit rows have no patient identity: `None`.
    """
    if uncertainty_unit != PATIENT_UNIT:
        return None
    return kind == SALIENCY_KIND


# --------------------------------------------------------------------------- comparisons


def _signed_rank_test(
    comparison: Dict[str, Any], differences: np.ndarray
) -> Tuple[float, float]:
    """Run the registered signed-rank `p_method` on one value per registered unit."""
    p_method = comparison["p_method"]
    if p_method == "permutation":
        return _permutation_signed_rank_test(differences, comparison["id"])
    if p_method == "approx":
        return _approx_signed_rank_test(
            differences, comparison["zero_method"], comparison["id"]
        )
    raise RuntimeError(
        f"comparison {comparison['id']!r}: p_method {p_method!r} needs the per-image "
        "0/1 pairs and is not a signed-rank test on one value per unit"
    )


def _saliency_test(
    comparison: Dict[str, Any], values_a: np.ndarray, values_b: np.ndarray
) -> Tuple[float, float]:
    """Run the registered `p_method` on the aligned image pairs; (statistic, p_raw)."""
    if comparison["p_method"] == "mcnemar_exact":
        return _mcnemar_exact_test(values_a, values_b, comparison["id"])
    return _signed_rank_test(comparison, values_b - values_a)


def saliency_paired_comparison(
    comparison: Dict[str, Any], plan_dir: Path
) -> Dict[str, Any]:
    """Paired saliency comparison: the registered `p_method` on `b - a` + bootstrap CI.

    `estimate` is the mean paired difference on the registered unit: the image mean, or
    for `uncertainty_unit: "patient"` the mean of the patient-mean differences (one
    value per patient, `n` counts patients and `n_images` is also recorded). Its
    percentile CI is written to the CSV `ci_low`/`ci_high`. The median and its CI are
    kept in the family JSON alongside the zero/tie counts the registered zero rule
    depends on. The p-value comes from the registered `p_method` (`permutation`,
    `approx` or `mcnemar_exact`), never from the sample size, the presence of
    ties/zeros, or SciPy's `method="auto"`.
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
    n_images = int(differences.size)
    if n_images < 1:
        raise RuntimeError(f"comparison {cid!r}: no aligned rows to test")

    patient_context = _patient_context(comparison, aligned_a, plan_dir)
    if patient_context is None:
        unit_differences = differences
        tie_quantity = "|b - a| (non-zero)"
        statistic, p_value = _saliency_test(comparison, values_a, values_b)
    else:
        unit_differences = patient_mean_differences(
            differences, patient_context["patient_ids"]
        )
        tie_quantity = "patient-mean |b - a| (non-zero)"
        statistic, p_value = _signed_rank_test(comparison, unit_differences)

    n = int(unit_differences.size)
    n_zero, n_ties = _zero_and_tie_counts(unit_differences)
    degenerate = bool(np.all(unit_differences == 0.0))

    bootstrap = bootstrap_mean_median_ci(
        unit_differences,
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
            "patient_manifest": (
                patient_context["manifest"] if patient_context is not None else None
            ),
        },
        "cohort": comparison["cohort"],
        "paired_difference": comparison["paired_difference"],
        "zero_method": comparison["zero_method"],
        "p_method": comparison["p_method"],
        "uncertainty_unit": comparison["uncertainty_unit"],
        "p_clustering_aware": _p_clustering_aware(
            SALIENCY_KIND, comparison["uncertainty_unit"]
        ),
        "bootstrap": {
            "n": comparison["bootstrap"]["n"],
            "seed": comparison["bootstrap"]["seed"],
            "n_used": bootstrap["n_used"],
        },
        "n": n,
        "estimate": float(unit_differences.mean()),
        "ci_low": bootstrap["ci_mean_low"],
        "ci_high": bootstrap["ci_mean_high"],
        "statistic": statistic,
        "p_raw": p_value,
        "n_zero_diffs": n_zero,
        "n_ties": n_ties,
        "tie_quantity": tie_quantity,
        "median_diff": float(np.median(unit_differences)),
        "ci_median_low": bootstrap["ci_median_low"],
        "ci_median_high": bootstrap["ci_median_high"],
        "degenerate": degenerate,
        **_patient_fields(patient_context, n_images),
    }


def classification_auroc_comparison(
    comparison: Dict[str, Any], plan_dir: Path
) -> Dict[str, Any]:
    """Paired DeLong test on aligned rows + bootstrap CI of the AUROC difference.

    `estimate` is AUROC(b) - AUROC(a), pooled over the aligned rows. `n_zero_diffs`/
    `n_ties` describe the paired probability differences (the registered `b - a`);
    DeLong itself uses the raw scores and applies its own midrank tie correction.

    With `uncertainty_unit: "patient"` the interval is a cluster bootstrap over patients
    and `p_clustering_aware` is `False`: the p-value stays the image-level paired DeLong
    test, which cannot model the repeated patients (no cluster-robust p is invented).
    """
    if comparison["kind"] != CLASSIFICATION_KIND:
        raise ValueError(f"not a {CLASSIFICATION_KIND} comparison: {comparison['id']!r}")
    cid = comparison["id"]
    inputs = comparison["inputs"]
    label_column = inputs["label_column"]
    prob_column_a = inputs["prob_column_a"]
    prob_column_b = inputs["prob_column_b"]
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

    probs_a = _numeric_column(aligned_a, prob_column_a, cid, "a")
    probs_b = _numeric_column(aligned_b, prob_column_b, cid, "b")
    for side, column, probs in (
        ("a", prob_column_a, probs_a),
        ("b", prob_column_b, probs_b),
    ):
        if np.any((probs < 0.0) | (probs > 1.0)):
            raise RuntimeError(
                f"comparison {cid!r}: input {side} {column!r} is not in [0, 1]; "
                "the plan compares probabilities, not logits"
            )

    statistic, p_value, degenerate = delong_paired(labels, probs_a, probs_b)
    auroc_a = float(roc_auc_score(labels, probs_a))
    auroc_b = float(roc_auc_score(labels, probs_b))
    patient_context = _patient_context(comparison, aligned_a, plan_dir)
    if patient_context is None:
        bootstrap = bootstrap_auroc_diff_ci(
            labels,
            probs_a,
            probs_b,
            comparison["bootstrap"]["n"],
            comparison["bootstrap"]["seed"],
        )
    else:
        bootstrap = bootstrap_auroc_diff_ci_clustered(
            labels,
            probs_a,
            probs_b,
            patient_context["patient_ids"],
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
            "prob_column_a": prob_column_a,
            "prob_column_b": prob_column_b,
            "label_column": label_column,
            "patient_manifest": (
                patient_context["manifest"] if patient_context is not None else None
            ),
        },
        "cohort": comparison["cohort"],
        "paired_difference": comparison["paired_difference"],
        "uncertainty_unit": comparison["uncertainty_unit"],
        "p_clustering_aware": _p_clustering_aware(
            CLASSIFICATION_KIND, comparison["uncertainty_unit"]
        ),
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
        **_patient_fields(patient_context, int(labels.size)),
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
                "p_method": record.get("p_method"),
                "n": record["n"],
                "estimate": record["estimate"],
                "ci_low": record["ci_low"],
                "ci_high": record["ci_high"],
                "statistic": record["statistic"],
                "p_raw": record["p_raw"],
                "p_bh": record["p_bh"],
                "reject": record["reject"],
                "plan_sha256": digest,
                "uncertainty_unit": record["uncertainty_unit"],
                "n_images": record["n_images"],
                "n_patients": record["n_patients"],
                "n_multi_image_patients": record["n_multi_image_patients"],
                "patient_manifest_sha256": record["patient_manifest_sha256"],
                "p_clustering_aware": record["p_clustering_aware"],
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pre-registered analysis plan (Section 8).")
    sub = parser.add_subparsers(dest="command", required=True)
    lock_cmd = sub.add_parser("lock", help="validate the plan and write <plan>.sha256")
    lock_cmd.add_argument("plan", type=Path)
    run_cmd = sub.add_parser("run", help="run a locked plan")
    run_cmd.add_argument("plan", type=Path)
    run_cmd.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "lock":
        print(lock_plan(args.plan))
    else:
        print(json.dumps(run_plan(args.plan, args.out_dir), indent=2))

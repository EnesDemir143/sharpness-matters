"""Stage-2 (B) analysis tests: locked plan, ID alignment, registered tests, one BH family.

Synthetic only: small ID-keyed CSVs in `tmp_path`, hand-computed differences and
hand-computed BH. Each test names the module docstring entry of
`sharpness_matters/ptx/corrected/analysis.py` it covers:

- PTX-05 / Section 8.1: alignment by ID (never position), duplicate/missing IDs raise,
  labels must agree, and the registered zero rule is the one used.
- PTX-22 / Section 8.2: the family is the plan's comparison list; a missing input raises
  instead of shrinking it; BH matches the hand computation.
- PTX-11 / Section 8.3: a family may not mix endpoint classes; classification AUROC is
  the `classification` endpoint and saliency is one of the three XAI endpoints.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import binomtest, wilcoxon
from sklearn.metrics import roc_auc_score

from sharpness_matters.ptx.corrected.analysis import (
    OUTPUT_COLUMNS,
    SALIENCY_P_METHODS,
    align_by_id,
    benjamini_hochberg,
    bootstrap_auroc_diff_ci,
    bootstrap_auroc_diff_ci_clustered,
    cluster_draw,
    delong_paired,
    lock_plan,
    patient_blocks,
    plan_sha256,
    run_plan,
    validate_plan,
)
from sharpness_matters.ptx.corrected.frozen import load_or_create_frozen, sha256_file

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_PLAN = (
    REPO_ROOT / "sharpness_matters" / "ptx" / "corrected" / "analysis_plan.example.json"
)


# --------------------------------------------------------------------------- helpers


def write_saliency(path, ids, ious, threshold=0.5, decisions=None):
    frame = pd.DataFrame(
        {
            "image_id": list(ids),
            "prob": 0.5,
            "decision": 1 if decisions is None else list(decisions),
            "threshold": threshold,
            "iou": list(ious),
        }
    )
    frame.to_csv(path, index=False)
    return path


def write_classification(path, ids, labels, probs, prob_column="prob"):
    pd.DataFrame(
        {"image_id": list(ids), "label": list(labels), prob_column: list(probs)}
    ).to_csv(path, index=False)
    return path


def write_paired_differences(tmp_path, differences, base=0.5):
    """Write `a.csv`/`b.csv` whose `b - a` is exactly `differences` after the CSV round trip.

    `base` and every difference are exact binary fractions, so the registered test sees
    the intended ties and zeros and not a floating-point near-tie.
    """
    differences = np.asarray(differences, dtype=float)
    ids = [f"p{index:02d}" for index in range(differences.size)]
    write_saliency(tmp_path / "a.csv", ids, [base] * differences.size)
    write_saliency(tmp_path / "b.csv", ids, base + differences)
    return ids


def write_patient_manifest(path, image_ids, patient_ids):
    """Freeze a synthetic `image_id` -> `patient_id` manifest (csv + .sha256 + record)."""
    frame = pd.DataFrame(
        {"image_id": list(image_ids), "patient_id": list(patient_ids)}
    )
    load_or_create_frozen(
        Path(path),
        lambda: frame,
        {"description": "synthetic image_id -> patient_id map for the analysis tests"},
    )
    return path


def saliency_comparison(
    comparison_id,
    a="a.csv",
    b="b.csv",
    metric_column="iou",
    threshold=0.5,
    zero_method="wilcox",
    p_method="approx",
    endpoint_class="localization",
    where=None,
    n_boot=200,
    seed=7,
    uncertainty_unit="image",
    patient_manifest=None,
):
    comparison = {
        "id": comparison_id,
        "kind": "saliency_paired",
        "endpoint_class": endpoint_class,
        "inputs": {
            "a": a,
            "b": b,
            "metric_column": metric_column,
            "threshold": {"column": "threshold", "value": threshold},
        },
        "cohort": {"where": {} if where is None else where},
        "paired_difference": "b - a",
        "uncertainty_unit": uncertainty_unit,
        "bootstrap": {"n": n_boot, "seed": seed},
    }
    if patient_manifest is not None:
        comparison["inputs"]["patient_manifest"] = patient_manifest
    if zero_method is not None:
        comparison["zero_method"] = zero_method
    if p_method is not None:
        comparison["p_method"] = p_method
    return comparison


def classification_comparison(
    comparison_id,
    a="a.csv",
    b="b.csv",
    endpoint_class="classification",
    where=None,
    n_boot=200,
    seed=7,
    prob_column="prob",
    prob_columns=None,
    uncertainty_unit="image",
    patient_manifest=None,
):
    inputs = {"a": a, "b": b, "label_column": "label"}
    if prob_columns is None:
        inputs["prob_column"] = prob_column
    else:
        inputs["prob_column_a"], inputs["prob_column_b"] = prob_columns
    if patient_manifest is not None:
        inputs["patient_manifest"] = patient_manifest
    return {
        "id": comparison_id,
        "kind": "classification_auroc",
        "endpoint_class": endpoint_class,
        "inputs": inputs,
        "cohort": {"where": {} if where is None else where},
        "paired_difference": "b - a",
        "uncertainty_unit": uncertainty_unit,
        "bootstrap": {"n": n_boot, "seed": seed},
    }


def family(name, comparisons, alpha=0.05, endpoint_class=None):
    payload = {
        "name": name,
        "correction": "bh",
        "alpha": alpha,
        "comparisons": comparisons,
    }
    if endpoint_class is not None:
        payload["endpoint_class"] = endpoint_class
    return payload


def write_plan(tmp_path, families, name="plan.json"):
    path = Path(tmp_path) / name
    path.write_text(
        json.dumps({"plan_version": 1, "families": families}, indent=2),
        encoding="utf-8",
    )
    return path


def locked_plan(tmp_path, families, name="plan.json"):
    path = write_plan(tmp_path, families, name=name)
    lock_plan(path)
    return path


def read_outputs(out_dir, family_name):
    out_dir = Path(out_dir)
    frame = pd.read_csv(out_dir / f"{family_name}.csv")
    payload = json.loads((out_dir / f"{family_name}.json").read_text(encoding="utf-8"))
    return frame, payload


# --------------------------------------------------------------------------- plan schema


def test_example_plan_validates():
    plan = validate_plan(json.loads(EXAMPLE_PLAN.read_text(encoding="utf-8")))
    assert [item["name"] for item in plan["families"]] == [
        "siim_k1_localization_iou_t050",
        "siim_classification_auroc",
        "chexlocalize_e4_localization_iou_patient",
        "chexlocalize_e4_classification_auroc_patient",
    ]
    assert [item["endpoint_class"] for item in plan["families"]] == [
        "localization",
        "classification",
        "localization",
        "classification",
    ]


def test_unknown_key_rejected():
    comparison = saliency_comparison("cmp")
    comparison["input"] = comparison.pop("inputs")
    with pytest.raises(ValueError, match="unknown key"):
        validate_plan({"plan_version": 1, "families": [family("f", [comparison])]})


def test_mixed_endpoint_classes_rejected():
    comparisons = [
        saliency_comparison("loc", endpoint_class="localization"),
        classification_comparison("auroc", endpoint_class="classification"),
    ]
    with pytest.raises(ValueError, match="must not mix endpoint classes"):
        validate_plan({"plan_version": 1, "families": [family("f", comparisons)]})


def test_classification_auroc_requires_classification_endpoint():
    comparison = classification_comparison("cmp", endpoint_class="sanity")
    with pytest.raises(ValueError, match="performance endpoint"):
        validate_plan({"plan_version": 1, "families": [family("f", [comparison])]})


def test_saliency_cannot_use_classification_endpoint():
    comparison = saliency_comparison("cmp", endpoint_class="classification")
    with pytest.raises(ValueError, match="saliency_paired must use one of"):
        validate_plan({"plan_version": 1, "families": [family("f", [comparison])]})


def test_endpoint_class_outside_registered_set_rejected():
    comparison = saliency_comparison("cmp", endpoint_class="clinical_benefit")
    with pytest.raises(ValueError, match="endpoint_class"):
        validate_plan({"plan_version": 1, "families": [family("f", [comparison])]})


def test_saliency_comparison_requires_threshold():
    comparison = saliency_comparison("cmp")
    del comparison["inputs"]["threshold"]
    with pytest.raises(ValueError, match="threshold"):
        validate_plan({"plan_version": 1, "families": [family("f", [comparison])]})


def test_classification_comparison_rejects_zero_method():
    comparison = classification_comparison("cmp")
    comparison["zero_method"] = "wilcox"
    with pytest.raises(ValueError, match="not applicable"):
        validate_plan({"plan_version": 1, "families": [family("f", [comparison])]})


def test_saliency_comparison_requires_p_method():
    comparison = saliency_comparison("cmp", p_method=None)
    with pytest.raises(ValueError, match=r"p_method.*required for saliency_paired"):
        validate_plan({"plan_version": 1, "families": [family("f", [comparison])]})


@pytest.mark.parametrize("value", ["auto", "exact", "t_test"])
def test_saliency_p_method_outside_registered_set_rejected(value):
    comparison = saliency_comparison("cmp", p_method=value)
    with pytest.raises(
        ValueError, match=r"p_method.*must be one of.*'permutation'.*got"
    ):
        validate_plan({"plan_version": 1, "families": [family("f", [comparison])]})


def test_classification_comparison_rejects_p_method():
    comparison = classification_comparison("cmp")
    comparison["p_method"] = "approx"
    with pytest.raises(ValueError, match="not applicable"):
        validate_plan({"plan_version": 1, "families": [family("f", [comparison])]})


def test_permutation_p_method_requires_wilcox_zero_method():
    comparison = saliency_comparison("cmp", zero_method="pratt", p_method="permutation")
    with pytest.raises(ValueError, match=r"permutation needs zero_method 'wilcox'"):
        validate_plan({"plan_version": 1, "families": [family("f", [comparison])]})


def test_example_plan_registers_a_p_method_per_comparison():
    plan = validate_plan(json.loads(EXAMPLE_PLAN.read_text(encoding="utf-8")))
    comparisons = [
        comparison
        for registered_family in plan["families"]
        for comparison in registered_family["comparisons"]
    ]
    saliency = [item for item in comparisons if item["kind"] == "saliency_paired"]
    classification = [
        item for item in comparisons if item["kind"] == "classification_auroc"
    ]
    assert saliency and classification
    assert all(item["p_method"] in SALIENCY_P_METHODS for item in saliency)
    # The example plan's two SIIM cohorts are large (>20 images): `approx`. The
    # CheXlocalize XAI patient cohort has 15 patients, so the exhaustive sign-flip test
    # is the registered one there.
    assert {
        item["p_method"]
        for item in saliency
        if item["uncertainty_unit"] == "image"
    } == {"approx"}
    assert {
        item["p_method"]
        for item in saliency
        if item["uncertainty_unit"] == "patient"
    } == {"permutation"}
    # DeLong is the registered classification test; no p-value method is registered.
    assert all(item["p_method"] is None for item in classification)


def test_mixed_single_and_per_side_prob_columns_rejected():
    comparison = classification_comparison("cmp", prob_columns=("score", "prob"))
    comparison["inputs"]["prob_column"] = "prob"
    with pytest.raises(ValueError, match="not both"):
        validate_plan({"plan_version": 1, "families": [family("f", [comparison])]})


def test_per_side_prob_columns_must_be_complete():
    comparison = classification_comparison("cmp", prob_columns=("score", "prob"))
    del comparison["inputs"]["prob_column_b"]
    with pytest.raises(ValueError, match=r"missing \['prob_column_b'\]"):
        validate_plan({"plan_version": 1, "families": [family("f", [comparison])]})


def test_lock_is_idempotent_and_rejects_edit_after_lock(tmp_path):
    plan = write_plan(tmp_path, [family("f", [saliency_comparison("cmp")])])
    digest = lock_plan(plan)
    assert (tmp_path / "plan.json.sha256").read_text().strip() == digest
    assert lock_plan(plan) == digest

    payload = json.loads(plan.read_text())
    payload["families"][0]["alpha"] = 0.01
    plan.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="changed after it was locked"):
        lock_plan(plan)


# --------------------------------------------------------------------------- locking gate


def test_run_plan_requires_lock(tmp_path):
    write_saliency(tmp_path / "a.csv", ["i1", "i2"], [0.2, 0.3])
    write_saliency(tmp_path / "b.csv", ["i1", "i2"], [0.25, 0.35])
    plan = write_plan(tmp_path, [family("f", [saliency_comparison("cmp")])])
    with pytest.raises(RuntimeError, match="not locked"):
        run_plan(plan, tmp_path / "out")


def test_run_plan_rejects_modified_plan(tmp_path):
    write_saliency(tmp_path / "a.csv", ["i1", "i2"], [0.2, 0.3])
    write_saliency(tmp_path / "b.csv", ["i1", "i2"], [0.25, 0.35])
    plan = locked_plan(tmp_path, [family("f", [saliency_comparison("cmp")])])
    payload = json.loads(plan.read_text())
    payload["families"][0]["alpha"] = 0.01
    plan.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        run_plan(plan, tmp_path / "out")


# --------------------------------------------------------------------------- ID alignment


def test_align_by_id_rejects_duplicate_ids():
    a = pd.DataFrame({"image_id": ["x", "y"], "value": [1, 2]})
    b = pd.DataFrame({"image_id": ["x", "x"], "value": [1, 2]})
    with pytest.raises(ValueError, match="not unique"):
        align_by_id(a, b)


def test_align_by_id_rejects_different_id_sets():
    a = pd.DataFrame({"image_id": ["x", "y"], "value": [1, 2]})
    b = pd.DataFrame({"image_id": ["y", "z"], "value": [1, 2]})
    with pytest.raises(ValueError) as error:
        align_by_id(a, b)
    message = str(error.value)
    assert "n_a=2" in message and "n_b=2" in message
    assert "only in a ['x']" in message and "only in b ['z']" in message


def test_align_by_id_uses_id_not_position():
    a = pd.DataFrame({"image_id": ["x", "y", "z"], "value": [1.0, 2.0, 3.0]})
    b = pd.DataFrame({"image_id": ["z", "x", "y"], "value": [30.0, 10.0, 20.0]})
    aligned_a, aligned_b = align_by_id(a, b)
    assert aligned_a["image_id"].tolist() == ["x", "y", "z"]
    assert aligned_b["image_id"].tolist() == ["x", "y", "z"]
    assert aligned_b["value"].tolist() == [10.0, 20.0, 30.0]


# --------------------------------------------------------------------------- family integrity


def test_missing_input_raises_and_writes_nothing(tmp_path):
    write_saliency(tmp_path / "a.csv", ["i1", "i2"], [0.2, 0.3])
    write_saliency(tmp_path / "b.csv", ["i1", "i2"], [0.25, 0.35])
    comparisons = [
        saliency_comparison("present", a="a.csv", b="b.csv"),
        saliency_comparison("missing", a="a.csv", b="absent.csv"),
    ]
    plan = locked_plan(tmp_path, [family("f", comparisons)])
    with pytest.raises(RuntimeError, match="input b not found"):
        run_plan(plan, tmp_path / "out")
    # A registered comparison is never dropped: nothing was written, not a 1-test family.
    assert not (tmp_path / "out").exists()


def test_refuses_to_overwrite_outputs(tmp_path):
    write_saliency(tmp_path / "a.csv", ["i1", "i2"], [0.2, 0.3])
    write_saliency(tmp_path / "b.csv", ["i1", "i2"], [0.25, 0.35])
    plan = locked_plan(tmp_path, [family("f", [saliency_comparison("cmp")])])
    run_plan(plan, tmp_path / "out")
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        run_plan(plan, tmp_path / "out")


# --------------------------------------------------------------------------- BH


def test_bh_matches_hand_computation():
    reject, adjusted = benjamini_hochberg([0.01, 0.04, 0.03, 0.20], 0.05)
    assert reject == [True, False, False, False]
    assert adjusted == pytest.approx([0.04, 0.16 / 3.0, 0.16 / 3.0, 0.20])


def test_bh_applied_once_per_family_over_registered_comparisons(tmp_path):
    from statsmodels.stats.multitest import multipletests

    rng = np.random.default_rng(31)
    ids = [f"i{index:03d}" for index in range(20)]
    values_a = rng.uniform(0.10, 0.50, len(ids))
    write_saliency(tmp_path / "a.csv", ids, values_a)
    comparisons = []
    for index, shift in enumerate((0.0, 0.03, 0.10)):
        write_saliency(
            tmp_path / f"b{index}.csv",
            ids,
            values_a + shift + rng.normal(0, 0.04, len(ids)),
        )
        comparisons.append(
            saliency_comparison(f"c{index}", a="a.csv", b=f"b{index}.csv")
        )
    plan = locked_plan(tmp_path, [family("f", comparisons)])
    run_plan(plan, tmp_path / "out")
    frame, payload = read_outputs(tmp_path / "out", "f")

    assert len(frame) == 3
    assert [record["comparison_id"] for record in payload["comparisons"]] == [
        "c0",
        "c1",
        "c2",
    ]
    reject, adjusted, _, _ = multipletests(
        frame["p_raw"].to_numpy(), alpha=0.05, method="fdr_bh"
    )
    assert frame["p_bh"].to_numpy() == pytest.approx(adjusted)
    assert frame["reject"].tolist() == reject.tolist()


# --------------------------------------------------------------------------- saliency test


def test_registered_zero_method_is_used(tmp_path):
    ids = [f"i{index:02d}" for index in range(12)]
    diffs = np.array([0, 0, 0, 1, 2, 3, -1, -2, -3, 4, 5, -4], dtype=float) * 0.01
    write_saliency(tmp_path / "a.csv", ids, [0.1] * len(ids))
    write_saliency(tmp_path / "b.csv", ids, 0.1 + diffs)

    wilcox_plan = locked_plan(
        tmp_path,
        [family("f_wilcox", [saliency_comparison("cmp", zero_method="wilcox")])],
        name="plan_wilcox.json",
    )
    pratt_plan = locked_plan(
        tmp_path,
        [family("f_pratt", [saliency_comparison("cmp", zero_method="pratt")])],
        name="plan_pratt.json",
    )
    run_plan(wilcox_plan, tmp_path / "out_wilcox")
    run_plan(pratt_plan, tmp_path / "out_pratt")

    wilcox_frame, wilcox_payload = read_outputs(tmp_path / "out_wilcox", "f_wilcox")
    pratt_frame, pratt_payload = read_outputs(tmp_path / "out_pratt", "f_pratt")
    assert wilcox_payload["comparisons"][0]["zero_method"] == "wilcox"
    assert pratt_payload["comparisons"][0]["zero_method"] == "pratt"
    assert wilcox_payload["comparisons"][0]["n_zero_diffs"] == 3
    # scipy on this hand-built vector: wilcox W=18 p=0.5927, pratt W=30 p=0.6337.
    assert wilcox_frame["p_raw"].iloc[0] != pratt_frame["p_raw"].iloc[0]
    assert wilcox_frame["statistic"].iloc[0] != pratt_frame["statistic"].iloc[0]


# --------------------------------------------------------------------------- p-value method


def run_saliency_plan(
    tmp_path,
    differences,
    zero_method="wilcox",
    p_method="permutation",
    out="out",
):
    write_paired_differences(tmp_path, differences)
    plan = locked_plan(
        tmp_path,
        [
            family(
                "f",
                [saliency_comparison("cmp", zero_method=zero_method, p_method=p_method)],
            )
        ],
    )
    run_plan(plan, tmp_path / out)
    return read_outputs(tmp_path / out, "f")


def test_permutation_matches_exhaustive_sign_flip_reference(tmp_path):
    # 12 positive / 4 negative differences, all |b - a| equal (one tie group of 16):
    # the conditional test counts 2 * (C(16,0)+..+C(16,4)) = 5034 of 2^16 sign patterns
    # with |W+ - mean| >= |102 - 68| = 2 * P(Bin(16, .5) >= 12).
    differences = [0.5] * 12 + [-0.5] * 4
    frame, payload = run_saliency_plan(tmp_path, differences)
    record = payload["comparisons"][0]

    assert frame["p_raw"].iloc[0] == 0.076812744140625
    assert frame["statistic"].iloc[0] == 34.0
    assert record["p_method"] == "permutation"
    assert record["n"] == 16 and record["n_zero_diffs"] == 0 and record["n_ties"] == 16
    # SciPy 1.10.1 `method="exact"` applies a tie correction here (p = 0.0833): the
    # registered test is the exhaustive sign-flip test, never scipy's default.
    assert frame["p_raw"].iloc[0] != pytest.approx(
        wilcoxon(differences, alternative="two-sided", method="exact").pvalue
    )


def test_permutation_drops_zero_differences_like_wilcox(tmp_path):
    # 4 zero + 8(+1) + 4(-1) with the registered `wilcox` rule: n = 12 non-zero
    # differences, ranks all tied at 6.5, W+ = 52 vs mean 39 -> 2 * (C(12,0)+..+C(12,4))
    # = 1588 of 2^12 sign patterns are at least as extreme.
    differences = [0.0] * 4 + [0.5] * 8 + [-0.5] * 4
    frame, payload = run_saliency_plan(tmp_path, differences)
    record = payload["comparisons"][0]

    assert frame["p_raw"].iloc[0] == 0.3876953125
    assert frame["statistic"].iloc[0] == 26.0
    assert record["n"] == 16 and record["n_zero_diffs"] == 4
    # Same counts, same two-sided binomial tail as the exact McNemar on 8 vs 4.
    assert frame["p_raw"].iloc[0] == pytest.approx(binomtest(8, 12, 0.5).pvalue)


def test_permutation_equals_scipy_exact_without_ties(tmp_path):
    differences = (
        np.array([1, -2, 3, 4, -5, 6, 7, -8, 9, 10, -11, 12, 13, -14, 15, 16]) / 64.0
    )
    assert np.unique(np.abs(differences)).size == differences.size
    frame, payload = run_saliency_plan(tmp_path, differences)

    expected = wilcoxon(differences, alternative="two-sided", method="exact")
    # The JSON keeps full float precision; pandas' CSV reader may move the last bit.
    assert payload["comparisons"][0]["p_raw"] == expected.pvalue
    assert payload["comparisons"][0]["statistic"] == expected.statistic
    assert frame["p_raw"].iloc[0] == pytest.approx(expected.pvalue, abs=1e-12)


def test_permutation_rejects_cohort_above_the_limit(tmp_path):
    differences = [0.5] * 11 + [-0.5] * 10
    write_paired_differences(tmp_path, differences)
    plan = locked_plan(
        tmp_path, [family("f", [saliency_comparison("cmp", p_method="permutation")])]
    )
    with pytest.raises(RuntimeError, match="use p_method 'approx' for large cohorts"):
        run_plan(plan, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_approx_path_is_the_explicit_normal_approximation(tmp_path):
    # The same tied 12/4 vector: `auto` (and `exact`) report 0.0833 on SciPy 1.10.1,
    # the registered `approx` method reports the normal approximation 0.0455.
    differences = [0.5] * 12 + [-0.5] * 4
    frame, payload = run_saliency_plan(tmp_path, differences, p_method="approx")
    record = payload["comparisons"][0]

    expected = wilcoxon(
        differences, zero_method="wilcox", alternative="two-sided", method="approx"
    )
    # The JSON keeps full float precision; pandas' CSV reader may move the last bit.
    assert record["p_raw"] == expected.pvalue
    assert record["statistic"] == expected.statistic
    assert frame["p_raw"].iloc[0] == pytest.approx(0.04550026389635839, abs=1e-12)
    assert frame["p_raw"].iloc[0] < 0.05  # never the 0.0833 `auto`/`exact` value
    assert record["p_method"] == "approx"


def test_mcnemar_exact_counts_only_discordant_pairs(tmp_path):
    # Binary hit, 16 images: b = 5 hits gained, c = 1 hit lost, 10 concordant pairs.
    ids = [f"p{index:02d}" for index in range(16)]
    values_a = np.zeros(16)
    values_b = np.zeros(16)
    values_a[5] = 1.0
    values_b[:5] = 1.0
    write_saliency(tmp_path / "a.csv", ids, values_a)
    write_saliency(tmp_path / "b.csv", ids, values_b)
    plan = locked_plan(
        tmp_path,
        [family("f", [saliency_comparison("cmp", p_method="mcnemar_exact")])],
    )
    run_plan(plan, tmp_path / "out")
    frame, payload = read_outputs(tmp_path / "out", "f")
    record = payload["comparisons"][0]

    # min(b, c) = 1 of 6 discordant pairs: 2 * P(Bin(6, .5) <= 1) = 14/64.
    assert frame["p_raw"].iloc[0] == 0.21875
    assert frame["statistic"].iloc[0] == 1.0
    assert frame["estimate"].iloc[0] == pytest.approx(4.0 / 16.0)
    # A zero difference is a concordant pair: `zero_method` is required by the kind but
    # not used by this test, and the 10 zeros are the concordant pairs.
    assert record["n"] == 16 and record["n_zero_diffs"] == 10
    assert record["zero_method"] == "wilcox"
    assert record["p_method"] == "mcnemar_exact"
    assert record["degenerate"] is False


def test_mcnemar_exact_all_concordant_pairs_is_degenerate(tmp_path):
    # Binary hits that agree on every image: no discordant pair, so p = 1 by definition.
    ids = [f"i{index:02d}" for index in range(12)]
    hits = [index % 2 for index in range(12)]
    write_saliency(tmp_path / "a.csv", ids, hits)
    write_saliency(tmp_path / "b.csv", ids, hits)
    plan = locked_plan(
        tmp_path,
        [family("f", [saliency_comparison("cmp", p_method="mcnemar_exact")])],
    )
    run_plan(plan, tmp_path / "out")
    frame, payload = read_outputs(tmp_path / "out", "f")
    record = payload["comparisons"][0]
    assert frame["p_raw"].iloc[0] == 1.0
    assert frame["statistic"].iloc[0] == 0.0
    assert record["degenerate"] is True


def test_approx_all_zero_differences_is_degenerate(tmp_path):
    frame, payload = run_saliency_plan(tmp_path, [0.0] * 12, p_method="approx")
    record = payload["comparisons"][0]
    assert frame["p_raw"].iloc[0] == 1.0
    assert frame["statistic"].iloc[0] == 0.0
    assert record["degenerate"] is True


def test_mcnemar_exact_rejects_non_binary_values(tmp_path):
    ids = ["i1", "i2", "i3"]
    write_saliency(tmp_path / "a.csv", ids, [0.0, 0.0, 0.0])
    write_saliency(tmp_path / "b.csv", ids, [1.0, 0.0, 0.5])
    plan = locked_plan(
        tmp_path,
        [family("f", [saliency_comparison("cmp", p_method="mcnemar_exact")])],
    )
    with pytest.raises(RuntimeError, match=r"binary.*found 0\.5 in input b"):
        run_plan(plan, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_mcnemar_exact_rejects_a_constant_non_binary_metric(tmp_path):
    # Constant 0.5 on both sides has only concordant (zero) differences; the binary
    # requirement is still checked instead of reporting the degenerate p = 1.
    ids = ["i1", "i2", "i3"]
    write_saliency(tmp_path / "a.csv", ids, [0.5, 0.5, 0.5])
    write_saliency(tmp_path / "b.csv", ids, [0.5, 0.5, 0.5])
    plan = locked_plan(
        tmp_path,
        [family("f", [saliency_comparison("cmp", p_method="mcnemar_exact")])],
    )
    with pytest.raises(RuntimeError, match=r"binary.*found 0\.5 in input a"):
        run_plan(plan, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_output_rows_and_json_record_the_registered_p_method(tmp_path):
    differences = [0.5] * 12 + [-0.5] * 4
    frame, payload = run_saliency_plan(tmp_path, differences)
    record = payload["comparisons"][0]

    assert "p_method" in OUTPUT_COLUMNS
    assert list(frame.columns) == OUTPUT_COLUMNS
    assert frame["p_method"].tolist() == ["permutation"]
    for key in ("n", "n_zero_diffs", "n_ties", "tie_quantity", "p_method"):
        assert key in record
    assert record["tie_quantity"] == "|b - a| (non-zero)"


# --------------------------------------------------------------------------- row order


def test_row_order_does_not_change_results_and_hash_is_stamped(tmp_path):
    rng = np.random.default_rng(3)
    ids = [f"img{index:03d}" for index in range(24)]
    values_a = rng.uniform(0.10, 0.60, len(ids))
    values_b = values_a + rng.normal(0.03, 0.04, len(ids))
    write_saliency(tmp_path / "a.csv", ids, values_a)
    write_saliency(tmp_path / "b.csv", ids, values_b)
    plan = locked_plan(tmp_path, [family("f", [saliency_comparison("cmp")])])
    digest = plan_sha256(plan)
    run_plan(plan, tmp_path / "out_sorted")

    shuffle = rng.permutation(len(ids))
    write_saliency(tmp_path / "a.csv", [ids[i] for i in shuffle], values_a[shuffle])
    write_saliency(tmp_path / "b.csv", [ids[i] for i in shuffle], values_b[shuffle])
    run_plan(plan, tmp_path / "out_shuffled")

    sorted_frame, sorted_payload = read_outputs(tmp_path / "out_sorted", "f")
    shuffled_frame, shuffled_payload = read_outputs(tmp_path / "out_shuffled", "f")
    pd.testing.assert_frame_equal(sorted_frame, shuffled_frame)
    assert list(sorted_frame.columns) == OUTPUT_COLUMNS
    assert set(sorted_frame["plan_sha256"]) == {digest}
    assert sorted_payload["plan_sha256"] == digest
    assert sorted_payload["comparisons"][0]["n"] == len(ids)
    assert shuffled_payload["comparisons"][0]["n"] == len(ids)


def test_bootstrap_uses_registered_seed(tmp_path):
    rng = np.random.default_rng(11)
    ids = [f"i{index:03d}" for index in range(30)]
    values_a = rng.uniform(0.1, 0.7, len(ids))
    values_b = values_a + rng.normal(0.05, 0.06, len(ids))
    write_saliency(tmp_path / "a.csv", ids, values_a)
    write_saliency(tmp_path / "b.csv", ids, values_b)

    plan_seed_7 = locked_plan(
        tmp_path,
        [family("f", [saliency_comparison("cmp", seed=7, n_boot=300)])],
        name="plan_seed7.json",
    )
    plan_seed_8 = locked_plan(
        tmp_path,
        [family("f", [saliency_comparison("cmp", seed=8, n_boot=300)])],
        name="plan_seed8.json",
    )
    run_plan(plan_seed_7, tmp_path / "out_seed7a")
    run_plan(plan_seed_7, tmp_path / "out_seed7b")
    run_plan(plan_seed_8, tmp_path / "out_seed8")

    frame_7a, _ = read_outputs(tmp_path / "out_seed7a", "f")
    frame_7b, _ = read_outputs(tmp_path / "out_seed7b", "f")
    frame_8, _ = read_outputs(tmp_path / "out_seed8", "f")
    pd.testing.assert_frame_equal(frame_7a, frame_7b)
    assert frame_7a["ci_low"].iloc[0] != frame_8["ci_low"].iloc[0]


def test_cohort_filter_restricts_paired_rows(tmp_path):
    ids = ["i1", "i2", "i3", "i4"]
    write_saliency(tmp_path / "a.csv", ids, [0.2, 0.3, 0.4, 0.5], decisions=[1, 1, 0, 0])
    write_saliency(tmp_path / "b.csv", ids, [0.25, 0.35, 0.42, 0.52], decisions=[1, 1, 0, 0])
    plan = locked_plan(
        tmp_path,
        [family("f", [saliency_comparison("cmp", where={"decision": 1})])],
    )
    run_plan(plan, tmp_path / "out")
    _, payload = read_outputs(tmp_path / "out", "f")
    assert payload["comparisons"][0]["n"] == 2


# --------------------------------------------------------------------------- classification


def _classification_inputs(tmp_path):
    rng = np.random.default_rng(21)
    ids = [f"c{index:03d}" for index in range(40)]
    labels = np.array([0, 1] * 20)
    probs_a = np.clip(0.45 + 0.15 * labels + rng.normal(0, 0.08, labels.size), 0.01, 0.99)
    probs_b = np.clip(0.45 + 0.32 * labels + rng.normal(0, 0.08, labels.size), 0.01, 0.99)
    write_classification(tmp_path / "a.csv", ids, labels, probs_a)
    write_classification(tmp_path / "b.csv", ids, labels, probs_b)
    return labels, probs_a, probs_b


def test_classification_auroc_difference_and_delong(tmp_path):
    labels, probs_a, probs_b = _classification_inputs(tmp_path)
    plan = locked_plan(tmp_path, [family("f", [classification_comparison("cmp")])])
    run_plan(plan, tmp_path / "out")
    frame, payload = read_outputs(tmp_path / "out", "f")
    record = payload["comparisons"][0]

    expected = roc_auc_score(labels, probs_b) - roc_auc_score(labels, probs_a)
    assert frame["estimate"].iloc[0] == pytest.approx(expected, abs=1e-12)
    assert record["n_positive"] == 20 and record["n_negative"] == 20
    assert record["delong_degenerate"] is False
    assert 0.0 <= frame["p_raw"].iloc[0] <= 1.0
    assert frame["statistic"].iloc[0] >= 0.0
    assert frame["n"].iloc[0] == labels.size
    # The single `prob_column` contract resolves to the same column on both sides.
    assert record["inputs"]["prob_column_a"] == "prob"
    assert record["inputs"]["prob_column_b"] == "prob"


def test_classification_compares_a_score_to_b_prob(tmp_path):
    rng = np.random.default_rng(23)
    ids = [f"c{index:03d}" for index in range(40)]
    labels = np.array([0, 1] * 20)
    scores_a = np.clip(
        0.40 + 0.20 * labels + rng.normal(0, 0.08, labels.size), 0.01, 0.99
    )
    probs_b = np.clip(
        0.45 + 0.32 * labels + rng.normal(0, 0.08, labels.size), 0.01, 0.99
    )
    # A's stage-1 CSV is keyed by image_id but carries `score` and a shuffled row order;
    # alignment must use IDs, and each side must read its own registered column.
    order = rng.permutation(len(ids))
    write_classification(
        tmp_path / "a.csv",
        [ids[index] for index in order],
        labels[order],
        scores_a[order],
        prob_column="score",
    )
    write_classification(tmp_path / "b.csv", ids, labels, probs_b)
    plan = locked_plan(
        tmp_path,
        [
            family(
                "f",
                [classification_comparison("cmp", prob_columns=("score", "prob"))],
            )
        ],
    )
    run_plan(plan, tmp_path / "out")
    frame, payload = read_outputs(tmp_path / "out", "f")
    record = payload["comparisons"][0]

    expected = roc_auc_score(labels, probs_b) - roc_auc_score(labels, scores_a)
    assert frame["estimate"].iloc[0] == pytest.approx(expected, abs=1e-12)
    assert record["n"] == labels.size
    assert record["inputs"]["prob_column_a"] == "score"
    assert record["inputs"]["prob_column_b"] == "prob"


def test_classification_missing_registered_prob_column_raises(tmp_path):
    _classification_inputs(tmp_path)
    # Both CSVs carry `prob`, but the plan names A's column `score`: never guess or fall
    # back to B's column name.
    plan = locked_plan(
        tmp_path,
        [
            family(
                "f",
                [classification_comparison("cmp", prob_columns=("score", "prob"))],
            )
        ],
    )
    with pytest.raises(RuntimeError, match="column 'score' missing from input a"):
        run_plan(plan, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_classification_label_mismatch_raises(tmp_path):
    ids = [f"c{index:02d}" for index in range(10)]
    labels = np.array([0, 1] * 5)
    write_classification(tmp_path / "a.csv", ids, labels, np.linspace(0.1, 0.9, 10))
    write_classification(
        tmp_path / "b.csv", ids, 1 - labels, np.linspace(0.1, 0.9, 10)
    )
    plan = locked_plan(tmp_path, [family("f", [classification_comparison("cmp")])])
    with pytest.raises(RuntimeError, match="labels disagree"):
        run_plan(plan, tmp_path / "out")


def test_classification_requires_both_classes(tmp_path):
    ids = [f"c{index:02d}" for index in range(10)]
    labels = np.ones(10, dtype=int)
    probs = np.linspace(0.1, 0.9, 10)
    write_classification(tmp_path / "a.csv", ids, labels, probs)
    write_classification(tmp_path / "b.csv", ids, labels, probs * 0.9)
    plan = locked_plan(tmp_path, [family("f", [classification_comparison("cmp")])])
    with pytest.raises(RuntimeError, match="single class"):
        run_plan(plan, tmp_path / "out")


def test_delong_identical_predictions_is_explicit(tmp_path):
    rng = np.random.default_rng(5)
    labels = np.array([0, 1] * 20)
    probs = rng.uniform(0.05, 0.95, labels.size)
    statistic, p_value, degenerate = delong_paired(labels, probs, probs.copy())
    # Installed MLstatkit returns (nan, nan) here; the plan records the limit explicitly.
    assert degenerate is True
    assert statistic == 0.0 and p_value == 1.0

    write_classification(tmp_path / "a.csv", [f"c{i:02d}" for i in range(40)], labels, probs)
    write_classification(tmp_path / "b.csv", [f"c{i:02d}" for i in range(40)], labels, probs)
    plan = locked_plan(tmp_path, [family("f", [classification_comparison("cmp")])])
    run_plan(plan, tmp_path / "out")
    frame, payload = read_outputs(tmp_path / "out", "f")
    assert frame["p_raw"].iloc[0] == 1.0
    assert frame["p_bh"].iloc[0] == 1.0
    assert not bool(frame["reject"].iloc[0])
    assert payload["comparisons"][0]["delong_degenerate"] is True


def test_delong_zero_variance_with_nonzero_difference_raises():
    # Installed MLstatkit returns (-inf, 0.0) here: AUROC_A=1, AUROC_B=0, zero variance.
    # That is not a p=1 result; inference is undefined and must raise.
    labels = np.array([0, 0, 1, 1])
    prob_a = np.array([0.1, 0.2, 0.8, 0.9])
    prob_b = np.array([0.9, 0.8, 0.2, 0.1])
    with pytest.raises(ValueError, match="variance is zero with a nonzero AUROC difference"):
        delong_paired(labels, prob_a, prob_b)


def test_delong_normal_case_is_unchanged():
    rng = np.random.default_rng(17)
    labels = np.array([0, 1] * 25)
    prob_a = np.clip(0.5 + 0.15 * labels + rng.normal(0, 0.10, labels.size), 0.01, 0.99)
    prob_b = np.clip(0.5 + 0.35 * labels + rng.normal(0, 0.10, labels.size), 0.01, 0.99)
    statistic, p_value, degenerate = delong_paired(labels, prob_a, prob_b)
    assert degenerate is False
    assert np.isfinite(statistic) and statistic > 0.0
    assert 0.0 < p_value < 1.0


# --------------------------------------------------------------------------- patient unit


def _repeated_patient_frame():
    """20 patients x 2 images; both images of a patient share its label and its scores.

    The 40 rows therefore carry 20 independent observations: an image-level bootstrap
    understates the uncertainty that a patient-level cluster bootstrap keeps.
    """
    rng = np.random.default_rng(13)
    labels = np.repeat(np.array([0, 1] * 10), 2)
    probs_a = np.repeat(np.clip(rng.uniform(0.20, 0.50, 20), 0.01, 0.99), 2)
    probs_b = np.repeat(np.clip(rng.uniform(0.40, 0.90, 20), 0.01, 0.99), 2)
    patient_ids = np.repeat([f"p{index:02d}" for index in range(20)], 2)
    return labels, probs_a, probs_b, patient_ids


def _patient_classification_inputs(tmp_path):
    labels, probs_a, probs_b, patient_ids = _repeated_patient_frame()
    ids = [f"c{index:02d}" for index in range(40)]
    write_classification(tmp_path / "a.csv", ids, labels, probs_a)
    write_classification(tmp_path / "b.csv", ids, labels, probs_b)
    return ids, patient_ids


def test_patient_unit_requires_patient_manifest():
    comparison = saliency_comparison("cmp", uncertainty_unit="patient")
    with pytest.raises(ValueError, match=r"missing key\(s\) \['patient_manifest'\]"):
        validate_plan({"plan_version": 1, "families": [family("f", [comparison])]})


def test_image_unit_rejects_patient_manifest():
    comparison = classification_comparison("cmp", patient_manifest="patients.csv")
    with pytest.raises(ValueError, match="patient_manifest is registered only with"):
        validate_plan({"plan_version": 1, "families": [family("f", [comparison])]})


def test_patient_unit_rejects_mcnemar_exact():
    comparison = saliency_comparison(
        "cmp",
        p_method="mcnemar_exact",
        uncertainty_unit="patient",
        patient_manifest="patients.csv",
    )
    with pytest.raises(ValueError, match="mcnemar_exact"):
        validate_plan({"plan_version": 1, "families": [family("f", [comparison])]})


def test_patient_manifest_hash_mismatch_raises_and_writes_nothing(tmp_path):
    ids, patient_ids = _patient_classification_inputs(tmp_path)
    manifest = write_patient_manifest(tmp_path / "patients.csv", ids, patient_ids)
    plan = locked_plan(
        tmp_path,
        [
            family(
                "f",
                [
                    classification_comparison(
                        "cmp",
                        uncertainty_unit="patient",
                        patient_manifest="patients.csv",
                    )
                ],
            )
        ],
    )
    # A frozen manifest is never reloaded after a byte changed.
    with open(manifest, "a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        run_plan(plan, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_patient_manifest_requires_image_and_patient_columns(tmp_path):
    write_saliency(tmp_path / "a.csv", ["i1", "i2"], [0.2, 0.3])
    write_saliency(tmp_path / "b.csv", ["i1", "i2"], [0.25, 0.35])
    load_or_create_frozen(
        tmp_path / "patients.csv",
        lambda: pd.DataFrame({"image_id": ["i1", "i2"]}),
        {"description": "manifest without patient_id"},
    )
    comparison = saliency_comparison(
        "cmp",
        uncertainty_unit="patient",
        patient_manifest="patients.csv",
        p_method="permutation",
    )
    plan = locked_plan(tmp_path, [family("f", [comparison])])
    with pytest.raises(RuntimeError, match=r"missing column\(s\) \['patient_id'\]"):
        run_plan(plan, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_patient_manifest_rejects_duplicate_image_ids(tmp_path):
    ids, patient_ids = _patient_classification_inputs(tmp_path)
    duplicate = ids + [ids[0]]
    load_or_create_frozen(
        tmp_path / "patients.csv",
        lambda: pd.DataFrame(
            {"image_id": duplicate, "patient_id": list(patient_ids) + ["p00"]}
        ),
        {"description": "manifest with a duplicated image_id"},
    )
    plan = locked_plan(
        tmp_path,
        [
            family(
                "f",
                [
                    classification_comparison(
                        "cmp",
                        uncertainty_unit="patient",
                        patient_manifest="patients.csv",
                    )
                ],
            )
        ],
    )
    with pytest.raises(RuntimeError, match="duplicated image_id"):
        run_plan(plan, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_patient_manifest_rejects_empty_patient_id(tmp_path):
    ids, patient_ids = _patient_classification_inputs(tmp_path)
    empty = np.asarray(patient_ids, dtype=object).copy()
    empty[3] = ""
    write_patient_manifest(tmp_path / "patients.csv", ids, empty)
    plan = locked_plan(
        tmp_path,
        [
            family(
                "f",
                [
                    classification_comparison(
                        "cmp",
                        uncertainty_unit="patient",
                        patient_manifest="patients.csv",
                    )
                ],
            )
        ],
    )
    with pytest.raises(RuntimeError, match="empty patient_id"):
        run_plan(plan, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_unmapped_aligned_image_raises_and_writes_nothing(tmp_path):
    ids, patient_ids = _patient_classification_inputs(tmp_path)
    # The manifest covers the cohort except the last image: every aligned image must map.
    write_patient_manifest(
        tmp_path / "patients.csv", ids[:-1], patient_ids[:-1]
    )
    plan = locked_plan(
        tmp_path,
        [
            family(
                "f",
                [
                    classification_comparison(
                        "cmp",
                        uncertainty_unit="patient",
                        patient_manifest="patients.csv",
                    )
                ],
            )
        ],
    )
    with pytest.raises(RuntimeError, match="no patient_id in the frozen manifest"):
        run_plan(plan, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def _draw_segments(draw, blocks):
    """Split a cluster draw into its whole-patient blocks, in draw order."""
    segments = []
    position = 0
    while position < draw.size:
        for block in blocks:
            end = position + block.size
            if end <= draw.size and np.array_equal(draw[position:end], block):
                segments.append(block)
                position = end
                break
        else:
            raise AssertionError(f"draw is not a concatenation of patient blocks: {draw}")
    return segments


def test_cluster_draw_keeps_whole_patient_groups():
    patient_ids = np.array(["p1", "p1", "p2", "p3", "p3", "p3", "p4"])
    blocks = patient_blocks(patient_ids)
    assert [list(block) for block in blocks] == [[0, 1], [2], [3, 4, 5], [6]]

    rng = np.random.default_rng(0)
    for _ in range(25):
        draw = cluster_draw(blocks, rng)
        segments = _draw_segments(draw, blocks)
        # Exactly n_patients clusters are resampled, each one a whole patient: no image
        # of a drawn patient is ever dropped, so within-patient dependence survives.
        assert len(segments) == len(blocks)
        assert np.array_equal(np.concatenate(segments), draw)
        for block in segments:
            assert len(set(patient_ids[block])) == 1


def test_cluster_draw_is_reproducible_from_the_registered_seed():
    patient_ids = np.array(["p1", "p1", "p2", "p2", "p3", "p4", "p4"])
    blocks = patient_blocks(patient_ids)
    first = [cluster_draw(blocks, np.random.default_rng(3)) for _ in range(4)]
    second = [cluster_draw(blocks, np.random.default_rng(3)) for _ in range(4)]
    for draw_a, draw_b in zip(first, second):
        assert np.array_equal(draw_a, draw_b)
        assert len(_draw_segments(draw_a, blocks)) == len(blocks)


def test_cluster_bootstrap_ci_is_at_least_as_wide_as_image_bootstrap():
    labels, probs_a, probs_b, patient_ids = _repeated_patient_frame()
    image = bootstrap_auroc_diff_ci(labels, probs_a, probs_b, n_boot=400, seed=5)
    cluster = bootstrap_auroc_diff_ci_clustered(
        labels, probs_a, probs_b, patient_ids, n_boot=400, seed=5
    )
    image_width = image["ci_high"] - image["ci_low"]
    cluster_width = cluster["ci_high"] - cluster["ci_low"]
    # 40 rows carry 20 independent observations; resampling whole patients keeps that
    # dependence, so the interval is never narrower than the independent-image one.
    assert cluster_width >= image_width
    assert cluster_width > 1.2 * image_width
    assert cluster["n"] == 400 and cluster["n_used"] == 400
    assert cluster["ci_low"] <= cluster["ci_high"]


def test_classification_patient_unit_reports_cluster_fields(tmp_path):
    ids, patient_ids = _patient_classification_inputs(tmp_path)
    labels, probs_a, probs_b, _ = _repeated_patient_frame()
    manifest = write_patient_manifest(tmp_path / "patients.csv", ids, patient_ids)
    plan = locked_plan(
        tmp_path,
        [
            family(
                "f",
                [
                    classification_comparison(
                        "cmp",
                        uncertainty_unit="patient",
                        patient_manifest="patients.csv",
                        n_boot=300,
                        seed=11,
                    )
                ],
            )
        ],
    )
    run_plan(plan, tmp_path / "out")
    frame, payload = read_outputs(tmp_path / "out", "f")
    record = payload["comparisons"][0]

    assert record["uncertainty_unit"] == "patient"
    assert record["n"] == 40 and record["n_images"] == 40
    assert record["n_patients"] == 20
    assert record["n_multi_image_patients"] == 20
    assert record["max_images_per_patient"] == 2
    assert record["patient_manifest"] == str(manifest.resolve())
    assert Path(record["patient_manifest"]).is_absolute()
    assert record["inputs"]["patient_manifest"] == record["patient_manifest"]
    digest = sha256_file(manifest)
    assert record["patient_manifest_sha256"] == digest

    # The pooled paired difference and the image-level paired DeLong p are unchanged:
    # only the interval becomes a cluster bootstrap, and the record says the p-value
    # does not model the clustering.
    expected = roc_auc_score(labels, probs_b) - roc_auc_score(labels, probs_a)
    assert record["estimate"] == pytest.approx(expected, abs=1e-12)
    _, delong_p, _ = delong_paired(labels, probs_a, probs_b)
    assert record["p_raw"] == delong_p
    assert record["p_clustering_aware"] is False

    for column in (
        "uncertainty_unit",
        "n_images",
        "n_patients",
        "n_multi_image_patients",
        "patient_manifest_sha256",
        "p_clustering_aware",
    ):
        assert column in OUTPUT_COLUMNS and column in frame.columns
    assert frame["uncertainty_unit"].tolist() == ["patient"]
    assert int(frame["n_images"].iloc[0]) == 40
    assert int(frame["n_patients"].iloc[0]) == 20
    assert int(frame["n_multi_image_patients"].iloc[0]) == 20
    assert frame["patient_manifest_sha256"].tolist() == [digest]
    assert frame["p_clustering_aware"].tolist() == [False]


def test_patient_manifest_is_recorded_as_an_absolute_resolved_path(tmp_path, monkeypatch):
    # Run the plan from inside its own directory with a relative plan path: the recorded
    # manifest must still be the absolute file a reader can open, not "./patients.csv".
    write_saliency(tmp_path / "a.csv", ["i1", "i2"], [0.2, 0.3])
    write_saliency(tmp_path / "b.csv", ["i1", "i2"], [0.25, 0.35])
    manifest = write_patient_manifest(tmp_path / "patients.csv", ["i1", "i2"], ["p1", "p1"])
    comparison = saliency_comparison(
        "cmp",
        uncertainty_unit="patient",
        patient_manifest="patients.csv",
        p_method="permutation",
    )
    write_plan(tmp_path, [family("f", [comparison])])
    monkeypatch.chdir(tmp_path)
    lock_plan(Path("plan.json"))
    run_plan(Path("plan.json"), Path("out"))

    _, payload = read_outputs(tmp_path / "out", "f")
    record = payload["comparisons"][0]
    assert record["patient_manifest"] == str(manifest.resolve())
    assert record["n_patients"] == 1 and record["max_images_per_patient"] == 2


def test_image_unit_rows_report_no_patient_fields(tmp_path):
    _classification_inputs(tmp_path)
    plan = locked_plan(tmp_path, [family("f", [classification_comparison("cmp")])])
    run_plan(plan, tmp_path / "out")
    frame, payload = read_outputs(tmp_path / "out", "f")
    record = payload["comparisons"][0]
    assert record["uncertainty_unit"] == "image"
    assert record["n_images"] == record["n"] == 40
    assert record["n_patients"] is None
    assert record["n_multi_image_patients"] is None
    assert record["max_images_per_patient"] is None
    assert record["patient_manifest"] is None
    assert record["patient_manifest_sha256"] is None
    assert record["p_clustering_aware"] is None
    assert pd.isna(frame["n_patients"].iloc[0])
    assert pd.isna(frame["p_clustering_aware"].iloc[0])


def test_saliency_patient_collapse_uses_patient_mean_differences(tmp_path):
    # Three patients, one with two images. Image differences: -0.5, +0.5 (patient p1,
    # mean 0.0), +0.25 (p2), +0.25 (p3). The registered estimand is the mean of the
    # patient-mean differences = (0.0 + 0.25 + 0.25) / 3 = 1/6, never the image mean
    # (-0.5 + 0.5 + 0.25 + 0.25) / 4 = 0.125.
    ids = ["i1", "i2", "i3", "i4"]
    write_saliency(tmp_path / "a.csv", ids, [0.75, 0.25, 0.50, 0.50])
    write_saliency(tmp_path / "b.csv", ids, [0.25, 0.75, 0.75, 0.75])
    write_patient_manifest(tmp_path / "patients.csv", ids, ["p1", "p1", "p2", "p3"])
    comparison = saliency_comparison(
        "cmp",
        uncertainty_unit="patient",
        patient_manifest="patients.csv",
        p_method="permutation",
        n_boot=300,
    )
    plan = locked_plan(tmp_path, [family("f", [comparison])])
    run_plan(plan, tmp_path / "out")
    frame, payload = read_outputs(tmp_path / "out", "f")
    record = payload["comparisons"][0]

    assert record["n"] == 3 and record["n_images"] == 4
    assert record["n_patients"] == 3
    assert record["n_multi_image_patients"] == 1
    assert record["max_images_per_patient"] == 2
    assert record["estimate"] == pytest.approx(1.0 / 6.0)
    assert record["estimate"] != pytest.approx(0.125)
    # Patient-level signed ranks: the 0.0 patient difference is dropped (wilcox), the
    # two 0.25 values tie at rank 1.5 -> 2 of the 4 sign patterns are at least as extreme.
    assert record["p_raw"] == 0.5
    assert record["statistic"] == 0.0
    assert record["n_zero_diffs"] == 1
    assert record["n_ties"] == 2
    assert record["tie_quantity"] == "patient-mean |b - a| (non-zero)"
    assert record["p_clustering_aware"] is True
    assert record["ci_low"] <= record["estimate"] <= record["ci_high"]
    assert frame["n"].iloc[0] == 3


def test_example_plan_registers_a_patient_unit_comparison_per_kind():
    plan = validate_plan(json.loads(EXAMPLE_PLAN.read_text(encoding="utf-8")))
    comparisons = [
        comparison
        for registered_family in plan["families"]
        for comparison in registered_family["comparisons"]
    ]
    patient = [item for item in comparisons if item["uncertainty_unit"] == "patient"]
    assert {item["kind"] for item in patient} == {"saliency_paired", "classification_auroc"}
    assert all(item["inputs"]["patient_manifest"].endswith(".csv") for item in patient)
    assert all(
        item["inputs"]["patient_manifest"] is None
        for item in comparisons
        if item["uncertainty_unit"] == "image"
    )

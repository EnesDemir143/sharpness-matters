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
from sklearn.metrics import roc_auc_score

from sharpness_matters.ptx.corrected.analysis import (
    OUTPUT_COLUMNS,
    align_by_id,
    benjamini_hochberg,
    delong_paired,
    lock_plan,
    plan_sha256,
    run_plan,
    validate_plan,
)

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


def write_classification(path, ids, labels, probs):
    pd.DataFrame(
        {"image_id": list(ids), "label": list(labels), "prob": list(probs)}
    ).to_csv(path, index=False)
    return path


def saliency_comparison(
    comparison_id,
    a="a.csv",
    b="b.csv",
    metric_column="iou",
    threshold=0.5,
    zero_method="wilcox",
    endpoint_class="localization",
    where=None,
    n_boot=200,
    seed=7,
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
        "uncertainty_unit": "image",
        "bootstrap": {"n": n_boot, "seed": seed},
    }
    if zero_method is not None:
        comparison["zero_method"] = zero_method
    return comparison


def classification_comparison(
    comparison_id,
    a="a.csv",
    b="b.csv",
    endpoint_class="classification",
    where=None,
    n_boot=200,
    seed=7,
):
    return {
        "id": comparison_id,
        "kind": "classification_auroc",
        "endpoint_class": endpoint_class,
        "inputs": {
            "a": a,
            "b": b,
            "prob_column": "prob",
            "label_column": "label",
        },
        "cohort": {"where": {} if where is None else where},
        "paired_difference": "b - a",
        "uncertainty_unit": "image",
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
    ]
    assert [item["endpoint_class"] for item in plan["families"]] == [
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

"""Joint coverage keeps the K1 denominator (PTX-04 / §7.4).

Pure scoring only: `summarize_scores` is called on hand-built per-image records, so no
GradCAM and no Lightning Trainer is constructed. In the joint cohort a wrong negative
(GT positive, decision 0) contributes 0 coverage, exactly like IoU / precision / hit,
even though its pixel coverage is still recorded in the CSV.
"""

import pytest

from sharpness_matters.ptx.utils.saliency_eval import summarize_scores

TAU = 0.5


def _row(
    image_id,
    decision,
    coverage,
    iou=0.5,
    precision=0.5,
    hit=1.0,
    pixel_auprc=0.5,
    threshold=TAU,
):
    return {
        "image_id": image_id,
        "threshold": threshold,
        "decision": decision,
        "iou": iou,
        "precision": precision,
        "coverage": coverage,
        "empty_pred_mask": 0,
        "hit": hit,
        "pixel_auprc": pixel_auprc,
    }


def test_joint_coverage_divides_by_k1_and_zeroes_the_wrong_negative():
    """3 GT positives, one wrong negative: mean over all 3, the FN contributes 0."""
    rows = [
        _row("tp1", 1, 0.6),
        _row("tp2", 1, 0.4),
        _row("fn1", 0, 0.8),  # wrong negative: its 0.8 must not be credited
    ]

    summary = summarize_scores(rows, [TAU])["0.5"]

    assert summary["tp_conditional"]["coverage_mean"] == pytest.approx(0.5)
    assert summary["tp_conditional"]["coverage_n"] == 2
    assert summary["all_positive"]["coverage_mean"] == pytest.approx(
        (0.6 + 0.4 + 0.8) / 3
    )
    assert summary["all_positive"]["coverage_n"] == 3

    joint = summary["joint"]
    assert joint["n"] == 3
    assert joint["coverage_n"] == joint["n"]
    assert joint["coverage_mean"] == pytest.approx((0.6 + 0.4 + 0.0) / 3)
    # Neither the tp_conditional mean nor the all_positive mean.
    assert joint["coverage_mean"] != pytest.approx(0.5)
    assert joint["coverage_mean"] != pytest.approx(0.6)


def test_joint_coverage_denominator_is_k1_not_the_scored_rows():
    rows = [
        _row("tp1", 1, 1.0),
        _row("fn1", 0, 0.5),
        _row("fn2", 0, 0.5),
    ]

    joint = summarize_scores(rows, [TAU])["0.5"]["joint"]

    assert joint["coverage_n"] == joint["n"] == 3
    assert joint["coverage_mean"] == pytest.approx(1.0 / 3.0)


def test_joint_coverage_without_true_positive_is_zero():
    rows = [_row("fn1", 0, 0.9)]

    summary = summarize_scores(rows, [TAU])["0.5"]

    assert summary["all_positive"]["coverage_mean"] == pytest.approx(0.9)
    assert summary["joint"]["coverage_mean"] == pytest.approx(0.0)
    assert summary["joint"]["coverage_n"] == 1


def test_joint_block_schema_is_unchanged():
    summary = summarize_scores([_row("tp1", 1, 0.6)], [TAU])["0.5"]

    assert set(summary["joint"]) == {
        "n",
        "iou_mean",
        "iou_std",
        "precision_mean",
        "precision_std",
        "coverage_mean",
        "coverage_n",
        "hit_mean",
        "pixel_auprc_mean",
    }
    assert set(summary["all_positive"]) == set(summary["tp_conditional"]) == (
        set(summary["joint"])
    )

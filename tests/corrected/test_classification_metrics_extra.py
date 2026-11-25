"""Behavioral tests for classification AUPRC, sensitivity and specificity.

These are isolated metric tests: the helper is called with hand-written probabilities,
and `eval_res` is exercised with inference monkeypatched away. They lock the protocol
conventions: AUPRC is sklearn average precision (not trapezoidal PR area), decisions are
`p >= tau` (inclusive), and a single-class cohort yields `None`, never a fabricated
AUROC/AUPRC/specificity. The metric dict must also survive strict JSON and CSV output.
"""

import json
import logging
import math

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import auc, average_precision_score, precision_recall_curve

from sharpness_matters.ptx.utils import classification_eval

LOGGER = logging.getLogger("test_classification_metrics_extra")

# Ties at 0.4 (one positive, one negative) and 0.6 (one positive, one negative).
LABELS = np.array([0, 1, 1, 0, 1, 0])
PROBS = np.array([0.2, 0.8, 0.4, 0.4, 0.6, 0.6])
TAU = 0.6


class _UnusedLoader:
    """Stand-in for DataLoader; inference is monkeypatched, so it is never iterated."""

    def __init__(self, *args, **kwargs):
        pass


class _SizeOnlyDataset:
    def __init__(self, size):
        self._size = int(size)

    def __len__(self):
        return self._size

    def __getitem__(self, index):
        raise AssertionError("evaluation loader must not be iterated in this test")


class _DummyModel:
    @classmethod
    def load_from_checkpoint(cls, path):
        return cls()

    def eval(self):
        return self

    def to(self, device):
        return self


def _run_eval_res(monkeypatch, tmp_path, labels, probs, tau):
    labels = np.asarray(labels, dtype=int)
    probs = np.asarray(probs, dtype=np.float64)
    checkpoint = tmp_path / "fold_0.ckpt"
    checkpoint.write_bytes(b"synthetic checkpoint for metrics only")
    monkeypatch.setattr(
        classification_eval, "fold_checkpoints", lambda record: [str(checkpoint)]
    )
    monkeypatch.setattr(classification_eval, "DataLoader", _UnusedLoader)
    monkeypatch.setattr(
        classification_eval,
        "evaluate_fold",
        lambda fold, loader, model: {
            "logits": probs.copy(),
            "labels": labels.copy(),
        },
    )
    ids = [f"id{index}" for index in range(labels.size)]
    return classification_eval.eval_res(
        _SizeOnlyDataset(labels.size),
        8,
        "resnet",
        _DummyModel,
        LOGGER,
        ids=ids,
        labels=labels,
        record={"k_folds": 1, "tau": float(tau)},
    )


def test_known_probabilities_and_ties_use_sklearn_average_precision():
    """Tied scores are one operating point; AUPRC is AP, not trapezoidal PR area."""
    metrics = classification_eval.binary_classification_metrics(LABELS, PROBS, TAU)

    # Manual step-wise AP: 1/3 (at 0.8) + 2/9 (at 0.6 tie) + 1/5 (at 0.4 tie).
    expected_ap = 1.0 / 3.0 + 2.0 / 9.0 + 1.0 / 5.0
    assert metrics["auprc"] == pytest.approx(expected_ap, abs=1e-12)
    assert metrics["auprc"] == pytest.approx(
        average_precision_score(LABELS, PROBS), abs=1e-12
    )

    # `precision_recall_curve` returns recall in decreasing order; sklearn's `auc`
    # orients the integration, so this is the positive trapezoidal PR area. It differs
    # from step-wise AP on the same tied input, which is the convention under test.
    precision, recall, _ = precision_recall_curve(LABELS, PROBS)
    trapezoid = float(auc(recall, precision))
    assert trapezoid > 0.0
    assert abs(metrics["auprc"] - trapezoid) > 1e-3

    # Ensemble cells at tau=0.6: predictions 0,1,0,0,1,1.
    assert metrics["auc"] == pytest.approx(7.0 / 9.0)
    assert metrics["acc"] == pytest.approx(4.0 / 6.0)
    assert metrics["f1"] == pytest.approx(4.0 / 6.0)
    assert metrics["sensitivity"] == pytest.approx(2.0 / 3.0)
    assert metrics["specificity"] == pytest.approx(2.0 / 3.0)
    assert metrics["prevalence"] == pytest.approx(0.5)
    assert metrics["n_pos"] == 3
    assert metrics["n_neg"] == 3


def test_tau_boundary_is_inclusive_at_the_exact_probability():
    labels = np.array([0, 1, 0, 1])
    probs = np.array([0.4, 0.6, 0.7, 0.9])

    at_boundary = classification_eval.binary_classification_metrics(labels, probs, 0.6)
    # The positive sample at exactly 0.6 must count as a true positive.
    assert at_boundary["sensitivity"] == pytest.approx(1.0)
    assert at_boundary["specificity"] == pytest.approx(0.5)

    above_boundary = classification_eval.binary_classification_metrics(
        labels, probs, 0.7
    )
    # Raising tau above 0.6 drops that positive: sensitivity halves.
    assert above_boundary["sensitivity"] == pytest.approx(0.5)
    assert above_boundary["specificity"] == pytest.approx(0.5)


def test_reported_tau_reproduces_decisions_at_full_precision(monkeypatch, tmp_path):
    """A positive sitting exactly on tau must stay positive when tau is re-applied.

    tau is one of the observed OOF probabilities, so rounding it up by even 5e-7
    (the old 6-decimal output) pushes that sample below the threshold.
    """
    tau = 0.47670790553092957
    labels = np.array([0, 0, 1, 1, 1])
    probs = np.array([0.10, 0.30, tau, 0.70, 0.90])
    assert round(tau, 6) > tau  # the case this test guards against

    _, metrics = _run_eval_res(monkeypatch, tmp_path, labels, probs, tau)

    assert metrics["tau"] == tau
    assert json.loads(json.dumps(metrics))["tau"] == tau
    reapplied = probs >= metrics["tau"]
    reproduced_true_positives = int(np.count_nonzero(reapplied & (labels == 1)))
    assert reproduced_true_positives == round(metrics["sensitivity"] * labels.sum())


def test_single_class_returns_none_for_undefined_cells():
    all_negative = classification_eval.binary_classification_metrics(
        [0, 0, 0], [0.1, 0.6, 0.9], 0.5
    )
    assert all_negative["auc"] is None
    assert all_negative["auprc"] is None
    assert all_negative["sensitivity"] is None
    assert all_negative["specificity"] == pytest.approx(1.0 / 3.0)
    assert all_negative["prevalence"] == 0.0
    assert all_negative["n_pos"] == 0
    assert all_negative["n_neg"] == 3

    all_positive = classification_eval.binary_classification_metrics(
        [1, 1, 1], [0.2, 0.6, 0.9], 0.5
    )
    # All-positive AUPRC is degenerate (precision is 1 at every recall); not 1.0.
    assert all_positive["auc"] is None
    assert all_positive["auprc"] is None
    assert all_positive["sensitivity"] == pytest.approx(2.0 / 3.0)
    assert all_positive["specificity"] is None
    assert all_positive["prevalence"] == 1.0
    assert all_positive["n_pos"] == 3
    assert all_positive["n_neg"] == 0

    for metrics in (all_negative, all_positive):
        for value in metrics.values():
            if isinstance(value, float):
                assert math.isfinite(value)

    ci = classification_eval.auroc_confidence_interval(
        np.zeros(3), np.array([0.1, 0.6, 0.9])
    )
    assert ci == (None, None)


def test_eval_res_emits_metrics_and_serializes_to_csv_and_strict_json(
    monkeypatch, tmp_path
):
    pred_df, metrics = _run_eval_res(monkeypatch, tmp_path, LABELS, PROBS, TAU)

    np.testing.assert_allclose(pred_df["prob"].to_numpy(), PROBS)
    assert list(metrics) == [
        "acc",
        "auc",
        "f1",
        "auprc",
        "sensitivity",
        "specificity",
        "prevalence",
        "n_pos",
        "n_neg",
        "auroc_ci",
        "tau",
    ]
    assert metrics["auprc"] == pytest.approx(0.7556, abs=1e-4)
    assert metrics["sensitivity"] == pytest.approx(0.6667, abs=1e-4)
    assert metrics["specificity"] == pytest.approx(0.6667, abs=1e-4)
    assert metrics["prevalence"] == pytest.approx(0.5)
    assert metrics["n_pos"] == 3 and metrics["n_neg"] == 3
    assert len(metrics["auroc_ci"]) == 2

    csv_path = tmp_path / "classification_resnet_test.csv"
    pd.DataFrame([metrics]).to_csv(csv_path, index=False)
    frame = pd.read_csv(csv_path)
    for column in (
        "auprc",
        "sensitivity",
        "specificity",
        "prevalence",
        "n_pos",
        "n_neg",
    ):
        assert column in frame.columns
    assert frame.loc[0, "auprc"] == pytest.approx(0.7556, abs=1e-4)
    assert frame.loc[0, "sensitivity"] == pytest.approx(0.6667, abs=1e-4)
    assert frame.loc[0, "specificity"] == pytest.approx(0.6667, abs=1e-4)
    assert frame.loc[0, "n_pos"] == 3

    payload = json.loads(json.dumps(metrics, allow_nan=False))
    assert payload["auprc"] == pytest.approx(0.7556, abs=1e-4)
    assert payload["sensitivity"] == pytest.approx(0.6667, abs=1e-4)
    assert payload["auroc_ci"][0] <= payload["auroc_ci"][1]


def test_eval_res_single_class_writes_null_not_fabricated_numbers(
    monkeypatch, tmp_path
):
    # E3-like positive-only cohort: no AUROC, no AUPRC, no specificity at tau.
    pred_df, metrics = _run_eval_res(
        monkeypatch, tmp_path, [1, 1, 1], [0.2, 0.6, 0.9], 0.5
    )

    assert list(pred_df["label"]) == [1, 1, 1]
    assert metrics["auc"] is None
    assert metrics["auprc"] is None
    assert metrics["specificity"] is None
    assert metrics["auroc_ci"] is None
    assert metrics["sensitivity"] == pytest.approx(0.6667, abs=1e-4)
    assert metrics["prevalence"] == 1.0
    assert metrics["n_pos"] == 3 and metrics["n_neg"] == 0

    payload = json.loads(json.dumps(metrics, allow_nan=False))
    assert payload["auc"] is None
    assert payload["auprc"] is None
    assert payload["specificity"] is None
    assert payload["auroc_ci"] is None

    csv_path = tmp_path / "classification_positive_only.csv"
    pd.DataFrame([metrics]).to_csv(csv_path, index=False)
    frame = pd.read_csv(csv_path)
    assert pd.isna(frame.loc[0, "auc"])
    assert pd.isna(frame.loc[0, "auprc"])
    assert pd.isna(frame.loc[0, "specificity"])
    assert frame.loc[0, "sensitivity"] == pytest.approx(0.6667, abs=1e-4)
    assert frame.loc[0, "n_neg"] == 0

"""Frozen-manifest integrity regressions for the B evaluation paths.

Synthetic manifests only: each test freezes a tiny csv with its `.sha256` and
`.record.json`, then checks that evaluation refuses a tampered, incomplete, swapped or
renamed artifact before any dataset/model/output work. The valid cases use a stubbed
`eval_res`/`run_experiment`, so no checkpoint, image or real cohort is needed.

Covers `sharpness_matters.ptx.evaluation.evaluate_kfold` (holdout, E2, E4) and
`sharpness_matters.ptx.evaluation.evaluate_saliency_chexlocalize` (E4 XAI). E5 is
evaluated only by run A (25 Kasım 2026 decision), so it has no path here.
"""

import hashlib
import json
import sys
import types
from pathlib import Path

import pandas as pd
import pytest
import torch
from torch.utils.data import Dataset

from sharpness_matters.ptx.evaluation import evaluate_kfold
from sharpness_matters.ptx.evaluation import evaluate_saliency_chexlocalize as eval_chx

IMG_SIZE = 512
MODEL_NAME = "resnet"


# --------------------------------------------------------------------------- helpers


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _freeze(path: Path, frame: pd.DataFrame, item: str = "test artifact") -> Path:
    """Write csv + .sha256 + .record.json the way `load_or_create_frozen` does."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    digest = _sha256(path)
    Path(str(path) + ".sha256").write_text(digest + "\n", encoding="utf-8")
    record = {"csv": str(path), "csv_sha256": digest, "item": item}
    Path(str(path) + ".record.json").write_text(json.dumps(record), encoding="utf-8")
    return path


def _explode(*args, **kwargs):
    raise AssertionError("evaluation rebuilt or consumed an artifact before verifying")


class _Cohort(Dataset):
    """Minimal dataset exposing the evaluation contract (`image_ids`, `labels`)."""

    def __init__(self, ids, labels):
        self.image_ids = [str(value) for value in ids]
        self._labels = [int(value) for value in labels]
        self.labels = dict(zip(self.image_ids, self._labels))

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        return {
            "data": torch.zeros(1, 1, 1),
            "label": torch.tensor(self._labels[index]),
            "image_id": self.image_ids[index],
        }


# ------------------------------------------------- removed E5 evaluation modes

@pytest.mark.parametrize("removed_mode", ["ood_reference_like", "ood_clean"])
def test_removed_e5_modes_are_not_valid_validation_modes(
    removed_mode, tmp_path, monkeypatch
):
    # B does not evaluate E5 (25 Kasım 2026 decision): run A owns that row, so the
    # dispatch must reject both E5 modes and name only the remaining cohorts.
    monkeypatch.setattr(evaluate_kfold, "manifest_dir", lambda: tmp_path / "manifests")
    monkeypatch.setattr(evaluate_kfold, "HOME", tmp_path / "home")
    monkeypatch.setattr(evaluate_kfold, "eval_res", _explode)

    with pytest.raises(ValueError, match="Invalid validation mode") as exc_info:
        evaluate_kfold.fetch_results_per_resolution(IMG_SIZE, MODEL_NAME, removed_mode)

    message = str(exc_info.value)
    assert removed_mode not in message
    for retained_mode in ("holdout", "vindr_test", "vindr_wide", "chexlocalize"):
        assert retained_mode in message
    assert evaluate_kfold.VALIDATION_MODES == (
        "holdout",
        "vindr_test",
        "vindr_wide",
        "chexlocalize",
    )
    assert not (tmp_path / "home" / "output").exists()


# --------------------------------------------------------------- E2 / E4 evaluation


def test_vindr_test_rejects_tampered_manifest_before_dataset(tmp_path, monkeypatch):
    root = tmp_path / "manifests"
    _freeze(
        root / "vindr_test.csv",
        pd.DataFrame({"image_id": ["a", "b"], "label": [1, 0]}),
        "E2 VinDr official test manifest",
    )
    manifest_csv = root / "vindr_test.csv"
    tampered = pd.read_csv(manifest_csv, dtype={"image_id": str})
    tampered.loc[0, "label"] = 0
    tampered.to_csv(manifest_csv, index=False)
    monkeypatch.setattr(evaluate_kfold, "manifest_dir", lambda: root)
    monkeypatch.setattr(evaluate_kfold, "HOME", tmp_path / "home")
    monkeypatch.setattr(evaluate_kfold, "VinDataset", _explode)
    monkeypatch.setattr(evaluate_kfold, "eval_res", _explode)

    with pytest.raises(RuntimeError, match="hash mismatch"):
        evaluate_kfold.fetch_results_per_resolution(
            IMG_SIZE, MODEL_NAME, "vindr_test"
        )

    assert not (tmp_path / "home" / "output").exists()


def test_chexlocalize_cls_rejects_tampered_manifest_before_dataset(
    tmp_path, monkeypatch
):
    root = tmp_path / "manifests"
    frame = pd.DataFrame(
        [
            {
                "image_id": "p1_s1_v1_frontal",
                "split": "val",
                "patient_id": "p1",
                "path": "val/p1/s1/v1_frontal.jpg",
                "label": 1,
            },
            {
                "image_id": "p2_s1_v1_frontal",
                "split": "test",
                "patient_id": "p2",
                "path": "test/p2/s1/v1_frontal.jpg",
                "label": 0,
            },
        ]
    )
    manifest_csv = _freeze(root / "chexlocalize_cls.csv", frame, "E4 cls manifest")
    tampered = pd.read_csv(manifest_csv, dtype={"image_id": str})
    tampered.loc[1, "label"] = 1
    tampered.to_csv(manifest_csv, index=False)

    class _ExplodingDataset:
        def __init__(self, *args, **kwargs):
            raise AssertionError("dataset constructed before hash verification")

    module = types.ModuleType("sharpness_matters.ptx.datasets.chexlocalize_dataset")
    module.CheXlocalizeDataset = _ExplodingDataset
    monkeypatch.setitem(
        sys.modules, "sharpness_matters.ptx.datasets.chexlocalize_dataset", module
    )
    monkeypatch.setattr(evaluate_kfold, "manifest_dir", lambda: root)
    monkeypatch.setattr(evaluate_kfold, "HOME", tmp_path / "home")
    monkeypatch.setattr(evaluate_kfold, "eval_res", _explode)

    with pytest.raises(RuntimeError, match="hash mismatch"):
        evaluate_kfold.fetch_results_per_resolution(
            IMG_SIZE, MODEL_NAME, "chexlocalize"
        )

    assert not (tmp_path / "home" / "output").exists()


def test_vindr_wide_reads_frozen_artifact_without_rebuild(tmp_path, monkeypatch):
    # `load_or_create_frozen` must never be reached: a missing artifact would then fail
    # instead of silently building the cohort from the label CSVs at evaluation time.
    root = tmp_path / "manifests"
    monkeypatch.setattr(evaluate_kfold, "manifest_dir", lambda: root)
    monkeypatch.setattr(evaluate_kfold, "HOME", tmp_path / "home")
    monkeypatch.setattr(evaluate_kfold, "VinDataset", _explode)
    monkeypatch.setattr(evaluate_kfold, "eval_res", _explode)

    with pytest.raises(RuntimeError, match="missing"):
        evaluate_kfold.fetch_results_per_resolution(
            IMG_SIZE, MODEL_NAME, "vindr_wide"
        )

    assert not (tmp_path / "home" / "output").exists()


# --------------------------------------------------------------------- E4 XAI wrapper


def _xai_frame():
    return pd.DataFrame(
        [
            {
                "image_id": "p1_s1_v1_frontal",
                "split": "val",
                "patient_id": "p1",
                "path": "val/p1/s1/v1_frontal.jpg",
                "label": 1,
            },
            {
                "image_id": "p2_s1_v1_frontal",
                "split": "test",
                "patient_id": "p2",
                "path": "test/p2/s1/v1_frontal.jpg",
                "label": 1,
            },
        ]
    )


def test_xai_wrapper_accepts_valid_frozen_manifest(tmp_path, monkeypatch):
    root = tmp_path / "manifests"
    _freeze(root / eval_chx.XAI_MANIFEST_NAME, _xai_frame(), "E4 XAI manifest")
    frame = _xai_frame()
    monkeypatch.setattr(eval_chx, "manifest_dir", lambda: root)
    monkeypatch.setattr(eval_chx, "summary_table", lambda results: [])
    calls = []

    def fake_run(dataset, img_size, model_name, dataset_name, **kwargs):
        calls.append(model_name)
        return {eval_chx.RULE_MAIN: {}, eval_chx.RULE_LEGACY: {}}

    monkeypatch.setattr(eval_chx, "run_experiment", fake_run)
    output_root = tmp_path / "out"
    results = eval_chx.run_chexlocalize_saliency(
        model_names=["resnet"],
        dataset=_Cohort(frame["image_id"].tolist(), frame["label"].tolist()),
        output_root=output_root,
    )

    assert list(results) == ["resnet"]
    assert calls == ["resnet"]
    assert (
        output_root / "saliency_thresholded_chexlocalize_resnet_main.csv"
    ).is_file()


def test_xai_wrapper_rejects_tampered_manifest_before_model_run(tmp_path, monkeypatch):
    root = tmp_path / "manifests"
    manifest_csv = _freeze(
        root / eval_chx.XAI_MANIFEST_NAME, _xai_frame(), "E4 XAI manifest"
    )
    tampered = pd.read_csv(manifest_csv, dtype={"image_id": str})
    tampered.loc[0, "label"] = 0
    tampered.to_csv(manifest_csv, index=False)
    monkeypatch.setattr(eval_chx, "manifest_dir", lambda: root)
    monkeypatch.setattr(eval_chx, "run_experiment", _explode)
    output_root = tmp_path / "out"

    with pytest.raises(RuntimeError, match="hash mismatch"):
        eval_chx.run_chexlocalize_saliency(
            model_names=["resnet"], output_root=output_root
        )

    assert not output_root.exists()


def test_xai_wrapper_rejects_missing_manifest_even_with_dataset(tmp_path, monkeypatch):
    root = tmp_path / "manifests"
    root.mkdir()
    monkeypatch.setattr(eval_chx, "manifest_dir", lambda: root)
    monkeypatch.setattr(eval_chx, "run_experiment", _explode)
    output_root = tmp_path / "out"

    with pytest.raises(RuntimeError, match="missing"):
        eval_chx.run_chexlocalize_saliency(
            model_names=["resnet"],
            dataset=_Cohort(["p1_s1_v1_frontal"], [1]),
            output_root=output_root,
        )

    assert not output_root.exists()


def test_xai_wrapper_rejects_unidentifiable_injected_dataset(tmp_path, monkeypatch):
    root = tmp_path / "manifests"
    _freeze(root / eval_chx.XAI_MANIFEST_NAME, _xai_frame(), "E4 XAI manifest")
    monkeypatch.setattr(eval_chx, "manifest_dir", lambda: root)
    monkeypatch.setattr(eval_chx, "run_experiment", _explode)
    output_root = tmp_path / "out"

    class _OpaqueDataset:
        pass

    with pytest.raises(RuntimeError, match="image_ids"):
        eval_chx.run_chexlocalize_saliency(
            model_names=["resnet"],
            dataset=_OpaqueDataset(),
            output_root=output_root,
        )

    assert not output_root.exists()


def test_xai_wrapper_rejects_dataset_order_mismatch(tmp_path, monkeypatch):
    root = tmp_path / "manifests"
    _freeze(root / eval_chx.XAI_MANIFEST_NAME, _xai_frame(), "E4 XAI manifest")
    frame = _xai_frame()
    monkeypatch.setattr(eval_chx, "manifest_dir", lambda: root)
    monkeypatch.setattr(eval_chx, "run_experiment", _explode)

    with pytest.raises(RuntimeError, match="order differs"):
        eval_chx.run_chexlocalize_saliency(
            model_names=["resnet"],
            dataset=_Cohort(
                list(reversed(frame["image_id"])), list(reversed(frame["label"]))
            ),
            output_root=tmp_path / "out",
        )

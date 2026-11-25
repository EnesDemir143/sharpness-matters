"""Frozen-manifest integrity regressions for the B evaluation paths.

Synthetic manifests only: each test freezes a tiny csv with its `.sha256` and
`.record.json`, then checks that evaluation refuses a tampered, incomplete, swapped or
renamed artifact before any dataset/model/output work. The valid cases use a stubbed
`eval_res`/`run_experiment`, so no checkpoint, image or real cohort is needed.

Covers `sharpness_matters.ptx.corrected.eval_manifests`,
`sharpness_matters.ptx.evaluation.evaluate_kfold` (E2, E2-wide, E4, E5) and
`sharpness_matters.ptx.evaluation.evaluate_saliency_chexlocalize` (E4 XAI).
"""

import hashlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import Dataset

from sharpness_matters.ptx.corrected import eval_manifests
from sharpness_matters.ptx.evaluation import evaluate_kfold
from sharpness_matters.ptx.evaluation import evaluate_saliency_chexlocalize as eval_chx

IMG_SIZE = 512
MODEL_NAME = "resnet"

E5_COMBINED = pd.DataFrame(
    {
        "image_id": ["vin_a", "vin_b", "vin_c"],
        "label": [0, 1, 0],
        "source_list": ["neg", "pos", "neg"],
    }
)
E5_CLEAN = pd.DataFrame(
    {"image_id": ["vin_a", "vin_b", "vin_d"], "label": [0, 1, 0]}
)
REFERENCE_ITEM = "PTX-03/PTX-08 E5 reference-like (unmodified author script)"
CLEAN_ITEM = "PTX-03/PTX-08 E5-clean negative manifest"
COMBINED_NAME = "reference_like_combined.csv"
CLEAN_NAME = "reference_like_clean.csv"


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


def _e5_dir(tmp_path: Path) -> Path:
    root = tmp_path / "manifests"
    _freeze(root / COMBINED_NAME, E5_COMBINED, REFERENCE_ITEM)
    _freeze(root / CLEAN_NAME, E5_CLEAN, CLEAN_ITEM)
    return root


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


def _fake_eval_res(dataset, img_size, model_name, model_cls, logger, **kwargs):
    ids = [str(value) for value in kwargs["ids"]]
    labels = [int(value) for value in kwargs["labels"]]
    probs = np.linspace(0.1, 0.9, len(ids))
    frame = pd.DataFrame(
        {
            "image_id": ids,
            "label": labels,
            "prob": probs,
            "prob_fold0": probs,
        }
    )
    return frame, {"auc": 0.5, "tau": 0.5}


# ----------------------------------------------------------------- E5 manifest binding


def test_e5_modes_load_their_canonical_frozen_manifest(tmp_path):
    root = _e5_dir(tmp_path)

    combined_path, combined = eval_manifests.load_e5_manifest(
        "ood_reference_like", None, root
    )
    clean_path, clean = eval_manifests.load_e5_manifest(
        "ood_clean", str(root / CLEAN_NAME), root
    )

    assert combined_path == root / COMBINED_NAME
    assert clean_path == root / CLEAN_NAME
    assert combined["image_id"].tolist() == E5_COMBINED["image_id"].tolist()
    assert clean["image_id"].tolist() == E5_CLEAN["image_id"].tolist()
    assert combined["label"].tolist() == [0, 1, 0]
    assert clean["label"].tolist() == [0, 1, 0]


def test_e5_unknown_mode_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="not an E5 mode"):
        eval_manifests.load_e5_manifest("holdout", None, _e5_dir(tmp_path))


def test_e5_swapped_modes_are_rejected(tmp_path):
    root = _e5_dir(tmp_path)

    with pytest.raises(RuntimeError, match="not the frozen artifact"):
        eval_manifests.load_e5_manifest(
            "ood_reference_like", str(root / CLEAN_NAME), root
        )
    with pytest.raises(RuntimeError, match="not the frozen artifact"):
        eval_manifests.load_e5_manifest(
            "ood_clean", str(root / COMBINED_NAME), root
        )


def test_e5_renamed_manifest_is_rejected(tmp_path):
    # The canonical clean path holds the reference-like cohort and record: only the
    # creation record's identity can tell the two apart once the file name is reused.
    root = tmp_path / "renamed"
    _freeze(root / COMBINED_NAME, E5_COMBINED, REFERENCE_ITEM)
    _freeze(root / CLEAN_NAME, E5_COMBINED, REFERENCE_ITEM)

    with pytest.raises(RuntimeError, match="E5-clean"):
        eval_manifests.load_e5_manifest("ood_clean", None, root)

    # A frozen copy of the clean cohort under another name is the wrong content for
    # reference-like, even though its own hash/record verify.
    copy = tmp_path / "copied" / "clean_copy.csv"
    _freeze(copy, E5_CLEAN, CLEAN_ITEM)
    with pytest.raises(RuntimeError, match="content differs"):
        eval_manifests.load_e5_manifest("ood_reference_like", str(copy), root)


def test_e5_tampered_manifest_fails_hash_check(tmp_path):
    root = _e5_dir(tmp_path)
    clean_csv = root / CLEAN_NAME
    tampered = pd.read_csv(clean_csv, dtype={"image_id": str})
    tampered.loc[0, "label"] = 1
    tampered.to_csv(clean_csv, index=False)

    with pytest.raises(RuntimeError, match="hash mismatch"):
        eval_manifests.load_e5_manifest("ood_clean", None, root)
    with pytest.raises(RuntimeError, match="hash mismatch"):
        eval_manifests.load_e5_manifest("ood_clean", str(clean_csv), root)


@pytest.mark.parametrize("suffix", [".sha256", ".record.json"])
def test_e5_missing_sidecar_is_rejected(tmp_path, suffix):
    root = _e5_dir(tmp_path)
    Path(str(root / CLEAN_NAME) + suffix).unlink()

    with pytest.raises(RuntimeError, match="missing"):
        eval_manifests.load_e5_manifest("ood_clean", None, root)


# ---------------------------------------------------------- E5 evaluation entry point


def test_e5_eval_rejects_swapped_mode_before_dataset_or_output(tmp_path, monkeypatch):
    root = _e5_dir(tmp_path)
    home = tmp_path / "home"
    monkeypatch.setattr(evaluate_kfold, "manifest_dir", lambda: root)
    monkeypatch.setattr(evaluate_kfold, "HOME", home)
    monkeypatch.setattr(evaluate_kfold, "VinDataset", _explode)
    monkeypatch.setattr(evaluate_kfold, "OODTestDataset", _explode)
    monkeypatch.setattr(evaluate_kfold, "eval_res", _explode)

    with pytest.raises(RuntimeError, match="not the frozen artifact"):
        evaluate_kfold.fetch_results_per_resolution(
            IMG_SIZE,
            MODEL_NAME,
            "ood_reference_like",
            e5_manifest=str(root / CLEAN_NAME),
        )

    assert not (home / "output").exists()


def test_e5_eval_loads_matching_frozen_manifests(tmp_path, monkeypatch):
    root = _e5_dir(tmp_path)
    home = tmp_path / "home"
    monkeypatch.setattr(evaluate_kfold, "manifest_dir", lambda: root)
    monkeypatch.setattr(evaluate_kfold, "HOME", home)
    monkeypatch.setattr(evaluate_kfold, "eval_res", _fake_eval_res)
    captured = {}

    def fake_vin(manifest_csv=None, root_dir=None, img_size=None, transform=None):
        captured["vin_manifest"] = manifest_csv
        frame = pd.read_csv(manifest_csv, dtype={"image_id": str})
        return _Cohort(frame["image_id"], frame["label"])

    def fake_ptx(img_size=None, transform=None):
        return _Cohort(["ptx_one"], [1])

    monkeypatch.setattr(evaluate_kfold, "VinDataset", fake_vin)
    monkeypatch.setattr(evaluate_kfold, "OODTestDataset", fake_ptx)

    evaluate_kfold.fetch_results_per_resolution(
        IMG_SIZE,
        MODEL_NAME,
        "ood_clean",
        e5_manifest=str(root / CLEAN_NAME),
    )

    assert captured["vin_manifest"] == str(root / CLEAN_NAME)
    ids_csv = (
        home
        / "output"
        / "classif_preds"
        / f"ood_cleanpred_{MODEL_NAME}_{IMG_SIZE}_ids.csv"
    )
    frame = pd.read_csv(ids_csv, dtype={"image_id": str})
    assert list(frame["image_id"]) == E5_CLEAN["image_id"].tolist() + ["ptx_one"]
    assert list(frame["source"]) == ["vindr"] * 3 + ["ptx498"]
    assert list(frame["label"]) == [0, 1, 0, 1]


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

"""Stage-2 (B) E2 (VinDr official test) and E4 (CheXlocalize) evaluation tests.

Synthetic only. The VinDr manifest builder is exercised on a hand-written
`image_labels_test.csv` plus empty `.dicom` files; the evaluator paths run against a
tiny LightningModule, a hand-written `cv_record.json` and fake datasets, so no real
image, label file or model weight is needed.
Check targets are named in `make_vindr_test.py` and `evaluate_kfold.py`.
"""

import hashlib
import json
import logging
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import pytorch_lightning as pl
import torch
from click.testing import CliRunner
from pytorch_lightning import LightningModule
from torch import nn
from torch.utils.data import Dataset

from sharpness_matters.config.load_config import cfg
from sharpness_matters.ptx.corrected import make_vindr_test
from sharpness_matters.ptx.evaluation import evaluate_kfold
from sharpness_matters.ptx.utils import classification_eval

LOGGER = logging.getLogger("test_e2_e4_eval")
IMG_SIZE = 512
MODEL_NAME = "resnet"
FOLD_BIASES = [0.0, 0.0, 0.0, 0.0, 0.0]

# Logits whose sigmoid probabilities are 0.6, 0.8, 0.4, 0.9; labels 0,1,0,1.
LOGITS = [0.4055, 1.3863, -0.4055, 2.1972]
LABELS = [0, 1, 0, 1]
IDS = [f"img{index}" for index in range(len(LOGITS))]


# --------------------------------------------------------------------------- fixtures


@pytest.fixture(autouse=True)
def no_worker_spawn(monkeypatch):
    """Deterministic single-process loading (the production default is 8 workers)."""
    real_loader = classification_eval.DataLoader

    def loader(*args, **kwargs):
        kwargs["num_workers"] = 0
        return real_loader(*args, **kwargs)

    monkeypatch.setattr(classification_eval, "DataLoader", loader)


@pytest.fixture
def ckpt_dir(tmp_path, monkeypatch):
    root = tmp_path / "checkpoints"
    monkeypatch.setattr(cfg.ptx.model, "checkpoint_dir", str(root))
    return root


# --------------------------------------------------------------------------- helpers


class TinyClassifier(LightningModule):
    """One-bias classifier: logit = mean of the input plane + bias."""

    def __init__(self, bias: float = 0.0):
        super().__init__()
        self.save_hyperparameters()
        self.bias = nn.Parameter(torch.tensor(float(bias)))

    def forward(self, x):
        return (x.mean(dim=(1, 2, 3)) + self.bias).unsqueeze(1)


class LogitDataset(Dataset):
    """Dataset-order logits + labels; exposes the `image_id -> label` dict of B."""

    def __init__(self, logits, labels, ids=None):
        self.logits = [float(value) for value in logits]
        self.item_labels = [int(value) for value in labels]
        self.image_ids = (
            list(ids)
            if ids is not None
            else [f"img{index}" for index in range(len(self.logits))]
        )
        self.labels = dict(zip(self.image_ids, self.item_labels))

    def __len__(self):
        return len(self.logits)

    def __getitem__(self, index):
        return {
            "data": torch.full((1, 4, 4), self.logits[index]),
            "label": torch.tensor(self.item_labels[index], dtype=torch.long),
        }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_checkpoint(path: Path, bias: float) -> None:
    model = TinyClassifier(bias=bias)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "hyper_parameters": {"bias": float(bias)},
            "epoch": 0,
            "global_step": 0,
            "pytorch-lightning_version": pl.__version__,
        },
        path,
    )


def _write_record(root: Path, biases, tau: float) -> Path:
    directory = root / MODEL_NAME / f"{MODEL_NAME}_{IMG_SIZE}_logs"
    directory.mkdir(parents=True, exist_ok=True)
    folds = []
    for fold, bias in enumerate(biases):
        checkpoint = directory / f"fold_{fold}.ckpt"
        _write_checkpoint(checkpoint, bias)
        folds.append(
            {
                "fold": fold,
                "best_epoch": 1,
                "stopped_epoch": 1,
                "checkpoint_path": str(checkpoint),
                "sha256": _sha256(checkpoint),
                "val_auroc": 0.75,
                "n_train": 4,
                "n_val": 4,
                "pos_weight": 1.0,
            }
        )
    record = {
        "model_name": MODEL_NAME,
        "img_size": IMG_SIZE,
        "k_folds": len(biases),
        "fold_manifest_sha256": "0" * 64,
        "folds": folds,
        "S_h": 0.75,
        "pooled_oof_auroc": 0.75,
        "tau": float(tau),
        "target_sensitivity": 0.95,
        "oof_csv": "oof_predictions.csv",
    }
    json_path = directory / "cv_record.json"
    json_path.write_text(json.dumps(record))
    return json_path


def _write_labels_csv(path: Path, rows) -> Path:
    pd.DataFrame(rows, columns=["image_id", "Pneumothorax"]).to_csv(path, index=False)
    return path


def _touch_dicoms(root: Path, image_ids) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for image_id in image_ids:
        (root / f"{image_id}.dicom").write_bytes(b"")
    return root


def _explode(*args, **kwargs):
    raise AssertionError("frozen manifest was rebuilt instead of loaded")


# --------------------------------------------------------------- E2 VinDr test manifest


def test_vindr_test_manifest_happy_path(tmp_path, monkeypatch):
    labels_csv = _write_labels_csv(
        tmp_path / "image_labels_test.csv", [("b", 1), ("a", 0), ("c", 1)]
    )
    test_root = _touch_dicoms(tmp_path / "vindr_test", ["a", "b", "c"])

    frame = make_vindr_test.build_vindr_test_manifest(labels_csv, test_root)

    assert list(frame.columns) == ["image_id", "label"]
    assert list(frame["image_id"]) == ["b", "a", "c"]  # csv order, nothing sorted
    assert list(frame["label"]) == [1, 0, 1]
    assert make_vindr_test.manifest_counts(frame) == {"n": 3, "n_pos": 2, "n_neg": 1}

    # Freeze once through the public loader, with the config pointing at tmp data.
    manifests = tmp_path / "manifests"
    monkeypatch.setattr(make_vindr_test, "manifest_dir", lambda: manifests)
    monkeypatch.setattr(cfg.ptx.vindr, "test_labels_csv", str(labels_csv))
    monkeypatch.setattr(cfg.ptx.vindr, "test_root_dir", str(test_root))

    frozen = make_vindr_test.load_vindr_test_manifest()

    assert list(frozen["image_id"]) == ["b", "a", "c"]
    assert list(frozen["label"]) == [1, 0, 1]
    csv_path = manifests / "vindr_test.csv"
    assert csv_path.exists()
    assert (manifests / "vindr_test.csv.sha256").exists()
    record = json.loads(
        (manifests / "vindr_test.csv.record.json").read_text(encoding="utf-8")
    )
    assert (record["n"], record["n_pos"], record["n_neg"]) == (3, 2, 1)
    assert record["label_column"] == "Pneumothorax"
    assert record["csv_sha256"] == _sha256(csv_path)

    # Load-first: with a frozen manifest present the builder is never called again.
    monkeypatch.setattr(make_vindr_test, "build_vindr_test_manifest", _explode)
    again = make_vindr_test.load_vindr_test_manifest()

    assert again.equals(frozen)


def test_vindr_test_manifest_duplicate_image_id_raises(tmp_path):
    labels_csv = _write_labels_csv(
        tmp_path / "image_labels_test.csv", [("a", 1), ("a", 0)]
    )
    test_root = _touch_dicoms(tmp_path / "vindr_test", ["a"])

    with pytest.raises(ValueError, match="more than once"):
        make_vindr_test.build_vindr_test_manifest(labels_csv, test_root)


def test_vindr_test_manifest_non_binary_label_raises(tmp_path):
    labels_csv = _write_labels_csv(
        tmp_path / "image_labels_test.csv", [("a", 1), ("b", 2)]
    )
    test_root = _touch_dicoms(tmp_path / "vindr_test", ["a", "b"])

    with pytest.raises(ValueError, match="non-binary"):
        make_vindr_test.build_vindr_test_manifest(labels_csv, test_root)


def test_vindr_test_manifest_missing_dicom_raises(tmp_path):
    labels_csv = _write_labels_csv(
        tmp_path / "image_labels_test.csv", [("a", 1), ("b", 0)]
    )
    test_root = _touch_dicoms(tmp_path / "vindr_test", ["a"])

    with pytest.raises(FileNotFoundError, match=r"no <test_root>/<image_id>\.dicom") as error:
        make_vindr_test.build_vindr_test_manifest(labels_csv, test_root)

    assert "b" in str(error.value)


# ------------------------------------------------------------------ E2 evaluation path


def test_vindr_test_mode_writes_id_csv(tmp_path, monkeypatch, ckpt_dir):
    _write_record(ckpt_dir, FOLD_BIASES, tau=0.7)
    manifests = tmp_path / "manifests"
    monkeypatch.setattr(evaluate_kfold, "manifest_dir", lambda: manifests)
    monkeypatch.setattr(evaluate_kfold, "HOME", tmp_path)
    monkeypatch.setattr(cfg.ptx.vindr, "test_root_dir", str(tmp_path / "vindr_test"))
    captured = {}

    def fake_vin(manifest_csv=None, root_dir=None, img_size=None, transform=None):
        captured.update(
            manifest_csv=manifest_csv, root_dir=root_dir, img_size=img_size
        )
        return LogitDataset(LOGITS, LABELS, ids=IDS)

    monkeypatch.setattr(evaluate_kfold, "VinDataset", fake_vin)
    monkeypatch.setattr(evaluate_kfold, "CNNBinaryClassifier", TinyClassifier)

    # E2 needs no --e5_manifest: the cohort comes from the frozen VinDr test manifest.
    metrics = evaluate_kfold.fetch_results_per_resolution(
        IMG_SIZE, MODEL_NAME, "vindr_test"
    )

    assert captured["manifest_csv"] == str(manifests / "vindr_test.csv")
    assert captured["root_dir"] == str(tmp_path / "vindr_test")
    assert captured["img_size"] == IMG_SIZE
    assert metrics["tau"] == pytest.approx(0.7)

    ids_csv = (
        tmp_path
        / "output"
        / "classif_preds"
        / f"vindr_testpred_{MODEL_NAME}_{IMG_SIZE}_ids.csv"
    )
    frame = pd.read_csv(ids_csv, dtype={"image_id": str})
    assert list(frame.columns[:4]) == ["image_id", "source", "label", "prob"]
    assert list(frame["image_id"]) == IDS
    assert list(frame["source"]) == ["vindr_test"] * len(IDS)
    assert list(frame["label"]) == LABELS
    assert "split" not in frame.columns
    assert "patient_id" not in frame.columns


# ------------------------------------------------------------------ E4 evaluation path


E4_ROWS = [
    ("patient1_study1_view1_frontal", "val", "patient1", 1),
    ("patient2_study1_view1_frontal", "test", "patient2", 0),
    ("patient3_study1_view1_frontal", "test", "patient3", 1),
    ("patient4_study1_view1_frontal", "val", "patient4", 0),
]


def _write_chexlocalize_manifest(path: Path) -> Path:
    pd.DataFrame(
        [
            {
                "image_id": image_id,
                "split": split,
                "patient_id": patient_id,
                "path": f"{split}/{patient_id}/study/view_frontal.jpg",
                "label": label,
            }
            for image_id, split, patient_id, label in E4_ROWS
        ]
    ).to_csv(path, index=False)
    return path


def _install_fake_chexlocalize(monkeypatch) -> None:
    """A stand-in for the sibling's dataset module (lazily imported in the branch)."""

    class FakeCheXlocalizeDataset(LogitDataset):
        def __init__(self, manifest_csv, img_size, with_mask=False):
            manifest = pd.read_csv(manifest_csv, dtype={"image_id": str})
            super().__init__(
                LOGITS,
                [int(value) for value in manifest["label"]],
                ids=[str(value) for value in manifest["image_id"]],
            )
            self.manifest_csv = manifest_csv
            self.img_size = img_size
            self.with_mask = with_mask

    module = types.ModuleType("sharpness_matters.ptx.datasets.chexlocalize_dataset")
    module.CheXlocalizeDataset = FakeCheXlocalizeDataset
    monkeypatch.setitem(
        sys.modules, "sharpness_matters.ptx.datasets.chexlocalize_dataset", module
    )


def test_chexlocalize_mode_keeps_split_and_patient_id(
    tmp_path, monkeypatch, ckpt_dir
):
    _write_record(ckpt_dir, FOLD_BIASES, tau=0.7)
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    _write_chexlocalize_manifest(manifests / "chexlocalize_cls.csv")
    monkeypatch.setattr(evaluate_kfold, "manifest_dir", lambda: manifests)
    monkeypatch.setattr(evaluate_kfold, "HOME", tmp_path)
    monkeypatch.setattr(evaluate_kfold, "CNNBinaryClassifier", TinyClassifier)
    _install_fake_chexlocalize(monkeypatch)

    evaluate_kfold.fetch_results_per_resolution(IMG_SIZE, MODEL_NAME, "chexlocalize")

    ids_csv = (
        tmp_path
        / "output"
        / "classif_preds"
        / f"chexlocalizepred_{MODEL_NAME}_{IMG_SIZE}_ids.csv"
    )
    frame = pd.read_csv(ids_csv, dtype={"image_id": str, "patient_id": str})
    assert list(frame.columns[:4]) == ["image_id", "source", "label", "prob"]
    assert list(frame["image_id"]) == [row[0] for row in E4_ROWS]
    assert list(frame["source"]) == [
        f"chexlocalize_{row[1]}" for row in E4_ROWS
    ]
    assert list(frame["split"]) == [row[1] for row in E4_ROWS]
    assert list(frame["patient_id"]) == [row[2] for row in E4_ROWS]
    assert list(frame["label"]) == [row[3] for row in E4_ROWS]


def test_chexlocalize_manifest_order_mismatch_raises(tmp_path):
    manifest_csv = _write_chexlocalize_manifest(tmp_path / "chexlocalize_cls.csv")

    with pytest.raises(RuntimeError, match="order mismatch"):
        evaluate_kfold._chexlocalize_cohort_columns(manifest_csv, IDS)


# -------------------------------------------------------------------------------- CLI


def test_cli_help_lists_e2_e4_modes():
    result = CliRunner().invoke(evaluate_kfold.main, ["--help"])

    assert result.exit_code == 0
    assert "vindr_test" in result.output
    assert "chexlocalize" in result.output

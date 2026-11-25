"""Stage-2 (B) E2 (VinDr-CXR) and E4 (CheXlocalize) evaluation tests.

E2 covers all 18,000 VinDr images (`vindr_wide`); `vindr_test` is its official-test
subset. Synthetic only. The VinDr manifest builder is exercised on a hand-written
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
from sklearn.metrics import roc_auc_score
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


def _freeze_manifest(path: Path, frame: pd.DataFrame, item: str) -> Path:
    """Write a csv plus the .sha256/.record.json pair load_frozen requires."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    digest = _sha256(path)
    Path(str(path) + ".sha256").write_text(digest + "\n", encoding="utf-8")
    record = {"csv": str(path), "csv_sha256": digest, "item": item}
    Path(str(path) + ".record.json").write_text(json.dumps(record), encoding="utf-8")
    return path


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


def _write_wide_train_csv(path: Path, votes_by_image) -> Path:
    """One row per (image_id, rad_id) vote, the shape of image_labels_train.csv."""
    rows = []
    for image_id, votes in votes_by_image.items():
        for index, vote in enumerate(votes):
            rows.append(
                {"image_id": image_id, "rad_id": f"R{index + 1}", "Pneumothorax": vote}
            )
    pd.DataFrame(rows).to_csv(path, index=False)
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
    assert record["rule"] == make_vindr_test.TEST_RULE
    assert record["labels_csv_sha256"] == _sha256(labels_csv)
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

    with pytest.raises(FileNotFoundError, match=r"no <root>/<image_id>\.dicom") as error:
        make_vindr_test.build_vindr_test_manifest(labels_csv, test_root)

    assert "b" in str(error.value)


# ---------------------------------- E2 all-18,000-image manifest (`vindr_wide`)

WIDE_TRAIN_VOTES = {
    "majority_pos": [1, 1, 0],  # 2 votes -> label 1, label_any 1
    "single_vote": [1, 0, 0],  # 1 vote  -> label 0, label_any 1
    "all_votes": [1, 1, 1],  # 3 votes -> label 1, label_any 1
    "negative": [0, 0, 0],  # 0 votes -> label 0, label_any 0
}
WIDE_TEST_ROWS = [("test_pos", 1), ("test_neg", 0)]


def _wide_tmp_inputs(tmp_path):
    train_csv = _write_wide_train_csv(
        tmp_path / "image_labels_train.csv", WIDE_TRAIN_VOTES
    )
    test_csv = _write_labels_csv(
        tmp_path / "image_labels_test.csv", WIDE_TEST_ROWS
    )
    train_root = _touch_dicoms(tmp_path / "train", WIDE_TRAIN_VOTES)
    test_root = _touch_dicoms(
        tmp_path / "test", [image_id for image_id, _ in WIDE_TEST_ROWS]
    )
    return train_csv, test_csv, train_root, test_root


def test_vindr_wide_manifest_rules_and_counts(tmp_path, monkeypatch):
    train_csv, test_csv, train_root, test_root = _wide_tmp_inputs(tmp_path)

    frame = make_vindr_test.build_vindr_wide_manifest(
        train_csv, test_csv, train_root, test_root
    )

    assert list(frame.columns) == make_vindr_test.WIDE_COLUMNS
    assert list(frame["split"]) == ["train"] * 4 + ["test"] * 2
    assert list(frame["image_id"]) == list(WIDE_TRAIN_VOTES) + ["test_pos", "test_neg"]
    # MAIN rule: >= 2 of 3 votes; SENSITIVITY rule: >= 1 vote.
    assert list(frame["label"]) == [1, 0, 1, 0, 1, 0]
    assert list(frame["label_any"]) == [1, 1, 1, 0, 1, 0]
    assert list(frame["n_readers"]) == [3, 3, 3, 3, 5, 5]
    assert list(frame["n_pos_votes"]) == [2, 1, 3, 0, 1, 0]
    assert make_vindr_test.wide_manifest_counts(frame) == {
        "train": {"n": 4, "label_pos": 2, "label_any_pos": 3},
        "test": {"n": 2, "label_pos": 1, "label_any_pos": 1},
        "total": {"n": 6, "label_pos": 3, "label_any_pos": 4},
    }

    # Freeze through the public loader, with config pointing at the tmp files.
    manifests = tmp_path / "manifests"
    monkeypatch.setattr(make_vindr_test, "manifest_dir", lambda: manifests)
    monkeypatch.setattr(cfg.ptx.vindr, "test_labels_csv", str(test_csv))
    monkeypatch.setattr(cfg.ptx.vindr, "root_dir", str(train_root))
    monkeypatch.setattr(cfg.ptx.vindr, "test_root_dir", str(test_root))

    frozen = make_vindr_test.load_vindr_wide_manifest()

    assert frozen.equals(frame)
    record = json.loads(
        (manifests / "vindr_wide.csv.record.json").read_text(encoding="utf-8")
    )
    assert record["main_rule"] == make_vindr_test.WIDE_MAIN_RULE
    assert record["sensitivity_rule"] == make_vindr_test.WIDE_SENSITIVITY_RULE
    assert record["train_labels_csv"] == str(train_csv)
    # The train path is derived from the test label path, never a config key.
    assert record["test_labels_csv"] == str(test_csv)
    assert record["train_labels_csv_sha256"] == _sha256(train_csv)
    assert record["test_labels_csv_sha256"] == _sha256(test_csv)
    assert record["counts"] == make_vindr_test.wide_manifest_counts(frame)
    assert record["counts"]["total"] == {"n": 6, "label_pos": 3, "label_any_pos": 4}

    # Load-first: a frozen wide manifest is never rebuilt.
    monkeypatch.setattr(make_vindr_test, "build_vindr_wide_manifest", _explode)
    assert make_vindr_test.load_vindr_wide_manifest().equals(frame)


def test_vindr_wide_reader_count_raises(tmp_path):
    train_csv = _write_wide_train_csv(
        tmp_path / "image_labels_train.csv", {"a": [1, 0], "b": [1, 1, 0]}
    )
    test_csv = _write_labels_csv(tmp_path / "image_labels_test.csv", [("t", 0)])

    with pytest.raises(ValueError, match="exactly 3 reader rows") as error:
        make_vindr_test.build_vindr_wide_manifest(
            train_csv, test_csv, tmp_path / "train", tmp_path / "test"
        )

    assert "a: 2" in str(error.value)


def test_vindr_wide_split_overlap_raises(tmp_path):
    train_csv = _write_wide_train_csv(
        tmp_path / "image_labels_train.csv", {"shared": [1, 1, 1]}
    )
    test_csv = _write_labels_csv(
        tmp_path / "image_labels_test.csv", [("shared", 1)]
    )

    with pytest.raises(ValueError, match="disjoint"):
        make_vindr_test.build_vindr_wide_manifest(
            train_csv, test_csv, tmp_path / "train", tmp_path / "test"
        )


def test_vindr_wide_missing_dicom_raises(tmp_path):
    train_csv = _write_wide_train_csv(
        tmp_path / "image_labels_train.csv", {"a": [1, 1, 1], "b": [0, 0, 0]}
    )
    test_csv = _write_labels_csv(tmp_path / "image_labels_test.csv", [("t", 0)])
    train_root = _touch_dicoms(tmp_path / "train", ["a"])
    test_root = _touch_dicoms(tmp_path / "test", ["t"])

    with pytest.raises(FileNotFoundError, match=r"VinDr train image") as error:
        make_vindr_test.build_vindr_wide_manifest(
            train_csv, test_csv, train_root, test_root
        )

    assert "b" in str(error.value)


# ------------------------------------------------------------------ E2 evaluation path


def test_vindr_test_mode_writes_id_csv(tmp_path, monkeypatch, ckpt_dir):
    _write_record(ckpt_dir, FOLD_BIASES, tau=0.7)
    manifests = tmp_path / "manifests"
    _freeze_manifest(
        manifests / "vindr_test.csv",
        pd.DataFrame({"image_id": IDS, "label": LABELS}),
        "E2 VinDr official test manifest",
    )
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

    # E2 reads its cohort from the frozen VinDr official-test manifest (`vindr_test`).
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
    cls_path = _write_chexlocalize_manifest(manifests / "chexlocalize_cls.csv")
    _freeze_manifest(
        cls_path,
        pd.read_csv(cls_path, dtype={"image_id": str}),
        "E4 CheXlocalize cls manifest",
    )
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


# --------------------------- E2 wide eval path (all 18,000 images, `vindr_wide`)

WIDE_LOGIT_BY_ID = {
    "majority_pos": 2.0,
    "single_vote": 1.5,
    "all_votes": 1.0,
    "negative": -2.0,
    "test_pos": 1.0,
    "test_neg": -1.0,
}


def _fake_wide_manifest() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "image_id": list(WIDE_LOGIT_BY_ID),
            "split": ["train"] * 4 + ["test"] * 2,
            "label": [1, 0, 1, 0, 1, 0],
            "label_any": [1, 1, 1, 0, 1, 0],
            "n_readers": [3, 3, 3, 3, 5, 5],
            "n_pos_votes": [2, 1, 3, 0, 1, 0],
        }
    )


def test_vindr_wide_mode_writes_both_labels(tmp_path, monkeypatch, ckpt_dir):
    _write_record(ckpt_dir, FOLD_BIASES, tau=0.7)
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    wide = _fake_wide_manifest()
    _freeze_manifest(manifests / "vindr_wide.csv", wide, "E2-wide manifest")
    monkeypatch.setattr(evaluate_kfold, "manifest_dir", lambda: manifests)
    monkeypatch.setattr(evaluate_kfold, "HOME", tmp_path)
    monkeypatch.setattr(evaluate_kfold, "CNNBinaryClassifier", TinyClassifier)
    captured = []

    def fake_vin(manifest_csv=None, root_dir=None, img_size=None, transform=None):
        part = pd.read_csv(manifest_csv, dtype={"image_id": str})
        image_ids = part["image_id"].tolist()
        captured.append((manifest_csv, root_dir, image_ids))
        return LogitDataset(
            [WIDE_LOGIT_BY_ID[image_id] for image_id in image_ids],
            part["label"].astype(int).tolist(),
            ids=image_ids,
        )

    monkeypatch.setattr(evaluate_kfold, "VinDataset", fake_vin)

    metrics = evaluate_kfold.fetch_results_per_resolution(
        IMG_SIZE, MODEL_NAME, "vindr_wide"
    )

    # One dataset per split, each with its own DICOM root, in train -> test order.
    assert [item[1] for item in captured] == [
        str(cfg.ptx.vindr.root_dir),
        str(cfg.ptx.vindr.test_root_dir),
    ]
    assert [item[2] for item in captured] == [
        ["majority_pos", "single_vote", "all_votes", "negative"],
        ["test_pos", "test_neg"],
    ]
    # The per-split csv views the loader needed are temporary and already removed.
    assert all(not Path(item[0]).exists() for item in captured)

    ids_csv = (
        tmp_path
        / "output"
        / "classif_preds"
        / f"vindr_widepred_{MODEL_NAME}_{IMG_SIZE}_ids.csv"
    )
    frame = pd.read_csv(ids_csv, dtype={"image_id": str})
    assert list(frame.columns) == (
        ["image_id", "source", "split", "label", "label_any", "prob"]
        + [f"prob_fold{fold}" for fold in range(len(FOLD_BIASES))]
    )
    assert list(frame["image_id"]) == list(wide["image_id"])
    assert list(frame["source"]) == ["vindr_wide_train"] * 4 + ["vindr_wide_test"] * 2
    assert list(frame["split"]) == ["train"] * 4 + ["test"] * 2
    assert list(frame["label"]) == [1, 0, 1, 0, 1, 0]
    assert list(frame["label_any"]) == [1, 1, 1, 0, 1, 0]

    # Both metric rows are recomputations of the SAME saved ensemble probability.
    probs = frame["prob"].to_numpy()
    main_labels = np.asarray(frame["label"], dtype=int)
    any_labels = np.asarray(frame["label_any"], dtype=int)
    assert metrics["auc"] == pytest.approx(round(roc_auc_score(main_labels, probs), 4))
    block = metrics["sensitivity_label_any"]
    assert block["auc"] == pytest.approx(round(roc_auc_score(any_labels, probs), 4))
    assert block["n_pos"] == 4
    assert block["n_neg"] == 2
    assert len(block["auroc_ci"]) == 2
    assert block["auroc_ci"][0] <= block["auroc_ci"][1]


# -------------------------------------------------------------------------------- CLI


def test_cli_help_lists_e2_e4_modes():
    result = CliRunner().invoke(evaluate_kfold.main, ["--help"])

    assert result.exit_code == 0
    assert "vindr_test" in result.output
    assert "vindr_wide" in result.output
    assert "chexlocalize" in result.output


def test_cli_no_longer_accepts_e5_evaluation():
    # B does not evaluate E5 (25 Kasım 2026 decision); run A owns that row.
    help_result = CliRunner().invoke(evaluate_kfold.main, ["--help"])
    assert help_result.exit_code == 0
    assert "--e5_manifest" not in help_result.output
    assert "ood_reference_like" not in help_result.output
    assert "ood_clean" not in help_result.output

    rejected = CliRunner().invoke(
        evaluate_kfold.main, ["--validation_mode", "ood_reference_like"]
    )
    assert rejected.exit_code != 0

"""Stage-2 (B) classification-evaluation tests: cv-record checkpoints, tau, ID keys.

Synthetic only: a tiny dataset, a tiny LightningModule and a hand-written
`cv_record.json` with hand-written checkpoints under a temporary checkpoint dir.
Check targets are named in the module docstring of
`sharpness_matters/ptx/utils/classification_eval.py`.
"""

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pytest
import pytorch_lightning as pl
import torch
from pytorch_lightning import LightningModule
from torch import nn
from torch.utils.data import Dataset

from sharpness_matters.config.load_config import cfg
from sharpness_matters.ptx.utils import classification_eval

LOGGER = logging.getLogger("test_classif_eval")
IMG_SIZE = 512
MODEL_NAME = "resnet"
FOLD_BIASES = [0.0, 0.0, 0.0, 0.0, 0.0]


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

    def __init__(self, logits, labels, label_dict=None, ids=None):
        self.logits = [float(value) for value in logits]
        self.item_labels = [int(value) for value in labels]
        self.image_ids = (
            list(ids)
            if ids is not None
            else [f"img{i}" for i in range(len(self.logits))]
        )
        self.labels = {
            image_id: int(value)
            for image_id, value in zip(
                self.image_ids, label_dict if label_dict is not None else labels
            )
        }

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


def _write_record(root: Path, biases, tau: float, k_folds=None) -> Path:
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
        "k_folds": len(biases) if k_folds is None else k_folds,
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


# Logits whose sigmoid probabilities are 0.6, 0.8, 0.4, 0.9; labels 0,1,0,1.
# 0.5 -> preds 1,1,0,1 (acc 0.75, f1 0.8); tau=0.7 -> preds 0,1,0,1 (acc 1.0).
LOGITS = [0.4055, 1.3863, -0.4055, 2.1972]
LABELS = [0, 1, 0, 1]
IDS = [f"img{index}" for index in range(len(LOGITS))]


# --------------------------------------------------------------------------- tests


def test_missing_fold_checkpoint_raises(ckpt_dir):
    _write_record(ckpt_dir, FOLD_BIASES, tau=0.7)
    missing = ckpt_dir / MODEL_NAME / f"{MODEL_NAME}_{IMG_SIZE}_logs" / "fold_2.ckpt"
    missing.unlink()

    with pytest.raises(RuntimeError, match="missing"):
        classification_eval.eval_res(
            LogitDataset(LOGITS, LABELS),
            IMG_SIZE,
            MODEL_NAME,
            TinyClassifier,
            LOGGER,
            ids=IDS,
        )


def test_sha256_mismatch_raises(ckpt_dir):
    _write_record(ckpt_dir, FOLD_BIASES, tau=0.7)
    tampered = ckpt_dir / MODEL_NAME / f"{MODEL_NAME}_{IMG_SIZE}_logs" / "fold_1.ckpt"
    _write_checkpoint(tampered, bias=5.0)  # valid checkpoint, different bytes

    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        classification_eval.eval_res(
            LogitDataset(LOGITS, LABELS),
            IMG_SIZE,
            MODEL_NAME,
            TinyClassifier,
            LOGGER,
            ids=IDS,
        )


def test_threshold_uses_tau(ckpt_dir):
    _write_record(ckpt_dir, FOLD_BIASES, tau=0.7)

    pred_df, metrics = classification_eval.eval_res(
        LogitDataset(LOGITS, LABELS),
        IMG_SIZE,
        MODEL_NAME,
        TinyClassifier,
        LOGGER,
        ids=IDS,
    )

    # tau=0.7: only 0.8 and 0.9 cross the threshold -> perfect agreement.
    assert metrics["tau"] == pytest.approx(0.7)
    assert metrics["acc"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0)
    np.testing.assert_allclose(
        pred_df["prob"].to_numpy(), [0.6, 0.8, 0.4, 0.9], atol=1e-3
    )
    # A fixed 0.5 would predict 1,1,0,1 and drop accuracy to 0.75.
    assert (pred_df["prob"].to_numpy() >= 0.5).astype(int).tolist() == [1, 1, 0, 1]


def test_ensemble_is_mean_of_fold_probabilities(ckpt_dir):
    biases = [0.0, 1.0, -1.0, 2.0, 0.5]
    _write_record(ckpt_dir, biases, tau=0.7)

    pred_df, metrics = classification_eval.eval_res(
        LogitDataset(LOGITS, LABELS),
        IMG_SIZE,
        MODEL_NAME,
        TinyClassifier,
        LOGGER,
        ids=IDS,
    )

    logits = np.asarray(LOGITS, dtype=np.float64)
    fold_probs = np.stack([1.0 / (1.0 + np.exp(-(logits + bias))) for bias in biases])
    for fold in range(len(biases)):
        np.testing.assert_allclose(
            pred_df[f"prob_fold{fold}"].to_numpy(), fold_probs[fold], atol=1e-5
        )
    # Decision A: mean of the fold sigmoid probabilities, not sigmoid of mean logits.
    np.testing.assert_allclose(
        pred_df["prob"].to_numpy(), fold_probs.mean(axis=0), atol=1e-6
    )
    logit_mean_prob = 1.0 / (1.0 + np.exp(-(logits + np.mean(biases))))
    assert np.abs(pred_df["prob"].to_numpy() - logit_mean_prob).max() > 1e-2
    assert metrics["auc"] == pytest.approx(1.0)


def test_output_id_order_matches_input(ckpt_dir):
    _write_record(ckpt_dir, FOLD_BIASES, tau=0.7)
    # Non-sorted IDs, as from a manifest or a DICOM listing.
    ids = ["c_1", "a_2", "d_0", "b_3"]
    dataset = LogitDataset(LOGITS, LABELS, ids=ids)

    pred_df, _ = classification_eval.eval_res(
        dataset, IMG_SIZE, MODEL_NAME, TinyClassifier, LOGGER, ids=ids
    )

    # The frame is in dataset order and every row carries that ID's label.
    assert list(pred_df.columns[:3]) == ["image_id", "label", "prob"]
    assert list(pred_df["image_id"]) == ids
    assert list(pred_df["label"]) == [dataset.labels[image_id] for image_id in ids]
    assert [f"prob_fold{fold}" for fold in range(5)] == list(pred_df.columns[3:])


def test_label_misalignment_raises(ckpt_dir):
    _write_record(ckpt_dir, FOLD_BIASES, tau=0.7)
    # The dataset's label dict disagrees with the labels its items return.
    dataset = LogitDataset(LOGITS, LABELS, label_dict=[1, 0, 1, 0])

    with pytest.raises(RuntimeError, match="label misalignment"):
        classification_eval.eval_res(
            dataset, IMG_SIZE, MODEL_NAME, TinyClassifier, LOGGER, ids=IDS
        )


def test_caller_labels_mismatch_raises(ckpt_dir):
    _write_record(ckpt_dir, FOLD_BIASES, tau=0.7)

    with pytest.raises(RuntimeError, match="label misalignment"):
        classification_eval.eval_res(
            LogitDataset(LOGITS, LABELS),
            IMG_SIZE,
            MODEL_NAME,
            TinyClassifier,
            LOGGER,
            ids=IDS,
            labels=np.asarray([1, 0, 1, 0]),
        )

"""Stage-2 (B) training tests: fold manifest, pos_weight, checkpoint restore, OOF.

Synthetic only: a tiny dataset, a tiny LightningModule and a monkeypatched fold manifest.
Check targets are named in the docstrings of the modules under test.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from pytorch_lightning import LightningModule
from torch import nn
from torch.utils.data import Dataset

from sharpness_matters.config.load_config import cfg
from sharpness_matters.ptx.corrected.cv_record import (
    fold_checkpoints,
    load_cv_record,
    run_dir,
    write_cv_record,
)
from sharpness_matters.ptx.corrected.threshold import oof_threshold
from sharpness_matters.ptx.train import train_kfold

TRAINER_KWARGS = dict(
    accelerator="cpu",
    devices=1,
    logger=False,
    enable_progress_bar=False,
    enable_model_summary=False,
    # keep the recorded validation epochs equal to the real epochs (no sanity pass)
    num_sanity_val_steps=0,
)


# --------------------------------------------------------------------------- fixtures


@pytest.fixture
def ckpt_dir(tmp_path, monkeypatch):
    root = tmp_path / "checkpoints"
    monkeypatch.setattr(cfg.ptx.model, "checkpoint_dir", str(root))
    return root


class TinyDataset(Dataset):
    """Minimal stand-in for ChestXrayDataset: `image_ids`, `labels`, `pid` in the batch."""

    def __init__(self, labels, img_size=8, augment=False):
        self.image_ids = list(labels)
        self.labels = {image_id: int(label) for image_id, label in labels.items()}
        self.img_size = img_size
        self.augment = augment
        self._generator = torch.Generator().manual_seed(0)
        pos = sum(1 for label in self.labels.values() if label == 1)
        self.class_count = {1: pos, 0: len(self.labels) - pos}

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        label = self.labels[image_id]
        data = torch.full((1, self.img_size, self.img_size), float(label))
        if self.augment:  # stands in for the random flip/rotation of the train path
            data = data + torch.rand(1, generator=self._generator)
        return {
            "data": data,
            "label": torch.tensor(label, dtype=torch.long),
            "pid": image_id,
        }


class TinyModel(LightningModule):
    """Tiny module with a real parameter; optionally scripted `val/loss` per epoch."""

    def __init__(
        self,
        lr=1e-3,
        pos_weight=1.0,
        img_size=8,
        model_name="dummy",
        script=None,
    ):
        super().__init__()
        self.lr = lr
        self.pos_weight = pos_weight
        self.script = None if script is None else list(script)
        self.linear = nn.Linear(1, 1)
        self.val_inputs_per_epoch = []
        self.train_inputs_per_epoch = []
        self._val_inputs = []
        self._train_inputs = []

    def forward(self, x):
        pooled = x.flatten(1).mean(dim=1, keepdim=True)
        return self.linear(pooled).squeeze(1)

    def on_train_epoch_start(self):
        self._train_inputs = []

    def on_train_epoch_end(self):
        self.train_inputs_per_epoch.append(self._train_inputs)

    def training_step(self, batch, batch_idx):
        self._train_inputs.append(batch["data"].detach().clone())
        loss = (self.linear(torch.zeros(1, 1, device=self.device)) ** 2).sum()
        self.log("train_loss", loss)
        return loss

    def on_validation_epoch_start(self):
        self._val_inputs = []

    def on_validation_epoch_end(self):
        self.val_inputs_per_epoch.append(self._val_inputs)

    def validation_step(self, batch, batch_idx):
        self._val_inputs.append(batch["data"].detach().clone())
        if self.script is None:
            logits = self(batch["data"])
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits, batch["label"].float()
            )
        else:
            loss = torch.tensor(
                self.script[min(self.current_epoch, len(self.script) - 1)],
                dtype=torch.float32,
                device=self.device,
            )
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/acc", torch.tensor(1.0, device=self.device))
        return {"val/loss": loss}

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


def _factory(script=None, captured=None):
    def build(**kwargs):
        if captured is not None:
            captured.append(dict(kwargs))
        return TinyModel(script=script, **kwargs)

    return build


def _manifest(path, fold_of_id, labels):
    rows = [
        {"image_id": image_id, "label": labels[image_id], "fold": fold}
        for image_id, fold in fold_of_id.items()
    ]
    frame = pd.DataFrame(rows, columns=["image_id", "label", "fold"])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


def _development_labels():
    # 8 images, both classes in every fold; fold 0 = 2 ids, fold 1 = 6 ids.
    return {
        "img0": 1,
        "img1": 0,
        "img2": 1,
        "img3": 0,
        "img4": 1,
        "img5": 0,
        "img6": 1,
        "img7": 0,
    }


# ------------------------------------------------------------------ oof_threshold


@pytest.mark.parametrize(
    "labels, probs, target, expected",
    [
        # all three positives must survive -> highest tau is the smallest positive prob
        ([0, 1, 0, 1, 1, 0], [0.1, 0.7, 0.2, 0.9, 0.6, 0.3], 0.95, 0.6),
        # 2/3 = 0.667 >= 0.66, so tau may rise to the second-highest positive prob
        ([0, 1, 0, 1, 1, 0], [0.1, 0.7, 0.2, 0.9, 0.6, 0.3], 0.66, 0.7),
        # ties at tau count positive: both positives sit exactly on 0.4
        ([0, 1, 0, 1], [0.1, 0.4, 0.2, 0.4], 0.95, 0.4),
        # target sensitivity 1.0 is attainable at the minimum positive probability
        ([0, 1, 0, 1], [0.1, 0.4, 0.2, 0.8], 1.0, 0.4),
        # a lower target lets tau move above every negative score
        ([1, 1, 0, 0], [0.9, 0.2, 0.5, 0.3], 0.5, 0.9),
    ],
)
def test_oof_threshold_hand_computed(labels, probs, target, expected):
    tau = oof_threshold(labels, probs, target)
    assert tau == pytest.approx(expected)
    # tau is always one of the observed OOF probabilities
    assert tau in set(np.asarray(probs, dtype=float).tolist())


def test_oof_threshold_returns_highest_qualifying_tau_not_first():
    # Rather than the first tau in ascending order, the highest qualifying one is used.
    labels = [1, 1, 0, 0]
    probs = [0.05, 0.55, 0.60, 0.99]
    assert oof_threshold(labels, probs, 1.0) == pytest.approx(0.05)


@pytest.mark.parametrize(
    "labels, probs, target",
    [
        ([0, 0, 0, 0], [0.1, 0.4, 0.2, 0.8], 0.95),  # no positive: undefined
        ([0, 1, 0, 1], [0.1, 0.4, 0.2, 0.8], 1.01),  # unattainable target
        ([0, 1, 0, 1], [0.1, 0.4, 0.2, 0.8], 0.0),  # target outside (0, 1]
        ([0, 1], [0.1, float("nan")], 0.9),  # non-finite probability
        ([0, 2], [0.1, 0.8], 0.9),  # non-binary label
        ([0, 1], [0.1], 0.9),  # length mismatch
    ],
)
def test_oof_threshold_rejects_undefined_inputs(labels, probs, target):
    with pytest.raises(ValueError):
        oof_threshold(labels, probs, target)


def test_oof_threshold_accepts_pandas_series():
    labels = pd.Series([0, 1, 0, 1])
    probs = pd.Series([0.1, 0.4, 0.2, 0.4])
    assert oof_threshold(labels, probs, 0.95) == pytest.approx(0.4)


# -------------------------------------------------------------------- cv_record


def _fake_record(tmp_path, k_folds=2, model_name="dummy", img_size=8):
    folds = []
    for fold in range(k_folds):
        checkpoint = tmp_path / f"fold_{fold}.ckpt"
        checkpoint.write_bytes(b"fold-%d" % fold)
        folds.append(
            {
                "fold": fold,
                "best_epoch": fold + 1,
                "stopped_epoch": fold + 3,
                "checkpoint_path": str(checkpoint),
                "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "val_auroc": 0.5 + 0.1 * fold,
                "n_train": 6,
                "n_val": 2,
                "pos_weight": 3.0,
            }
        )
    record = {
        "model_name": model_name,
        "img_size": img_size,
        "k_folds": k_folds,
        "fold_manifest_sha256": "0" * 64,
        "fold_manifest_provenance_version": 1,
        "folds": folds,
        "S_h": float(np.mean([entry["val_auroc"] for entry in folds])),
        "pooled_oof_auroc": 0.75,
        "tau": 0.4,
        "target_sensitivity": 0.95,
    }
    oof = pd.DataFrame(
        {
            "image_id": [f"img{i}" for i in range(8)],
            "label": [0, 1, 0, 1, 0, 1, 0, 1],
            "fold": [0, 0, 0, 0, 1, 1, 1, 1],
            "prob": [0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6],
        }
    )
    return record, oof


def test_cv_record_roundtrip_and_fold_checkpoints(tmp_path, ckpt_dir):
    record, oof = _fake_record(tmp_path)
    json_path = write_cv_record(record, oof)
    assert json_path == run_dir("dummy", 8) / "cv_record.json"
    assert (json_path.parent / "oof_predictions.csv").is_file()

    loaded = load_cv_record("dummy", 8)
    assert loaded["k_folds"] == 2
    assert loaded["oof_csv"] == "oof_predictions.csv"
    assert len(fold_checkpoints(loaded)) == 2
    assert [Path(path).name for path in fold_checkpoints(loaded)] == [
        "fold_0.ckpt",
        "fold_1.ckpt",
    ]


def test_cv_record_load_raises_when_record_missing(tmp_path, ckpt_dir):
    with pytest.raises(RuntimeError, match="cv record not found"):
        load_cv_record("dummy", 8)


def test_cv_record_load_raises_on_missing_fold(tmp_path, ckpt_dir):
    record, oof = _fake_record(tmp_path)
    json_path = write_cv_record(record, oof)
    broken = json.loads(json_path.read_text())
    broken["folds"] = broken["folds"][:1]
    json_path.write_text(json.dumps(broken))

    with pytest.raises(RuntimeError, match="fold indices"):
        load_cv_record("dummy", 8)


def test_cv_record_write_rejects_wrong_provenance_version(tmp_path, ckpt_dir):
    record, oof = _fake_record(tmp_path)
    record["fold_manifest_provenance_version"] = 0
    with pytest.raises(RuntimeError, match="provenance_version"):
        write_cv_record(record, oof)


def test_cv_record_write_rejects_missing_checkpoint_entry(tmp_path, ckpt_dir):
    with pytest.raises(ValueError, match="missing keys"):
        write_cv_record({"model_name": "dummy", "img_size": 8, "folds": []}, pd.DataFrame())


def test_cv_record_load_raises_when_checkpoint_missing(tmp_path, ckpt_dir):
    record, oof = _fake_record(tmp_path)
    write_cv_record(record, oof)
    Path(record["folds"][0]["checkpoint_path"]).unlink()

    with pytest.raises(RuntimeError, match="missing"):
        load_cv_record("dummy", 8)


def test_cv_record_load_raises_on_hash_mismatch(tmp_path, ckpt_dir):
    record, oof = _fake_record(tmp_path)
    write_cv_record(record, oof)
    Path(record["folds"][1]["checkpoint_path"]).write_bytes(b"tampered")

    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        load_cv_record("dummy", 8)


def test_cv_record_write_rejects_duplicate_oof_ids(tmp_path, ckpt_dir):
    record, oof = _fake_record(tmp_path)
    oof.loc[1, "image_id"] = "img0"
    with pytest.raises(ValueError, match="duplicated image_id"):
        write_cv_record(record, oof)


# ------------------------------------------------------------------ train_with_kfold


def test_pos_weight_hand_count(tmp_path, ckpt_dir):
    # Four positives (a-d) and six negatives (e-j), so the whole split gives 6/4 = 1.5 —
    # the author's per-dataset ratio. Fold 0 validates d, e, f: training is 3 pos / 4 neg
    # -> 4/3. Fold 1 validates a, b, c, g, h, i, j: training is 1 pos / 2 neg -> 2.0.
    labels = {
        "a": 1,
        "b": 1,
        "c": 1,
        "d": 1,
        "e": 0,
        "f": 0,
        "g": 0,
        "h": 0,
        "i": 0,
        "j": 0,
    }
    fold_of_id = {image_id: 1 for image_id in labels}
    fold_of_id.update({"d": 0, "e": 0, "f": 0})
    manifest_path = tmp_path / "siim_folds.csv"
    manifest = _manifest(manifest_path, fold_of_id, labels)

    captured = []
    record = train_kfold.train_with_kfold(
        TinyDataset(labels),
        TinyDataset(labels),
        k_folds=2,
        num_epochs=1,
        batch_size=4,
        num_workers=0,
        img_size=8,
        fold_manifest=manifest,
        fold_manifest_path=manifest_path,
        model_name="dummy",
        model_factory=_factory(captured=captured),
        trainer_kwargs=TRAINER_KWARGS,
    )
    assert [entry["pos_weight"] for entry in captured] == pytest.approx([4 / 3, 2.0])
    assert [entry["pos_weight"] for entry in record["folds"]] == pytest.approx(
        [4 / 3, 2.0]
    )
    assert [entry["n_train"] for entry in record["folds"]] == [7, 3]
    assert [entry["n_val"] for entry in record["folds"]] == [3, 7]


def test_manifest_ids_must_match_dataset(tmp_path, ckpt_dir):
    labels = _development_labels()
    partial = {image_id: 0 for image_id in list(labels)[:6]}
    manifest_path = tmp_path / "siim_folds.csv"
    manifest = _manifest(manifest_path, partial, labels)

    with pytest.raises(RuntimeError, match="differ from the dataset"):
        train_kfold.train_with_kfold(
            TinyDataset(labels),
            TinyDataset(labels),
            k_folds=1,
            num_epochs=1,
            num_workers=0,
            img_size=8,
            fold_manifest=manifest,
            fold_manifest_path=manifest_path,
            model_name="dummy",
            model_factory=_factory(),
            trainer_kwargs=TRAINER_KWARGS,
        )


def test_manifest_label_mismatch_is_rejected(tmp_path, ckpt_dir):
    labels = _development_labels()
    wrong = dict(labels)
    wrong["img0"] = 0
    fold_of_id = {image_id: 0 for image_id in labels}
    manifest_path = tmp_path / "siim_folds.csv"
    manifest = _manifest(manifest_path, fold_of_id, wrong)

    with pytest.raises(RuntimeError, match="labels disagree"):
        train_kfold.train_with_kfold(
            TinyDataset(labels),
            TinyDataset(labels),
            k_folds=1,
            num_epochs=1,
            num_workers=0,
            img_size=8,
            fold_manifest=manifest,
            fold_manifest_path=manifest_path,
            model_name="dummy",
            model_factory=_factory(),
            trainer_kwargs=TRAINER_KWARGS,
        )


def test_eval_dataset_is_required(tmp_path, ckpt_dir):
    labels = _development_labels()
    with pytest.raises(ValueError, match="eval_dataset is required"):
        train_kfold.train_with_kfold(TinyDataset(labels), None, num_workers=0)


def _run_scripted(tmp_path, ckpt_dir, script, patience=2, max_epochs=5):
    labels = _development_labels()
    # Two folds, so every fold has a training split; fold 0 is the one inspected.
    fold_of_id = {
        image_id: int(index >= len(labels) // 2)
        for index, image_id in enumerate(labels)
    }
    manifest_path = tmp_path / "siim_folds.csv"
    manifest = _manifest(manifest_path, fold_of_id, labels)
    return train_kfold.train_with_kfold(
        TinyDataset(labels),
        TinyDataset(labels),
        k_folds=2,
        num_epochs=max_epochs,
        patience=patience,
        batch_size=8,
        num_workers=0,
        img_size=8,
        fold_manifest=manifest,
        fold_manifest_path=manifest_path,
        model_name="dummy",
        model_factory=_factory(script=script),
        trainer_kwargs=dict(TRAINER_KWARGS, max_epochs=max_epochs),
    )


def test_checkpoint_monitor_restores_best_epoch_and_ties_keep_first(
    tmp_path, ckpt_dir
):
    # val/loss per epoch: 0.9, 0.5, 0.7, 0.5, 0.8 -> best is epoch 2; epoch 4 ties 0.5 and
    # must NOT replace it (Lightning compares strictly). patience=2 stops after epoch 4.
    record = _run_scripted(tmp_path, ckpt_dir, [0.9, 0.5, 0.7, 0.5, 0.8])

    fold = record["folds"][0]
    assert fold["best_epoch"] == 2
    assert fold["stopped_epoch"] == 4
    checkpoint = Path(fold["checkpoint_path"])
    assert checkpoint.is_file()
    assert checkpoint.name == "best-epoch=1.ckpt"  # filename="best-{epoch}", 0-based epoch
    assert fold["sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    loaded = load_cv_record("dummy", 8)
    assert loaded["folds"][0]["best_epoch"] == 2


def test_patience_governs_early_stopping_length(tmp_path, ckpt_dir):
    # No improvement after epoch 2 (best 0.4); patience=3 in epochs stops after epoch 5,
    # one epoch before max_epochs=6 would have ended the run.
    record = _run_scripted(
        tmp_path, ckpt_dir, [0.9, 0.4, 0.8, 0.85, 0.95, 0.99], patience=3, max_epochs=6
    )
    fold = record["folds"][0]
    assert fold["best_epoch"] == 2
    assert fold["stopped_epoch"] == 5


def test_train_with_kfold_end_to_end_writes_valid_record_and_oof(tmp_path, ckpt_dir):
    labels = _development_labels()
    fold_of_id = {
        "img0": 0,
        "img1": 0,
        "img2": 0,
        "img3": 0,
        "img4": 1,
        "img5": 1,
        "img6": 1,
        "img7": 1,
    }
    manifest_path = tmp_path / "siim_folds.csv"
    manifest = _manifest(manifest_path, fold_of_id, labels)

    record = train_kfold.train_with_kfold(
        TinyDataset(labels, augment=True),
        TinyDataset(labels, augment=False),
        k_folds=2,
        num_epochs=2,
        batch_size=4,
        num_workers=0,
        img_size=8,
        fold_manifest=manifest,
        fold_manifest_path=manifest_path,
        model_name="dummy",
        model_factory=_factory(),
        trainer_kwargs=TRAINER_KWARGS,
    )

    directory = run_dir("dummy", 8)
    on_disk = json.loads((directory / "cv_record.json").read_text())
    assert on_disk["k_folds"] == 2
    assert on_disk["fold_manifest_sha256"] == hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    assert on_disk["img_size"] == 8
    assert on_disk["target_sensitivity"] == pytest.approx(
        float(cfg.ptx.corrected.target_sensitivity)
    )
    assert [entry["fold"] for entry in on_disk["folds"]] == [0, 1]
    for entry in on_disk["folds"]:
        assert entry["n_train"] + entry["n_val"] == len(labels)
        assert 1 <= entry["best_epoch"] <= entry["stopped_epoch"] <= 2
        assert 0.0 <= entry["val_auroc"] <= 1.0
        assert Path(entry["checkpoint_path"]).is_file()

    oof = pd.read_csv(directory / "oof_predictions.csv")
    assert list(oof.columns) == ["image_id", "label", "fold", "prob"]
    assert sorted(oof["image_id"]) == sorted(labels)
    assert not oof["image_id"].duplicated().any()
    assert dict(zip(oof["image_id"], oof["fold"])) == fold_of_id
    assert dict(zip(oof["image_id"], oof["label"].astype(int))) == labels
    assert oof["prob"].between(0.0, 1.0).all()

    # S_h is the mean fold AUROC; pooled OOF AUROC and tau are recomputed independently.
    assert record["S_h"] == pytest.approx(
        float(np.mean([entry["val_auroc"] for entry in record["folds"]]))
    )
    pooled = float(
        np.mean(
            [
                (p > q) + 0.5 * (p == q)
                for p in oof.loc[oof.label == 1, "prob"]
                for q in oof.loc[oof.label == 0, "prob"]
            ]
        )
    )
    assert record["pooled_oof_auroc"] == pytest.approx(pooled)
    assert record["tau"] == pytest.approx(
        oof_threshold(
            oof["label"].to_numpy(),
            oof["prob"].to_numpy(),
            float(cfg.ptx.corrected.target_sensitivity),
        )
    )

    loaded = load_cv_record("dummy", 8)
    assert fold_checkpoints(loaded) == [
        entry["checkpoint_path"] for entry in on_disk["folds"]
    ]


def test_training_rejects_frame_that_does_not_match_hashed_manifest(
    tmp_path, ckpt_dir
):
    """The cv record's fold_manifest_sha256 must describe the frame actually trained."""
    labels = _development_labels()
    fold_of_id = {
        image_id: int(index >= len(labels) // 2)
        for index, image_id in enumerate(labels)
    }
    manifest_path = tmp_path / "siim_folds.csv"
    manifest = _manifest(manifest_path, fold_of_id, labels)

    leaked = manifest.copy()
    leaked.loc[0, "fold"] = (int(leaked.loc[0, "fold"]) + 1) % 2
    with pytest.raises(RuntimeError, match="provenance mismatch"):
        train_kfold.train_with_kfold(
            TinyDataset(labels),
            TinyDataset(labels),
            k_folds=2,
            num_epochs=1,
            batch_size=4,
            num_workers=0,
            img_size=8,
            model_name="dummy",
            fold_manifest=leaked,
            fold_manifest_path=manifest_path,
            model_factory=_factory(),
            trainer_kwargs=TRAINER_KWARGS,
        )

    extra = pd.concat(
        [
            manifest,
            pd.DataFrame({"image_id": ["held-out"], "label": [0], "fold": [0]}),
        ],
        ignore_index=True,
    )
    with pytest.raises(RuntimeError, match="provenance mismatch"):
        train_kfold.train_with_kfold(
            TinyDataset(labels),
            TinyDataset(labels),
            k_folds=2,
            num_epochs=1,
            batch_size=4,
            num_workers=0,
            img_size=8,
            model_name="dummy",
            fold_manifest=extra,
            fold_manifest_path=manifest_path,
            model_factory=_factory(),
            trainer_kwargs=TRAINER_KWARGS,
        )


def test_reordered_manifest_frame_matches_the_hashed_file(tmp_path, ckpt_dir):
    """Row order is not provenance: the same ID/label/fold map trains and records."""
    labels = _development_labels()
    fold_of_id = {
        image_id: int(index >= len(labels) // 2)
        for index, image_id in enumerate(labels)
    }
    manifest_path = tmp_path / "siim_folds.csv"
    manifest = _manifest(manifest_path, fold_of_id, labels)

    record = train_kfold.train_with_kfold(
        TinyDataset(labels),
        TinyDataset(labels),
        k_folds=2,
        num_epochs=1,
        batch_size=4,
        num_workers=0,
        img_size=8,
        model_name="dummy",
        fold_manifest=manifest.iloc[::-1].reset_index(drop=True),
        fold_manifest_path=manifest_path,
        model_factory=_factory(),
        trainer_kwargs=TRAINER_KWARGS,
    )
    assert record["fold_manifest_sha256"] == hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    assert record["fold_manifest_provenance_version"] == 1


def test_default_manifest_loader_and_path_are_used(tmp_path, ckpt_dir, monkeypatch):
    labels = _development_labels()
    fold_of_id = {
        image_id: int(index >= len(labels) // 2)
        for index, image_id in enumerate(labels)
    }
    manifest_path = tmp_path / "siim_folds.csv"
    manifest = _manifest(manifest_path, fold_of_id, labels)
    monkeypatch.setattr(train_kfold, "_load_fold_manifest", lambda: manifest)
    monkeypatch.setattr(train_kfold, "_fold_manifest_path", lambda: manifest_path)

    record = train_kfold.train_with_kfold(
        TinyDataset(labels),
        TinyDataset(labels),
        k_folds=2,
        num_epochs=1,
        batch_size=4,
        num_workers=0,
        img_size=8,
        model_name="dummy",
        model_factory=_factory(),
        trainer_kwargs=TRAINER_KWARGS,
    )
    assert record["fold_manifest_sha256"] == hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    assert [entry["fold"] for entry in record["folds"]] == [0, 1]


def test_validation_path_is_deterministic_and_train_path_is_not(tmp_path, ckpt_dir):
    labels = _development_labels()
    fold_of_id = {
        image_id: int(index >= len(labels) // 2)
        for index, image_id in enumerate(labels)
    }
    manifest_path = tmp_path / "siim_folds.csv"
    manifest = _manifest(manifest_path, fold_of_id, labels)

    model_holder = {}

    def factory(**kwargs):
        model = TinyModel(**kwargs)
        model_holder.setdefault("models", []).append(model)
        return model

    train_kfold.train_with_kfold(
        TinyDataset(labels, augment=True),
        TinyDataset(labels, augment=False),
        k_folds=2,
        num_epochs=2,
        batch_size=8,
        num_workers=0,
        img_size=8,
        fold_manifest=manifest,
        fold_manifest_path=manifest_path,
        model_name="dummy",
        model_factory=factory,
        trainer_kwargs=TRAINER_KWARGS,
    )

    model = model_holder["models"][0]
    assert len(model.val_inputs_per_epoch) == 2
    # 5.1: the same validation image gives the same tensor on the next epoch...
    assert torch.equal(
        model.val_inputs_per_epoch[0][0], model.val_inputs_per_epoch[1][0]
    )
    # ...while the training path is still augmented (different tensor per access).
    assert not torch.equal(
        model.train_inputs_per_epoch[0][0], model.train_inputs_per_epoch[1][0]
    )

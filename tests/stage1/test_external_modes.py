"""Stage-1 (A) checks for the added external rows (E2, E2-wide, E4). Synthetic only.

Pins: manifest hash verification, ID order, the author's 0.5 threshold and
probability averaging, the E2-wide sensitivity block, and the CheXlocalize loader.
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import pytest
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from sklearn.metrics import roc_auc_score

from sharpness_matters.config.load_config import cfg
from sharpness_matters.ptx.datasets import external_eval_datasets as ext
from sharpness_matters.ptx.evaluation import evaluate_kfold as ek
from sharpness_matters.ptx.utils import classification_eval as ce


class DummyClassifier(pl.LightningModule):
    def __init__(self, bias: float = 0.0):
        super().__init__()
        self.save_hyperparameters()
        self.lin = nn.Linear(1, 1)
        with torch.no_grad():
            self.lin.weight.fill_(1.0)
            self.lin.bias.fill_(bias)

    def forward(self, x):
        return self.lin(x.flatten(1).mean(1, keepdim=True))


class FakeDS(Dataset):
    def __init__(self, ids, labels):
        self.image_ids = list(ids)
        self.labels = dict(zip(ids, labels))

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        i = self.image_ids[idx]
        return {"data": torch.zeros(1, 4, 4), "label": self.labels[i], "pid": i}


class PixelDS(FakeDS):
    """Data value +1 for positives, -1 for negatives (module level: DataLoader workers)."""

    def __getitem__(self, idx):
        i = self.image_ids[idx]
        value = 1.0 if self.labels[i] == 1 else -1.0
        return {"data": torch.full((1, 4, 4), value), "label": self.labels[i], "pid": i}


def _freeze(directory: Path, name: str, frame: pd.DataFrame) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    frame.to_csv(path, index=False)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    Path(f"{path}.sha256").write_text(f"{digest}  {name}\n")
    return path


def test_manifest_hash_verified(tmp_path, monkeypatch):
    frame = pd.DataFrame({"image_id": ["a", "b"], "label": [0, 1]})
    _freeze(tmp_path, "m.csv", frame)
    monkeypatch.setitem(cfg.ptx, "manifests_dir", str(tmp_path))
    assert list(ext.load_manifest("m.csv")["image_id"]) == ["a", "b"]
    # An edited manifest must be refused (run A never regenerates/edits manifests).
    frame.assign(label=[1, 1]).to_csv(tmp_path / "m.csv", index=False)
    with pytest.raises(RuntimeError, match="hash mismatch"):
        ext.load_manifest("m.csv")
    with pytest.raises(FileNotFoundError):
        ext.load_manifest("absent.csv")


def test_author_threshold_and_probability_average_unchanged(tmp_path, monkeypatch):
    """0.5 (strict >) and fold-probability averaging are the author's semantics."""
    for fold in range(5):
        directory = f"{tmp_path}/resnet/resnet_64_logs/lightning_logs/version_{fold}/checkpoints"
        os.makedirs(directory)
        model = DummyClassifier(bias=0.0)
        trainer = pl.Trainer(logger=False, enable_checkpointing=False, enable_progress_bar=False)
        trainer.strategy.connect(model)
        trainer.save_checkpoint(f"{directory}/epoch=0.ckpt")
    monkeypatch.setattr(ce, "CHECKPOINT_DIR", str(tmp_path))

    # data +1 -> logit +1 -> prob 0.731; data -1 -> prob 0.269
    ds = PixelDS(["p", "n"], [1, 0])
    labels, probs, metrics = ce.eval_res(
        ds, 64, "resnet", DummyClassifier, __import__("logging").getLogger("t")
    )
    assert np.allclose(probs, [0.7310586, 0.26894143], atol=1e-5)  # 5 identical folds
    assert metrics["acc"] == 1.0  # 0.731 > 0.5, 0.269 not > 0.5


@pytest.mark.parametrize(
    "mode,expected_columns",
    [
        ("vindr_test", ["image_id", "source", "label", "score"]),
        ("vindr_wide", ["image_id", "source", "label", "score", "split", "label_any"]),
        ("chexlocalize", ["image_id", "source", "label", "score", "split"]),
    ],
)
def test_added_modes_write_id_csv(tmp_path, monkeypatch, mode, expected_columns):
    monkeypatch.setitem(cfg.ptx, "manifests_dir", str(tmp_path))
    monkeypatch.setattr(ek, "HOME", str(tmp_path))
    scores = np.array([0.2, 0.9, 0.4, 0.8])

    if mode == "vindr_test":
        frame = pd.DataFrame({"image_id": ["b", "a", "d", "c"], "label": [0, 1, 0, 1]})
        _freeze(tmp_path, "vindr_test.csv", frame)
        monkeypatch.setattr(
            ek, "ManifestVinDataset",
            lambda **kw: FakeDS(list(frame.image_id), list(frame.label)),
        )
    elif mode == "vindr_wide":
        frame = pd.DataFrame(
            {
                "image_id": ["t0", "t1", "s0", "s1"],
                "split": ["train", "train", "test", "test"],
                "label": [0, 1, 0, 1],
                "label_any": [1, 1, 0, 1],
            }
        )
        _freeze(tmp_path, "vindr_wide.csv", frame)

        def fake_vin(**kw):
            stats_key = kw["stats_key"]
            rows = frame[frame.split == ("train" if "train" in stats_key else "test")]
            return FakeDS(list(rows.image_id), list(rows.label))

        monkeypatch.setattr(ek, "ManifestVinDataset", fake_vin)
    else:
        frame = pd.DataFrame(
            {
                "image_id": ["x0", "x1", "x2", "x3"],
                "split": ["val", "test", "val", "test"],
                "label": [0, 1, 0, 1],
            }
        )
        _freeze(tmp_path, "chexlocalize_cls.csv", frame)

        class FakeCheX(FakeDS):
            def __init__(self, **kw):
                super().__init__(list(frame.image_id), list(frame.label))
                self.splits = dict(zip(frame.image_id, frame.split))

        monkeypatch.setattr(ek, "CheXlocalizeEvalDataset", lambda **kw: FakeCheX())

    monkeypatch.setattr(
        ek, "eval_res",
        lambda *a, **kw: (
            np.asarray([0, 1, 0, 1]),
            scores.astype(float),
            {"acc": 1.0, "auc": 0.9, "f1": 1.0, "auroc_ci": (0.8, 1.0)},
        ),
    )
    metrics = ek.fetch_results_per_resolution(64, "resnet", mode)
    df = pd.read_csv(
        tmp_path / "output" / "classif_preds" / f"{mode}pred_resnet_64_ids.csv",
        dtype={"image_id": str},
    )
    assert list(df.columns) == expected_columns
    if mode == "vindr_test":
        # Manifest order is preserved (not sorted) and the score column is the ensemble.
        assert list(df.image_id) == ["b", "a", "d", "c"]
        assert np.allclose(df.score, scores)
    if mode == "vindr_wide":
        from sklearn.metrics import roc_auc_score

        block = metrics["sensitivity_label_any"]
        assert block["n_pos"] == 3 and block["n_neg"] == 1
        assert abs(block["auc"] - roc_auc_score([1, 1, 0, 1], scores)) < 1e-12
        assert list(df.label_any) == [1, 1, 0, 1]
        assert list(df.split) == ["train", "train", "test", "test"]
    if mode == "chexlocalize":
        assert list(df.source) == ["chexlocalize_val", "chexlocalize_test"] * 2


def test_chexlocalize_loader_preprocessing_and_mask(tmp_path, monkeypatch):
    image_dir = tmp_path / "CheXpert" / "val" / "patient1" / "study1"
    image_dir.mkdir(parents=True)
    gradient = np.tile(np.linspace(0, 255, 8, dtype=np.uint8), (6, 1))
    Image.fromarray(gradient).save(image_dir / "view1_frontal.jpg")
    manifest = tmp_path / "xai.csv"
    pd.DataFrame(
        {
            "image_id": ["patient1_study1_view1_frontal"],
            "split": ["val"],
            "path": ["val/patient1/study1/view1_frontal.jpg"],
            "label": [1],
        }
    ).to_csv(manifest, index=False)
    annotations = tmp_path / "ann"
    annotations.mkdir()
    # size [6, 8]; runs [5, 43] -> first 5 pixels background, rest foreground (column-major)
    (annotations / "gt_segmentations_val.json").write_text(
        json.dumps(
            {
                "patient1_study1_view1_frontal": {
                    "Pneumothorax": {"size": [6, 8], "counts": _encode_counts([5, 43])}
                }
            }
        )
    )
    monkeypatch.setitem(cfg.ptx.chexlocalize, "annotations_dir", str(annotations))
    monkeypatch.setitem(cfg.ptx.chexlocalize, "chexpert_dir", str(tmp_path / "CheXpert"))
    ds = ext.CheXlocalizeEvalDataset(str(manifest), img_size=8, with_mask=True)
    item = ds[0]
    assert item["data"].shape == (1, 8, 8)
    assert 0.0 <= float(item["data"].min()) and float(item["data"].max()) <= 1.0
    assert item["mask"].shape == (8, 8)
    assert float(item["mask"].max()) > 0 and float(item["mask"].min()) >= 0


def _encode_counts(counts):
    """Minimal pycocotools-compatible encoder for <=2 runs (test fixture only)."""
    chars = []
    for value in counts:
        more = True
        while more:
            chunk = value & 0x1F
            value >>= 5
            more = value != 0
            if more:
                chunk |= 0x20
            chars.append(chr(chunk + 48))
    return "".join(chars)


def test_holdout_mode_still_works(tmp_path, monkeypatch):
    """The author holdout branch must keep working after adding the external modes."""
    monkeypatch.setattr(ek, "HOME", str(tmp_path))
    monkeypatch.setattr(ek, "ChestXrayDataset", lambda *a, **k: FakeDS(["h0", "h1"], [0, 1]))
    monkeypatch.setattr(
        ek, "eval_res",
        lambda *a, **k: (np.array([0, 1]), np.array([0.2, 0.8]), {"acc": 1.0}),
    )
    ek.fetch_results_per_resolution(64, "resnet", "holdout")
    df = pd.read_csv(
        tmp_path / "output" / "classif_preds" / "holdoutpred_resnet_64_ids.csv",
        dtype={"image_id": str},
    )
    assert list(df.columns) == ["image_id", "source", "label", "score"]
    assert list(df.image_id) == ["h0", "h1"] and list(df.source) == ["siim_acr"] * 2

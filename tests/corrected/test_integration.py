"""Cross-slice checks for the Stage-2 (B) package. Synthetic data only, CPU, no downloads.

train_kfold -> cv_record -> classification_eval, and
train_kfold (real CNNBinaryClassifier, random init) -> evaluate_saliency.run_experiment.
"""
import json
import logging

import numpy as np
import pandas as pd
import pytest
import pytorch_lightning as pl
import timm
import torch
from torch import nn
from torch.utils.data import Dataset

from sharpness_matters.config.load_config import cfg


class Dummy(pl.LightningModule):
    def __init__(self, pos_weight=1.0, lr=1e-2, img_size=8, model_name="densenet"):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(img_size * img_size, 1))
        self.crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))

    def forward(self, x):
        return self.net(x)

    def training_step(self, b, i):
        return self.crit(self(b["data"]).squeeze(1), b["label"].float())

    def validation_step(self, b, i):
        self.log("val/loss", self.crit(self(b["data"]).squeeze(1), b["label"].float()))

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)


class ClassifDS(Dataset):
    def __init__(self, n=60):
        rng = np.random.default_rng(0)
        self.image_ids = [f"img{i:03d}" for i in range(n)]
        lab = (rng.random(n) < 0.3).astype(int)
        self.labels = dict(zip(self.image_ids, lab.tolist()))
        self.x = {
            i: torch.tensor(rng.normal(l, 1.0, (1, 8, 8)), dtype=torch.float32)
            for i, l in zip(self.image_ids, lab)
        }

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, k):
        i = self.image_ids[k]
        return {"data": self.x[i], "label": torch.tensor(self.labels[i]), "pid": i, "image_id": i}


class MaskedDS(Dataset):
    """SIIM-like masked items: 64x64 input, 1024x1024 uint8 {0,1} GT (contract 6)."""

    def __init__(self, n=20, size=64):
        rng = np.random.default_rng(1)
        self.image_ids = [f"1.2.{i}" for i in range(n)]
        lab = np.array([1, 0] * (n // 2))
        self.labels = dict(zip(self.image_ids, lab.tolist()))
        self.items = {}
        for i, l in zip(self.image_ids, lab):
            m = np.zeros((1024, 1024), np.uint8)
            if l:
                m[200:600, 300:301] = 1  # 1-px thin line
                m[700:760, 700:760] = 1
            self.items[i] = (torch.tensor(rng.random((1, size, size)), dtype=torch.float32), m)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, k):
        i = self.image_ids[k]
        x, m = self.items[i]
        return {"data": x, "mask": m, "label": torch.tensor(self.labels[i]), "pid": i, "image_id": i}


def _train(ds, eval_ds, tmp_path, img_size, model_factory, **kw):
    from sharpness_matters.ptx.corrected.make_fold_manifest import build_fold_manifest
    from sharpness_matters.ptx.train.train_kfold import train_with_kfold

    fm = build_fold_manifest(ds.image_ids, [ds.labels[i] for i in ds.image_ids])
    fm_path = tmp_path / "siim_folds.csv"
    fm.to_csv(fm_path, index=False)
    return train_with_kfold(
        ds,
        eval_dataset=eval_ds,
        img_size=img_size,
        model_name="densenet",
        num_workers=0,
        fold_manifest=fm,
        fold_manifest_path=fm_path,
        model_factory=model_factory,
        trainer_kwargs=dict(accelerator="cpu", devices=1, logger=False, enable_progress_bar=False),
        **kw,
    )


def test_train_record_feeds_classification_eval(tmp_path, monkeypatch):
    from sharpness_matters.ptx.corrected.cv_record import load_cv_record, run_dir
    from sharpness_matters.ptx.utils import classification_eval as ce

    monkeypatch.setitem(cfg.ptx.model, "checkpoint_dir", str(tmp_path / "ckpt"))
    ds, ev = ClassifDS(), ClassifDS()
    rec = _train(ds, ev, tmp_path, 8, lambda **k: Dummy(**k), num_epochs=3, lr=1e-2, batch_size=8)

    oof = pd.read_csv(run_dir("densenet", 8) / "oof_predictions.csv", dtype={"image_id": str})
    assert oof.image_id.is_unique and sorted(oof.image_id) == sorted(ds.image_ids)
    assert all(1 <= f["best_epoch"] <= f["stopped_epoch"] for f in rec["folds"])
    pos = oof[oof.label == 1].prob
    assert (pos >= rec["tau"]).mean() >= cfg.ptx.corrected.target_sensitivity

    pred, metrics = ce.eval_res(ev, 8, "densenet", Dummy, logging.getLogger("integ"), ids=ev.image_ids)
    assert list(pred.image_id) == ev.image_ids
    assert abs(metrics["tau"] - rec["tau"]) < 1e-6
    fold_cols = [c for c in pred.columns if c.startswith("prob_fold")]
    assert len(fold_cols) == 5 and np.allclose(pred[fold_cols].mean(axis=1), pred.prob)

    with open(rec["folds"][2]["checkpoint_path"], "ab") as f:
        f.write(b"x")
    with pytest.raises(RuntimeError, match="sha256"):
        load_cv_record("densenet", 8)


def test_real_densenet_saliency_end_to_end(tmp_path, monkeypatch):
    """Real author model class (random init, no download) through GradCAM scoring."""
    from sharpness_matters.ptx.models.cnn import CNNBinaryClassifier
    from sharpness_matters.ptx.evaluation import evaluate_saliency as es

    orig = timm.create_model
    monkeypatch.setattr(timm, "create_model", lambda name, pretrained=False, **kw: orig(name, pretrained=False, **kw))
    monkeypatch.setitem(cfg.ptx.model, "checkpoint_dir", str(tmp_path / "ckpt"))
    monkeypatch.setattr(es, "HOME", tmp_path)
    size = 64
    ds = MaskedDS(size=size)
    _train(
        ds,
        ds,
        tmp_path,
        size,
        lambda pos_weight, lr, img_size, model_name: CNNBinaryClassifier(
            lr=lr, img_size=img_size, model_name="densenet121", pos_weight=pos_weight
        ),
        num_epochs=1,
        batch_size=4,
    )
    out = tmp_path / "sal_out"
    es.run_experiment(ds, size, "densenet", "synthetic", chunk_size=4, save_maps=False,
                      output_dir=str(out), device=torch.device("cpu"))
    files = sorted(p.name for p in out.iterdir())
    main_csv = [f for f in files if "main" in f and f.endswith(".csv") and "negatives" not in f]
    assert len(main_csv) == 1
    df = pd.read_csv(out / main_csv[0], dtype={"image_id": str})
    positives = {i for i in ds.image_ids if ds.labels[i] == 1}
    assert set(df.image_id) == positives
    assert df[["iou", "precision", "coverage"]].notna().all().all()
    summary = json.load(open(out / [f for f in files if "main" in f and f.endswith(".json")][0]))
    assert summary["sensitivity_analysis"] is False and summary["n_positive"] == len(positives)
    # unique per-run cache removed after success (PTX-18 / Y5)
    assert not list((tmp_path / "output").glob("chunk_predictions_*"))

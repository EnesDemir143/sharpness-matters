"""Stage 1 (reference-like A) checks for baseline_correction.md §3 (3.1-3.7). Synthetic only."""
import hashlib
import logging
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import Dataset

from sharpness_matters.config.load_config import cfg
import sharpness_matters.ptx.utils.classification_eval as ce
import sharpness_matters.ptx.evaluation.evaluate_kfold as ek


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
        return {"data": torch.full((1, 4, 4), float(idx) / 10 - 0.5), "label": self.labels[i]}


def _make_ckpts(root, model_name, img_size, bias):
    for fold in range(5):
        d = f"{root}/{model_name}/{model_name}_{img_size}_logs/lightning_logs/version_{fold}/checkpoints"
        os.makedirs(d, exist_ok=True)
        m = DummyClassifier(bias=bias + fold * 0.1)
        trainer = pl.Trainer(logger=False, enable_checkpointing=False, enable_progress_bar=False)
        trainer.strategy.connect(m)
        trainer.save_checkpoint(f"{d}/epoch=0.ckpt")


IDS = [f"id{k:02d}" for k in range(10)]
LABELS = [0, 1] * 5


def test_3_1_checkpoint_dir_is_ptx_key():
    assert cfg.ptx.model.checkpoint_dir != cfg.pneumonia.model.checkpoint_dir  # 3.2
    assert ce.CHECKPOINT_DIR == cfg.ptx.model.checkpoint_dir


def test_3_1_eval_res_logs_ptx_checkpoint_sha256(tmp_path, monkeypatch, caplog):
    ptx_root, pneu_root = str(tmp_path / "ptx"), str(tmp_path / "pneu")
    _make_ckpts(ptx_root, "resnet", 64, bias=0.0)
    _make_ckpts(pneu_root, "resnet", 64, bias=5.0)  # decoy
    monkeypatch.setattr(ce, "CHECKPOINT_DIR", ptx_root)
    lg = logging.getLogger("stage1_test")
    with caplog.at_level(logging.INFO, logger="stage1_test"):
        ce.eval_res(FakeDS(IDS, LABELS), 64, "resnet", DummyClassifier, lg)
    msgs = [r.getMessage() for r in caplog.records if "sha256=" in r.getMessage()]
    assert len(msgs) == 5
    for msg in msgs:
        path = msg.split("checkpoint=")[1].split(" sha256=")[0]
        assert path.startswith(ptx_root)
        assert msg.split("sha256=")[1] == hashlib.sha256(open(path, "rb").read()).hexdigest()


@pytest.mark.parametrize(
    "mode,exp_ids,exp_src",
    [
        ("holdout", IDS, ["siim_acr"] * 10),
        ("ood", ["v0", "v1", "v2", "v3", "1_10", "2_20", "3_30"], ["vindr"] * 4 + ["ptx498"] * 3),
    ],
)
def test_3_7_id_keyed_predictions(tmp_path, monkeypatch, mode, exp_ids, exp_src):
    _make_ckpts(str(tmp_path / "ptx"), "resnet", 64, bias=0.0)
    monkeypatch.setattr(ce, "CHECKPOINT_DIR", str(tmp_path / "ptx"))
    monkeypatch.setattr(ek, "HOME", str(tmp_path))
    monkeypatch.setattr(ek, "CNNBinaryClassifier", DummyClassifier)
    monkeypatch.setattr(ek, "ChestXrayDataset", lambda *a, **k: FakeDS(IDS, LABELS))
    monkeypatch.setattr(ek, "VinDataset", lambda **k: FakeDS(["v0", "v1", "v2", "v3"], [0, 0, 1, 0]))
    monkeypatch.setattr(ek, "OODTestDataset", lambda **k: FakeDS(["1_10", "2_20", "3_30"], [1, 1, 1]))
    ek.fetch_results_per_resolution(64, "resnet", mode)
    out = tmp_path / "output" / "classif_preds"
    df = pd.read_csv(out / f"{mode}pred_resnet_64_ids.csv", dtype={"image_id": str})
    pred = np.load(out / f"{mode}pred_resnet_64.npy")
    gt = np.load(out / f"{mode}gt.npy")
    assert list(df.image_id) == exp_ids and list(df.source) == exp_src
    assert np.allclose(df.score.values, pred) and (df.label.values == gt).all()


def test_3_7_label_misalignment_raises(tmp_path, monkeypatch):
    _make_ckpts(str(tmp_path / "ptx"), "resnet", 64, bias=0.0)
    monkeypatch.setattr(ce, "CHECKPOINT_DIR", str(tmp_path / "ptx"))
    monkeypatch.setattr(ek, "HOME", str(tmp_path))
    monkeypatch.setattr(ek, "CNNBinaryClassifier", DummyClassifier)
    monkeypatch.setattr(ek, "VinDataset", lambda **k: FakeDS(["v0", "v1"], [0, 1]))
    monkeypatch.setattr(ek, "OODTestDataset", lambda **k: FakeDS(["1_10"], [0]))  # expected 1
    with pytest.raises(RuntimeError):
        ek.fetch_results_per_resolution(64, "resnet", "ood")


def test_3_3_missing_vindr_key_raises_keyerror():
    code = (
        "from sharpness_matters.config.load_config import cfg\n"
        "del cfg.ptx['vindr']\n"
        "import sharpness_matters.ptx.datasets.vin_dataset\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode != 0 and "KeyError: 'vindr'" in r.stderr


@pytest.mark.parametrize(
    "module",
    [
        "sharpness_matters.ptx.datasets.vin_dataset",
        "sharpness_matters.ptx.evaluation.evaluate_kfold",
        "sharpness_matters.ptx.datasets.extract_demographics",
    ],
)
def test_3_4_modules_import(module):
    r = subprocess.run([sys.executable, "-c", f"import {module}"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-500:]


def test_3_5_delong_returns_two_values():
    from MLstatkit.stats import Delong_test

    rng = np.random.default_rng(0)
    out = Delong_test(rng.integers(0, 2, 200), rng.random(200), rng.random(200))
    assert isinstance(out, tuple) and len(out) == 2


def test_3_5_requirements_match_lock_pins():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    req = open(os.path.join(root, "requirements.txt")).read().lower()
    for pin in ["mlstatkit==0.1.7", "torch==2.1.0", "torchvision==0.16.0", "statsmodels==0.14.1", "nibabel==5.2.1"]:
        assert pin in req, pin

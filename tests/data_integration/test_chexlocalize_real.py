"""Real-data regression checks for E4 (CheXlocalize). Read-only; skipped when the data is not on disk.

Run: uv run python -m pytest -q tests/data_integration
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sharpness_matters.config.load_config import cfg
from sharpness_matters.ptx.corrected.frozen import manifest_dir, sha256_file

ANN_DIR = Path(str(cfg.ptx.chexlocalize.annotations_dir))
CHEXPERT_DIR = Path(str(cfg.ptx.chexlocalize.chexpert_dir))
CLS = manifest_dir() / "chexlocalize_cls.csv"
XAI = manifest_dir() / "chexlocalize_xai.csv"

pytestmark = pytest.mark.skipif(
    not (ANN_DIR.is_dir() and CHEXPERT_DIR.is_dir() and CLS.exists() and XAI.exists()),
    reason="CheXlocalize data or frozen E4 manifests not present",
)


@pytest.mark.parametrize("path", [CLS, XAI], ids=["cls", "xai"])
def test_frozen_manifest_hash_matches(path):
    recorded = Path(f"{path}.sha256").read_text().split()[0]
    assert sha256_file(path) == recorded


def test_cohort_counts_match_frozen_record():
    cls = pd.read_csv(CLS, dtype={"image_id": str})
    xai = pd.read_csv(XAI, dtype={"image_id": str})
    # Counts frozen on 2026-11-23 (baseline_correction.md §14): frontal only.
    assert len(cls) == 720 and int(cls["label"].sum()) == 16
    assert len(xai) == 16 and set(xai["image_id"]) <= set(cls.loc[cls["label"] == 1, "image_id"])
    assert cls["image_id"].is_unique and not cls["image_id"].str.contains("lateral").any()


def test_every_real_rle_decodes_to_its_declared_size():
    from sharpness_matters.ptx.datasets.chexlocalize_dataset import rle_decode

    nonempty = {"frontal": 0, "lateral": 0}
    n = 0
    for split in ("val", "test"):
        segs = json.loads((ANN_DIR / f"gt_segmentations_{split}.json").read_text())
        for cxr_id, tasks in segs.items():
            rle = tasks["Pneumothorax"]
            mask = rle_decode(rle)
            assert mask.shape == tuple(rle["size"]), cxr_id
            assert set(np.unique(mask)) <= {0, 1}
            n += 1
            if mask.any():
                nonempty["lateral" if cxr_id.endswith("lateral") else "frontal"] += 1
    assert n == 686
    assert nonempty == {"frontal": 16, "lateral": 2}


def test_xai_loader_masks_match_images_and_are_nonempty():
    from sharpness_matters.ptx.datasets.chexlocalize_dataset import CheXlocalizeDataset

    ds = CheXlocalizeDataset(str(XAI), 64, with_mask=True)
    for i in range(len(ds)):
        item = ds[i]
        assert item["mask"].ndim == 2 and item["mask"].dtype == np.uint8
        assert item["mask"].any() and int(item["label"]) == 1
        assert item["data"].shape == (1, 64, 64)

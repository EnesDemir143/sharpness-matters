"""Stage-1 (A) read-only smoke test on the real CheXlocalize data; skipped when absent."""
from pathlib import Path

import pytest

from sharpness_matters.config.load_config import cfg
from sharpness_matters.ptx.datasets.external_eval_datasets import (
    CheXlocalizeEvalDataset,
    load_manifest,
)

CHEXPERT = Path(str(cfg.ptx.chexlocalize.chexpert_dir))
ANNOTATIONS = Path(str(cfg.ptx.chexlocalize.annotations_dir))
MANIFESTS = Path(str(cfg.ptx.manifests_dir))

pytestmark = pytest.mark.skipif(
    not (
        CHEXPERT.is_dir()
        and ANNOTATIONS.is_dir()
        and (MANIFESTS / "chexlocalize_xai.csv").is_file()
    ),
    reason="CheXlocalize data or frozen E4 manifests not present",
)


def test_frozen_classification_cohort_counts():
    cls = load_manifest("chexlocalize_cls.csv")  # verifies the recorded SHA-256
    assert len(cls) == 720 and int(cls["label"].sum()) == 16
    assert set(cls["split"]) == {"val", "test"} and cls["image_id"].is_unique


def test_xai_masks_load_with_author_preprocessing():
    import tempfile

    frame = load_manifest("chexlocalize_xai.csv")
    assert len(frame) == 16
    directory = tempfile.mkdtemp(prefix="stage1_xai_view_")
    path = Path(directory) / "xai.csv"
    frame[["image_id", "split", "path", "label"]].to_csv(path, index=False)
    ds = CheXlocalizeEvalDataset(str(path), img_size=64, with_mask=True)
    assert len(ds) == 16
    for idx in range(len(ds)):
        item = ds[idx]
        assert item["data"].shape == (1, 64, 64)
        assert 0.0 <= float(item["data"].min()) and float(item["data"].max()) <= 1.0
        assert item["mask"].shape == (64, 64) and float(item["mask"].max()) > 0
        assert int(item["label"]) == 1


def test_saliency_script_wires_the_xai_manifest():
    """The E4 saliency entry point builds its dataset from the frozen XAI manifest."""
    from sharpness_matters.ptx.evaluation import evaluate_saliency_chexlocalize as script

    frame = load_manifest("chexlocalize_xai.csv")
    view = script._write_xai_view(frame)
    dataset = CheXlocalizeEvalDataset(view, img_size=64, with_mask=True)
    assert len(dataset) == 16 and int(frame["label"].sum()) == 16
    assert script.collate_fn([dataset[0], dataset[1]])["data"].shape == (2, 1, 64, 64)

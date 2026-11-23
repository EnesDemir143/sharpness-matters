"""Stage-2 (B) saliency tests: GT grid, fold rule, chunk identity, K1 cohort.

Synthetic only: fold saliency maps are hand-built arrays and the masked dataset is a list
wrapper, so no checkpoint, no real model and no DICOM/NIfTI data is needed. Each test
names the correction item and the module docstring entry it covers:

- PTX-02 / §7.1: the GT mask stays on its reference grid; the CAM is resized to it.
- PTX-17 / §7.2: mean -> min-max -> threshold is the main rule, mean -> threshold is a
  labelled sensitivity rule.
- PTX-18 / §7.3: a missing chunk, a mismatched ImageId set or another checkpoint's chunk
  raises; the cache directory is unique per run.
- PTX-04 / §7.4: every GT-positive image is scored from its CAM (K1), a wrong negative is
  not forced to 0 there, and the joint summary keeps its own denominator.
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from sharpness_matters.config.load_config import cfg
from sharpness_matters.ptx.utils.saliency_eval import (
    NEGATIVE_SCORE_COLUMNS,
    RULE_LEGACY,
    RULE_MAIN,
    SCORE_COLUMNS,
    cache_dir_for_run,
    compute_scores_chunked,
    create_cache_dir,
    fold_chunk_paths,
    save_fold_chunk,
    score_fold_saliency,
    write_saliency_outputs,
)

K_FOLDS = int(cfg.ptx.corrected.k_folds)


class _MaskedDataset:
    """Minimal masked loader: input-grid image, reference-grid mask, ImageId key."""

    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def _item(image_id: str, label: int, mask: np.ndarray, img_size: int = 16) -> dict:
    data = torch.zeros(1, img_size, img_size)
    data[0, img_size // 2, img_size // 2] = 1.0
    return {
        "data": data,
        "label": torch.tensor(label),
        "pid": image_id,
        "image_id": image_id,
        "mask": np.asarray(mask, dtype=np.uint8),
    }


def _hot_cam(size: int, row: int, col: int) -> np.ndarray:
    """One-hot CAM: every fold map is independently min-max normalised by GradCAM."""
    cam = np.zeros((size, size), dtype=np.float32)
    cam[row, col] = 1.0
    return cam


def _stored(cam: np.ndarray) -> np.ndarray:
    """Shape written by `predict_fold_chunked`: (s, s, 1), as the author's transpose."""
    return np.expand_dims(cam, axis=-1)


def _record(tau: float = 0.5, shas=None) -> dict:
    shas = shas if shas is not None else [f"sha{fold}" for fold in range(K_FOLDS)]
    return {
        "k_folds": K_FOLDS,
        "tau": tau,
        "folds": [{"fold": fold, "sha256": shas[fold]} for fold in range(K_FOLDS)],
    }


def _write_run(cache_dir, items, probs, cams, record, chunk_size: int = 2):
    """Write one chunk set per fold, as a completed `predict_fold_chunked` run would."""
    ids = [item["image_id"] for item in items]
    shas = [entry["sha256"] for entry in record["folds"]]
    for chunk_index, start in enumerate(range(0, len(ids), chunk_size)):
        chunk_ids = ids[start : start + chunk_size]
        for fold in range(K_FOLDS):
            save_fold_chunk(
                cache_dir,
                fold,
                chunk_index,
                {image_id: probs[image_id] for image_id in chunk_ids},
                {image_id: _stored(cams[image_id][fold]) for image_id in chunk_ids},
                shas[fold],
            )


def _scores(cache_dir, dataset, record, chunk_size: int = 2):
    return compute_scores_chunked(
        dataset,
        cache_dir,
        record,
        "resnet",
        64,
        "synthetic",
        chunk_size=chunk_size,
        save_maps=False,
    )


def _two_image_run(tmp_path):
    """Two GT-positive images on different GT grids (32^2 and 40^2), CAMs on 16^2."""
    gt_small = np.zeros((32, 32), dtype=np.uint8)
    gt_small[8:16, 8:16] = 1
    gt_large = np.zeros((40, 40), dtype=np.uint8)
    gt_large[10:20, 10:20] = 1
    items = [_item("p1", 1, gt_small), _item("p2", 1, gt_large)]
    probs = {"p1": 0.9, "p2": 0.8}
    cams = {
        image_id: [_hot_cam(16, 4 + fold, 4 + fold) for fold in range(K_FOLDS)]
        for image_id in ("p1", "p2")
    }
    record = _record()
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    _write_run(cache_dir, items, probs, cams, record)
    return cache_dir, _MaskedDataset(items), record


def test_fold_mean_renormalisation_rescues_high_threshold():
    """PTX-17 / §7.2: the synthetic counter-example of the report."""
    gt = np.zeros((1024, 1024), dtype=np.uint8)
    gt[256:768, 256:768] = 1
    # Three folds peak at A, one at B, one at C; every peak is inside the GT block.
    peaks = [(24, 24), (24, 24), (24, 24), (24, 40), (40, 24)]
    fold_maps = [_hot_cam(64, row, col) for row, col in peaks]

    result = score_fold_saliency(fold_maps, gt, [0.25, 0.5, 0.75])

    # CAM geometry: five 64^2 maps averaged first, then bilinear-resized to the 1024^2 GT.
    assert result["ensemble_saliency"].shape == gt.shape
    # Peak positions differ, so the fold mean tops out well below 1.
    assert float(result["ensemble_saliency"].max()) == pytest.approx(0.6, abs=0.05)
    main = result["scores"][RULE_MAIN][0.75]
    legacy = result["scores"][RULE_LEGACY][0.75]
    # Main rule renormalises per image, so the peak is selected and it is inside the GT.
    assert main["empty_pred_mask"] is False
    assert main["precision"] > 0.0
    assert main["coverage"] > 0.0
    # Legacy rule thresholds the un-renormalised mean: no pixel reaches 0.75.
    assert legacy["empty_pred_mask"] is True
    assert legacy["precision"] == 0.0
    assert result["scores"][RULE_LEGACY][0.5]["precision"] > 0.0


@pytest.mark.parametrize(
    "shape_name, foreground",
    [
        ("thin_line", 1024),
        ("small", 64),
        ("border", 2047),
        ("empty", 0),
        ("wide", 810000),
    ],
)
def test_gt_grid_preserved_for_thin_small_border_empty_wide_masks(
    shape_name, foreground
):
    """PTX-02 / §7.1: a 64^2 CAM against 1024^2 GT masks of every fragile shape."""
    gt = np.zeros((1024, 1024), dtype=np.uint8)
    if shape_name == "thin_line":
        gt[512, :] = 1
    elif shape_name == "small":
        gt[500:508, 500:508] = 1
    elif shape_name == "border":
        gt[0, :] = 1
        gt[:, 0] = 1
    elif shape_name == "wide":
        gt[50:950, 50:950] = 1
    fold_maps = [_hot_cam(64, 32, 32) for _ in range(K_FOLDS)]

    result = score_fold_saliency(fold_maps, gt, [0.5])

    # The scorer never resizes or re-binarises the GT: foreground count is unchanged.
    assert int(gt.sum()) == foreground
    assert int(np.count_nonzero(gt)) == foreground
    assert result["ensemble_saliency"].shape == gt.shape
    for rule in (RULE_MAIN, RULE_LEGACY):
        metric = result["scores"][rule][0.5]
        for key in ("iou", "precision", "coverage"):
            assert np.isfinite(metric[key])
            assert 0.0 <= metric[key] <= 1.0


def test_constant_mean_map_yields_empty_prediction():
    """PTX-17 / §7.2: no contrast -> `constant_image_value` -> empty mask."""
    gt = np.zeros((32, 32), dtype=np.uint8)
    gt[8:16, 8:16] = 1
    fold_maps = [np.full((16, 16), 0.4, dtype=np.float32) for _ in range(K_FOLDS)]

    result = score_fold_saliency(fold_maps, gt, [0.25])

    assert np.allclose(
        result["normalized_saliency"],
        float(cfg.ptx.corrected.constant_image_value),
    )
    metric = result["scores"][RULE_MAIN][0.25]
    assert metric["empty_pred_mask"] is True
    assert metric["coverage"] == 0.0
    assert metric["precision"] == 0.0


def test_missing_fold_chunk_raises(tmp_path):
    """PTX-18 / §7.3: a missing fold file raises instead of a partial ensemble."""
    cache_dir, dataset, record = _two_image_run(tmp_path)
    for path in fold_chunk_paths(cache_dir, 2, 0):
        os.remove(path)

    with pytest.raises(RuntimeError, match="incomplete"):
        _scores(cache_dir, dataset, record)


def test_mismatched_fold_image_ids_raise(tmp_path):
    """PTX-18 / §7.3: chunks of one chunk index must hold the same ImageId set."""
    cache_dir, dataset, record = _two_image_run(tmp_path)
    logits_path, sm_path, meta_path = fold_chunk_paths(cache_dir, 1, 0)
    for path in (logits_path, sm_path, meta_path):
        os.remove(path)
    save_fold_chunk(
        cache_dir,
        1,
        0,
        {"p1": 0.5, "x9": 0.5},
        {"p1": _stored(_hot_cam(16, 1, 1)), "x9": _stored(_hot_cam(16, 1, 1))},
        record["folds"][1]["sha256"],
    )

    with pytest.raises(RuntimeError, match="do not match the dataset slice"):
        _scores(cache_dir, dataset, record)


def test_chunk_with_other_checkpoint_sha_raises(tmp_path):
    """PTX-18 / §7.3: a chunk written by another checkpoint is rejected."""
    cache_dir, dataset, record = _two_image_run(tmp_path)
    logits_path, sm_path, meta_path = fold_chunk_paths(cache_dir, 3, 0)
    for path in (logits_path, sm_path, meta_path):
        os.remove(path)
    save_fold_chunk(
        cache_dir,
        3,
        0,
        {"p1": 0.5, "p2": 0.5},
        {"p1": _stored(_hot_cam(16, 1, 1)), "p2": _stored(_hot_cam(16, 1, 1))},
        "another-checkpoint-sha",
    )

    with pytest.raises(RuntimeError, match="was written by checkpoint"):
        _scores(cache_dir, dataset, record)


def test_cache_dir_is_unique_per_checkpoint_set(tmp_path):
    """PTX-18 / §7.3 (Y5): the cache directory is not reused across runs."""
    shas = [f"sha{fold}" for fold in range(K_FOLDS)]
    path = create_cache_dir(tmp_path, "siim", "resnet", 64, shas)
    assert path.is_dir()
    assert "resnet" in path.name and "64" in path.name
    assert (
        cache_dir_for_run(tmp_path, "siim", "resnet", 64, shas[:4] + ["other"]) != path
    )
    assert cache_dir_for_run(tmp_path, "siim", "resnet", 128, shas) != path
    with pytest.raises(RuntimeError, match="already exists"):
        create_cache_dir(tmp_path, "siim", "resnet", 64, shas)
    with pytest.raises(ValueError, match="expected 5 checkpoint hashes"):
        cache_dir_for_run(tmp_path, "siim", "resnet", 64, shas[:4])


def test_wrong_negative_is_scored_in_k1(tmp_path):
    """PTX-04 / §7.4: a wrong negative keeps its CAM score in K1 and counts as 0 in joint."""
    gt_fn = np.zeros((32, 32), dtype=np.uint8)
    gt_fn[8:20, 8:20] = 1
    gt_tp = np.zeros((32, 32), dtype=np.uint8)
    gt_tp[4:12, 4:12] = 1
    items = [_item("fn1", 1, gt_fn), _item("tp1", 1, gt_tp)]
    probs = {"fn1": 0.2, "tp1": 0.9}
    cams = {
        "fn1": [_hot_cam(16, 6 + fold, 6 + fold) for fold in range(K_FOLDS)],
        "tp1": [_hot_cam(16, 2 + fold, 2 + fold) for fold in range(K_FOLDS)],
    }
    record = _record(tau=0.5)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    _write_run(cache_dir, items, probs, cams, record)

    results = _scores(cache_dir, _MaskedDataset(items), record)
    main_rows = {
        row["image_id"]: row
        for row in results[RULE_MAIN]["rows"]
        if row["threshold"] == 0.5
    }

    assert main_rows["fn1"]["decision"] == 0
    assert main_rows["fn1"]["iou"] > 0.0  # CAM still localises; not forced to 0
    summary = results[RULE_MAIN]["summaries"]["0.5"]
    assert summary["all_positive"]["n"] == 2
    assert summary["tp_conditional"]["n"] == 1
    assert summary["joint"]["n"] == 2
    # Joint (reference bridge): the wrong negative contributes 0.
    assert summary["joint"]["iou_mean"] == pytest.approx(main_rows["tp1"]["iou"] / 2)
    assert summary["joint"]["iou_mean"] < summary["tp_conditional"]["iou_mean"]
    assert results[RULE_MAIN]["n_tp"] == 1
    assert results[RULE_MAIN]["sensitivity"] == pytest.approx(0.5)


def test_k1_key_set_equals_positive_image_ids(tmp_path):
    """PTX-04 / §7.4: one CSV per rule keyed by exactly the GT-positive ImageId set."""
    gt_pos = np.zeros((32, 32), dtype=np.uint8)
    gt_pos[8:16, 8:16] = 1
    gt_neg = np.zeros((32, 32), dtype=np.uint8)
    items = [
        _item("p1", 1, gt_pos),
        _item("p2", 1, gt_pos),
        _item("n1", 0, gt_neg),
    ]
    probs = {"p1": 0.9, "p2": 0.3, "n1": 0.1}
    cams = {
        image_id: [_hot_cam(16, 4 + fold, 4 + fold) for fold in range(K_FOLDS)]
        for image_id in ("p1", "p2", "n1")
    }
    record = _record(tau=0.5)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    _write_run(cache_dir, items, probs, cams, record)
    positive_ids = {"p1", "p2"}

    results = _scores(cache_dir, _MaskedDataset(items), record)

    for rule in (RULE_MAIN, RULE_LEGACY):
        assert {row["image_id"] for row in results[rule]["rows"]} == positive_ids
        assert {row["image_id"] for row in results[rule]["negative_rows"]} == {"n1"}
        assert results[rule]["n_positive"] == 2
        assert results[rule]["n_negative"] == 1
        assert set(results[rule]["summaries"]) == {"0.25", "0.5", "0.75"}

    written = write_saliency_outputs(
        results, tmp_path / "out", "synthetic", "resnet", 64
    )
    assert written[RULE_MAIN]["csv"] != written[RULE_LEGACY]["csv"]
    frame = pd.read_csv(written[RULE_MAIN]["csv"], dtype={"image_id": str})
    assert list(frame.columns) == SCORE_COLUMNS
    assert set(frame["image_id"]) == positive_ids
    assert set(frame["threshold"]) == set(cfg.ptx.corrected.saliency_thresholds)
    negatives = pd.read_csv(
        written[RULE_MAIN]["negatives_csv"], dtype={"image_id": str}
    )
    assert list(negatives.columns) == NEGATIVE_SCORE_COLUMNS
    assert set(negatives["image_id"]) == {"n1"}

    main_summary = json.loads(Path(written[RULE_MAIN]["summary"]).read_text())
    legacy_summary = json.loads(Path(written[RULE_LEGACY]["summary"]).read_text())
    assert main_summary["n_positive"] == 2
    assert main_summary["n_negative"] == 1
    assert main_summary["sensitivity_analysis"] is False
    assert legacy_summary["sensitivity_analysis"] is True
    # Empty predicted masks are reported per threshold with their denominator.
    assert main_summary["summaries"]["0.75"]["empty_pred_mask"]["n"] == 2


def test_predict_fold_chunked_records_checkpoint_identity(tmp_path):
    """PTX-18 / §7.3: the writer stores the checkpoint SHA-256 with every chunk."""
    from torch.utils.data import DataLoader

    from sharpness_matters.ptx.evaluation.evaluate_saliency import collate_fn
    from sharpness_matters.ptx.utils.saliency_eval import (
        load_fold_chunk,
        predict_fold_chunked,
    )

    items = [
        _item("p1", 1, np.zeros((16, 16), dtype=np.uint8), img_size=8),
        _item("p2", 1, np.zeros((20, 20), dtype=np.uint8), img_size=8),
    ]
    loader = DataLoader(
        _MaskedDataset(items), batch_size=2, shuffle=False, collate_fn=collate_fn
    )
    model = _toy_model()
    predict_fold_chunked(
        loader,
        model,
        [model.features[0]],
        0,
        checkpoint_sha256="sha0",
        checkpoint_path="ckpt.pt",
        chunk_size=2,
        output_dir=str(tmp_path),
    )

    probabilities, saliency = load_fold_chunk(tmp_path, 0, 0, "sha0")
    assert sorted(probabilities) == ["p1", "p2"]
    assert saliency["p1"].shape == (8, 8, 1)
    with pytest.raises(RuntimeError, match="was written by checkpoint"):
        load_fold_chunk(tmp_path, 0, 0, "other-sha")


def _toy_model():
    """Tiny convolutional module so GradCAM runs without a pretrained network."""

    class _ToyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.features = torch.nn.Sequential(
                torch.nn.Conv2d(1, 4, 3, padding=1), torch.nn.ReLU()
            )
            self.head = torch.nn.Sequential(
                torch.nn.AdaptiveAvgPool2d(1),
                torch.nn.Flatten(),
                torch.nn.Linear(4, 1),
            )

        def forward(self, x):
            return self.head(self.features(x))

    return _ToyModel()


def test_run_experiment_writes_outputs_and_cleans_cache(tmp_path, monkeypatch):
    """PTX-18/PTX-04 end to end: recorded checkpoints, unique cache, per-rule outputs."""
    from sharpness_matters.ptx.evaluation import evaluate_saliency as script
    from torch.utils.data import DataLoader

    gt = np.zeros((32, 32), dtype=np.uint8)
    gt[8:16, 8:16] = 1
    items = [_item("p1", 1, gt, img_size=8), _item("p2", 1, gt, img_size=8)]
    record = _record()

    monkeypatch.setattr(script, "HOME", tmp_path)
    monkeypatch.setattr(script, "load_cv_record", lambda model, size: record)
    monkeypatch.setattr(
        script,
        "fold_checkpoints",
        lambda rec: [f"ckpt{fold}.pt" for fold in range(K_FOLDS)],
    )
    # The author's loader uses 8 workers; in-process keeps the test fast.
    monkeypatch.setattr(
        script,
        "DataLoader",
        lambda *args, **kwargs: DataLoader(*args, **{**kwargs, "num_workers": 0}),
    )

    def _fake_load(path, model_name, device):
        model = _toy_model()
        return model, [model.features[0]]

    monkeypatch.setattr(script, "_load_fold_model", _fake_load)

    results = script.run_experiment(
        _MaskedDataset(items),
        8,
        "resnet",
        "synthetic",
        chunk_size=2,
        save_maps=False,
        output_dir=str(tmp_path / "out"),
        device=torch.device("cpu"),
    )

    assert set(results) == {RULE_MAIN, RULE_LEGACY}
    for rule in (RULE_MAIN, RULE_LEGACY):
        assert (tmp_path / "out" / f"saliency_synthetic_resnet_8_{rule}.csv").is_file()
        assert (tmp_path / "out" / f"saliency_synthetic_resnet_8_{rule}.json").is_file()
    # The run-owned chunk cache never survives, so it cannot be reused silently.
    assert not list((tmp_path / "output").glob("chunk_predictions_*"))


class _VisualisingDataset(_MaskedDataset):
    """`_MaskedDataset` plus a display-grid `visualize_segmentation` (author semantics)."""

    def visualize_segmentation(self, image, mask, alpha=0.5, mask_color=(1, 0, 0)):
        image = np.array(image * 255).astype(np.uint8)
        foreground = np.where(np.asarray(mask) > 0, 255, 0).astype(np.uint8)
        overlay = np.zeros_like(image, dtype=np.uint8)
        for channel in range(3):
            overlay[..., channel] = foreground * mask_color[channel]
        return overlay


def test_positive_with_empty_gt_raises(tmp_path):
    """PTX-04 / §7.4: a label==1 image with an empty reference mask is not scorable."""
    gt = np.zeros((32, 32), dtype=np.uint8)
    gt[8:16, 8:16] = 1
    items = [
        _item("p1", 1, gt),
        _item("bad1", 1, np.zeros((32, 32), dtype=np.uint8)),
        _item("bad2", 1, np.zeros((40, 40), dtype=np.uint8)),
    ]
    probs = {"p1": 0.9, "bad1": 0.8, "bad2": 0.7}
    cams = {
        image_id: [_hot_cam(16, 4 + fold, 4 + fold) for fold in range(K_FOLDS)]
        for image_id in probs
    }
    record = _record()
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    _write_run(cache_dir, items, probs, cams, record)  # two chunks: [p1, bad1], [bad2]

    with pytest.raises(RuntimeError, match="empty") as error:
        _scores(cache_dir, _MaskedDataset(items), record)

    message = str(error.value)
    # Every offending ImageId is listed, across chunk boundaries; healthy ids are not.
    assert "bad1" in message
    assert "bad2" in message
    assert "p1" not in message


def test_save_maps_moves_maps_to_the_display_grid(tmp_path, monkeypatch):
    """PTX-02 / §7.1: the author's three visualisations still render after the GT grid change."""
    from sharpness_matters.ptx.utils import saliency_eval as scorer

    gt = np.zeros((32, 32), dtype=np.uint8)
    gt[8:16, 8:16] = 1
    items = [_item("p1", 1, gt), _item("p2", 1, gt)]
    probs = {"p1": 0.9, "p2": 0.2}
    cams = {
        image_id: [_hot_cam(16, 4 + fold, 4 + fold) for fold in range(K_FOLDS)]
        for image_id in probs
    }
    record = _record(tau=0.5)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    _write_run(cache_dir, items, probs, cams, record)
    monkeypatch.setattr(scorer, "HOME", tmp_path)

    scorer.compute_scores_chunked(
        _VisualisingDataset(items),
        str(cache_dir),
        record,
        "resnet",
        64,
        "synthetic",
        chunk_size=2,
        save_maps=True,
    )

    base = tmp_path / "output" / "resnet_maps" / "maps_holdout" / "maps_64"
    # One GT overlay and one CAM overlay per image, plus three thresholded masks.
    assert len(list((base / "gt").glob("*.jpg"))) == 2
    assert len(list((base / "sm").glob("*.jpg"))) == 2
    for threshold in cfg.ptx.corrected.saliency_thresholds:
        assert len(list((base / f"masks_{threshold}").glob("*.jpg"))) == 2


def test_collate_keeps_masks_as_a_list():
    """PTX-02 / §7.1: ragged reference-grid masks must not be stacked by the loader."""
    from sharpness_matters.ptx.evaluation.evaluate_saliency import collate_fn

    batch = [
        _item("a", 1, np.zeros((32, 32), dtype=np.uint8), img_size=8),
        _item("b", 1, np.zeros((40, 40), dtype=np.uint8), img_size=8),
    ]

    collated = collate_fn(batch)

    assert collated["data"].shape == (2, 1, 8, 8)
    assert isinstance(collated["mask"], list)
    assert [mask.shape for mask in collated["mask"]] == [(32, 32), (40, 40)]

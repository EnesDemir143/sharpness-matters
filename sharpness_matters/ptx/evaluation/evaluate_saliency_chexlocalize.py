"""Stage-1 (A) saliency evaluation on the added E4 (CheXlocalize) row.

The author's saliency rule is used unchanged: per-fold Grad-CAM chunks with the
author's `predict_fold_chunked`, fold mean, then the author's fixed thresholds
(0.25/0.5/0.75), the author's decision-conditional scoring (`compute_scores_chunked`,
operating point 0.5), and the author's summary fields (including `num_accurate`, which
is reported as the author reports it).

Only the dataset and the output directory labels differ from `evaluate_saliency.py`:
images come from the frozen E4 XAI manifest, GT masks are resized to the input size and
scaled by 255 exactly as the author's masked loaders do.

Usage:
    uv run python evaluation/evaluate_saliency_chexlocalize.py
"""

import gc
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from sharpness_matters.config.load_config import cfg
from sharpness_matters.ptx.datasets.external_eval_datasets import (
    CheXlocalizeEvalDataset,
    load_manifest,
)
from sharpness_matters.ptx.models.cnn import CNNBinaryClassifier
from sharpness_matters.ptx.utils.logging_utils import initialize_logger
from sharpness_matters.ptx.utils.saliency_eval import (
    compute_scores_chunked,
    predict_fold_chunked,
)

logger = initialize_logger(level="debug")
HOME = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = cfg.ptx.model.checkpoint_dir


def collate_fn(batch):
    """The author's collate function (`evaluate_saliency.py`)."""
    collated = {}
    for key in batch[0]:
        if key == "data":
            collated[key] = torch.stack([b[key] for b in batch], dim=0)
        else:
            collated[key] = [b[key] for b in batch]
    return collated


def _write_xai_view(frame: pd.DataFrame) -> str:
    """Read-only view of the frozen XAI manifest with the columns the loader needs."""
    import tempfile

    directory = tempfile.mkdtemp(prefix="stage1_chexlocalize_xai_")
    path = os.path.join(directory, "chexlocalize_xai_view.csv")
    frame[["image_id", "split", "path", "label"]].to_csv(path, index=False)
    return path


def run_experiment(
    dataset,
    img_size: int,
    model_name: str,
    threshold: float = 0.5,
    chunk_size: int = 5,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    """Mirror of the author's `evaluate_saliency.run_experiment`, E4 dataset/labels only."""
    pred_loader = DataLoader(
        dataset, batch_size=4, shuffle=False, num_workers=8, pin_memory=True
    )
    cache_dir = f"{HOME}/output/chunk_predictions_{model_name}_{img_size}_chexlocalize"
    os.makedirs(cache_dir, exist_ok=True)
    for fold in range(5):
        ckpt_dir = f"{CHECKPOINT_DIR}/{model_name}/{model_name}_{img_size}_logs/lightning_logs/version_{fold}/checkpoints/"
        ckpt_file = os.listdir(ckpt_dir)[0]
        checkpoint_path = os.path.join(ckpt_dir, ckpt_file)
        if device.type == "cpu":
            logger.warning("Device is CPU; saliency evaluation may be slow.")
        model = CNNBinaryClassifier.load_from_checkpoint(
            checkpoint_path, map_location=device
        )
        target_layers = (
            [model.model.features.denseblock4]
            if "densenet" in model_name
            else [model.model.layer4[-1].conv3]
        )
        model.eval()
        model.to(device)
        predict_fold_chunked(
            pred_loader,
            model,
            target_layers,
            fold,
            chunk_size=chunk_size,
            output_dir=cache_dir,
        )
        del model
        gc.collect()
    iou, precision, coverage = compute_scores_chunked(
        dataset,
        cache_dir,
        model_name,
        img_size,
        collate_fn,
        logger,
        validation_mode="chexlocalize",
        threshold=threshold,
        chunk_size=chunk_size,
    )
    complete_miss = sum(1 for v in iou.values() if v == 0)
    iou_values = np.array(list(iou.values()))
    precision_values = np.array(list(precision.values()))
    summary = {
        "iou": f"{iou_values.mean():.4f} +- {iou_values.std():.4f}",
        "precision": f"{precision_values.mean():.4f} +- {precision_values.std():.4f}",
        "coverage": float(np.mean(list(coverage.values()))),
        "num_accurate": f"{np.round(100 * len(iou) / len(dataset), 2)}%",
        "complete_miss": f"{np.round(100 * complete_miss / len(dataset), 2)}%",
    }
    save_path = f"{HOME}/output/saliency_eval/{model_name}"
    os.makedirs(save_path, exist_ok=True)
    pd.DataFrame(
        [{"image_id": key, "iou": value, "precision": precision[key], "coverage": coverage.get(key)} for key, value in iou.items()]
    ).to_csv(
        os.path.join(save_path, f"chexlocalize_iou_{img_size}_{threshold}.csv"),
        index=False,
    )
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    gc.collect()
    return summary


if __name__ == "__main__":
    seed = 42
    pl.seed_everything(seed)
    frame = load_manifest("chexlocalize_xai.csv")  # hash-verified, read-only
    thresholds = [0.25, 0.75, 0.5]  # the author's threshold set
    model_names = ["resnet", "densenet"]
    for threshold in thresholds:
        for model_name in model_names:
            out = []
            for img_size in [512]:  # stage-1 run scope: 512 only (protocol §6)
                torch.cuda.empty_cache()
                dataset = CheXlocalizeEvalDataset(
                    manifest_csv=_write_xai_view(frame), img_size=img_size, with_mask=True
                )
                logger.info(
                    f"Running CheXlocalize: size {img_size}, threshold {threshold}, "
                    f"model {model_name}"
                )
                res = run_experiment(
                    dataset, img_size, model_name, threshold=threshold, chunk_size=10
                )
                res["img_size"] = img_size
                out.append(res)
                df = pd.DataFrame(out)
                df = df[["img_size"] + [c for c in df.columns if c != "img_size"]]
                df.to_csv(
                    f"{HOME}/output/saliency_thresholded_chexlocalize_{model_name}_{threshold}.csv",
                    index=False,
                )

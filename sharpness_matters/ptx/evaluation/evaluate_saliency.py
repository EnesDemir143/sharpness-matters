"""
Module for performing saliency evaluation experiments on the Holdout Dataset.

This module evaluates saliency maps produced by different model architectures (e.g., DenseNet, ResNet).
It selects a subset of the dataset based on the specified evaluation mode, runs predictions in chunks
(Computation is done chunk wise to prevent crashes due to CPU memory over-utilization)
for each cross-validation fold, computes metrics such as Intersection over Union (IoU),
Precision (%-IN), and coverage, and finally summarizes the results.

B (PTX-18 / §7.3): current -> the fold checkpoint was `os.listdir(...)[0]` and the chunk cache
was a fixed directory reused with `exist_ok=True`; problem -> the evaluated weights were not the
recorded ones and stale chunks could be averaged in; new -> checkpoints come from the cv record
(`load_cv_record`, `fold_checkpoints`), every chunk records its checkpoint SHA-256, and the cache
directory is unique per run. A missing fold, a mismatched ImageId set or a different checkpoint
raises instead of being averaged into a partial ensemble.
B (PTX-04 / §7.4): one CSV and one JSON summary per (model, img_size, dataset, rule); saliency is
scored for every GT-positive image from its positive-class CAM, independent of the decision.
B (protocol §6): B evaluates input size 512 only; `train_kfold.main` and `evaluate_kfold` use
[512] and a cv record exists only for 512, so the author's 64..1024 sweep is not run here.

Usage:
    Run this module directly to execute the experiments:
        python3 evaluate_saliency.py
    Output is stored as
    `output/saliency_eval/{model_name}/saliency_{dataset}_{model}_{img_size}_{rule}.csv`
    and a per-model table `output/saliency_eval/saliency_thresholded_{dataset}_{model}_{rule}.csv`.
"""

import os
import glob
import gc
import shutil
from pathlib import Path
import torch
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from typing import Dict, Any, List, Optional, Tuple, Union

from sharpness_matters.ptx.corrected.cv_record import fold_checkpoints, load_cv_record
from sharpness_matters.ptx.datasets.siim_acr_dataset import ChestXrayDataset
from sharpness_matters.ptx.models.cnn import CNNBinaryClassifier
from sharpness_matters.ptx.utils.logging_utils import initialize_logger
from sharpness_matters.ptx.utils.saliency_eval import (
    RULE_LEGACY,
    RULE_MAIN,
    compute_scores_chunked,
    create_cache_dir,
    fold_checkpoint_shas,
    predict_fold_chunked,
    summary_table,
    write_saliency_outputs,
)
from sharpness_matters.config.load_config import cfg

logger = initialize_logger(level="debug")
HOME = Path(__file__).resolve().parent.parent


def collate_fn(
    batch: List[Dict[str, Any]]
) -> Dict[str, Union[torch.Tensor, List[Any]]]:
    """
    Simple collate function that stacks tensors for key 'data' and aggregates other fields in lists.

    `mask` must stay a list: the GT mask lives on the reference annotation grid (contract 6),
    which differs per image for PTX-498, so stacking it would fail or silently resize it
    (PTX-02 / §7.1).

    Args:
        batch (List[Dict[str, Any]]): List of sample dictionaries from dataset

    Returns:
        Dict[str, Union[torch.Tensor, List[Any]]]: Collated batch with stacked tensors and aggregated lists
    """
    collated = {}
    for key in batch[0]:
        if key == "data":
            collated[key] = torch.stack([b[key] for b in batch], dim=0)
        else:
            collated[key] = [b[key] for b in batch]
    return collated


def _load_fold_model(
    checkpoint_path: str, model_name: str, device: torch.device
) -> Tuple[torch.nn.Module, List[Any]]:
    """Load one fold checkpoint with the author's target layer (unchanged CAM method)."""
    if "densenet" not in model_name and "resnet" not in model_name:
        raise ValueError(f"unknown model name: {model_name!r}")
    model = CNNBinaryClassifier.load_from_checkpoint(
        checkpoint_path, map_location=device
    )
    if "densenet" in model_name:
        target_layers = [model.model.features.denseblock4]
    else:
        target_layers = [model.model.layer4[-1].conv3]
    model.eval()
    model.to(device)
    return model, target_layers


def run_experiment(
    dataset: Dataset,
    img_size: int,
    model_name: str,
    dataset_name: str,
    validation_mode: str = "holdout",
    chunk_size: int = 5,
    save_maps: bool = True,
    output_dir: Optional[str] = None,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> Dict[str, Dict[str, Any]]:
    """
    Run saliency evaluation experiment for the holdout dataset with specified model and parameters.

    Args:
        dataset (Dataset): masked PyTorch dataset; `mask` is uint8 {0,1} on the reference grid
        img_size (int): Input image size for the model
        model_name (str): Name of the model architecture ('resnet' or 'densenet')
        dataset_name (str): dataset label used in the output file names and the summary
        validation_mode (str): 'holdout' or 'ood' (visualisation directory only)
        chunk_size (int): Size of chunks for batch processing to manage memory
        save_maps (bool): Write the author's map visualisations under `ptx/output`
        output_dir (Optional[str]): directory of the per-run CSV/JSON; defaults to
            `output/saliency_eval/{model_name}`

    Returns:
        Dict[str, Dict[str, Any]]: one result per fold-aggregation rule (`main`, `legacy`)
    """
    record = load_cv_record(model_name, img_size)
    checkpoints = fold_checkpoints(record)
    checkpoint_sha256s = fold_checkpoint_shas(record)
    if len(checkpoints) != len(checkpoint_sha256s):
        raise RuntimeError(
            f"cv record lists {len(checkpoints)} checkpoints but "
            f"{len(checkpoint_sha256s)} checkpoint hashes"
        )
    cache_dir = create_cache_dir(
        f"{HOME}/output", dataset_name, model_name, img_size, checkpoint_sha256s
    )
    if output_dir is None:
        output_dir = f"{HOME}/output/saliency_eval/{model_name}"
    # Batch size 4 as in the author's runner; the explicit collate keeps `mask` a list.
    pred_loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    try:
        for fold, checkpoint_path in enumerate(checkpoints):
            if device.type == "cpu":
                logger.warning("Device is CPU; saliency evaluation may be slow.")
            model, target_layers = _load_fold_model(checkpoint_path, model_name, device)
            predict_fold_chunked(
                pred_loader,
                model,
                target_layers,
                fold,
                checkpoint_sha256=checkpoint_sha256s[fold],
                checkpoint_path=checkpoint_path,
                chunk_size=chunk_size,
                output_dir=str(cache_dir),
            )
            del model
            gc.collect()
        results = compute_scores_chunked(
            dataset,
            str(cache_dir),
            record,
            model_name,
            img_size,
            dataset_name,
            validation_mode=validation_mode,
            chunk_size=chunk_size,
            save_maps=save_maps,
        )
    finally:
        # The cache directory is unique per run (PTX-18), so it is removed in every
        # outcome; a killed run leaves one behind and the next run raises on it instead
        # of silently reusing partial chunks.
        shutil.rmtree(cache_dir, ignore_errors=True)
    write_saliency_outputs(results, output_dir, dataset_name, model_name, img_size)
    return results


if __name__ == "__main__":
    seed = 42
    pl.seed_everything(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)

    DATA_DIR = cfg.ptx.siim_acr.root_dir
    train_dicom_dir = os.path.join(DATA_DIR, "dicom-images-train")
    train_glob = f"{train_dicom_dir}/*/*/*.dcm"
    train_names = [f for f in sorted(glob.glob(train_glob))]

    dataset_name = "siim_holdout"
    model_names = ["resnet", "densenet"]
    output_root = f"{HOME}/output/saliency_eval"
    for model_name in model_names:
        table_rows = {RULE_MAIN: [], RULE_LEGACY: []}
        for img_size in [512]:  # B (protocol §6): B evaluates 512 only
            torch.cuda.empty_cache()
            dataset = ChestXrayDataset(
                train_names,
                f"{DATA_DIR}/train-rle.csv",
                split="test",
                img_size=img_size,
            )
            logger.info(f"Running: Image size: {img_size}, Model: {model_name}")
            results = run_experiment(
                dataset,
                img_size,
                model_name,
                dataset_name,
                validation_mode="holdout",
                chunk_size=10,
                output_dir=f"{output_root}/{model_name}",
            )
            for rule in (RULE_MAIN, RULE_LEGACY):
                table_rows[rule].extend(summary_table(results[rule]))
            logger.debug(f"Results: {img_size} {model_name} done")
        for rule in (RULE_MAIN, RULE_LEGACY):
            df = pd.DataFrame(table_rows[rule])
            table_path = f"{output_root}/saliency_thresholded_{dataset_name}_{model_name}_{rule}.csv"
            df.to_csv(table_path, index=False)
            logger.debug(f"Saved {table_path}")

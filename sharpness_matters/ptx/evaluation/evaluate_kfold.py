"""
Module for evaluating chest X-ray classification performance across multiple resolutions.

This module uses a chest X-ray dataset along with a binary classification model (CNNBinaryClassifier)
to evaluate classification performance metrics such as accuracy, AUC, and F1 score across different
image resolutions and model architectures (e.g., DenseNet, ResNet). It includes a custom collate function
to handle bounding box information in the dataset and fetches evaluation results per resolution.
The predicted logits and ground truth labels are saved for further analysis.

Usage:
    Run this module directly:
        python3 evaluate_kfold.py --validation_mode=<validation_dataset>
    Output CSV files with evaluation metrics for each model will be saved in 'classification_{model_name}.csv'.
"""

import os
import glob
import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import ConcatDataset, Dataset
import pandas as pd
import sys
import click
from pathlib import Path
from typing import Dict, Any, Tuple

from sharpness_matters.ptx.datasets.siim_acr_classification_dataset import (
    ChestXrayDataset,
)
from sharpness_matters.ptx.datasets.vin_dataset import VinDataset
from sharpness_matters.ptx.datasets.ptx498_dataset import OODTestDataset
from sharpness_matters.ptx.models.cnn import CNNBinaryClassifier
from sharpness_matters.ptx.utils.classification_eval import eval_res
from sharpness_matters.ptx.utils.logging_utils import initialize_logger
from sharpness_matters.config.load_config import cfg

logger = initialize_logger(level="debug")
DATA_DIR = cfg.ptx.siim_acr.root_dir
HOME = Path(__file__).resolve().parent.parent
logger.debug(f"Output Root: {HOME}")


def fetch_results_per_resolution(
    img_size: int = 512, model_name: str = "resnet", validation_mode: str = "holdout"
) -> Dict[str, Any]:
    """
    Evaluate model performance for a specific image resolution and validation dataset.

    Args:
        img_size (int): Input image size for model evaluation
        model_name (str): Model architecture ('resnet' or 'densenet')
        validation_mode (str): Validation dataset type ('holdout' or 'ood')

    Returns:
        Dict[str, Any]: Dictionary containing evaluation metrics
    """
    # Extract training files required to initialize dataset
    if validation_mode == "holdout":
        train_dicom_dir = os.path.join(DATA_DIR, "dicom-images-train")
        train_glob = f"{train_dicom_dir}/*/*/*.dcm"
        train_names = [f for f in sorted(glob.glob(train_glob))]
        dataset = ChestXrayDataset(
            train_names, f"{DATA_DIR}/train-rle.csv", split="test", img_size=img_size
        )
        ids = list(dataset.image_ids)
        sources = ["siim_acr"] * len(ids)
        expected = [int(dataset.labels[i]) for i in ids]
    elif validation_mode == "ood":
        vin_ds = VinDataset(img_size=img_size)
        ptx_ds = OODTestDataset(img_size=img_size)
        dataset = ConcatDataset([vin_ds, ptx_ds])
        ids = list(vin_ds.image_ids) + list(ptx_ds.image_ids)
        sources = ["vindr"] * len(vin_ds.image_ids) + ["ptx498"] * len(
            ptx_ds.image_ids
        )
        expected = [int(vin_ds.labels[i]) for i in vin_ds.image_ids] + [1] * len(
            ptx_ds.image_ids
        )
    else:
        raise ValueError("Invalid validation mode: Choose one of holdout, ood")
    # Compute evaluation metrics for given resolution
    all_labels, all_logits_ensemble, ensemble_metrics = eval_res(
        dataset, img_size, model_name, CNNBinaryClassifier, logger
    )
    # Stage 1: verify prediction order matches dataset index order
    if len(ids) != len(all_labels):
        raise RuntimeError(
            f"Prediction/label length mismatch: {len(ids)} dataset ids vs "
            f"{len(all_labels)} labels"
        )
    if len(ids) != len(all_logits_ensemble):
        raise RuntimeError(
            f"Prediction/label length mismatch: {len(ids)} dataset ids vs "
            f"{len(all_logits_ensemble)} logits"
        )
    if not np.array_equal(np.asarray(expected), np.asarray(all_labels).astype(int)):
        raise RuntimeError(
            "Prediction order mismatch: dataset-order labels differ from eval labels"
        )
    # Save predictions to avoid re-running for stat analysis
    os.makedirs(f"{HOME}/output/classif_preds/", exist_ok=True)
    np.save(
        f"{HOME}/output/classif_preds/{validation_mode}pred_{model_name}_{img_size}.npy",
        all_logits_ensemble,
    )
    np.save(f"{HOME}/output/classif_preds/{validation_mode}gt.npy", all_labels)
    # Stage 1: ID-keyed predictions (score = fold-mean sigmoid probability, as in eval_res)
    pd.DataFrame(
        {
            "image_id": ids,
            "source": sources,
            "label": np.asarray(all_labels).astype(int),
            "score": all_logits_ensemble,
        }
    ).to_csv(
        f"{HOME}/output/classif_preds/{validation_mode}pred_{model_name}_{img_size}_ids.csv",
        index=False,
    )
    return ensemble_metrics


@click.command()
@click.option(
    "--validation_mode",
    type=str,
    required=True,
    help="Choose validation dataset: One of holdout, ood",
)
def main(validation_mode: str) -> None:
    """
    Main function to evaluate model performance across multiple resolutions and architectures.

    Args:
        validation_mode (str): Validation dataset type ('holdout' or 'ood')
    """
    seed = 42
    pl.seed_everything(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    model_names = ["resnet", "densenet"]
    for model_name in model_names:
        out = []
        for img_size in [64, 128, 224, 512, 768, 1024]:
            res = fetch_results_per_resolution(img_size, model_name, validation_mode)
            res["img_size"] = img_size
            out.append(res)
            df = pd.DataFrame(out)
            columns_order = ["img_size"] + [
                col for col in df.columns if col != "img_size"
            ]
            df = df[columns_order]
            df.to_csv(
                f"{HOME}/output/classification_{model_name}_{validation_mode}.csv",
                index=False,
            )


if __name__ == "__main__":
    main()

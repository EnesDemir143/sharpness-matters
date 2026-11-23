"""
Module for evaluating chest X-ray classification performance.

This module uses a chest X-ray dataset along with a binary classification model (CNNBinaryClassifier)
to evaluate classification performance metrics such as accuracy, AUC, and F1 score.
The fold checkpoints and the operating threshold come from the run's cv record; the
predictions are saved keyed by ImageId for further analysis.

# B (PTX-18 / PTX-07, Section 6.1): the fold checkpoints are read from the run's
#   `cv_record.json` by `eval_res` (`load_cv_record`), which verifies every fold and its
#   sha256; the author's `os.listdir(...)[0]` discovery is gone.
# B (PTX-08 / Section 6.4): predictions are written as an ID-keyed CSV
#   (`image_id,source,label,prob,prob_fold*`) named by validation_mode.
# B (Section 6.3): the decision threshold is the baseline's own OOF tau.
# B (Section 6.5 / §8.1): the legacy `.npy` files are still written because
#   `p_value_classification.py` reads them (`oodpred_*`, `holdoutpred_*`, `*gt.npy`);
#   that script is not part of B, but keeping the files avoids breaking a reader on the
#   branch. E5-clean has no legacy name, so it gets its own `ood_clean` prefix.

Usage:
    Run this module directly:
        python3 evaluate_kfold.py --validation_mode=<validation_dataset>
        python3 evaluate_kfold.py --validation_mode=ood_clean --e5_manifest=manifests/reference_like_clean.csv
    Output CSV files with evaluation metrics for each model will be saved in
    'classification_{model_name}_{validation_mode}.csv'.
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


def _legacy_npy_prefix(validation_mode: str) -> str:
    """Prefix of the author's `.npy` files, which `p_value_classification.py` reads.

    The author's E5 run is the reference-like manifest, so that run keeps the `ood`
    prefix; E5-clean gets its own prefix rather than overwriting it.
    """
    if validation_mode == "holdout":
        return "holdout"
    if validation_mode == "ood_reference_like":
        return "ood"
    return validation_mode


def fetch_results_per_resolution(
    img_size: int = 512,
    model_name: str = "resnet",
    validation_mode: str = "holdout",
    e5_manifest: str = None,
) -> Dict[str, Any]:
    """
    Evaluate model performance for a specific image resolution and validation dataset.

    Args:
        img_size (int): Input image size for model evaluation
        model_name (str): Model architecture ('resnet' or 'densenet')
        validation_mode (str): Validation dataset ('holdout', 'ood_reference_like' or
            'ood_clean'); the E5 variants are separate runs with separate manifests.
        e5_manifest (str): ID+label manifest of the E5 run (required for the E5 modes)

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
    elif validation_mode in ("ood_reference_like", "ood_clean"):
        if not e5_manifest:
            raise ValueError(
                f"validation_mode={validation_mode} needs the E5 manifest path "
                "(reference-like and clean are different ID sets)"
            )
        vin_ds = VinDataset(manifest_csv=e5_manifest, img_size=img_size)
        ptx_ds = OODTestDataset(img_size=img_size)
        dataset = ConcatDataset([vin_ds, ptx_ds])
        ids = list(vin_ds.image_ids) + list(ptx_ds.image_ids)
        sources = ["vindr"] * len(vin_ds.image_ids) + ["ptx498"] * len(ptx_ds.image_ids)
        expected = [int(vin_ds.labels[i]) for i in vin_ds.image_ids] + [1] * len(
            ptx_ds.image_ids
        )
    else:
        raise ValueError(
            "Invalid validation mode: Choose one of holdout, ood_reference_like, ood_clean"
        )
    # Compute evaluation metrics for given resolution; predictions are ID-keyed
    pred_df, ensemble_metrics = eval_res(
        dataset,
        img_size,
        model_name,
        CNNBinaryClassifier,
        logger,
        ids=ids,
        labels=np.asarray(expected, dtype=int),
    )
    # Stage 1 / Section 6.4: verify the ID order and labels match the dataset order
    if len(ids) != len(pred_df):
        raise RuntimeError(
            f"Prediction/label length mismatch: {len(ids)} dataset ids vs "
            f"{len(pred_df)} predictions"
        )
    if list(pred_df["image_id"]) != ids:
        raise RuntimeError(
            "Prediction order mismatch: returned image_ids differ from dataset order"
        )
    if not np.array_equal(
        np.asarray(expected), pred_df["label"].to_numpy().astype(int)
    ):
        raise RuntimeError(
            "Prediction order mismatch: dataset-order labels differ from eval labels"
        )
    # Save predictions to avoid re-running for stat analysis
    out_dir = Path(HOME) / "output" / "classif_preds"
    os.makedirs(out_dir, exist_ok=True)
    prefix = _legacy_npy_prefix(validation_mode)
    np.save(
        out_dir / f"{prefix}pred_{model_name}_{img_size}.npy",
        pred_df["prob"].to_numpy(),
    )
    np.save(out_dir / f"{prefix}gt.npy", pred_df["label"].to_numpy().astype(int))
    # Section 6.4: ID-keyed predictions (score = fold-mean sigmoid probability)
    pred_df.insert(1, "source", sources)
    pred_df.to_csv(
        out_dir / f"{validation_mode}pred_{model_name}_{img_size}_ids.csv",
        index=False,
    )
    return ensemble_metrics


@click.command()
@click.option(
    "--validation_mode",
    type=click.Choice(["holdout", "ood_reference_like", "ood_clean"]),
    required=True,
    help="Choose validation dataset: One of holdout, ood_reference_like, ood_clean",
)
@click.option(
    "--e5_manifest",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="ID+label manifest of the E5 run (required for ood_reference_like / ood_clean)",
)
def main(validation_mode: str, e5_manifest: str) -> None:
    """
    Main function to run the evaluation for the baseline resolution.

    Args:
        validation_mode (str): Validation dataset ('holdout', 'ood_reference_like'
            or 'ood_clean')
        e5_manifest (str): E5 manifest path (required for the E5 modes)
    """
    if validation_mode.startswith("ood") and not e5_manifest:
        raise click.UsageError(
            "--e5_manifest is required for ood_reference_like / ood_clean"
        )
    seed = 42
    pl.seed_everything(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    model_names = ["resnet", "densenet"]
    for model_name in model_names:
        out = []
        # B (Section 5.1): only the baseline resolution is evaluated in stage 2
        for img_size in [512]:
            res = fetch_results_per_resolution(
                img_size, model_name, validation_mode, e5_manifest
            )
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

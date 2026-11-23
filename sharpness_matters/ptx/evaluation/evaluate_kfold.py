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
import tempfile
import shutil
import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import ConcatDataset, Dataset
import pandas as pd
import sys
import click
from pathlib import Path
from typing import Dict, Any, Tuple

from sklearn.metrics import roc_auc_score
from sharpness_matters.ptx.datasets.siim_acr_classification_dataset import (
    ChestXrayDataset,
)
from sharpness_matters.ptx.datasets.vin_dataset import VinDataset
from sharpness_matters.ptx.datasets.ptx498_dataset import OODTestDataset
from sharpness_matters.ptx.datasets.external_eval_datasets import (
    CheXlocalizeEvalDataset,
    ManifestVinDataset,
    load_manifest,
)
from sharpness_matters.ptx.models.cnn import CNNBinaryClassifier
from sharpness_matters.ptx.utils.classification_eval import eval_res
from sharpness_matters.ptx.utils.logging_utils import initialize_logger
from sharpness_matters.config.load_config import cfg

logger = initialize_logger(level="debug")
DATA_DIR = cfg.ptx.siim_acr.root_dir
HOME = Path(__file__).resolve().parent.parent
logger.debug(f"Output Root: {HOME}")


def _write_split_csv(frame: pd.DataFrame, key: str) -> str:
    """Write a per-split `image_id,label` view of a frozen manifest to a temp file.

    The loaders read `image_id,label` only; the frozen manifest stays untouched and is
    hashed before use, so this is a read-only view, not a new manifest.
    """
    directory = tempfile.mkdtemp(prefix=f"stage1_{key}_")
    path = os.path.join(directory, f"{key}.csv")
    frame[["image_id", "label"]].to_csv(path, index=False)
    return path


def _temp_manifest_dataset(frame: pd.DataFrame, key: str, factory) -> Dataset:
    """Build a dataset from a temporary `image_id,label` view, then remove the view.

    Loaders read their manifest only in `__init__`, so the temporary file is not
    needed afterwards; the frozen manifest itself stays untouched.
    """
    path = _write_split_csv(frame, key)
    try:
        return factory(path)
    finally:
        shutil.rmtree(os.path.dirname(path), ignore_errors=True)


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
    # Extra ID-aligned columns for the added rows (empty for the author's modes).
    label_any = None
    extra_columns = {}
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
    elif validation_mode == "vindr_test":
        # E2 (added row): official VinDr-CXR test split; IDs/labels from the frozen
        # manifest produced on B. Author VinDataset preprocessing is reused as-is.
        frame = load_manifest("vindr_test.csv")
        dataset = _temp_manifest_dataset(
            frame,
            "vindr_test",
            lambda path: ManifestVinDataset(
                manifest_csv=path,
                root_dir=str(cfg.ptx["vindr"]["test_root_dir"]),
                img_size=img_size,
                stats_key="vindr_test",
            ),
        )
        ids = list(dataset.image_ids)
        sources = ["vindr_test"] * len(ids)
        expected = [int(dataset.labels[i]) for i in ids]
    elif validation_mode == "vindr_wide":
        # E2-wide (added row): all 18,000 published VinDr images; train and test DICOMs
        # live in different roots. Main label = majority (>=2/3 readers) on the train
        # part, consensus on the test part; `label_any` = >=1/3 readers (author's E5
        # positive rule) is reported as a sensitivity analysis from the SAME scores.
        frame = load_manifest("vindr_wide.csv")
        if "label_any" not in frame.columns or "split" not in frame.columns:
            raise RuntimeError("vindr_wide.csv must hold split and label_any columns")
        train_frame = frame[frame["split"] == "train"]
        test_frame = frame[frame["split"] == "test"]
        vin_ds = _temp_manifest_dataset(
            train_frame,
            "wide_train",
            lambda path: ManifestVinDataset(
                manifest_csv=path,
                root_dir=str(cfg.ptx["vindr"]["root_dir"]),
                img_size=img_size,
                stats_key="vindr_wide_train",
            ),
        )
        ptx_ds = _temp_manifest_dataset(
            test_frame,
            "wide_test",
            lambda path: ManifestVinDataset(
                manifest_csv=path,
                root_dir=str(cfg.ptx["vindr"]["test_root_dir"]),
                img_size=img_size,
                stats_key="vindr_wide_test",
            ),
        )
        dataset = ConcatDataset([vin_ds, ptx_ds])
        ids = list(vin_ds.image_ids) + list(ptx_ds.image_ids)
        sources = ["vindr_wide_train"] * len(vin_ds.image_ids) + [
            "vindr_wide_test"
        ] * len(ptx_ds.image_ids)
        expected = [int(vin_ds.labels[i]) for i in vin_ds.image_ids] + [
            int(ptx_ds.labels[i]) for i in ptx_ds.image_ids
        ]
        label_by_id = dict(
            zip(ids, [int(v) for v in frame.set_index("image_id").loc[ids, "label_any"]])
        )
        label_any = [label_by_id[image_id] for image_id in ids]
        extra_columns = {
            "split": [str(value) for value in frame.set_index("image_id").loc[ids, "split"]],
            "label_any": label_any,
        }
    elif validation_mode == "chexlocalize":
        # E4 (added row): CheXlocalize frontal cohort, author preprocessing chain.
        frame = load_manifest("chexlocalize_cls.csv")
        dataset = _temp_manifest_dataset(
            frame,
            "chexlocalize_cls",
            lambda path: CheXlocalizeEvalDataset(manifest_csv=path, img_size=img_size),
        )
        ids = list(dataset.image_ids)
        sources = [f"chexlocalize_{dataset.splits[i]}" for i in ids]
        expected = [int(dataset.labels[i]) for i in ids]
        extra_columns = {"split": [str(dataset.splits[i]) for i in ids]}
    else:
        raise ValueError(
            "Invalid validation mode: Choose one of holdout, ood, vindr_test, "
            "vindr_wide, chexlocalize"
        )
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
    if label_any is not None:
        # E2-wide sensitivity analysis: the SAME fold-averaged scores, second label
        # rule (>=1/3 readers). No threshold, no re-run, no separate model.
        label_any_array = np.asarray(label_any).astype(int)
        if len(np.unique(label_any_array)) < 2:
            raise RuntimeError("label_any cohort is single-class; AUROC undefined")
        ensemble_metrics["sensitivity_label_any"] = {
            "auc": float(roc_auc_score(label_any_array, all_logits_ensemble)),
            "n_pos": int(label_any_array.sum()),
            "n_neg": int(len(label_any_array) - label_any_array.sum()),
        }
    # Save predictions to avoid re-running for stat analysis
    os.makedirs(f"{HOME}/output/classif_preds/", exist_ok=True)
    np.save(
        f"{HOME}/output/classif_preds/{validation_mode}pred_{model_name}_{img_size}.npy",
        all_logits_ensemble,
    )
    np.save(f"{HOME}/output/classif_preds/{validation_mode}gt.npy", all_labels)
    # Stage 1: ID-keyed predictions (score = fold-mean sigmoid probability, as in eval_res)
    predictions = {
        "image_id": ids,
        "source": sources,
        "label": np.asarray(all_labels).astype(int),
        "score": all_logits_ensemble,
    }
    predictions.update(extra_columns)
    pd.DataFrame(predictions).to_csv(
        f"{HOME}/output/classif_preds/{validation_mode}pred_{model_name}_{img_size}_ids.csv",
        index=False,
    )
    return ensemble_metrics


@click.command()
@click.option(
    "--validation_mode",
    type=click.Choice(
        ["holdout", "ood", "vindr_test", "vindr_wide", "chexlocalize"],
        case_sensitive=True,
    ),
    required=True,
    help=(
        "Choose validation dataset: holdout, ood (author rows); vindr_test, "
        "vindr_wide, chexlocalize (added external rows, frozen manifests from B)"
    ),
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
        for img_size in [512]:  # Stage 1 run scope: run A is 512 only (protocol §6); author list 64..1024
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

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
# B (§4.4 E2 / E4): current -> the evaluator knew only the internal holdout and the two
#   E5 VinDr+PTX-498 runs, so the protocol's external rows had no prediction file.
#   New -> `vindr_test` evaluates the VinDr official test manifest (E2) and
#   `chexlocalize` evaluates the CheXlocalize classification manifest (E4); both keep
#   the ID-keyed CSV and the run's own tau, `source` is `vindr_test` /
#   `chexlocalize_<split>`, and the E4 CSV additionally keeps `split` and
#   `patient_id`. E2 needs no `--e5_manifest` (its ids come from the frozen E2
#   manifest).
#   check: tests/corrected/test_e2_e4_eval.py::test_vindr_test_mode_writes_id_csv,
#   ::test_chexlocalize_mode_keeps_split_and_patient_id,
#   ::test_cli_help_lists_e2_e4_modes

Usage:
    Run this module directly:
        python3 evaluate_kfold.py --validation_mode=<validation_dataset>
        python3 evaluate_kfold.py --validation_mode=ood_clean --e5_manifest=manifests/reference_like_clean.csv
        python3 evaluate_kfold.py --validation_mode=vindr_test   # E2
        python3 evaluate_kfold.py --validation_mode=chexlocalize  # E4
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
from sharpness_matters.ptx.corrected.frozen import manifest_dir
from sharpness_matters.ptx.utils.classification_eval import eval_res
from sharpness_matters.ptx.utils.logging_utils import initialize_logger
from sharpness_matters.config.load_config import cfg

logger = initialize_logger(level="debug")
DATA_DIR = cfg.ptx.siim_acr.root_dir
HOME = Path(__file__).resolve().parent.parent
logger.debug(f"Output Root: {HOME}")

# Manifest names live in their builders (make_vindr_test / make_chexlocalize); the
# evaluator only reads the frozen csv, so the names are mirrored here (§4.4 E2/E4).
VINDR_TEST_MANIFEST = "vindr_test.csv"
CHEXLOCALIZE_CLS_MANIFEST = "chexlocalize_cls.csv"
VALIDATION_MODES = (
    "holdout",
    "ood_reference_like",
    "ood_clean",
    "vindr_test",
    "chexlocalize",
)


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


def _chexlocalize_cohort_columns(
    manifest_csv: Path, ids: list
) -> Dict[str, list]:
    """Align the frozen E4 classification manifest's split/patient_id to dataset order.

    The dataset keeps manifest order (`image_ids`), so the metadata columns are read
    straight from the same manifest and checked against that order; a reordered or
    duplicated cohort raises instead of mislabelling a prediction row.
    """
    manifest = pd.read_csv(
        manifest_csv, dtype={"image_id": str, "split": str, "patient_id": str}
    )
    missing = {"image_id", "split", "patient_id"} - set(manifest.columns)
    if missing:
        raise KeyError(f"{manifest_csv} is missing columns {sorted(missing)}")
    manifest_ids = [str(image_id) for image_id in manifest["image_id"]]
    if manifest_ids != list(ids):
        raise RuntimeError(
            f"CheXlocalize manifest order mismatch: {manifest_csv} lists ids that "
            "differ from the dataset's image_ids, so split/patient_id cannot be "
            "attached to the prediction rows"
        )
    return {
        "split": [str(split) for split in manifest["split"]],
        "patient_id": [str(patient_id) for patient_id in manifest["patient_id"]],
    }


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
        validation_mode (str): Validation dataset ('holdout', 'ood_reference_like',
            'ood_clean', 'vindr_test' or 'chexlocalize'); the E5 variants are separate
            runs with separate manifests, E2/E4 read their own frozen manifests.
        e5_manifest (str): ID+label manifest of the E5 run (required for the E5 modes;
            unused by E2/E4, whose cohorts come from their frozen manifests)

    Returns:
        Dict[str, Any]: Dictionary containing evaluation metrics
    """
    # Extra ID-keyed columns beyond `source`; the E4 CSV keeps split and patient_id.
    extra_columns: Dict[str, list] = {}
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
    elif validation_mode == "vindr_test":
        # B (§4.4 E2): official test split only; no PTX-498 images are mixed in, so the
        # positives and negatives come from the same validated VinDr cohort.
        manifest_csv = manifest_dir() / VINDR_TEST_MANIFEST
        dataset = VinDataset(
            manifest_csv=str(manifest_csv),
            root_dir=str(cfg.ptx.vindr.test_root_dir),
            img_size=img_size,
        )
        ids = list(dataset.image_ids)
        sources = ["vindr_test"] * len(ids)
        expected = [int(dataset.labels[i]) for i in ids]
    elif validation_mode == "chexlocalize":
        # B (§4.4 E4): complementary external cohort; split and patient_id are kept so
        # the report can count val/test/combined separately and watch clustering.
        from sharpness_matters.ptx.datasets.chexlocalize_dataset import (
            CheXlocalizeDataset,
        )

        manifest_csv = manifest_dir() / CHEXLOCALIZE_CLS_MANIFEST
        dataset = CheXlocalizeDataset(str(manifest_csv), img_size)
        ids = list(dataset.image_ids)
        expected = [int(dataset.labels[i]) for i in ids]
        cohort = _chexlocalize_cohort_columns(manifest_csv, ids)
        extra_columns = {
            "split": cohort["split"],
            "patient_id": cohort["patient_id"],
        }
        sources = [f"chexlocalize_{split}" for split in cohort["split"]]
    else:
        raise ValueError(
            "Invalid validation mode: Choose one of "
            + ", ".join(VALIDATION_MODES)
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
    for column, values in extra_columns.items():
        pred_df[column] = values
    pred_df.to_csv(
        out_dir / f"{validation_mode}pred_{model_name}_{img_size}_ids.csv",
        index=False,
    )
    return ensemble_metrics


@click.command()
@click.option(
    "--validation_mode",
    type=click.Choice(list(VALIDATION_MODES)),
    required=True,
    help="Choose validation dataset: holdout, ood_reference_like, ood_clean "
    "(internal holdout and the two E5 runs) or vindr_test, chexlocalize (E2, E4)",
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
        validation_mode (str): Validation dataset ('holdout', 'ood_reference_like',
            'ood_clean', 'vindr_test' or 'chexlocalize')
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

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
# B (§4.4 E2-wide, secondary row): current -> a single prediction file could only carry
#   E2's one consensus label. New -> `vindr_wide` evaluates the frozen train+test wide
#   manifest as one ConcatDataset (train root, test root) and writes both pre-registered
#   labels (`label`, `label_any`) with the prediction, so the sensitivity AUROC comes
#   from the same predictions under `sensitivity_label_any` instead of a second model
#   run. The per-split `image_id,label` csv files the loader needs are temporary views
#   of the frozen manifest and are never frozen themselves.
#   check: tests/corrected/test_e2_e4_eval.py::test_vindr_wide_mode_writes_both_labels,
#   ::test_vindr_wide_sensitivity_auroc_uses_same_predictions

# B (Section 4.7 / 6.1): every cohort below is read from an existing frozen artifact
#   (`frozen.load_frozen` checks csv + .sha256 + .record.json) and the loaded dataset is
#   compared with that frame; nothing is rebuilt. E5 additionally binds each mode to its
#   canonical manifest (`eval_manifests.load_e5_manifest`), so E5-clean passed to
#   ood_reference_like (or a renamed copy of it) is refused before inference.
#   check: tests/corrected/test_manifest_eval_integrity.py
#
# Usage:
    Run this module directly:
        python3 evaluate_kfold.py --validation_mode=<validation_dataset>
        python3 evaluate_kfold.py --validation_mode=ood_clean --e5_manifest=manifests/reference_like_clean.csv
        python3 evaluate_kfold.py --validation_mode=vindr_test   # E2
        python3 evaluate_kfold.py --validation_mode=vindr_wide   # E2-wide (secondary)
        python3 evaluate_kfold.py --validation_mode=chexlocalize  # E4
    Output CSV files with evaluation metrics for each model will be saved in
    'classification_{model_name}_{validation_mode}.csv'.
"""

import os
import glob
import shutil
import tempfile
import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import ConcatDataset, Dataset
import pandas as pd
import sys
import click
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from sklearn.metrics import roc_auc_score

from sharpness_matters.ptx.datasets.siim_acr_classification_dataset import (
    ChestXrayDataset,
)
from sharpness_matters.ptx.datasets.vin_dataset import VinDataset
from sharpness_matters.ptx.datasets.ptx498_dataset import OODTestDataset
from sharpness_matters.ptx.models.cnn import CNNBinaryClassifier
from sharpness_matters.ptx.corrected.frozen import (
    load_frozen,
    manifest_dir,
    sha256_file,
)
from sharpness_matters.ptx.corrected.eval_manifests import (
    load_e5_manifest,
    verify_dataset_against_frozen,
)
from sharpness_matters.ptx.corrected.make_chexlocalize import (
    CLS_MANIFEST_NAME as CHEXLOCALIZE_CLS_MANIFEST,
)
from sharpness_matters.ptx.corrected.make_vindr_test import (
    MANIFEST_NAME as VINDR_TEST_MANIFEST,
    WIDE_COLUMNS as VINDR_WIDE_COLUMNS,
    WIDE_MANIFEST_NAME as VINDR_WIDE_MANIFEST,
    WIDE_SPLITS,
)
from sharpness_matters.ptx.utils.classification_eval import (
    auroc_confidence_interval,
    eval_res,
)
from sharpness_matters.ptx.utils.logging_utils import initialize_logger
from sharpness_matters.config.load_config import cfg

logger = initialize_logger(level="debug")
DATA_DIR = cfg.ptx.siim_acr.root_dir
HOME = Path(__file__).resolve().parent.parent
logger.debug(f"Output Root: {HOME}")

# Manifest names come from their builders (single source of truth, §4.4 E2/E2-wide/E4).
VALIDATION_MODES = (
    "holdout",
    "ood_reference_like",
    "ood_clean",
    "vindr_test",
    "vindr_wide",
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
    manifest_csv: Path, ids: list, manifest_frame: Optional[pd.DataFrame] = None
) -> Dict[str, list]:
    """Align the frozen E4 classification manifest's split/patient_id to dataset order.

    The dataset keeps manifest order (`image_ids`), so the metadata columns are read
    straight from the same manifest and checked against that order; a reordered or
    duplicated cohort raises instead of mislabelling a prediction row. Evaluation
    passes the already hash-verified frame so the metadata cannot come from a second,
    different read of the path.
    """
    manifest = (
        manifest_frame
        if manifest_frame is not None
        else pd.read_csv(
            manifest_csv, dtype={"image_id": str, "split": str, "patient_id": str}
        )
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
            'ood_clean', 'vindr_test', 'vindr_wide' or 'chexlocalize'); the E5 variants
            are separate runs with separate manifests, E2/E2-wide/E4 read their own
            frozen manifests.
        e5_manifest (str): ID+label manifest of the E5 run (required for the E5 modes;
            unused by E2/E2-wide/E4, whose cohorts come from their frozen manifests)

    Returns:
        Dict[str, Any]: Dictionary containing evaluation metrics; `vindr_wide` adds the
            `sensitivity_label_any` AUROC block computed from the same predictions.
    """
    # Extra ID-keyed prediction columns as (insert position, name, values); position
    # None appends. The E2-wide row pins split/label_any between source and label.
    extra_columns: List[Tuple[Optional[int], str, list]] = []
    label_any_labels: Optional[np.ndarray] = None
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
        # B (Section 4.7 / 6.1): the mode owns one frozen cohort. The path must be
        # frozen and carry that cohort's content; the other mode's manifest (or a
        # renamed copy of it) is rejected here, before any dataset or checkpoint.
        e5_path, e5_frame = load_e5_manifest(
            validation_mode, e5_manifest, manifest_dir()
        )
        vin_ds = VinDataset(manifest_csv=str(e5_path), img_size=img_size)
        verify_dataset_against_frozen(
            vin_ds, e5_frame, what=f"{validation_mode} VinDr cohort"
        )
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
        frozen_test = load_frozen(manifest_csv)
        dataset = VinDataset(
            manifest_csv=str(manifest_csv),
            root_dir=str(cfg.ptx.vindr.test_root_dir),
            img_size=img_size,
        )
        verify_dataset_against_frozen(dataset, frozen_test, what="VinDr test cohort")
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
        frozen_cls = load_frozen(manifest_csv)
        dataset = CheXlocalizeDataset(str(manifest_csv), img_size)
        verify_dataset_against_frozen(
            dataset, frozen_cls, what="CheXlocalize classification cohort"
        )
        ids = list(dataset.image_ids)
        expected = [int(dataset.labels[i]) for i in ids]
        cohort = _chexlocalize_cohort_columns(
            manifest_csv, ids, manifest_frame=frozen_cls
        )
        extra_columns = [
            (None, "split", cohort["split"]),
            (None, "patient_id", cohort["patient_id"]),
        ]
        sources = [f"chexlocalize_{split}" for split in cohort["split"]]
    elif validation_mode == "vindr_wide":
        # B (§4.4 E2-wide): secondary row next to E2. One frozen wide manifest carries
        # both pre-registered labels; VinDataset needs one ID+label csv per root, so the
        # per-split views are written to a temporary dir (never frozen separately) and
        # removed once the datasets have parsed them.
        manifest_csv = manifest_dir() / VINDR_WIDE_MANIFEST
        wide = load_frozen(manifest_csv)
        missing_wide = set(VINDR_WIDE_COLUMNS) - set(wide.columns)
        if missing_wide:
            raise KeyError(
                f"{manifest_csv} is missing columns {sorted(missing_wide)}; the "
                "frozen E2-wide manifest must carry both labels and split"
            )
        split_dir = Path(
            tempfile.mkdtemp(prefix=f"vindr_wide_{sha256_file(manifest_csv)[:12]}_")
        )
        split_datasets = []
        ids = []
        sources = []
        expected = []
        split_values = []
        label_any_values = []
        for split in WIDE_SPLITS:
            part = wide.loc[wide["split"].astype(str) == split]
            if part.empty:
                raise RuntimeError(
                    f"Frozen {VINDR_WIDE_MANIFEST} has no {split} rows; the wide "
                    "cohort needs both splits"
                )
            split_csv = split_dir / f"{split}.csv"
            part[["image_id", "label"]].to_csv(split_csv, index=False)
            root = (
                cfg.ptx.vindr.root_dir
                if split == "train"
                else cfg.ptx.vindr.test_root_dir
            )
            split_ds = VinDataset(
                manifest_csv=str(split_csv), root_dir=str(root), img_size=img_size
            )
            split_ids = [str(image_id) for image_id in split_ds.image_ids]
            if split_ids != part["image_id"].astype(str).tolist():
                raise RuntimeError(
                    f"{split}: VinDataset order differs from the frozen wide manifest"
                )
            frozen_split_labels = {
                str(image_id): int(label)
                for image_id, label in zip(part["image_id"], part["label"])
            }
            split_mismatched = [
                image_id
                for image_id in split_ids
                if int(split_ds.labels[image_id]) != frozen_split_labels[image_id]
            ]
            if split_mismatched:
                raise RuntimeError(
                    f"{split}: VinDataset labels differ from the frozen wide manifest "
                    f"for {len(split_mismatched)} ids, e.g. {split_mismatched[:5]}"
                )
            split_datasets.append(split_ds)
            ids.extend(split_ids)
            sources.extend([f"vindr_wide_{split}"] * len(split_ids))
            expected.extend(int(split_ds.labels[image_id]) for image_id in split_ids)
            split_values.extend([split] * len(split_ids))
            label_any_values.extend(int(value) for value in part["label_any"])
        dataset = ConcatDataset(split_datasets)
        # VinDataset parses its manifest in __init__; the temporary views can go now.
        shutil.rmtree(split_dir, ignore_errors=True)
        label_any_labels = np.asarray(label_any_values, dtype=int)
        extra_columns = [
            (2, "split", split_values),
            (4, "label_any", label_any_values),
        ]
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
    for position, column, values in extra_columns:
        if position is None:
            pred_df[column] = values
        else:
            pred_df.insert(position, column, values)
    pred_df.to_csv(
        out_dir / f"{validation_mode}pred_{model_name}_{img_size}_ids.csv",
        index=False,
    )
    if label_any_labels is not None:
        # §4.4 E2-wide: the sensitivity row reuses the SAME ensemble predictions; only
        # the label definition changes (>= 1 reader vote), so the model is not re-run.
        if len(np.unique(label_any_labels)) < 2:
            raise RuntimeError(
                "sensitivity_label_any AUROC is undefined: the E2-wide cohort has one "
                "class under the any-vote rule"
            )
        any_probs = pred_df["prob"].to_numpy().astype(np.float64)
        ci_low, ci_high = auroc_confidence_interval(label_any_labels, any_probs)
        ensemble_metrics["sensitivity_label_any"] = {
            "auc": float(np.round(roc_auc_score(label_any_labels, any_probs), 4)),
            "auroc_ci": [
                float(np.round(ci_low, 4)),
                float(np.round(ci_high, 4)),
            ],
            "n_pos": int((label_any_labels == 1).sum()),
            "n_neg": int((label_any_labels == 0).sum()),
        }
    return ensemble_metrics


@click.command()
@click.option(
    "--validation_mode",
    type=click.Choice(list(VALIDATION_MODES)),
    required=True,
    help="Choose validation dataset: holdout, ood_reference_like, ood_clean "
    "(internal holdout and the two E5 runs) or vindr_test, vindr_wide, chexlocalize "
    "(E2, E2-wide, E4)",
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

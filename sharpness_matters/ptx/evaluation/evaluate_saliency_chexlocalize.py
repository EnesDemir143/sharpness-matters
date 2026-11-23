"""CheXlocalize E4 XAI saliency evaluation entry point.

# B (protocol Section 4.4 E4): current -> the author's saliency runner had SIIM and
#   PTX-498 entry points only; problem -> E4's complementary external XAI row could
#   not be produced; new -> the frozen `manifests/chexlocalize_xai.csv` cohort is
#   scored with `evaluate_saliency.run_experiment` under dataset_name
#   `chexlocalize`, input size 512 only, reusing the same fold checkpoints, chunk
#   cache, K1 / TP-conditional / joint summaries and hit/pixel-AUPRC columns.
#   check: tests/corrected/test_chexlocalize.py::test_saliency_*

Usage:
    python -m sharpness_matters.ptx.evaluation.evaluate_saliency_chexlocalize
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import pandas as pd
import pytorch_lightning as pl
import torch

from sharpness_matters.config.load_config import cfg
from sharpness_matters.ptx.corrected.frozen import manifest_dir
from sharpness_matters.ptx.datasets.chexlocalize_dataset import CheXlocalizeDataset
from sharpness_matters.ptx.evaluation.evaluate_saliency import run_experiment
from sharpness_matters.ptx.utils.logging_utils import initialize_logger
from sharpness_matters.ptx.utils.saliency_eval import (
    RULE_LEGACY,
    RULE_MAIN,
    summary_table,
)

logger = initialize_logger(level="debug")
HOME = Path(__file__).resolve().parent.parent

DATASET_NAME = "chexlocalize"
XAI_MANIFEST_NAME = "chexlocalize_xai.csv"
IMG_SIZE = 512


def build_dataset(
    img_size: int = IMG_SIZE,
    annotations_dir: Optional[str] = None,
    chexpert_dir: Optional[str] = None,
) -> CheXlocalizeDataset:
    """The frozen XAI cohort with native-grid masks, at one input size."""
    manifest_csv = manifest_dir() / XAI_MANIFEST_NAME
    return CheXlocalizeDataset(
        str(manifest_csv),
        int(img_size),
        with_mask=True,
        annotations_dir=annotations_dir
        if annotations_dir is not None
        else cfg.ptx.chexlocalize.annotations_dir,
        chexpert_dir=chexpert_dir
        if chexpert_dir is not None
        else cfg.ptx.chexlocalize.chexpert_dir,
    )


def run_chexlocalize_saliency(
    model_names: Sequence[str] = ("resnet", "densenet"),
    img_size: int = IMG_SIZE,
    chunk_size: int = 20,
    save_maps: bool = True,
    output_root: Optional[Union[str, Path]] = None,
    dataset: Optional[CheXlocalizeDataset] = None,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Run the main/legacy saliency rules for every model on the E4 XAI cohort."""
    if int(img_size) != IMG_SIZE:
        raise ValueError(
            f"CheXlocalize XAI is evaluated at {IMG_SIZE} only (protocol Section 6); "
            f"got img_size={img_size}"
        )
    cohort = dataset if dataset is not None else build_dataset(img_size=img_size)
    output_root = Path(
        output_root if output_root is not None else f"{HOME}/output/saliency_eval"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    results_by_model: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for model_name in model_names:
        torch.cuda.empty_cache()
        logger.info(
            f"Running: dataset={DATASET_NAME}, image size={img_size}, model={model_name}"
        )
        results = run_experiment(
            cohort,
            int(img_size),
            model_name,
            DATASET_NAME,
            validation_mode="ood",
            chunk_size=chunk_size,
            save_maps=save_maps,
            output_dir=str(output_root / model_name),
        )
        results_by_model[model_name] = results
        table_rows = {RULE_MAIN: [], RULE_LEGACY: []}
        for rule in (RULE_MAIN, RULE_LEGACY):
            table_rows[rule].extend(summary_table(results[rule]))
        for rule in (RULE_MAIN, RULE_LEGACY):
            table_path = (
                output_root
                / f"saliency_thresholded_{DATASET_NAME}_{model_name}_{rule}.csv"
            )
            pd.DataFrame(table_rows[rule]).to_csv(table_path, index=False)
            logger.debug(f"Saved {table_path}")
    return results_by_model


if __name__ == "__main__":
    seed = 42
    pl.seed_everything(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    run_chexlocalize_saliency()

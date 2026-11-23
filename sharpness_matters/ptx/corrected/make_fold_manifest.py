"""SIIM development-split fold manifest (PTX-16, stratification).

# B (PTX-16 / Section 4.1): train_kfold.py used plain KFold(shuffle=True) -> folds
#   not stratified. New: StratifiedKFold(k, shuffle=True, random_state=seed) over the
#   author's development split (ChestXrayDataset split="train", same
#   train_test_split), written once to manifests/siim_folds.csv with image_id,label,
#   fold; training reads the manifest instead of re-splitting.
#   check: tests/corrected/test_data.py::test_fold_manifest_*
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from sharpness_matters.config.load_config import cfg
from sharpness_matters.ptx.corrected.frozen import (
    hash_path,
    load_or_create_frozen,
    manifest_dir,
    record_path,
)

FOLD_MANIFEST_NAME = "siim_folds.csv"


def build_fold_manifest(
    image_ids: Sequence[str], labels: Sequence[int]
) -> pd.DataFrame:
    """Return ``image_id,label,fold`` with stratified folds, in input order.

    ``fold`` runs 0..k-1 where k = cfg.ptx.corrected.k_folds; the split is
    ``StratifiedKFold(shuffle=True, random_state=cfg.ptx.corrected.seed)``.
    Every development image appears exactly once.
    """
    image_ids = list(image_ids)
    labels = [int(value) for value in labels]
    if len(image_ids) != len(labels):
        raise ValueError(
            f"image_ids and labels differ in length: {len(image_ids)} vs {len(labels)}"
        )
    if len(image_ids) == 0:
        raise ValueError("Cannot build a fold manifest for an empty image list")
    if len(set(image_ids)) != len(image_ids):
        raise ValueError("image_ids must be unique to define disjoint folds")

    k_folds = int(cfg.ptx.corrected.k_folds)
    seed = int(cfg.ptx.corrected.seed)
    fold_of = np.full(len(image_ids), -1, dtype=int)
    splitter = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
    dummy_x = np.zeros(len(image_ids), dtype=np.int8)
    for fold, (_, val_idx) in enumerate(splitter.split(dummy_x, labels)):
        fold_of[val_idx] = fold
    if (fold_of < 0).any():  # pragma: no cover - StratifiedKFold guarantees coverage
        raise RuntimeError("StratifiedKFold left images without a fold")
    return pd.DataFrame(
        {"image_id": image_ids, "label": labels, "fold": fold_of.astype(int)}
    )


def _author_development_split() -> Tuple[list, list]:
    """Image IDs and labels of the author's development split, or raise.

    Mirrors the construction in the author's training script: all DICOMs under
    ``dicom-images-train`` with ``split="train"`` and the author's split seed.
    """
    data_dir = cfg.ptx.siim_acr.root_dir
    train_glob = os.path.join(str(data_dir), "dicom-images-train", "*", "*", "*.dcm")
    file_paths = sorted(glob.glob(train_glob))
    label_csv = os.path.join(str(data_dir), "train-rle.csv")
    if not file_paths or not os.path.exists(label_csv):
        raise RuntimeError(
            "Fold manifest is absent and the SIIM development data is not available: "
            f"found {len(file_paths)} DICOM files and label csv exists="
            f"{os.path.exists(label_csv)} under {data_dir}. Place the data first; a "
            "fold manifest is never built from a partial split."
        )
    from sharpness_matters.ptx.datasets.siim_acr_classification_dataset import (
        ChestXrayDataset,
    )

    dataset = ChestXrayDataset(file_paths, label_csv, split="train")
    return list(dataset.image_ids), [int(dataset.labels[i]) for i in dataset.image_ids]


def load_fold_manifest() -> pd.DataFrame:
    """Load ``manifests/siim_folds.csv``; build and freeze it once if absent."""
    path = manifest_dir() / FOLD_MANIFEST_NAME
    frozen_state = any(
        p.exists() for p in (path, hash_path(path), record_path(path))
    )

    if frozen_state:
        def _build():  # pragma: no cover - only reached for a broken frozen state
            raise RuntimeError(f"Frozen fold manifest {path} is incomplete")

        record = {}
    else:
        image_ids, labels = _author_development_split()
        record = {
            "item": "PTX-16 (Section 4.1) SIIM stratified fold manifest",
            "source_split": "ChestXrayDataset(split='train'), author train_test_split",
            "k_folds": int(cfg.ptx.corrected.k_folds),
            "seed": int(cfg.ptx.corrected.seed),
            "n_images": len(image_ids),
            "n_positive": int(sum(labels)),
        }

        def _build():
            return build_fold_manifest(image_ids, labels)

    frame = load_or_create_frozen(path, _build, record)
    frame["label"] = frame["label"].astype(int)
    frame["fold"] = frame["fold"].astype(int)
    return frame


if __name__ == "__main__":
    # Build-and-freeze (first run) or load-and-verify (later runs) manifests/siim_folds.csv.
    folds = load_fold_manifest()
    print(
        f"siim_folds.csv: {len(folds)} images, {int(folds['label'].sum())} positive; "
        f"per fold (n, positives): "
        f"{folds.groupby('fold')['label'].agg(['size', 'sum']).values.tolist()}"
    )

"""Stage-2 (B) k-fold training for the PTX baseline.

# B (§5.1 / PTX-06): train and validation shared one dataset and one random transform,
#   so flip/rotation also ran at validation time -> two instances from the same split
#   arguments (augmented train, `transform=None` eval).
# B (§5.2 / PTX-16): `pos_weight` came from the whole development split -> now counted
#   on `labels[train_idx]` only.
# B (§4.1 / PTX-16): plain `KFold(shuffle=True, random_state=seed)` -> frozen
#   `manifests/siim_folds.csv`, validated against the dataset.
# B (§5.5-§5.8 / PTX-07): `last-checkpoint` with no monitor and mean `val/loss`/`val/acc`
#   only -> best checkpoint on `val/loss`, epoch-end validation with epoch patience,
#   per-fold AUROC from the restored checkpoint, pooled OOF AUROC and OOF threshold.
#   check: tests/corrected/test_train.py
"""

import numpy as np
import os
import glob
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import click
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset, Dataset

from sharpness_matters.ptx.corrected.cv_record import (
    FOLD_MANIFEST_PROVENANCE_VERSION,
    file_sha256,
    run_dir,
    write_cv_record,
)
from sharpness_matters.ptx.corrected.threshold import oof_threshold
from sharpness_matters.ptx.datasets.siim_acr_classification_dataset import (
    ChestXrayDataset,
)
from sharpness_matters.ptx.models.cnn import CNNBinaryClassifier
from sharpness_matters.ptx.utils.logging_utils import initialize_logger
from sharpness_matters.config.load_config import cfg

HOME = Path(__file__).resolve().parent.parent
DATA_DIR = cfg.ptx.siim_acr.root_dir
logger = initialize_logger(level="debug")


def _load_fold_manifest() -> pd.DataFrame:
    """B (§4.1): fold membership comes from the frozen manifest (contract 3)."""
    from sharpness_matters.ptx.corrected.make_fold_manifest import (
        load_fold_manifest,
    )

    return load_fold_manifest()


def _fold_manifest_path() -> Path:
    """Hashable location of the frozen fold manifest (contract 1/3)."""
    from sharpness_matters.ptx.corrected.frozen import manifest_dir

    return Path(manifest_dir()) / "siim_folds.csv"


def _train_transform(img_size: int):
    """B (§5.3): the author's augmentation on the shared resize policy (contract 2)."""
    from sharpness_matters.ptx.corrected.preprocess import train_transform

    return train_transform(img_size)


def _default_model_factory(
    pos_weight: float, lr: float, img_size: int, model_name: str
) -> pl.LightningModule:
    """The author's model selection (`train_kfold.py` L68-91), unchanged."""
    if "densenet" in model_name:
        return CNNBinaryClassifier(
            lr=lr,
            img_size=img_size,
            model_name="densenet121",
            pos_weight=pos_weight,
        )
    if "resnet" in model_name:
        return CNNBinaryClassifier(
            lr=lr,
            img_size=img_size,
            model_name="resnet152.a2_in1k",
            pos_weight=pos_weight,
        )
    raise ValueError(
        f"Unsupported model_name '{model_name}'. Expected one of ['densenet', 'resnet']."
    )


def _image_ids(dataset: Dataset) -> List[str]:
    return [str(image_id) for image_id in dataset.image_ids]


def _labels_by_id(dataset: Dataset) -> Dict[str, int]:
    return {str(image_id): int(label) for image_id, label in dataset.labels.items()}


def _check_same_development_split(train_dataset: Dataset, eval_dataset: Dataset) -> None:
    """B (§5.1): the train and eval instances must carry the same split in the same order.

    The two `ChestXrayDataset` instances are built with identical split arguments and
    differ only in `transform`; any drift would silently misalign fold indices.
    """
    train_ids = _image_ids(train_dataset)
    eval_ids = _image_ids(eval_dataset)
    if train_ids != eval_ids:
        raise RuntimeError(
            "train and eval datasets do not expose the same development split: "
            f"{len(set(train_ids) ^ set(eval_ids))} image ids differ "
            f"({len(train_ids)} vs {len(eval_ids)} images)"
        )


def _fold_of_image(
    fold_manifest: pd.DataFrame,
    image_ids: List[str],
    labels: Dict[str, int],
    k_folds: int,
) -> Dict[str, int]:
    """Map image_id -> fold from the frozen manifest, validated against the dataset.

    B (§4.1 / PTX-16): current -> `KFold(shuffle=True, random_state=seed)` over the
    dataset; problem -> fold membership is not persisted and cannot be audited;
    new behaviour -> the manifest on disk defines membership and must cover exactly the
    development split, once, with the dataset's labels.
    """
    missing_columns = [c for c in ("image_id", "label", "fold") if c not in fold_manifest.columns]
    if missing_columns:
        raise RuntimeError(
            f"fold manifest is missing columns {missing_columns}; expected "
            "'image_id,label,fold'"
        )

    manifest_ids = [str(value) for value in fold_manifest["image_id"].tolist()]
    duplicated = pd.Series(manifest_ids).duplicated()
    if bool(duplicated.any()):
        raise RuntimeError(
            "fold manifest lists an image_id more than once: "
            f"{sorted(set(pd.Series(manifest_ids)[duplicated]))[:5]}"
        )
    if set(manifest_ids) != set(image_ids):
        raise RuntimeError(
            "fold manifest image ids differ from the dataset: "
            f"manifest {len(manifest_ids)} ids, dataset {len(image_ids)} ids, "
            f"{len(set(manifest_ids) ^ set(image_ids))} ids differ"
        )

    folds = [int(value) for value in fold_manifest["fold"].tolist()]
    outside = sorted({fold for fold in folds if not 0 <= fold < k_folds})
    if outside:
        raise RuntimeError(
            f"fold manifest contains folds {outside} outside 0..{k_folds - 1}"
        )

    manifest_labels = {
        image_id: int(label)
        for image_id, label in zip(manifest_ids, fold_manifest["label"].tolist())
    }
    mismatched = [
        image_id
        for image_id in image_ids
        if manifest_labels[image_id] != labels[image_id]
    ]
    if mismatched:
        raise RuntimeError(
            f"fold manifest labels disagree with the dataset for {len(mismatched)} ids, "
            f"e.g. {mismatched[:5]}"
        )

    return dict(zip(manifest_ids, folds))


def _manifest_index(frame: pd.DataFrame, source: str) -> Dict[str, Tuple[int, int]]:
    """Unique image_id -> (label, fold) for a fold manifest frame, or raise."""
    missing = [c for c in ("image_id", "label", "fold") if c not in frame.columns]
    if missing:
        raise RuntimeError(
            f"{source} is missing columns {missing}; expected 'image_id,label,fold'"
        )
    index: Dict[str, Tuple[int, int]] = {}
    for image_id, label, fold in zip(
        frame["image_id"], frame["label"], frame["fold"]
    ):
        key = str(image_id)
        if key in index:
            raise RuntimeError(f"{source} lists an image_id more than once: {key}")
        try:
            index[key] = (int(label), int(fold))
        except (TypeError, ValueError) as error:
            raise RuntimeError(
                f"{source} has a non-integer label/fold for {key}: "
                f"{label!r}, {fold!r}"
            ) from error
    return index


def _check_fold_manifest_matches_file(
    fold_manifest: pd.DataFrame, manifest_path: Path
) -> None:
    """Prove the frame used for training is the file whose SHA-256 is recorded.

    B (§4.1 / §5.5-§5.8): the cv record binds ``fold_manifest_sha256`` from the file,
    so training must not accept an independently supplied frame. A caller could bind
    a disjoint file while training on different (e.g. holdout) IDs; comparing the
    image_id -> (label, fold) maps closes that gap. Row order and extra columns are
    irrelevant.
    """
    on_disk = pd.read_csv(manifest_path, dtype={"image_id": str})
    recorded = _manifest_index(on_disk, f"fold manifest file {manifest_path}")
    supplied = _manifest_index(fold_manifest, "in-memory fold manifest")
    if supplied == recorded:
        return
    only_supplied = sorted(set(supplied) - set(recorded))
    only_file = sorted(set(recorded) - set(supplied))
    changed = sorted(
        image_id
        for image_id in set(supplied) & set(recorded)
        if supplied[image_id] != recorded[image_id]
    )
    raise RuntimeError(
        "fold manifest provenance mismatch: the frame passed to training does not "
        f"match {manifest_path} (whose sha256 is written to the cv record); "
        f"ids only in frame {only_supplied[:5]}, ids only in file {only_file[:5]}, "
        f"changed label/fold {changed[:5]}. Refusing to train on a frame that is not "
        "the recorded manifest."
    )


def _fold_indices(fold_of: Dict[str, int], image_ids: List[str], fold: int):
    """Dataset-order indices of one fold's validation and training members."""
    val_idx = [idx for idx, image_id in enumerate(image_ids) if fold_of[image_id] == fold]
    train_idx = [idx for idx, image_id in enumerate(image_ids) if fold_of[image_id] != fold]
    if not val_idx:
        raise RuntimeError(f"fold {fold} has no validation image in the manifest")
    if not train_idx:
        raise RuntimeError(f"fold {fold} has no training image in the manifest")
    return train_idx, val_idx


def _predict_val_fold(
    model: pl.LightningModule, val_loader: DataLoader, device: torch.device
) -> Tuple[List[str], np.ndarray]:
    """Sigmoid probabilities of the restored best checkpoint on one validation fold.

    B (§5.7/§5.8, PTX-07): current -> the author validates `val/loss`/`val/acc` and keeps
    no per-image probability; problem -> fold AUROC, pooled OOF AUROC and the OOF
    threshold cannot be computed from the run; new behaviour -> probabilities are read
    from the restored best checkpoint in eval mode without gradients, keyed by ImageId.
    B (OOF device): `strategy.teardown()` moves the module to CPU when `fit` returns, so
    `next(model.parameters()).device` scored every fold on the CPU while training ran on
    the GPU (~0.84 s/image, minutes per fold, GPU idle) and produced probabilities from a
    different numeric path than the GPU evaluators. `device` is the trainer's root
    device and is required; the module is moved there for the pass and back to CPU after
    it, so GPU memory does not accumulate across folds.
    """
    device = torch.device(device)
    was_training = model.training
    model.to(device)
    model.eval()
    image_ids: List[str] = []
    probs: List[float] = []
    try:
        with torch.no_grad():
            for batch in val_loader:
                if "pid" not in batch:
                    raise RuntimeError(
                        "validation batch has no 'pid' key; OOF rows must be keyed by ImageId"
                    )
                logits = model(batch["data"].to(device))
                probs.extend(
                    torch.sigmoid(logits)
                    .reshape(-1)
                    .detach()
                    .cpu()
                    .numpy()
                    .tolist()
                )
                image_ids.extend([str(pid) for pid in batch["pid"]])
    finally:
        model.train(was_training)
        # B (OOF device): do not keep the fold's activations/weights on the GPU.
        model.to(torch.device("cpu"))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return image_ids, np.asarray(probs, dtype=np.float64)


def train_with_kfold(
    dataset: Dataset,
    eval_dataset: Optional[Dataset] = None,
    k_folds: Optional[int] = None,
    batch_size: Optional[int] = None,
    num_epochs: Optional[int] = None,
    lr: Optional[float] = None,
    img_size: int = 512,
    seed: Optional[int] = None,
    patience: Optional[int] = None,
    min_delta: Optional[float] = None,
    model_name: str = "densenet",
    num_workers: int = 2,
    fold_manifest: Optional[pd.DataFrame] = None,
    fold_manifest_path: Optional[Union[str, Path]] = None,
    model_factory: Optional[Callable[..., pl.LightningModule]] = None,
    trainer_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Cross-validate one model/resolution and write the CV record + OOF predictions.

    Every value that no correction item changes stays at the author's value, pinned in
    `cfg.ptx.corrected` (`baseline_correction.md` §11): passing `None` reads the config.

    Args:
        dataset: training split instance (its `transform` carries the augmentation).
        eval_dataset: the same split built with `transform=None`, so the validation
            subset is deterministic (5.1). Required.
        k_folds: fold count; the manifest must contain exactly these folds.
        batch_size / num_epochs / lr / seed / patience / min_delta: see config.
        img_size: input resolution; B runs 512 only (protocol §6).
        model_name: 'densenet' or 'resnet' (kept from the author's CLI).
        num_workers: DataLoader workers (author: 2; tests use 0).
        fold_manifest: frozen fold membership; defaults to `load_fold_manifest()`.
        fold_manifest_path: file hashed into `fold_manifest_sha256`; defaults to the
            frozen manifest location.
        model_factory: injectable constructor used once per fold; defaults to the
            author's `CNNBinaryClassifier` selection.
        trainer_kwargs: overrides merged over the author's `Trainer` arguments (tests
            use it for CPU and short runs).

    Returns:
        The CV record that was written to `run_dir(model_name, img_size)`.
    """
    corrected = cfg.ptx.corrected
    k_folds = int(corrected.k_folds if k_folds is None else k_folds)
    batch_size = int(corrected.batch_size if batch_size is None else batch_size)
    num_epochs = int(corrected.max_epochs if num_epochs is None else num_epochs)
    lr = float(corrected.lr if lr is None else lr)
    seed = int(corrected.seed if seed is None else seed)
    patience = int(corrected.patience if patience is None else patience)
    min_delta = float(corrected.min_delta if min_delta is None else min_delta)
    target_sensitivity = float(corrected.target_sensitivity)

    pl.seed_everything(seed)

    # B (§5.1, PTX-06): one dataset instance shared train and validation, so the random
    # flip/rotation also ran at validation time. Two instances now: augmented train,
    # deterministic eval.
    if eval_dataset is None:
        raise ValueError(
            "eval_dataset is required: validation must not reuse the augmented training "
            "dataset (5.1/PTX-06)"
        )
    _check_same_development_split(dataset, eval_dataset)

    image_ids = _image_ids(dataset)
    labels = _labels_by_id(dataset)
    if len(labels) != len(image_ids):
        raise RuntimeError(
            f"dataset exposes {len(image_ids)} image ids but {len(labels)} labels"
        )

    if fold_manifest is None:
        fold_manifest = _load_fold_manifest()
    manifest_path = Path(
        _fold_manifest_path() if fold_manifest_path is None else fold_manifest_path
    )
    if not manifest_path.is_file():
        raise RuntimeError(f"fold manifest file not found: {manifest_path}")
    fold_manifest_sha256 = file_sha256(manifest_path)
    _check_fold_manifest_matches_file(fold_manifest, manifest_path)

    fold_of = _fold_of_image(fold_manifest, image_ids, labels, k_folds)
    factory = _default_model_factory if model_factory is None else model_factory
    output_dir = run_dir(model_name, img_size)

    fold_records: List[Dict[str, Any]] = []
    oof_rows: List[Dict[str, Any]] = []

    for fold in range(k_folds):
        logger.info(f"\nTraining Fold {fold + 1}/{k_folds}...")
        train_idx, val_idx = _fold_indices(fold_of, image_ids, fold)

        # B (§5.2, PTX-16): the author weighed by the whole development split
        # (`train_subset.dataset.class_count`); the weight now counts only fold-train.
        train_labels = [labels[image_ids[idx]] for idx in train_idx]
        n_pos = int(sum(1 for label in train_labels if label == 1))
        n_neg = len(train_labels) - n_pos
        if n_pos == 0:
            raise RuntimeError(
                f"fold {fold} has no positive training image; pos_weight is undefined"
            )
        pos_weight = n_neg / n_pos

        train_loader = DataLoader(
            Subset(dataset, train_idx),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        val_loader = DataLoader(
            Subset(eval_dataset, val_idx),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        model = factory(
            pos_weight=pos_weight, lr=lr, img_size=img_size, model_name=model_name
        )

        # B (§5.5, PTX-07): the author saved `last-checkpoint` with no monitor, so the
        # restored weights were not the loss minimum. Monitor `val/loss` (min); ties keep
        # the first epoch because Lightning compares strictly (`torch.lt`).
        checkpoint_callback = ModelCheckpoint(
            dirpath=str(output_dir / "checkpoints" / f"fold_{fold}"),
            filename="best-{epoch}",
            monitor="val/loss",
            mode="min",
            save_top_k=1,
        )
        callbacks = [checkpoint_callback]
        # B (§5.6): epoch-end validation and patience in epochs, so `best_epoch` is an
        # integer epoch unit (`E_FD = ceil(mean(best_epoch))`). The author used
        # `val_check_interval=0.5` with patience counted in half-epoch checks.
        callbacks.append(
            EarlyStopping(
                monitor="val/loss",
                patience=patience,
                min_delta=min_delta,
                mode="min",
                verbose=True,
            )
        )

        trainer_config: Dict[str, Any] = dict(
            max_epochs=num_epochs,
            devices=[0],
            default_root_dir=str(output_dir),
            callbacks=callbacks,
            accelerator="auto",  # Automatically use GPU if available
            precision=32,
            log_every_n_steps=10,
            val_check_interval=1.0,
        )
        if trainer_kwargs:
            trainer_config.update(trainer_kwargs)
        trainer = Trainer(**trainer_config)

        trainer.fit(model, train_loader, val_loader)

        best_model_path = checkpoint_callback.best_model_path
        if not best_model_path:
            raise RuntimeError(
                f"fold {fold}: no checkpoint was written for monitor 'val/loss'"
            )
        checkpoint = torch.load(best_model_path, map_location="cpu")
        best_epoch = int(checkpoint["epoch"]) + 1
        # Lightning 2.1.1 reports the number of completed epochs here, so this is the
        # 1-based epoch at which fitting ended (early stop or max_epochs).
        stopped_epoch = int(trainer.current_epoch)
        if not 1 <= best_epoch <= stopped_epoch:
            raise RuntimeError(
                f"fold {fold}: best checkpoint epoch {best_epoch} is outside "
                f"1..{stopped_epoch} epochs actually trained"
            )

        # 5.7/5.8: score the fold with the restored best checkpoint, not the last epoch.
        # B (OOF device): `teardown()` already put the module back on CPU, so the OOF
        # pass is told the device `fit` ran on instead of reading the module's device.
        model.load_state_dict(checkpoint["state_dict"])
        pred_ids, probs = _predict_val_fold(
            model, val_loader, trainer.strategy.root_device
        )
        if len(pred_ids) != len(val_idx) or set(pred_ids) != {
            image_ids[idx] for idx in val_idx
        }:
            raise RuntimeError(
                f"fold {fold}: validation predictions do not match the fold members "
                f"({len(pred_ids)} predictions, {len(val_idx)} expected)"
            )

        y_val = np.asarray([labels[image_id] for image_id in pred_ids], dtype=int)
        try:
            fold_auroc = float(roc_auc_score(y_val, probs))
        except ValueError as error:
            raise RuntimeError(
                f"fold {fold}: AUROC is undefined for {len(val_idx)} validation images"
            ) from error

        fold_records.append(
            {
                "fold": fold,
                "best_epoch": best_epoch,
                "stopped_epoch": stopped_epoch,
                "checkpoint_path": str(best_model_path),
                "sha256": file_sha256(best_model_path),
                "val_auroc": fold_auroc,
                "n_train": len(train_idx),
                "n_val": len(val_idx),
                "pos_weight": float(pos_weight),
            }
        )
        oof_rows.extend(
            {
                "image_id": image_id,
                "label": labels[image_id],
                "fold": fold,
                "prob": float(prob),
            }
            for image_id, prob in zip(pred_ids, probs.tolist())
        )
        logger.info(
            f"Fold {fold + 1}: best_epoch={best_epoch} stopped_epoch={stopped_epoch} "
            f"pos_weight={pos_weight:.4f} AUROC={fold_auroc:.4f}"
        )

    # B (§5.8): every development image exactly once, otherwise the pooled OOF numbers
    # and the threshold describe a subset.
    oof = pd.DataFrame(oof_rows, columns=["image_id", "label", "fold", "prob"])
    if len(oof) != len(image_ids):
        raise RuntimeError(
            f"OOF has {len(oof)} rows for {len(image_ids)} development images"
        )
    if bool(oof["image_id"].duplicated().any()):
        duplicates = sorted(set(oof["image_id"][oof["image_id"].duplicated()]))[:5]
        raise RuntimeError(f"OOF lists an image_id more than once, e.g. {duplicates}")
    if set(oof["image_id"]) != set(image_ids):
        raise RuntimeError("OOF image ids differ from the development split")

    try:
        pooled_oof_auroc = float(roc_auc_score(oof["label"], oof["prob"]))
    except ValueError as error:
        raise RuntimeError(
            "pooled OOF AUROC is undefined: the OOF set has a single class"
        ) from error
    tau = oof_threshold(
        oof["label"].to_numpy(), oof["prob"].to_numpy(), target_sensitivity
    )

    record: Dict[str, Any] = {
        "model_name": model_name,
        "img_size": int(img_size),
        "k_folds": k_folds,
        "fold_manifest_sha256": fold_manifest_sha256,
        # Set only after `_check_fold_manifest_matches_file` proved the frame trained
        # on is the hashed file; readers require this version for holdout evaluation.
        "fold_manifest_provenance_version": FOLD_MANIFEST_PROVENANCE_VERSION,
        "folds": fold_records,
        "S_h": float(np.mean([entry["val_auroc"] for entry in fold_records])),
        "pooled_oof_auroc": pooled_oof_auroc,
        "tau": tau,
        "target_sensitivity": target_sensitivity,
    }
    record_path = write_cv_record(record, oof)
    logger.info(
        f"\nK-Fold Cross-Validation Results: S(h)={record['S_h']:.4f} "
        f"pooled OOF AUROC={pooled_oof_auroc:.4f} tau={tau:.6f} "
        f"target_sensitivity={target_sensitivity} -> {record_path}"
    )
    return record


@click.command()
@click.option(
    "--model_name",
    "-m",
    type=str,
    required=True,
    help="Choose between resnet or densenet",
)
def main(model_name):
    train_dicom_dir = os.path.join(DATA_DIR, "dicom-images-train")
    test_dicom_dir = os.path.join(DATA_DIR, "dicom-images-test")
    logger.debug(f"Test Len: {len(os.listdir(test_dicom_dir))}")
    logger.debug(f"Train Len: {len(os.listdir(train_dicom_dir))}")
    train_glob = f"{train_dicom_dir}/*/*/*.dcm"
    test_glob = f"{test_dicom_dir}/*/*/*.dcm"
    train_names = [f for f in sorted(glob.glob(train_glob))]
    # B (protocol §6): Stage 2 runs the dev split at 512 only; other resolutions were a
    # local shortcut in the author's loop.
    for img_size in [512]:
        train_transforms = _train_transform(img_size)
        # B (§5.1): same split arguments, two instances; only the training one augments.
        ds = ChestXrayDataset(
            train_names,
            f"{DATA_DIR}/train-rle.csv",
            img_size=img_size,
            transform=train_transforms,
        )
        eval_ds = ChestXrayDataset(
            train_names, f"{DATA_DIR}/train-rle.csv", img_size=img_size
        )
        train_with_kfold(
            ds,
            eval_ds,
            img_size=img_size,
            model_name=model_name,
        )


if __name__ == "__main__":
    main()

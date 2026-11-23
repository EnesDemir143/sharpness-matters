"""
Utility file for evaluation of classification performance
"""

import hashlib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from scipy.stats import norm
import scipy.stats as st
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Union, Type
from sharpness_matters.ptx.corrected.cv_record import fold_checkpoints, load_cv_record

# B (PTX-18 / PTX-07, Section 6.1): current -> the fold checkpoint was discovered with
#   `os.listdir(...)[0]` and a fold without a checkpoint was skipped (`continue`); a
#   run with 4/5 checkpoints silently passed as a 5-fold ensemble. Problem -> no fold
#   identity and no integrity check. New -> checkpoints and their SHA-256 come from the
#   run's `cv_record.json` (`load_cv_record`); a missing fold, a missing file or a hash
#   mismatch raises instead of shrinking the ensemble. Path + sha256 are still logged.
#   check: tests/corrected/test_classif_eval.py::test_missing_fold_checkpoint_raises,
#   ::test_sha256_mismatch_raises
# B (PTX-09 / Section 6.2): decision A -> fold outputs are averaged as sigmoid
#   probabilities exactly as the author, no logit averaging.
#   check: ::test_ensemble_is_mean_of_fold_probabilities
# B (Section 6.3): current -> fold and ensemble decisions used a fixed 0.5; new -> the
#   baseline's own OOF threshold `record["tau"]`, decision `p >= tau`, reported in the
#   metrics dict. check: ::test_threshold_uses_tau
# B (PTX-08 / Section 6.4): current -> `eval_res` returned bare arrays and the caller
#   wrote anonymous `.npy` files; new -> an ID-keyed DataFrame
#   (`image_id,label,prob,prob_fold0..k`) in dataset order, verified against the
#   dataset-order labels. check: ::test_output_id_order_matches_input,
#   ::test_label_misalignment_raises


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def auroc_confidence_interval(
    y_true: np.ndarray, y_scores: np.ndarray, confidence=0.95, n_bootstrap=2000
):
    """
    Compute the confidence interval for AUROC using bootstrapping.
    Args:
        y_true (np.ndarray): Ground truth binary labels (0 or 1).
        y_scores (np.ndarray): Predicted scores (probabilities or raw scores).
        confidence (float): Confidence level (default: 0.95).
        n_bootstrap (int): Number of bootstrap samples (default: 1000).
    Returns:
        (float, float): Lower and upper bounds of the AUROC confidence interval.
    """
    bootstrapped_aurocs = []
    n = len(y_true)
    # Perform bootstrapping
    for _ in range(n_bootstrap):
        indices = np.random.choice(np.arange(n), size=n, replace=True)
        if len(np.unique(y_true[indices])) < 2:
            continue  # Skip if there's only one class in the resampled data
        bootstrapped_aurocs.append(roc_auc_score(y_true[indices], y_scores[indices]))
    # Compute confidence interval from percentiles
    lower_percentile = (1 - confidence) / 2 * 100
    upper_percentile = (1 + confidence) / 2 * 100
    ci_lower = np.percentile(bootstrapped_aurocs, lower_percentile)
    ci_upper = np.percentile(bootstrapped_aurocs, upper_percentile)

    return ci_lower, ci_upper


def delong_roc_test(y_true: np.ndarray, scores1: np.ndarray, scores2: np.ndarray):
    """
    Perform DeLong's test to compare two AUROCs.
    Parameters:
    y_true: Ground truth binary labels
    scores1: Predicted scores from the first classifier
    scores2: Predicted scores from the second classifier
    Returns:
    z_stat: Z-statistic
    p_value: P-value for the significance test
    """
    # Compute AUROCs
    auc1 = roc_auc_score(y_true, scores1)
    auc2 = roc_auc_score(y_true, scores2)

    def compute_delong_covariance(scores, y_true):
        order = np.argsort(scores)[::-1]
        scores = scores[order]
        y_true = y_true[order]

        unique_scores = np.unique(scores)
        m, n = np.sum(y_true == 1), np.sum(y_true == 0)
        auc_partial_sums = np.zeros_like(unique_scores)

        for idx, score in enumerate(unique_scores):
            y_positive = np.sum(y_true[scores == score] == 1)
            auc_partial_sums[idx] = y_positive / (m * n)

        covariance = np.cov(auc_partial_sums)
        return covariance

    # Compute covariance matrices
    cov1 = compute_delong_covariance(scores1, y_true)
    cov2 = compute_delong_covariance(scores2, y_true)
    diff = auc1 - auc2
    var = cov1 + cov2
    z_stat = diff / np.sqrt(var)
    p_value = 2 * (1 - norm.cdf(abs(z_stat)))
    return z_stat, p_value


# Evaluation function for a single fold
def evaluate_fold(fold_number: int, val_loader: DataLoader, model: torch.nn.Module):
    """
    Evaluate a single cross-validation fold, collecting model sigmoid outputs and labels.
    Args:
        fold_number (int): Index of the current fold (for logging only).
        val_loader (DataLoader): Validation data loader yielding batches with keys "data" and "label".
        model (torch.nn.Module): Trained (or current) model returning raw logits of shape (B, 1) or (B,).
    Returns:
        dict: {
            "logits": np.ndarray of shape (N,), sigmoid probabilities for each sample,
            "labels": np.ndarray of shape (N,), ground-truth labels
        }
    """

    all_labels = []
    all_logits = []
    # Evaluate on validation data
    with torch.no_grad():
        for batch in tqdm(val_loader):
            pixel_values = batch["data"].to(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            labels = batch["label"].to("cuda" if torch.cuda.is_available() else "cpu")
            logits = model(pixel_values).squeeze(1)
            preds = torch.sigmoid(logits).cpu().numpy()
            all_labels.extend(labels.cpu().numpy())
            all_logits.append(preds)
    # Stack logits for this fold
    all_logits = np.concatenate(all_logits, axis=0)
    print(f"Fold {fold_number} - Collected {len(all_logits)} logits")
    return {"logits": all_logits, "labels": np.array(all_labels)}


def classification_collate(
    batch: List[Dict[str, torch.Tensor]]
) -> Dict[str, torch.Tensor]:
    """Collate only the classification inputs: `data` stacked, `label` as a tensor.

    # B (§6.6 / §7.1): current -> the evaluation loader used the default collate;
    #   problem -> the classification datasets carry a `mask` whose type and grid
    #   differ per source (VinDataset: [1,s,s] tensor, PTX498: native-grid numpy), so a
    #   batch crossing the Vin->PTX498 boundary of the ConcatDataset crashed on a key
    #   the classifier never reads; new -> every other key is ignored here.
    #   check: tests/corrected/test_classif_eval.py::test_collate_spans_mixed_masks
    """
    data = torch.stack([item["data"] for item in batch], dim=0)
    labels = torch.as_tensor([int(item["label"]) for item in batch], dtype=torch.long)
    return {"data": data, "label": labels}


def _dataset_order_labels(dataset: Dataset, ids: List[str]) -> Optional[np.ndarray]:
    """Dataset-order labels when the dataset exposes an `image_id -> label` dict.

    Used as a fallback when the caller does not pass `labels`; `ConcatDataset` is
    walked slice by slice so the ID order of the concatenation is preserved.
    """
    labels = getattr(dataset, "labels", None)
    if isinstance(labels, dict):
        if any(image_id not in labels for image_id in ids):
            return None
        return np.asarray([int(labels[image_id]) for image_id in ids])
    subsets = getattr(dataset, "datasets", None)
    if subsets is not None:
        parts = []
        start = 0
        for subset in subsets:
            stop = start + len(subset)
            part = _dataset_order_labels(subset, ids[start:stop])
            if part is None:
                return None
            parts.append(part)
            start = stop
        return np.concatenate(parts) if parts else None
    return None


def eval_res(
    dataset: Dataset,
    img_size: int,
    model_name: str,
    model_class: Type[torch.nn.Module],
    logger,
    ids: List[str],
    labels: Optional[np.ndarray] = None,
):
    """
    Evaluate a cross-validated classification model and aggregate ensemble metrics.
    Loads each fold's best checkpoint from the run's cv record, collects sigmoid
    probabilities and labels, computes per-fold (acc, auc, f1) and ensemble metrics by
    averaging sigmoid probabilities across folds, and returns ID-keyed predictions.
    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Validation dataset (same for all folds).
    img_size : int
        Image size suffix of the run whose cv record is read.
    model_name : str
        Base model name used in directory structure.
    model_class : Type[torch.nn.Module]
        LightningModule class with `load_from_checkpoint`.
    logger : logging.Logger
        Logger for debug output.
    ids : List[str]
        Image IDs in dataset order; defines the order of the returned frame.
    labels : Optional[np.ndarray]
        Expected labels in dataset order. When omitted, they are taken from the
        dataset's `labels` dict if it has one.
    Returns
    -------
    pred_df : pd.DataFrame
        Columns `image_id,label,prob,prob_fold0..prob_foldk-1` in dataset order;
        `prob` is the mean of the fold sigmoid probabilities.
    ensemble_metrics : dict
        Dict with keys: 'acc', 'auc', 'f1', 'auroc_ci', 'tau'.
    """
    ids = [str(image_id) for image_id in ids]
    if len(ids) != len(dataset):
        raise RuntimeError(
            f"ID/dataset length mismatch: {len(ids)} ids vs {len(dataset)} samples"
        )
    # Checkpoints and the operating threshold come from the run's cv record, which
    # verifies every fold is present and every recorded sha256 matches (Section 6.1).
    record = load_cv_record(model_name, img_size)
    checkpoints = fold_checkpoints(record)
    k_folds = len(checkpoints)
    if int(record["k_folds"]) != k_folds:
        raise RuntimeError(
            f"cv record covers {k_folds} folds but declares k_folds={record['k_folds']}"
        )
    # Section 6.3: the baseline's own OOF threshold, not a fixed 0.5.
    tau = float(record["tau"])
    expected_labels = (
        np.asarray(labels).astype(int).reshape(-1)
        if labels is not None
        else _dataset_order_labels(dataset, ids)
    )
    if expected_labels is not None and len(expected_labels) != len(ids):
        raise RuntimeError(
            f"label/id length mismatch: {len(expected_labels)} labels vs {len(ids)} ids"
        )

    val_loader = DataLoader(
        dataset,
        batch_size=32,
        num_workers=8,
        shuffle=False,
        collate_fn=classification_collate,
    )
    fold_probs = []
    fold_labels = []
    metrics = {"acc": [], "auc": [], "f1": []}
    print(f"Evaluating {model_name} model with image size {img_size}...")
    # Compute fold-wise metrics
    for fold, checkpoint_path in enumerate(checkpoints):
        print(f"\nEvaluating Fold {fold + 1}/{k_folds}...")
        # Evaluate this fold
        model = model_class.load_from_checkpoint(checkpoint_path)
        logger.info(
            f"Fold {fold}: checkpoint={checkpoint_path} sha256={_sha256(checkpoint_path)}"
        )
        model.eval()
        model.to("cuda" if torch.cuda.is_available() else "cpu")
        fold_results = evaluate_fold(fold, val_loader, model)
        probs = np.asarray(fold_results["logits"], dtype=np.float64).reshape(-1)
        fold_label = np.asarray(fold_results["labels"]).astype(int).reshape(-1)
        if len(probs) != len(ids):
            raise RuntimeError(
                f"fold {fold}: {len(probs)} predictions for {len(ids)} dataset ids"
            )
        # Labels are the same for every fold; a change means the loader reordered the
        # dataset between folds, so the IDs no longer describe the predictions.
        if fold_labels and not np.array_equal(fold_labels[0], fold_label):
            raise RuntimeError(
                f"fold {fold}: label order changed between folds; predictions are not "
                "ID-aligned"
            )
        if expected_labels is not None and not np.array_equal(
            expected_labels, fold_label
        ):
            raise RuntimeError(
                f"fold {fold}: label misalignment, dataset-order labels differ from "
                "the evaluated labels"
            )
        fold_probs.append(probs)
        fold_labels.append(fold_label)
        fold_preds = (probs >= tau).astype(int)
        acc = accuracy_score(fold_label, fold_preds)
        auc = roc_auc_score(fold_label, probs)
        f1 = f1_score(fold_label, fold_preds)
        metrics["acc"].append(acc)
        metrics["auc"].append(auc)
        metrics["f1"].append(f1)
        logger.debug(
            f"Fold {fold} confusion matrix at tau={tau}: "
            f"{confusion_matrix(fold_label, fold_preds).tolist()}"
        )
    # Compute average metrics from all folds
    for key in metrics:
        metrics[key] = np.round(np.mean(metrics[key]), 4)
    # Section 6.2 (decision A): the ensemble is the mean of the fold sigmoid
    # probabilities, exactly as the author; logits are never averaged.
    all_probs_ensemble = np.mean(fold_probs, axis=0)
    all_labels = fold_labels[0]
    # Compute ensemble metrics at the baseline's own threshold
    all_preds = (all_probs_ensemble >= tau).astype(int)
    accuracy = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs_ensemble)
    f1 = f1_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    ci = auroc_confidence_interval(all_labels, all_probs_ensemble)
    # Save metrics as dict for ease of reporting csv
    ensemble_metrics = {}
    ensemble_metrics["acc"] = np.round(accuracy, 4)
    ensemble_metrics["auc"] = np.round(auc, 4)
    ensemble_metrics["f1"] = np.round(f1, 4)
    ensemble_metrics["auroc_ci"] = np.round(ci, 4)
    ensemble_metrics["tau"] = np.round(tau, 6)
    # Log metrics
    logger.debug("\nEnsemble Evaluation Results:")
    logger.debug(f"Threshold tau: {tau}")
    logger.debug(f"Accuracy: {accuracy:.4f}")
    logger.debug(f"AUC: {auc:.4f}")
    logger.debug(f"AUROC CI: {ci}")
    logger.debug(f"F1: {f1:.4f}")
    logger.debug(f"Confusion Matrix: {cm}")
    # Section 6.4: predictions are keyed by ImageId, in dataset order.
    pred_df = pd.DataFrame(
        {"image_id": ids, "label": all_labels, "prob": all_probs_ensemble}
    )
    for fold, probs in enumerate(fold_probs):
        pred_df[f"prob_fold{fold}"] = probs
    return pred_df, ensemble_metrics

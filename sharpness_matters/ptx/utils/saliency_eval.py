"""
Utility file for saliency map evaluation
"""

import os
import hashlib
import json
import math
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import preprocess_image, show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Tuple, List, Union, Optional, Any
import matplotlib.pyplot as plt

from sharpness_matters.config.load_config import cfg
from sharpness_matters.ptx.corrected.preprocess import minmax_per_image

HOME = Path(__file__).resolve().parent.parent

# B (PTX-17 / §7.2): the fold-aggregation rule is recorded next to every number. The main
# rule is mean -> per-image min-max -> threshold; the author's mean -> threshold is kept
# only as a separately labelled sensitivity output. The two are never merged in a table.
# B (PTX-04 / §7.4): every saliency output is keyed by ImageId (contract 6 `image_id`) and
# carries the ensemble probability and the decision `prob >= tau`.
RULE_MAIN = "main"
RULE_LEGACY = "legacy"
RULE_DESCRIPTIONS = {
    RULE_MAIN: (
        "folds averaged on the input grid -> bilinear resize to the GT grid -> "
        "per-image min-max -> threshold"
    ),
    RULE_LEGACY: (
        "folds averaged on the input grid -> bilinear resize to the GT grid -> "
        "threshold without renormalisation; sensitivity analysis only"
    ),
}
SCORE_COLUMNS = [
    "image_id",
    "prob",
    "decision",
    "threshold",
    "iou",
    "precision",
    "coverage",
    "empty_pred_mask",
]
NEGATIVE_SCORE_COLUMNS = [
    "image_id",
    "prob",
    "decision",
    "threshold",
    "coverage",
    "empty_pred_mask",
]


def visualize_and_save(
    image: Image.Image,
    rollout_map: np.ndarray,
    save_path: str = "attention_rollout.png",
) -> Image.Image:
    """
    Save the aggregated attention map overlayed on the image.

    Args:
        image (Image.Image): Input PIL image
        rollout_map (np.ndarray): Aggregated attention map
        save_path (str): Path to save the overlayed image

    Returns:
        Image.Image: Blended image with attention overlay
    """
    rollout_map_resized = rollout_map / rollout_map.max()  # Normalize
    rollout_map_resized = np.kron(
        rollout_map_resized, np.ones((16, 16))
    )  # Upscale to 224x224
    # Convert rollout map to RGB heatmap
    cmap = plt.get_cmap("jet")
    heatmap = cmap(rollout_map_resized)  # Apply colormap
    heatmap = (heatmap[:, :, :3] * 255).astype(np.uint8)  # Convert to RGB
    # Convert image and heatmap to numpy
    original_image = image.convert("RGB")
    overlay = Image.blend(original_image, Image.fromarray(heatmap), alpha=0.4)
    return overlay


def tensor_to_pil(input_data: Union[torch.Tensor, np.ndarray]) -> Image.Image:
    """
    Convert tensor or numpy array to PIL Image.

    Args:
        input_data (Union[torch.Tensor, np.ndarray]): Input tensor or array

    Returns:
        Image.Image: Converted PIL Image
    """
    # If the input is a torch tensor, convert to numpy
    if isinstance(input_data, torch.Tensor):
        array = input_data.detach().cpu().numpy()
    elif isinstance(input_data, np.ndarray):
        array = input_data
    else:
        raise TypeError("Input must be a torch.Tensor or numpy.ndarray")
    # Check if the array is in channel-first format (C, H, W) and convert it to (H, W, C)
    if array.ndim == 3 and array.shape[0] in [1, 3]:
        array = np.transpose(array, (1, 2, 0))
    # If the array contains floats, assume range [0, 1] and convert to [0, 255]
    if np.issubdtype(array.dtype, np.floating):
        array = (array * 255).astype(np.uint8)
    return Image.fromarray(array)


def overlay_map(
    overlayviz: Union[torch.Tensor, np.ndarray],
    segviz: Union[torch.Tensor, np.ndarray],
    alpha: float = 0.5,
) -> Image.Image:
    """
    Overlay two images with specified transparency.

    Args:
        overlayviz (Union[torch.Tensor, np.ndarray]): Base image tensor/array
        segviz (Union[torch.Tensor, np.ndarray]): Overlay image tensor/array
        alpha (float): Transparency factor for blending

    Returns:
        Image.Image: Blended PIL Image
    """
    # Convert both tensors to PIL Images
    base_img = tensor_to_pil(overlayviz)
    overlay_img = tensor_to_pil(segviz)
    # Ensure both images are the same size (resize overlay if necessary)
    if overlay_img.size != base_img.size:
        overlay_img = overlay_img.resize(base_img.size)
    # Convert images to RGB (in case they aren't)
    base_img = base_img.convert("RGB")
    overlay_img = overlay_img.convert("RGB")
    # Blend the images using the provided alpha (0.0 to 1.0)
    blended = Image.blend(base_img, overlay_img, alpha)
    return blended


def calculate_precision(
    ground_truth: np.ndarray, segmentation: np.ndarray, threshold: float = 0.5
) -> float:
    """
    Calculate the Effective Heat Ratio (EHR) between a ground truth mask and a segmentation map.

    Args:
        ground_truth (np.ndarray): Binary ground truth mask (H, W), values are 0 or 1
        segmentation (np.ndarray): Predicted segmentation map (H, W), with values between 0 and 1
        threshold (float): Threshold to binarize the segmentation map

    Returns:
        float: Precision value
    """
    if ground_truth.shape != segmentation.shape:
        raise ValueError("Ground truth and segmentation must have the same shape.")
    binary_segmentation = (segmentation >= threshold).astype(np.float32)
    threshold_area_inside_gt = np.sum(binary_segmentation * ground_truth)
    complete_threshold_area = np.sum(binary_segmentation)
    if complete_threshold_area == 0:
        return 0.0
    precision = threshold_area_inside_gt / complete_threshold_area
    return precision


def collage_maker(image1: Image.Image, image2: Image.Image, name: str) -> None:
    """
    Create a horizontal collage of two images and save it.

    Args:
        image1 (Image.Image): First PIL image
        image2 (Image.Image): Second PIL image
        name (str): Output filename for the collage
    """
    i1 = np.array(image1)
    i2 = np.array(image2)
    collage = np.hstack([i1, i2])
    image = Image.fromarray(collage)
    image.save(name)


def fold_chunk_paths(
    cache_dir: Union[str, Path], fold: int, chunk_index: int
) -> Tuple[str, str, str]:
    """Paths of one (fold, chunk) record: probabilities, saliency maps, identity.

    B (PTX-18 / §7.3): the author's `.npy` pair carried no identity, so a chunk written
    by a different checkpoint was indistinguishable from the current one. New: a third
    sidecar file records the checkpoint SHA-256 and the ImageIds of the chunk.
    check: tests/corrected/test_saliency.py::test_chunk_with_other_checkpoint_sha_raises
    """
    base = os.path.join(str(cache_dir), f"fold_{fold}")
    return (
        f"{base}_logits_chunk_{chunk_index}.npy",
        f"{base}_sm_chunk_{chunk_index}.npy",
        f"{base}_meta_chunk_{chunk_index}.json",
    )


def save_fold_chunk(
    output_dir: Union[str, Path],
    fold: int,
    chunk_index: int,
    fold_probabilities: Dict[str, float],
    fold_saliency: Dict[str, np.ndarray],
    checkpoint_sha256: str,
    checkpoint_path: Optional[Union[str, Path]] = None,
) -> None:
    """Write one fold chunk with the identity of the checkpoint that produced it.

    The metadata file is written last: a crash between the `.npy` writes and the sidecar
    leaves a chunk that `load_fold_chunk` refuses to read instead of an anonymous chunk.
    """
    logits_path, sm_path, meta_path = fold_chunk_paths(output_dir, fold, chunk_index)
    if list(fold_probabilities) != list(fold_saliency):
        raise ValueError(
            "a fold chunk must hold the same ImageIds in its probability and saliency maps"
        )
    for path in (logits_path, sm_path, meta_path):
        if os.path.exists(path):
            raise RuntimeError(
                f"refusing to overwrite fold chunk file {path}; "
                "a chunk cache directory is unique per run (PTX-18)"
            )
    np.save(logits_path, fold_probabilities, allow_pickle=True)
    np.save(sm_path, fold_saliency, allow_pickle=True)
    meta = {
        "fold": int(fold),
        "chunk_index": int(chunk_index),
        "checkpoint_sha256": str(checkpoint_sha256),
        "checkpoint_path": None if checkpoint_path is None else str(checkpoint_path),
        "image_ids": list(fold_probabilities),
    }
    with open(meta_path, "w") as handle:
        json.dump(meta, handle, indent=2)


def load_fold_chunk(
    cache_dir: Union[str, Path],
    fold: int,
    chunk_index: int,
    checkpoint_sha256: str,
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    """Read one fold chunk; every missing piece or identity mismatch raises.

    B (PTX-18 / §7.3): current -> a missing fold file `continue`s and the remaining folds
    are averaged as if the ensemble were complete; problem -> a partial ensemble is
    reported as a normal result; new -> missing file, missing metadata or a chunk written
    by another checkpoint raises.
    check: tests/corrected/test_saliency.py::test_missing_fold_chunk_raises
    """
    logits_path, sm_path, meta_path = fold_chunk_paths(cache_dir, fold, chunk_index)
    missing = [
        path for path in (logits_path, sm_path, meta_path) if not os.path.isfile(path)
    ]
    if missing:
        raise RuntimeError(
            f"fold {fold} chunk {chunk_index} is incomplete; every fold must be "
            f"present, missing: {missing}"
        )
    with open(meta_path) as handle:
        meta = json.load(handle)
    if meta.get("checkpoint_sha256") != str(checkpoint_sha256):
        raise RuntimeError(
            f"fold {fold} chunk {chunk_index} was written by checkpoint "
            f"{meta.get('checkpoint_sha256')}, expected {checkpoint_sha256}"
        )
    if int(meta.get("fold", -1)) != int(fold) or int(
        meta.get("chunk_index", -1)
    ) != int(chunk_index):
        raise RuntimeError(
            f"chunk metadata at {meta_path} describes fold {meta.get('fold')} chunk "
            f"{meta.get('chunk_index')}, expected fold {fold} chunk {chunk_index}"
        )
    probabilities = np.load(logits_path, allow_pickle=True).item()
    saliency = np.load(sm_path, allow_pickle=True).item()
    image_ids = list(meta.get("image_ids", []))
    if list(probabilities) != image_ids or list(saliency) != image_ids:
        raise RuntimeError(
            f"fold {fold} chunk {chunk_index}: payload ImageIds do not match the "
            "chunk metadata"
        )
    return probabilities, saliency


def fold_checkpoint_shas(record: Dict[str, Any]) -> List[str]:
    """Checkpoint SHA-256 values of a cv record, ordered by fold (contract 5)."""
    folds = sorted(record["folds"], key=lambda entry: int(entry["fold"]))
    k_folds = int(record["k_folds"])
    if [int(entry["fold"]) for entry in folds] != list(range(k_folds)):
        raise RuntimeError(
            f"cv record folds {[entry['fold'] for entry in folds]} are not 0..{k_folds - 1}"
        )
    return [str(entry["sha256"]) for entry in folds]


def cache_dir_for_run(
    base_dir: Union[str, Path],
    dataset_name: str,
    model_name: str,
    img_size: int,
    checkpoint_sha256s: List[str],
) -> Path:
    """Chunk cache directory of one run, named after the checkpoints it belongs to.

    B (PTX-18 / §7.3, Y5): current -> `chunk_predictions_{model}_{size}` reused with
    `exist_ok=True`, so stale chunks of another checkpoint set are averaged in; new -> the
    name carries a short hash of the five checkpoint SHA-256s.
    check: tests/corrected/test_saliency.py::test_cache_dir_is_unique_per_checkpoint_set
    """
    k_folds = int(cfg.ptx.corrected.k_folds)
    if len(checkpoint_sha256s) != k_folds:
        raise ValueError(
            f"expected {k_folds} checkpoint hashes, got {len(checkpoint_sha256s)}"
        )
    digest = hashlib.sha256("".join(checkpoint_sha256s).encode("utf-8")).hexdigest()[
        :12
    ]
    return Path(base_dir) / (
        f"chunk_predictions_{dataset_name}_{model_name}_{img_size}_{digest}"
    )


def create_cache_dir(
    base_dir: Union[str, Path],
    dataset_name: str,
    model_name: str,
    img_size: int,
    checkpoint_sha256s: List[str],
) -> Path:
    """Create this run's chunk cache directory; an existing directory raises.

    A directory left behind by an earlier run with the same checkpoints is never reused
    silently: its chunks may be partial, so the owner has to remove it explicitly.
    """
    path = cache_dir_for_run(
        base_dir, dataset_name, model_name, img_size, checkpoint_sha256s
    )
    if path.exists():
        raise RuntimeError(
            f"chunk cache directory already exists: {path}; a previous run is not "
            "reused silently - inspect and delete it before evaluating again"
        )
    path.mkdir(parents=True)
    return path


def predict_fold_chunked(
    val_loader: DataLoader,
    model: torch.nn.Module,
    target_layers: List[Any],
    fold: int,
    checkpoint_sha256: str,
    checkpoint_path: Optional[Union[str, Path]] = None,
    chunk_size: int = 5,
    output_dir: str = f"{HOME}/output/chunk_predictions",
) -> None:
    """
    Process predictions and saliency maps in chunks for a specific fold and save them to disk.

    This function iterates over the validation DataLoader, computes model predictions and GradCAM saliency maps
    for each batch, and aggregates results into chunks. Once the number of processed samples reaches the chunk_size,
    the probabilities and saliency maps are saved as .npy files to the specified output directory, with filenames
    indicating the fold and chunk index, next to a metadata file that records the checkpoint SHA-256 the chunk
    came from (PTX-18 / §7.3).

    Args:
        val_loader (DataLoader): DataLoader providing validation batches; the collate must keep `mask` a list
        model (torch.nn.Module): The model used for prediction
        target_layers (List[Any]): List of target layers for computing GradCAM
        fold (int): The current fold index (used in output filenames)
        checkpoint_sha256 (str): SHA-256 of the checkpoint that produced this chunk
        checkpoint_path (Optional[Union[str, Path]]): Checkpoint path, recorded for auditing
        chunk_size (int): Number of samples per chunk
        output_dir (str): Directory where chunk files will be saved
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    # Initialize tracking variables
    all_logits = {}
    all_sm = {}
    sample_count = 0
    chunk_index = 0
    # Initialize GradCAM with the provided model and target layers.
    with GradCAM(model=model, target_layers=target_layers) as cam:
        for idx, batch in tqdm(
            enumerate(val_loader), total=len(val_loader), desc=f"Fold {fold}"
        ):
            pixel_values = batch["data"].to(device)
            # Contract 6: `image_id` is the ImageId key every saliency output is keyed by.
            image_ids = batch["image_id"]
            with torch.no_grad():
                # Sigmoid output: the ensemble averages probabilities (PTX-09 decision A).
                probabilities = (
                    torch.sigmoid(model(pixel_values).squeeze(1)).cpu().numpy()
                )
            # Since we have binary classificaiton, we will always get gradients corresponding to the positive class,
            # when using an index of 0 for ClassifierOutputTarget
            targets = [ClassifierOutputTarget(0) for _ in range(pixel_values.shape[0])]
            pixel_values.requires_grad = True
            grayscale_cam = cam(input_tensor=pixel_values, targets=targets)
            for k, image_id in enumerate(image_ids):
                all_logits[image_id] = probabilities[k]
                all_sm[image_id] = np.expand_dims(grayscale_cam[k], axis=0).transpose(
                    1, 2, 0
                )
                sample_count += 1
                # If the current chunk is full, save it to disk.
                if sample_count == chunk_size:
                    save_fold_chunk(
                        output_dir,
                        fold,
                        chunk_index,
                        all_logits,
                        all_sm,
                        checkpoint_sha256,
                        checkpoint_path,
                    )
                    chunk_index += 1
                    sample_count = 0
                    all_logits = {}
                    all_sm = {}
        # Save any remaining samples that did not fill up the last chunk.
        if sample_count > 0:
            save_fold_chunk(
                output_dir,
                fold,
                chunk_index,
                all_logits,
                all_sm,
                checkpoint_sha256,
                checkpoint_path,
            )
    return


def _image_id(item: Dict[str, Any]) -> str:
    """ImageId of one dataset item (contract 6: every masked item carries `image_id`)."""
    if "image_id" not in item:
        raise KeyError(
            "dataset item has no 'image_id' key; saliency outputs are keyed by ImageId "
            "(baseline_correction.md §7.4)"
        )
    return str(item["image_id"])


def _as_saliency_map(value: Any) -> np.ndarray:
    """Squeeze one stored fold saliency map (s, s, 1) to a 2-D input-grid array."""
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim != 2:
        raise ValueError(
            f"fold saliency map must be 2-D on the input grid, got shape {array.shape}"
        )
    return array


def resize_to_gt_grid(saliency: np.ndarray, gt_shape: Tuple[int, int]) -> np.ndarray:
    """Bilinear resize of one input-grid saliency map to the reference GT grid.

    B (PTX-02 / §7.1): current -> the GT mask is `Resize(antialias=True)`-d to the input
    grid and the scorer then applies `/255 -> uint8`, which erases mixed border pixels;
    new -> the GT mask stays on its reference annotation grid and the CAM is moved to it.
    The image was squashed (no crop, no pad) to s x s, so a full-frame bilinear resize of
    the CAM is the defined inverse of that squash.
    check: tests/corrected/test_saliency.py::test_gt_grid_preserved_for_thin_small_border_empty_wide_masks
    """
    array = np.asarray(saliency, dtype=np.float32)
    # A constant map resizes to the same constant; returning it directly avoids float32
    # rounding in the interpolated weights, which would otherwise turn a contrast-free
    # map into a full-range one at the min-max step (PTX-17 / §7.2).
    if array.size and float(array.max()) == float(array.min()):
        return np.full(
            (int(gt_shape[0]), int(gt_shape[1])), float(array.max()), dtype=np.float32
        )
    tensor = torch.from_numpy(array)[None, None]
    resized = F.interpolate(
        tensor,
        size=(int(gt_shape[0]), int(gt_shape[1])),
        mode="bilinear",
        align_corners=False,
    )
    return resized[0, 0].numpy()


def ensemble_fold_saliency(
    fold_saliency: List[np.ndarray], gt_shape: Tuple[int, int]
) -> np.ndarray:
    """Average the fold CAMs on the input grid, then resize the mean to the GT grid.

    Averaging before the resize keeps the author's fold-combination order: the five maps
    are combined first, then the single ensemble map is moved to the GT grid.
    """
    maps = [_as_saliency_map(value) for value in fold_saliency]
    if not maps:
        raise ValueError("no fold saliency maps to ensemble")
    shapes = {array.shape for array in maps}
    if len(shapes) != 1:
        raise ValueError(
            f"fold saliency maps must share one input grid, got {sorted(shapes)}"
        )
    mean_map = np.mean(np.stack(maps, axis=0), axis=0).astype(np.float32)
    return resize_to_gt_grid(mean_map, gt_shape)


def _iou(ground_truth: np.ndarray, prediction: np.ndarray) -> float:
    """Jaccard index of two binary masks; two empty masks are identical (1.0)."""
    gt = np.asarray(ground_truth).astype(bool)
    pred = np.asarray(prediction).astype(bool)
    union = int(np.logical_or(gt, pred).sum())
    if union == 0:
        return 1.0
    return float(np.logical_and(gt, pred).sum()) / union


def score_fold_saliency(
    fold_saliency: List[np.ndarray],
    ground_truth: np.ndarray,
    thresholds: List[float],
) -> Dict[str, Any]:
    """Score one image's fold saliency maps on its reference GT grid.

    B (PTX-02 / §7.1, PTX-17 / §7.2): current -> GT resized to the input grid, `/255 ->
    uint8`, fold mean thresholded without renormalisation; new -> GT stays on its grid,
    the CAM is resized to it, the main rule is mean -> min-max -> threshold, and the
    author's mean -> threshold is returned separately as `legacy`. A constant map has no
    contrast, so min-max maps it to `cfg.ptx.corrected.constant_image_value` and the
    predicted mask is empty. `>= threshold` counts a pixel exactly at the threshold as
    selected, the same convention as `calculate_precision`.
    check: tests/corrected/test_saliency.py::test_fold_mean_renormalisation_rescues_high_threshold
    """
    gt = np.asarray(ground_truth)
    if gt.ndim != 2:
        raise ValueError(
            "ground truth must be 2-D on the reference annotation grid "
            f"(contract 6), got shape {gt.shape}"
        )
    gt_binary = gt > 0
    ensemble = ensemble_fold_saliency(fold_saliency, gt.shape)
    normalized = minmax_per_image(ensemble)
    scores: Dict[str, Dict[float, Dict[str, Any]]] = {RULE_MAIN: {}, RULE_LEGACY: {}}
    for rule, saliency in ((RULE_MAIN, normalized), (RULE_LEGACY, ensemble)):
        for threshold in thresholds:
            threshold = float(threshold)
            predicted = saliency >= threshold
            scores[rule][threshold] = {
                "iou": _iou(gt_binary, predicted),
                "precision": calculate_precision(
                    gt_binary.astype(np.uint8), saliency, threshold
                ),
                "coverage": float(predicted.sum()) / float(predicted.size),
                "empty_pred_mask": bool(not predicted.any()),
            }
    return {
        "ensemble_saliency": ensemble,
        "normalized_saliency": normalized,
        "scores": scores,
    }


def _resize_nearest_2d(array: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Nearest resize used only for map visualisation, never for scoring."""
    tensor = torch.from_numpy(np.asarray(array, dtype=np.float32))[None, None]
    resized = F.interpolate(tensor, size=(int(shape[0]), int(shape[1])), mode="nearest")
    return resized[0, 0].numpy()


def _save_map_visualisations(
    item: Dict[str, Any],
    dataset: Dataset,
    ensemble_saliency: np.ndarray,
    ground_truth: np.ndarray,
    thresholds: List[float],
    save_dirs: Dict[str, Any],
    image_id: str,
) -> None:
    """Write the author's three visualisations per threshold on the display grid only.

    The mask overlay thresholds the raw fold mean exactly as the author did; it is a
    visualisation of the legacy rule, not the main-rule mask the CSV scores.
    """
    data = item["data"]
    if data.ndim != 3:
        raise ValueError(f"'data' must be (C, H, W), got shape {tuple(data.shape)}")
    display_shape = (int(data.shape[1]), int(data.shape[2]))
    binary_mask_viz = data.permute(1, 2, 0).repeat(1, 1, 3).detach().cpu()
    sm_display = _resize_nearest_2d(ensemble_saliency, display_shape)
    segviz = show_cam_on_image(
        np.repeat(sm_display[..., None], 3, axis=-1), sm_display, use_rgb=True
    )
    overlayviz = binary_mask_viz.clone()
    segviz_overlay = overlay_map(overlayviz, segviz, alpha=0.5)
    segviz_overlay.save(os.path.join(save_dirs["sm"], f"{image_id}.jpg"))
    # The GT overlay does not depend on the saliency threshold; save it once per image.
    gt_display = _resize_nearest_2d(
        (ground_truth > 0).astype(np.float32), display_shape
    )
    gtviz = dataset.visualize_segmentation(
        binary_mask_viz, torch.from_numpy(gt_display * 255.0)
    )
    Image.fromarray(gtviz).save(os.path.join(save_dirs["gt"], f"{image_id}.jpg"))
    for threshold in thresholds:
        mask_dir = save_dirs["masks"][float(threshold)]
        predicted = (ensemble_saliency >= float(threshold)).astype(np.float32)
        mask_display = _resize_nearest_2d(predicted, display_shape)
        maskviz = dataset.visualize_segmentation(
            binary_mask_viz, torch.from_numpy(mask_display * 255.0)
        )
        Image.fromarray(maskviz).save(os.path.join(mask_dir, f"{image_id}.jpg"))


def _mean_std(values: List[float]) -> Tuple[float, float]:
    """Mean and population standard deviation; empty input yields NaN, never an error."""
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return float("nan"), float("nan")
    return float(array.mean()), float(array.std())


def summarize_scores(
    rows: List[Dict[str, Any]], thresholds: List[float]
) -> Dict[str, Any]:
    """Per-threshold summaries with explicit denominators (PTX-04 / §7.4).

    Three cohorts are reported side by side and are never collapsed into one number:
    `all_positive` (K1, every GT-positive image, decision independent),
    `tp_conditional` (positive decision only) and `joint` (K1 denominator, a wrong
    negative contributes 0; coverage keeps its own positive-decision denominator because
    the author records no coverage for a wrong negative).
    """
    summaries: Dict[str, Any] = {}
    for threshold in thresholds:
        threshold = float(threshold)
        subset = [row for row in rows if row["threshold"] == threshold]
        positive = [row for row in subset if row["decision"] == 1]
        iou_mean, iou_std = _mean_std([row["iou"] for row in subset])
        precision_mean, precision_std = _mean_std([row["precision"] for row in subset])
        coverage_mean, _ = _mean_std([row["coverage"] for row in subset])
        tp_iou_mean, tp_iou_std = _mean_std([row["iou"] for row in positive])
        tp_precision_mean, tp_precision_std = _mean_std(
            [row["precision"] for row in positive]
        )
        tp_coverage_mean, _ = _mean_std([row["coverage"] for row in positive])
        joint_iou_mean, joint_iou_std = _mean_std(
            [row["iou"] if row["decision"] == 1 else 0.0 for row in subset]
        )
        joint_precision_mean, joint_precision_std = _mean_std(
            [row["precision"] if row["decision"] == 1 else 0.0 for row in subset]
        )
        empty = sum(1 for row in subset if int(row["empty_pred_mask"]) == 1)
        summaries[f"{threshold:g}"] = {
            "threshold": threshold,
            "all_positive": {
                "n": len(subset),
                "iou_mean": iou_mean,
                "iou_std": iou_std,
                "precision_mean": precision_mean,
                "precision_std": precision_std,
                "coverage_mean": coverage_mean,
                "coverage_n": len(subset),
            },
            "tp_conditional": {
                "n": len(positive),
                "iou_mean": tp_iou_mean,
                "iou_std": tp_iou_std,
                "precision_mean": tp_precision_mean,
                "precision_std": tp_precision_std,
                "coverage_mean": tp_coverage_mean,
                "coverage_n": len(positive),
            },
            "joint": {
                "n": len(subset),
                "iou_mean": joint_iou_mean,
                "iou_std": joint_iou_std,
                "precision_mean": joint_precision_mean,
                "precision_std": joint_precision_std,
                "coverage_mean": tp_coverage_mean,
                "coverage_n": len(positive),
            },
            "empty_pred_mask": {
                "n": len(subset),
                "n_empty": empty,
                "fraction": (float(empty) / len(subset)) if subset else float("nan"),
            },
        }
    return summaries


def compute_scores_chunked(
    val_dataset: Dataset,
    cache_dir: Union[str, Path],
    record: Dict[str, Any],
    model_name: str,
    img_size: int,
    dataset_name: str,
    validation_mode: str = "holdout",
    thresholds: Optional[List[float]] = None,
    chunk_size: int = 5,
    save_maps: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Score every image from the pre-saved fold chunks, decision independently (K1).

    B (PTX-04 / §7.4): current -> a wrong negative (GT positive, `prob < tau`) got IoU and
    precision 0 without its CAM being looked at, a wrong positive was scored without
    coverage, and `num_accurate = len(iou) / len(dataset)` was reported as if it were
    accuracy; new -> every GT-positive image is scored from its positive-class CAM
    regardless of the decision, and the `all_positive` (K1), `tp_conditional` and `joint`
    summaries carry their own denominators. No `num_accurate`/`fraction_scored` is
    reported: K1 has no unscored image. A GT-positive image (`label == 1`) whose
    reference-grid mask is empty cannot be scored for localization: it is neither scored
    nor dropped, the call raises `RuntimeError` listing its ImageId(s) before any result
    is returned or written.
    check: tests/corrected/test_saliency.py::test_wrong_negative_is_scored_in_k1

    Args:
        val_dataset (Dataset): masked dataset; `mask` is uint8 {0,1} on the reference grid
        cache_dir (Union[str, Path]): directory holding the fold chunks of this run
        record (Dict[str, Any]): cv record of the trained ensemble (contract 5); supplies
            tau and the checkpoint SHA-256s every chunk is verified against
        model_name (str): model whose maps are scored (result metadata, visualisation dirs)
        img_size (int): input resolution of this run (result metadata, visualisation dirs)
        dataset_name (str): dataset label carried in the result metadata
        validation_mode (str): 'holdout' or 'ood' (visualisation dirs only)
        thresholds (Optional[List[float]]): saliency thresholds; default
            `cfg.ptx.corrected.saliency_thresholds`
        chunk_size (int): samples per chunk file, identical for every fold
        save_maps (bool): write the author's map visualisations under `ptx/output`

    Returns:
        Dict[str, Dict[str, Any]]: one result per rule (`main`, `legacy`) with per-image
        `rows`, `negative_rows` (coverage only), the K1 `positive_ids` and `summaries`.
    """
    k_folds = int(cfg.ptx.corrected.k_folds)
    if thresholds is None:
        thresholds = list(cfg.ptx.corrected.saliency_thresholds)
    thresholds = [float(threshold) for threshold in thresholds]
    if not thresholds:
        raise ValueError("at least one saliency threshold is required")
    if int(record["k_folds"]) != k_folds:
        raise RuntimeError(
            f"cv record has {record['k_folds']} folds, config requires {k_folds}"
        )
    checkpoint_sha256s = fold_checkpoint_shas(record)
    tau = float(record["tau"])
    n_images = len(val_dataset)
    if n_images == 0:
        raise RuntimeError("cannot score saliency on an empty dataset")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    save_dirs: Dict[str, Any] = {}
    if save_maps:
        base = f"{HOME}/output/{model_name}_maps/maps_{validation_mode}/maps_{img_size}"
        save_dirs = {
            "gt": f"{base}/gt",
            "sm": f"{base}/sm",
            "masks": {t: f"{base}/masks_{t}" for t in thresholds},
        }
        os.makedirs(save_dirs["gt"], exist_ok=True)
        os.makedirs(save_dirs["sm"], exist_ok=True)
        for mask_dir in save_dirs["masks"].values():
            os.makedirs(mask_dir, exist_ok=True)
    rows: Dict[str, List[Dict[str, Any]]] = {RULE_MAIN: [], RULE_LEGACY: []}
    negative_rows: Dict[str, List[Dict[str, Any]]] = {RULE_MAIN: [], RULE_LEGACY: []}
    positive_ids: List[str] = []
    negative_ids: List[str] = []
    unscorable: List[str] = []
    num_chunks = int(math.ceil(n_images / chunk_size))
    for chunk_index in range(num_chunks):
        start = chunk_index * chunk_size
        stop = min(n_images, start + chunk_size)
        items = [val_dataset[index] for index in range(start, stop)]
        chunk_ids = [_image_id(item) for item in items]
        if len(set(chunk_ids)) != len(chunk_ids):
            raise RuntimeError(
                f"dataset chunk {chunk_index} repeats an ImageId: {chunk_ids}"
            )
        fold_probabilities: Dict[int, Dict[str, float]] = {}
        fold_saliency: Dict[int, Dict[str, np.ndarray]] = {}
        for fold in range(k_folds):
            probabilities, saliency = load_fold_chunk(
                cache_dir, fold, chunk_index, checkpoint_sha256s[fold]
            )
            if set(probabilities) != set(chunk_ids):
                raise RuntimeError(
                    f"fold {fold} chunk {chunk_index} ImageIds {sorted(probabilities)} "
                    f"do not match the dataset slice {sorted(chunk_ids)}"
                )
            if set(saliency) != set(chunk_ids):
                raise RuntimeError(
                    f"fold {fold} chunk {chunk_index} saliency ImageIds "
                    f"{sorted(saliency)} do not match the dataset slice {sorted(chunk_ids)}"
                )
            fold_probabilities[fold] = probabilities
            fold_saliency[fold] = saliency
        for item, image_id in zip(items, chunk_ids):
            # PTX-09 decision A: fold probabilities are averaged; decision is `p >= tau`.
            prob = float(
                np.mean([fold_probabilities[fold][image_id] for fold in range(k_folds)])
            )
            decision = 1 if prob >= tau else 0
            fold_maps = [
                _as_saliency_map(fold_saliency[fold][image_id])
                for fold in range(k_folds)
            ]
            ground_truth = np.asarray(item["mask"])
            label = int(np.asarray(item["label"]).reshape(-1)[0])
            if label == 1 and not (ground_truth > 0).any():
                # Localization is undefined without a reference-grid foreground: the image
                # is neither scored nor dropped, the whole call fails with its ImageId.
                unscorable.append(image_id)
                continue
            scored = score_fold_saliency(fold_maps, ground_truth, thresholds)
            if label == 1:
                positive_ids.append(image_id)
                for rule in (RULE_MAIN, RULE_LEGACY):
                    for threshold in thresholds:
                        metric = scored["scores"][rule][float(threshold)]
                        rows[rule].append(
                            {
                                "image_id": image_id,
                                "prob": prob,
                                "decision": decision,
                                "threshold": float(threshold),
                                "iou": metric["iou"],
                                "precision": metric["precision"],
                                "coverage": metric["coverage"],
                                "empty_pred_mask": int(metric["empty_pred_mask"]),
                            }
                        )
                if save_maps:
                    _save_map_visualisations(
                        item,
                        val_dataset,
                        scored["ensemble_saliency"],
                        ground_truth,
                        thresholds,
                        save_dirs,
                        image_id,
                    )
            else:
                negative_ids.append(image_id)
                # A negative image has no pixel GT: only coverage of its CAM is defined.
                for rule in (RULE_MAIN, RULE_LEGACY):
                    for threshold in thresholds:
                        metric = scored["scores"][rule][float(threshold)]
                        negative_rows[rule].append(
                            {
                                "image_id": image_id,
                                "prob": prob,
                                "decision": decision,
                                "threshold": float(threshold),
                                "coverage": metric["coverage"],
                                "empty_pred_mask": int(metric["empty_pred_mask"]),
                            }
                        )
    if unscorable:
        raise RuntimeError(
            "cannot score localization for GT-positive images whose reference-grid mask "
            f"is empty (label == 1): {unscorable}"
        )
    n_positive = len(positive_ids)
    n_tp = sum(
        1
        for row in rows[RULE_MAIN]
        if row["threshold"] == thresholds[0] and row["decision"] == 1
    )
    results: Dict[str, Dict[str, Any]] = {}
    for rule in (RULE_MAIN, RULE_LEGACY):
        results[rule] = {
            "rule": rule,
            "rule_description": RULE_DESCRIPTIONS[rule],
            "sensitivity_analysis": rule == RULE_LEGACY,
            "model_name": model_name,
            "img_size": int(img_size),
            "dataset": dataset_name,
            "validation_mode": validation_mode,
            "tau": tau,
            "thresholds": thresholds,
            "n_images": n_images,
            "n_positive": n_positive,
            "n_negative": len(negative_ids),
            "n_tp": n_tp,
            "sensitivity": (float(n_tp) / n_positive) if n_positive else float("nan"),
            "positive_ids": positive_ids,
            "rows": rows[rule],
            "negative_rows": negative_rows[rule],
            "summaries": summarize_scores(rows[rule], thresholds),
        }
    return results


def _json_safe(value: Any) -> Any:
    """Replace non-finite floats by null so the summary is strict JSON.

    An undefined cohort (e.g. no true positive at this operating point) is reported as
    null instead of a NaN token that strict JSON readers reject.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def write_saliency_outputs(
    results: Dict[str, Dict[str, Any]],
    output_dir: Union[str, Path],
    dataset_name: str,
    model_name: str,
    img_size: int,
) -> Dict[str, Dict[str, str]]:
    """Write one CSV and one JSON summary per (model, img_size, dataset, rule).

    B (PTX-04 / §7.4): the CSV is keyed by ImageId and carries `prob`, `decision`,
    `threshold`, `iou`, `precision`, `coverage` and `empty_pred_mask`; the JSON keeps the
    K1 / TP-conditional / joint denominators. The K1 key set is asserted to be exactly the
    GT-positive ImageId set, so a dropped image cannot pass as a normal result.
    check: tests/corrected/test_saliency.py::test_k1_key_set_equals_positive_image_ids
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    written: Dict[str, Dict[str, str]] = {}
    for rule, result in results.items():
        frame = pd.DataFrame(result["rows"], columns=SCORE_COLUMNS)
        positive_ids = set(result["positive_ids"])
        if set(frame["image_id"]) != positive_ids:
            raise RuntimeError(
                f"{rule}: scored ImageIds {sorted(set(frame['image_id']))} are not the "
                f"GT-positive ImageId set {sorted(positive_ids)}"
            )
        csv_path = output_path / (
            f"saliency_{dataset_name}_{model_name}_{img_size}_{rule}.csv"
        )
        frame.to_csv(csv_path, index=False)
        files = {"csv": str(csv_path)}
        if result["negative_rows"]:
            negative_frame = pd.DataFrame(
                result["negative_rows"], columns=NEGATIVE_SCORE_COLUMNS
            )
            negative_path = output_path / (
                f"saliency_{dataset_name}_{model_name}_{img_size}_{rule}_negatives.csv"
            )
            negative_frame.to_csv(negative_path, index=False)
            files["negatives_csv"] = str(negative_path)
        summary = {
            key: value
            for key, value in result.items()
            if key not in ("rows", "negative_rows", "positive_ids")
        }
        json_path = output_path / (
            f"saliency_{dataset_name}_{model_name}_{img_size}_{rule}.json"
        )
        with open(json_path, "w") as handle:
            json.dump(_json_safe(summary), handle, indent=2)
        files["summary"] = str(json_path)
        written[rule] = files
    return written


def summary_table(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten one rule's per-threshold summaries into rows for the per-model CSV."""
    table: List[Dict[str, Any]] = []
    for summary in result["summaries"].values():
        for cohort in ("all_positive", "tp_conditional", "joint"):
            entry = summary[cohort]
            table.append(
                {
                    "rule": result["rule"],
                    "model_name": result["model_name"],
                    "dataset": result["dataset"],
                    "img_size": result["img_size"],
                    "threshold": summary["threshold"],
                    "cohort": cohort,
                    "n": entry["n"],
                    "iou_mean": entry["iou_mean"],
                    "iou_std": entry["iou_std"],
                    "precision_mean": entry["precision_mean"],
                    "precision_std": entry["precision_std"],
                    "coverage_mean": entry["coverage_mean"],
                    "coverage_n": entry["coverage_n"],
                    "empty_count": summary["empty_pred_mask"]["n_empty"],
                    "empty_fraction": summary["empty_pred_mask"]["fraction"],
                    "tau": result["tau"],
                    "n_positive": result["n_positive"],
                    "n_tp": result["n_tp"],
                    "sensitivity": result["sensitivity"],
                    "sensitivity_analysis": result["sensitivity_analysis"],
                }
            )
    return table

import os
from pathlib import Path
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import subprocess
import glob
import pandas as pd
from tqdm import tqdm
import cv2
import nibabel as nib
from typing import Dict, Any, Tuple, Union, List, Optional

from sharpness_matters.ptx.utils.logging_utils import initialize_logger
from sharpness_matters.ptx.corrected.preprocess import eval_transform, minmax_per_image
from sharpness_matters.ptx.corrected.frozen import load_frozen, manifest_dir
from sharpness_matters.config.load_config import cfg

HOME = Path(__file__).resolve().parent.parent
logger = initialize_logger(level="info")

COHORT_MANIFEST_NAME = "ptx498_cohort.csv"
SITES = ("A", "B", "C")
IMAGE_SUFFIX = ".4.img.nii.gz"
MASK_SUFFIX = ".5.mask.nii.gz"
_MAX_EXAMPLES = 5


def default_cohort_manifest() -> Path:
    """Frozen PTX-498 cohort manifest: ``manifests/ptx498_cohort.csv``."""
    return manifest_dir() / COHORT_MANIFEST_NAME


def _examples(items: List[str]) -> str:
    """Up to five example values, with the remainder counted."""
    shown = items[:_MAX_EXAMPLES]
    text = ", ".join(shown)
    if len(items) > _MAX_EXAMPLES:
        text += f", ... ({len(items) - _MAX_EXAMPLES} more)"
    return text


def verify_cohort_files(
    root_dir: Union[str, Path], image_ids: List[str]
) -> None:
    """Require the directory to hold exactly the frozen cohort's NIfTI pairs.

    Every manifest ID must have both ``N.4.img.nii.gz`` and ``N.5.mask.nii.gz`` under
    ``Site<site>``; a case present on disk but absent from the manifest, a missing or
    empty site directory, or an unknown site in the manifest fails the run.

    # B (PTX-18 / Section 7.3, follow-up): the cohort was ``os.listdir`` of SiteA/B/C,
    #   so an existing but EMPTY SiteC/ gave 426 IDs (247+179+0) instead of 498, and a
    #   case with only PNGs was accepted at parse time. New: the frozen manifest is
    #   the cohort and this check proves the directory matches it, before
    #   ``image_ids`` is published.
    #   check: tests/corrected/test_ptx498_cohort.py
    """
    root = Path(root_dir)
    if not root.is_dir():
        raise RuntimeError(
            f"PTX-498 root directory {root} does not exist; the frozen cohort "
            f"({len(image_ids)} cases) cannot be located"
        )

    expected: Dict[str, List[str]] = {}
    for image_id in image_ids:
        site, case = image_id.split("_", 1)
        expected.setdefault(site, []).append(case)

    issues: List[str] = []
    unknown_sites = sorted(set(expected) - set(SITES))
    if unknown_sites:
        issues.append(
            f"the frozen manifest names site(s) {unknown_sites}; expected {list(SITES)}"
        )

    for site in SITES:
        cases = expected.get(site, [])
        site_ids = [f"{site}_{case}" for case in cases]
        site_dir = root / f"Site{site}"
        if not site_dir.is_dir():
            issues.append(
                f"Site{site} directory is missing under {root}; the frozen manifest "
                f"expects {len(cases)} case(s)"
                + (f" (e.g. {_examples(site_ids)})" if cases else "")
            )
            continue
        entries = sorted(entry.name for entry in site_dir.iterdir())
        if not entries:
            issues.append(
                f"Site{site} directory {site_dir} is empty; the frozen manifest "
                f"expects {len(cases)} case(s)"
                + (f" (missing IDs e.g. {_examples(site_ids)})" if cases else "")
            )
            continue
        missing = []
        for case in cases:
            absent = [
                suffix
                for suffix in (IMAGE_SUFFIX, MASK_SUFFIX)
                if not (site_dir / f"{case}{suffix}").is_file()
            ]
            if absent:
                missing.append(f"{site}_{case} (missing {', '.join(absent)})")
        if missing:
            issues.append(
                f"Site{site}: {len(missing)} frozen case(s) lack their NIfTI files "
                f"(e.g. {_examples(missing)})"
            )
        present = {
            name[: -len(suffix)]
            for name in entries
            for suffix in (IMAGE_SUFFIX, MASK_SUFFIX)
            if name.endswith(suffix)
        }
        extra = sorted(present - set(cases))
        if extra:
            issues.append(
                f"Site{site}: {len(extra)} case(s) on disk are not in the frozen "
                f"manifest (e.g. {_examples([f'{site}_{case}' for case in extra])})"
            )

    if issues:
        raise RuntimeError(
            "PTX-498 cohort verification failed against the frozen manifest "
            f"({len(image_ids)} cases, root {root}): " + "; ".join(issues)
        )


def orient_ptx498(tensor: torch.Tensor) -> torch.Tensor:
    """Put a PTX-498 image or mask into the dataset's frame (last two axes).

    The author rotates the NIfTI array by 90 degrees; that keeps up/down but mirrors
    left/right against the dataset's own PNG (readable "R" on the image left). The
    dataset frame is the author's rotation followed by a horizontal flip (last axis,
    i.e. ``np.fliplr`` of the rotated array), which matches the PNG for every image in
    the real-data check. Image and mask both go through this one helper so they cannot
    diverge.

    # B (§9.1 / Y6): author rot90 mirrors left-right vs the dataset PNG (497/498
    # author-mirrored; fixed helper matches 498/498; reports/tools/corr.py) ->
    # rot90 + horizontal flip (last axis) for image and mask
    #   check: tests/corrected/test_data.py::test_ptx498_orientation_is_rot90_plus_horizontal_flip
    """
    rotated = torch.rot90(tensor, k=1, dims=(-2, -1))
    return torch.flip(rotated, dims=(-1,))



class OODTestDataset(Dataset):
    """
    PyTorch Dataset for PTX498 out-of-distribution test data with NIfTI images.
    """

    def __init__(
        self,
        root_dir: str = cfg.ptx.ptx498.root_dir,
        img_size: int = 224,
        transform: Optional[Any] = None,
        manifest_csv: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize PTX498 dataset from the frozen cohort manifest.

        # B (PTX-27 / Section 5.4): no dataset mean/std and no stats pickle; the
        #   image is scaled per image with min-max. # B (PTX-26 / Section 5.3): the
        #   default preprocessing is the shared eval_transform policy. # B (PTX-02 /
        #   Section 7.1): the mask stays on the native NIfTI grid (no resize, no
        #   *255, uint8 {0,1}). # B (§9.1 / Y6): image and mask go through the same
        #   orientation helper (author rot90 + horizontal flip on the last axis); the
        #   author's rot90 alone mirrored left/right vs the dataset PNG.
        # B (PTX-18 / Section 7.3, follow-up): the cohort is the frozen manifest, and
        #   the directory must match it; a partial site cannot shrink the test set.
        #   check: tests/corrected/test_data.py::test_ptx498_*,
        #          tests/corrected/test_ptx498_cohort.py

        Args:
            root_dir (str): Root directory containing Site folders with NIfTI files
            img_size (int): Target image size for resizing
            transform (Optional[Any]): Explicit preprocessing; defaults to eval_transform
            manifest_csv (Optional[Union[str, Path]]): Frozen cohort manifest
                (image_id,site,case); defaults to manifests/ptx498_cohort.csv
        """
        manifest_path = (
            Path(manifest_csv) if manifest_csv is not None else default_cohort_manifest()
        )
        manifest = load_frozen(manifest_path)
        if "image_id" not in manifest.columns:
            raise RuntimeError(
                f"frozen PTX-498 manifest {manifest_path} has no image_id column"
            )
        image_ids = [str(image_id) for image_id in manifest["image_id"]]
        verify_cohort_files(root_dir, image_ids)
        self.root_dir = root_dir
        self.img_size = img_size
        self.image_ids = image_ids
        self.transform = (
            transform if transform is not None else eval_transform(img_size)
        )

    def visualize_segmentation(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        alpha: float = 0.5,
        mask_color: Tuple[float, float, float] = (1, 0, 0),
    ) -> np.ndarray:
        """
        Visualize a segmentation mask overlaid on an image.

        Args:
            image (np.ndarray): The base image (H, W, 3)
            mask (np.ndarray): The segmentation mask (H, W), binary or multi-class
            alpha (float): Transparency of the mask overlay (0-1)
            mask_color (Tuple[float, float, float]): RGB color of the mask overlay

        Returns:
            np.ndarray: Image with overlaid segmentation mask
        """
        # Ensure the image and mask have the same dimensions
        if image.shape[:2] != mask.shape:
            raise ValueError(
                f"Image and mask must have the same height and width. Image shape: {image.shape}, Mask shape: {mask.shape}"
            )
        image = np.array(image * 255).astype(np.uint8)
        # Convert binary mask to RGB if necessary. Masks are uint8 {0,1} (PTX-02 /
        # Section 7.1), so the foreground is scaled to 255 for display; a 0/255 mask
        # (e.g. the thresholded CAM) is unchanged.
        if len(mask.shape) == 2:  # Single-channel binary or multi-class mask
            mask = np.asarray(mask)
            foreground = np.where(mask > 0, 255, 0).astype(np.uint8)
            colored_mask = np.zeros_like(image, dtype=np.uint8)
            for c in range(3):
                colored_mask[..., c] = foreground * mask_color[c]  # Apply mask color
        else:
            colored_mask = mask  # Assume already RGB for multi-class

        # Overlay the mask on the image
        overlaid_image = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)
        return overlaid_image

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(
        self, idx: int
    ) -> Dict[str, Union[torch.Tensor, np.ndarray, int, str]]:
        """
        Get a single sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve

        Returns:
            Dict[str, Union[torch.Tensor, np.ndarray, int, str]]: Dictionary containing:
                - data: Processed image tensor
                - mask: Binary mask array on the native NIfTI grid, uint8 {0,1}
                - label: Binary label (always 1 for pneumothorax)
                - pid / image_id: image ID string
        """
        # Extract image and mask paths
        image_id = self.image_ids[idx]
        site = image_id.split("_")[0]
        number = image_id.split("_")[1]
        image_path = os.path.join(
            self.root_dir, f"Site{site}", f"{number}.4.img.nii.gz"
        )
        mask_path = os.path.join(
            self.root_dir, f"Site{site}", f"{number}.5.mask.nii.gz"
        )
        # Load from paths. # B (PTX-02 / Section 7.1): the mask keeps its native
        # grid (no resize) and is binarised with `> 0` (Section 9.2 value range is an
        # open data check, so no `* 255` and no assumed {0,1}/{0,255} convention).
        mask = nib.load(mask_path).get_fdata()
        mask = np.squeeze(mask)
        if mask.ndim != 2:
            raise RuntimeError(
                f"PTX-498 mask {mask_path} is not a 2D grid after squeeze: "
                f"shape {mask.shape}"
            )
        mask = (mask > 0).astype(np.uint8)
        image = nib.load(image_path).get_fdata().astype(np.float32)
        image = np.expand_dims(image, -1)
        image = minmax_per_image(image)
        image = self.transform(image)  # eval_transform: ToTensor -> Resize
        # §9.1 decided: the author's rot90 alone mirrors left/right vs the dataset PNG.
        # B applies the mirror fix through the shared helper, for image and mask
        # together (rot90 + horizontal flip on the last axis).
        image = orient_ptx498(image)
        mask = orient_ptx498(torch.from_numpy(mask)).numpy()
        return {
            "data": image,
            "mask": mask,
            "label": 1,
            "pid": f"{site}_{number}",
            "image_id": f"{site}_{number}",
        }


if __name__ == "__main__":
    dataset = OODTestDataset()
    for idx in range(len(dataset)):
        batch = dataset[idx]
        logger.debug(f"Image id: {dataset.image_ids[idx]}")
        img = batch["data"]
        mask = batch["mask"]
        logger.debug(f"Image type: {type(img)}, Shape: {img.shape}")
        logger.debug(f"Mean: {img.mean()}, Max: {img.max()}, Min: {img.min()}")
        logger.debug(f"Mask type: {type(mask)}, Shape: {mask.shape}")
        logger.debug(f"Mean: {mask.mean()}, Max: {mask.max()}, Min: {mask.min()}")
        # Visualize segmentation and save
        out_dir = Path(HOME) / "output" / "seg_viz"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Prepare image (H,W,3) in [0,1]
        if torch.is_tensor(img):
            img_np = img.squeeze().cpu().numpy()  # (H,W)
        else:
            img_np = np.squeeze(img)
        img_rgb = np.stack([img_np] * 3, axis=-1)  # (H,W,3)

        # Prepare mask (H,W) binary
        mask_arr = mask
        if torch.is_tensor(mask_arr):
            mask_arr = mask_arr.squeeze().cpu().numpy()
        else:
            mask_arr = np.squeeze(mask_arr)

        seg_viz = dataset.visualize_segmentation(
            image=img_rgb, mask=mask_arr, alpha=0.4, mask_color=(1, 0, 0)
        )

        save_path = out_dir / f"{batch['pid']}.png"
        cv2.imwrite(str(save_path), cv2.cvtColor(seg_viz, cv2.COLOR_RGB2BGR))
        logger.debug(f"Saved segmentation visualization to {save_path}")
        break

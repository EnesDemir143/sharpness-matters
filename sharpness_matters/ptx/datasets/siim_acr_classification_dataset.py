import os
from pathlib import Path
import torch
from torch.utils.data import Dataset
import pydicom
import numpy as np
from PIL import Image
import subprocess
import glob
import pandas as pd
from tqdm import tqdm
import cv2
from sklearn.model_selection import train_test_split
from typing import Dict, Any, Tuple, Union, List, Optional


from sharpness_matters.ptx.utils.logging_utils import initialize_logger
from sharpness_matters.ptx.corrected.preprocess import eval_transform, minmax_per_image
from sharpness_matters.config.load_config import cfg

HOME_DIR = Path(__file__).resolve().parent.parent
logger = initialize_logger(level="info")


def index_dicom_files(file_paths: List[str]) -> Tuple[Dict[str, str], List[str]]:
    """Map ImageId -> DICOM path by exact filename stem; list ambiguous stems.

    # B (Section 10.3 / Y10): the old lookup was `image_id in path` over an unsorted
    #   list, so a substring match (or the first of several files) decided the study.
    #   New: the ImageId is the DICOM filename stem, matched exactly, in one dict.
    #   check: tests/corrected/test_data.py::test_siim_file_map_rejects_*
    """
    file_map: Dict[str, str] = {}
    duplicate_ids: List[str] = []
    for path in file_paths:
        stem = Path(path).stem
        if stem in file_map:
            if str(file_map[stem]) != str(path) and stem not in duplicate_ids:
                duplicate_ids.append(stem)
            continue
        file_map[stem] = path
    return file_map, duplicate_ids


def require_unique_files(
    image_ids: List[str], file_map: Dict[str, str], duplicate_ids: List[str]
) -> None:
    """Raise unless every ImageId maps to exactly one file (Section 10.3 / Y10)."""
    missing = sorted(image_id for image_id in image_ids if image_id not in file_map)
    if missing:
        raise RuntimeError(
            f"{len(missing)} ImageId(s) have no DICOM file with a matching filename "
            f"stem, e.g. {missing[:5]}"
        )
    duplicated = set(duplicate_ids)
    ambiguous = sorted(image_id for image_id in image_ids if image_id in duplicated)
    if ambiguous:
        raise RuntimeError(
            f"{len(ambiguous)} ImageId(s) map to more than one DICOM file, e.g. "
            f"{ambiguous[:5]}"
        )


def siim_labels_from_rle(label_df: pd.DataFrame) -> Dict[str, int]:
    """ImageId -> 1 when any RLE row differs from ``" -1"`` (the author's rule).

    # B (PTX-27 / Section 5.4): labels used to be cached together with a global
    #   mean/std that was computed by reading every DICOM. Labels are pure CSV work;
    #   the statistics are gone, so the cache is gone and both SIIM datasets derive
    #   the split from this one function (identical labels -> identical split).
    """
    encoded = label_df[" EncodedPixels"].astype(str)
    has_mask = (encoded != " -1").groupby(label_df["ImageId"]).any()
    return {image_id: int(value) for image_id, value in has_mask.items()}


def siim_manifest_labels(
    manifest_csv: Union[str, Path], label_df: pd.DataFrame
) -> Tuple[List[str], Dict[str, int]]:
    """IDs and labels of a frozen ``image_id,label`` cohort, verified against the RLE CSV.

    The frozen manifest defines membership and label order; the training CSV is used
    only as an order-independent rule lookup (``siim_labels_from_rle``), so reordering
    ``train-rle.csv`` can never change which images or labels are evaluated. A missing
    RLE row, a duplicated manifest ID, a non-binary label or a label that disagrees
    with the RLE rule raises instead of being patched from either source.
    """
    frame = pd.read_csv(manifest_csv, dtype={"image_id": str})
    missing_columns = [c for c in ("image_id", "label") if c not in frame.columns]
    if missing_columns:
        raise RuntimeError(
            f"frozen SIIM cohort {manifest_csv} is missing columns {missing_columns}; "
            "expected 'image_id,label'"
        )
    image_ids = [str(value) for value in frame["image_id"].tolist()]
    duplicated = sorted(set(pd.Series(image_ids)[pd.Series(image_ids).duplicated()]))
    if duplicated:
        raise RuntimeError(
            f"frozen SIIM cohort {manifest_csv} lists an image_id more than once: "
            f"{duplicated[:5]}"
        )
    labels: Dict[str, int] = {}
    invalid = []
    for image_id, value in zip(image_ids, frame["label"].tolist()):
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = float("nan")
        if numeric not in (0.0, 1.0):
            invalid.append((image_id, value))
            continue
        labels[image_id] = int(numeric)
    if invalid:
        raise RuntimeError(
            f"frozen SIIM cohort {manifest_csv} has non-binary labels, e.g. {invalid[:5]}"
        )
    rule_labels = siim_labels_from_rle(label_df)
    missing_ids = sorted(image_id for image_id in image_ids if image_id not in rule_labels)
    if missing_ids:
        raise RuntimeError(
            f"{len(missing_ids)} ImageId(s) of frozen SIIM cohort {manifest_csv} have no "
            f"row in the RLE CSV, e.g. {missing_ids[:5]}"
        )
    mismatched = sorted(
        image_id for image_id in image_ids if rule_labels[image_id] != labels[image_id]
    )
    if mismatched:
        raise RuntimeError(
            f"frozen SIIM cohort {manifest_csv} disagrees with the RLE CSV for "
            f"{len(mismatched)} ImageId(s), e.g. {mismatched[:5]}"
        )
    return image_ids, labels


class ChestXrayDataset(Dataset):
    """
    PyTorch Dataset for SIIM-ACR chest X-ray classification (binary pneumothorax detection).
    """

    def __init__(
        self,
        file_paths: List[str],
        label_dir: str,
        img_size: int = 224,
        split: str = "train",
        transform: Optional[Any] = None,
        test_size: float = 0.1,
        split_seed: int = 42,
        manifest_csv: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize SIIM-ACR chest X-ray classification dataset.

        # B (PTX-27 / Section 5.4): the dataset no longer computes or loads a global
        #   mean/std (the old stats pickle is gone); __getitem__ applies per-image
        #   min-max. # B (PTX-26 / Section 5.3): the default preprocessing is the
        #   shared eval_transform policy. # B (Section 10.3 / Y10): ImageIds resolve
        #   through an exact filename-stem map built once here.
        # B (Section 4.1 / 6.x): ``manifest_csv`` selects a frozen ``image_id,label``
        #   cohort verbatim (membership and order from the manifest, labels checked
        #   against the RLE rule) instead of re-running the author's split. The
        #   internal holdout is evaluated this way, so reordering ``train-rle.csv``
        #   cannot change the cohort.

        Args:
            file_paths (List[str]): List of paths to DICOM files
            label_dir (str): Path to CSV file with labels
            img_size (int): Target image size for resizing
            split (str): Dataset split ('train' or 'test'); ignored when
                ``manifest_csv`` is given
            transform (Optional[Any]): Image transformations to apply
            test_size (float): Fraction of data for test split
            split_seed (int): Random seed for train/test split
            manifest_csv (Optional[Union[str, Path]]): frozen ``image_id,label``
                cohort that defines membership instead of a fresh split
        """
        self.file_paths = file_paths
        self.file_map, self.duplicate_ids = index_dicom_files(file_paths)
        label_df = pd.read_csv(label_dir)
        if manifest_csv is not None:
            self.image_ids, self.labels = siim_manifest_labels(
                manifest_csv, label_df
            )
        else:
            image_id_arr = label_df["ImageId"].unique()
            self.image_ids = list(image_id_arr)
            self.labels = self.process_labels(label_df)

            indices = np.arange(len(self.image_ids))
            labels = [self.labels[i] for i in self.image_ids]
            train_idx, test_idx = train_test_split(
                indices,
                stratify=labels,
                test_size=test_size,
                random_state=split_seed,
            )
            if split == "train":
                indices = train_idx
            else:
                indices = test_idx
            self.image_ids = list(np.array(self.image_ids)[indices])
            self.labels = {i: self.labels[i] for i in self.image_ids}
        require_unique_files(self.image_ids, self.file_map, self.duplicate_ids)
        pos, neg = 0, 0
        for val in self.labels.values():
            if val == 1:
                pos += 1
            else:
                neg += 1
        self.class_count = {1: pos, 0: neg}
        logger.info(f"Class counts: {self.class_count}")
        # Transformations: one resize policy (PTX-26), per-image min-max (PTX-27)
        self.preprocess = eval_transform(img_size)
        self.transform = transform
        logger.info(f"Initialzed dataset of size {len(self.labels)}")

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        image_id = self.image_ids[idx]
        # Load the DICOM file (exact ImageId == filename stem, Section 10.3 / Y10)
        dicom_path = self.file_map[image_id]
        image, pid = self.load_dicom(dicom_path)

        # Apply transformations
        image = minmax_per_image(image)
        if self.transform:
            image = self.transform(image)
        else:
            image = self.preprocess(image)

        # image = torch.tensor(image, dtype=torch.float32) # Add channel dimension
        image = image.float()
        label = torch.tensor(self.labels[image_id], dtype=torch.long)
        return {"data": image, "label": label, "pid": image_id, "image_id": image_id}

    def get_weights(self) -> List[float]:
        """
        Calculate class weights for balanced sampling.

        Returns:
            List[float]: List of weights for each sample based on class frequency
        """
        weights = []
        for val in self.labels.values():
            weights.append(1 / self.class_count[val])
        return weights

    def process_labels(self, df: pd.DataFrame) -> Dict[str, int]:
        """
        Process labels from CSV file (no dataset statistics are computed).

        # B (PTX-27 / Section 5.4): the old version also read every DICOM to average
        #   per-image mean/std into a "global" statistic. That statistic is no longer
        #   used, so labels are pure CSV work here.

        Args:
            df (pd.DataFrame): DataFrame containing image IDs and RLE encodings

        Returns:
            Dict[str, int]: Labels keyed by image ID
        """
        return siim_labels_from_rle(df)

    @staticmethod
    def load_dicom(file_path: str) -> Tuple[np.ndarray, Optional[str]]:
        """
        Load a DICOM file and extract the pixel array.

        Args:
            file_path (str): Path to the DICOM file

        Returns:
            Tuple[np.ndarray, Optional[str]]: Pixel data array and patient ID
        """
        dicom = pydicom.dcmread(file_path)
        metadata = dicom.file_meta
        patient_id = dicom.get("PatientID", None)
        image = dicom.pixel_array  # Get the raw pixel data
        return image, patient_id

    def visualize_as_jpg(self, idx: int, output_path: str) -> None:
        """
        Save the specified DICOM image as a JPG for visualization.

        Args:
            idx (int): Index of the image in the dataset
            output_path (str): Path to save the JPG file
        """
        # Load and preprocess the image (exact file map, per-image min-max)
        image_id = self.image_ids[idx]
        dicom_path = self.file_map[image_id]
        image, _ = self.load_dicom(dicom_path)
        image = minmax_per_image(image)
        image = self.preprocess(image).squeeze(0).cpu().numpy()

        # Convert to 8-bit image (range [0, 255])
        image_8bit = (image * 255).astype(np.uint8)

        # Save as JPG using Pillow
        image_pil = Image.fromarray(image_8bit)
        image_pil = image_pil.resize((1024, 1024))
        image_pil.save(output_path)
        logger.info(f"Image saved as JPG at: {output_path}")

    def visualize_segmentation(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        alpha: float = 0.5,
        mask_color: Tuple[float, float, float] = (0, 0, 1),
    ) -> np.ndarray:
        """
        Visualize a segmentation mask overlaid on an image.

        Args:
            image (np.ndarray): The base image array
            mask (np.ndarray): The segmentation mask array
            alpha (float): Transparency of the mask overlay (0-1)
            mask_color (Tuple[float, float, float]): RGB color of the mask overlay

        Returns:
            np.ndarray: Image with overlaid segmentation mask
        """
        # Ensure the image and mask have the same dimensions
        if image.shape[:2] != mask.shape:
            raise ValueError("Image and mask must have the same height and width.")
        image = np.array(image * 255).astype(np.uint8)
        # Convert binary mask to RGB if necessary
        if len(mask.shape) == 2:  # Single-channel binary or multi-class mask
            colored_mask = np.zeros_like(image, dtype=np.uint8)
            for c in range(3):
                colored_mask[..., c] = mask * mask_color[c]  # Apply mask color
        else:
            colored_mask = mask  # Assume already RGB for multi-class

        # Overlay the mask on the image
        overlaid_image = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)
        return overlaid_image


if __name__ == "__main__":
    DATA_DIR = cfg.ptx.siim_acr.root_dir
    train_dicom_dir = os.path.join(DATA_DIR, "dicom-images-train")
    test_dicom_dir = os.path.join(DATA_DIR, "dicom-images-test")
    csv_path = os.path.join(DATA_DIR, "train-rle.csv")
    train_glob = f"{train_dicom_dir}/*/*/*.dcm"
    test_glob = f"{test_dicom_dir}/*/*/*.dcm"
    train_names = [f for f in sorted(glob.glob(train_glob))]
    test_names = [f for f in sorted(glob.glob(test_glob))]
    logger.debug(f"Home dir: {HOME_DIR}")
    if not os.path.exists(f"{HOME_DIR}/output/samples"):
        os.mkdir(f"{HOME_DIR}/output/samples")
    if not os.path.exists(f"{HOME_DIR}/output/masks"):
        os.mkdir(f"{HOME_DIR}/output/masks")

    img_size = 224
    train_ds = ChestXrayDataset(train_names, csv_path, split="train", img_size=img_size)
    test_ds = ChestXrayDataset(train_names, csv_path, split="test", img_size=img_size)
    logger.debug(f"Len train: {len(train_ds)}, Len Test: {len(test_ds)}")
    ds = test_ds
    train_image_ids = train_ds.image_ids
    test_image_ids = test_ds.image_ids
    logger.debug(
        f"Number of overlapping indices: {len(list(set(train_image_ids) & set(test_image_ids)))}"
    )
    pids = []
    for idx in tqdm(range(len(ds))):
        if idx > 500:
            break
        batch = ds[idx]
        img = batch["data"]
        label = batch["label"]
        logger.debug(f"PID: {batch['pid']}")
        # print(f"Label : {label}")
        # pids.append(batch['pid'])
        if label == 0:
            continue
        ds.visualize_as_jpg(idx, f"/{HOME_DIR}/output/samples/sample_{idx}.jpg")
    logger.debug(f"Number of unique pids: {len(set(pids))}")

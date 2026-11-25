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

HOME_DIR = Path(__file__).resolve().parent.parent

from sharpness_matters.ptx.utils.logging_utils import initialize_logger
from sharpness_matters.ptx.corrected.preprocess import eval_transform, minmax_per_image
from sharpness_matters.ptx.datasets.siim_acr_classification_dataset import (
    index_dicom_files,
    require_unique_files,
    siim_labels_from_rle,
    siim_manifest_labels,
)
from sharpness_matters.config.load_config import cfg

logger = initialize_logger(level="debug")


def mask2rle(img: np.ndarray, width: int, height: int) -> str:
    """
    Convert binary mask to run-length encoding (RLE) format.

    Args:
        img (np.ndarray): Binary mask image
        width (int): Image width
        height (int): Image height

    Returns:
        str: RLE encoded string
    """
    rle = []
    lastColor = 0
    currentPixel = 0
    runStart = -1
    runLength = 0

    for x in range(width):
        for y in range(height):
            currentColor = img[x][y]
            if currentColor != lastColor:
                if currentColor == 255:
                    runStart = currentPixel
                    runLength = 1
                else:
                    rle.append(str(runStart))
                    rle.append(str(runLength))
                    runStart = -1
                    runLength = 0
                    currentPixel = 0
            elif runStart > -1:
                runLength += 1
            lastColor = currentColor
            currentPixel += 1

    return " ".join(rle)


def rle2mask(rle: str, width: int, height: int) -> np.ndarray:
    """
    Convert run-length encoding (RLE) format to binary mask.

    Args:
        rle (str): RLE encoded string
        width (int): Image width
        height (int): Image height

    Returns:
        np.ndarray: Binary mask array
    """
    mask = np.zeros(width * height)
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]
    current_position = 0
    for index, start in enumerate(starts):
        current_position += start
        mask[current_position : current_position + lengths[index]] = 255
        current_position += lengths[index]
    return mask.reshape(width, height)


class ChestXrayDataset(Dataset):
    """
    PyTorch Dataset for SIIM-ACR chest X-ray classification with pneumothorax detection.
    """

    def __init__(
        self,
        file_paths: List[str],
        label_dir: str,
        img_size: int = 224,
        split: str = "test",
        transform: Optional[Any] = None,
        test_size: float = 0.1,
        split_seed: int = 42,
        manifest_csv: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize SIIM-ACR chest X-ray dataset.

        # B (PTX-02 / Section 7.1): the GT mask is no longer resized to the input size;
        #   it stays on the 1024x1024 RLE grid as uint8 {0,1}. # B (PTX-27 / Section
        #   5.4): per-image min-max, no dataset mean/std and no stats pickle. # B
        #   (PTX-26 / Section 5.3): default preprocessing is eval_transform. # B
        #   (Section 7.4 / Y4 and Section 10.3 / Y10): `pid`/`image_id` are the
        #   ImageId and files resolve by exact filename stem.
        # B (Section 4.1 / 6.x): ``manifest_csv`` fixes the cohort to a frozen
        #   ``image_id,label`` list and its order; the RLE CSV is only a
        #   mask/label lookup, so annotation reordering cannot change membership.
        #   check: tests/corrected/test_data.py::test_siim_masked_loader_*,
        #   tests/corrected/test_siim_holdout.py::test_*

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
            self.image_ids, manifest_labels = siim_manifest_labels(
                manifest_csv, label_df
            )
        else:
            image_id_arr = label_df["ImageId"].unique()
            self.image_ids = list(image_id_arr)
            label_lookup = siim_labels_from_rle(label_df)

            labels = [label_lookup[i] for i in self.image_ids]
            indices = np.arange(len(self.image_ids))
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
        require_unique_files(self.image_ids, self.file_map, self.duplicate_ids)
        self.labels = self.process_labels(label_df, self.image_ids)
        if manifest_csv is not None:
            mismatched = sorted(
                image_id
                for image_id in self.image_ids
                if self.labels[image_id]["label"] != manifest_labels[image_id]
            )
            if mismatched:
                raise RuntimeError(
                    f"masked SIIM loader disagrees with frozen cohort {manifest_csv} "
                    f"for {len(mismatched)} ImageId(s), e.g. {mismatched[:5]}"
                )
        pos, neg = 0, 0
        for i in self.labels.values():
            val = i["label"]
            if val == 1:
                pos += 1
            else:
                neg += 1
        self.class_count = {1: pos, 0: neg}
        logger.info(f"Class counts: {self.class_count}")
        # Transformations: one resize policy (PTX-26); the mask is never resized
        self.preprocess = eval_transform(img_size)
        self.transform = transform
        logger.info(f"Initialzed dataset of size {len(self.labels)}")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        image_id = self.image_ids[idx]
        # Load the DICOM file (exact ImageId == filename stem, Section 10.3 / Y10)
        dicom_path = self.file_map[image_id]
        image, _ = self.load_dicom(dicom_path)
        # GT mask on the reference annotation grid (1024x1024), uint8 {0,1}: no resize
        mask = torch.from_numpy(self.labels[image_id]["mask"])
        label = torch.tensor(self.labels[image_id]["label"], dtype=torch.long)

        # Apply transformations
        image = minmax_per_image(image)
        if self.transform:
            image = self.transform(image)
        else:
            image = self.preprocess(image)
        image = image.float()
        return {
            "data": image,
            "mask": mask,
            "label": label,
            "pid": image_id,
            "image_id": image_id,
        }

    def process_labels(
        self, df: pd.DataFrame, image_ids: List[str]
    ) -> Dict[str, Dict[str, Union[np.ndarray, int]]]:
        """
        Process masks and labels for the selected image IDs.

        # B (PTX-02 / Section 7.1): masks are built on the 1024x1024 RLE grid and
        #   binarised to uint8 {0,1} (the old version stored 0/255 and let the caller
        #   resize, which eroded thin foreground). Only the split's IDs are parsed;
        #   labels come from the shared CSV rule so the split matches the
        #   classification dataset exactly.

        Args:
            df (pd.DataFrame): DataFrame containing image IDs and RLE encodings
            image_ids (List[str]): Image IDs to build masks for

        Returns:
            Dict[str, Dict[str, Union[np.ndarray, int]]]: image ID -> mask and label
        """
        labels = {}
        records_by_id = {image_id: rows for image_id, rows in df.groupby("ImageId")}
        for image_id in tqdm(image_ids, desc="Processing labels"):
            record_arr = records_by_id.get(image_id)
            if record_arr is None:
                raise RuntimeError(f"No RLE rows for ImageId {image_id}")
            # Visualize patient has multi segment
            label = 0
            mask = np.zeros((1024, 1024), dtype=np.uint8)
            for _, row in record_arr.iterrows():
                if row[" EncodedPixels"] != " -1":
                    mask[rle2mask(row[" EncodedPixels"], 1024, 1024).T == 255] = 1
                    label = 1
            labels[image_id] = {"mask": mask, "label": label}
        return labels

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
        # metadata = dicom.file_meta
        # logger.debug(f"Metadata: {dicom}")
        patient_id = dicom.get("PatientID", None)
        image = dicom.pixel_array  # Get the raw pixel data
        return image, patient_id

    def extract_demographics(
        self, image_id: str
    ) -> Dict[str, Optional[Union[str, int]]]:
        """
        Extract age and sex metadata from a DICOM file.

        Args:
            image_id (str): The image ID to extract demographics for

        Returns:
            Dict[str, Optional[Union[str, int]]]: Dictionary containing age, sex, and patient_id
        """
        # Find the file path for this image_id (exact filename stem, Section 10.3)
        dicom_path = self.file_map.get(image_id)
        if dicom_path is None:
            logger.error(f"Could not find file path for image_id: {image_id}")
            return {"image_id": image_id, "age": None, "sex": None, "patient_id": None}

        try:
            dicom = pydicom.dcmread(dicom_path)

            age = dicom.get("PatientAge", None)
            sex = dicom.get("PatientSex", None)
            patient_id = dicom.get("PatientID", None)

            # Clean age data - handle various age formats
            if age is not None:
                try:
                    if isinstance(age, str):
                        # Remove common suffixes and clean the string
                        age_str = age.strip().upper()
                        if age_str.endswith("Y"):
                            age_str = age_str[:-1]
                        elif age_str.endswith("YEARS"):
                            age_str = age_str[:-5]
                        elif age_str.endswith("YRS"):
                            age_str = age_str[:-3]

                        # Try to extract numeric part
                        import re

                        age_match = re.search(r"\d+", age_str)
                        if age_match:
                            age = int(age_match.group())
                        else:
                            age = None
                    else:
                        # Try to convert directly to int
                        age = int(age)

                    # Validate age range (reasonable human age)
                    if age is not None and (age < 0 or age > 120):
                        age = None

                except (ValueError, TypeError, AttributeError):
                    age = None

            # Clean sex data - standardize to M/F
            if sex and isinstance(sex, str):
                sex = sex.upper().strip()
                if sex not in ["M", "F"]:
                    sex = None

            return {
                "image_id": image_id,
                "age": age,
                "sex": sex,
                "patient_id": patient_id,
            }
        except Exception as e:
            logger.error(f"Error extracting demographics for {image_id}: {e}")
            return {"image_id": image_id, "age": None, "sex": None, "patient_id": None}

    def get_all_demographics(self) -> List[Dict[str, Optional[Union[str, int]]]]:
        """
        Extract demographics for all images in the dataset.

        Returns:
            List[Dict[str, Optional[Union[str, int]]]]: List of dictionaries containing demographics for each image
        """
        demographics = []
        for image_id in tqdm(self.image_ids, desc="Extracting demographics"):
            demo_data = self.extract_demographics(image_id)
            demo_data["label"] = self.labels[image_id][
                "label"
            ]  # Add the pneumothorax label
            demographics.append(demo_data)
        return demographics

    def visualize_as_jpg(self, image: torch.Tensor, idx: int, output_path: str) -> None:
        """
        Save the specified DICOM image as a JPG for visualization.

        Args:
            image (torch.Tensor): Image tensor to save
            idx (int): Index of the image in the dataset
            output_path (str): Path to save the JPG file
        """
        # Convert to 8-bit image (range [0, 255])
        image = image.numpy()
        image_8bit = (image * 255).astype(np.uint8)

        # Save as JPG using Pillow
        image_pil = Image.fromarray(image_8bit, "L")
        image_pil.save(output_path)
        logger.info(f"Image saved as JPG at: {output_path}")

    def visualize_segmentation(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        alpha: float = 0.5,
        mask_color: Tuple[float, float, float] = (1, 0, 0),
    ) -> np.ndarray:
        """
        Visualize a segmentation mask overlaid on an image.

        Args:
            image (torch.Tensor): The base image tensor
            mask (torch.Tensor): The segmentation mask tensor
            alpha (float): Transparency of the mask overlay (0-1)
            mask_color (Tuple[float, float, float]): RGB color of the mask overlay

        Returns:
            np.ndarray: Image with overlaid segmentation mask
        """
        # Ensure the image and mask have the same dimensions
        image = cv2.resize(
            image.numpy(),
            dsize=(mask.shape[0], mask.shape[1]),
            interpolation=cv2.INTER_CUBIC,
        )
        if image.shape[:2] != mask.shape:
            raise ValueError("Image and mask must have the same height and width.")
        image = np.array(image * 255).astype(np.uint8)
        # Convert binary mask to RGB if necessary. GT masks are uint8 {0,1}
        # (PTX-02 / Section 7.1), so the foreground is scaled to 255 for display;
        # a 0/255 mask (e.g. the thresholded CAM passed by the scorer) is unchanged.
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


if __name__ == "__main__":
    DATA_DIR = cfg.ptx.siim_acr.root_dir
    train_dicom_dir = os.path.join(DATA_DIR, "dicom-images-train")
    test_dicom_dir = os.path.join(DATA_DIR, "dicom-images-test")
    csv_path = os.path.join(DATA_DIR, "train-rle.csv")
    train_glob = f"{train_dicom_dir}/*/*/*.dcm"
    test_glob = f"{test_dicom_dir}/*/*/*.dcm"
    train_names = [f for f in sorted(glob.glob(train_glob))]
    test_names = [f for f in sorted(glob.glob(test_glob))]
    if not os.path.exists(f"{HOME_DIR}/output/samples"):
        os.mkdir(f"{HOME_DIR}/output/samples")
    if not os.path.exists(f"{HOME_DIR}/output/masks"):
        os.mkdir(f"{HOME_DIR}/output/masks")
    img_size = 1024
    ds = ChestXrayDataset(train_names, csv_path, img_size=img_size)
    for idx in range(500):
        batch = ds[idx]
        img = batch["data"]
        label = batch["label"]
        mask = batch["mask"]
        pid = batch["pid"]
        if label == 0:
            continue
        if len(mask.shape) > 1:
            ds.visualize_as_jpg(
                img.squeeze(0), idx, f"{HOME_DIR}/output/samples/{pid}.jpg"
            )
            seg = ds.visualize_segmentation(
                img.permute(1, 2, 0).expand(*img.shape[1:], 3), mask
            )
            Image.fromarray(seg).save(f"{HOME_DIR}/output/masks/{pid}.jpg")
            logger.debug(f"Saved {pid} to {HOME_DIR}/output/masks/{pid}.jpg")

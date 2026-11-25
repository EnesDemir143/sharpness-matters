import os
from pathlib import Path
from typing import Optional, Tuple

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
from pydicom.pixel_data_handlers.util import apply_voi_lut

from sharpness_matters.ptx.utils.logging_utils import initialize_logger
from sharpness_matters.ptx.corrected.preprocess import eval_transform, minmax_per_image
from sharpness_matters.config.load_config import cfg

logger = initialize_logger(level="debug")
HOME = Path(__file__).resolve().parent.parent


class VinDataset(Dataset):
    def __init__(
        self,
        manifest_csv: str = None,
        root_dir: str = cfg.ptx["vindr"]["root_dir"],
        img_size: int = 224,
        transform=None,
    ):
        """
        Initialize the VinDr cohort from a frozen ID+label manifest.

        # B (PTX-03 / PTX-08, Section 4.2): the author read vin500neg.txt +
        #   vin500pos.txt, merged them without deduplication and labelled by POS
        #   membership, so an ID in both lists counted as a positive twice. New:
        #   IDs and labels come from a manifest csv (`image_id,label`) in file order,
        #   with no deduplication (E5 and E5-clean are separate manifests).
        # B (PTX-27 / Section 5.4): per-image min-max, no dataset mean/std.
        # B (PTX-26 / Section 5.3): default preprocessing is eval_transform.
        #   check: tests/corrected/test_data.py::test_vin_manifest_order_labels

        Args:
            manifest_csv (str): Manifest with `image_id,label` (required)
            root_dir (str): Directory containing `<image_id>.dicom`
            img_size (int): Target image size for resizing
            transform: Explicit preprocessing; defaults to eval_transform
        """
        if not manifest_csv:
            raise RuntimeError(
                "VinDataset in stage 2 reads an ID+label manifest and no longer reads "
                "the author's .txt lists. Pass manifest_csv="
                "manifests/reference_like_combined.csv (E5) or "
                "manifests/reference_like_clean.csv (E5-clean)."
            )
        self.root_dir = root_dir
        manifest = pd.read_csv(manifest_csv, dtype={"image_id": str})
        missing = {"image_id", "label"} - set(manifest.columns)
        if missing:
            raise KeyError(f"{manifest_csv} is missing columns {sorted(missing)}")
        self.image_ids = [str(image_id) for image_id in manifest["image_id"]]
        self.labels = {}
        for image_id, label in zip(manifest["image_id"], manifest["label"]):
            key = str(image_id)
            value = int(label)
            if key in self.labels and self.labels[key] != value:
                raise RuntimeError(
                    f"{manifest_csv} gives image_id {key} conflicting labels "
                    f"({self.labels[key]} and {value}); a manifest must label each "
                    "image consistently"
                )
            self.labels[key] = value
        duplicate_rows = len(self.image_ids) - len(self.labels)
        if duplicate_rows:
            # The E5 reference-like manifest keeps an ID that the author's lists had
            # in both NEG and POS (Section 4.2, author semantics: counted twice, both
            # rows positive). The rows stay; nothing is deduplicated here.
            logger.info(
                f"{manifest_csv}: {duplicate_rows} duplicate image_id row(s) kept, "
                "as in the author loader"
            )
        logger.debug(f"Manifest rows: {len(self.image_ids)}")
        self.transform = (
            transform if transform is not None else eval_transform(img_size)
        )
        logger.info(f"Len of Vin dataset: {len(self.image_ids)}")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> dict:
        image_id = self.image_ids[idx]
        dicom_path = os.path.join(self.root_dir, f"{self.image_ids[idx]}.dicom")
        image, pid = self.load_dicom(dicom_path)
        image = minmax_per_image(image)
        image = self.transform(image)
        return {
            "data": image,
            "label": self.labels[image_id],
            "mask": image,
            "pid": image_id,
            "image_id": image_id,
        }

    @staticmethod
    def load_dicom(file_path: str) -> Tuple[np.ndarray, Optional[str]]:
        """
        Load a DICOM file and decode it to a MONOCHROME2-polarity pixel array.

        Parameters:
        - file_path (str): Path to the DICOM file.

        Returns:
        - tuple: The float32 pixel data with display polarity normalised, and the
          patient ID.
        """
        dicom = pydicom.dcmread(file_path)
        dicom.BitsStored = (
            16  # This was added due to a warning recommending this change
        )
        patient_id = dicom.get("PatientID", None)
        image = dicom.pixel_array  # Get the raw pixel data
        # image = apply_voi_lut(dicom.pixel_array, dicom)
        image = image.astype(np.float32)
        # B: DICOM MONOCHROME1 displays low stored values as white, the opposite of
        #   MONOCHROME2. Negating here (before min-max in __getitem__) restores the
        #   displayed intensities, so an inverted file no longer enters the pipeline
        #   upside down. A constant image stays constant and min-max maps it to 0.
        photometric = dicom.get("PhotometricInterpretation", None)
        if (
            photometric is not None
            and str(photometric).strip().upper() == "MONOCHROME1"
        ):
            image = -image
        return image, patient_id

    def extract_demographics(self, image_id: str) -> dict:
        """
        Extract age and sex metadata from a DICOM file.

        Parameters:
        - image_id (str): The image ID to extract demographics for.

        Returns:
        - dict: Dictionary containing age, sex, and patient_id
        """
        dicom_path = os.path.join(self.root_dir, f"{image_id}.dicom")
        try:
            dicom = pydicom.dcmread(dicom_path)
            dicom.BitsStored = 16

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
            else:
                age = None
                logger.warning(f"Couldn't find age for image {image_id}")

            # Clean sex data - standardize to M/F
            if sex and isinstance(sex, str):
                sex = sex.upper().strip()
                if sex not in ["M", "F"]:
                    sex = None
            else:
                logger.warning(f"Couldn't find sex")

            return {
                "image_id": image_id,
                "age": age,
                "sex": sex,
                "patient_id": patient_id,
            }
        except Exception as e:
            logger.error(f"Error extracting demographics for {image_id}: {e}")
            return {"image_id": image_id, "age": None, "sex": None, "patient_id": None}

    def get_all_demographics(self) -> list:
        """
        Extract demographics for all images in the dataset.

        Returns:
        - list: List of dictionaries containing demographics for each image
        """
        demographics = []
        for image_id in tqdm(self.image_ids, desc="Extracting demographics"):
            demo_data = self.extract_demographics(image_id)
            demo_data["label"] = self.labels[image_id]  # Add the pneumothorax label
            demographics.append(demo_data)
        return demographics

    def visualize_as_jpg(self, idx: int, output_path: str):
        """
        Save the specified DICOM image as a JPG for visualization.

        Parameters:
        - idx (int): Index of the image in the dataset.
        - output_path (str): Path to save the JPG file.
        """
        # Load and preprocess the image
        image_id = self.image_ids[idx]
        dicom_path = os.path.join(self.root_dir, f"{self.image_ids[idx]}.dicom")
        image, pid = self.load_dicom(dicom_path)
        image = minmax_per_image(image)

        image = self.transform(image).squeeze(0).numpy()

        # Convert to 8-bit image (range [0, 255])
        image_8bit = (image * 255.0).astype(np.uint8)

        # Save as JPG using Pillow
        image_pil = Image.fromarray(image_8bit)
        image_pil.save(output_path)
        logger.info(f"Image saved as JPG at: {output_path}")


if __name__ == "__main__":
    from sharpness_matters.ptx.corrected.frozen import manifest_dir

    img_size = 1024
    dataset = VinDataset(
        manifest_csv=str(manifest_dir() / "reference_like_combined.csv"),
        img_size=img_size,
    )
    logger.debug(dataset[0]["data"].shape)
    # dataset.visualize_as_jpg(0, '/cnvrg/test.jpg')

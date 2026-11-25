"""CheXlocalize (E4) classification / XAI dataset and COCO RLE decoder.

# B (protocol Section 4.4 E4): CheXlocalize was not evaluated, and a CheXpert
#   label row is not the same cohort as an annotated segmentation. New: a frozen
#   frontal-only classification manifest (CheXpert 0/1 `Pneumothorax`) and a
#   separate XAI manifest (label 1 AND non-empty Pneumothorax mask) are read here;
#   the mask stays on its native grid (uint8 {0,1}), never resized.
# B (protocol Section 4.5 / Section 4.7): a frozen manifest carries `split` and
#   `patient_id`, so val/test/combined counts and patient clustering stay visible.
#   check: tests/corrected/test_chexlocalize.py::test_*
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from sharpness_matters.config.load_config import cfg
from sharpness_matters.ptx.corrected.preprocess import eval_transform, minmax_per_image
from sharpness_matters.ptx.utils.logging_utils import initialize_logger

logger = initialize_logger(level="info")

FINDING = "Pneumothorax"
REQUIRED_MANIFEST_COLUMNS = ("image_id", "split", "patient_id", "path", "label")
SEGMENTATION_FILES = {
    "val": "gt_segmentations_val.json",
    "test": "gt_segmentations_test.json",
}


def _posix_parts(path: Any) -> List[str]:
    """Path components with either separator, empty components dropped."""
    return [part for part in str(path).replace("\\", "/").split("/") if part]


def cxr_id_from_path(path: Any) -> str:
    """CheXlocalize ID: last three components ``patientX_studyY_viewZ_frontal``.

    The CheXpert label CSV ``Path`` values are either bare
    ``patientX/studyY/viewZ_frontal.jpg`` or prefixed with ``CheXpert/v1.0/valid``;
    both forms reduce to the key used by ``gt_segmentations_{val,test}.json``.
    """
    parts = _posix_parts(path)
    if len(parts) < 3:
        raise ValueError(
            f"CheXpert path {path!r} has fewer than three components; expected "
            "patientX/studyY/viewZ_frontal.jpg"
        )
    patient, study, filename = parts[-3:]
    stem = Path(filename).stem
    if not patient or not study or not stem:
        raise ValueError(f"CheXpert path {path!r} has an empty component")
    return f"{patient}_{study}_{stem}"


def view_from_path(path: Any) -> str:
    """``frontal`` or ``lateral`` from the final ``_frontal``/``_lateral`` suffix."""
    parts = _posix_parts(path)
    if not parts:
        raise ValueError(f"Empty CheXpert path: {path!r}")
    stem = Path(parts[-1]).stem
    for view in ("frontal", "lateral"):
        if stem.endswith("_" + view):
            return view
    raise ValueError(
        f"CheXpert path {path!r} does not end in _frontal or _lateral; the view "
        "cannot be verified"
    )


def patient_id_from_path(path: Any) -> str:
    """``patientX`` component of a CheXpert path (the report's patient key)."""
    parts = _posix_parts(path)
    if len(parts) < 3:
        raise ValueError(f"CheXpert path {path!r} has fewer than three components")
    return parts[-3]


def rle_counts(rle: Mapping[str, Any]) -> List[int]:
    """Decode a pycocotools compressed RLE string into run lengths.

    The string is a sequence of 6-bit chunks: ``ord(char) - 48``; bit ``0x20``
    means another chunk follows; bit ``0x10`` on the final chunk sign-extends the
    value. Runs after the first two are delta-coded against the run two positions
    earlier. Counts alternate starting with background (0).
    """
    if not isinstance(rle, Mapping):
        raise TypeError(f"RLE must be a mapping with size/counts, got {type(rle)}")
    try:
        encoded = rle["counts"]
    except KeyError as exc:
        raise KeyError("RLE is missing the 'counts' key") from exc
    if not isinstance(encoded, str):
        raise TypeError(
            "RLE counts must be the pycocotools compressed string; got "
            f"{type(encoded)}"
        )
    counts: List[int] = []
    position = 0
    length = len(encoded)
    while position < length:
        value = 0
        shift = 0
        more = True
        while more:
            if position >= length:
                raise ValueError("truncated compressed RLE string")
            code = ord(encoded[position]) - 48
            position += 1
            if code < 0 or code > 63:
                raise ValueError(
                    f"invalid compressed RLE character {chr(code + 48)!r}: "
                    "6-bit chunks are offset by 48 and must be in [0, 63]"
                )
            value |= (code & 0x1F) << (5 * shift)
            more = bool(code & 0x20)
            shift += 1
            if not more and (code & 0x10):
                value |= -1 << (5 * shift)
        if len(counts) > 2:
            value += counts[-2]
        counts.append(int(value))
    return counts


def _rle_size_and_counts(rle: Mapping[str, Any]) -> Tuple[int, int, List[int]]:
    """Validated ``(height, width, counts)`` for a compressed RLE mask."""
    if not isinstance(rle, Mapping):
        raise TypeError(f"RLE must be a mapping with size/counts, got {type(rle)}")
    try:
        size = rle["size"]
    except KeyError as exc:
        raise KeyError("RLE is missing the 'size' key") from exc
    if len(size) != 2:
        raise ValueError(f"RLE size must be [height, width], got {size!r}")
    height, width = int(size[0]), int(size[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"RLE size must be positive, got {size!r}")
    counts = rle_counts(rle)
    if any(count < 0 for count in counts):
        raise ValueError(f"RLE has a negative run length: {counts[:5]}")
    total = height * width
    if sum(counts) != total:
        raise ValueError(
            f"RLE runs sum to {sum(counts)} but size {height}x{width} needs {total}"
        )
    return height, width, counts


def rle_foreground_pixels(rle: Mapping[str, Any]) -> int:
    """Number of foreground pixels without materialising the mask.

    Runs alternate background/foreground starting with background, so the odd
    runs are the foreground. Used to select the XAI cohort while keeping the
    decision identical to decoding the mask and counting ``mask.sum()``.
    """
    _, _, counts = _rle_size_and_counts(rle)
    return int(sum(counts[1::2]))


def rle_decode(rle: Mapping[str, Any]) -> np.ndarray:
    """Decode a COCO/pycocotools compressed RLE mask to uint8 ``{0,1}``.

    ``rle = {"size": [height, width], "counts": "<compressed>"}``. The runs fill
    column-major (Fortran) order: linear index ``p`` is row ``p % height`` and
    column ``p // height``.
    """
    height, width, counts = _rle_size_and_counts(rle)
    flat = np.empty(height * width, dtype=np.uint8)
    offset = 0
    value = 0
    for count in counts:
        if count:
            flat[offset : offset + count] = value
            offset += count
        value ^= 1
    if offset != height * width:  # _rle_size_and_counts already guarantees this
        raise RuntimeError("RLE run lengths do not cover the mask")
    return flat.reshape((height, width), order="F")


class CheXlocalizeDataset(Dataset):
    """CheXlocalize images with optional native-grid Pneumothorax masks.

    # B (protocol Section 4.4 E4): current -> the author had no CheXlocalize
    #   loader; problem -> E4 could not be measured; new -> images come from the
    #   frozen cls/xai manifest (manifest order), are converted to 8-bit grayscale,
    #   min-max scaled per image and resized only by the shared eval_transform.
    #   With ``with_mask=True`` the decoded GT stays on the native image grid; a
    #   size mismatch raises instead of resizing the mask.
    #   check: tests/corrected/test_chexlocalize.py::test_dataset_*

    Manifest columns: ``image_id, split, patient_id, path, label``. A relative
    ``path`` is resolved against ``chexpert_dir``; ``pid`` is ``patient_id`` and
    ``image_id`` is the CheXlocalize key.
    """

    def __init__(
        self,
        manifest_csv: str,
        img_size: int,
        with_mask: bool = False,
        transform: Optional[Any] = None,
        annotations_dir: Optional[str] = None,
        chexpert_dir: Optional[str] = None,
    ):
        self.manifest_csv = str(manifest_csv)
        self.img_size = int(img_size)
        self.with_mask = bool(with_mask)
        self.annotations_dir = str(
            annotations_dir
            if annotations_dir is not None
            else cfg.ptx.chexlocalize.annotations_dir
        )
        self.chexpert_dir = str(
            chexpert_dir
            if chexpert_dir is not None
            else cfg.ptx.chexlocalize.chexpert_dir
        )
        manifest = pd.read_csv(
            self.manifest_csv,
            dtype={
                "image_id": str,
                "split": str,
                "patient_id": str,
                "path": str,
            },
        )
        missing = set(REQUIRED_MANIFEST_COLUMNS) - set(manifest.columns)
        if missing:
            raise KeyError(f"{self.manifest_csv} is missing columns {sorted(missing)}")
        if len(manifest) == 0:
            raise ValueError(f"{self.manifest_csv} lists no images")

        self.image_ids: List[str] = []
        self.labels: Dict[str, int] = {}
        self.splits: Dict[str, str] = {}
        self.patient_ids: Dict[str, str] = {}
        self.paths: List[str] = []
        for image_id, split, patient_id, path, label in zip(
            manifest["image_id"],
            manifest["split"],
            manifest["patient_id"],
            manifest["path"],
            manifest["label"],
        ):
            key = str(image_id)
            if key in self.labels:
                raise RuntimeError(
                    f"{self.manifest_csv} lists image_id {key} more than once; a "
                    "frozen manifest must contain each image exactly once"
                )
            value = int(label)
            if value not in (0, 1):
                raise RuntimeError(
                    f"{self.manifest_csv} gives image_id {key} label {value}; "
                    "CheXlocalize manifests carry only verified 0/1 labels"
                )
            self.image_ids.append(key)
            self.labels[key] = value
            self.splits[key] = str(split)
            self.patient_ids[key] = str(patient_id)
            self.paths.append(str(path))

        self.transform = (
            transform if transform is not None else eval_transform(self.img_size)
        )
        self._segmentations: Dict[str, Dict[str, Any]] = {}
        if self.with_mask:
            for split in sorted(set(self.splits.values())):
                self._segmentations[split] = self._load_segmentations(split)
        logger.info(f"Len of CheXlocalize dataset: {len(self.image_ids)}")

    def _load_segmentations(self, split: str) -> Dict[str, Any]:
        """Read one ``gt_segmentations_{split}.json`` (val/test only)."""
        if split not in SEGMENTATION_FILES:
            raise ValueError(
                f"Unknown CheXlocalize split {split!r}; expected one of "
                f"{sorted(SEGMENTATION_FILES)}"
            )
        path = Path(self.annotations_dir) / SEGMENTATION_FILES[split]
        if not path.exists():
            raise FileNotFoundError(f"CheXlocalize segmentations not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise RuntimeError(f"{path} must map image_id -> finding -> RLE")
        return data

    def _mask_for(self, image_id: str, split: str, native_shape: Tuple[int, int]) -> np.ndarray:
        """Decode the Pneumothorax GT and require the image's native grid."""
        entries = self._segmentations[split]
        entry = entries.get(image_id)
        if entry is None:
            raise RuntimeError(
                f"{image_id} has no entry in gt_segmentations_{split}.json"
            )
        if FINDING not in entry:
            raise RuntimeError(
                f"{image_id} has no {FINDING} mask in gt_segmentations_{split}.json"
            )
        mask = rle_decode(entry[FINDING])
        if mask.shape != tuple(native_shape):
            raise RuntimeError(
                f"{image_id}: {FINDING} GT grid {mask.shape} does not match the "
                f"image grid {tuple(native_shape)}; the mask is never resized"
            )
        return mask

    def visualize_segmentation(
        self,
        image: Union[np.ndarray, torch.Tensor],
        mask: Union[np.ndarray, torch.Tensor],
        alpha: float = 0.5,
        mask_color: Tuple[float, float, float] = (1, 0, 0),
    ) -> np.ndarray:
        """Overlay a binary mask on the image at the mask's full resolution.

        # B (protocol Section 4.4 E4): current -> the CheXlocalize loader had no
        #   `visualize_segmentation`, so the scorer's default `save_maps=True` crashed
        #   in `_save_map_visualisations` before writing any map; problem -> E4 visuals
        #   could only be produced by suppressing them; new -> the same author
        #   convention as the SIIM/PTX-498 loaders: the ``[0, 1]`` HWC RGB image is
        #   resized to the mask grid with cubic interpolation (so a native, non-square
        #   mask is rendered at full resolution), a 2-D mask becomes ``mask_color``
        #   where it is positive, and `cv2.addWeighted` returns a uint8 RGB overlay.
        #   The scorer passes a torch ``(H, W, 3)`` display-grid image and a 2-D
        #   ``0/255`` thresholded map; numpy arrays are accepted too.
        #   check: tests/corrected/test_chexlocalize.py::test_save_maps_*

        Args:
            image: Base image ``(H, W, 3)`` in ``[0, 1]``; torch tensor or numpy array
            mask: ``(h, w)`` binary mask (`{0, 1}` GT or `0/255` CAM) or an
                ``(h, w, 3)`` RGB mask; torch tensor or numpy array
            alpha: Opacity of the mask overlay in ``[0, 1]``
            mask_color: RGB colour applied to a binary mask's foreground

        Returns:
            np.ndarray: uint8 ``(h, w, 3)`` RGB image with the mask overlaid on the
            mask's grid.
        """
        if torch.is_tensor(image):
            image = image.detach().cpu().numpy()
        if torch.is_tensor(mask):
            mask = mask.detach().cpu().numpy()
        image = np.asarray(image)
        mask = np.asarray(mask)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"image must be (H, W, 3) RGB, got shape {image.shape}")
        if mask.ndim not in (2, 3):
            raise ValueError(f"mask must be (H, W) or (H, W, 3), got {mask.shape}")
        if mask.ndim == 3 and mask.shape[2] != 3:
            raise ValueError(f"an RGB mask must be (H, W, 3), got {mask.shape}")
        if image.shape[:2] != mask.shape[:2]:
            # Display-grid image -> mask grid; a no-op for the scorer's equal grids,
            # the native-resolution path for a mask on the image's original grid.
            image = cv2.resize(
                np.ascontiguousarray(image),
                dsize=(int(mask.shape[1]), int(mask.shape[0])),
                interpolation=cv2.INTER_CUBIC,
            )
            # INTER_CUBIC can ring past the [0, 1] range at sharp edges; clamping
            # keeps a cast to uint8 from wrapping highlights and shadows.
            image = np.clip(image, 0.0, 1.0)
        image = (np.asarray(image) * 255).astype(np.uint8)
        if mask.ndim == 2:
            foreground = np.where(mask > 0, 255, 0).astype(np.uint8)
            colored_mask = np.zeros_like(image, dtype=np.uint8)
            for channel in range(3):
                colored_mask[..., channel] = foreground * mask_color[channel]
        else:
            colored_mask = np.asarray(mask, dtype=np.uint8)
        return cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        image_id = self.image_ids[index]
        image_path = Path(self.paths[index])
        if not image_path.is_absolute():
            image_path = Path(self.chexpert_dir) / image_path
        with Image.open(image_path) as image_file:
            # CheXpert JPGs are 8-bit; convert("L") gives one grayscale plane.
            array = np.asarray(image_file.convert("L"), dtype=np.float32)
        if array.ndim != 2:
            raise RuntimeError(
                f"{image_id}: expected a grayscale image, got shape {array.shape}"
            )
        native_shape = (int(array.shape[0]), int(array.shape[1]))
        array = minmax_per_image(array)
        data = self.transform(array)
        item: Dict[str, Any] = {
            "data": data,
            "label": int(self.labels[image_id]),
            "pid": self.patient_ids[image_id],
            "image_id": image_id,
            "split": self.splits[image_id],
        }
        if self.with_mask:
            item["mask"] = self._mask_for(image_id, self.splits[image_id], native_shape)
        return item

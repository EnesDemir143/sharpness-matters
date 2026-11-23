"""Stage-1 (A) loaders for the added external rows (E2, E2-wide, E4).

The author's method is unchanged: VinDr uses the author's `VinDataset` preprocessing
(global z-normalisation, then per-image min-max, `ToTensor` -> `Resize(antialias=True)`)
and the same model input pipeline; CheXlocalize images use the author's preprocessing
chain (z-normalisation, then per-image min-max) with the author's resize setting.
Only the ID list comes from a frozen manifest instead of `output/vin500*.txt`.

Manifests are produced once on `stage2-corrected` (B); this module only reads them and
verifies the recorded SHA-256. It never regenerates a manifest.

The COCO RLE decoder below is copied verbatim from
`stage2-corrected:sharpness_matters/ptx/datasets/chexlocalize_dataset.py` (verified
against pycocotools on the real CheXlocalize masks). Decoding is data reading, not
preprocessing.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from sharpness_matters.config.load_config import cfg
from sharpness_matters.ptx.datasets.vin_dataset import VinDataset
from sharpness_matters.ptx.utils.logging_utils import initialize_logger

logger = initialize_logger(level="debug")
HOME = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------- manifest access
def manifest_path(name: str) -> Path:
    """Frozen manifest produced on `stage2-corrected` (B). Read-only."""
    return Path(str(cfg.ptx.manifests_dir)) / name


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(name: str) -> pd.DataFrame:
    """Read a frozen manifest and verify its recorded SHA-256.

    A missing manifest or a hash mismatch raises: run A never regenerates or edits a
    manifest (manifests are frozen once on B).
    """
    path = manifest_path(name)
    hash_path = Path(f"{path}.sha256")
    if not path.is_file() or not hash_path.is_file():
        raise FileNotFoundError(
            f"frozen manifest or its .sha256 not found: {path}. Freeze it on "
            "stage2-corrected (B) and point ptx.manifests_dir at that manifests/ "
            "directory; run A only reads manifests."
        )
    recorded = hash_path.read_text().split()[0].strip().lower()
    digest = _sha256(path)
    if digest != recorded:
        raise RuntimeError(
            f"manifest hash mismatch for {path}: recorded {recorded}, file {digest}. "
            "Refusing to evaluate against an edited manifest."
        )
    frame = pd.read_csv(path, dtype={"image_id": str})
    frame["image_id"] = frame["image_id"].astype(str)
    return frame


# ---------------------------------------------------------------- COCO RLE (verbatim from B)
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


# ---------------------------------------------------------------- VinDr (E2/E2-wide)
class ManifestVinDataset(VinDataset):
    """Author VinDataset preprocessing, ID list from a frozen manifest.

    `__init__` is replaced (the author reads `output/vin500*.txt`); `__getitem__`,
    `z_normalize`, `load_dicom` and the transform (including `antialias=True`) are the
    author's, unchanged.
    """

    def __init__(
        self,
        manifest_csv: str,
        root_dir: Optional[str] = None,
        img_size: int = 512,
        stats_key: str = "vindr",
    ):
        frame = pd.read_csv(manifest_csv, dtype={"image_id": str})
        if "image_id" not in frame.columns or "label" not in frame.columns:
            raise RuntimeError(f"{manifest_csv} must hold image_id,label columns")
        if frame["image_id"].duplicated().any():
            raise RuntimeError(f"{manifest_csv} lists an image_id more than once")
        self.root_dir = (
            str(cfg.ptx["vindr"]["root_dir"]) if root_dir is None else str(root_dir)
        )
        self.image_ids = [str(value) for value in frame["image_id"]]
        self.labels = dict(zip(self.image_ids, [int(value) for value in frame["label"]]))
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((img_size, img_size), antialias=True),
            ]
        )
        stats_path = f"{HOME}/output/cache/vin_stats_{stats_key}.pkl"
        # The author's loader assumes the cache directory already exists (it is created
        # by the label cache on the SIIM path); a fresh checkout has no output/cache.
        os.makedirs(os.path.dirname(stats_path), exist_ok=True)
        self.mean, self.std = self.get_global_mean_std(stats_path)
        logger.info(f"Len of {stats_key} Vin dataset: {len(self.image_ids)}")


# ---------------------------------------------------------------- CheXlocalize (E4)
class CheXlocalizeEvalDataset(Dataset):
    """Author preprocessing chain on CheXpert JPGs, IDs/labels from a frozen manifest.

    Preprocessing (author's chain, unchanged): grayscale float -> z-normalisation ->
    per-image min-max -> `ToTensor` -> `Resize((s, s), antialias=True)`. The z step is
    affine, so the following per-image min-max cancels it up to float rounding; the
    chain is kept because the author's code keeps it.

    With ``with_mask=True`` the Pneumothorax GT is decoded on its native grid and then
    resized to the input size and scaled by 255, which is the author's mask convention
    (the author's scorer divides by 255 and casts to uint8).
    """

    def __init__(
        self,
        manifest_csv: str,
        chexpert_dir: Optional[str] = None,
        img_size: int = 512,
        with_mask: bool = False,
    ):
        frame = pd.read_csv(manifest_csv, dtype={"image_id": str})
        required = {"image_id", "path", "label"}
        missing = required - set(frame.columns)
        if missing:
            raise RuntimeError(f"{manifest_csv} is missing columns {sorted(missing)}")
        self.root_dir = Path(
            str(cfg.ptx.chexlocalize.chexpert_dir)
            if chexpert_dir is None
            else str(chexpert_dir)
        )
        self.image_ids = [str(value) for value in frame["image_id"]]
        self.paths = [str(value) for value in frame["path"]]
        self.labels = dict(zip(self.image_ids, [int(v) for v in frame["label"]]))
        self.splits = dict(
            zip(self.image_ids, [str(v) for v in frame.get("split", [""] * len(frame))])
        )
        self.with_mask = bool(with_mask)
        self.img_size = int(img_size)
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((img_size, img_size), antialias=True),
            ]
        )
        self.mask_resize = transforms.Resize((img_size, img_size), antialias=True)
        if self.with_mask:
            self._masks = self._load_masks()

    def _resolve(self, relative: str) -> Path:
        candidate = Path(relative)
        if candidate.is_file():
            return candidate
        for prefix in ("", "CheXpert-v1.0/valid/", "CheXpert-v1.0/test/"):
            path = (self.root_dir / prefix / relative) if prefix else (self.root_dir / relative)
            if path.is_file():
                return path
        raise FileNotFoundError(
            f"CheXpert image not found for {relative} under {self.root_dir}"
        )

    def _load_masks(self) -> Dict[str, np.ndarray]:
        gt_dir = Path(str(cfg.ptx.chexlocalize.annotations_dir))
        masks: Dict[str, np.ndarray] = {}
        for split in sorted({value for value in self.splits.values() if value}):
            candidate = gt_dir / f"gt_segmentations_{split}.json"
            if not candidate.is_file():
                continue
            segmentations = json.loads(candidate.read_text())
            for cxr_id, tasks in segmentations.items():
                if "Pneumothorax" in tasks:
                    masks[cxr_id] = rle_decode(tasks["Pneumothorax"])
        missing = [image_id for image_id in self.image_ids if image_id not in masks]
        if missing:
            raise RuntimeError(
                f"{len(missing)} CheXlocalize image(s) have no Pneumothorax "
                f"segmentation, e.g. {missing[:5]}"
            )
        return {image_id: masks[image_id] for image_id in self.image_ids}

    def __len__(self) -> int:
        return len(self.image_ids)

    @staticmethod
    def _z_normalize_minmax(xray_array: np.ndarray) -> np.ndarray:
        """The author's two steps (masked-loader semantics): z, then per-image min-max."""
        std = np.std(xray_array)
        if std == 0:  # the author returns the input unchanged when std == 0
            return xray_array
        normalized = (xray_array - np.mean(xray_array)) / std
        min_val, max_val = np.min(normalized), np.max(normalized)
        if max_val == min_val:
            return xray_array
        return (normalized - min_val) / (max_val - min_val)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        image_id = self.image_ids[idx]
        with Image.open(self._resolve(self.paths[idx])) as handle:
            image = np.asarray(handle.convert("L"), dtype=np.float32)
        image = self._z_normalize_minmax(image)
        image = self.transform(image).float()
        item: Dict[str, object] = {
            "data": image,
            "label": self.labels[image_id],
            "pid": image_id,
            "image_id": image_id,
        }
        if self.with_mask:
            mask = torch.tensor(self._masks[image_id], dtype=torch.float32)
            mask = self.mask_resize(mask.unsqueeze(0)).squeeze(0)
            item["mask"] = mask * 255.0  # the author's convention; the scorer /255s
        return item

    def visualize_segmentation(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        alpha: float = 0.5,
        mask_color: Tuple[float, float, float] = (1, 0, 0),
    ) -> np.ndarray:
        """The author's visualisation helper (`siim_acr_dataset.visualize_segmentation`)."""
        image = cv2.resize(
            image.numpy(),
            dsize=(mask.shape[0], mask.shape[1]),
            interpolation=cv2.INTER_CUBIC,
        )
        if image.shape[:2] != mask.shape:
            raise ValueError("Image and mask must have the same height and width.")
        image = np.array(image * 255).astype(np.uint8)
        if len(mask.shape) == 2:
            colored_mask = np.zeros_like(image, dtype=np.uint8)
            for c in range(3):
                colored_mask[..., c] = mask * mask_color[c]
        else:
            colored_mask = mask
        return cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)

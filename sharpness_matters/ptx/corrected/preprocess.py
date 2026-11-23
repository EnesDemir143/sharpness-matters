"""Stage-2 (B) preprocessing policy: per-image min-max and one resize policy.

# B (PTX-27 / Section 5.4): z-normalize + min-max with an unprotected denominator
#   -> the z step is an affine map that the following per-image min-max cancels, and
#   a constant image divides by zero (NaN). New: min-max only, constant image mapped
#   to cfg.ptx.corrected.constant_image_value (0.0). No global/dataset statistics.
#   check: tests/corrected/test_data.py::test_minmax_*
# B (PTX-26 / Section 5.3): train Default `Resize` had no `antialias` while evaluation
#   Resizes passed `antialias=True` -> train/eval preprocessing differed. New: one
#   policy from config, `antialias` passed explicitly, identical for train/val/test.
#   check: tests/corrected/test_data.py::test_siim_eval_path_equals_eval_transform
"""

from __future__ import annotations

import numpy as np
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from sharpness_matters.config.load_config import cfg


def _resize_interpolation() -> InterpolationMode:
    """Interpolation mode from cfg.ptx.corrected.resize.interpolation ("bilinear")."""
    name = str(cfg.ptx.corrected.resize.interpolation).upper()
    try:
        return getattr(InterpolationMode, name)
    except AttributeError as exc:
        raise KeyError(
            f"Unknown resize interpolation {name!r}; "
            "cfg.ptx.corrected.resize.interpolation must be a torchvision "
            "InterpolationMode name"
        ) from exc


def minmax_per_image(x: np.ndarray) -> np.ndarray:
    """Scale one image to [0, 1] with its own min and max; never NaN.

    ``(x - min) / (max - min)`` in float64, returned as float32. When
    ``max == min`` the image carries no contrast and the whole output is
    ``cfg.ptx.corrected.constant_image_value`` (0.0) instead of 0/0.

    The statistics are taken on the input dtype (float64 for float inputs) so a
    near-constant image is not collapsed by an earlier float32 cast.
    """
    arr = np.asarray(x)
    if arr.size == 0:
        return arr.astype(np.float32, copy=True)
    work = arr.astype(np.float64, copy=False)
    min_val = work.min()
    max_val = work.max()
    if max_val == min_val:
        value = float(cfg.ptx.corrected.constant_image_value)
        return np.full(arr.shape, value, dtype=np.float32)
    return ((work - min_val) / (max_val - min_val)).astype(np.float32)


def eval_transform(img_size: int) -> transforms.Compose:
    """Deterministic preprocessing: ToTensor, then Resize with antialias.

    This is the single resize policy of stage 2 (contract module 2). Order,
    interpolation and antialias come from cfg.ptx.corrected.resize and are passed
    explicitly on every call.
    """
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(
                (img_size, img_size),
                interpolation=_resize_interpolation(),
                antialias=bool(cfg.ptx.corrected.resize.antialias),
            ),
        ]
    )


def train_transform(img_size: int) -> transforms.Compose:
    """Training preprocessing: eval steps, then the author's augmentations.

    Author order (``train_kfold.py``: ToTensor, Resize, RandomHorizontalFlip(0.5),
    RandomRotation(15)). Only the resize step is replaced by the shared policy, so
    a validation pass can use ``eval_transform`` and see the same resize.
    """
    steps = list(eval_transform(img_size).transforms)
    steps.extend(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
        ]
    )
    return transforms.Compose(steps)

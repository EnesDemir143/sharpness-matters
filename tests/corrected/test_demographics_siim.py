"""SIIM demographics loader must index only the top-level ``dicom-images-train`` tree.

The Kaggle archive can be extracted with a second copy of itself nested under the
root (``siim-acr/pneumothorax/dicom-images-train``). A recursive scan then maps every
ImageId to two files and the ambiguity check rejects the whole cohort.
"""

from pathlib import Path

import pandas as pd
import pytest

from sharpness_matters.ptx.datasets import extract_demographics

TRAIN_SUBDIR = "dicom-images-train"
N_IMAGES = 40


def _write_dicom_tree(base: Path, image_ids) -> None:
    for image_id in image_ids:
        study = base / TRAIN_SUBDIR / f"study-{image_id}" / f"series-{image_id}"
        study.mkdir(parents=True)
        (study / f"{image_id}.dcm").write_bytes(b"placeholder")


def _write_labels(root: Path, image_ids) -> None:
    positive = " 100 10"
    negative = " -1"
    rows = [
        {"ImageId": image_id, " EncodedPixels": positive if i % 2 else negative}
        for i, image_id in enumerate(image_ids)
    ]
    pd.DataFrame(rows).to_csv(root / "train-rle.csv", index=False)


@pytest.fixture
def siim_root_with_nested_copy(tmp_path, monkeypatch):
    image_ids = [f"1.2.3.{i}" for i in range(N_IMAGES)]
    root = tmp_path / "siim-acr"
    root.mkdir()
    _write_labels(root, image_ids)
    _write_dicom_tree(root, image_ids)
    _write_dicom_tree(root / "pneumothorax", image_ids)
    monkeypatch.setattr(extract_demographics, "SIIM_ROOT_DIR", str(root))
    return root


def test_nested_second_copy_is_not_indexed(siim_root_with_nested_copy):
    root = siim_root_with_nested_copy

    dataset = extract_demographics.load_siim_dataset(split="test")

    assert len(dataset.image_ids) == N_IMAGES // 10
    assert dataset.duplicate_ids == []
    top_level = str(root / TRAIN_SUBDIR)
    assert all(path.startswith(top_level) for path in dataset.file_map.values())


def test_missing_top_level_tree_raises_even_if_nested_copy_exists(
    tmp_path, monkeypatch
):
    image_ids = [f"1.2.3.{i}" for i in range(N_IMAGES)]
    root = tmp_path / "siim-acr"
    root.mkdir()
    _write_labels(root, image_ids)
    _write_dicom_tree(root / "pneumothorax", image_ids)
    monkeypatch.setattr(extract_demographics, "SIIM_ROOT_DIR", str(root))

    with pytest.raises(FileNotFoundError, match="No DICOM files"):
        extract_demographics.load_siim_dataset(split="test")

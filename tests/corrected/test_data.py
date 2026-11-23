"""Synthetic checks for the stage-2 (B) data slice.

Covers contract modules 1-3 and 6-7: frozen manifests, per-image min-max, the
shared resize policy, the SIIM fold manifest, E5 reference/clean derivation, the
exact SIIM ImageId -> file map and the masked loaders' native-grid masks.
No real data, no author-script run, no training.
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torchvision import transforms

from sharpness_matters.config.load_config import cfg
from sharpness_matters.ptx.corrected.frozen import (
    hash_path,
    load_or_create_frozen,
    manifest_dir,
    record_path,
    repo_root,
    sha256_file,
)
from sharpness_matters.ptx.corrected.preprocess import (
    eval_transform,
    minmax_per_image,
    train_transform,
)
from sharpness_matters.ptx.corrected.make_fold_manifest import build_fold_manifest
from sharpness_matters.ptx.corrected import make_e5
from sharpness_matters.ptx.datasets import siim_acr_classification_dataset as siim_cls
from sharpness_matters.ptx.datasets import siim_acr_dataset as siim_masked
from sharpness_matters.ptx.datasets import ptx498_dataset as ptx498
from sharpness_matters.ptx.datasets.vin_dataset import VinDataset


# --------------------------------------------------------------------- helpers


class FakeDicom:
    """Minimal pydicom stand-in: pixel array plus metadata lookups."""

    def __init__(self, pixels, patient_id="PATIENT_X"):
        self.pixel_array = pixels
        self._patient_id = patient_id
        self.file_meta = None

    def get(self, key, default=None):
        if key == "PatientID":
            return self._patient_id
        return default


class FakeNifti:
    def __init__(self, data):
        self._data = data

    def get_fdata(self):
        return self._data


def write_siim_label_csv(
    path: Path, image_ids, positive_rles: dict
) -> Path:
    """Write a train-rle.csv lookalike (the author column is ' EncodedPixels')."""
    rows = [
        {"ImageId": image_id, " EncodedPixels": positive_rles.get(image_id, " -1")}
        for image_id in image_ids
    ]
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def siim_file_paths(root: Path, image_ids, subdir: str = "a") -> list:
    """DICOM paths whose filename stems are exactly ``image_ids`` (files may be absent)."""
    return [
        str(root / "dicom-images-train" / subdir / "study" / f"{image_id}.dcm")
        for image_id in image_ids
    ]


# ---------------------------------------------------------------------- frozen


def test_frozen_create_load_and_do_not_rebuild(tmp_path):
    path = tmp_path / "manifest.csv"
    frame = pd.DataFrame({"image_id": ["a", "b"], "label": [1, 0]})
    calls = {"n": 0}

    def build():
        calls["n"] += 1
        return frame

    first = load_or_create_frozen(path, build, {"item": "unit"})
    assert calls["n"] == 1
    assert path.exists() and hash_path(path).exists() and record_path(path).exists()
    assert sha256_file(path) == hash_path(path).read_text().split()[0]
    assert record_path(path).read_text().find('"item": "unit"') >= 0

    second = load_or_create_frozen(path, build, {"item": "unit"})
    assert calls["n"] == 1  # loaded, never rebuilt
    assert list(first["image_id"]) == list(second["image_id"]) == ["a", "b"]
    assert second["image_id"].dtype == object  # ID columns stay strings


def test_frozen_raises_on_hash_mismatch(tmp_path):
    path = tmp_path / "manifest.csv"
    frame = pd.DataFrame({"image_id": ["a"], "label": [1]})
    load_or_create_frozen(path, lambda: frame, {"item": "unit"})
    path.write_text("image_id,label\nb,0\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        load_or_create_frozen(path, lambda: frame, {"item": "unit"})


def test_frozen_raises_on_missing_hash(tmp_path):
    path = tmp_path / "manifest.csv"
    path.write_text("image_id,label\na,1\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Inconsistent frozen"):
        load_or_create_frozen(
            path, lambda: pd.DataFrame({"image_id": ["a"], "label": [1]}), {}
        )


def test_frozen_raises_when_hash_has_no_csv(tmp_path):
    path = tmp_path / "manifest.csv"
    hash_path(path).write_text("0" * 64 + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Inconsistent frozen"):
        load_or_create_frozen(
            path, lambda: pd.DataFrame({"image_id": ["a"], "label": [1]}), {}
        )


# --------------------------------------------------------------------- min-max


def test_minmax_constant_image_is_configured_value_not_nan():
    out = minmax_per_image(np.full((8, 8), 7.25))
    assert out.dtype == np.float32
    assert not np.isnan(out).any()
    assert np.allclose(out, float(cfg.ptx.corrected.constant_image_value))


def test_minmax_near_constant_image_keeps_range():
    image = np.full((8, 8), 0.1, dtype=np.float64)
    image[0, 0] = 0.1 + 1e-12
    out = minmax_per_image(image)
    assert not np.isnan(out).any()
    assert out.min() == 0.0
    assert out.max() == pytest.approx(1.0, abs=1e-6)
    assert ((out >= 0.0) & (out <= 1.0)).all()


def test_minmax_single_bright_pixel():
    image = np.zeros((16, 16), dtype=np.uint16)
    image[3, 4] = 255
    out = minmax_per_image(image)
    assert out[3, 4] == 1.0
    assert np.count_nonzero(out) == 1
    assert ((out >= 0.0) & (out <= 1.0)).all()


def test_minmax_empty_array_is_finite():
    out = minmax_per_image(np.array([], dtype=np.float64))
    assert out.shape == (0,)
    assert out.dtype == np.float32
    assert not np.isnan(out).any()


def test_minmax_flat_image_inside_random_stack():
    stack = np.random.RandomState(0).randint(0, 4000, size=(2, 8, 8))
    stack[0] = 1234  # a constant slice must not poison the others
    for image in stack:
        out = minmax_per_image(image)
        assert not np.isnan(out).any()
        assert out.min() >= 0.0 and out.max() <= 1.0


# -------------------------------------------------------------------- transforms


def test_eval_transform_passes_antialias_explicitly():
    transform = eval_transform(64)
    resize = transform.transforms[-1]
    assert isinstance(resize, transforms.Resize)
    assert resize.antialias is True
    assert tuple(resize.size) == (64, 64)


def test_train_transform_is_eval_plus_author_augmentations():
    transform = train_transform(64)
    names = [type(step).__name__ for step in transform.transforms]
    assert names == [
        "ToTensor",
        "Resize",
        "RandomHorizontalFlip",
        "RandomRotation",
    ]


def test_siim_eval_path_equals_eval_transform(tmp_path, monkeypatch):
    image_ids = [f"img{i:02d}" for i in range(20)]
    positives = {image_id: " 0 4" for image_id in image_ids[:10]}
    label_csv = write_siim_label_csv(tmp_path / "train-rle.csv", image_ids, positives)
    file_paths = siim_file_paths(tmp_path, image_ids)
    synthetic = (
        (np.arange(24 * 32, dtype=np.float64).reshape(24, 32) % 251) * 3.0
    )
    monkeypatch.setattr(
        siim_cls.pydicom, "dcmread", lambda path: FakeDicom(synthetic)
    )

    dataset = siim_cls.ChestXrayDataset(
        file_paths, str(label_csv), img_size=16, split="train"
    )
    item = dataset[0]
    expected = eval_transform(16)(minmax_per_image(synthetic))
    assert torch.equal(item["data"], expected.float())
    assert item["pid"] == dataset.image_ids[0]


# ------------------------------------------------------------- SIIM file map


def test_siim_file_map_rejects_duplicate_stems(tmp_path):
    image_ids = [f"id{i:02d}" for i in range(20)]
    positives = {image_id: " 0 4" for image_id in image_ids[:10]}
    label_csv = write_siim_label_csv(tmp_path / "train-rle.csv", image_ids, positives)
    file_paths = siim_file_paths(tmp_path, image_ids, subdir="a") + siim_file_paths(
        tmp_path, image_ids, subdir="b"
    )
    with pytest.raises(RuntimeError, match="more than one DICOM file"):
        siim_cls.ChestXrayDataset(file_paths, str(label_csv), img_size=16)


def test_siim_file_map_rejects_missing_stems(tmp_path):
    image_ids = [f"id{i:02d}" for i in range(20)]
    positives = {image_id: " 0 4" for image_id in image_ids[:10]}
    label_csv = write_siim_label_csv(tmp_path / "train-rle.csv", image_ids, positives)
    with pytest.raises(RuntimeError, match="no DICOM file"):
        siim_cls.ChestXrayDataset([], str(label_csv), img_size=16)


def test_siim_file_map_matches_exact_stem_only(tmp_path):
    image_ids = ["abc123", "abc1234"]
    file_paths = [
        str(tmp_path / "dicom-images-train" / "a" / "s" / "abc1234.dcm")
    ]
    file_map, duplicates = siim_cls.index_dicom_files(file_paths)
    assert duplicates == []
    assert "abc1234" in file_map
    # The old substring search would have matched "abc123" to "abc1234.dcm".
    assert "abc123" not in file_map


# --------------------------------------------------------------- fold manifest


def test_fold_manifest_is_disjoint_stratified_and_complete():
    rng = np.random.RandomState(7)
    image_ids = [f"img{i:04d}" for i in range(600)]
    labels = list(rng.randint(0, 2, size=600))
    frame = build_fold_manifest(image_ids, labels)

    k_folds = int(cfg.ptx.corrected.k_folds)
    assert list(frame.columns) == ["image_id", "label", "fold"]
    assert sorted(frame["fold"].unique().tolist()) == list(range(k_folds))
    assert set(frame["image_id"]) == set(image_ids)
    assert len(frame) == len(image_ids)

    global_rate = float(np.mean(labels))
    seen = []
    for fold in range(k_folds):
        part = frame[frame["fold"] == fold]
        seen.extend(part["image_id"].tolist())
        positives = int(part["label"].sum())
        assert abs(positives - global_rate * len(part)) <= 1.0
    assert len(seen) == len(set(seen)) == len(image_ids)  # disjoint and complete

    again = build_fold_manifest(image_ids, labels)
    assert frame.equals(again)  # deterministic for the configured seed


def test_fold_manifest_rejects_duplicate_ids():
    with pytest.raises(ValueError, match="unique"):
        build_fold_manifest(["a", "a", "b"], [1, 0, 1])


# -------------------------------------------------------------------- E5 clean


def test_e5_clean_without_duplicates_is_identity():
    neg = ["n1", "n2", "n3"]
    pos = ["p1", "p2"]
    neg_clean, report = make_e5.build_e5_clean(
        neg, pos, ["n1", "n2", "n3", "p1", "p2", "c1"], seed=42
    )
    assert neg_clean == neg
    assert report["n_dup"] == 0
    assert report["filled"] == 0
    assert report["complete"] is True
    assert all(report["checks"].values())


def test_e5_clean_fills_exactly_n_dup_and_passes_checks():
    neg = ["n1", "n2", "p1"]
    pos = ["p1", "p2"]
    candidates = ["n1", "n2", "p1", "p2", "c1", "c2", "c3"]
    neg_clean, report = make_e5.build_e5_clean(neg, pos, candidates, seed=42)

    assert report["n_dup"] == 1
    assert report["pool_size"] == 3
    assert report["filled"] == 1
    assert report["complete"] is True
    assert len(neg_clean) == 3 == report["target_neg"]
    assert set(neg_clean) & set(pos) == set()
    assert len(set(neg_clean)) == len(neg_clean)
    assert {"n1", "n2"}.issubset(set(neg_clean))
    assert neg_clean[:2] == ["n1", "n2"]  # frozen order is kept
    assert report["checks"]["disjoint_from_pos"] is True
    assert report["checks"]["unique"] is True
    assert report["checks"]["kept_neg_frozen_minus_pos"] is True
    assert report["checks"]["count_complete"] is True

    again, _ = make_e5.build_e5_clean(neg, pos, candidates, seed=42)
    assert again == neg_clean  # recorded seed -> same draw


def test_e5_clean_reports_shortfall_without_padding():
    neg = ["n1", "p1", "p2", "p3"]
    pos = ["p1", "p2", "p3"]
    neg_clean, report = make_e5.build_e5_clean(neg, pos, ["n1", "p1", "p2", "p3"], 42)
    assert report["n_dup"] == 3
    assert report["pool_size"] == 0
    assert report["shortfall"] == 3
    assert report["complete"] is False
    assert neg_clean == ["n1"]
    assert report["checks"]["count_complete"] is False
    assert report["checks"]["disjoint_from_pos"] is True
    assert report["checks"]["unique"] is True


def test_e5_clean_rejects_duplicate_pos():
    with pytest.raises(RuntimeError, match="POS list contains duplicate"):
        make_e5.build_e5_clean(["n1"], ["p1", "p1"], ["n1", "p1"], 42)


# ------------------------------------------------------- E5 reference freezing


def _fake_txt_pair(tmp_path, neg_rows, pos_rows):
    neg = tmp_path / "vin500neg.txt"
    pos = tmp_path / "vin500pos.txt"
    neg.write_text("".join(f"{row}\n" for row in neg_rows), encoding="utf-8")
    pos.write_text("".join(f"{row}\n" for row in pos_rows), encoding="utf-8")
    return neg, pos


def test_e5_reference_freeze_builds_author_order_manifest(tmp_path):
    neg_txt, pos_txt = _fake_txt_pair(tmp_path, ["a", "b", "c"], ["c", "d"])
    ref_dir = tmp_path / "ref"
    combined = tmp_path / "combined.csv"
    report = make_e5.freeze_reference_lists(neg_txt, pos_txt, ref_dir, combined)

    assert report["n_neg"] == 3 and report["n_pos"] == 2
    assert report["n_dup"] == 1
    assert report["n_combined_rows"] == 5
    assert report["n_unique_ids"] == 4
    assert (ref_dir / "vin500neg.txt").exists()
    assert (ref_dir / "vin500neg.txt.sha256").exists()
    assert (ref_dir / "vin500pos.txt.sha256").exists()
    assert (ref_dir / "reference.json").exists()

    frame = pd.read_csv(combined, dtype={"image_id": str})
    assert list(frame["image_id"]) == ["a", "b", "c", "c", "d"]
    assert list(frame["label"].astype(int)) == [0, 0, 1, 1, 1]
    assert list(frame["source_list"]) == ["neg", "neg", "neg", "pos", "pos"]

    # A second freeze verifies and reuses the frozen artifacts.
    report2 = make_e5.freeze_reference_lists(neg_txt, pos_txt, ref_dir, combined)
    assert report2["n_dup"] == 1
    assert sha256_file(combined) == report2["combined_csv_sha256"]


def test_e5_reference_freeze_refuses_changed_frozen_list(tmp_path):
    neg_txt, pos_txt = _fake_txt_pair(tmp_path, ["a", "b"], ["c"])
    ref_dir = tmp_path / "ref"
    combined = tmp_path / "combined.csv"
    make_e5.freeze_reference_lists(neg_txt, pos_txt, ref_dir, combined)

    (ref_dir / "vin500neg.txt").write_text("a\nz\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        make_e5.freeze_reference_lists(neg_txt, pos_txt, ref_dir, combined)


def test_e5_reference_freeze_rejects_missing_trailing_newline(tmp_path):
    neg = tmp_path / "vin500neg.txt"
    neg.write_text("a\nb", encoding="utf-8")
    pos = tmp_path / "vin500pos.txt"
    pos.write_text("c\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="trailing newline"):
        make_e5.freeze_reference_lists(neg, pos, tmp_path / "ref", tmp_path / "c.csv")


def test_e5_shim_redirects_the_hardcoded_csv(tmp_path):
    annotations = tmp_path / "annotations.csv"
    pd.DataFrame(
        {"image_id": ["x"], "class_name": ["Aortic enlargement"]}
    ).to_csv(annotations, index=False)
    script = tmp_path / "fake_author.py"
    output = tmp_path / "out.txt"
    script.write_text(
        "import pandas as pd\n"
        "df = pd.read_csv('/nowhere/hardcoded/train_vin.csv')\n"
        f"open({str(output)!r}, 'w').write(str(len(df)) + ' ' + df['image_id'].iloc[0])\n",
        encoding="utf-8",
    )
    shim = make_e5.build_shim(script, annotations)
    assert str(annotations.resolve()) in shim
    assert "run_path" in shim
    assert "PYTHONHASHSEED" not in shim  # the env var is set by the caller

    env = dict(os.environ, PYTHONHASHSEED="0")
    completed = subprocess.run(
        [sys.executable, "-c", shim], capture_output=True, text=True, env=env
    )
    assert completed.returncode == 0, completed.stderr
    assert output.read_text() == "1 x"


def test_e5_shim_fails_when_author_csv_is_never_read(tmp_path):
    annotations = tmp_path / "annotations.csv"
    pd.DataFrame({"image_id": ["x"], "class_name": ["A"]}).to_csv(
        annotations, index=False
    )
    script = tmp_path / "other_script.py"
    script.write_text("print('no csv here')\n", encoding="utf-8")
    shim = make_e5.build_shim(script, annotations)
    completed = subprocess.run(
        [sys.executable, "-c", shim], capture_output=True, text=True
    )
    assert completed.returncode != 0
    assert "never read train_vin.csv" in completed.stderr


def test_e5_candidate_pool_uses_class_name_rule(tmp_path):
    annotations = tmp_path / "annotations.csv"
    pd.DataFrame(
        {
            "image_id": ["p", "n1", "n2", "p"],
            "class_name": [
                "Pneumothorax",
                "Aortic enlargement",
                "Cardiomegaly",
                "Pneumothorax",
            ],
        }
    ).to_csv(annotations, index=False)
    candidates = make_e5.candidate_negative_ids(annotations)
    assert candidates == ["n1", "n2"]  # POS ids are not excluded here (E5-clean does that)


# ----------------------------------------------------------------- masked SIIM


def test_siim_masked_loader_keeps_1024_grid_and_thin_line(tmp_path, monkeypatch):
    image_ids = [f"id{i:02d}" for i in range(20)]
    # "0 1024" = one 1024-pixel run -> after rle2mask(...).T a single 1-px column.
    positives = {image_id: " 0 1024" for image_id in image_ids[:10]}
    label_csv = write_siim_label_csv(tmp_path / "train-rle.csv", image_ids, positives)
    file_paths = siim_file_paths(tmp_path, image_ids)
    pixels = (np.arange(8 * 8, dtype=np.uint16).reshape(8, 8) * 7)
    monkeypatch.setattr(
        siim_masked.pydicom, "dcmread", lambda path: FakeDicom(pixels)
    )

    dataset = siim_masked.ChestXrayDataset(
        file_paths, str(label_csv), img_size=16, split="train"
    )
    target = next(image_id for image_id in dataset.image_ids if image_id in positives)
    item = dataset[dataset.image_ids.index(target)]

    mask = item["mask"]
    assert mask.shape == (1024, 1024)
    assert mask.dtype == torch.uint8
    assert set(mask.unique().tolist()) == {0, 1}
    assert int(mask.sum()) == 1024  # the thin line is not eroded or resized away
    mask_np = mask.numpy()
    assert np.count_nonzero(mask_np.any(axis=0)) == 1  # a single 1-px column
    assert int(mask_np[:, 0].sum()) == 1024
    assert int(item["label"]) == 1
    assert item["pid"] == target
    assert item["image_id"] == target
    assert tuple(item["data"].shape) == (1, 16, 16)


def test_siim_masked_loader_negative_row_has_empty_mask(tmp_path, monkeypatch):
    image_ids = [f"id{i:02d}" for i in range(20)]
    positives = {image_id: " 0 1024" for image_id in image_ids[:10]}
    label_csv = write_siim_label_csv(tmp_path / "train-rle.csv", image_ids, positives)
    file_paths = siim_file_paths(tmp_path, image_ids)
    monkeypatch.setattr(
        siim_masked.pydicom, "dcmread", lambda path: FakeDicom(np.zeros((8, 8)))
    )
    dataset = siim_masked.ChestXrayDataset(
        file_paths, str(label_csv), img_size=16, split="train"
    )
    target = next(image_id for image_id in dataset.image_ids if image_id not in positives)
    item = dataset[dataset.image_ids.index(target)]
    assert int(item["label"]) == 0
    assert int(item["mask"].sum()) == 0
    assert item["mask"].shape == (1024, 1024)


# -------------------------------------------------------------------- PTX-498


def test_ptx498_mask_is_native_binary_and_rotated_like_the_image(tmp_path, monkeypatch):
    root = tmp_path / "ptx498"
    for site in ["A", "B", "C"]:
        site_dir = root / f"Site{site}"
        site_dir.mkdir(parents=True)
        (site_dir / "1.4.img.nii.gz").write_bytes(b"")
        (site_dir / "1.5.mask.nii.gz").write_bytes(b"")

    image = np.linspace(0.0, 1.0, 16, dtype=np.float64).reshape(4, 4)
    mask = np.zeros((4, 4), dtype=np.float64)
    mask[1, :] = 255.0  # also proves the {0,255} range is binarised, not multiplied

    def fake_load(path):
        return FakeNifti(mask if str(path).endswith("5.mask.nii.gz") else image)

    monkeypatch.setattr(ptx498.nib, "load", fake_load)
    dataset = ptx498.OODTestDataset(root_dir=str(root), img_size=2)
    index = dataset.image_ids.index("A_1")
    item = dataset[index]

    expected = torch.rot90(
        torch.from_numpy((mask > 0).astype(np.uint8)), 1, dims=(0, 1)
    ).numpy()
    assert item["mask"].dtype == np.uint8
    assert item["mask"].shape == (4, 4)  # native grid, not the 2x2 input size
    assert set(np.unique(item["mask"]).tolist()) == {0, 1}
    assert np.array_equal(item["mask"], expected)
    assert int(item["mask"].sum()) == 4
    assert item["image_id"] == item["pid"] == "A_1"
    assert tuple(item["data"].shape) == (1, 2, 2)


def test_ptx498_image_id_order_is_deterministic(tmp_path, monkeypatch):
    root = tmp_path / "ptx498"
    for site in ["A", "B", "C"]:
        site_dir = root / f"Site{site}"
        site_dir.mkdir(parents=True)
        for number in [3, 1, 2]:  # creation order must not matter
            (site_dir / f"{number}.4.img.nii.gz").write_bytes(b"")

    dataset = ptx498.OODTestDataset(root_dir=str(root), img_size=8)
    expected = [f"{site}_{number}" for site in "ABC" for number in ["1", "2", "3"]]
    assert dataset.image_ids == expected

    # Reversing the directory listing must not change the order (PYTHONHASHSEED and
    # filesystem order are both out of the picture).
    real_listdir = os.listdir

    def reversed_listdir(path):
        return list(reversed(real_listdir(path)))

    monkeypatch.setattr(ptx498.os, "listdir", reversed_listdir)
    reordered = ptx498.OODTestDataset(root_dir=str(root), img_size=8)
    assert reordered.image_ids == dataset.image_ids == expected


# ------------------------------------------------------------------- VinDataset


def test_vin_manifest_order_and_labels(tmp_path):
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame({"image_id": ["b", "a", "c"], "label": [1, 0, 1]}).to_csv(
        manifest, index=False
    )
    dataset = VinDataset(manifest_csv=str(manifest), img_size=16)
    assert dataset.image_ids == ["b", "a", "c"]
    assert dataset.labels == {"b": 1, "a": 0, "c": 1}
    assert len(dataset) == 3


def test_vin_requires_manifest(tmp_path):
    with pytest.raises(RuntimeError, match="manifest_csv"):
        VinDataset(img_size=16)


def test_vin_keeps_duplicate_rows_from_the_e5_reference_manifest(tmp_path):
    """The E5 reference-like manifest keeps an ID present in both author lists."""
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame({"image_id": ["a", "a", "b"], "label": [1, 1, 0]}).to_csv(
        manifest, index=False
    )
    dataset = VinDataset(manifest_csv=str(manifest), img_size=16)
    assert dataset.image_ids == ["a", "a", "b"]  # no deduplication
    assert len(dataset) == 3
    assert dataset.labels == {"a": 1, "b": 0}


def test_vin_rejects_conflicting_labels(tmp_path):
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame({"image_id": ["a", "a"], "label": [0, 1]}).to_csv(
        manifest, index=False
    )
    with pytest.raises(RuntimeError, match="conflicting labels"):
        VinDataset(manifest_csv=str(manifest), img_size=16)


def test_vin_manifest_default_transform_is_eval_transform(tmp_path):
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame({"image_id": ["a"], "label": [0]}).to_csv(manifest, index=False)
    dataset = VinDataset(manifest_csv=str(manifest), img_size=32)
    names = [type(step).__name__ for step in dataset.transform.transforms]
    assert names == ["ToTensor", "Resize"]


# ------------------------------------------------------------- demographics


def test_demographics_dedupe_by_image_id():
    from sharpness_matters.ptx.datasets.extract_demographics import (
        dedupe_demographics_by_image_id,
    )

    frame = pd.DataFrame(
        {
            "image_id": ["a", "a", "b", "c"],
            "age": [40, 40, 50, 60],
            "sex": ["M", "M", "F", "M"],
            "label": [1, 1, 0, 1],
        }
    )
    deduped, removed = dedupe_demographics_by_image_id(frame)
    assert removed == 1
    assert list(deduped["image_id"]) == ["a", "b", "c"]
    assert list(deduped["age"]) == [40, 50, 60]
    assert deduped.index.tolist() == [0, 1, 2]

    unique = frame.drop_duplicates(subset="image_id")
    unchanged, removed_none = dedupe_demographics_by_image_id(unique)
    assert removed_none == 0
    assert len(unchanged) == len(unique)


def test_demographics_cli_requires_vindr_manifest():
    script = (
        Path(__file__).resolve().parents[2]
        / "sharpness_matters"
        / "ptx"
        / "datasets"
        / "extract_demographics.py"
    )
    completed = subprocess.run(
        [sys.executable, str(script), "--dataset", "vindr"],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert "--vindr-manifest is required" in completed.stderr

    help_run = subprocess.run(
        [sys.executable, str(script), "--help"], capture_output=True, text=True
    )
    assert help_run.returncode == 0
    assert "--vindr-manifest" in help_run.stdout


# --------------------------------------------------------------- manifest dirs


def test_manifest_dir_follows_config():
    assert manifest_dir() == repo_root() / str(cfg.ptx.corrected.manifest_dir)

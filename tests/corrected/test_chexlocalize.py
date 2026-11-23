"""E4 CheXlocalize tests: RLE, dataset, frozen manifests, saliency wiring.

Synthetic only except the read-only real-JSON smoke test. The RLE encoder below is
an independent reference implementation (pycocotools compressed string) used to
check the production decoder, including a hand-made literal vector.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from sharpness_matters.config.load_config import cfg
from sharpness_matters.ptx.corrected import make_chexlocalize as mchx
from sharpness_matters.ptx.corrected.frozen import record_path
from sharpness_matters.ptx.datasets.chexlocalize_dataset import (
    FINDING,
    CheXlocalizeDataset,
    cxr_id_from_path,
    patient_id_from_path,
    rle_decode,
    rle_foreground_pixels,
    view_from_path,
)
from sharpness_matters.ptx.evaluation import evaluate_saliency_chexlocalize as eval_chx
from sharpness_matters.ptx.utils.saliency_eval import RULE_LEGACY, RULE_MAIN


# --- reference pycocotools encoder (independent of the production decoder) ---


def _encode_value(value):
    """One delta as 5-bit chunks; a chunk's 0x10 bit is its signed value's sign."""
    characters = []
    while True:
        chunk = value & 0x1F
        remaining = value >> 5
        signed = chunk - 32 if (chunk & 0x10) else chunk
        if remaining in (0, -1) and signed == value:
            characters.append(chr(chunk + 48))
            return "".join(characters)
        characters.append(chr((chunk | 0x20) + 48))
        value = remaining


def _reference_encode_string(counts):
    """Encode run lengths exactly like pycocotools ``rleToString``."""
    return "".join(
        _encode_value(
            int(count) - (int(counts[index - 2]) if index > 2 else 0)
        )
        for index, count in enumerate(counts)
    )


def _reference_rle(mask):
    """Compressed RLE of a binary mask, runs in Fortran (column-major) order."""
    array = np.asarray(mask, dtype=np.uint8)
    flat = array.flatten(order="F")
    counts = []
    current = 0
    run = 0
    for pixel in flat:
        value = int(pixel)
        if value == current:
            run += 1
        else:
            counts.append(run)
            current = value
            run = 1
    counts.append(run)
    return {
        "size": [int(array.shape[0]), int(array.shape[1])],
        "counts": _reference_encode_string(counts),
    }


def _mask_with_runs():
    mask = np.zeros((10, 60), dtype=np.uint8)
    mask[:, 10:30] = 1
    mask[:, 35:36] = 1
    return mask


_RANDOM_MASK = (
    np.random.RandomState(7).randint(0, 2, size=(17, 13)).astype(np.uint8)
)
_ROUNDTRIP_MASKS = [
    np.zeros((2, 5), dtype=np.uint8),
    np.ones((2, 5), dtype=np.uint8),
    np.array([[0, 1, 1, 1], [0, 0, 1, 0], [0, 0, 0, 0]], dtype=np.uint8),
    np.zeros((40, 40), dtype=np.uint8),
    _mask_with_runs(),
    _RANDOM_MASK,
]


# --- RLE ---


def test_rle_decoder_matches_hand_made_vector():
    # counts [3, 1, 2, 2, 1, 1, 2]; deltas after index 2 are [3, 1, 2, 1, -1, -1, 1]
    # -> '3', '1', '2', '1', 'O'(-1), 'O'(-1), '1'.
    rle = {"size": [3, 4], "counts": "3121OO1"}
    expected = np.array(
        [[0, 1, 1, 1], [0, 0, 1, 0], [0, 0, 0, 0]], dtype=np.uint8
    )
    np.testing.assert_array_equal(rle_decode(rle), expected)
    assert _reference_encode_string([3, 1, 2, 2, 1, 1, 2]) == "3121OO1"
    assert rle_foreground_pixels(rle) == int(expected.sum()) == 4


@pytest.mark.parametrize(
    "size,counts",
    [
        ((9, 18), [50, 30, 10, 46, 26]),  # +16 deltas need a second chunk
        ((7, 37), [100, 50, 20, 29, 60]),  # -21 delta needs a second chunk
    ],
)
def test_rle_signed_five_bit_deltas(size, counts):
    encoded = _reference_encode_string(counts)
    assert len(encoded) > len(counts)
    flat = np.concatenate(
        [np.full(count, index % 2, dtype=np.uint8) for index, count in enumerate(counts)]
    )
    expected = flat.reshape(size, order="F")
    decoded = rle_decode({"size": list(size), "counts": encoded})
    np.testing.assert_array_equal(decoded, expected)


def test_rle_decoder_fortran_order():
    rle = {"size": [2, 3], "counts": _reference_encode_string([1, 2, 1, 2])}
    expected = np.array([[0, 1, 1], [1, 0, 1]], dtype=np.uint8)
    np.testing.assert_array_equal(rle_decode(rle), expected)


@pytest.mark.parametrize("mask", _ROUNDTRIP_MASKS)
def test_rle_roundtrip_with_reference_encoder(mask):
    rle = _reference_rle(mask)
    decoded = rle_decode(rle)
    assert decoded.dtype == np.uint8
    assert decoded.shape == mask.shape
    np.testing.assert_array_equal(decoded, mask)
    assert rle_foreground_pixels(rle) == int(mask.sum())
    assert set(np.unique(decoded)).issubset({0, 1})


def test_rle_rejects_bad_input():
    with pytest.raises(ValueError):
        rle_decode({"size": [2, 2], "counts": _reference_encode_string([3, 3])})
    with pytest.raises(TypeError):
        rle_decode({"size": [2, 2], "counts": [4]})
    with pytest.raises(ValueError):
        rle_decode({"size": [2, 2], "counts": "z"})
    with pytest.raises(ValueError):
        rle_decode({"size": [2, 2], "counts": chr(48 + 0x20)})
    with pytest.raises(ValueError):
        rle_decode({"size": [0, 2], "counts": "0"})


def test_real_json_decodes_three_pneumothorax_masks():
    path = Path(cfg.ptx.chexlocalize.annotations_dir) / "gt_segmentations_val.json"
    if not path.is_file():
        pytest.skip(f"real CheXlocalize json not available: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    non_empty = sorted(
        image_id
        for image_id, entry in data.items()
        if rle_foreground_pixels(entry[FINDING]) > 0
    )
    assert len(non_empty) >= 3
    for image_id in non_empty[:3]:
        rle = data[image_id][FINDING]
        mask = rle_decode(rle)
        assert mask.shape == (rle["size"][0], rle["size"][1])
        assert mask.dtype == np.uint8
        assert set(np.unique(mask)).issubset({0, 1})
        assert int(mask.sum()) == rle_foreground_pixels(rle) > 0
    empty_id = next(
        image_id
        for image_id, entry in data.items()
        if rle_foreground_pixels(entry[FINDING]) == 0
    )
    assert int(rle_decode(data[empty_id][FINDING]).sum()) == 0


# --- path helpers ---


def test_cxr_id_and_view_from_path():
    assert (
        cxr_id_from_path(
            "CheXpert/v1.0/valid/patient64541/study1/view1_frontal.jpg"
        )
        == "patient64541_study1_view1_frontal"
    )
    assert (
        cxr_id_from_path("patient1/study2/view3_lateral.jpg")
        == "patient1_study2_view3_lateral"
    )
    assert view_from_path("patient1/study1/view1_frontal.jpg") == "frontal"
    assert view_from_path("patient1/study1/view2_lateral.jpg") == "lateral"
    assert patient_id_from_path("patient1/study2/view3_lateral.jpg") == "patient1"
    with pytest.raises(ValueError):
        cxr_id_from_path("view1_frontal.jpg")
    with pytest.raises(ValueError):
        view_from_path("patient1/study1/view1_oblique.jpg")


# --- synthetic fixtures ---


def _write_jpg(path, height=16, width=20):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = np.arange(height, dtype=np.uint16)[:, None] * 3
    columns = np.arange(width, dtype=np.uint16)[None, :] * 7
    array = ((rows + columns) % 256).astype(np.uint8)
    Image.fromarray(array, mode="L").save(path, format="JPEG", quality=95)


def _write_segmentations(path, masks):
    payload = {
        image_id: {FINDING: _reference_rle(mask)} for image_id, mask in masks.items()
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _mask_pair(height=16, width=20):
    empty = np.zeros((height, width), dtype=np.uint8)
    positive = empty.copy()
    positive[3:6, 4:9] = 1
    return empty, positive


class _Fixture:
    def __init__(self, root):
        self.root = Path(root)
        self.chexpert = self.root / "chexpert"
        self.annotations = self.root / "annotations"


def _make_dataset_fixture(tmp_path, mask_height=16):
    fixture = _Fixture(tmp_path)
    image_ids = [
        "patient1_study1_view1_frontal",
        "patient2_study1_view2_frontal",
    ]
    paths = [
        "val/patient1/study1/view1_frontal.jpg",
        "val/patient2/study1/view2_frontal.jpg",
    ]
    empty, positive = _mask_pair()
    for path in paths:
        _write_jpg(fixture.chexpert / path)
    masks = {
        image_ids[0]: positive,
        image_ids[1]: np.zeros((mask_height, 20), dtype=np.uint8),
    }
    _write_segmentations(
        fixture.annotations / "gt_segmentations_val.json", masks
    )
    manifest = fixture.root / "manifest.csv"
    pd.DataFrame(
        {
            "image_id": image_ids,
            "split": ["val", "val"],
            "patient_id": ["patient1", "patient2"],
            "path": paths,
            "label": [1, 0],
        }
    ).to_csv(manifest, index=False)
    return fixture, manifest, image_ids, paths


def _dataset(fixture, manifest, **kwargs):
    return CheXlocalizeDataset(
        str(manifest),
        kwargs.pop("img_size", 512),
        annotations_dir=str(fixture.annotations),
        chexpert_dir=str(fixture.chexpert),
        **kwargs,
    )


def test_dataset_image_mask_and_metadata(tmp_path):
    fixture, manifest, image_ids, paths = _make_dataset_fixture(tmp_path)
    dataset = _dataset(fixture, manifest, with_mask=True)
    assert dataset.image_ids == image_ids
    assert dataset.labels == {image_ids[0]: 1, image_ids[1]: 0}

    item = dataset[0]
    assert tuple(item["data"].shape) == (1, 512, 512)
    assert float(item["data"].min()) >= 0.0
    assert float(item["data"].max()) <= 1.0
    assert item["label"] == 1
    assert item["pid"] == "patient1"
    assert item["image_id"] == image_ids[0]
    assert item["split"] == "val"
    assert item["mask"].dtype == np.uint8
    assert item["mask"].shape == (16, 20)
    assert set(np.unique(item["mask"])).issubset({0, 1})
    assert int(item["mask"].sum()) == 15

    without_mask = _dataset(fixture, manifest, with_mask=False)
    assert "mask" not in without_mask[0]


def test_dataset_rejects_mask_grid_mismatch(tmp_path):
    fixture, manifest, image_ids, _ = _make_dataset_fixture(
        tmp_path, mask_height=17
    )
    dataset = _dataset(fixture, manifest, with_mask=True)
    with pytest.raises(RuntimeError, match="does not match"):
        dataset[1]


# --- manifest builder ---


def _make_builder_fixture(tmp_path, bad_mask=False):
    fixture = _Fixture(tmp_path)
    rows = {
        "patient1/study1/view1_frontal.jpg": 1.0,
        "patient1/study1/view2_frontal.jpg": 0.0,
        "patient2/study1/view1_frontal.jpg": -1.0,
        "patient3/study1/view1_frontal.jpg": None,
        "patient5/study1/view1_frontal.jpg": 1.0,
        "patient6/study1/view1_frontal.jpg": 0.0,
        "patient7/study1/view1_frontal.jpg": 1.0,
    }
    for relative in rows:
        _write_jpg(fixture.chexpert / "val" / relative)
    val_rows = [
        {"Path": f"CheXpert/v1.0/valid/{relative}", "Pneumothorax": label}
        for relative, label in rows.items()
    ]
    val_rows.append(
        {
            "Path": "CheXpert/v1.0/valid/patient4/study1/view1_lateral.jpg",
            "Pneumothorax": 1.0,
        }
    )
    pd.DataFrame(val_rows).to_csv(
        fixture.chexpert / "val_labels.csv", index=False
    )

    _write_jpg(fixture.chexpert / "test/patient8/study1/view1_frontal.jpg")
    pd.DataFrame(
        [
            {
                "Path": "CheXpert/v1.0/test/patient8/study1/view1_frontal.jpg",
                "Pneumothorax": 0.0,
            }
        ]
    ).to_csv(fixture.chexpert / "test_labels.csv", index=False)

    empty, positive = _mask_pair()
    val_masks = {
        "patient1_study1_view1_frontal": positive,
        "patient1_study1_view2_frontal": empty,
        "patient5_study1_view1_frontal": empty,
        "patient6_study1_view1_frontal": positive,
        "patient7_study1_view1_frontal": positive,
    }
    if bad_mask:
        wrong_grid = np.zeros((17, 20), dtype=np.uint8)
        wrong_grid[3:6, 4:9] = 1
        val_masks["patient1_study1_view1_frontal"] = wrong_grid
    _write_segmentations(
        fixture.annotations / "gt_segmentations_val.json", val_masks
    )
    _write_segmentations(
        fixture.annotations / "gt_segmentations_test.json",
        {"patient8_study1_view1_frontal": empty},
    )
    return fixture


def test_resolve_image_accepts_chexpert_valid_layout(tmp_path):
    root = tmp_path / "chexpert"
    path = root / "valid/patient1/study1/view1_frontal.jpg"
    _write_jpg(path)
    raw = "CheXpert/v1.0/valid/patient1/study1/view1_frontal.jpg"
    assert mchx._resolve_image(raw, "val", root) == path


def test_builder_cohorts_and_counts(tmp_path):
    fixture = _make_builder_fixture(tmp_path)
    cls_frame, xai_frame, counts = mchx.build_chexlocalize_manifests(
        chexpert_dir=str(fixture.chexpert),
        annotations_dir=str(fixture.annotations),
    )

    assert list(cls_frame.columns) == mchx.MANIFEST_COLUMNS
    assert list(xai_frame.columns) == mchx.MANIFEST_COLUMNS
    assert list(cls_frame["image_id"]) == [
        "patient1_study1_view1_frontal",
        "patient1_study1_view2_frontal",
        "patient5_study1_view1_frontal",
        "patient6_study1_view1_frontal",
        "patient7_study1_view1_frontal",
        "patient8_study1_view1_frontal",
    ]
    assert list(xai_frame["image_id"]) == [
        "patient1_study1_view1_frontal",
        "patient7_study1_view1_frontal",
    ]
    assert set(cls_frame["label"]) == {0, 1}
    assert all(
        path.startswith(("val/", "test/")) for path in cls_frame["path"]
    )

    val = counts["val"]
    assert val["rows_seen"] == 8
    assert val["images"] == 5
    assert val["patients"] == 4
    assert val["positives"] == 3
    assert val["negatives"] == 2
    assert val["laterals_excluded"] == 1
    assert val["uncertain_excluded"] == 2
    assert val["positives_without_mask"] == 1
    assert val["masks_without_positive_label"] == 1
    assert val["nonempty_masks"] == 3
    assert val["xai_images"] == 2
    assert val["xai_patients"] == 2
    assert val["nonempty_masks_outside_xai"] == 1

    test = counts["test"]
    assert test["images"] == 1
    assert test["patients"] == 1
    assert test["positives"] == 0
    assert test["negatives"] == 1
    assert test["xai_images"] == 0

    combined = counts["combined"]
    assert combined["images"] == 6
    assert combined["patients"] == 5
    assert combined["positives"] == 3
    assert combined["negatives"] == 3
    assert combined["laterals_excluded"] == 1
    assert combined["uncertain_excluded"] == 2
    assert combined["positives_without_mask"] == 1
    assert combined["masks_without_positive_label"] == 1
    assert combined["nonempty_masks"] == 3
    assert combined["xai_images"] == 2
    assert combined["xai_patients"] == 2
    assert combined["nonempty_masks_outside_xai"] == 1


def test_builder_rejects_mask_grid_mismatch(tmp_path):
    fixture = _make_builder_fixture(tmp_path, bad_mask=True)
    with pytest.raises(RuntimeError, match="does not match"):
        mchx.build_chexlocalize_manifests(
            chexpert_dir=str(fixture.chexpert),
            annotations_dir=str(fixture.annotations),
        )


def test_frozen_manifests_are_written_then_loaded_first(tmp_path, monkeypatch):
    fixture = _make_builder_fixture(tmp_path)
    cls_frame, xai_frame, counts = mchx.build_chexlocalize_manifests(
        chexpert_dir=str(fixture.chexpert),
        annotations_dir=str(fixture.annotations),
    )
    manifests = tmp_path / "manifests"
    monkeypatch.setattr(mchx, "manifest_dir", lambda: manifests)
    monkeypatch.setattr(
        mchx,
        "build_chexlocalize_manifests",
        lambda *args, **kwargs: (cls_frame, xai_frame, counts),
    )
    monkeypatch.setattr(mchx, "_source_hashes", lambda: {})

    loaded_cls = mchx.load_chexlocalize_cls_manifest()
    loaded_xai = mchx.load_chexlocalize_xai_manifest()
    assert list(loaded_cls["image_id"]) == list(cls_frame["image_id"])
    assert list(loaded_xai["image_id"]) == list(xai_frame["image_id"])
    for name in (mchx.CLS_MANIFEST_NAME, mchx.XAI_MANIFEST_NAME):
        assert (manifests / name).is_file()
        assert (manifests / (name + ".sha256")).is_file()
        assert record_path(manifests / name).is_file()

    record = json.loads(
        record_path(manifests / mchx.CLS_MANIFEST_NAME).read_text(
            encoding="utf-8"
        )
    )
    assert record["counts"]["val"]["images"] == 5
    assert record["counts"]["combined"]["xai_images"] == 2

    def _rebuilt(*args, **kwargs):
        raise AssertionError("a frozen CheXlocalize manifest must not be rebuilt")

    monkeypatch.setattr(mchx, "build_chexlocalize_manifests", _rebuilt)
    assert list(mchx.load_chexlocalize_cls_manifest()["image_id"]) == list(
        cls_frame["image_id"]
    )
    assert list(mchx.load_chexlocalize_xai_manifest()["image_id"]) == list(
        xai_frame["image_id"]
    )


# --- saliency wrapper wiring ---


def test_saliency_wrapper_uses_xai_manifest_and_512(tmp_path, monkeypatch):
    captured = {}

    class _FakeDataset:
        def __init__(self, *args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            self.with_mask = kwargs.get("with_mask")

    monkeypatch.setattr(eval_chx, "CheXlocalizeDataset", _FakeDataset)
    monkeypatch.setattr(eval_chx, "manifest_dir", lambda: tmp_path)
    dataset = eval_chx.build_dataset()
    assert captured["args"][0] == str(tmp_path / eval_chx.XAI_MANIFEST_NAME)
    assert captured["args"][1] == 512
    assert captured["kwargs"]["with_mask"] is True

    calls = []

    def _fake_run(dataset_arg, img_size, model_name, dataset_name, **kwargs):
        calls.append((dataset_arg, img_size, model_name, dataset_name, kwargs))
        return {RULE_MAIN: {}, RULE_LEGACY: {}}

    monkeypatch.setattr(eval_chx, "run_experiment", _fake_run)
    monkeypatch.setattr(eval_chx, "summary_table", lambda result: [])
    results = eval_chx.run_chexlocalize_saliency(
        model_names=["resnet"],
        dataset=dataset,
        output_root=tmp_path / "output",
    )
    assert set(results) == {"resnet"}
    dataset_arg, img_size, model_name, dataset_name, kwargs = calls[0]
    assert dataset_arg is dataset
    assert img_size == 512
    assert model_name == "resnet"
    assert dataset_name == "chexlocalize"
    assert kwargs["validation_mode"] == "ood"
    assert (
        tmp_path
        / "output"
        / "saliency_thresholded_chexlocalize_resnet_main.csv"
    ).is_file()

    with pytest.raises(ValueError):
        eval_chx.run_chexlocalize_saliency(
            model_names=["resnet"], img_size=256, dataset=dataset
        )

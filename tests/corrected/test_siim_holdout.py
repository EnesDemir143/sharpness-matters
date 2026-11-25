"""Behavioral checks for the frozen SIIM internal holdout (contract 4.1 / 6).

Synthetic only: a small ``train-rle.csv`` lookalike, a hand-written frozen
development manifest and the frozen holdout artifact. No DICOM is read; both loaders
resolve membership and labels without touching pixel data at construction.

Covered:
- the holdout is exactly the labelled cohort minus the development manifest;
- both the classification and the masked saliency loaders honour the frozen ID+label
  list when the RLE CSV is reordered (membership is not a fresh split);
- overlap, missing/extra IDs and label drift fail closed;
- the frozen artifact cannot be rebuilt at evaluation, and a tampered csv/record,
  absent provenance or mismatched development provenance raise.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from sharpness_matters.ptx.corrected.frozen import (
    hash_path,
    load_or_create_frozen,
    record_path,
    sha256_file,
)
from sharpness_matters.ptx.corrected.make_siim_holdout import (
    build_holdout_frame,
    load_and_verify_holdout,
    prepare_holdout,
    verify_holdout_membership,
)
from sharpness_matters.ptx.datasets import siim_acr_classification_dataset as siim_cls
from sharpness_matters.ptx.datasets import siim_acr_dataset as siim_masked


# --------------------------------------------------------------------- helpers

ALL_IDS = [f"s{index:02d}" for index in range(12)]
POSITIVES = set(ALL_IDS[:6])
DEVELOPMENT_IDS = ["s00", "s01", "s08", "s09"]  # 2 positive, 2 negative
HOLDOUT_IDS = sorted(set(ALL_IDS) - set(DEVELOPMENT_IDS))  # 4 positive, 4 negative


def _labels(image_ids):
    return {image_id: int(image_id in POSITIVES) for image_id in image_ids}


def _write_label_csv(path: Path, image_ids) -> Path:
    rows = [
        {
            "ImageId": image_id,
            " EncodedPixels": " 0 4" if image_id in POSITIVES else " -1",
        }
        for image_id in image_ids
    ]
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _development_frame(image_ids=DEVELOPMENT_IDS) -> pd.DataFrame:
    labels = _labels(image_ids)
    return pd.DataFrame(
        {
            "image_id": list(image_ids),
            "label": [labels[image_id] for image_id in image_ids],
            "fold": [index % 5 for index in range(len(image_ids))],
        }
    )


def _file_paths(root: Path, image_ids) -> list:
    return [
        str(root / "dicom-images-train" / "a" / "study" / f"{image_id}.dcm")
        for image_id in image_ids
    ]


def _freeze(path: Path, frame: pd.DataFrame, record=None) -> pd.DataFrame:
    return load_or_create_frozen(path, lambda: frame, record or {"item": "unit"})


def _fixture(tmp_path: Path, reordered: bool = False):
    """Frozen development manifest, RLE csv and frozen holdout for the synthetic ids."""
    order = list(reversed(ALL_IDS)) if reordered else list(ALL_IDS)
    label_csv = _write_label_csv(tmp_path / f"train-rle-{int(reordered)}.csv", order)
    development_path = tmp_path / "siim_folds.csv"
    _freeze(development_path, _development_frame(), {"item": "development"})
    dev_sha = sha256_file(development_path)

    holdout = build_holdout_frame(ALL_IDS, _labels(ALL_IDS), DEVELOPMENT_IDS)
    holdout_path = tmp_path / "siim_holdout.csv"
    _freeze(
        holdout_path,
        holdout,
        {"item": "holdout", "dev_manifest_sha256": dev_sha},
    )
    record = {
        "fold_manifest_sha256": dev_sha,
        "fold_manifest_provenance_version": 1,
    }
    return label_csv, development_path, holdout_path, holdout, record


def _save_frame(path: Path, frame: pd.DataFrame) -> None:
    """Rewrite a frozen frame coherently (csv, sidecar hash and record hash)."""
    frame.to_csv(path, index=False)
    digest = sha256_file(path)
    hash_path(path).write_text(digest + "\n", encoding="utf-8")
    creation_record = json.loads(record_path(path).read_text(encoding="utf-8"))
    creation_record["csv_sha256"] = digest
    record_path(path).write_text(json.dumps(creation_record), encoding="utf-8")


# ------------------------------------------------------------ complement build


def test_build_holdout_is_exact_complement_and_order_invariant():
    forward = build_holdout_frame(ALL_IDS, _labels(ALL_IDS), DEVELOPMENT_IDS)
    reversed_ids = list(reversed(ALL_IDS))
    backward = build_holdout_frame(reversed_ids, _labels(reversed_ids), DEVELOPMENT_IDS)

    assert forward.equals(backward)  # membership and order independent of CSV order
    assert list(forward.columns) == ["image_id", "label"]
    assert list(forward["image_id"]) == sorted(HOLDOUT_IDS)
    assert set(forward["image_id"]) == set(ALL_IDS) - set(DEVELOPMENT_IDS)
    assert not set(forward["image_id"]) & set(DEVELOPMENT_IDS)
    assert dict(zip(forward["image_id"], forward["label"])) == {
        image_id: _labels(ALL_IDS)[image_id] for image_id in forward["image_id"]
    }


def test_prepare_holdout_freezes_once_and_then_loads(tmp_path):
    label_csv = _write_label_csv(tmp_path / "train-rle.csv", ALL_IDS)
    development_path = tmp_path / "siim_folds.csv"
    _freeze(development_path, _development_frame(), {"item": "development"})
    holdout_path = tmp_path / "siim_holdout.csv"

    first, info = prepare_holdout(
        label_csv, development_path, out_path=holdout_path, verify_author_split=False
    )
    assert holdout_path.exists() and hash_path(holdout_path).exists()
    assert record_path(holdout_path).exists()
    assert info == {
        "n_images": 8,
        "n_positive": 4,
        "n_negative": 4,
        "n_development": len(DEVELOPMENT_IDS),
        "n_labelled": len(ALL_IDS),
    }
    frozen_record = json.loads(record_path(holdout_path).read_text())
    assert frozen_record["dev_manifest_sha256"] == sha256_file(development_path)

    second, _ = prepare_holdout(
        label_csv, development_path, out_path=holdout_path, verify_author_split=False
    )
    assert first.equals(second)


# ------------------------------------------- frozen loaders, order invariance


def test_classification_loader_uses_frozen_membership_when_csv_is_reordered(tmp_path):
    label_csv, development_path, holdout_path, holdout, _ = _fixture(tmp_path)
    reordered_csv, _, _, _, _ = _fixture(tmp_path, reordered=True)

    def load(csv_path):
        return siim_cls.ChestXrayDataset(
            _file_paths(tmp_path, ALL_IDS),
            str(csv_path),
            img_size=8,
            manifest_csv=str(holdout_path),
        )

    original = load(label_csv)
    reordered = load(reordered_csv)

    assert original.image_ids == reordered.image_ids == list(holdout["image_id"])
    assert original.labels == reordered.labels
    assert original.labels == dict(zip(holdout["image_id"], holdout["label"]))
    assert len(original) == len(holdout) == 8


def test_masked_saliency_loader_uses_frozen_membership_when_csv_is_reordered(tmp_path):
    label_csv, _, holdout_path, holdout, _ = _fixture(tmp_path)
    reordered_csv, _, _, _, _ = _fixture(tmp_path, reordered=True)

    def load(csv_path):
        return siim_masked.ChestXrayDataset(
            _file_paths(tmp_path, ALL_IDS),
            str(csv_path),
            img_size=8,
            manifest_csv=str(holdout_path),
        )

    original = load(label_csv)
    reordered = load(reordered_csv)

    assert original.image_ids == reordered.image_ids == list(holdout["image_id"])
    assert {i: original.labels[i]["label"] for i in original.image_ids} == {
        i: reordered.labels[i]["label"] for i in reordered.image_ids
    }
    assert all(
        original.labels[image_id]["label"] == label
        for image_id, label in zip(holdout["image_id"], holdout["label"])
    )
    assert all(
        original.labels[image_id]["mask"].shape == (1024, 1024)
        for image_id in original.image_ids
    )


def test_loader_rejects_frozen_label_drift(tmp_path):
    label_csv, _, holdout_path, holdout, _ = _fixture(tmp_path)
    tampered = holdout.copy()
    tampered.loc[0, "label"] = 1 - int(tampered.loc[0, "label"])
    _save_frame(holdout_path, tampered)  # valid hash, wrong label

    with pytest.raises(RuntimeError, match="disagrees with the RLE"):
        siim_cls.ChestXrayDataset(
            _file_paths(tmp_path, ALL_IDS),
            str(label_csv),
            img_size=8,
            manifest_csv=str(holdout_path),
        )
    with pytest.raises(RuntimeError, match="disagrees with the RLE"):
        siim_masked.ChestXrayDataset(
            _file_paths(tmp_path, ALL_IDS),
            str(label_csv),
            img_size=8,
            manifest_csv=str(holdout_path),
        )


def test_loader_rejects_manifest_id_without_rle_row(tmp_path):
    label_csv, _, holdout_path, _, _ = _fixture(tmp_path)
    missing_csv = _write_label_csv(
        tmp_path / "train-rle-missing.csv", ALL_IDS[:-1]
    )  # the first holdout id is absent

    with pytest.raises(RuntimeError, match="no row in the RLE CSV"):
        siim_cls.ChestXrayDataset(
            _file_paths(tmp_path, ALL_IDS),
            str(missing_csv),
            img_size=8,
            manifest_csv=str(holdout_path),
        )


# --------------------------------------------------------- verification gates


def test_verify_rejects_overlap_with_development(tmp_path):
    label_csv = _write_label_csv(tmp_path / "train-rle.csv", ALL_IDS)
    development = _development_frame()
    holdout = build_holdout_frame(ALL_IDS, _labels(ALL_IDS), DEVELOPMENT_IDS)
    leaked = pd.concat(
        [holdout, development.iloc[[0]][["image_id", "label"]]], ignore_index=True
    )
    with pytest.raises(RuntimeError, match="overlaps the development manifest"):
        verify_holdout_membership(leaked, development, label_csv)


def test_verify_rejects_missing_holdout_id(tmp_path):
    label_csv = _write_label_csv(tmp_path / "train-rle.csv", ALL_IDS)
    development = _development_frame()
    holdout = build_holdout_frame(ALL_IDS, _labels(ALL_IDS), DEVELOPMENT_IDS)
    with pytest.raises(RuntimeError, match="not the complement"):
        verify_holdout_membership(holdout.iloc[1:], development, label_csv)


def test_verify_rejects_extra_holdout_id(tmp_path):
    label_csv = _write_label_csv(tmp_path / "train-rle.csv", ALL_IDS)
    development = _development_frame()
    holdout = build_holdout_frame(ALL_IDS, _labels(ALL_IDS), DEVELOPMENT_IDS)
    extra = pd.concat(
        [holdout, pd.DataFrame({"image_id": ["unknown"], "label": [0]})],
        ignore_index=True,
    )
    with pytest.raises(RuntimeError, match="not the complement"):
        verify_holdout_membership(extra, development, label_csv)


def test_verify_rejects_development_label_drift(tmp_path):
    label_csv = _write_label_csv(tmp_path / "train-rle.csv", ALL_IDS)
    development = _development_frame()
    development.loc[0, "label"] = 1 - int(development.loc[0, "label"])
    holdout = build_holdout_frame(ALL_IDS, _labels(ALL_IDS), DEVELOPMENT_IDS)
    with pytest.raises(RuntimeError, match="development manifest disagrees"):
        verify_holdout_membership(holdout, development, label_csv)


# ------------------------------------------------------- frozen artifact gates


def test_prepare_holdout_rejects_same_ids_different_fold_manifest(tmp_path):
    """Identical holdout IDs are not enough: the dev manifest hash must be the source."""
    label_csv = _write_label_csv(tmp_path / "train-rle.csv", ALL_IDS)
    development_path = tmp_path / "siim_folds.csv"
    _freeze(development_path, _development_frame(), {"item": "development"})
    holdout_path = tmp_path / "siim_holdout.csv"
    prepare_holdout(
        label_csv, development_path, out_path=holdout_path, verify_author_split=False
    )

    alternate = _development_frame()
    alternate["fold"] = (alternate["fold"].astype(int) + 1) % 5
    alternate_path = tmp_path / "siim_folds_alt.csv"
    _freeze(alternate_path, alternate, {"item": "development-alt"})

    with pytest.raises(RuntimeError, match="provenance mismatch"):
        prepare_holdout(
            label_csv,
            alternate_path,
            out_path=holdout_path,
            verify_author_split=False,
        )


def test_prepare_holdout_rejects_missing_dev_manifest_binding(tmp_path):
    label_csv = _write_label_csv(tmp_path / "train-rle.csv", ALL_IDS)
    development_path = tmp_path / "siim_folds.csv"
    _freeze(development_path, _development_frame(), {"item": "development"})
    holdout_path = tmp_path / "siim_holdout.csv"
    prepare_holdout(
        label_csv, development_path, out_path=holdout_path, verify_author_split=False
    )
    frozen_record = json.loads(record_path(holdout_path).read_text())
    frozen_record.pop("dev_manifest_sha256")
    record_path(holdout_path).write_text(json.dumps(frozen_record), encoding="utf-8")

    with pytest.raises(RuntimeError, match="no dev_manifest_sha256"):
        prepare_holdout(
            label_csv,
            development_path,
            out_path=holdout_path,
            verify_author_split=False,
        )


def test_holdout_record_must_bind_csv_sha256_not_only_sidecar(tmp_path):
    """A rewritten csv + .sha256 must not pass when the record lacks csv_sha256."""
    label_csv, development_path, holdout_path, holdout, record = _fixture(tmp_path)
    tampered = holdout.copy()
    tampered.loc[0, "label"] = 1 - int(tampered.loc[0, "label"])
    tampered.to_csv(holdout_path, index=False)
    hash_path(holdout_path).write_text(
        sha256_file(holdout_path) + "\n", encoding="utf-8"
    )
    frozen_record = json.loads(record_path(holdout_path).read_text())
    frozen_record.pop("csv_sha256")
    record_path(holdout_path).write_text(json.dumps(frozen_record), encoding="utf-8")

    with pytest.raises(RuntimeError, match="no csv_sha256"):
        load_and_verify_holdout(
            record,
            label_csv,
            holdout_path=holdout_path,
            development_manifest=development_path,
        )


def test_load_and_verify_holdout_rejects_tampered_csv(tmp_path):
    label_csv, development_path, holdout_path, holdout, record = _fixture(tmp_path)
    holdout_path.write_text(
        holdout.assign(label=1 - holdout["label"]).to_csv(index=False),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="hash mismatch"):
        load_and_verify_holdout(
            record,
            label_csv,
            holdout_path=holdout_path,
            development_manifest=development_path,
        )


def test_load_frozen_rejects_record_csv_disagreement(tmp_path):
    label_csv, development_path, holdout_path, _, record = _fixture(tmp_path)
    # Rewrite csv + sidecar but leave the creation record's csv_sha256 stale.
    frame = pd.read_csv(holdout_path, dtype={"image_id": str}).iloc[:-1]
    frame.to_csv(holdout_path, index=False)
    hash_path(holdout_path).write_text(
        sha256_file(holdout_path) + "\n", encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="record and csv disagree"):
        load_and_verify_holdout(
            record,
            label_csv,
            holdout_path=holdout_path,
            development_manifest=development_path,
        )


def test_load_and_verify_holdout_never_builds_a_missing_artifact(tmp_path):
    label_csv, development_path, holdout_path, _, record = _fixture(tmp_path)
    holdout_path.unlink()
    hash_path(holdout_path).unlink()
    record_path(holdout_path).unlink()

    with pytest.raises(RuntimeError, match="is missing"):
        load_and_verify_holdout(
            record,
            label_csv,
            holdout_path=holdout_path,
            development_manifest=development_path,
        )
    assert not holdout_path.exists()
    assert not hash_path(holdout_path).exists()


def test_load_and_verify_holdout_rejects_absent_provenance(tmp_path):
    label_csv, development_path, holdout_path, _, _ = _fixture(tmp_path)
    with pytest.raises(RuntimeError, match="no fold_manifest_sha256"):
        load_and_verify_holdout(
            {"model_name": "dummy"},
            label_csv,
            holdout_path=holdout_path,
            development_manifest=development_path,
        )


def test_load_and_verify_holdout_rejects_mismatched_provenance(tmp_path):
    label_csv, development_path, holdout_path, _, _ = _fixture(tmp_path)
    with pytest.raises(RuntimeError, match="provenance mismatch"):
        load_and_verify_holdout(
            {
                "fold_manifest_sha256": "0" * 64,
                "fold_manifest_provenance_version": 1,
            },
            label_csv,
            holdout_path=holdout_path,
            development_manifest=development_path,
        )


def test_load_and_verify_holdout_rejects_legacy_hash_without_version(tmp_path):
    """An old record hashes a file but never proved the trained frame was that file."""
    label_csv, development_path, holdout_path, _, record = _fixture(tmp_path)
    legacy = {"fold_manifest_sha256": record["fold_manifest_sha256"]}
    with pytest.raises(RuntimeError, match="provenance is missing or outdated"):
        load_and_verify_holdout(
            legacy,
            label_csv,
            holdout_path=holdout_path,
            development_manifest=development_path,
        )


def test_load_and_verify_holdout_rejects_holdout_frozen_against_other_dev(tmp_path):
    label_csv, development_path, holdout_path, _, record = _fixture(tmp_path)
    frozen_record = json.loads(record_path(holdout_path).read_text())
    frozen_record["dev_manifest_sha256"] = "f" * 64
    record_path(holdout_path).write_text(json.dumps(frozen_record), encoding="utf-8")

    with pytest.raises(RuntimeError, match="holdout provenance mismatch"):
        load_and_verify_holdout(
            record,
            label_csv,
            holdout_path=holdout_path,
            development_manifest=development_path,
        )


def test_load_and_verify_holdout_happy_path_returns_frozen_frame(tmp_path):
    label_csv, development_path, holdout_path, holdout, record = _fixture(tmp_path)
    loaded = load_and_verify_holdout(
        record,
        label_csv,
        holdout_path=holdout_path,
        development_manifest=development_path,
    )
    assert loaded.equals(holdout)

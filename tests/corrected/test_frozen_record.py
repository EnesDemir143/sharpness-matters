"""`load_or_create_frozen` requires all three frozen files (contract module 1).

A csv + `.sha256` without `.record.json` is a partial frozen state: it must raise the
same `RuntimeError` as the other partial states (and as `load_frozen`), naming the
missing file, instead of silently loading the csv.
"""

import pandas as pd
import pytest

from sharpness_matters.ptx.corrected.frozen import (
    hash_path,
    load_frozen,
    load_or_create_frozen,
    record_path,
    sha256_file,
)

FRAME = pd.DataFrame({"image_id": ["a", "b"], "label": [1, 0]})


def _write_csv_and_hash(path):
    """Write the csv and its .sha256, deliberately leaving the record out."""
    FRAME.to_csv(path, index=False)
    hash_path(path).write_text(sha256_file(path) + "\n", encoding="utf-8")


def _refuse_build():
    raise AssertionError("a frozen manifest must never be rebuilt")


def test_csv_and_hash_without_record_raises_and_names_the_file(tmp_path):
    path = tmp_path / "manifest.csv"
    _write_csv_and_hash(path)

    with pytest.raises(RuntimeError) as excinfo:
        load_or_create_frozen(path, _refuse_build, {"item": "unit"})

    message = str(excinfo.value)
    assert "record.json" in message
    assert "manifest.csv" in message
    # Refuses; it neither loads the csv nor writes the missing record.
    assert not record_path(path).exists()


def test_missing_record_refusal_matches_load_frozen(tmp_path):
    path = tmp_path / "manifest.csv"
    _write_csv_and_hash(path)

    with pytest.raises(RuntimeError, match="record.json"):
        load_or_create_frozen(path, _refuse_build, {})
    with pytest.raises(RuntimeError, match="record.json"):
        load_frozen(path)


def test_record_deleted_after_freezing_raises_instead_of_loading(tmp_path):
    path = tmp_path / "manifest.csv"
    load_or_create_frozen(path, lambda: FRAME, {"item": "unit"})
    record_path(path).unlink()

    with pytest.raises(RuntimeError, match="record.json"):
        load_or_create_frozen(path, _refuse_build, {"item": "unit"})


def test_complete_frozen_set_still_loads_without_rebuild(tmp_path):
    path = tmp_path / "manifest.csv"
    load_or_create_frozen(path, lambda: FRAME, {"item": "unit"})

    loaded = load_or_create_frozen(path, _refuse_build, {"item": "unit"})

    assert list(loaded["image_id"]) == ["a", "b"]
    assert path.exists() and hash_path(path).exists() and record_path(path).exists()


def test_absent_artifact_is_built_and_freezes_all_three_files(tmp_path):
    path = tmp_path / "manifest.csv"

    frame = load_or_create_frozen(path, lambda: FRAME, {"item": "unit"})

    assert list(frame["image_id"]) == ["a", "b"]
    assert path.exists() and hash_path(path).exists() and record_path(path).exists()


def test_record_without_csv_still_raises_inconsistent_state(tmp_path):
    path = tmp_path / "manifest.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    FRAME.to_csv(path, index=False)
    hash_path(path).write_text(sha256_file(path) + "\n", encoding="utf-8")
    record_path(path).write_text("{}", encoding="utf-8")
    path.unlink()

    with pytest.raises(RuntimeError, match="Inconsistent frozen"):
        load_or_create_frozen(path, _refuse_build, {})

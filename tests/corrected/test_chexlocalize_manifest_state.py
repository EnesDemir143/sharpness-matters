"""CheXlocalize readers refuse a partial data directory (protocol §4.7).

The cls and xai manifests are one pair: if one is frozen (csv + .sha256 +
.record.json) while the other is entirely absent, a reader must raise instead of
rebuilding and re-freezing the missing half. Only when both are absent are they
built and frozen together.
"""

import json

import pandas as pd
import pytest

from sharpness_matters.ptx.corrected import make_chexlocalize as mchx
from sharpness_matters.ptx.corrected.frozen import hash_path, record_path

CLS = mchx.CLS_MANIFEST_NAME
XAI = mchx.XAI_MANIFEST_NAME


def _frames():
    cls = pd.DataFrame(
        {
            "image_id": ["patient1_study1_view1_frontal"],
            "split": ["val"],
            "patient_id": ["patient1"],
            "path": ["val/patient1/study1/view1_frontal.jpg"],
            "label": [1],
        }
    )
    xai = cls.copy()
    counts = {"combined": {"images": 1, "xai_images": 1}}
    return cls, xai, counts


def _patch(monkeypatch, tmp_path):
    manifests = tmp_path / "manifests"
    calls = []
    monkeypatch.setattr(mchx, "manifest_dir", lambda: manifests)
    monkeypatch.setattr(mchx, "_source_hashes", lambda: {})

    def _build(*args, **kwargs):
        calls.append(1)
        return _frames()

    monkeypatch.setattr(mchx, "build_chexlocalize_manifests", _build)
    return manifests, calls


def _artifact_files(manifests, name):
    path = manifests / name
    return [path, hash_path(path), record_path(path)]


def test_both_absent_are_built_and_frozen_together(tmp_path, monkeypatch):
    manifests, calls = _patch(monkeypatch, tmp_path)

    cls = mchx.load_chexlocalize_cls_manifest()
    xai = mchx.load_chexlocalize_xai_manifest()

    assert calls == [1]  # one build for the pair
    assert list(cls["image_id"]) == ["patient1_study1_view1_frontal"]
    assert list(xai["image_id"]) == ["patient1_study1_view1_frontal"]
    for name in (CLS, XAI):
        for path in _artifact_files(manifests, name):
            assert path.is_file(), path
    record = json.loads(record_path(manifests / CLS).read_text(encoding="utf-8"))
    assert record["counts"]["combined"]["xai_images"] == 1


def test_absent_xai_with_frozen_cls_raises_without_rebuild(tmp_path, monkeypatch):
    manifests, calls = _patch(monkeypatch, tmp_path)
    mchx.load_chexlocalize_cls_manifest()  # freezes both
    for path in _artifact_files(manifests, XAI):
        path.unlink()
    calls.clear()

    with pytest.raises(RuntimeError) as excinfo:
        mchx.load_chexlocalize_xai_manifest()

    message = str(excinfo.value)
    assert "partial" in message
    assert XAI in message and CLS in message
    assert calls == []  # never rebuilt from the partial data directory
    assert not (manifests / XAI).exists()
    # The frozen half still loads.
    assert list(mchx.load_chexlocalize_cls_manifest()["image_id"]) == [
        "patient1_study1_view1_frontal"
    ]


def test_absent_cls_with_frozen_xai_raises_without_rebuild(tmp_path, monkeypatch):
    manifests, calls = _patch(monkeypatch, tmp_path)
    mchx.load_chexlocalize_cls_manifest()  # freezes both
    for path in _artifact_files(manifests, CLS):
        path.unlink()
    calls.clear()

    with pytest.raises(RuntimeError) as excinfo:
        mchx.load_chexlocalize_cls_manifest()

    message = str(excinfo.value)
    assert "partial" in message
    assert CLS in message and XAI in message
    assert calls == []
    assert not (manifests / CLS).exists()
    assert list(mchx.load_chexlocalize_xai_manifest()["image_id"]) == [
        "patient1_study1_view1_frontal"
    ]


def test_partial_self_state_still_raises_inconsistent(tmp_path, monkeypatch):
    manifests, calls = _patch(monkeypatch, tmp_path)
    mchx.load_chexlocalize_cls_manifest()  # freezes both
    record_path(manifests / XAI).unlink()
    calls.clear()

    with pytest.raises(RuntimeError, match="Inconsistent frozen"):
        mchx.load_chexlocalize_xai_manifest()
    assert calls == []

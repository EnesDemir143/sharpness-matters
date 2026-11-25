"""PTX-498 cohort: the frozen manifest is the cohort, the directory must match it.

# B (PTX-18 / Section 7.3, follow-up): ``OODTestDataset`` derived the cohort from
#   ``os.listdir`` of SiteA/B/C, so an existing but EMPTY SiteC/ silently yielded 426
#   IDs (247+179+0) instead of 498, and a case with only PNG files or an image
#   without its mask was accepted at parse time and failed during inference. These
#   tests pin the new behaviour: ``manifests/ptx498_cohort.csv`` is the cohort (built
#   once by ``make_ptx498_manifest``, never at evaluation time), and the dataset
#   verifies the directory against it before publishing ``image_ids``.
#   check: this module

No real data, no image decoding: every file is an empty placeholder.
"""

import os
import shutil
from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner

from sharpness_matters.ptx.corrected import make_ptx498_manifest
from sharpness_matters.ptx.corrected.frozen import hash_path, load_or_create_frozen
from sharpness_matters.ptx.datasets import ptx498_dataset as ptx498

# Lexicographic on the case string, not numeric: A_1 < A_10 < A_2. The manifest
# order must survive verbatim (saliency chunk files and E5 rows depend on it).
COHORT = [
    ("A_1", "A", "1"),
    ("A_10", "A", "10"),
    ("A_2", "A", "2"),
    ("B_1", "B", "1"),
    ("C_1", "C", "1"),
    ("C_2", "C", "2"),
]
ID_COLUMNS = ["image_id", "site", "case"]


def freeze_cohort(path, rows=COHORT):
    """Write and freeze a synthetic cohort manifest (csv + .sha256 + .record.json)."""
    path = Path(path)
    frame = pd.DataFrame(list(rows), columns=ID_COLUMNS)
    load_or_create_frozen(path, lambda: frame, {"item": "synthetic PTX-498 cohort"})
    return path


def write_case(root, site, case, nifti=True):
    """Create one case: both NIfTI files, or only its three PNG files."""
    site_dir = Path(root) / f"Site{site}"
    site_dir.mkdir(parents=True, exist_ok=True)
    if nifti:
        (site_dir / f"{case}.4.img.nii.gz").write_bytes(b"")
        (site_dir / f"{case}.5.mask.nii.gz").write_bytes(b"")
    else:
        for suffix in ("1.img.png", "2.mask.png", "3.merge.png"):
            (site_dir / f"{case}.{suffix}").write_bytes(b"")


def write_cohort_root(root, rows=COHORT):
    for _, site, case in rows:
        write_case(root, site, case)
    return Path(root)


def build(tmp_path, root, manifest=None, rows=COHORT):
    manifest = (
        Path(manifest) if manifest is not None else freeze_cohort(tmp_path / "cohort.csv", rows)
    )
    return ptx498.OODTestDataset(
        root_dir=str(root), manifest_csv=manifest, img_size=8
    )


# ------------------------------------------------------------------ valid cohort


def test_manifest_order_is_the_cohort_order(tmp_path):
    root = write_cohort_root(tmp_path / "root")
    dataset = build(tmp_path, root)
    assert dataset.image_ids == [row[0] for row in COHORT]
    assert len(dataset) == len(COHORT)


# ------------------------------------------------------------ directory mismatches


def test_empty_site_directory_is_rejected(tmp_path):
    """The reported repro: SiteC/ exists but is empty -> 426 IDs instead of 498."""
    root = write_cohort_root(tmp_path / "root")
    for name in os.listdir(root / "SiteC"):
        (root / "SiteC" / name).unlink()

    with pytest.raises(RuntimeError) as excinfo:
        build(tmp_path, root)
    message = str(excinfo.value)
    assert "SiteC" in message
    assert "empty" in message
    assert "C_1" in message  # the missing IDs are named


def test_missing_mask_file_is_rejected(tmp_path):
    root = write_cohort_root(tmp_path / "root")
    (root / "SiteB" / "1.5.mask.nii.gz").unlink()

    with pytest.raises(RuntimeError) as excinfo:
        build(tmp_path, root)
    message = str(excinfo.value)
    assert "B_1" in message
    assert ".5.mask.nii.gz" in message


def test_case_missing_from_the_manifest_is_rejected(tmp_path):
    root = write_cohort_root(tmp_path / "root")
    write_case(root, "A", "9")  # both NIfTIs, but the frozen cohort has no A_9

    with pytest.raises(RuntimeError, match="A_9"):
        build(tmp_path, root)


def test_png_only_case_is_rejected(tmp_path):
    """PNG presence cannot stand in for the NIfTI pair the loader reads."""
    root = write_cohort_root(tmp_path / "root")
    (root / "SiteC" / "2.4.img.nii.gz").unlink()
    (root / "SiteC" / "2.5.mask.nii.gz").unlink()
    write_case(root, "C", "2", nifti=False)

    with pytest.raises(RuntimeError) as excinfo:
        build(tmp_path, root)
    message = str(excinfo.value)
    assert "C_2" in message
    assert ".4.img.nii.gz" in message


def test_missing_site_directory_is_rejected(tmp_path):
    root = write_cohort_root(tmp_path / "root")
    shutil.rmtree(root / "SiteB")

    with pytest.raises(RuntimeError) as excinfo:
        build(tmp_path, root)
    message = str(excinfo.value)
    assert "SiteB" in message
    assert "missing" in message


def test_tampered_manifest_is_rejected(tmp_path):
    root = write_cohort_root(tmp_path / "root")
    manifest = freeze_cohort(tmp_path / "cohort.csv")
    with open(manifest, "a", encoding="utf-8") as handle:
        handle.write("X_1,X,1\n")

    with pytest.raises(RuntimeError, match="hash mismatch"):
        build(tmp_path, root, manifest=manifest)


def test_manifest_without_sidecars_is_never_built(tmp_path):
    root = write_cohort_root(tmp_path / "root")
    manifest = freeze_cohort(tmp_path / "cohort.csv")
    hash_path(manifest).unlink()

    with pytest.raises(RuntimeError, match="missing"):
        build(tmp_path, root, manifest=manifest)
    assert not hash_path(manifest).exists()  # never rebuilt at load time


# --------------------------------------------------------------------- builder


def test_builder_rejects_incomplete_reference_root_and_writes_nothing(tmp_path):
    root = tmp_path / "ptx498"
    for site, count in make_ptx498_manifest.REFERENCE_CASE_COUNTS.items():
        for number in range(1, count + 1):
            write_case(root, site, str(number))
    (root / "SiteC" / "72.5.mask.nii.gz").unlink()  # present but incomplete
    (root / "SiteA" / "200.4.img.nii.gz").unlink()  # case absent entirely
    (root / "SiteA" / "200.5.mask.nii.gz").unlink()
    out_dir = tmp_path / "manifests"

    result = CliRunner().invoke(
        make_ptx498_manifest.main,
        ["--root-dir", str(root), "--manifest-dir", str(out_dir)],
    )

    assert result.exit_code != 0
    message = f"{result.output}\n{result.exception}"
    assert "A_200" in message
    assert "C_72" in message
    assert ".5.mask.nii.gz" in message
    assert not out_dir.exists() or list(out_dir.iterdir()) == []

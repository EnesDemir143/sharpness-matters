"""Synthetic checks for the stage-2 pre-run data checks (Section 9).

Everything here is synthetic: tiny DICOMs written with pydicom, tiny NIfTI masks
written with nibabel and a ``train-rle.csv`` lookalike. No real dataset, no network,
no training. The tests also prove the script never writes into a dataset root.

One recorded limit: with pydicom 2.4.0 and the numpy/gdcm decoders, ``BitsStored`` is
not used when decoding uncompressed pixel data, so forcing it cannot change those
values (checked empirically). The equal-decode case is therefore exercised with a
real DICOM whose stored values exceed the 12-bit limit, and the differing branch is
exercised by injecting the two decoders — the branch logic, not a codec claim, is
under test.
"""

import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from click.testing import CliRunner
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import UID, ExplicitVRLittleEndian, generate_uid
from sklearn.model_selection import train_test_split

from sharpness_matters.ptx.corrected import data_checks


# --------------------------------------------------------------------- fixtures


def write_dicom(
    path: Path,
    pixels,
    photometric: str = "MONOCHROME2",
    bits_stored: int = 12,
    bits_allocated: int = 16,
    pixel_representation: int = 0,
    transfer_syntax: str = ExplicitVRLittleEndian,
) -> Path:
    """Write a minimal uncompressed DICOM with the given pixel array."""
    path.parent.mkdir(parents=True, exist_ok=True)
    transfer_syntax = UID(str(transfer_syntax))
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.7"
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = transfer_syntax
    dataset = FileDataset(
        str(path),
        {},
        file_meta=file_meta,
        preamble=b"\x00" * 128,
        is_implicit_VR=transfer_syntax.is_implicit_VR,
        is_little_endian=transfer_syntax.is_little_endian,
    )
    pixels = np.asarray(pixels)
    dataset.Rows, dataset.Columns = pixels.shape
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = photometric
    dataset.BitsAllocated = bits_allocated
    dataset.BitsStored = bits_stored
    dataset.HighBit = bits_stored - 1
    dataset.PixelRepresentation = pixel_representation
    dtype = np.uint16 if bits_allocated == 16 else np.uint8
    dataset.PixelData = pixels.astype(dtype).tobytes()
    dataset.save_as(str(path), write_like_original=False)
    return path


def write_nifti(path: Path, array) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.asarray(array), np.eye(4)), str(path))
    return path


def write_rle_csv(path: Path, image_ids, encodings) -> Path:
    """``train-rle.csv`` lookalike (the author's column keeps a leading space)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "ImageId": list(image_ids),
            " EncodedPixels": [encodings.get(image_id, " -1") for image_id in image_ids],
        }
    )
    frame.to_csv(path, index=False)
    return path


def make_siim_root(base: Path, image_ids, positives, pixels) -> Path:
    """SIIM root with one DICOM per image id and the matching train-rle.csv."""
    root = base / "siim"
    write_rle_csv(
        root / "train-rle.csv",
        image_ids,
        {image_id: " 1 4" for image_id in positives},
    )
    for image_id in image_ids:
        write_dicom(
            root / "dicom-images-train" / "a" / "b" / f"{image_id}.dcm", pixels
        )
    return root


def snapshot(root: Path) -> dict:
    """(relative path, size, mtime_ns, sha256) for every file under root."""
    state = {}
    for path in sorted(Path(root).rglob("*")):
        if not path.is_file():
            continue
        stat = path.stat()
        state[str(path.relative_to(root))] = (
            stat.st_size,
            stat.st_mtime_ns,
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
    return state


PIXELS_4X4 = np.arange(16, dtype=np.uint16).reshape(4, 4)


# ------------------------------------------------------------------ 9.2 masks


def test_ptx498_mask_values_flag_non_binary(tmp_path):
    root = tmp_path / "ptx498"
    write_nifti(root / "SiteA" / "1.5.mask.nii.gz", np.zeros((4, 4), dtype=np.uint8))
    non_binary = np.zeros((4, 4), dtype=np.uint8)
    non_binary[0, :] = 255
    write_nifti(root / "SiteA" / "2.5.mask.nii.gz", non_binary)

    check = data_checks.check_ptx498_masks(root)

    assert check["status"] == "ISSUE → B item 9.2"
    assert check["code"] == "ISSUE"
    assert check["data"]["n_mask_files"] == 2
    assert check["data"]["n_non_binary_files"] == 1
    values = {row["value"] for row in check["data"]["value_counts"]}
    assert "255" in values and "1" not in values


def test_ptx498_binary_masks_are_no_change(tmp_path):
    root = tmp_path / "ptx498"
    binary = np.zeros((4, 4), dtype=np.uint8)
    binary[1:3, 1:3] = 1
    write_nifti(root / "SiteB" / "7.5.mask.nii.gz", binary)

    check = data_checks.check_ptx498_masks(root)

    assert check["status"].startswith("NO CHANGE NEEDED")
    assert check["data"]["n_non_binary_files"] == 0
    assert check["data"]["value_counts"] == [
        {"value": "0", "pixels": 12},
        {"value": "1", "pixels": 4},
    ]


# ------------------------------------------------------------ 9.3a photometric


def test_photometric_monochrome1_is_issue(tmp_path):
    root = tmp_path / "siim"
    write_dicom(
        root / "dicom-images-train" / "a" / "b" / "img_mono1.dcm",
        PIXELS_4X4,
        photometric="MONOCHROME1",
    )
    write_dicom(root / "dicom-images-train" / "a" / "b" / "img_mono2.dcm", PIXELS_4X4)

    check = data_checks.check_photometric(
        root, "SIIM-ACR", data_checks.SIIM_DICOM_PATTERNS
    )

    assert check["status"] == "ISSUE → B item 9.3a"
    assert check["data"]["monochrome1_count"] == 1
    assert check["data"]["monochrome1_ids"] == ["img_mono1"]
    assert check["data"]["pattern"] == "dicom-images-train/*/*/*.dcm"
    distribution = {
        row["photometric_interpretation"]: row["n_files"]
        for row in check["data"]["distribution"]
    }
    assert distribution == {"MONOCHROME1": 1, "MONOCHROME2": 1}


def test_photometric_all_monochrome2_is_no_change(tmp_path):
    root = tmp_path / "vindr"
    write_dicom(root / "v1.dicom", PIXELS_4X4)
    write_dicom(root / "v2.dicom", PIXELS_4X4)

    check = data_checks.check_photometric(root, "VinDr", data_checks.VINDR_DICOM_PATTERNS)

    assert check["status"].startswith("NO CHANGE NEEDED")
    assert check["data"]["monochrome1_count"] == 0
    assert check["data"]["pattern"] == "*.dicom"


# ------------------------------------------------------------- 9.3b bit depth


def test_vindr_equal_decodes_are_no_change(tmp_path):
    root = tmp_path / "vindr"
    # BitsStored=12 but values above 2**12-1: the forced decode cannot change them
    # with the installed decoders, which is exactly what the check must record.
    pixels = np.array(
        [[0, 1, 4095, 4096], [5000, 65535, 12, 300], [7, 8, 9, 10], [200, 400, 800, 1600]],
        dtype=np.uint16,
    )
    write_dicom(root / "v1.dicom", pixels, bits_stored=12)
    write_dicom(root / "v2.dicom", pixels, bits_stored=12)

    check = data_checks.check_vindr_bit_depth(root, samples_per_group=2)

    assert check["status"].startswith("NO CHANGE NEEDED")
    assert check["data"]["n_sampled"] == 2
    assert check["data"]["n_equal"] == 2 and check["data"]["n_differing"] == 0
    sample = check["data"]["samples"][0]
    assert sample["BitsStored"] == 12
    assert sample["BitsAllocated"] == 16
    assert sample["HighBit"] == 11
    assert sample["TransferSyntax"] == str(ExplicitVRLittleEndian)
    assert sample["exact_equal"] is True
    assert sample["n_changed"] == 0
    assert sample["fraction_changed"] == 0.0
    assert sample["original_max"] == 65535
    assert sample["original_bits_used"] == 16
    assert sample["original_exceeds_bits_stored"] is True
    assert sample["forced_max"] == 65535
    detail = check["data"]["samples_detail"][0]
    assert detail["comparison"]["bits_stored_limit"] == 4095
    assert detail["comparison"]["n_pixels"] == 16
    assert isinstance(detail["warnings_unmodified"]["messages"], list)


def test_vindr_decode_difference_is_open(tmp_path, monkeypatch):
    root = tmp_path / "vindr"
    write_dicom(root / "v1.dicom", PIXELS_4X4, bits_stored=12)
    original = np.zeros((4, 4), dtype=np.uint16)
    forced = original.copy()
    forced[0, 0] = 5000
    forced[1, 1] = 4096
    monkeypatch.setattr(data_checks, "decode_unmodified", lambda path: (original, []))
    monkeypatch.setattr(
        data_checks,
        "decode_forced_bits_stored",
        lambda path, bits=16: (forced, ["RuntimeWarning: synthetic decoder notice"]),
    )

    check = data_checks.check_vindr_bit_depth(root, samples_per_group=1)

    assert check["status"].startswith("OPEN")
    assert "which decode is correct" in check["status"]
    assert check["data"]["n_differing"] == 1
    sample = check["data"]["samples"][0]
    assert sample["exact_equal"] is False
    assert sample["n_changed"] == 2
    assert sample["fraction_changed"] == 0.125
    assert sample["max_abs_diff"] == 5000
    assert sample["forced_max"] == 5000
    assert sample["forced_exceeds_bits_stored"] is True
    comparison = check["data"]["samples_detail"][0]["comparison"]
    assert comparison["n_pixels"] == 16


# ------------------------------------------------------ 9.4 SIIM ImageId mapping


def test_siim_mapping_duplicate_stem_is_issue(tmp_path):
    image_ids = [f"id{index:02d}" for index in range(20)]
    root = make_siim_root(tmp_path, image_ids, image_ids[:10], PIXELS_4X4)
    write_dicom(
        root / "dicom-images-train" / "c" / "d" / f"{image_ids[0]}.dcm", PIXELS_4X4
    )

    check = data_checks.check_siim_mapping(root)

    assert check["status"] == "ISSUE → B item 9.4-SIIM"
    assert check["data"]["n_duplicate_image_ids"] == 1
    assert check["data"]["duplicate_examples"] == [image_ids[0]]


def test_siim_mapping_missing_file_is_open(tmp_path):
    image_ids = [f"id{index:02d}" for index in range(20)]
    root = make_siim_root(tmp_path, image_ids, image_ids[:10], PIXELS_4X4)
    (root / "dicom-images-train" / "a" / "b" / f"{image_ids[3]}.dcm").unlink()

    check = data_checks.check_siim_mapping(root)

    assert check["status"].startswith("OPEN")
    assert "no DICOM file" in check["status"]
    assert check["data"]["n_missing_files"] == 1
    assert check["data"]["missing_examples"] == [image_ids[3]]


def test_siim_mapping_split_reproduces_author_call(tmp_path, monkeypatch):
    image_ids = [f"id{index:02d}" for index in range(20)]
    positives = set(image_ids[:10])
    root = make_siim_root(tmp_path, image_ids, sorted(positives), PIXELS_4X4)
    labels = [1 if image_id in positives else 0 for image_id in image_ids]
    train_idx, test_idx = train_test_split(
        np.arange(len(image_ids)), stratify=labels, test_size=0.1, random_state=42
    )
    test_pos = int(sum(labels[index] for index in test_idx))
    expected = {
        "labelled_ids": len(image_ids),
        "dev_ids": int(len(train_idx)),
        "test_ids": int(len(test_idx)),
        "test_pos": test_pos,
        "test_neg": int(len(test_idx)) - test_pos,
    }
    monkeypatch.setattr(data_checks, "EXPECTED_SIIM", expected)

    check = data_checks.check_siim_mapping(root)

    assert check["status"].startswith("NO CHANGE NEEDED")
    assert check["data"]["observed"] == expected
    assert check["data"]["mismatches"] == []
    assert expected["test_pos"] == 1 and expected["test_neg"] == 1


def test_siim_mapping_tiny_label_set_is_open(tmp_path):
    image_ids = ["p1", "p2", "n1", "n2"]
    root = make_siim_root(tmp_path, image_ids, image_ids[:2], PIXELS_4X4)

    check = data_checks.check_siim_mapping(root)

    assert check["status"].startswith("OPEN")
    assert "stratified split is undefined" in check["status"]
    assert check["data"]["n_labelled_ids"] == 4
    assert "split_error" in check["data"]


def test_siim_mapping_count_mismatch_is_open(tmp_path):
    image_ids = [f"id{index:02d}" for index in range(20)]
    root = make_siim_root(tmp_path, image_ids, image_ids[:10], PIXELS_4X4)

    check = data_checks.check_siim_mapping(root)

    assert check["status"].startswith("OPEN")
    assert "differ from the protocol expectation" in check["status"]
    fields = {item["field"] for item in check["data"]["mismatches"]}
    assert fields == {"labelled_ids", "dev_ids", "test_ids", "test_pos", "test_neg"}
    assert check["data"]["observed"]["labelled_ids"] == 20


# ------------------------------------------------------- empty GT masks (K1)


def test_empty_positive_ptx_mask_is_issue(tmp_path):
    root = tmp_path / "ptx498"
    write_nifti(root / "SiteA" / "1.5.mask.nii.gz", np.zeros((4, 4), dtype=np.uint8))
    foreground = np.zeros((4, 4), dtype=np.uint8)
    foreground[1:3, 1:3] = 1
    write_nifti(root / "SiteB" / "2.5.mask.nii.gz", foreground)

    check = data_checks.check_empty_gt_masks(ptx498_root=root)

    assert check["status"] == "ISSUE → B item 7.4"
    block = check["data"]["datasets"]["ptx498"]
    assert block["n_masks"] == 2 and block["n_empty"] == 1
    assert block["empty_ids"] == ["SiteA/1"]
    assert check["data"]["n_empty_total"] == 1


def test_empty_positive_siim_mask_is_issue(tmp_path):
    root = tmp_path / "siim"
    write_rle_csv(
        root / "train-rle.csv",
        ["p1", "p2", "n1", "n2"],
        {"p1": " 1 4", "p2": " 1 0"},
    )

    check = data_checks.check_empty_gt_masks(siim_root=root)

    assert check["status"] == "ISSUE → B item 7.4"
    block = check["data"]["datasets"]["siim"]
    assert block["n_positive_ids"] == 2 and block["n_empty"] == 1
    assert block["empty_ids"] == ["p2"]


def test_empty_positive_masks_none_is_no_change(tmp_path):
    ptx_root = tmp_path / "ptx498"
    foreground = np.zeros((4, 4), dtype=np.uint8)
    foreground[0, :] = 1
    write_nifti(ptx_root / "SiteA" / "1.5.mask.nii.gz", foreground)
    siim_root = tmp_path / "siim"
    write_rle_csv(siim_root / "train-rle.csv", ["p1", "n1"], {"p1": " 1 4"})

    check = data_checks.check_empty_gt_masks(siim_root=siim_root, ptx498_root=ptx_root)

    assert check["status"].startswith("NO CHANGE NEEDED")
    assert check["data"]["n_empty_total"] == 0
    assert check["data"]["datasets"]["ptx498"]["n_masks"] == 1
    assert check["data"]["datasets"]["siim"]["n_positive_ids"] == 1


def test_empty_gt_missing_roots_not_run(tmp_path):
    check = data_checks.check_empty_gt_masks(
        siim_root=tmp_path / "nope_siim", ptx498_root=tmp_path / "nope_ptx"
    )

    assert check["status"] == "NOT RUN (path missing)"
    assert check["data"]["n_empty_total"] == 0


def test_empty_gt_one_root_missing_is_open(tmp_path):
    ptx_root = tmp_path / "ptx498"
    foreground = np.zeros((4, 4), dtype=np.uint8)
    foreground[0, :] = 1
    write_nifti(ptx_root / "SiteA" / "1.5.mask.nii.gz", foreground)

    check = data_checks.check_empty_gt_masks(
        siim_root=tmp_path / "nope_siim", ptx498_root=ptx_root
    )

    assert check["status"].startswith("OPEN")
    assert "completeness unproven" in check["status"]


# ---------------------------------------------------------------- 9.1 overlays


def test_orientation_overlays_written_and_open(tmp_path):
    image_ids = [f"img{index:02d}" for index in range(20)]
    siim_root = make_siim_root(
        tmp_path,
        image_ids,
        image_ids[:10],
        np.arange(64 * 64, dtype=np.uint16).reshape(64, 64),
    )
    ptx_root = tmp_path / "ptx498"
    write_nifti(
        ptx_root / "SiteA" / "1.4.img.nii.gz",
        np.arange(32 * 32, dtype=np.uint16).reshape(32, 32),
    )
    write_nifti(ptx_root / "SiteA" / "1.5.mask.nii.gz", np.ones((32, 32), np.uint8))

    payload = data_checks.run_checks(
        out=tmp_path / "reports",
        siim_root=str(siim_root),
        vindr_root=str(tmp_path / "missing_vindr"),
        ptx498_root=str(ptx_root),
        samples_per_group=1,
        as_of="2026-11-23",
    )

    orientation = next(check for check in payload["checks"] if check["id"] == "9.1")
    assert orientation["status"] == "OPEN (human visual decision required)"
    assert orientation["data"]["n_ptx498"] == 1
    assert orientation["data"]["n_siim"] == 1
    files = [Path(path) for path in orientation["data"]["files"]]
    assert len(files) == 1 and files[0].is_file()
    assert files[0].parent == tmp_path / "reports" / "data_checks_2026-11-23_overlays"
    assert files[0].stat().st_size > 0


# -------------------------------------------------------------- orchestration


def build_full_tree(base: Path) -> dict:
    """One tiny tree per dataset, with a MONOCHROME1 SIIM file."""
    siim_root = base / "siim"
    image_ids = [f"img{index:02d}" for index in range(20)]
    positives = image_ids[:10]
    write_rle_csv(
        siim_root / "train-rle.csv",
        image_ids,
        {image_id: " 1 4" for image_id in positives},
    )
    for image_id in image_ids:
        write_dicom(
            siim_root / "dicom-images-train" / "a" / "b" / f"{image_id}.dcm",
            PIXELS_4X4,
            photometric="MONOCHROME1" if image_id == image_ids[-1] else "MONOCHROME2",
        )
    vindr_root = base / "vindr"
    write_dicom(vindr_root / "v1.dicom", PIXELS_4X4, bits_stored=12)
    ptx_root = base / "ptx498"
    write_nifti(
        ptx_root / "SiteA" / "1.4.img.nii.gz",
        np.arange(16, dtype=np.uint16).reshape(4, 4),
    )
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[0, :] = 1
    write_nifti(ptx_root / "SiteA" / "1.5.mask.nii.gz", mask)
    return {"siim": siim_root, "vindr": vindr_root, "ptx498": ptx_root}


def test_run_checks_reports_every_section(tmp_path):
    roots = build_full_tree(tmp_path)

    payload = data_checks.run_checks(
        out=tmp_path / "reports",
        siim_root=str(roots["siim"]),
        vindr_root=str(roots["vindr"]),
        ptx498_root=str(roots["ptx498"]),
        samples_per_group=1,
        as_of="2026-11-23",
    )

    statuses = {check["id"]: check["status"] for check in payload["checks"]}
    assert [check["id"] for check in payload["checks"]] == [
        "9.2",
        "9.3a",
        "9.3a",
        "9.3b",
        "9.4-SIIM",
        "GT-EMPTY",
        "9.1",
    ]
    photometric = {
        check["title"].split()[0]: check["status"]
        for check in payload["checks"]
        if check["id"] == "9.3a"
    }
    assert photometric["SIIM-ACR"] == "ISSUE → B item 9.3a"
    assert photometric["VinDr"].startswith("NO CHANGE NEEDED")
    assert statuses["9.2"].startswith("NO CHANGE NEEDED")
    assert statuses["9.3b"].startswith("NO CHANGE NEEDED")
    assert statuses["9.4-SIIM"].startswith("OPEN")
    assert statuses["GT-EMPTY"].startswith("NO CHANGE NEEDED")
    assert statuses["9.1"] == "OPEN (human visual decision required)"
    assert payload["status_counts"]["ISSUE"] == 1
    assert payload["status_counts"]["OPEN"] == 2


def test_run_checks_missing_roots_are_not_run(tmp_path):
    missing = tmp_path / "nowhere"

    payload = data_checks.run_checks(
        out=tmp_path / "reports",
        siim_root=str(missing),
        vindr_root=str(missing / "vindr"),
        ptx498_root=str(missing / "ptx498"),
        as_of="2026-11-23",
    )

    for check in payload["checks"]:
        assert check["status"] == "NOT RUN (path missing)"
        assert check["code"] == "NOT RUN"
    assert payload["status_counts"]["NOT RUN"] == len(payload["checks"])

    markdown = Path(payload["outputs"]["markdown"])
    report_json = Path(payload["outputs"]["json"])
    assert markdown.name == "data_checks_2026-11-23.md"
    assert report_json.name == "data_checks_2026-11-23.json"
    text = markdown.read_text()
    assert "NOT RUN (path missing)" in text
    assert "## Status summary" in text
    reloaded = json.loads(report_json.read_text())
    assert reloaded["date"] == "2026-11-23"
    assert reloaded["roots_present"] == {"siim": False, "vindr": False, "ptx498": False}
    assert len(reloaded["checks"]) == len(payload["checks"])


def test_run_checks_never_touches_dataset_roots(tmp_path):
    roots = build_full_tree(tmp_path)
    before = {name: snapshot(path) for name, path in roots.items()}

    payload = data_checks.run_checks(
        out=tmp_path / "reports",
        siim_root=str(roots["siim"]),
        vindr_root=str(roots["vindr"]),
        ptx498_root=str(roots["ptx498"]),
        samples_per_group=1,
        as_of="2026-11-23",
    )

    after = {name: snapshot(path) for name, path in roots.items()}
    assert before == after
    for name, root in roots.items():
        assert not list(Path(root).rglob("data_checks*")), name
    assert Path(payload["outputs"]["markdown"]).is_file()
    assert Path(payload["outputs"]["json"]).is_file()


def test_cli_missing_roots(tmp_path):
    missing = tmp_path / "nowhere"
    result = CliRunner().invoke(
        data_checks.cli,
        [
            "--out",
            str(tmp_path / "reports"),
            "--siim-root",
            str(missing),
            "--vindr-root",
            str(missing),
            "--ptx498-root",
            str(missing),
            "--samples-per-group",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "9.2: NOT RUN (path missing)" in result.output
    assert "GT-EMPTY: NOT RUN (path missing)" in result.output
    assert "markdown:" in result.output
    assert list((tmp_path / "reports").glob("data_checks_*.md"))

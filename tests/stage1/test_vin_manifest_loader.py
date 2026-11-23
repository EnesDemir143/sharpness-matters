"""The E2/E2-wide thin loader must reproduce the author VinDataset item exactly."""
import numpy as np
import pandas as pd
import torch
from pydicom.dataset import Dataset as DcmDataset

from sharpness_matters.ptx.datasets import external_eval_datasets as ext
from sharpness_matters.ptx.datasets.vin_dataset import VinDataset


def _write_dicom(path) -> None:
    meta = DcmDataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.1"
    meta.MediaStorageSOPInstanceUID = "1.2.3"
    meta.TransferSyntaxUID = "1.2.840.10008.1.2"  # implicit VR little endian
    dataset = DcmDataset()
    dataset.file_meta = meta
    dataset.is_little_endian = True
    dataset.is_implicit_VR = True
    dataset.Rows, dataset.Columns = 8, 8
    dataset.BitsAllocated, dataset.BitsStored = 16, 16
    dataset.HighBit, dataset.PixelRepresentation = 15, 0
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.PixelData = (np.arange(64, dtype=np.uint16).reshape(8, 8) * 100).tobytes()
    dataset.PatientID = "pid-test"
    dataset.preamble = b"\x00" * 128
    dataset.save_as(str(path), write_like_original=False)


def test_manifest_vin_loader_matches_author_preprocessing(tmp_path):
    _write_dicom(tmp_path / "case.dicom")
    manifest = tmp_path / "m.csv"
    pd.DataFrame({"image_id": ["case"], "label": [1]}).to_csv(manifest, index=False)
    manifest_ds = ext.ManifestVinDataset(
        str(manifest), root_dir=str(tmp_path), img_size=8, stats_key="unit"
    )

    # The author's own __getitem__ on the same instance attributes: this is the
    # equality check that the thin loader did not change any preprocessing step.
    author = object.__new__(VinDataset)
    author.root_dir = manifest_ds.root_dir
    author.image_ids = manifest_ds.image_ids
    author.labels = manifest_ds.labels
    author.transform = manifest_ds.transform
    author.mean, author.std = manifest_ds.mean, manifest_ds.std

    ours, theirs = manifest_ds[0], author[0]
    assert torch.equal(ours["data"], theirs["data"])
    assert ours["label"] == theirs["label"] == 1
    assert ours["data"].shape == (1, 8, 8)


def test_loader_rejects_duplicate_manifest_ids(tmp_path):
    _write_dicom(tmp_path / "case.dicom")
    manifest = tmp_path / "m.csv"
    pd.DataFrame({"image_id": ["case", "case"], "label": [1, 1]}).to_csv(
        manifest, index=False
    )
    try:
        ext.ManifestVinDataset(str(manifest), root_dir=str(tmp_path), img_size=8)
    except RuntimeError as error:
        assert "more than once" in str(error)
    else:  # pragma: no cover
        raise AssertionError("duplicate manifest ids must be rejected")

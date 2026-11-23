# §9.1 PTX-498 orientation — decision (2026-11-23)

**Status:** ISSUE → B (fixed in `ptx/datasets/ptx498_dataset.py`, `orient_ptx498`). A keeps the author transform.

## Evidence
- Data: `PTX-498-v2-fix` (498 NIfTI image/mask pairs; the dataset also ships `N.1.img.png` and `N.2.mask.png`).
- The dataset PNG shows standard radiograph orientation (readable "R" marker on image left). The author's NIfTI + `rotate(90)` output shows the marker on the right and mirrored; cardiac silhouette on the wrong side.
- `reports/tools/corr.py` (normalised correlation at 128², all 498 images): author output matches the horizontally mirrored PNG for 497/498, 1 ambiguous (same direction, margin 0.04), 0 match the author orientation. Same pattern in SiteA, SiteB, SiteC (`reports/tools/probe.py` → `data_checks_2026-11-23_overlays/ptx498_sites_vs_dataset_png.png`).
- After the fix, the B loader (`OODTestDataset`) matches the PNG orientation for 498/498 images, and its native-grid mask equals the dataset `N.2.mask.png` exactly (IoU 1.0 for all 498).

## Interpretation
The author's comment says the 90° rotation was added "for some reason" to undo a rotation; a 90° rotation restores up/down but leaves left/right mirrored. Image and mask are rotated together, so GT alignment and saliency scoring are not affected; the model's external input is mirrored relative to SIIM training images. This is classified as an unintended implementation error (not an author choice). Its effect on scores is not measured and is not claimed.

## Change (B only)
`orient_ptx498(t) = flip(rot90(t, k=1, dims=(-2,-1)), dims=(-1,))` applied to image and mask through the same helper. Test: `tests/corrected/test_data.py::test_ptx498_orientation_is_rot90_plus_horizontal_flip`.

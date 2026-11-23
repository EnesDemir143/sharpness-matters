# Pre-run data checks — SIIM-ACR / VinDr / PTX-498

- Date: 2026-11-23
- Generated (UTC): 2026-11-23T18:56:57+00:00
- Script: `sharpness_matters/ptx/corrected/data_checks.py`
- Repo: `/Users/enesdemir/Documents/ap_articles/ref-source-code/sharpness-matters`
- Read-only: no training, no model load, no write/move inside a dataset root
- Report: `/Users/enesdemir/Documents/ap_articles/ref-source-code/sharpness-matters/reports/data_checks_2026-11-23.md` / `/Users/enesdemir/Documents/ap_articles/ref-source-code/sharpness-matters/reports/data_checks_2026-11-23.json`

## Roots

| Dataset | Root | Present |
|---|---|---|
| siim | `/Users/enesdemir/Documents/ap_articles/data/siim-acr` | yes |
| vindr | `/path/to/your/datasets` | no |
| ptx498 | `/Users/enesdemir/Documents/ap_articles/data/PTX-498-v2-fix` | yes |

## Status summary

| Check | Title | Status |
|---|---|---|
| 9.2 | PTX-498 mask values (row 9.2) | NO CHANGE NEEDED (evidence: all 498 mask files hold only values 0 and 1) |
| 9.3a | SIIM-ACR PhotometricInterpretation (row 9.3a) | NO CHANGE NEEDED (evidence: 0 of 10712 file(s) are MONOCHROME1) |
| 9.3a | VinDr PhotometricInterpretation (row 9.3a) | NOT RUN (path missing) |
| 9.3b | VinDr bit depth: unmodified vs forced BitsStored=16 (row 9.3b) | NOT RUN (path missing) |
| 9.4-SIIM | SIIM-ACR ImageId -> DICOM mapping and author split (row 9.4) | NO CHANGE NEEDED (evidence: 10675 labelled ImageIds, one file each; split 9607/1068 and test 238/830 match) |
| GT-EMPTY | Positive images with an empty GT mask (saliency K1 guard) | NO CHANGE NEEDED (evidence: 0 empty GT masks in 498 PTX-498 mask(s) and 2379 SIIM positive ImageId(s)) |
| 9.1 | PTX-498 orientation vs SIIM (row 9.1, human decision) | OPEN (human visual decision required) |

Counts: ISSUE=0, NO CHANGE NEEDED=4, OPEN=1, NOT RUN=2

Status vocabulary: `ISSUE → B item <no>` / `NO CHANGE NEEDED (evidence: …)` / `OPEN (reason: …)` / `NOT RUN (path missing)`. An ambiguous result is OPEN.

## 9.2 — PTX-498 mask values (row 9.2)

**Status:** NO CHANGE NEEDED (evidence: all 498 mask files hold only values 0 and 1)

- mask files: 498; distinct values: 2
- top values: 0 x 496336119, 1 x 25854729
- every mask value is 0 or 1

### Evidence

- **root**: /Users/enesdemir/Documents/ap_articles/data/PTX-498-v2-fix
- **n mask files**: 498
- **binarisation**: stage-2 loaders use `mask > 0` (contract module 6)
- **n non binary files**: 0
- **non binary files**: —
- **n unreadable files**: 0
- **unreadable files**: —
- **value counts** (2 row(s))

  | value | pixels |
  |---|---|
  | 0 | 496336119 |
  | 1 | 25854729 |

- **total pixels**: 522190848

## 9.3a — SIIM-ACR PhotometricInterpretation (row 9.3a)

**Status:** NO CHANGE NEEDED (evidence: 0 of 10712 file(s) are MONOCHROME1)

- distribution: MONOCHROME2 x 10712
- no MONOCHROME1 image

### Evidence

- **root**: /Users/enesdemir/Documents/ap_articles/data/siim-acr
- **dataset**: SIIM-ACR
- **pattern**: dicom-images-train/*/*/*.dcm
- **n dicom files**: 10712
- **distribution** (1 row(s))

  | photometric_interpretation | n_files |
  |---|---|
  | MONOCHROME2 | 10712 |

- **monochrome1 count**: 0
- **monochrome1 ids**: —
- **monochrome1 ids truncated**: 0
- **missing tag count**: 0
- **missing tag ids**: —
- **n unreadable**: 0
- **unreadable files**: —

## 9.3a — VinDr PhotometricInterpretation (row 9.3a)

**Status:** NOT RUN (path missing)

- root not found: /path/to/your/datasets
- nothing was inspected for this check; this is not a pass

### Evidence

- **root**: /path/to/your/datasets

## 9.3b — VinDr bit depth: unmodified vs forced BitsStored=16 (row 9.3b)

**Status:** NOT RUN (path missing)

- root not found: /path/to/your/datasets
- nothing was inspected for this check; this is not a pass

### Evidence

- **root**: /path/to/your/datasets

## 9.4-SIIM — SIIM-ACR ImageId -> DICOM mapping and author split (row 9.4)

**Status:** NO CHANGE NEEDED (evidence: 10675 labelled ImageIds, one file each; split 9607/1068 and test 238/830 match)

- observed: labelled_ids=10675, dev_ids=9607, test_ids=1068, test_pos=238, test_neg=830
- expected: labelled_ids=10675, dev_ids=9607, test_ids=1068, test_pos=238, test_neg=830
- DICOM files: 10712 (pattern 'dicom-images-train/*/*/*.dcm'); unique stems: 10712
- every labelled ImageId maps to exactly one file; counts match

### Evidence

- **root**: /Users/enesdemir/Documents/ap_articles/data/siim-acr
- **dicom pattern**: dicom-images-train/*/*/*.dcm
- **n dicom files**: 10712
- **rle csv**: /Users/enesdemir/Documents/ap_articles/data/siim-acr/train-rle.csv
- **expected**
  - **labelled ids**: 10675
  - **dev ids**: 9607
  - **test ids**: 1068
  - **test pos**: 238
  - **test neg**: 830
- **observed**
  - **labelled ids**: 10675
  - **dev ids**: 9607
  - **test ids**: 1068
  - **test pos**: 238
  - **test neg**: 830
- **mismatches**: —
- **n missing files**: 0
- **missing examples**: —
- **n duplicate image ids**: 0
- **duplicate examples**: —
- **n unique file stems**: 10712
- **label rule disagreements**: 0
- **split**: train_test_split(test_size=0.1, random_state=42)

## GT-EMPTY — Positive images with an empty GT mask (saliency K1 guard)

**Status:** NO CHANGE NEEDED (evidence: 0 empty GT masks in 498 PTX-498 mask(s) and 2379 SIIM positive ImageId(s))

- inspected datasets: ptx498, siim
- ptx498: 0 empty GT mask(s) of 498 mask(s)
- siim: 0 empty GT mask(s) of 2379 positive ImageId(s)
- PTX-498 example ids: none
- SIIM example ids: none
- no positive image has an empty GT mask

### Evidence

- **datasets**
  - **ptx498**
    - **root**: /Users/enesdemir/Documents/ap_articles/data/PTX-498-v2-fix
    - **available**: True
    - **n masks**: 498
    - **n empty**: 0
    - **empty ids**: —
    - **empty ids truncated**: 0
    - **n unreadable**: 0
    - **unreadable files**: —
  - **siim**
    - **root**: /Users/enesdemir/Documents/ap_articles/data/siim-acr
    - **available**: True
    - **rle csv**: /Users/enesdemir/Documents/ap_articles/data/siim-acr/train-rle.csv
    - **n positive ids**: 2379
    - **n empty**: 0
    - **empty ids**: —
    - **empty ids truncated**: 0
    - **n malformed**: 0
    - **malformed masks**: —
- **n empty total**: 0
- **b item**: 7.4
- **scoring effect**: score_saliency raises RuntimeError when a GT-positive image has an empty reference-grid mask (saliency_eval.py, PTX-04 / Section 7.4)

## 9.1 — PTX-498 orientation vs SIIM (row 9.1, human decision)

**Status:** OPEN (human visual decision required)

- overlay files: 5 (5 PTX-498, 5 SIIM)
- the human compares the two columns and records the decision in the report
- no automatic orientation/mirroring decision is made

### Evidence

- **dir**: /Users/enesdemir/Documents/ap_articles/ref-source-code/sharpness-matters/reports/data_checks_2026-11-23_overlays
- **n ptx498**: 5
- **n siim**: 5
- **files**: /Users/enesdemir/Documents/ap_articles/ref-source-code/sharpness-matters/reports/data_checks_2026-11-23_overlays/overlay_01.png, /Users/enesdemir/Documents/ap_articles/ref-source-code/sharpness-matters/reports/data_checks_2026-11-23_overlays/overlay_02.png, /Users/enesdemir/Documents/ap_articles/ref-source-code/sharpness-matters/reports/data_checks_2026-11-23_overlays/overlay_03.png, /Users/enesdemir/Documents/ap_articles/ref-source-code/sharpness-matters/reports/data_checks_2026-11-23_overlays/overlay_04.png, /Users/enesdemir/Documents/ap_articles/ref-source-code/sharpness-matters/reports/data_checks_2026-11-23_overlays/overlay_05.png
- **items** (5 row(s))

  | file | ptx498_id | ptx498_image | ptx498_mask | ptx_mask_resized | siim_id | siim_image | siim_mask_resized | error |
  |---|---|---|---|---|---|---|---|---|
  | /Users/enesdemir/Documents/ap_articles/ref-source-code/sharpness-matters/reports/data_checks_2026-11-23_overlays/overlay_01.png | SiteA/1 | /Users/enesdemir/Documents/ap_articles/data/PTX-498-v2-fix/SiteA/1.4.img.nii.gz | /Users/enesdemir/Documents/ap_articles/data/PTX-498-v2-fix/SiteA/1.5.mask.nii.gz | False | 1.2.276.0.7230010.3.1.4.8323329.4904.1517875185.355709 | /Users/enesdemir/Documents/ap_articles/data/siim-acr/dicom-images-train/1.2.276.0.7230010.3.1.2.8323329.4904.1517875185.355708/1.2.276.0.7230010.3.1.3.8323329.4904.1517875185.355707/1.2.276.0.7230010.3.1.4.8323329.4904.1517875185.355709.dcm | False | — |
  | /Users/enesdemir/Documents/ap_articles/ref-source-code/sharpness-matters/reports/data_checks_2026-11-23_overlays/overlay_02.png | SiteA/10 | /Users/enesdemir/Documents/ap_articles/data/PTX-498-v2-fix/SiteA/10.4.img.nii.gz | /Users/enesdemir/Documents/ap_articles/data/PTX-498-v2-fix/SiteA/10.5.mask.nii.gz | False | 1.2.276.0.7230010.3.1.4.8323329.32579.1517875161.299312 | /Users/enesdemir/Documents/ap_articles/data/siim-acr/dicom-images-train/1.2.276.0.7230010.3.1.2.8323329.32579.1517875161.299311/1.2.276.0.7230010.3.1.3.8323329.32579.1517875161.299310/1.2.276.0.7230010.3.1.4.8323329.32579.1517875161.299312.dcm | False | — |
  | /Users/enesdemir/Documents/ap_articles/ref-source-code/sharpness-matters/reports/data_checks_2026-11-23_overlays/overlay_03.png | SiteA/100 | /Users/enesdemir/Documents/ap_articles/data/PTX-498-v2-fix/SiteA/100.4.img.nii.gz | /Users/enesdemir/Documents/ap_articles/data/PTX-498-v2-fix/SiteA/100.5.mask.nii.gz | False | 1.2.276.0.7230010.3.1.4.8323329.1314.1517875167.222290 | /Users/enesdemir/Documents/ap_articles/data/siim-acr/dicom-images-train/1.2.276.0.7230010.3.1.2.8323329.1314.1517875167.222289/1.2.276.0.7230010.3.1.3.8323329.1314.1517875167.222288/1.2.276.0.7230010.3.1.4.8323329.1314.1517875167.222290.dcm | False | — |
  | /Users/enesdemir/Documents/ap_articles/ref-source-code/sharpness-matters/reports/data_checks_2026-11-23_overlays/overlay_04.png | SiteA/101 | /Users/enesdemir/Documents/ap_articles/data/PTX-498-v2-fix/SiteA/101.4.img.nii.gz | /Users/enesdemir/Documents/ap_articles/data/PTX-498-v2-fix/SiteA/101.5.mask.nii.gz | False | 1.2.276.0.7230010.3.1.4.8323329.4440.1517875182.865105 | /Users/enesdemir/Documents/ap_articles/data/siim-acr/dicom-images-train/1.2.276.0.7230010.3.1.2.8323329.4440.1517875182.865104/1.2.276.0.7230010.3.1.3.8323329.4440.1517875182.865103/1.2.276.0.7230010.3.1.4.8323329.4440.1517875182.865105.dcm | False | — |
  | /Users/enesdemir/Documents/ap_articles/ref-source-code/sharpness-matters/reports/data_checks_2026-11-23_overlays/overlay_05.png | SiteA/102 | /Users/enesdemir/Documents/ap_articles/data/PTX-498-v2-fix/SiteA/102.4.img.nii.gz | /Users/enesdemir/Documents/ap_articles/data/PTX-498-v2-fix/SiteA/102.5.mask.nii.gz | False | 1.2.276.0.7230010.3.1.4.8323329.4982.1517875185.837576 | /Users/enesdemir/Documents/ap_articles/data/siim-acr/dicom-images-train/1.2.276.0.7230010.3.1.2.8323329.4982.1517875185.837575/1.2.276.0.7230010.3.1.3.8323329.4982.1517875185.837574/1.2.276.0.7230010.3.1.4.8323329.4982.1517875185.837576.dcm | False | — |


## Notes

- B items are the Section 9 rows of baseline_correction.md; '9.4-SIIM' is the SIIM ImageId -> file mapping row of the Section 9.4 table.
- 9.1 is a human decision: the PNG overlays are inputs, the status stays OPEN.
- This report never asserts that a decode is correct; it records equality.

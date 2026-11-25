# Sharpness Matters: Rethinking the Impact of Image Resolution on Medical Image Classification

[![Python 3.8.10](https://img.shields.io/badge/python-3.8.10-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Datasets Used](#datasets-used)
- [Prepare Config File](#prepare-config-file)
- [Usage](#usage)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)
- [Acknowledgments](#acknowledgments)

## Overview

This repository contains the official implementation of our paper **"Sharpness Matters: Rethinking the Impact of Image Resolution on Medical Image Classification"**.

Training deep learning chest radiograph classifiers at higher resolutions resulted in better performance generalizability to external test sets and more precise explanations, highlighting the importance of training data image resolution for clinical AI systems.

<!-- Pipeline overview with enforced white background -->
<div style="background:#ffffff;padding:8px;border:1px solid #ddd;display:inline-block;">
    <img src="assets/overview.png" alt="Sharpness Matters Pipeline" style="display:block;background:#ffffff;">
</div>

## Installation

> **Branch `stage2-corrected`:** corrected baseline-CV (B) of the PTX pipeline. The environment is pinned in `uv.lock`; `pyproject.toml` is the single source and `requirements.txt` is exported from the lock.

### Prerequisites
- [uv](https://docs.astral.sh/uv/) (tested with 0.7.3)
- Python 3.8.x (`~=3.8.10`; uv installs 3.8.20 if missing)
- CUDA or Apple MPS GPU recommended for training

### Quick Start

```bash
git clone https://github.com/EnesDemir143/sharpness-matters.git
cd sharpness-matters
git switch stage2-corrected

uv python install 3.8.20
uv sync --python 3.8.20              # creates .venv from uv.lock (torch 2.1.0, torchvision 0.16.0)
uv sync --python 3.8.20 --extra dev  # optional: pytest for the test suite
```

Run every command below from the repository root with `uv run` (or `source .venv/bin/activate` and drop `uv run`).

`pip install -r requirements.txt` installs the same pinned versions but uv is the supported path.

### Tests

```bash
uv run python -m pytest -q tests/corrected          # synthetic data only, no downloads
uv run python -m pytest -q tests/data_integration   # read-only checks on real data + frozen manifests; skipped when the data is absent
```

## Datasets Used

This project uses five public datasets:

### SIIM-ACR Pneumothorax Challenge Dataset
- **Source**: [Kaggle Competition](https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation)
- **Images**: ~12,000 chest X-rays
- **Format**: DICOM files

### RSNA Pneumonia Detection Challenge Dataset  
- **Source**: [Kaggle Competition](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge)
- **Images**: ~26,000 chest X-rays  
- **Format**: DICOM files

### SIIM-FISABIO-RSNA COVID-19 Detection Dataset
- **Source**: [Kaggle Competition](https://www.kaggle.com/c/siim-covid19-detection)
- **Images**: ~6,400 chest CT/X-ray studies
- **Format**: DICOM files

### VinBigData Chest X-ray Abnormalities Detection Dataset
- **Source**: [Kaggle Competition](https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection)
- **Images**: ~18,000 chest X-rays
- **Format**: DICOM files
- **Preprocessing**: the stage-2 loader honors `PhotometricInterpretation`; a `MONOCHROME1` file is inverted to `MONOCHROME2` polarity before per-image min-max.

### PTX498 Dataset
- **Source**: [DeepSDM](https://github.com/wangyunpengbio/DeepSDM)
- **Images**: 498 chest X-rays with expert annotations
- **Format**: NIFTI files

## Prepare Config File

Edit `sharpness_matters/config/config.yaml`:

```yaml
ptx:
  siim_acr:
    root_dir: "/data/siim-acr"            # contains dicom-images-train/ and train-rle.csv
  vindr:
      root_dir: "/data/vindr/train"       # <image_id>.dicom files
  ptx498:
      root_dir: "/data/ptx498"            # SiteA/ SiteB/ SiteC/ NIfTI files
  model:
      checkpoint_dir: "/data/checkpoints/ptx"
  vindr_annotations_csv: "/data/vindr/train.csv"
  corrected:                              # B settings; keep the author values (baseline_correction.md §11)
      ...
```

The `ptx.corrected` block holds the pinned B settings (folds, epochs, patience, LR, batch size, resize policy, OOF target sensitivity, E5 seeds). Do not edit it for a B run.

## Usage — corrected baseline (B), in order

Steps 1–6 need no training. Step 7 trains and downloads ImageNet weights.

**1. Pre-run data checks (read-only):** writes `reports/data_checks_<date>.md/.json` and orientation overlays. Every check ends as `ISSUE`, `NO CHANGE NEEDED` or `OPEN`; resolve `ISSUE`/`OPEN` before continuing.
```bash
uv run python -m sharpness_matters.ptx.corrected.data_checks --out reports/
```

**2. SIIM manifests:** builds and freezes `manifests/siim_folds.csv` (StratifiedKFold, 5 folds, seed 42) or verifies it if it already exists, then freezes the internal holdout as the exact complement of that manifest inside `train-rle.csv`.
```bash
uv run python -m sharpness_matters.ptx.corrected.make_fold_manifest
uv run python -m sharpness_matters.ptx.corrected.make_siim_holdout
```

`manifests/siim_holdout.csv` (`image_id,label`, 1068 images / 238 positive for the author split) is frozen with `.sha256` and `.record.json`; it records both the `siim_folds.csv` SHA-256 it was derived from and its own CSV SHA-256. Evaluation loads it by membership (CSV row order cannot change the cohort) and fails closed if it overlaps the development manifest bound to the run's `cv_record.json`, if that provenance is absent or mismatched, if the creation record lacks its `csv_sha256`, or if the holdout is not exactly the development complement. Training verifies that the fold frame it trains on equals the hashed file and records `fold_manifest_provenance_version` next to `fold_manifest_sha256`; older records (no version) are unverifiable and cannot evaluate the holdout.

**2b. PTX-498 cohort (E3; E5 is scored by run A):** freezes `manifests/ptx498_cohort.csv` (`image_id,site,case`, 498 cases: 247+179+72) or verifies it if it already exists; the builder rejects a root whose SiteA/B/C case sets differ from the reference before anything is written.
```bash
uv run python -m sharpness_matters.ptx.corrected.make_ptx498_manifest
```

**3. Reference-like E5 manifest (record only: E5 is evaluated by run A only; B does not evaluate E5 — decision of 25 Kasım 2026):** runs the unmodified author script `create_vin_dr_subset.py` once (fixed `PYTHONHASHSEED`) and freezes `manifests/reference_like_combined.csv`, both lists and `N_dup`.
```bash
uv run python -m sharpness_matters.ptx.corrected.make_e5 reference
```

**4. E5-clean manifest (record only, same decision as step 3):** derived from the frozen lists (positives removed from negatives, refilled to 594) → `manifests/reference_like_clean.csv`.
```bash
uv run python -m sharpness_matters.ptx.corrected.make_e5 clean
```

**5. Multi-draw sensitivity record (optional record, never an E5 input):**
```bash
uv run python -m sharpness_matters.ptx.corrected.e5_multidraw --base-seed 0 --draws 20 --hashseeds 0,1,2
```

**5b. E2 and E4 manifests:** VinDr-CXR v1.0.0 official test (PhysioNet `image_labels_test.csv` + `test/*.dicom`) → `manifests/vindr_test.csv`; CheXlocalize (CheXpert val/test images + label CSVs + `gt_segmentations_*.json`) → `manifests/chexlocalize_cls.csv`, `manifests/chexlocalize_xai.csv`.
```bash
uv run python -m sharpness_matters.ptx.corrected.make_vindr_test          # E2 official-test subset: 5-radiologist consensus (needs test/*.dicom)
uv run python -m sharpness_matters.ptx.corrected.make_vindr_test --wide   # E2 all 18,000 images: train part majority >=2/3 readers, sensitivity label >=1/3 (needs train/ + test/)
uv run python -m sharpness_matters.ptx.corrected.make_chexlocalize        # E4 (needs CheXpert val/test images + label CSVs)
```

All manifests are frozen with `.sha256` + `.record.json`; an existing manifest is loaded, never rebuilt, and a hash mismatch stops the run. Commit `manifests/` after this step.

**6. (Optional) demographics over unique image IDs:**
```bash
uv run python sharpness_matters/ptx/datasets/extract_demographics.py --dataset siim_acr
uv run python sharpness_matters/ptx/datasets/extract_demographics.py --dataset vindr --vindr-manifest manifests/reference_like_clean.csv
```

**7. Training (512² only, 5 folds):** writes per-fold best (min `val/loss`) checkpoints, `cv_record.json` (checkpoint hashes, best epochs, fold AUROC, `S(h)`, OOF threshold τ) and `oof_predictions.csv` under `checkpoint_dir`.
```bash
uv run python -m sharpness_matters.ptx.train.train_kfold --model_name densenet
uv run python -m sharpness_matters.ptx.train.train_kfold --model_name resnet
```

**8. Classification evaluation (ID-keyed predictions, decision `p >= τ`):** E5 is not evaluated on B (run A owns it — decision of 25 Kasım 2026), so no E5 mode or `--e5_manifest` option exists.
```bash
uv run python -m sharpness_matters.ptx.evaluation.evaluate_kfold --validation_mode holdout
uv run python -m sharpness_matters.ptx.evaluation.evaluate_kfold --validation_mode vindr_test     # E2 official-test subset
uv run python -m sharpness_matters.ptx.evaluation.evaluate_kfold --validation_mode chexlocalize   # E4 classification
uv run python -m sharpness_matters.ptx.evaluation.evaluate_kfold --validation_mode vindr_wide     # E2 all 18,000 images (+ label_any sensitivity AUROC)
```

**9. Saliency evaluation (GT kept on its reference grid; decision-independent K1 cohort):**
```bash
uv run python -m sharpness_matters.ptx.evaluation.evaluate_saliency       # SIIM internal test
uv run python -m sharpness_matters.ptx.evaluation.evaluate_saliency_ood   # PTX-498
uv run python -m sharpness_matters.ptx.evaluation.evaluate_saliency_chexlocalize   # E4 XAI
```

**10. Statistics (pre-registered, hash-locked plan):** copy `sharpness_matters/ptx/corrected/analysis_plan.example.json`, fill in the inputs, lock it before looking at results, then run.
```bash
uv run python -m sharpness_matters.ptx.corrected.analysis lock plans/analysis_plan.json
uv run python -m sharpness_matters.ptx.corrected.analysis run plans/analysis_plan.json --out-dir reports/analysis
```

Classification inputs name their probability column per side: use `prob_column` when both CSVs use the same name, or `prob_column_a`/`prob_column_b` when they differ (for example the A branch's stage-1 `score` vs the B branch's `prob`). Both names are part of the locked plan, and a missing column raises instead of falling back to the other side.
```json
"inputs": {"a": ".../stage1_scores.csv", "b": ".../stage2_ids.csv",
           "prob_column_a": "score", "prob_column_b": "prob", "label_column": "label"}
```

Every `saliency_paired` comparison also registers `p_method`, so the p-value is never SciPy's data-dependent `method="auto"`: `"approx"` is the explicit normal approximation (use it for large cohorts, e.g. n=498), `"permutation"` is the exhaustive 2^n sign-flip test on the tie-averaged signed ranks (small continuous cohorts, at most 20 non-zero differences, requires `"zero_method": "wilcox"`), and `"mcnemar_exact"` is the two-sided exact binomial on the discordant pairs of a binary 0/1 hit metric (`zero_method` is still required and recorded but not applied, since a zero difference is a concordant pair). A missing key, `"auto"`, `"exact"` or another value fails validation, and `p_method` on a `classification_auroc` comparison is rejected (DeLong is fixed by the plan).
```json
"zero_method": "wilcox", "p_method": "approx"
```

Comparisons whose cohort repeats patients (the CheXlocalize E4 cohorts: 720 images / 700 patients and 16 images / 15 patients) register `"uncertainty_unit": "patient"` plus `"patient_manifest"` in `inputs` — a frozen `image_id`/`patient_id` CSV with its `.sha256` and `.record.json`, loaded with `load_frozen`, so every aligned image must map to a non-empty patient or the run raises before writing anything. The estimate stays the pooled paired difference, but `classification_auroc` takes a cluster percentile bootstrap that resamples patients and keeps all images of each drawn patient; its p-value is still the image-level paired DeLong test, which cannot model the clustering, and the row says so with `p_clustering_aware: false`. `saliency_paired` at patient level collapses to one value per patient (the mean of that patient's `b - a`) and runs the registered `p_method` and bootstrap on those patient values, so `n` counts patients (`n_images` is also recorded) and `mcnemar_exact` is rejected there. Rows carry `n_patients`, `n_multi_image_patients`, `patient_manifest_sha256` and `p_clustering_aware`; image-unit rows leave the patient fields empty.
```json
"uncertainty_unit": "patient", "inputs": {"a": "...", "b": "...",
  "patient_manifest": "manifests/chexlocalize_cls.csv"}
```

The author's `p_value_classification.py`, `p_value_saliency.py`, `train_sweep.py` and `compare_cv_tune.py` are not part of B.

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{kulkarni2024sharpness,
  title={Sharpness Matters: Rethinking the Impact of Image Resolution on Medical Image Classification},
  author={Kulkarni, Aditya and Zech, John and Yi, Paul},
  journal={Journal Name},
  year={2025},
  volume={XX},
  pages={XX-XX}
}
```

## Contact

- **Repository Issues**: Please use [GitHub Issues](https://github.com/your-username/sharpness_matters/issues) for bug reports and feature requests
- **General Questions**: Open a [Discussion](https://github.com/your-username/sharpness_matters/discussions)
- **Paper Authors**: Aditya Kulkarni, John Zech, Paul Yi

## Acknowledgments

- St. Jude Children's Research Hospital for computational resources
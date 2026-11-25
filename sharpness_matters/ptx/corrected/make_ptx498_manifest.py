"""Frozen PTX-498 external cohort (E3/E5), verified against the reference case sets.

# B (PTX-18 / Section 7.3, follow-up): ``OODTestDataset`` derived the cohort from
#   ``os.listdir`` of SiteA/B/C. An existing but EMPTY SiteC/ silently yielded 426 IDs
#   instead of 498 (247+179+72), and a case that only had PNG files, or an image
#   without its mask, was accepted at parse time and failed during inference. New
#   behaviour: ``manifests/ptx498_cohort.csv`` (``image_id,site,case``) is frozen once
#   with ``.sha256`` and ``.record.json``. This builder requires exactly the reference
#   case set per site (each case with both NIfTI files) and refuses before writing
#   anything; evaluation loads the frozen file and never rebuilds it.
#   check: tests/corrected/test_ptx498_cohort.py

Usage:
    python -m sharpness_matters.ptx.corrected.make_ptx498_manifest
    python -m sharpness_matters.ptx.corrected.make_ptx498_manifest --root-dir /data/PTX-498-v2-fix
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import click
import pandas as pd

from sharpness_matters.config.load_config import cfg
from sharpness_matters.ptx.corrected.frozen import (
    load_or_create_frozen,
    manifest_dir,
    sha256_file,
)
from sharpness_matters.ptx.datasets.ptx498_dataset import (
    COHORT_MANIFEST_NAME,
    IMAGE_SUFFIX,
    MASK_SUFFIX,
    SITES,
    default_cohort_manifest,
)

# Reference cohort of Zenodo record 8266529, release "PTX-498-v2-fix"
# (PTX-498-v2-fix.zip): cases are numbered 1..N per site, and every case ships
# N.1.img.png, N.2.mask.png, N.3.merge.png, N.4.img.nii.gz, N.5.mask.nii.gz.
REFERENCE_CASE_COUNTS: Dict[str, int] = {"A": 247, "B": 179, "C": 72}
MANIFEST_COLUMNS = ["image_id", "site", "case"]
_MAX_EXAMPLES = 5


def _examples(items: List[str]) -> str:
    """Up to five example values, with the remainder counted."""
    shown = items[:_MAX_EXAMPLES]
    text = ", ".join(shown)
    if len(items) > _MAX_EXAMPLES:
        text += f", ... ({len(items) - _MAX_EXAMPLES} more)"
    return text


def reference_cases(site: str) -> List[str]:
    """Reference case numbers of a site, as the strings used on disk."""
    return [str(number) for number in range(1, REFERENCE_CASE_COUNTS[site] + 1)]


def _nifti_suffixes_by_case(site_dir: Path) -> Dict[str, List[str]]:
    """Case stem -> the NIfTI suffixes that exist for it in ``site_dir``."""
    found: Dict[str, List[str]] = {}
    for entry in site_dir.iterdir():
        if not entry.is_file():
            continue
        for suffix in (IMAGE_SUFFIX, MASK_SUFFIX):
            if entry.name.endswith(suffix):
                found.setdefault(entry.name[: -len(suffix)], []).append(suffix)
    return found


def verify_reference_root(root_dir: Union[str, Path]) -> Dict[str, int]:
    """Require exactly the reference case set per site, each case with both NIfTIs.

    Returns the per-site case counts. Raises ``RuntimeError`` naming the counts and
    examples of every offending case, so nothing is frozen from a partial cohort.
    """
    root = Path(root_dir)
    if not root.is_dir():
        raise RuntimeError(f"PTX-498 root directory {root} does not exist")
    problems: List[str] = []
    for site in SITES:
        expected = set(reference_cases(site))
        site_dir = root / f"Site{site}"
        if not site_dir.is_dir():
            problems.append(
                f"Site{site} directory is missing under {root}; it must hold "
                f"{len(expected)} case(s) (e.g. {_examples([f'{site}_{case}' for case in reference_cases(site)])})"
            )
            continue
        on_disk = _nifti_suffixes_by_case(site_dir)
        missing = sorted(expected - set(on_disk))
        if missing:
            problems.append(
                f"Site{site}: {len(missing)} reference case(s) are absent "
                f"(e.g. {_examples([f'{site}_{case}' for case in missing])})"
            )
        extra = sorted(set(on_disk) - expected)
        if extra:
            problems.append(
                f"Site{site}: {len(extra)} case(s) on disk are not in the reference "
                f"set (e.g. {_examples([f'{site}_{case}' for case in extra])})"
            )
        incomplete = sorted(
            case for case in expected & set(on_disk) if len(set(on_disk[case])) < 2
        )
        if incomplete:
            details = [
                f"{site}_{case} (missing "
                + ", ".join(
                    suffix
                    for suffix in (IMAGE_SUFFIX, MASK_SUFFIX)
                    if suffix not in on_disk[case]
                )
                + ")"
                for case in incomplete
            ]
            problems.append(
                f"Site{site}: {len(incomplete)} case(s) lack one NIfTI file "
                f"(e.g. {_examples(details)})"
            )
    if problems:
        raise RuntimeError(
            f"PTX-498 root {root} does not match the reference cohort "
            f"({' + '.join(str(REFERENCE_CASE_COUNTS[site]) for site in SITES)} = "
            f"{sum(REFERENCE_CASE_COUNTS.values())} cases); nothing was written: "
            + "; ".join(problems)
        )
    return {site: REFERENCE_CASE_COUNTS[site] for site in SITES}


def build_cohort_frame(root_dir: Union[str, Path]) -> pd.DataFrame:
    """Reference cohort frame in loader order: SiteA/B/C, case string ascending.

    The order is the old loader's order (``sorted`` on the case string: A_1, A_10,
    A_2, ...), so saliency caches and E5 rows keep their existing row order.
    """
    verify_reference_root(root_dir)
    rows = [
        {"image_id": f"{site}_{case}", "site": site, "case": case}
        for site in SITES
        for case in sorted(reference_cases(site))
    ]
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


def _provenance(root_dir: Union[str, Path]) -> Dict[str, Any]:
    return {
        "item": "PTX-498 external test cohort (E3/E5), frozen by case",
        "source_archive": "PTX-498-v2-fix.zip",
        "source_record": "Zenodo record 8266529, release PTX-498-v2-fix",
        "root_dir": str(root_dir),
        "per_site_counts": {site: REFERENCE_CASE_COUNTS[site] for site in SITES},
        "n_cases": sum(REFERENCE_CASE_COUNTS.values()),
        "reference_cases": {
            site: f"1..{REFERENCE_CASE_COUNTS[site]}" for site in SITES
        },
        "order": "SiteA, SiteB, SiteC; case number compared as a string (lexicographic)",
        "required_files": [IMAGE_SUFFIX, MASK_SUFFIX],
    }


def prepare_cohort(
    root_dir: Union[str, Path],
    out_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Verify the root against the reference cohort, then freeze (or load) the csv."""
    root = Path(root_dir)
    path = Path(out_path) if out_path is not None else default_cohort_manifest()
    frame = build_cohort_frame(root)  # raises before anything is written
    return load_or_create_frozen(path, lambda: frame, _provenance(root))


@click.command()
@click.option(
    "--root-dir",
    type=click.Path(file_okay=False),
    default=None,
    help="PTX-498 root holding SiteA/B/C (default: ptx.ptx498.root_dir)",
)
@click.option(
    "--manifest-dir",
    "manifest_directory",
    type=click.Path(file_okay=False),
    default=None,
    help="directory that holds ptx498_cohort.csv (default: configured manifests/)",
)
def main(root_dir: Optional[str], manifest_directory: Optional[str]) -> None:
    """Freeze (first run) or verify (later runs) manifests/ptx498_cohort.csv."""
    root = Path(root_dir) if root_dir is not None else Path(str(cfg.ptx.ptx498.root_dir))
    out_dir = (
        Path(manifest_directory) if manifest_directory is not None else manifest_dir()
    )
    path = out_dir / COHORT_MANIFEST_NAME
    frame = prepare_cohort(root, path)
    counts = ", ".join(
        f"{site}={REFERENCE_CASE_COUNTS[site]}" for site in SITES
    )
    print(
        f"{COHORT_MANIFEST_NAME}: {len(frame)} cases ({counts}); "
        f"root {root}; sha256={sha256_file(path)}"
    )


if __name__ == "__main__":
    main()

"""E5 multi-draw sensitivity — baseline_correction.md Section 4.2, item 3.

This is a separate record, not a correction step and not an input to E5:

* it never changes ``reference_like_combined.csv`` or ``reference_like_clean.csv``
  and is never read by them (the E5 manifests come from ``make_e5.reference`` /
  ``make_e5.clean`` only);
* it does not select a draw and does not replace the author's own run; it only
  records how sensitive ``N_dup`` is to the draw seed and to ``PYTHONHASHSEED``.

The author's selector draws ``random.sample`` from ``list(set(candidates))``, whose
order depends on ``PYTHONHASHSEED``. Each draw therefore runs in a subprocess with the
pre-registered ``PYTHONHASHSEED``, rebuilding ``list(set(...))`` in that process (the
author's order), not a sorted list — sorting would erase the hash-seed effect that is
being measured.

# B (PTX-03 / Section 4.2 item 3): current -> no multi-draw record existed (a single
#   run was treated as the cohort); new -> pre-registered (seeds x hash seeds) draws
#   under manifests/e5_multidraw/, each reporting N_dup; output is refused if it
#   exists, and nothing here feeds the E5 manifests.
#   check: tests/corrected/test_data.py::test_e5_multidraw_*
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence

import click

from sharpness_matters.config.load_config import cfg
from sharpness_matters.ptx.corrected.frozen import (
    manifest_dir,
    repo_root,
    sha256_file,
)
from sharpness_matters.ptx.corrected.make_e5 import (
    AUTHOR_SCRIPT,
    EXPECTED_NEG_COUNT,
    NEGATIVE_RULE,
    verify_negative_rule,
    write_json,
)

OUTPUT_DIRNAME = "e5_multidraw"
PREREGISTRATION_NAME = "preregistration.json"
SUMMARY_NAME = "summary.json"
DRAWS_DIRNAME = "draws"
DRAW_TEMPLATE = "draw_{index:03d}.json"

ROLE = (
    "record-only multi-draw sensitivity analysis (Section 4.2 item 3); never an "
    "input to E5/E5-clean and no draw is selected"
)


def output_dir() -> Path:
    return manifest_dir() / OUTPUT_DIRNAME


def build_draw_shim(
    annotations_csv: Path, seed: int, k: int, output_json: Path
) -> str:
    """Build the ``python -c`` source for one draw (author-exact set ordering)."""
    csv_literal = repr(str(Path(annotations_csv).resolve()))
    out_literal = repr(str(Path(output_json).resolve()))
    return "\n".join(
        [
            "import hashlib",
            "import json",
            "import os",
            "import random",
            "import pandas as pd",
            "",
            f"_CSV = {csv_literal}",
            f"_OUT = {out_literal}",
            f"_SEED = {int(seed)}",
            f"_K = {int(k)}",
            "",
            "frame = pd.read_csv(_CSV)",
            "missing = {'image_id', 'class_name'} - set(frame.columns)",
            "if missing:",
            "    raise SystemExit(f'annotation csv missing columns {sorted(missing)}')",
            "classes = frame['class_name'].astype(str)",
            "ids = frame['image_id'].astype(str)",
            "candidates = ids[classes != 'Pneumothorax'].tolist()",
            "positives = ids[classes == 'Pneumothorax'].tolist()",
            "# author rule: list(set(...)); the order depends on PYTHONHASHSEED",
            "candidates = list(set(candidates))",
            "positives = list(set(positives))",
            "if len(candidates) < _K:",
            "    raise SystemExit(f'draw pool {len(candidates)} < k={_K}')",
            "picked = random.Random(_SEED).sample(candidates, _K)",
            "payload = {",
            "    'seed': _SEED,",
            "    'k': _K,",
            "    'pythonhashseed': os.environ.get('PYTHONHASHSEED'),",
            "    'population_size': len(candidates),",
            "    'population_sha256': hashlib.sha256(",
            "        '\\n'.join(candidates).encode('utf-8')",
            "    ).hexdigest(),",
            "    'positive_count': len(positives),",
            "    'positives_sha256': hashlib.sha256(",
            "        '\\n'.join(positives).encode('utf-8')",
            "    ).hexdigest(),",
            "    'sample_ids': picked,",
            "    'sample_sha256': hashlib.sha256(",
            "        '\\n'.join(picked).encode('utf-8')",
            "    ).hexdigest(),",
            "    'n_dup': len(set(picked) & set(positives)),",
            "}",
            "with open(_OUT, 'w') as handle:",
            "    json.dump(payload, handle, indent=2, sort_keys=True)",
            "",
        ]
    )


def _seed_list(
    seeds: Sequence[int] = None,
    base_seed: int = None,
    draws: int = None,
) -> List[int]:
    """Draw seeds from an explicit list or from ``base_seed`` + ``draws``."""
    if seeds:
        if base_seed is not None or draws is not None:
            raise ValueError("Give either --seeds or --base-seed/--draws, not both")
        return [int(seed) for seed in seeds]
    if base_seed is None or draws is None:
        raise ValueError(
            "Pre-register the draws: pass --seeds, or both --base-seed and --draws"
        )
    if int(draws) < 1:
        raise ValueError("--draws must be >= 1")
    return [int(base_seed) + index for index in range(int(draws))]


def run_multidraw(
    seeds: Sequence[int] = None,
    hashseeds: Sequence[str] = None,
    annotations_csv: str = None,
    k: int = EXPECTED_NEG_COUNT,
    base_seed: int = None,
    draws: int = None,
    out_dir: Path = None,
) -> Dict[str, object]:
    """Pre-register and run the multi-draw sensitivity analysis.

    Every (seed, ``PYTHONHASHSEED``) combination is one draw in its own subprocess.
    The output directory must not exist (nothing is ever overwritten), the
    pre-registration record is written before the first draw, and the summary holds
    all ``N_dup`` values with their distribution.
    """
    verify_negative_rule()
    seeds = _seed_list(seeds, base_seed, draws)
    hashseeds = [str(value) for value in (hashseeds or [])]
    if not hashseeds:
        raise ValueError("Pre-register at least one PYTHONHASHSEED value (--hashseeds)")
    if int(k) < 1:
        raise ValueError("k must be >= 1")

    annotations = Path(str(annotations_csv or cfg.ptx.vindr_annotations_csv))
    if not annotations.exists():
        raise FileNotFoundError(f"Annotation CSV does not exist: {annotations}")

    target = Path(out_dir) if out_dir is not None else output_dir()
    if target.exists():
        raise RuntimeError(
            f"Multi-draw output {target} already exists. This record is never "
            "overwritten; remove the directory deliberately to run a new set of draws."
        )
    draws_dir = target / DRAWS_DIRNAME
    draws_dir.mkdir(parents=True)

    n_draws = len(seeds) * len(hashseeds)
    preregistration = {
        "item": "Section 4.2 item 3 E5 multi-draw sensitivity (record only)",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "written_before_draws": True,
        "n_draws": n_draws,
        "seeds": [int(seed) for seed in seeds],
        "hashseeds": hashseeds,
        "k": int(k),
        "rule": NEGATIVE_RULE,
        "population_rule": (
            "author-exact: list(set(candidates)) under the draw's PYTHONHASHSEED, "
            "then random.Random(seed).sample(population, k)"
        ),
        "annotations_csv": str(annotations),
        "annotations_csv_sha256": sha256_file(annotations),
        "author_script": str(AUTHOR_SCRIPT),
        "author_script_sha256": sha256_file(AUTHOR_SCRIPT),
        "module_sha256": sha256_file(Path(__file__)),
        "role": ROLE,
    }
    preregistration_path = target / PREREGISTRATION_NAME
    write_json(preregistration_path, preregistration)

    results: List[Dict[str, object]] = []
    index = 0
    for seed in seeds:
        for hashseed in hashseeds:
            draw_path = draws_dir / DRAW_TEMPLATE.format(index=index)
            env = os.environ.copy()
            env["PYTHONHASHSEED"] = hashseed
            try:
                subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        build_draw_shim(annotations, int(seed), int(k), draw_path),
                    ],
                    cwd=str(repo_root()),
                    env=env,
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except subprocess.CalledProcessError as exc:
                stderr = (exc.stderr or "").strip().splitlines()[-5:]
                raise RuntimeError(
                    f"Draw {index} (seed={seed}, PYTHONHASHSEED={hashseed}) failed. "
                    f"Exit code {exc.returncode}. Last stderr lines: {stderr}"
                ) from exc
            payload = json.loads(draw_path.read_text(encoding="utf-8"))
            if str(payload.get("pythonhashseed")) != hashseed:
                raise RuntimeError(
                    f"Draw {index} ran with PYTHONHASHSEED="
                    f"{payload.get('pythonhashseed')} instead of {hashseed}"
                )
            results.append(
                {
                    "draw": index,
                    "seed": int(seed),
                    "hashseed": hashseed,
                    "k": int(payload["k"]),
                    "n_dup": int(payload["n_dup"]),
                    "population_size": int(payload["population_size"]),
                    "positive_count": int(payload["positive_count"]),
                    "population_sha256": payload["population_sha256"],
                    "positives_sha256": payload["positives_sha256"],
                    "sample_sha256": payload["sample_sha256"],
                    "file": str(draw_path),
                }
            )
            index += 1

    n_dup_values = [int(row["n_dup"]) for row in results]
    distribution: Dict[str, int] = {}
    for value in n_dup_values:
        distribution[str(value)] = distribution.get(str(value), 0) + 1
    summary = {
        "item": "Section 4.2 item 3 E5 multi-draw sensitivity (record only)",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "role": ROLE,
        "preregistration": str(preregistration_path),
        "preregistration_sha256": sha256_file(preregistration_path),
        "annotations_csv": str(annotations),
        "annotations_csv_sha256": sha256_file(annotations),
        "n_draws": len(results),
        "k": int(k),
        "seeds": [int(seed) for seed in seeds],
        "hashseeds": hashseeds,
        "n_dup_values": n_dup_values,
        "n_dup_distribution": distribution,
        "n_dup_min": min(n_dup_values),
        "n_dup_max": max(n_dup_values),
        "n_dup_mean": float(sum(n_dup_values)) / len(n_dup_values),
        "unique_sample_sets": len({row["sample_sha256"] for row in results}),
        "draws": results,
    }
    write_json(target / SUMMARY_NAME, summary)
    return summary


def _parse_int_list(text: str, name: str) -> List[int]:
    values = [part.strip() for part in str(text).split(",") if part.strip()]
    if not values:
        raise click.BadParameter(f"{name} must not be empty")
    return [int(value) for value in values]


def _parse_str_list(text: str) -> List[str]:
    values = [part.strip() for part in str(text).split(",") if part.strip()]
    if not values:
        raise click.BadParameter("--hashseeds must not be empty")
    return values


@click.command()
@click.option(
    "--seeds",
    default=None,
    help="Comma-separated draw seeds (alternative to --base-seed/--draws)",
)
@click.option(
    "--base-seed",
    type=int,
    default=None,
    help="First draw seed; requires --draws",
)
@click.option(
    "--draws",
    type=int,
    default=None,
    help="Number of consecutive seeds starting at --base-seed",
)
@click.option(
    "--hashseeds",
    required=True,
    help="Comma-separated PYTHONHASHSEED values (one subprocess per value)",
)
@click.option(
    "--annotations-csv",
    default=None,
    help="Annotation CSV (default: cfg.ptx.vindr_annotations_csv)",
)
@click.option(
    "--k",
    type=int,
    default=EXPECTED_NEG_COUNT,
    show_default=True,
    help="Negative sample size of each draw (the author's 594)",
)
@click.option(
    "--out-dir",
    default=None,
    help=f"Output directory (default: manifests/{OUTPUT_DIRNAME})",
)
def cli(
    seeds: str,
    base_seed: int,
    draws: int,
    hashseeds: str,
    annotations_csv: str,
    k: int,
    out_dir: str,
) -> None:
    """Run the pre-registered multi-draw sensitivity record (never an E5 input)."""
    summary = run_multidraw(
        seeds=_parse_int_list(seeds, "--seeds") if seeds else None,
        base_seed=base_seed,
        draws=draws,
        hashseeds=_parse_str_list(hashseeds),
        annotations_csv=annotations_csv,
        k=k,
        out_dir=Path(out_dir) if out_dir else None,
    )
    click.echo(json.dumps(summary, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    cli()

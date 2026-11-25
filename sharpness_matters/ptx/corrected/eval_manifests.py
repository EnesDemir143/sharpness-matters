"""Frozen-manifest verification shared by the B evaluation entry points.

# B (Section 4.7 / 6.1): evaluation may only read an artifact that was frozen before
#   the run. The author's evaluators read whatever path the caller passed, so a
#   renamed manifest produced a full metrics CSV under another protocol row. New:
#   every evaluation cohort is loaded through `frozen.load_frozen` (csv + .sha256 +
#   .record.json, hash checked) and the loaded dataset is compared row-for-row with
#   that frozen frame.
#   check: tests/corrected/test_manifest_eval_integrity.py
# E5 is evaluated only by run A (25 Kasım 2026 decision); no E5 binding lives here.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def verify_dataset_against_frozen(
    dataset: Any,
    frame: pd.DataFrame,
    *,
    what: str,
    id_attribute: str = "image_ids",
    labels_attribute: str = "labels",
) -> None:
    """Fail unless ``dataset`` exposes exactly the frozen frame's ids, order and labels.

    Evaluation datasets re-read the manifest path themselves, so this is the second
    half of the hash check: the rows the model will actually see must be the rows that
    were verified. A reordered, shortened or relabelled cohort raises instead of
    producing a prediction CSV under the frozen manifest's name.
    """
    dataset_ids = [str(value) for value in getattr(dataset, id_attribute)]
    frozen_ids = [str(value) for value in frame["image_id"]]
    if dataset_ids != frozen_ids:
        raise RuntimeError(
            f"{what}: loader order differs from the frozen manifest "
            f"({len(dataset_ids)} dataset ids vs {len(frozen_ids)} manifest ids)"
        )
    dataset_labels = getattr(dataset, labels_attribute)
    frozen_labels = {
        str(image_id): int(label)
        for image_id, label in zip(frame["image_id"], frame["label"])
    }
    mismatched = [
        image_id
        for image_id in dataset_ids
        if int(dataset_labels[image_id]) != frozen_labels[image_id]
    ]
    if mismatched:
        raise RuntimeError(
            f"{what}: loader labels differ from the frozen manifest for "
            f"{len(mismatched)} ids, e.g. {mismatched[:5]}"
        )

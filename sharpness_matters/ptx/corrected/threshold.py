"""OOF operating threshold for the Stage-2 (B) baseline.

B (§5.8 / protocol §5.3): current -> the author's evaluation compares at a fixed 0.5
(`classification_eval.py` L183, L199) and trains no OOF record; problem -> 0.5 is not the
baseline's own operating point and cannot be justified per model/resolution; new
behaviour -> the baseline threshold is the highest OOF probability tau whose decision
`p >= tau` still reaches the nominal target sensitivity, and tau is one of the observed
OOF probabilities (no interpolation, no threshold outside the data).
Check: tests/corrected/test_train.py::test_oof_threshold_*.
"""

from typing import Sequence, Union

import numpy as np

ArrayLike = Union[Sequence[float], np.ndarray]


def oof_threshold(
    labels: ArrayLike, probs: ArrayLike, target_sensitivity: float
) -> float:
    """Highest observed probability tau with OOF sensitivity of `p >= tau` >= target.

    Ties at tau count as positive decisions (`p >= tau`). Because tau is restricted to
    the observed OOF probabilities, tau = min(probs) always yields sensitivity 1.0, so
    every target in (0, 1] is attainable as long as at least one positive exists.

    Raises:
        ValueError: empty input, length mismatch, non-finite probabilities, labels
            outside {0, 1}, a target outside (0, 1] (unattainable), or no positive
            label (sensitivity is undefined).
    """
    labels = np.asarray(labels)
    probs = np.asarray(probs, dtype=np.float64)

    if labels.shape != probs.shape:
        raise ValueError(
            f"labels and probs must have the same shape, got {labels.shape} and {probs.shape}"
        )
    if labels.size == 0:
        raise ValueError("cannot compute an OOF threshold from an empty array")
    if not np.isfinite(probs).all():
        raise ValueError("probabilities must be finite")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("labels must be binary {0, 1}")
    if not 0.0 < float(target_sensitivity) <= 1.0:
        raise ValueError(
            f"target_sensitivity must be in (0, 1], got {target_sensitivity}"
        )

    positive = labels == 1
    n_pos = int(positive.sum())
    if n_pos == 0:
        raise ValueError("sensitivity is undefined: no positive label in the OOF set")

    taus = np.unique(probs)  # ascending, only observed probabilities
    # Sensitivity is non-increasing in tau, so the last qualifying tau is the highest one.
    true_positives = np.array([int(np.count_nonzero(positive & (probs >= t))) for t in taus])
    sensitivity = true_positives / n_pos
    qualifying = np.flatnonzero(sensitivity >= float(target_sensitivity))
    if qualifying.size == 0:
        # Unreachable for target <= 1 (tau = min(probs) gives sensitivity 1); kept as a
        # guard so a future change to the candidate set cannot silently return nothing.
        raise ValueError(
            f"no observed probability reaches sensitivity {target_sensitivity}"
        )
    return float(taus[qualifying[-1]])

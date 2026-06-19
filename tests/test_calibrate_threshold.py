"""Tests for the pure helpers of the threshold-calibration script."""

from __future__ import annotations

import numpy as np

from scripts.calibrate_threshold import candidate_thresholds, false_positive_rate


def test_false_positive_rate_counts_exceedances() -> None:
    errors = np.array([0.0, 1.0, 2.0, 3.0])
    # Two of four samples exceed 1.5.
    assert false_positive_rate(errors, 1.5) == 0.5
    # Nothing exceeds the max.
    assert false_positive_rate(errors, 3.0) == 0.0


def test_candidate_thresholds_are_ordered_and_labelled() -> None:
    errors = np.linspace(0.0, 1.0, 1000)
    candidates = dict(candidate_thresholds(errors))
    assert {"mu+2sigma", "mu+3sigma", "mu+4sigma", "p99", "p99.9", "p99.99"} <= set(
        candidates
    )
    # Higher sigma multiples and higher percentiles give stricter thresholds.
    assert candidates["mu+2sigma"] < candidates["mu+3sigma"] < candidates["mu+4sigma"]
    assert candidates["p99"] < candidates["p99.9"] < candidates["p99.99"]

from __future__ import annotations

import numpy as np

from mddf.benchmark.operating_point import auroc


def test_auroc_matches_known_values() -> None:
    # perfectly separable
    assert auroc([0.1, 0.2, 0.9, 0.8], [0, 0, 1, 1]) == 1.0
    # perfectly wrong
    assert auroc([0.9, 0.8, 0.1, 0.2], [0, 0, 1, 1]) == 0.0
    # random-ish -> ~0.5
    rng = np.random.default_rng(0)
    s = rng.random(400)
    y = (rng.random(400) > 0.5).astype(int)
    assert 0.4 < auroc(s, y) < 0.6


def test_auroc_degenerate() -> None:
    assert np.isnan(auroc([1.0, 2.0], [0, 0]))
    assert np.isnan(auroc([], []))


def test_auroc_ties() -> None:
    # all identical scores -> 0.5
    assert auroc([0.5, 0.5, 0.5, 0.5], [0, 1, 0, 1]) == 0.5

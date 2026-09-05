from __future__ import annotations

import numpy as np

from mddf.benchmark.operating_point import operating_points


def test_perfectly_separable() -> None:
    scores = np.array([0.1, 0.2, 0.15, 0.9, 0.95, 0.8])
    labels = np.array([0, 0, 0, 1, 1, 1])
    pts = {p.name: p for p in operating_points(scores, labels)}
    p = pts["recall>=1.00"]
    assert p.recall == 1.0
    assert p.false_alarm_rate == 0.0
    assert p.miss_rate == 0.0
    assert pts["f1_optimal"].f1 == 1.0


def test_target_recall_costs_false_alarms() -> None:
    # 10 good, 10 defect, partial overlap
    rng = np.random.default_rng(0)
    good = rng.normal(0.3, 0.1, 50)
    defect = rng.normal(0.6, 0.1, 50)
    scores = np.concatenate([good, defect])
    labels = np.concatenate([np.zeros(50), np.ones(50)])

    pts = {p.name: p for p in operating_points(scores, labels)}
    assert pts["recall>=0.90"].recall >= 0.90
    assert pts["recall>=0.99"].recall >= 0.99
    # higher recall target -> at least as many false alarms
    assert pts["recall>=0.99"].false_alarm_rate >= pts["recall>=0.90"].false_alarm_rate
    assert 0.0 <= pts["recall>=0.90"].false_alarm_rate <= 1.0


def test_degenerate_inputs_return_empty() -> None:
    assert operating_points([], []) == []
    assert operating_points([1.0, 2.0], [0, 0]) == []  # no positives
    assert operating_points([1.0, 2.0], [1, 1]) == []  # no negatives

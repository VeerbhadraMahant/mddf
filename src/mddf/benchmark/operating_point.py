"""Operating-point analysis for an anomaly detector.

AUROC summarises ranking quality; a factory line needs a concrete threshold and
the cost it implies. Given per-image anomaly scores and binary labels
(1 = defect), :func:`operating_points` reports, for each target defect-recall
(sensitivity):

* the score threshold that achieves it,
* precision at that threshold,
* the **false-alarm rate** (FPR) — good parts wrongly rejected,
* the miss rate (FNR = 1 - recall).

Plus the F1-optimal point. No sklearn dependency — sorted-threshold sweep.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

DEFAULT_TARGET_RECALLS = (0.90, 0.95, 0.99, 1.00)


@dataclass(frozen=True)
class OperatingPoint:
    name: str
    threshold: float
    recall: float
    precision: float
    false_alarm_rate: float  # FPR: good parts flagged
    miss_rate: float  # FNR: defects passed
    f1: float


def _sweep(scores: np.ndarray, labels: np.ndarray) -> dict[str, np.ndarray]:
    order = np.argsort(-scores, kind="stable")
    s = scores[order]
    y = labels[order].astype(np.int64)
    pos = int(y.sum())
    neg = int(len(y) - pos)
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    # thresholds = each distinct score (predict "defect" if score >= s[i])
    recall = tp / max(pos, 1)
    precision = tp / np.maximum(tp + fp, 1)
    fpr = fp / max(neg, 1)
    f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
    return {"thr": s, "recall": recall, "precision": precision, "fpr": fpr, "f1": f1}


def operating_points(
    scores: np.ndarray | list[float],
    labels: np.ndarray | list[int],
    *,
    target_recalls: tuple[float, ...] = DEFAULT_TARGET_RECALLS,
) -> list[OperatingPoint]:
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if s.size == 0 or y.sum() == 0 or y.sum() == y.size:
        return []

    sw = _sweep(s, y)
    points: list[OperatingPoint] = []

    for target in target_recalls:
        idx_candidates = np.where(sw["recall"] >= target - 1e-9)[0]
        if idx_candidates.size == 0:
            continue
        i = int(idx_candidates[0])  # smallest set that reaches the target recall
        points.append(
            OperatingPoint(
                name=f"recall>={target:.2f}",
                threshold=float(sw["thr"][i]),
                recall=float(sw["recall"][i]),
                precision=float(sw["precision"][i]),
                false_alarm_rate=float(sw["fpr"][i]),
                miss_rate=float(1.0 - sw["recall"][i]),
                f1=float(sw["f1"][i]),
            )
        )

    j = int(np.argmax(sw["f1"]))
    points.append(
        OperatingPoint(
            name="f1_optimal",
            threshold=float(sw["thr"][j]),
            recall=float(sw["recall"][j]),
            precision=float(sw["precision"][j]),
            false_alarm_rate=float(sw["fpr"][j]),
            miss_rate=float(1.0 - sw["recall"][j]),
            f1=float(sw["f1"][j]),
        )
    )
    return points


def point_to_dict(p: OperatingPoint) -> dict[str, float | str]:
    return asdict(p)


__all__ = [
    "DEFAULT_TARGET_RECALLS",
    "OperatingPoint",
    "operating_points",
    "point_to_dict",
]

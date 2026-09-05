"""Decision threshold + score normalisation.

The adaptive (F1-optimal) threshold is captured at training time. We expose:
  * ``verdict`` — normal / defect at that threshold,
  * ``normalized_score`` — score mapped to [0, 1] with 0.5 exactly at the
    threshold, so a gauge reads intuitively regardless of the raw scale.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

Verdict = str  # "normal" | "defect"


@dataclass(frozen=True)
class ThresholdModel:
    threshold: float
    # Scale of the score distribution; used only to shape the 0-1 mapping.
    scale: float = 1.0

    def normalize(self, score: float) -> float:
        if self.scale <= 0 or not math.isfinite(self.scale):
            return 1.0 if score >= self.threshold else 0.0
        # Logistic centred on the threshold; ~0.5 at threshold, saturating away from it.
        z = (score - self.threshold) / self.scale
        return 1.0 / (1.0 + math.exp(-z))

    def verdict(self, score: float) -> Verdict:
        return "defect" if score >= self.threshold else "normal"


def build(
    threshold: float | None,
    *,
    ref_scores: tuple[float, float] | None = None,
) -> ThresholdModel:
    """``ref_scores`` = (min, max) observed scores, if known, to size the logistic scale."""
    thr = float(threshold) if threshold is not None else 0.5
    scale = 1.0
    if ref_scores:
        lo, hi = ref_scores
        span = max(hi - lo, 1e-6)
        scale = span / 6.0  # ~99% of the span maps into (0, 1)
    return ThresholdModel(threshold=thr, scale=scale)


__all__ = ["ThresholdModel", "Verdict", "build"]

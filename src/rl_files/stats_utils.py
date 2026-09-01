"""
Shared statistical primitives used by both evaluate_experiment_seeds.py (per-episode /
cross-seed CIs on final metrics) and experiment_statistics.py (cross-seed CIs on entire
training curves). Split out from evaluate_experiment_seeds.py so experiment_statistics.py can
import these without an import cycle (evaluate_experiment_seeds.py is the one calling into
experiment_statistics.py, not the other way around).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from scipy import stats


@dataclass
class ConfidenceInterval:
    """Stores a metric with its confidence interval."""
    mean: float
    ci_lower: float
    ci_upper: float
    ci_width: float
    n: int
    method: str
    count: Optional[int] = None
    std: Optional[float] = None


def wilson_score_interval(successes: int, total: int, confidence: float = 0.95) -> ConfidenceInterval:
    """Wilson score CI for binomial proportions."""
    ##############################
    # Handle edge case of zero trials
    ##############################
    if total == 0:
        return ConfidenceInterval(0.0, 0.0, 0.0, 0.0, 0, "wilson_score", count=0)

    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p = successes / total

    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denominator

    return ConfidenceInterval(
        mean=float(p),
        ci_lower=float(max(0, center - margin)),
        ci_upper=float(min(1, center + margin)),
        ci_width=float(min(1, center + margin) - max(0, center - margin)),
        n=total,
        method="wilson_score",
        count=successes
    )


def t_distribution_ci(data: List[float], confidence: float = 0.95) -> ConfidenceInterval:
    """t-distribution CI for continuous data."""
    arr = np.array(data)
    n = len(arr)

    ##############################
    # Handle case of very few data points (less than 2 samples)
    ##############################
    if n < 2:
        mean = float(np.mean(arr)) if n > 0 else 0.0
        return ConfidenceInterval(mean, mean, mean, 0.0, n, "t_distribution", std=0.0)

    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1))
    sem = stats.sem(arr)
    h = sem * stats.t.ppf((1 + confidence) / 2, n - 1)

    return ConfidenceInterval(
        mean=mean,
        ci_lower=float(mean - h),
        ci_upper=float(mean + h),
        ci_width=float(2 * h),
        n=n,
        method="t_distribution",
        std=std
    )

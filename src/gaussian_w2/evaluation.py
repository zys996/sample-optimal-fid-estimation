"""Statistics shared by experiment analysis and the paper reproduction scripts.

Trial SD and between-case SD use the sample convention (ddof=1). RMSE
averages squared errors over the observed trials (denominator T). A failed
estimate must be handled explicitly by the caller; it is never dropped here.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np


def _finite_vector(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.float64)
    if array.ndim != 1 or not len(array):
        raise ValueError("Expected a nonempty one-dimensional sequence.")
    if not np.isfinite(array).all():
        raise ValueError("Nonfinite observations require explicit failure handling.")
    return array


def trial_metrics(estimates: Iterable[float], target: float) -> dict:
    """Describe repeated estimates for one fixed case and one method.

    ``center_error`` is abs(mean(estimate) - target), not mean absolute error.
    The target can be known population truth or an explicitly chosen proxy.
    Sample SD is undefined for a single observation and is returned as NaN.
    """
    values = _finite_vector(estimates)
    if not np.isfinite(target):
        raise ValueError("The target must be finite.")
    errors = values - float(target)
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")
    return {
        "n_trials": int(len(values)),
        "mean_estimate": mean,
        "signed_error": mean - float(target),
        "center_error": abs(mean - float(target)),
        "sd": sd,
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "median_absolute_error": float(np.median(np.abs(errors))),
    }


def across_cases(values: Iterable[float]) -> dict:
    """Give each case equal weight when reporting its scalar performance metric."""
    array = _finite_vector(values)
    return {
        "n_cases": int(len(array)),
        "mean": float(np.mean(array)),
        "sd": float(np.std(array, ddof=1)) if len(array) > 1 else float("nan"),
    }


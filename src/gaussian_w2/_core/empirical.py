"""Sample moments for real (n, d) arrays with the requested normalization."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .linalg import symmetrize


FloatArray = NDArray[np.float64]


def _normalization_denominator(n: int, normalization: str) -> Tuple[float, str]:
    key = str(normalization).strip().lower().replace(" ", "")
    if key in {"n", "1/n", "mle", "biased"}:
        return float(n), "1/N"
    if key in {"n-1", "1/(n-1)", "unbiased"}:
        if n < 2:
            raise ValueError("1/(N-1) covariance normalization requires at least two samples")
        return float(n - 1), "1/(N-1)"
    raise ValueError("normalization must be '1/N' or '1/(N-1)'")


def empirical_covariance(
    samples: ArrayLike,
    *,
    centered: bool = False,
    normalization: str = "1/N",
) -> Tuple[FloatArray, FloatArray, str]:
    """Return empirical covariance, empirical/known mean, and normalization.

    With ``centered=False`` the mean is known to be zero and the uncentered
    second moment is used, as required by the main experiments.
    """

    array = np.asarray(samples, dtype=np.float64)
    if centered:
        mean = np.mean(array, axis=0, dtype=np.float64)
        residuals = array - mean
    else:
        mean = np.zeros(array.shape[1], dtype=np.float64)
        residuals = array
    denominator, label = _normalization_denominator(array.shape[0], normalization)
    covariance = symmetrize((residuals.T @ residuals) / denominator)
    if not np.isfinite(covariance).all() or not np.isfinite(mean).all():
        raise FloatingPointError("empirical moments produced non-finite output")
    return covariance, mean, label


__all__ = ["empirical_covariance"]

"""Stable Gaussian W2/FID ground-truth calculations."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .linalg import DEFAULT_EIG_ATOL, DEFAULT_EIG_RTOL, psd_eigh, psd_sqrt, symmetrize


FloatArray = NDArray[np.float64]


def _covariance_pair(a: ArrayLike, b: ArrayLike) -> tuple[FloatArray, FloatArray]:
    a_array = symmetrize(a)
    b_array = symmetrize(b)
    if a_array.shape != b_array.shape:
        raise ValueError(
            f"covariances must have the same shape; got {a_array.shape}, {b_array.shape}"
        )
    return a_array, b_array


def _mean(mean: Optional[ArrayLike], dimension: int, name: str) -> FloatArray:
    if mean is None:
        return np.zeros(dimension, dtype=np.float64)
    array = np.asarray(mean, dtype=np.float64)
    if array.ndim == 0 and dimension == 1:
        array = array.reshape(1)
    if array.shape != (dimension,):
        raise ValueError(f"{name} must have shape ({dimension},); got {array.shape}")
    return array


def covariance_fidelity(
    a: ArrayLike,
    b: ArrayLike,
    *,
    rtol: float = DEFAULT_EIG_RTOL,
    atol: float = DEFAULT_EIG_ATOL,
) -> float:
    """Return ``tr(sqrt(sqrt(A) B sqrt(A)))`` for PSD covariances."""

    a_array, b_array = _covariance_pair(a, b)
    # Validate B as PSD even in degenerate cases where multiplication by A could
    # otherwise hide a negative direction.
    psd_eigh(b_array, rtol=rtol, atol=atol)
    sqrt_a = psd_sqrt(a_array, rtol=rtol, atol=atol)
    middle = symmetrize(sqrt_a @ b_array @ sqrt_a)
    middle_values, _, _ = psd_eigh(middle, rtol=rtol, atol=atol)
    result = float(np.sum(np.sqrt(middle_values), dtype=np.float64))
    if not np.isfinite(result):
        raise FloatingPointError("covariance fidelity produced non-finite output")
    return result


def gaussian_w2_sq_truth(
    mu_a: Optional[ArrayLike],
    a: ArrayLike,
    mu_b: Optional[ArrayLike],
    b: ArrayLike,
    *,
    rtol: float = DEFAULT_EIG_RTOL,
    atol: float = DEFAULT_EIG_ATOL,
) -> Dict[str, object]:
    """Return the Gaussian squared-distance target; ``None`` means zero mean."""

    a_array, b_array = _covariance_pair(a, b)
    dimension = int(a_array.shape[0])
    mean_a = _mean(mu_a, dimension, "mu_a")
    mean_b = _mean(mu_b, dimension, "mu_b")
    psd_eigh(a_array, rtol=rtol, atol=atol)
    psd_eigh(b_array, rtol=rtol, atol=atol)

    if np.array_equal(a_array, b_array):
        # The null pair is an important experimental regime.  Preserve its
        # exact mathematical truth instead of exposing trace/fidelity
        # cancellation at the 1e-13 scale and then amplifying it via sqrt.
        fidelity = float(np.trace(a_array))
        covariance_unclipped = 0.0
        covariance_w2_sq = 0.0
    else:
        fidelity = covariance_fidelity(a_array, b_array, rtol=rtol, atol=atol)
        covariance_unclipped = float(
            np.trace(a_array) + np.trace(b_array) - 2.0 * fidelity
        )
        if not np.isfinite(covariance_unclipped):
            raise FloatingPointError("Gaussian covariance distance is non-finite")
        covariance_w2_sq = max(covariance_unclipped, 0.0)
    mean_w2_sq = float(np.dot(mean_a - mean_b, mean_a - mean_b))
    w2_sq = max(mean_w2_sq + covariance_w2_sq, 0.0)
    if not np.isfinite(w2_sq):
        raise FloatingPointError("Gaussian distance produced non-finite output")
    return {"w2_sq_true": w2_sq}


__all__ = ["covariance_fidelity", "gaussian_w2_sq_truth"]

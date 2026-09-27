"""Float64 CuPy matrix helpers and Gaussian FD truth, imported lazily."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

from .gpu_backend import require_cupy

DEFAULT_EIG_RTOL = 1.0e-10
DEFAULT_EIG_ATOL = 1.0e-12


@dataclass(frozen=True)
class _SupportFactors:
    full_sqrt: Any
    pinv_sqrt: Any
    support_vectors: Any
    support_values: Any
    eigenvalues: Any
    threshold: float
    rank: int


def _symmetrize(matrix: Any) -> Any:
    return (matrix + matrix.T) * 0.5


def _psd_eigh(
    matrix: Any,
    *,
    rtol: float,
    atol: float,
    cp: Any,
) -> tuple[Any, Any, float]:
    symmetric = _symmetrize(matrix)
    values, vectors = cp.linalg.eigh(symmetric)
    if not bool(cp.all(cp.isfinite(values))):
        raise FloatingPointError("covariance eigendecomposition produced non-finite eigenvalues")
    max_abs = float(cp.max(cp.abs(values))) if int(values.size) else 0.0
    threshold = float(max(atol, rtol * max(1.0, max_abs)))
    minimum = float(values[0])
    if minimum < -threshold:
        raise ValueError(
            "matrix is not positive semidefinite within tolerance: "
            f"minimum eigenvalue {minimum:.6e}, threshold {threshold:.6e}"
        )
    return cp.maximum(values, 0.0), vectors, threshold


def _support_factors(
    matrix: Any,
    *,
    rtol: float,
    atol: float,
    cp: Any,
) -> _SupportFactors:
    values, vectors, threshold = _psd_eigh(matrix, rtol=rtol, atol=atol, cp=cp)
    full_sqrt = (vectors * cp.sqrt(values)[None, :]) @ vectors.T
    positive = values > threshold
    rank = int(cp.count_nonzero(positive))
    support_values = values[positive]
    dimension = int(matrix.shape[0])
    if rank:
        support_vectors = vectors[:, positive]
        pinv_sqrt = (
            support_vectors * (1.0 / cp.sqrt(support_values))[None, :]
        ) @ support_vectors.T
    else:
        support_vectors = cp.empty((dimension, 0), dtype=cp.float64)
        pinv_sqrt = cp.zeros((dimension, dimension), dtype=cp.float64)
    return _SupportFactors(
        full_sqrt=_symmetrize(full_sqrt),
        pinv_sqrt=_symmetrize(pinv_sqrt),
        support_vectors=support_vectors,
        support_values=support_values,
        eigenvalues=values,
        threshold=threshold,
        rank=rank,
    )


def _covariance_fidelity(
    a: Any,
    b: Any,
    *,
    rtol: float,
    atol: float,
    cp: Any,
    factors_a: Optional[_SupportFactors] = None,
    factors_b: Optional[_SupportFactors] = None,
) -> float:
    # Validate B independently, including directions hidden by singular A.
    if factors_b is None:
        _psd_eigh(b, rtol=rtol, atol=atol, cp=cp)
    factors = factors_a or _support_factors(a, rtol=rtol, atol=atol, cp=cp)
    # CPU ``psd_sqrt`` retains small positive eigenvalues even when they are
    # below the numerical-rank threshold; do the same for fidelity.
    middle = _symmetrize(factors.full_sqrt @ b @ factors.full_sqrt)
    middle_values, _, _ = _psd_eigh(middle, rtol=rtol, atol=atol, cp=cp)
    return float(cp.sum(cp.sqrt(middle_values), dtype=cp.float64))


def _normalization_denominator(n: int, normalization: str) -> tuple[float, str]:
    if normalization == "1/N":
        return float(n), normalization
    if normalization == "1/(N-1)":
        if n < 2:
            raise ValueError("1/(N-1) covariance normalization requires at least two samples")
        return float(n - 1), normalization
    raise ValueError("normalization must be '1/N' or '1/(N-1)'")


def gpu_gaussian_w2_sq_truth(
    mu_a: Any,
    A: Any,
    mu_b: Any,
    B: Any,
    *,
    rtol: float = DEFAULT_EIG_RTOL,
    atol: float = DEFAULT_EIG_ATOL,
    device: Optional[int] = None,
) -> dict[str, Any]:
    """Compute Gaussian W2 ground truth entirely on the selected GPU."""

    cp = require_cupy()
    selected = int(cp.cuda.Device().id if device is None else device)
    with cp.cuda.Device(selected):
        if any(cp.iscomplexobj(x) for x in (A, B, mu_a, mu_b)):
            raise ValueError("Gaussian means and covariances must be real")
        a, b = cp.asarray(A, dtype=cp.float64), cp.asarray(B, dtype=cp.float64)
        if a.shape != b.shape:
            raise ValueError(f"A and B must have the same shape; got {a.shape}, {b.shape}")
        dimension = int(a.shape[0])
        mean_a = cp.zeros(dimension, dtype=cp.float64) if mu_a is None else cp.asarray(mu_a, dtype=cp.float64)
        mean_b = cp.zeros(dimension, dtype=cp.float64) if mu_b is None else cp.asarray(mu_b, dtype=cp.float64)
        if mean_a.shape != (dimension,) or mean_b.shape != (dimension,):
            raise ValueError(f"means must have shape ({dimension},)")
        factors_a = _support_factors(a, rtol=rtol, atol=atol, cp=cp)
        factors_b = _support_factors(b, rtol=rtol, atol=atol, cp=cp)
        if bool(cp.array_equal(a, b)):
            fidelity = float(cp.trace(a))
            covariance_unclipped = 0.0
            covariance_w2_sq = 0.0
        else:
            fidelity = _covariance_fidelity(
                a,
                b,
                rtol=rtol,
                atol=atol,
                cp=cp,
                factors_a=factors_a,
                factors_b=factors_b,
            )
            covariance_unclipped = (
                float(cp.trace(a) + cp.trace(b)) - 2.0 * fidelity
            )
            if not math.isfinite(covariance_unclipped):
                raise FloatingPointError("Gaussian covariance distance is non-finite")
            covariance_w2_sq = max(covariance_unclipped, 0.0)
        delta = mean_a - mean_b
        mean_w2_sq = float(delta @ delta)
        w2_sq = max(mean_w2_sq + covariance_w2_sq, 0.0)
        if not math.isfinite(w2_sq):
            raise FloatingPointError("Gaussian distance is non-finite")
        return {"w2_sq_true": float(w2_sq)}


def _balanced_spans(n_samples: int, n_groups: int) -> tuple[tuple[int, int], ...]:
    if n_groups < 1 or n_groups > n_samples:
        raise ValueError("n_groups must satisfy 1 <= n_groups <= n_samples")
    quotient, remainder = divmod(n_samples, n_groups)
    spans: list[tuple[int, int]] = []
    start = 0
    for index in range(n_groups):
        stop = start + quotient + (1 if index < remainder else 0)
        spans.append((start, stop))
        start = stop
    return tuple(spans)

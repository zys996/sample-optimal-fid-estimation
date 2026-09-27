"""Degree selection and support handling for one-sided RTD."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Union

import numpy as np

from .linalg import psd_eigh

Array = np.ndarray


class _InsufficientSamplesError(ValueError):
    """Raised internally when no degree satisfying ``2 <= L <= n // 2`` exists."""


class _PilotRankError(RuntimeError):
    """Raised when fresh observations reveal support missing from the pilot."""


@dataclass(frozen=True)
class _SupportFactors:
    pinv_sqrt: Array
    projector: Array
    support_vectors: Array
    support_values: Array
    rank: int


def _support_factors(matrix: Array, *, rtol: float, atol: float) -> _SupportFactors:
    values, vectors, threshold = psd_eigh(matrix, rtol=rtol, atol=atol)
    values = np.asarray(values, dtype=float)
    vectors = np.asarray(vectors, dtype=float)
    positive = values > threshold
    rank = int(np.count_nonzero(positive))
    support_values = values[positive]
    support_vectors = vectors[:, positive]
    if rank:
        pinv_sqrt = (support_vectors * (1.0 / np.sqrt(support_values))) @ support_vectors.T
        projector = support_vectors @ support_vectors.T
    else:
        dimension = matrix.shape[0]
        pinv_sqrt = np.zeros((dimension, dimension), dtype=float)
        projector = np.zeros_like(pinv_sqrt)
    return _SupportFactors(
        pinv_sqrt=pinv_sqrt,
        projector=projector,
        support_vectors=support_vectors,
        support_values=support_values,
        rank=rank,
    )


def _support_overlap_svd(a, b, *, rtol: float, atol: float, xp=np):
    """RTD overlap on the same retained eigenspaces as its whitening.

    If A = Q_A diag(a) Q_A.T and B = Q_B diag(b) on retained support,
    the nonzero overlap singular values come from
    diag(sqrt(a)) Q_A.T Q_B diag(sqrt(b)). Lift the right singular vectors
    back to feature coordinates for the correction. General PSD square roots
    keep all positive eigenvalues; only RTD uses this numerical support.
    """
    if not a.rank or not b.rank:
        return (xp.empty(0, dtype=xp.float64),
                xp.empty((0, b.support_vectors.shape[0]), dtype=xp.float64))
    overlap = (xp.sqrt(a.support_values)[:, None]
               * (a.support_vectors.T @ b.support_vectors)
               * xp.sqrt(b.support_values)[None, :])
    u, singular_values, vt = xp.linalg.svd(overlap, full_matrices=False)
    if not all(bool(xp.all(xp.isfinite(array))) for array in (u, singular_values, vt)):
        raise FloatingPointError("overlap SVD produced non-finite output")
    threshold = max(atol, rtol * max(1.0, float(singular_values[0])))
    keep = singular_values > threshold
    return singular_values[keep], vt[keep, :] @ b.support_vectors.T


def recombine_atomic_blocks(n_atoms: int, degree: int) -> tuple[Array, ...]:
    """Balance ``n_atoms`` atom IDs into ``degree`` disjoint collections."""

    n_atoms = int(n_atoms)
    degree = int(degree)
    if n_atoms < 1 or not 1 <= degree <= n_atoms:
        raise ValueError("degree must satisfy 1 <= degree <= n_atoms")
    return tuple(np.asarray(group, dtype=int) for group in np.array_split(np.arange(n_atoms), degree))


def _dimension_cap_value(
    degree_cap: Optional[Union[int, Mapping[Any, Any]]], dimension: int
) -> Optional[int]:
    if degree_cap is None:
        return None
    if isinstance(degree_cap, Mapping):
        value = degree_cap.get(dimension, degree_cap.get(str(dimension)))
        return None if value is None else int(value)
    return int(degree_cap)


def choose_taylor_degree(
    d: int,
    n: int,
    *,
    rho_design: float = 0.25,
    C_T: float = 1.0,
    adaptive_degree: bool = True,
    degree_cap: Optional[Union[int, Mapping[Any, Any]]] = None,
    requested_degree: Optional[int] = None,
) -> tuple[int, int, str]:
    """Choose the paper Taylor depth and apply the sample/optional runtime caps.

    A returned ``used`` value of zero means that ``n < 4`` and hence no degree
    can satisfy ``2 <= L <= floor(n/2)``.  This non-raising form lets an
    experiment runner record the skipped trial explicitly.
    """

    d, n = int(d), int(n)
    if d < 1 or n < 1:
        raise ValueError("d and n must be positive")
    if requested_degree is not None:
        requested = int(requested_degree)
    elif adaptive_degree:
        if not 0.0 < float(rho_design) < 1.0:
            raise ValueError("rho_design must lie in (0, 1)")
        if float(C_T) <= 0.0:
            raise ValueError("C_T must be positive")
        epsilon_proxy = min(1.0, math.sqrt(d / n))
        argument = float(C_T) * d / epsilon_proxy
        requested = max(2, int(math.ceil(math.log(argument) / math.log(1.0 / rho_design))))
    else:
        requested = 2
    if requested < 2:
        raise ValueError("requested Taylor degree must be at least 2")

    if n // 2 < 2:
        return requested, 0, "insufficient_correction_samples"

    used = requested
    reasons: list[str] = []
    cap = _dimension_cap_value(degree_cap, d)
    if cap is not None:
        if cap < 2:
            raise ValueError("degree cap must be at least 2")
        if used > cap:
            used = cap
            reasons.append("runtime_dimension_cap")
    feasibility_cap = n // 2
    if used > feasibility_cap:
        used = feasibility_cap
        reasons.append("sample_feasibility_cap")
    return requested, used, ";".join(reasons) if reasons else "none"


def _failure_result(
    dimension: int,
    *,
    status: str,
    error: Union[BaseException, str],
    metadata: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Keep failed trials explicit rather than returning a usable estimate."""
    return {
        "estimate_w2_sq_raw": math.nan,
        "status": status,
        "error_message": str(error),
        "dimension": int(dimension),
        **dict(metadata or {}),
    }

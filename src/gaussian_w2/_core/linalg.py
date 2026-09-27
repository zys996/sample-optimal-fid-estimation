"""Numerically consistent linear-algebra helpers for symmetric PSD matrices.

All PSD matrix functions in the project go through :func:`psd_eigh`.  This is
important for two reasons: matrices are symmetrized before diagonalization, and
the same absolute/relative tolerance determines every reported numerical rank.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray


DEFAULT_EIG_RTOL = 1.0e-10
DEFAULT_EIG_ATOL = 1.0e-12

FloatArray = NDArray[np.float64]


def _real_square_matrix(matrix: ArrayLike, *, name: str = "matrix") -> FloatArray:
    """Convert real square input without scanning its entries."""

    array = np.asarray(matrix)
    if array.ndim != 2 or array.shape[0] != array.shape[1] or array.shape[0] == 0:
        raise ValueError(f"{name} must be a nonempty square matrix; got {array.shape}")
    if np.iscomplexobj(array):
        raise TypeError("covariance matrices must be real")
    return np.asarray(array, dtype=np.float64)


def symmetrize(matrix: ArrayLike) -> FloatArray:
    """Return ``(matrix + matrix.T) / 2`` after validating its shape."""

    array = _real_square_matrix(matrix)
    return 0.5 * (array + array.T)


def psd_eigh(
    matrix: ArrayLike,
    rtol: float = DEFAULT_EIG_RTOL,
    atol: float = DEFAULT_EIG_ATOL,
) -> Tuple[FloatArray, FloatArray, float]:
    """Diagonalize a symmetric PSD matrix with the project's rank threshold.

    Eigenvalues in ``[-threshold, 0)`` are interpreted as roundoff and clipped
    to zero.  A more negative eigenvalue is evidence that the input is not PSD
    and raises :class:`ValueError`.  Small *positive* eigenvalues are returned
    unchanged; callers determine numerical support with ``values > threshold``.
    """

    symmetric = symmetrize(matrix)
    values, vectors = np.linalg.eigh(symmetric)
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(vectors)):
        raise FloatingPointError("eigendecomposition produced non-finite output")
    max_abs = float(np.max(np.abs(values))) if values.size else 0.0
    threshold = float(max(atol, rtol * max(1.0, max_abs)))
    min_value = float(values[0])
    if min_value < -threshold:
        raise ValueError(
            "matrix is not positive semidefinite within tolerance: "
            f"minimum eigenvalue {min_value:.6e}, threshold {threshold:.6e}"
        )
    values = np.maximum(values, 0.0)
    return values, vectors, threshold


def _spectral_matrix_function(vectors: FloatArray, values: FloatArray) -> FloatArray:
    result = (vectors * values[np.newaxis, :]) @ vectors.T
    result = 0.5 * (result + result.T)
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("matrix square root produced non-finite output")
    return result


def psd_sqrt(
    matrix: ArrayLike,
    rtol: float = DEFAULT_EIG_RTOL,
    atol: float = DEFAULT_EIG_ATOL,
) -> FloatArray:
    """Return the principal symmetric PSD square root."""

    values, vectors, _ = psd_eigh(matrix, rtol=rtol, atol=atol)
    return _spectral_matrix_function(vectors, np.sqrt(values))


__all__ = [
    "DEFAULT_EIG_ATOL",
    "DEFAULT_EIG_RTOL",
    "psd_eigh",
    "psd_sqrt",
    "symmetrize",
]

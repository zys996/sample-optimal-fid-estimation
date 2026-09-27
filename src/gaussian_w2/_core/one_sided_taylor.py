"""Direct one-sided Taylor contractions for the Gaussian fidelity term.

For a positive diagonal expansion centre ``C = diag(s)`` and a relative
covariance direction ``V``, RTD's square-root recurrence expands

``X(z) = (C (I + z V) C) ** (1 / 2) = C + sum_r X_r z**r``.

The coefficients are obtained from an ``r x r`` Sylvester recursion. Mixed
contractions use exact subset polarization and compensated scalar summation.
CuPy is imported lazily, so importing this module is safe on CPU-only hosts.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Sequence

import numpy as np


Array = np.ndarray


def _real_array(value: Any, *, imag_tol: float) -> Array:
    """Convert a real array, allowing only negligible imaginary roundoff."""
    array = np.asarray(value)
    if np.iscomplexobj(array):
        if np.max(np.abs(array.imag)) > imag_tol * (1.0 + np.max(np.abs(array.real))):
            raise FloatingPointError("Taylor input has non-negligible imaginary entries")
        array = array.real
    return np.asarray(array, dtype=np.float64)


def _validate_degree(degree: int) -> int:
    degree = int(degree)
    if degree < 2:
        raise ValueError("degree must be at least 2")
    return degree


def _neumaier_sum(
    values: Sequence[np.floating[Any]], dtype: np.dtype
) -> np.floating[Any]:
    total = dtype.type(0.0)
    compensation = dtype.type(0.0)
    for raw in values:
        value = dtype.type(raw)
        updated = total + value
        if abs(total) >= abs(value):
            compensation += (total - updated) + value
        else:
            compensation += (value - updated) + total
        total = updated
    return total + compensation


def _cpu_scalar_sum(
    values: Sequence[np.floating[Any]],
    *,
    dtype: np.dtype,
    compensated: bool,
) -> np.floating[Any]:
    if compensated:
        return _neumaier_sum(values, dtype)
    return dtype.type(sum(values, dtype.type(0.0)))


def _cpu_homogeneous(
    singular_values: Array,
    direction: Array,
    degree: int,
    *,
    dtype: np.dtype,
) -> np.floating[Any]:
    """Return ``Q_degree(direction)`` using the direct Sylvester recursion."""

    s = np.asarray(singular_values, dtype=dtype)
    v = np.asarray(direction, dtype=dtype)
    denominator = s[:, None] + s[None, :]
    minimum_denominator = float(np.min(denominator))
    if minimum_denominator <= 0.0 or not np.all(np.isfinite(denominator)):
        raise FloatingPointError(
            "Sylvester denominator is not strictly positive and finite"
        )

    # C V C for C = diag(s).  X_0=C is not stored because all higher-order
    # right-hand sides only involve X_1, ..., X_{j-1}.
    rhs_first = s[:, None] * v * s[None, :]
    coefficients: list[Array] = [np.zeros((s.size, s.size), dtype=dtype)]
    for order in range(1, degree + 1):
        if order == 1:
            rhs = rhs_first.copy()
        else:
            rhs = np.zeros_like(rhs_first)
            for ell in range(1, order):
                with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                    product = coefficients[ell] @ coefficients[order - ell]
                rhs -= product
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            coefficient = rhs / denominator
        if not np.all(np.isfinite(coefficient)):
            raise FloatingPointError(
                f"non-finite coefficient at Sylvester order {order}"
            )
        coefficients.append(coefficient)

    # The Gaussian covariance fidelity contains -2 Tr X(z).
    return dtype.type(-2.0) * np.trace(coefficients[degree])


def evaluate_one_sided_homogeneous_taylor_coefficient(
    singular_values: Any,
    direction: Any,
    degree: int,
    *,
    imag_tol: float = 1.0e-10,
) -> float:
    """Evaluate ``Q_k(V)`` with the direct one-sided ``r x r`` recurrence.

    ``singular_values`` contains only the positive overlap singular values.
    Rank-deficient ambient covariance matrices are therefore supported by
    passing their positive overlap support and expressing ``direction`` in
    that support basis.
    """

    degree = _validate_degree(degree)
    s = _real_array(singular_values, imag_tol=imag_tol).reshape(-1)
    if np.any(s <= 0.0):
        raise ValueError("singular_values must contain only positive values")
    v = _real_array(direction, imag_tol=imag_tol)
    if v.shape != (s.size, s.size):
        raise ValueError("direction shape must match the overlap support")
    dtype = np.dtype(np.float64)
    value = _cpu_homogeneous(
        s.astype(dtype), v.astype(dtype), degree, dtype=dtype
    )
    result = float(value)
    if not math.isfinite(result):
        raise FloatingPointError("homogeneous Taylor coefficient is non-finite")
    return result


def evaluate_one_sided_mixed_taylor_contraction(
    singular_values: Any,
    directions: Sequence[Any],
    *,
    compensated_sum: bool = True,
    imag_tol: float = 1.0e-10,
) -> float:
    """Evaluate the exact mixed one-sided contraction by polarization.

    The ``k`` supplied directions define
    ``T_k[V_1, ..., V_k]``.  Binary-reflected Gray-code traversal changes one
    direction at a time and every subset receives a fresh Sylvester recursion.
    """

    degree = _validate_degree(len(directions))
    s = _real_array(singular_values, imag_tol=imag_tol).reshape(-1)
    if np.any(s <= 0.0):
        raise ValueError("singular_values must contain only positive values")
    matrices = [_real_array(value, imag_tol=imag_tol) for value in directions]
    if any(value.shape != (s.size, s.size) for value in matrices):
        raise ValueError("direction shapes must match the overlap support")
    dtype = np.dtype(np.float64)
    s_work = s.astype(dtype)
    matrices_work = [value.astype(dtype) for value in matrices]

    subset_direction = np.zeros((s.size, s.size), dtype=dtype)
    terms: list[np.floating[Any]] = []
    previous_gray = 0
    for counter in range(1 << degree):
        gray = counter ^ (counter >> 1)
        if counter:
            changed = previous_gray ^ gray
            bit = changed.bit_length() - 1
            if gray & changed:
                subset_direction += matrices_work[bit]
            else:
                subset_direction -= matrices_work[bit]
        coefficient = _cpu_homogeneous(
            s_work, subset_direction, degree, dtype=dtype
        )
        if not np.isfinite(coefficient):
            raise FloatingPointError("non-finite homogeneous polarization term")
        subset_size = bin(gray).count("1")
        sign = -1.0 if (degree - subset_size) % 2 else 1.0
        terms.append(dtype.type(sign) * coefficient)
        previous_gray = gray

    alternating_sum = _cpu_scalar_sum(
        terms, dtype=dtype, compensated=bool(compensated_sum)
    )
    result = float(alternating_sum / dtype.type(math.factorial(degree)))
    if not math.isfinite(result):
        raise FloatingPointError("polarization produced a non-finite result")
    return result


def _resolve_cupy(cp: Optional[Any]) -> Any:
    if cp is not None:
        return cp
    from .gpu_backend import require_cupy

    return require_cupy()


def _gpu_float(value: Any) -> float:
    return float(value.item() if hasattr(value, "item") else value)


def _gpu_bool(value: Any) -> bool:
    return bool(value.item() if hasattr(value, "item") else value)


def _gpu_real_array(value: Any, *, imag_tol: float, cp: Any) -> Any:
    array = cp.asarray(value)
    if array.dtype.kind == "c":
        if _gpu_float(cp.max(cp.abs(array.imag))) > imag_tol * (1.0 + _gpu_float(cp.max(cp.abs(array.real)))):
            raise FloatingPointError("Taylor input has non-negligible imaginary entries")
        array = array.real
    return cp.asarray(array, dtype=cp.float64)


def _gpu_homogeneous(
    singular_values: Any,
    direction: Any,
    degree: int,
    *,
    cp: Any,
) -> Any:
    denominator = singular_values[:, None] + singular_values[None, :]
    rhs_first = (
        singular_values[:, None] * direction * singular_values[None, :]
    )
    coefficients = [cp.zeros_like(rhs_first)]
    for order in range(1, degree + 1):
        if order == 1:
            rhs = rhs_first.copy()
        else:
            rhs = cp.zeros_like(rhs_first)
            for ell in range(1, order):
                rhs -= coefficients[ell] @ coefficients[order - ell]
        coefficient = rhs / denominator
        if not _gpu_bool(cp.all(cp.isfinite(coefficient))):
            raise FloatingPointError(
                f"non-finite coefficient at Sylvester order {order}"
            )
        coefficients.append(coefficient)
    return -2.0 * cp.trace(coefficients[degree])


def evaluate_one_sided_homogeneous_taylor_coefficient_gpu(
    singular_values: Any,
    direction: Any,
    degree: int,
    *,
    cp: Optional[Any] = None,
    imag_tol: float = 1.0e-10,
) -> float:
    """CuPy version of the direct one-sided homogeneous coefficient."""

    degree = _validate_degree(degree)
    cp = _resolve_cupy(cp)
    s = _gpu_real_array(singular_values, imag_tol=imag_tol, cp=cp).reshape(-1)
    if _gpu_bool(cp.any(s <= 0.0)):
        raise ValueError("singular_values must contain only positive values")
    v = _gpu_real_array(direction, imag_tol=imag_tol, cp=cp)
    if v.shape != (s.size, s.size):
        raise ValueError("direction shape must match the overlap support")
    value = _gpu_homogeneous(s, v, degree, cp=cp)
    result = _gpu_float(value)
    if not math.isfinite(result):
        raise FloatingPointError("homogeneous Taylor coefficient is non-finite")
    return result


def evaluate_one_sided_mixed_taylor_contraction_gpu(
    singular_values: Any,
    directions: Sequence[Any],
    *,
    compensated_sum: bool = True,
    cp: Optional[Any] = None,
    imag_tol: float = 1.0e-10,
) -> float:
    """CuPy direct recurrence with exact host-compensated polarization."""

    degree = _validate_degree(len(directions))
    cp = _resolve_cupy(cp)
    s = _gpu_real_array(singular_values, imag_tol=imag_tol, cp=cp).reshape(-1)
    if _gpu_bool(cp.any(s <= 0.0)):
        raise ValueError("singular_values must contain only positive values")
    matrices = [_gpu_real_array(value, imag_tol=imag_tol, cp=cp) for value in directions]
    if any(value.shape != (s.size, s.size) for value in matrices):
        raise ValueError("direction shapes must match the overlap support")

    subset_direction = cp.zeros((int(s.size), int(s.size)), dtype=cp.float64)
    terms: list[Any] = []
    previous_gray = 0
    for counter in range(1 << degree):
        gray = counter ^ (counter >> 1)
        if counter:
            changed = previous_gray ^ gray
            bit = changed.bit_length() - 1
            if gray & changed:
                subset_direction += matrices[bit]
            else:
                subset_direction -= matrices[bit]
        coefficient = _gpu_homogeneous(s, subset_direction, degree, cp=cp)
        subset_size = bin(gray).count("1")
        sign = -1.0 if (degree - subset_size) % 2 else 1.0
        terms.append(sign * coefficient)
        previous_gray = gray

    host_terms = cp.asnumpy(cp.stack(terms)).astype(np.float64, copy=False)
    if not np.all(np.isfinite(host_terms)):
        raise FloatingPointError("non-finite homogeneous polarization term")
    alternating_sum = (
        float(_neumaier_sum(host_terms, np.dtype(np.float64)))
        if compensated_sum
        else float(np.sum(host_terms, dtype=np.float64))
    )
    result = float(alternating_sum / math.factorial(degree))
    if not math.isfinite(result):
        raise FloatingPointError("polarization produced a non-finite result")
    return result


__all__ = [
    "evaluate_one_sided_homogeneous_taylor_coefficient",
    "evaluate_one_sided_homogeneous_taylor_coefficient_gpu",
    "evaluate_one_sided_mixed_taylor_contraction",
    "evaluate_one_sided_mixed_taylor_contraction_gpu",
]

"""OLS and VA extrapolation: sample-size grid → weights → raw intercept.

``fit_inverse_sample_polynomial`` returns only the intercept and its weights.
For repeated observations on the same grid, design the weights once and reuse
``apply_extrapolation_weights``.
"""
from .._core.fid_infinity import (
    CHEBYSHEV_INVERSE_N,
    ORDINARY_OLS,
    UNIFORM_INVERSE_N,
    UNIFORM_N,
    VARIANCE_AWARE,
    apply_extrapolation_weights,
    extrapolation_sample_sizes,
    fid_infinity_sample_sizes,
    fit_inverse_sample_polynomial,
    inverse_sample_extrapolation_weights,
)

__all__ = [
    "UNIFORM_N", "UNIFORM_INVERSE_N", "CHEBYSHEV_INVERSE_N",
    "ORDINARY_OLS", "VARIANCE_AWARE", "extrapolation_sample_sizes",
    "fid_infinity_sample_sizes", "fit_inverse_sample_polynomial",
    "inverse_sample_extrapolation_weights", "apply_extrapolation_weights",
]

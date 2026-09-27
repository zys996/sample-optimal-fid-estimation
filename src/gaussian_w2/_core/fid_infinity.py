"""OLS and variance-aware extrapolation from finite-sample plug-in FD values.

The experiment runners construct the plug-in curve. This module supplies its
sample-size grid, designs weights, and evaluates the raw extrapolated value.
Negative estimates remain raw.
Callers supply positive integer node counts and orders in the documented shapes.
"""
from __future__ import annotations

import math
from typing import Any, Sequence, Tuple

import numpy as np

ORDINARY_OLS = "ordinary_ols"
VARIANCE_AWARE = "variance_aware"
EXTRAPOLATION_WEIGHTINGS = frozenset({ORDINARY_OLS, VARIANCE_AWARE})
UNIFORM_N = "uniform_n"
UNIFORM_INVERSE_N = "uniform_inverse_n"
CHEBYSHEV_INVERSE_N = "chebyshev_inverse_n"
EXTRAPOLATION_SAMPLE_SCHEDULES = frozenset(
    {UNIFORM_N, UNIFORM_INVERSE_N, CHEBYSHEV_INVERSE_N}
)


def fid_infinity_sample_sizes(
    maximum_samples: int,
    *,
    minimum_samples: int = 5_000,
    num_points: int = 15,
) -> Tuple[int, ...]:
    """Return the paper-style regular grid of finite sample counts.

    The public FID-infinity code uses ``linspace(...).astype(int32)``. This
    function applies the same integer truncation rule and requires distinct
    sample sizes.
    """

    return extrapolation_sample_sizes(
        maximum_samples, minimum_samples=minimum_samples,
        num_points=num_points, schedule=UNIFORM_N,
    )


def extrapolation_sample_sizes(
    maximum_samples: int,
    *,
    minimum_samples: int = 5_000,
    num_points: int = 15,
    schedule: str = UNIFORM_N,
) -> Tuple[int, ...]:
    """Return an integer regression grid for a named extrapolation schedule.

    ``uniform_n`` is the original FID-infinity grid. ``uniform_inverse_n``
    spaces points uniformly in the regression coordinate ``1/n`` and
    ``chebyshev_inverse_n`` uses Chebyshev--Lobatto nodes in that coordinate.
    All schedules include both requested endpoints exactly.
    """

    maximum, minimum, points = maximum_samples, minimum_samples, num_points
    resolved = str(schedule).strip().lower()
    if resolved not in EXTRAPOLATION_SAMPLE_SCHEDULES:
        choices = ", ".join(sorted(EXTRAPOLATION_SAMPLE_SCHEDULES))
        raise ValueError(f"schedule must be one of: {choices}")
    if points < 2:
        raise ValueError("num_points must be at least two")
    if maximum <= minimum:
        raise ValueError("maximum_samples must be greater than minimum_samples")

    if resolved == UNIFORM_N:
        # Exact positive truncation used by the original FID-infinity grid.
        span = maximum - minimum
        grid = np.asarray([minimum + j * span // (points - 1) for j in range(points)], dtype=np.int64)
    else:
        position = np.linspace(0.0, 1.0, points, dtype=np.float64)
        if resolved == CHEBYSHEV_INVERSE_N:
            position = 0.5 * (1.0 - np.cos(np.pi * position))
        inverse = (1.0 - position) / float(minimum) + position / float(maximum)
        grid = np.rint(1.0 / inverse).astype(np.int64)
        grid[0], grid[-1] = minimum, maximum
    if np.any(np.diff(grid) <= 0):
        raise ValueError(
            "integer sample-size grid contains duplicates; reduce num_points "
            "or increase the sample-size interval"
        )
    return tuple(int(value) for value in grid)


def inverse_sample_extrapolation_weights(
    sample_sizes: Sequence[int],
    *,
    order: int,
    weighting: str = ORDINARY_OLS,
) -> dict[str, Any]:
    """Return weights that retain constants and cancel powers 1/n through 1/n**p.

    OLS minimizes ``sum(w_i**2)``. VA minimizes ``sum(w_i**2 / c_i)``, where
    ``c_i = n_i / max(n)``. Both are solved as constrained minimum-norm systems.
    A Chebyshev basis on an affine transform of ``max(n)/n`` avoids the severe
    column scaling of the raw power basis; it leaves the intercept unchanged.
    """
    resolved_order = order
    if weighting not in EXTRAPOLATION_WEIGHTINGS:
        choices = ", ".join(sorted(EXTRAPOLATION_WEIGHTINGS))
        raise ValueError(f"weighting must be one of: {choices}")
    sizes = np.asarray(sample_sizes, dtype=np.float64)
    if sizes.ndim != 1:
        raise ValueError("sample_sizes must be a vector")
    if len(np.unique(sizes)) != sizes.size:
        raise ValueError("sample_sizes must be unique")
    if sizes.size < resolved_order + 1:
        raise ValueError(
            f"at least {resolved_order + 1} extrapolation points are required "
            f"for order {resolved_order}"
        )

    maximum = float(np.max(sizes))
    inverse = maximum / sizes
    center = 0.5 * (float(np.min(inverse)) + float(np.max(inverse)))
    half_range = 0.5 * (float(np.max(inverse)) - float(np.min(inverse)))
    if not math.isfinite(half_range) or half_range <= 0.0:
        raise ValueError("inverse-sample extrapolation design is rank deficient")
    coordinates = (inverse - center) / half_range
    design = np.polynomial.chebyshev.chebvander(coordinates, resolved_order)
    singular_values = np.linalg.svd(design, compute_uv=False)
    tolerance = (
        np.finfo(np.float64).eps
        * max(design.shape)
        * float(np.max(singular_values))
    )
    if int(np.count_nonzero(singular_values > tolerance)) != resolved_order + 1:
        raise ValueError("inverse-sample extrapolation design is rank deficient")
    intercept_basis = np.polynomial.chebyshev.chebvander(
        np.asarray([-center / half_range], dtype=np.float64), resolved_order
    )[0]

    # Set w = scale * u, so the chosen objective becomes ||u||_2**2.
    relative_sizes = sizes / maximum
    scale = (
        np.ones_like(relative_sizes)
        if weighting == ORDINARY_OLS
        else np.sqrt(relative_sizes)
    )
    constraint_operator = design.T * scale[None, :]
    transformed_weights, _, rank, _ = np.linalg.lstsq(
        constraint_operator, intercept_basis, rcond=None
    )
    if int(rank) != resolved_order + 1:
        raise ValueError("inverse-sample extrapolation design is rank deficient")
    weights = scale * transformed_weights
    if not np.all(np.isfinite(weights)):
        raise ValueError("extrapolation weights must be finite")
    return {"weights": weights.tolist()}


def apply_extrapolation_weights(
    plugin_estimates: Sequence[float], weights: Sequence[float]
) -> float:
    """Apply precomputed extrapolation weights to one plug-in curve.

    ``math.fsum`` reduces cancellation error for high-order, alternating
    extrapolation weights.
    """

    values = np.asarray(plugin_estimates, dtype=np.float64)
    resolved_weights = np.asarray(weights, dtype=np.float64)
    if values.ndim != 1 or resolved_weights.shape != values.shape:
        raise ValueError("plugin_estimates and weights must be equal-length vectors")
    estimate = float(
        math.fsum(
            float(weight) * float(value)
            for weight, value in zip(resolved_weights, values)
        )
    )
    if not math.isfinite(estimate):
        raise FloatingPointError("extrapolation produced a non-finite output")
    return estimate


def fit_inverse_sample_polynomial(
    sample_sizes: Sequence[int],
    plugin_estimates: Sequence[float],
    *,
    order: int,
    weighting: str = ORDINARY_OLS,
) -> dict[str, Any]:
    """Return the unclipped intercept and weights of a polynomial fit in 1/n.

    Only the intercept is needed for FD estimation: compute its weights once,
    then apply them with compensated summation. No fitted curve is constructed.
    """
    weights = inverse_sample_extrapolation_weights(
        sample_sizes, order=order, weighting=weighting
    )["weights"]
    return {
        "intercept": apply_extrapolation_weights(plugin_estimates, weights),
        "extrapolation_weights": weights,
    }


def fit_fid_infinity(
    sample_sizes: Sequence[int], plugin_estimates: Sequence[float]
) -> dict[str, Any]:
    """Return the order-one OLS intercept used by the FID-infinity baseline."""
    return fit_inverse_sample_polynomial(sample_sizes, plugin_estimates, order=1)

r"""Analytic covariance pairs for one-sided fixed-reference stress tests.

The constructions in this module are deliberately analytic.  They avoid a
numerical search for a requested population distance and do not add a ridge
to singular covariances.  ``A`` is the exact, fixed covariance and ``B`` is
the covariance from which the experiment samples.

All builders accept a NumPy-like array namespace through ``xp``.  In
particular, callers may pass :mod:`cupy` without this CPU-safe module importing
CuPy.  Construction costs only the two dense output arrays; the structured
rank-deficient and ill-conditioned families do not form dense rotation or
factor matrices on the way to their answers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np


SCALED_IDENTITY = "scaled_identity"
RANK_DEFICIENT = "rank_deficient"
ILL_CONDITIONED = "ill_conditioned"

SUPPORTED_RANK_FRACTIONS = (0.25, 0.50, 0.75)
MAXIMUM_CONDITION_NUMBER = 1.0e6


@dataclass(frozen=True)
class FixedReferencePair:
    """One deterministic population covariance pair and its exact metadata."""

    A: Any
    B: Any
    metadata: Mapping[str, Any]

    @property
    def dimension(self) -> int:
        return int(self.A.shape[0])


def _dimension(value: int) -> int:
    if value <= 0 or value % 2:
        raise ValueError("dimension must be a positive even integer")
    return int(value)


def _positive_float(value: float, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")
    return result


def _base_metadata(
    *,
    family: str,
    construction: str,
    dimension: int,
    target_w2_sq: float,
) -> dict[str, Any]:
    return {
        "family": family,
        "construction": construction,
        "dimension": dimension,
        "target_w2_sq": target_w2_sq,
        "true_w2_sq": target_w2_sq,
        "truth_source": "analytic",
        "fixed_covariance": "A",
        "sampled_covariance": "B",
    }


def scaled_identity_pair(
    dimension: int,
    *,
    target_w2_sq: float = 5.0,
    generated_scale: float = 1.0,
    xp: Any = np,
) -> FixedReferencePair:
    r"""Return ``A=a I`` and ``B=b I`` with exact covariance distance ``D``.

    Here ``b = generated_scale`` and ``a = (sqrt(b) + sqrt(D / d))**2``.
    Thus ``W2(A, B)**2 = d * (sqrt(a) - sqrt(b))**2 = D``.
    The default ``b=1`` preserves the historical isotropic construction.
    """

    dimension = _dimension(dimension)
    target = _positive_float(target_w2_sq, name="target_w2_sq")
    scale_b = _positive_float(generated_scale, name="generated_scale")
    sqrt_scale_a = math.sqrt(scale_b) + math.sqrt(target / dimension)
    scale_a = sqrt_scale_a * sqrt_scale_a
    identity = xp.eye(dimension, dtype=xp.float64)
    A = scale_a * identity
    B = identity.copy() if scale_b == 1.0 else scale_b * identity
    metadata = _base_metadata(
        family=SCALED_IDENTITY,
        construction=SCALED_IDENTITY,
        dimension=dimension,
        target_w2_sq=target,
    )
    metadata.update(
        {
            "rank_A": dimension,
            "rank_B": dimension,
            "rank_fraction_A": 1.0,
            "rank_fraction_B": 1.0,
            "condition_number_A": 1.0,
            "condition_number_B": 1.0,
            "sqrt_scale_A": sqrt_scale_a,
            "scale_A": scale_a,
            "scale_B": scale_b,
            "trace_A": dimension * scale_a,
            "trace_B": dimension * scale_b,
        }
    )
    return FixedReferencePair(A=A, B=B, metadata=metadata)


def _rank_fraction(value: float, dimension: int) -> tuple[float, int]:
    fraction = float(value)
    if not any(fraction == supported for supported in SUPPORTED_RANK_FRACTIONS):
        raise ValueError(
            f"rank_fraction must be one of {SUPPORTED_RANK_FRACTIONS}"
        )
    rank_value = fraction * dimension
    rank = int(rank_value)
    if rank_value != rank:
        raise ValueError(
            "dimension * rank_fraction must be an integer; quarter-rank "
            "families require dimension divisible by four"
        )
    return fraction, rank


def _validate_angle_randomization(randomize_angles: bool, population_seed: Optional[int]) -> None:
    if not isinstance(randomize_angles, bool):
        raise ValueError("randomize_angles must be boolean")
    if not randomize_angles:
        if population_seed is not None:
            raise ValueError("population_seed requires randomize_angles=True")
    elif (isinstance(population_seed, (bool, np.bool_))
          or not isinstance(population_seed, (int, np.integer))
          or population_seed < 0):
        raise ValueError("randomized angles require a nonnegative integer population_seed")


def _randomized_distance_profile(
    count: int, target: float, cap: float, population_seed: int,
) -> np.ndarray:
    """Draw bounded, nonconstant per-plane distances with prescribed sum.

    Draw iid U(-1,1) with NumPy PCG64, center the draws, and divide by
    max(1, max(abs(centered))). Add half the available distance-to-boundary
    times this zero-sum vector to target/count. The scale floor preserves
    random amplitudes even with only two planes. This is a specified sampling
    distribution over profiles, not uniform sampling of all covariance pairs.
    """
    if count < 2:
        raise ValueError("randomized angles require at least two rotation planes")
    if not 0.0 < target < count * cap:
        raise ValueError("randomized angles require a strictly interior feasible target_w2_sq")
    rng = np.random.Generator(np.random.PCG64(int(population_seed)))
    centered = rng.uniform(-1.0, 1.0, size=count)
    centered -= centered.mean()
    largest = float(np.max(np.abs(centered)))
    if largest == 0.0:
        raise ValueError("population_seed generated a constant angle profile")
    centered /= max(1.0, largest)
    mean = target / count
    amplitude = 0.5 * min(mean, cap - mean)
    profile = mean + amplitude * centered
    # Correct only floating-point summation error, leaving a half-margin to
    # both boundaries. The analytic target is preserved without a search.
    profile[-1] += target - math.fsum(profile.tolist())
    if not np.all((profile > 0.0) & (profile < cap)):
        raise ValueError("randomized distance profile is outside the feasible range")
    if float(np.ptp(profile)) == 0.0:
        raise ValueError("target_w2_sq is too close to a boundary to randomize in float64")
    return profile


def _profile_metadata(profile: np.ndarray, angles: np.ndarray, seed: int) -> dict[str, Any]:
    values = profile.tolist()
    return {
        "randomize_angles": True,
        "population_seed": int(seed),
        "profile_seed_generator": "numpy.random.PCG64",
        "profile_distribution": "centered_uniform_bounded_distance_v1",
        "profile_spread_fraction": 0.5,
        "distance_contribution_profile": values,
        "angle_profile_radians": angles.tolist(),
        "analytic_w2_sq": math.fsum(values),
    }


def rank_deficient_pair(
    dimension: int,
    *,
    rank_fraction: float = 0.50,
    target_w2_sq: float = 5.0,
    randomize_angles: bool = False,
    population_seed: Optional[int] = None,
    xp: Any = np,
) -> FixedReferencePair:
    r"""Return two exact-rank projectors at a prescribed Bures distance.

    The positive spectrum of each covariance consists only of ones.  Their
    supports have the smallest intersection forced by dimension and differ by
    a common principal angle in every remaining two-coordinate plane.  If
    ``s = min(rank, d-rank)``, the resulting truth is

    ``W2(A, B)**2 = 2*s*(1-cos(theta))``. With ``randomize_angles=True``,
    the specified population seed draws distinct per-plane distance
    contributions using ``_randomized_distance_profile``; their sum remains D.
    """

    dimension = _dimension(dimension)
    fraction, rank = _rank_fraction(rank_fraction, dimension)
    target = _positive_float(target_w2_sq, name="target_w2_sq")

    rotated_directions = min(rank, dimension - rank)
    maximum_target = 2.0 * rotated_directions
    if target > maximum_target:
        raise ValueError(
            "target_w2_sq is infeasible for the requested rank: require "
            f"target_w2_sq <= {maximum_target:g}"
        )
    _validate_angle_randomization(randomize_angles, population_seed)
    if randomize_angles:
        profile = _randomized_distance_profile(rotated_directions, target, 2.0, population_seed)
        cosines = 1.0 - profile / 2.0
        sines = np.sqrt((profile / 2.0) * (2.0 - profile / 2.0))
        angles = np.arctan2(sines, cosines)
        cosine, sine = xp.asarray(cosines, dtype=xp.float64), xp.asarray(sines, dtype=xp.float64)
        angle = None
    else:
        cosine = 1.0 - target / maximum_target
        # Preserve the historical common-angle construction exactly.
        cosine = min(1.0, max(0.0, cosine))
        sine = math.sqrt(max(0.0, 1.0 - cosine * cosine))
        angle = math.acos(cosine)
    intersection = rank - rotated_directions

    A = xp.zeros((dimension, dimension), dtype=xp.float64)
    B = xp.zeros((dimension, dimension), dtype=xp.float64)
    diagonal = xp.arange(rank)
    A[diagonal, diagonal] = 1.0
    if intersection:
        common = xp.arange(intersection)
        B[common, common] = 1.0
    offsets = xp.arange(rotated_directions)
    support_indices = intersection + offsets
    null_indices = rank + offsets
    B[support_indices, support_indices] = cosine * cosine
    B[null_indices, null_indices] = sine * sine
    cross = cosine * sine
    B[support_indices, null_indices] = cross
    B[null_indices, support_indices] = cross

    metadata = _base_metadata(
        family=RANK_DEFICIENT,
        construction=RANK_DEFICIENT,
        dimension=dimension,
        target_w2_sq=target,
    )
    metadata.update(
        {
            "rank_A": rank,
            "rank_B": rank,
            "rank_fraction_A": fraction,
            "rank_fraction_B": fraction,
            "positive_eigenvalue_A": 1.0,
            "positive_eigenvalue_B": 1.0,
            "condition_number_A": 1.0,
            "condition_number_B": 1.0,
            "support_intersection_dimension": intersection,
            "rotated_directions": rotated_directions,
            "principal_angle_radians": angle,
            "principal_angle_cosine": None if randomize_angles else cosine,
            "maximum_target_w2_sq": maximum_target,
            "trace_A": float(rank),
            "trace_B": float(rank),
        }
    )
    if randomize_angles:
        metadata.pop("principal_angle_radians")
        metadata.pop("principal_angle_cosine")
        metadata.update(_profile_metadata(profile, angles, population_seed))
    return FixedReferencePair(A=A, B=B, metadata=metadata)


def ill_conditioned_pair(
    dimension: int,
    *,
    condition_number: float = 1.0e4,
    target_w2_sq: float = 5.0,
    randomize_angles: bool = False,
    population_seed: Optional[int] = None,
    xp: Any = np,
) -> FixedReferencePair:
    r"""Return a full-rank, noncommuting pair with exact condition ``kappa``.

    Each two-dimensional block of ``A`` has spectrum ``(1, 1/kappa)``.
    ``B`` has the same spectrum after a common rotation between its high- and
    low-variance directions.  The angle is obtained analytically from

    ``D = d * ((1+l) - sqrt((1+l)^2 - (1-l)^2 sin(theta)^2))``,

    where ``l = 1/kappa``.  Requiring strict inequality at the maximum keeps
    the pair genuinely noncommuting rather than merely swapping eigenvectors.
    With ``randomize_angles=True``, independently seeded, bounded per-block
    distance contributions replace the common angle while preserving their
    sum and both spectra. At least two blocks are required.
    """

    dimension = _dimension(dimension)
    kappa = _positive_float(condition_number, name="condition_number")
    if kappa <= 1.0 or kappa > MAXIMUM_CONDITION_NUMBER:
        raise ValueError(
            "condition_number must satisfy 1 < condition_number <= 1e6"
        )
    target = _positive_float(target_w2_sq, name="target_w2_sq")

    low = 1.0 / kappa
    maximum_target = dimension * (1.0 - math.sqrt(low)) ** 2
    if target >= maximum_target:
        raise ValueError(
            "target_w2_sq is infeasible for a noncommuting paired rotation: "
            f"require target_w2_sq < {maximum_target:.17g}"
        )
    _validate_angle_randomization(randomize_angles, population_seed)
    if randomize_angles:
        count = dimension // 2
        profile = _randomized_distance_profile(count, target, 2.0 * (1.0 - math.sqrt(low)) ** 2, population_seed)
        delta = profile / 2.0
        sine_squares = delta * (2.0 * (1.0 + low) - delta) / ((1.0 - low) ** 2)
        if not np.all((sine_squares > 0.0) & (sine_squares < 1.0)):
            raise ValueError("analytic rotation is outside the noncommuting range")
        sines, cosines = np.sqrt(sine_squares), np.sqrt(1.0 - sine_squares)
        angles = np.arctan2(sines, cosines)
        sine, cosine = xp.asarray(sines, dtype=xp.float64), xp.asarray(cosines, dtype=xp.float64)
        angle, sine_squared = None, None
    else:
        delta = target / dimension
        sine_squared = (
            delta * (2.0 * (1.0 + low) - delta) / ((1.0 - low) ** 2)
        )
        if not 0.0 < sine_squared < 1.0:  # defensive after feasibility checks
            raise ValueError("analytic rotation is outside the noncommuting range")
        sine = math.sqrt(sine_squared)
        cosine = math.sqrt(1.0 - sine_squared)
        angle = math.asin(sine)

    A = xp.zeros((dimension, dimension), dtype=xp.float64)
    B = xp.zeros((dimension, dimension), dtype=xp.float64)
    high_indices = 2 * xp.arange(dimension // 2)
    low_indices = high_indices + 1
    A[high_indices, high_indices] = 1.0
    A[low_indices, low_indices] = low
    B[high_indices, high_indices] = cosine * cosine + low * sine * sine
    B[low_indices, low_indices] = sine * sine + low * cosine * cosine
    cross = (1.0 - low) * cosine * sine
    B[high_indices, low_indices] = cross
    B[low_indices, high_indices] = cross

    trace = 0.5 * dimension * (1.0 + low)
    metadata = _base_metadata(
        family=ILL_CONDITIONED,
        construction=ILL_CONDITIONED,
        dimension=dimension,
        target_w2_sq=target,
    )
    metadata.update(
        {
            "rank_A": dimension,
            "rank_B": dimension,
            "rank_fraction_A": 1.0,
            "rank_fraction_B": 1.0,
            "condition_number_A": kappa,
            "condition_number_B": kappa,
            "lambda_max_A": 1.0,
            "lambda_max_B": 1.0,
            "lambda_min_A": low,
            "lambda_min_B": low,
            "high_multiplicity": dimension // 2,
            "low_multiplicity": dimension // 2,
            "paired_rotation_angle_radians": angle,
            "paired_rotation_sine_squared": sine_squared,
            "maximum_target_w2_sq": maximum_target,
            "trace_A": trace,
            "trace_B": trace,
        }
    )
    if randomize_angles:
        metadata.pop("paired_rotation_angle_radians")
        metadata.pop("paired_rotation_sine_squared")
        metadata.update(_profile_metadata(profile, angles, population_seed))
    return FixedReferencePair(A=A, B=B, metadata=metadata)


def build_fixed_reference_pair(
    family: str,
    dimension: int,
    *,
    target_w2_sq: float = 5.0,
    rank_fraction: Optional[float] = None,
    condition_number: Optional[float] = None,
    generated_scale: Optional[float] = None,
    randomize_angles: bool = False,
    population_seed: Optional[int] = None,
    xp: Any = np,
) -> FixedReferencePair:
    """Build an isotropic, rank-deficient, or ill-conditioned paper population.

    Parameters are explicit in the experiment configuration; historical fixed
    aliases are unnecessary. Reject parameters belonging to another family.
    """
    _validate_angle_randomization(randomize_angles, population_seed)
    common = dict(target_w2_sq=target_w2_sq, xp=xp)
    if family == "isotropic":
        if rank_fraction is not None or condition_number is not None or randomize_angles:
            raise ValueError("isotropic accepts generated_scale, but no rank or angle parameters")
        pair = scaled_identity_pair(
            dimension, generated_scale=1.0 if generated_scale is None else generated_scale,
            **common,
        )
    elif family == RANK_DEFICIENT:
        if condition_number is not None or generated_scale is not None:
            raise ValueError("rank_deficient does not accept condition_number or generated_scale")
        pair = rank_deficient_pair(
            dimension, rank_fraction=0.5 if rank_fraction is None else rank_fraction,
            randomize_angles=randomize_angles, population_seed=population_seed, **common,
        )
    elif family == ILL_CONDITIONED:
        if rank_fraction is not None or generated_scale is not None:
            raise ValueError("ill_conditioned does not accept rank_fraction or generated_scale")
        pair = ill_conditioned_pair(
            dimension, condition_number=1e4 if condition_number is None else condition_number,
            randomize_angles=randomize_angles, population_seed=population_seed, **common,
        )
    else:
        raise ValueError(f"unsupported fixed-reference family {family!r}")
    return FixedReferencePair(A=pair.A, B=pair.B, metadata={**pair.metadata, "family": family})

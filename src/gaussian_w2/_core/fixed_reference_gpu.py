"""Float64 CUDA empirical FD, extrapolation, and one-sided RTD.

The reference Gaussian is fixed. Known-zero-mean Gaussian experiments use raw
rows and 1/N covariance normalization; ImageNet uses N-1 Helmert contrasts
and a separate sample-mean term. RTD streams bounded chunks into its pilot
scatter and correction atoms, without storing another N-by-d matrix.
CuPy remains optional and is loaded only when an estimator is called.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np

from .fid_infinity import (
    ORDINARY_OLS,
    UNIFORM_N,
    extrapolation_sample_sizes,
    fit_inverse_sample_polynomial,
)
from .rtd_utils import (
    _InsufficientSamplesError, _PilotRankError, _support_overlap_svd, choose_taylor_degree,
)
from .gpu_backend import require_cupy
from .gpu_linalg import (
    DEFAULT_EIG_ATOL, DEFAULT_EIG_RTOL, _balanced_spans,
    _normalization_denominator, _psd_eigh, _support_factors, _symmetrize,
)
from .one_sided_taylor import (
    evaluate_one_sided_mixed_taylor_contraction_gpu,
)


@dataclass(frozen=True)
class GPUFixedReferenceStatistics:
    """Device-resident Gaussian statistics for a fixed real-data reference.

    Keeping only the mean and covariance makes repeated feature trials cheap:
    the full reference feature matrix can be released after this object is
    constructed.
    """

    mean: Any
    covariance: Any
    sqrt_covariance: Any
    factors: Any
    trace: float
    rank: int
    eig_threshold: float
    n_samples: Optional[int]
    dimension: int
    device_id: int
    covariance_normalization: str
    precompute_seconds: float
    reference_mode: str = "fixed_empirical_gaussian"


ReferenceInput = Union[GPUFixedReferenceStatistics, Any]


def _require_covariance_normalization(
    normalization: str, *, known_zero_mean: bool
) -> None:
    expected = "1/N" if known_zero_mean else "1/(N-1)"
    if normalization != expected:
        raise ValueError(f"this mean convention requires covariance normalization {expected}")


def _as_gpu_feature_matrix(
    value: Any,
    *,
    name: str,
    cp: Any,
    device: int,
) -> Any:
    with cp.cuda.Device(device):
        if cp.iscomplexobj(value):
            raise ValueError(f"{name} must be real")
        array = cp.asarray(value, dtype=cp.float64, order="C")
        if array.ndim == 1:
            array = array[:, None]
        if array.ndim != 2 or 0 in array.shape:
            raise ValueError(f"{name} must have nonempty shape (n, d)")
        return array


def prepare_gpu_fixed_reference(
    reference_features: Any,
    *,
    covariance_normalization: str = "1/(N-1)",
    known_zero_mean: bool = False,
    rtol: float = DEFAULT_EIG_RTOL,
    atol: float = DEFAULT_EIG_ATOL,
    device: Optional[int] = None,
    synchronize_timing: bool = True,
) -> GPUFixedReferenceStatistics:
    """Compute fixed real-data mean/covariance once on the selected GPU."""

    resolved_known_zero = bool(known_zero_mean)
    cp = require_cupy()
    selected = int(cp.cuda.Device().id if device is None else device)
    reference = _as_gpu_feature_matrix(
        reference_features,
        name="reference_features",
        cp=cp,
        device=selected,
    )
    n_samples, dimension = int(reference.shape[0]), int(reference.shape[1])
    denominator, normalization = _normalization_denominator(
        n_samples, covariance_normalization
    )
    with cp.cuda.Device(selected):
        if synchronize_timing:
            cp.cuda.Device(selected).synchronize()
        started = time.perf_counter()
        if resolved_known_zero:
            mean = cp.zeros(dimension, dtype=cp.float64)
            residual = reference
        else:
            mean = cp.mean(reference, axis=0, dtype=cp.float64)
            residual = reference - mean
        covariance = _symmetrize((residual.T @ residual) / denominator)
        factors = _support_factors(
            covariance,
            rtol=rtol,
            atol=atol,
            cp=cp,
        )
        trace = float(cp.trace(covariance))
        if synchronize_timing:
            cp.cuda.Device(selected).synchronize()
        elapsed = time.perf_counter() - started
        if not math.isfinite(trace):
            raise FloatingPointError("reference covariance trace is non-finite")
        return GPUFixedReferenceStatistics(
            mean=mean,
            covariance=covariance,
            sqrt_covariance=factors.full_sqrt,
            factors=factors,
            trace=trace,
            rank=factors.rank,
            eig_threshold=factors.threshold,
            n_samples=n_samples,
            dimension=dimension,
            device_id=selected,
            covariance_normalization=normalization,
            precompute_seconds=elapsed,
            reference_mode="fixed_empirical_gaussian",
        )


def prepare_gpu_fixed_reference_from_statistics(
    mean: Any,
    covariance: Any,
    *,
    sample_count: Optional[int] = 50_000,
    covariance_normalization: str = "1/(N-1)",
    reference_mode: str = "fixed_empirical_gaussian",
    rtol: float = DEFAULT_EIG_RTOL,
    atol: float = DEFAULT_EIG_ATOL,
    device: Optional[int] = None,
    synchronize_timing: bool = True,
) -> GPUFixedReferenceStatistics:
    """Load compact fixed-reference moments onto the selected GPU.

    This is the preferred server path after the reference pool has
    been reduced to a durable mean/covariance NPZ.  ``sample_count`` is
    provenance metadata; the supplied covariance is not rescaled.  An exact
    population reference may instead use ``sample_count=None`` together with
    ``reference_mode="exact_population_covariance"``.
    """

    cp = require_cupy()
    selected = int(cp.cuda.Device().id if device is None else device)
    # Supplied sample_count/reference_mode are provenance; moments are not rescaled.
    _, normalization = _normalization_denominator(2, covariance_normalization)
    with cp.cuda.Device(selected):
        if cp.iscomplexobj(covariance) or cp.iscomplexobj(mean):
            raise ValueError("reference means and covariances must be real")
        covariance_array = _symmetrize(cp.asarray(covariance, dtype=cp.float64))
        dimension = int(covariance_array.shape[0])
        mean_array = cp.asarray(mean, dtype=cp.float64)
        if mean_array.shape != (dimension,):
            raise ValueError(f"mean must have shape ({dimension},)")

        if synchronize_timing:
            cp.cuda.Device(selected).synchronize()
        started = time.perf_counter()
        factors = _support_factors(
            covariance_array,
            rtol=rtol,
            atol=atol,
            cp=cp,
        )
        trace = float(cp.trace(covariance_array))
        if synchronize_timing:
            cp.cuda.Device(selected).synchronize()
        elapsed = time.perf_counter() - started
        if not math.isfinite(trace):
            raise FloatingPointError("reference covariance trace is non-finite")
        return GPUFixedReferenceStatistics(
            mean=mean_array,
            covariance=covariance_array,
            sqrt_covariance=factors.full_sqrt,
            factors=factors,
            trace=trace,
            rank=factors.rank,
            eig_threshold=factors.threshold,
            n_samples=sample_count,
            dimension=dimension,
            device_id=selected,
            covariance_normalization=normalization,
            precompute_seconds=elapsed,
            reference_mode=reference_mode,
        )


def _resolve_reference(
    reference: ReferenceInput,
    *,
    covariance_normalization: str,
    known_zero_mean: bool,
    rtol: float,
    atol: float,
    device: Optional[int],
    synchronize_timing: bool,
) -> tuple[GPUFixedReferenceStatistics, bool]:
    if isinstance(reference, GPUFixedReferenceStatistics):
        return reference, True
    return (
        prepare_gpu_fixed_reference(
            reference,
            covariance_normalization=covariance_normalization,
            known_zero_mean=known_zero_mean,
            rtol=rtol,
            atol=atol,
            device=device,
            synchronize_timing=synchronize_timing,
        ),
        False,
    )


def _generated_plugin_value(
    reference: GPUFixedReferenceStatistics,
    generated: Any,
    *,
    known_zero_mean: bool,
    covariance_normalization: str,
    rtol: float,
    atol: float,
    cp: Any,
) -> float:
    n_samples = int(generated.shape[0])
    denominator, normalization = _normalization_denominator(
        n_samples, covariance_normalization
    )
    if known_zero_mean:
        mean_b = cp.zeros(reference.dimension, dtype=cp.float64)
        residual_b = generated
    else:
        mean_b = cp.mean(generated, axis=0, dtype=cp.float64)
        residual_b = generated - mean_b
    covariance_b = _symmetrize((residual_b.T @ residual_b) / denominator)
    _psd_eigh(covariance_b, rtol=rtol, atol=atol, cp=cp)
    middle = _symmetrize(
        reference.sqrt_covariance @ covariance_b @ reference.sqrt_covariance
    )
    middle_values, _ = cp.linalg.eigh(middle)
    if not bool(cp.all(cp.isfinite(middle_values))):
        raise FloatingPointError("fidelity eigendecomposition produced non-finite eigenvalues")
    maximum = float(cp.max(cp.abs(middle_values))) if int(middle_values.size) else 0.0
    threshold = float(max(atol, rtol * max(1.0, maximum)))
    minimum = float(middle_values[0]) if int(middle_values.size) else 0.0
    if minimum < -threshold:
        raise ValueError(
            "fixed-reference fidelity matrix is not positive semidefinite within "
            f"tolerance: minimum eigenvalue {minimum:.6e}"
        )
    fidelity = float(
        cp.sum(cp.sqrt(cp.maximum(middle_values, 0.0)), dtype=cp.float64)
    )
    covariance_term = (
        reference.trace + float(cp.trace(covariance_b)) - 2.0 * fidelity
    )
    if known_zero_mean:
        mean_term = 0.0
    else:
        mean_delta = reference.mean - mean_b
        mean_term = float(mean_delta @ mean_delta)
    raw = float(mean_term + covariance_term)
    if not all(math.isfinite(v) for v in (raw, mean_term, covariance_term)):
        raise FloatingPointError("fixed-reference empirical FD produced non-finite output")
    return raw


def gpu_fixed_reference_empirical(
    reference: ReferenceInput,
    generated_features: Any,
    *,
    covariance_normalization: str = "1/(N-1)",
    known_zero_mean: bool = False,
    rtol: float = DEFAULT_EIG_RTOL,
    atol: float = DEFAULT_EIG_ATOL,
    device: Optional[int] = None,
    synchronize_timing: bool = True,
) -> dict[str, Any]:
    """Standard plug-in FID with fixed real statistics and generated samples."""

    resolved_known_zero = bool(known_zero_mean)
    _require_covariance_normalization(
        covariance_normalization,
        known_zero_mean=resolved_known_zero,
    )
    cp = require_cupy()
    stats, reused = _resolve_reference(
        reference,
        covariance_normalization=covariance_normalization,
        known_zero_mean=resolved_known_zero,
        rtol=rtol,
        atol=atol,
        device=device,
        synchronize_timing=synchronize_timing,
    )
    _require_covariance_normalization(
        stats.covariance_normalization,
        known_zero_mean=resolved_known_zero,
    )
    generated = _as_gpu_feature_matrix(
        generated_features,
        name="generated_features",
        cp=cp,
        device=stats.device_id,
    )
    if int(generated.shape[1]) != stats.dimension:
        raise ValueError("reference and generated feature dimensions differ")
    with cp.cuda.Device(stats.device_id):
        if synchronize_timing:
            cp.cuda.Device(stats.device_id).synchronize()
        started = time.perf_counter()
        value = _generated_plugin_value(
            stats,
            generated,
            known_zero_mean=resolved_known_zero,
            covariance_normalization=covariance_normalization,
            rtol=rtol,
            atol=atol,
            cp=cp,
        )
        if synchronize_timing:
            cp.cuda.Device(stats.device_id).synchronize()
        elapsed = time.perf_counter() - started
    return {
        "estimate_w2_sq_raw": value,
        "status": "ok",
        "error_message": "",
        "dimension": stats.dimension,
        "known_zero_mean": resolved_known_zero,
        "covariance_normalization": stats.covariance_normalization,
        "n_samples_A": stats.n_samples,
        "n_samples_B": int(generated.shape[0]),
        "gpu_method_seconds": elapsed,
        "standalone_estimator_runtime_seconds": elapsed + (0.0 if reused else stats.precompute_seconds),
    }


def gpu_fixed_reference_plugin_curve(
    reference: ReferenceInput,
    generated_features: Any,
    *,
    minimum_samples: int = 5_000,
    num_points: int = 15,
    sample_schedule: str = UNIFORM_N,
    regression_orders: tuple[int, ...] = (1, 2, 3),
    regression_weightings: tuple[str, ...] = (ORDINARY_OLS,),
    seed: int = 0,
    covariance_normalization: str = "1/(N-1)",
    known_zero_mean: bool = False,
    rtol: float = DEFAULT_EIG_RTOL,
    atol: float = DEFAULT_EIG_ATOL,
    device: Optional[int] = None,
    synchronize_timing: bool = True,
) -> dict[str, Any]:
    """Evaluate one shared plug-in curve and several Section-4 fits.

    The endpoint uses all provided rows and is the ordinary empirical plug-in estimator.
    Every requested inverse-sample polynomial order is fitted to the *same*
    finite-sample values and therefore uses neither additional observations nor
    additional covariance eigendecompositions.  Raw intercepts are returned;
    callers decide whether a display-only nonnegative clipping is appropriate.
    """

    resolved_known_zero = bool(known_zero_mean)
    _require_covariance_normalization(
        covariance_normalization,
        known_zero_mean=resolved_known_zero,
    )
    cp = require_cupy()
    stats, reused = _resolve_reference(
        reference,
        covariance_normalization=covariance_normalization,
        known_zero_mean=resolved_known_zero,
        rtol=rtol,
        atol=atol,
        device=device,
        synchronize_timing=synchronize_timing,
    )
    generated = _as_gpu_feature_matrix(
        generated_features,
        name="generated_features",
        cp=cp,
        device=stats.device_id,
    )
    if int(generated.shape[1]) != stats.dimension:
        raise ValueError("reference and generated feature dimensions differ")
    maximum = int(generated.shape[0])
    sizes = extrapolation_sample_sizes(
        maximum,
        minimum_samples=minimum_samples,
        num_points=num_points,
        schedule=sample_schedule,
    )
    if max(regression_orders) >= len(sizes):
        raise ValueError("the plug-in curve needs more points than the largest order")
    rng = np.random.default_rng(seed)
    values: list[float] = []
    point_seconds: list[float] = []
    with cp.cuda.Device(stats.device_id):
        if synchronize_timing:
            cp.cuda.Device(stats.device_id).synchronize()
        curve_started = time.perf_counter()
        for sample_size in sizes:
            if sample_size == maximum:
                point_samples = generated
            else:
                host_indices = rng.permutation(maximum)[:sample_size]
                indices = cp.asarray(host_indices, dtype=cp.int64)
                point_samples = generated[indices]
            if synchronize_timing:
                cp.cuda.Device(stats.device_id).synchronize()
            point_started = time.perf_counter()
            value = _generated_plugin_value(
                stats,
                point_samples,
                known_zero_mean=resolved_known_zero,
                covariance_normalization=covariance_normalization,
                rtol=rtol,
                atol=atol,
                cp=cp,
            )
            if synchronize_timing:
                cp.cuda.Device(stats.device_id).synchronize()
            point_seconds.append(time.perf_counter() - point_started)
            values.append(value)
        curve_seconds = time.perf_counter() - curve_started

    fits_by_weighting: dict[str, dict[int, dict[str, Any]]] = {}
    for weighting in regression_weightings:
        fits_by_weighting[weighting] = {}
        for order in regression_orders:
            fits_by_weighting[weighting][order] = fit_inverse_sample_polynomial(
                sizes,
                values,
                order=order,
                weighting=weighting,
            )
    return {
        "sample_sizes": sizes,
        "plugin_values": tuple(values),
        "endpoint_plugin_w2_sq": values[-1],
        "fits_by_weighting": fits_by_weighting,
        "curve_seconds": curve_seconds,
        "point_seconds": tuple(point_seconds),
        "status": "ok",
        "error_message": "",
    }


def _helmert_from_validated(generated: Any, *, cp: Any) -> Any:
    n_samples = int(generated.shape[0])
    counts = cp.arange(1, n_samples, dtype=cp.float64)
    # This output buffer starts as prefix sums and is then updated in place,
    # avoiding a second N-by-d temporary.
    contrasts = cp.cumsum(generated[:-1], axis=0, dtype=cp.float64)
    contrasts /= counts[:, None]
    contrasts -= generated[1:]
    contrasts *= cp.sqrt(counts / (counts + 1.0))[:, None]
    return contrasts


def _stream_chunk_size(
    dimension: int,
    observation_count: int,
    requested: Optional[int],
) -> int:
    if requested is None:
        return max(1, min(int(dimension), int(observation_count)))
    if requested < 1:
        raise ValueError("stream_chunk_size must be positive to advance the stream")
    return min(requested, dimension, observation_count)


def _stream_covariance_scatter(
    generated: Any,
    *,
    observation_start: int,
    observation_stop: int,
    known_zero_mean: bool,
    chunk_size: int,
    cp: Any,
) -> Any:
    """Accumulate a covariance-observation scatter without an ``N x d`` copy."""

    dimension = int(generated.shape[1])
    scatter = cp.zeros((dimension, dimension), dtype=cp.float64)
    position = int(observation_start)
    stop = int(observation_stop)
    prefix = (
        None
        if known_zero_mean
        else cp.sum(generated[:position], axis=0, dtype=cp.float64)
    )
    while position < stop:
        chunk_stop = min(position + int(chunk_size), stop)
        if known_zero_mean:
            observations = generated[position:chunk_stop]
        else:
            prefixes = cp.cumsum(
                generated[position:chunk_stop], axis=0, dtype=cp.float64
            )
            prefixes += prefix
            counts = cp.arange(
                position + 1, chunk_stop + 1, dtype=cp.float64
            )
            observations = (
                prefixes / counts[:, None]
                - generated[position + 1 : chunk_stop + 1]
            )
            observations *= cp.sqrt(counts / (counts + 1.0))[:, None]
            prefix = prefixes[-1].copy()
        scatter += observations.T @ observations
        position = chunk_stop
    return _symmetrize(scatter)


def _stream_one_sided_atoms(
    generated: Any,
    *,
    observation_start: int,
    observation_count: int,
    known_zero_mean: bool,
    transform_b: Any,
    n_atoms: int,
    chunk_size: int,
    factors_b: Any,
    cp: Any,
) -> tuple[Any, Any, np.ndarray, tuple[tuple[int, int], ...], float]:
    """Stream covariance observations directly into atomic scatter sums."""

    spans = _balanced_spans(int(observation_count), n_atoms)
    rank = int(transform_b.shape[0])
    covariance = cp.zeros((n_atoms, rank, rank), dtype=cp.float64)
    norm_sq = cp.zeros(n_atoms, dtype=cp.float64)
    counts = np.zeros(n_atoms, dtype=np.int64)
    leakage_sq = cp.asarray(0.0, dtype=cp.float64)
    total_norm_sq = cp.asarray(0.0, dtype=cp.float64)
    position_base = int(observation_start)
    prefix = (
        None
        if known_zero_mean
        else cp.sum(generated[:position_base], axis=0, dtype=cp.float64)
    )
    full_support = factors_b.rank == int(generated.shape[1])
    for atom, (relative_start, relative_stop) in enumerate(spans):
        position = position_base + relative_start
        atom_stop = position_base + relative_stop
        counts[atom] = relative_stop - relative_start
        while position < atom_stop:
            chunk_stop = min(position + int(chunk_size), atom_stop)
            if known_zero_mean:
                observations = generated[position:chunk_stop]
            else:
                prefixes = cp.cumsum(
                    generated[position:chunk_stop], axis=0, dtype=cp.float64
                )
                prefixes += prefix
                helmert_counts = cp.arange(
                    position + 1, chunk_stop + 1, dtype=cp.float64
                )
                observations = (
                    prefixes / helmert_counts[:, None]
                    - generated[position + 1 : chunk_stop + 1]
                )
                observations *= cp.sqrt(
                    helmert_counts / (helmert_counts + 1.0)
                )[:, None]
                prefix = prefixes[-1].copy()
            transformed = observations @ transform_b.T
            if rank:
                covariance[atom] += transformed.T @ transformed
            chunk_norm_sq = cp.einsum("ij,ij->", observations, observations)
            norm_sq[atom] += chunk_norm_sq
            total_norm_sq += chunk_norm_sq
            if not full_support:
                supported = observations @ factors_b.support_vectors
                residual = observations - supported @ factors_b.support_vectors.T
                leakage_sq += cp.einsum("ij,ij->", residual, residual)
            position = chunk_stop
    leakage = math.sqrt(float(leakage_sq)) / max(
        1.0, math.sqrt(float(total_norm_sq))
    )
    return covariance, norm_sq, counts, spans, leakage


def _rtd_reference_factors(stats, *, rtol, atol, cp):
    """Reuse retained reference support only for the current RTD threshold."""
    maximum = float(stats.factors.eigenvalues[-1])
    threshold = max(atol, rtol * max(1.0, maximum))
    if threshold == stats.factors.threshold:
        return stats.factors
    return _support_factors(stats.covariance, rtol=rtol, atol=atol, cp=cp)


def gpu_rtd_fixed_reference(
    reference: ReferenceInput,
    generated_features: Any,
    *,
    m0: Optional[int] = None,
    L: Optional[int] = None,
    covariance_normalization: str = "1/(N-1)",
    known_zero_mean: bool = False,
    rho_design: float = 0.25,
    C_T: float = 1.0,
    degree_cap: Any = None,
    compensated_sum: bool = True,
    rtol: float = DEFAULT_EIG_RTOL,
    atol: float = DEFAULT_EIG_ATOL,
    device: Optional[int] = None,
    synchronize_timing: bool = True,
    stream_chunk_size: Optional[int] = None,
) -> dict[str, Any]:
    """One-sided RTD: sample pilot, then independent correction observations.

    For known-zero-mean Gaussian data the observations are raw rows (1/N).
    Otherwise, N rows yield N-1 Helmert contrasts (1/(N-1)) and the sample
    mean supplies the mean term. ``m0`` counts covariance observations.
    Both paths accumulate pilot and correction scatters in bounded chunks.
    """
    resolved_known_zero = bool(known_zero_mean)
    _require_covariance_normalization(
        covariance_normalization, known_zero_mean=resolved_known_zero,
    )
    cp = require_cupy()
    stats, reused = _resolve_reference(
        reference, covariance_normalization=covariance_normalization,
        known_zero_mean=resolved_known_zero, rtol=rtol,
        atol=atol, device=device, synchronize_timing=synchronize_timing,
    )
    _require_covariance_normalization(
        stats.covariance_normalization, known_zero_mean=resolved_known_zero,
    )
    generated = _as_gpu_feature_matrix(
        generated_features, name="generated_features", cp=cp, device=stats.device_id,
    )
    if int(generated.shape[1]) != stats.dimension:
        raise ValueError("reference and generated feature dimensions differ")
    n_raw = int(generated.shape[0])
    n_covariance = n_raw if resolved_known_zero else n_raw - 1
    resolved_m0 = n_covariance // 2 if m0 is None else m0
    chunk_size = _stream_chunk_size(stats.dimension, max(1, n_covariance), stream_chunk_size)
    metadata = {
        "dimension": stats.dimension,
        "m0": resolved_m0,
        "n_correction": n_covariance - resolved_m0,
        "covariance_observation_count": n_covariance,
        "stream_chunk_size": chunk_size,
        "known_zero_mean": resolved_known_zero,
        "covariance_normalization": stats.covariance_normalization,
        "n_samples_A": stats.n_samples,
        "n_samples_B": n_raw,
    }
    try:
        if n_raw < 2:
            raise _InsufficientSamplesError("at least two generated features are required")
        if not 1 <= resolved_m0 < n_covariance:
            raise _InsufficientSamplesError(
                "m0 must be positive and leave correction covariance observations"
            )
        n_correction = n_covariance - resolved_m0
        requested, used, reason = choose_taylor_degree(
            stats.dimension, n_correction, rho_design=float(rho_design),
            C_T=float(C_T), adaptive_degree=L is None, degree_cap=degree_cap,
            requested_degree=L,
        )
        metadata.update(L_requested=requested, L_used=used, degree_truncation_reason=reason)
        if used == 0:
            raise _InsufficientSamplesError(
                "no feasible degree: require 2 <= L_used <= floor(n_correction/2)"
            )
        with cp.cuda.Device(stats.device_id):
            if synchronize_timing:
                cp.cuda.Device(stats.device_id).synchronize()
            started = time.perf_counter()

            # 1. Estimate the Taylor center from the pilot alone.
            pilot_scatter = _stream_covariance_scatter(
                generated, observation_start=0, observation_stop=resolved_m0,
                known_zero_mean=resolved_known_zero, chunk_size=chunk_size, cp=cp,
            )
            covariance_b0 = _symmetrize(pilot_scatter / resolved_m0)
            factors_b = _support_factors(
                covariance_b0, rtol=rtol, atol=atol, cp=cp,
            )
            factors_a = _rtd_reference_factors(stats, rtol=rtol, atol=atol, cp=cp)
            singular_values, vt = _support_overlap_svd(
                factors_a, factors_b, rtol=rtol, atol=atol, xp=cp,
            )
            v = vt.T
            rank = int(singular_values.size)

            # 2. Stream fresh observations into L balanced atomic blocks.
            transform_b = v.T @ factors_b.pinv_sqrt
            atom_covariance, atom_norm_sq, atom_counts, _, leakage = _stream_one_sided_atoms(
                generated, observation_start=resolved_m0,
                observation_count=n_correction, known_zero_mean=resolved_known_zero,
                transform_b=transform_b, n_atoms=used, chunk_size=chunk_size,
                factors_b=factors_b, cp=cp,
            )
            tolerance = max(100.0 * atol, 100.0 * rtol, 1.0e-12)
            if not math.isfinite(leakage):
                raise FloatingPointError("pilot support leakage is non-finite")
            if leakage > tolerance:
                raise _PilotRankError(
                    "fresh generated covariance observations leave the pilot "
                    f"support (relative leakage B={leakage:.3e})"
                )
            identity = cp.eye(rank, dtype=cp.float64)

            # 3. Regroup the atoms for each Taylor degree and contract its term.
            nonlinear = 0.0
            if rank:
                for degree in range(2, used + 1):
                    directions = []
                    for start, stop in _balanced_spans(used, degree):
                        count = int(np.sum(atom_counts[start:stop]))
                        directions.append(cp.sum(atom_covariance[start:stop], axis=0) / count - identity)
                    nonlinear += evaluate_one_sided_mixed_taylor_contraction_gpu(
                        singular_values, directions, compensated_sum=compensated_sum, cp=cp,
                    )
            q_bar = cp.sum(atom_covariance, axis=0) / n_correction - identity
            mean_norm_sq = float(cp.sum(atom_norm_sq)) / n_correction
            linear = (mean_norm_sq - float(cp.trace(covariance_b0))
                      - float(cp.dot(singular_values, cp.diag(q_bar))))
            base = (stats.trace + float(cp.trace(covariance_b0))
                    - 2.0 * float(cp.sum(singular_values, dtype=cp.float64)))
            covariance_estimate = base + linear + nonlinear
            mean_estimate = 0.0
            if not resolved_known_zero:
                delta = cp.mean(generated, axis=0, dtype=cp.float64) - stats.mean
                mean_estimate = float(delta @ delta)
            raw = covariance_estimate + mean_estimate
            if synchronize_timing:
                cp.cuda.Device(stats.device_id).synchronize()
            elapsed = time.perf_counter() - started
        if not all(math.isfinite(v) for v in (raw, covariance_estimate, mean_estimate)):
            raise FloatingPointError("fixed-reference RTD produced non-finite output")
        return {
            **metadata, "estimate_w2_sq_raw": float(raw), "status": "ok", "error_message": "",
            "covariance_w2_sq_estimate": float(covariance_estimate),
            "mean_w2_sq_estimate": float(mean_estimate), "gpu_method_seconds": elapsed,
            "standalone_estimator_runtime_seconds": elapsed + (0.0 if reused else stats.precompute_seconds),
        }
    except _PilotRankError as error:
        status, message = "pilot_rank_failure", str(error)
    except _InsufficientSamplesError as error:
        status, message = "skipped_runtime_cap", str(error)
    except Exception as error:
        status, message = "numerical_failure", f"{type(error).__name__}: {error}"
    return {**metadata, "estimate_w2_sq_raw": math.nan, "status": status, "error_message": message}


__all__ = [
    "GPUFixedReferenceStatistics", "prepare_gpu_fixed_reference",
    "prepare_gpu_fixed_reference_from_statistics", "gpu_fixed_reference_empirical",
    "gpu_fixed_reference_plugin_curve",
    "gpu_rtd_fixed_reference",
]

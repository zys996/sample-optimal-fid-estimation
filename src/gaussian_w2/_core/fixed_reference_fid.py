"""Fixed-reference FID estimators for generated features.

The reference distribution in ordinary FID evaluation is not resampled at
every generated-sample budget: one first computes ``(mu_A, Sigma_A)`` from the
real feature set and then regards those statistics as fixed.  The estimands in
this module are therefore conditional on that empirical reference.

The general-order estimator needs independent zero-mean covariance
observations even though Inception features have an unknown mean.  For raw
generated features ``Y_1, ..., Y_N`` we use the deterministic Helmert
contrasts

``H_k = sqrt(k / (k + 1)) * (mean(Y_1, ..., Y_k) - Y_{k+1})``.

Under the working Gaussian feature model the ``N-1`` contrasts are iid
``N(0, Sigma_B)`` and are independent of the sample mean.  This makes it
legitimate to use disjoint contrast rows for pilot and correction.  The mean
term is the ordinary plug-in statistic ``||mean(Y) - mu_A||^2``.
Orthogonal contrasts are only *uncorrelated*, not generally independent, for
non-Gaussian features.

Inputs follow the experiment contract: real (n, d) sample arrays, d-vector
means, PSD covariance matrices, and integer sample counts and degrees.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .empirical import empirical_covariance
from .rtd_utils import (
    _InsufficientSamplesError,
    _PilotRankError,
    _failure_result,
    _support_factors,
    _support_overlap_svd,
    choose_taylor_degree,
    recombine_atomic_blocks,
)
from .linalg import (
    DEFAULT_EIG_ATOL,
    DEFAULT_EIG_RTOL,
    psd_eigh,
    psd_sqrt,
    symmetrize,
)
from .one_sided_taylor import evaluate_one_sided_mixed_taylor_contraction


FloatArray = NDArray[np.float64]
ReferenceInput = Union["FixedGaussianReference", ArrayLike]


@dataclass(frozen=True)
class FixedGaussianReference:
    """Cached empirical Gaussian used as the fixed side of every estimator.

    Construct instances with :func:`make_fixed_reference` or
    :func:`make_fixed_reference_from_statistics`.  Arrays returned by those
    factories are read-only so a trial cannot accidentally mutate its target.
    ``sample_count`` may be ``None`` when only externally computed statistics
    are available.
    """

    mean: FloatArray
    covariance: FloatArray
    sqrt_covariance: FloatArray
    sample_count: Optional[int]
    covariance_normalization: str

    @property
    def dimension(self) -> int:
        return int(self.mean.size)

    @property
    def covariance_trace(self) -> float:
        return float(np.trace(self.covariance))


def _canonical_normalization(value: str) -> str:
    key = str(value).strip().lower().replace(" ", "")
    if key in {"n", "1/n", "mle", "biased"}:
        return "1/N"
    if key in {"n-1", "1/(n-1)", "unbiased"}:
        return "1/(N-1)"
    raise ValueError("covariance_normalization must be '1/N' or '1/(N-1)'")


def _readonly_copy(array: ArrayLike) -> FloatArray:
    result = np.array(array, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def make_fixed_reference(
    reference_features: ArrayLike,
    *,
    covariance_normalization: str = "1/(N-1)",
    rtol: float = DEFAULT_EIG_RTOL,
    atol: float = DEFAULT_EIG_ATOL,
) -> FixedGaussianReference:
    """Freeze empirical real-feature mean/covariance for repeated FID trials.

    The canonical FID convention is selected by the default: center by the
    empirical mean and divide the covariance by ``N-1``.  Once constructed,
    the returned reference is treated as deterministic; uncertainty from the
    finite real set is not included in repeated generated-sample trial errors.
    """

    features = np.asarray(reference_features, dtype=np.float64)
    normalization = _canonical_normalization(covariance_normalization)
    covariance, mean, label = empirical_covariance(
        features,
        centered=True,
        normalization=normalization,
    )
    return make_fixed_reference_from_statistics(
        mean,
        covariance,
        sample_count=features.shape[0],
        covariance_normalization=label,
        rtol=rtol,
        atol=atol,
    )


def make_fixed_reference_from_statistics(
    mean: ArrayLike,
    covariance: ArrayLike,
    *,
    sample_count: Optional[int] = None,
    covariance_normalization: str = "1/(N-1)",
    rtol: float = DEFAULT_EIG_RTOL,
    atol: float = DEFAULT_EIG_ATOL,
) -> FixedGaussianReference:
    """Build a fixed reference from already-computed real FID statistics."""

    resolved_mean = np.asarray(mean, dtype=np.float64)
    resolved_covariance = np.asarray(covariance, dtype=np.float64)
    if resolved_mean.ndim != 1 or resolved_covariance.shape != (resolved_mean.size, resolved_mean.size):
        raise ValueError("mean and covariance dimensions must agree")
    resolved_covariance = symmetrize(resolved_covariance)
    normalization = _canonical_normalization(covariance_normalization)
    return FixedGaussianReference(
        mean=_readonly_copy(resolved_mean),
        covariance=_readonly_copy(resolved_covariance),
        sqrt_covariance=_readonly_copy(
            psd_sqrt(resolved_covariance, rtol=float(rtol), atol=float(atol))
        ),
        sample_count=sample_count,
        covariance_normalization=normalization,
    )


def _resolve_reference(
    reference: ReferenceInput,
    *,
    covariance_normalization: str,
    rtol: float,
    atol: float,
) -> FixedGaussianReference:
    if isinstance(reference, FixedGaussianReference):
        return reference
    return make_fixed_reference(
        reference,
        covariance_normalization=covariance_normalization,
        rtol=rtol,
        atol=atol,
    )


def _plugin_components(
    reference: FixedGaussianReference,
    generated_features: FloatArray,
    *,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    covariance_b, mean_b, normalization = empirical_covariance(
        generated_features,
        centered=True,
        normalization=reference.covariance_normalization,
    )
    covariance_b = symmetrize(covariance_b)
    middle = symmetrize(
        reference.sqrt_covariance @ covariance_b @ reference.sqrt_covariance
    )
    middle_values, _, _ = psd_eigh(middle, rtol=rtol, atol=atol)
    fidelity = float(np.sum(np.sqrt(middle_values), dtype=np.float64))
    covariance_term = float(
        reference.covariance_trace + np.trace(covariance_b) - 2.0 * fidelity
    )
    mean_delta = reference.mean - mean_b
    mean_term = float(mean_delta @ mean_delta)
    raw = mean_term + covariance_term
    if not math.isfinite(raw):
        raise FloatingPointError("empirical FD produced a non-finite output")
    return {
        "raw": raw,
        "covariance_term": covariance_term,
        "mean_term": mean_term,
        "normalization": normalization,
    }


def fixed_reference_empirical_fid(
    reference: ReferenceInput,
    generated_features: ArrayLike,
    *,
    covariance_normalization: str = "1/(N-1)",
    rtol: float = DEFAULT_EIG_RTOL,
    atol: float = DEFAULT_EIG_ATOL,
) -> dict[str, Any]:
    """Compute raw plug-in FID while keeping the reference statistics fixed."""
    resolved_reference = _resolve_reference(
        reference, covariance_normalization=covariance_normalization,
        rtol=float(rtol), atol=float(atol),
    )
    generated = np.asarray(generated_features, dtype=np.float64)
    if generated.shape[1] != resolved_reference.dimension:
        raise ValueError("reference and generated feature dimensions differ")
    components = _plugin_components(
        resolved_reference, generated, rtol=float(rtol), atol=float(atol),
    )
    return {
        "estimate_w2_sq_raw": float(components["raw"]),
        "covariance_w2_sq_hat_raw": float(components["covariance_term"]),
        "mean_w2_sq_hat": float(components["mean_term"]),
        "covariance_normalization": str(components["normalization"]),
        "n_samples_A": resolved_reference.sample_count,
        "n_samples_B": int(generated.shape[0]),
        "dimension": int(generated.shape[1]),
        "status": "ok",
        "error_message": "",
    }


def fixed_reference_mean_sq(
    reference_mean: ArrayLike,
    generated_features: ArrayLike,
) -> float:
    """Return the ordinary plug-in statistic ``||mean(Y) - mu_A||^2``."""

    mean = np.asarray(reference_mean, dtype=np.float64)
    generated = np.asarray(generated_features, dtype=np.float64)
    if mean.shape != (generated.shape[1],):
        raise ValueError("reference_mean and generated feature dimensions differ")
    delta = np.mean(generated, axis=0, dtype=np.float64) - mean
    value = float(delta @ delta)
    if not math.isfinite(value):
        raise FloatingPointError("mean FD term produced a non-finite output")
    return value


def helmert_contrasts(samples: ArrayLike) -> FloatArray:
    """Return the ``N-1`` sequential Helmert contrasts of ``N`` row samples.

    The transformation is deterministic, consumes all observations, and
    satisfies the exact scatter identity

    ``H.T @ H = sum_i (Y_i - mean(Y)) (Y_i - mean(Y)).T``

    up to floating-point error.  Thus ``H.T @ H / (N-1)`` is the ordinary
    unbiased sample covariance.  For iid Gaussian rows the contrast rows are
    independent ``N(0, Sigma)`` observations and are independent of the raw
    sample mean.
    """

    array = np.asarray(samples, dtype=np.float64)
    sample_count, dimension = array.shape
    if sample_count < 2:
        raise ValueError("at least two samples are required for Helmert contrasts")
    result = np.empty((sample_count - 1, dimension), dtype=np.float64)
    np.cumsum(array[:-1], axis=0, dtype=np.float64, out=result)
    counts = np.arange(1, sample_count, dtype=np.float64)
    result /= counts[:, None]
    result -= array[1:]
    result *= np.sqrt(counts / (counts + 1.0))[:, None]
    if not np.isfinite(result).all():
        raise FloatingPointError("Helmert contrasts produced non-finite output")
    return result


def _balanced_spans(length: int, parts: int) -> tuple[tuple[int, int], ...]:
    """Partition ``range(length)`` into consecutive spans without index arrays."""

    length, parts = int(length), int(parts)
    if length < 1 or not 1 <= parts <= length:
        raise ValueError("parts must satisfy 1 <= parts <= length")
    quotient, remainder = divmod(length, parts)
    spans: list[tuple[int, int]] = []
    start = 0
    for part in range(parts):
        stop = start + quotient + (1 if part < remainder else 0)
        spans.append((start, stop))
        start = stop
    return tuple(spans)


def _stream_chunk_size(
    dimension: int,
    observation_count: int,
    requested: Optional[int],
) -> int:
    """Choose the processing chunk width, defaulting to at most ``d`` rows."""

    if requested is None:
        return max(1, min(int(dimension), int(observation_count)))
    value = int(requested)
    if value < 1:
        raise ValueError("stream_chunk_size must advance the stream")
    return min(value, int(dimension), int(observation_count))


def _stream_helmert_scatter(
    samples: FloatArray,
    *,
    contrast_start: int,
    contrast_stop: int,
    chunk_size: int,
    known_zero_mean: bool = False,
) -> FloatArray:
    """Accumulate ``sum h_i h_i^T`` without materialising Helmert rows."""

    dimension = int(samples.shape[1])
    scatter = np.zeros((dimension, dimension), dtype=np.float64)
    position = int(contrast_start)
    stop = int(contrast_stop)
    prefix = np.sum(samples[:position], axis=0, dtype=np.float64)
    while position < stop:
        chunk_stop = min(position + int(chunk_size), stop)
        if known_zero_mean:
            contrasts = samples[position:chunk_stop]
        else:
            prefixes = np.cumsum(samples[position:chunk_stop], axis=0, dtype=np.float64)
            prefixes += prefix
            counts = np.arange(position + 1, chunk_stop + 1, dtype=np.float64)
            contrasts = prefixes / counts[:, None] - samples[position + 1:chunk_stop + 1]
            contrasts *= np.sqrt(counts / (counts + 1.0))[:, None]
            prefix = prefixes[-1].copy()
        scatter += contrasts.T @ contrasts
        position = chunk_stop
    return symmetrize(scatter)


def _stream_one_sided_atoms(
    samples: FloatArray,
    *,
    contrast_start: int,
    contrast_count: int,
    transform_b: FloatArray,
    n_atoms: int,
    chunk_size: int,
    support_projector: FloatArray,
    full_support: bool,
    known_zero_mean: bool = False,
) -> tuple[
    FloatArray,
    FloatArray,
    NDArray[np.int64],
    float,
]:
    """Stream covariance observations directly into atomic scatter sums."""

    spans = _balanced_spans(contrast_count, n_atoms)
    rank = int(transform_b.shape[0])
    covariance = np.zeros((n_atoms, rank, rank), dtype=np.float64)
    norm_sq = np.zeros(n_atoms, dtype=np.float64)
    counts = np.zeros(n_atoms, dtype=np.int64)
    leakage_sq = 0.0
    total_norm_sq = 0.0
    absolute_start = int(contrast_start)
    prefix = np.sum(samples[:absolute_start], axis=0, dtype=np.float64)
    for atom, (relative_start, relative_stop) in enumerate(spans):
        position = absolute_start + relative_start
        atom_stop = absolute_start + relative_stop
        counts[atom] = relative_stop - relative_start
        while position < atom_stop:
            chunk_stop = min(position + int(chunk_size), atom_stop)
            if known_zero_mean:
                contrasts = samples[position:chunk_stop]
            else:
                prefixes = np.cumsum(samples[position:chunk_stop], axis=0, dtype=np.float64)
                prefixes += prefix
                helmert_counts = np.arange(position + 1, chunk_stop + 1, dtype=np.float64)
                contrasts = prefixes / helmert_counts[:, None] - samples[position + 1:chunk_stop + 1]
                contrasts *= np.sqrt(helmert_counts / (helmert_counts + 1.0))[:, None]
                prefix = prefixes[-1].copy()
            transformed = contrasts @ transform_b.T
            covariance[atom] += transformed.T @ transformed
            chunk_norm_sq = float(
                np.einsum("ij,ij->", contrasts, contrasts, dtype=np.float64)
            )
            norm_sq[atom] += chunk_norm_sq
            total_norm_sq += chunk_norm_sq
            if not full_support:
                residual = contrasts - contrasts @ support_projector
                leakage_sq += float(
                    np.einsum("ij,ij->", residual, residual, dtype=np.float64)
                )
            position = chunk_stop
    leakage = math.sqrt(leakage_sq) / max(1.0, math.sqrt(total_norm_sq))
    return covariance, norm_sq, counts, leakage


def rtd_fixed_reference(
    reference: ReferenceInput,
    generated_features: ArrayLike,
    *,
    m0: Optional[int] = None,
    known_zero_mean: bool = False,
    L: Optional[int] = None,
    covariance_normalization: str = "1/(N-1)",
    rtol: float = DEFAULT_EIG_RTOL,
    atol: float = DEFAULT_EIG_ATOL,
    imag_tol: float = 1.0e-10,
    compensated_sum: bool = True,
    rho_design: float = 0.25,
    C_T: float = 1.0,
    degree_cap: Optional[Union[int, Mapping[Any, Any]]] = None,
    stream_chunk_size: Optional[int] = None,
) -> dict[str, Any]:
    """One-sided RTD: sample pilot, then independent Taylor corrections.

    The reference covariance is fixed. For unknown means, stream the N-1
    Helmert contrasts; their first m0 rows form the pilot, and the remaining
    rows form correction atoms. For known zero means with 1/N normalization,
    use the raw N rows instead. Gaussian contrasts are independent of each
    other and of the sample mean. The mean term remains the ordinary plug-in.

    ``m0`` defaults to half the covariance observations; the paper configs
    explicitly select their pilot fraction. No full contrast matrix or Taylor
    coefficient tensor is materialized. Raw estimates are never clipped.
    """
    dimension = 0
    metadata: dict[str, Any] = {}
    try:
        resolved_rtol, resolved_atol = float(rtol), float(atol)
        resolved_imag_tol = float(imag_tol)
        resolved_reference = _resolve_reference(
            reference, covariance_normalization=covariance_normalization,
            rtol=resolved_rtol, atol=resolved_atol,
        )
        generated = np.asarray(generated_features, dtype=np.float64)
        dimension = int(generated.shape[1])
        if dimension != resolved_reference.dimension:
            raise ValueError("reference and generated feature dimensions differ")
        if known_zero_mean and (resolved_reference.covariance_normalization != "1/N"
                                or np.any(resolved_reference.mean != 0.0)):
            raise ValueError("known_zero_mean requires zero reference mean and normalization 1/N")

        observation_count = int(generated.shape[0] - (0 if known_zero_mean else 1))
        chunk_size = _stream_chunk_size(dimension, max(1, observation_count), stream_chunk_size)
        resolved_m0 = observation_count // 2 if m0 is None else m0
        metadata.update({"m0": resolved_m0, "covariance_observation_count": observation_count})
        if resolved_m0 < 1 or resolved_m0 >= observation_count:
            raise _InsufficientSamplesError("m0 must leave fresh correction observations")
        n_correction = observation_count - resolved_m0
        requested, used, reason = choose_taylor_degree(
            dimension, n_correction, rho_design=float(rho_design), C_T=float(C_T),
            adaptive_degree=L is None, degree_cap=degree_cap,
            requested_degree=L,
        )
        metadata.update({
            "L_requested": requested, "L_used": used,
            "degree_truncation_reason": reason, "n_correction": n_correction,
        })
        if used == 0:
            raise _InsufficientSamplesError(
                "no feasible degree: require 2 <= L_used <= floor(n_correction/2)"
            )

        # Pilot coordinates: positive overlap support of sqrt(A) sqrt(B0).
        a0 = resolved_reference.covariance
        pilot_scatter = _stream_helmert_scatter(
            generated, contrast_start=0, contrast_stop=resolved_m0,
            known_zero_mean=known_zero_mean, chunk_size=chunk_size,
        )
        b0 = symmetrize(pilot_scatter / resolved_m0)
        factors_a = _support_factors(a0, rtol=resolved_rtol, atol=resolved_atol)
        factors_b = _support_factors(b0, rtol=resolved_rtol, atol=resolved_atol)
        singular_values, vt = _support_overlap_svd(
            factors_a, factors_b, rtol=resolved_rtol, atol=resolved_atol,
        )
        singular_values = np.asarray(singular_values, dtype=np.float64)
        v = np.asarray(vt, dtype=np.float64).T
        overlap_rank = int(singular_values.size)
        transform_b = v.T @ factors_b.pinv_sqrt
        atom_covariance, atom_norm_sq, atom_counts, leakage_b = _stream_one_sided_atoms(
            generated, contrast_start=resolved_m0, contrast_count=n_correction,
            transform_b=transform_b, n_atoms=used, chunk_size=chunk_size,
            support_projector=factors_b.projector,
            full_support=factors_b.rank == dimension, known_zero_mean=known_zero_mean,
        )
        support_tolerance = max(100.0 * resolved_atol, 100.0 * resolved_rtol, 1.0e-12)
        if not math.isfinite(leakage_b):
            raise FloatingPointError("pilot support leakage is non-finite")
        if leakage_b > support_tolerance:
            raise _PilotRankError(
                "fresh correction observations leave the B pilot support "
                f"(relative leakage={leakage_b:.3e})"
            )

        # Each degree uses all atoms, regrouped into disjoint balanced groups.
        identity = np.eye(overlap_rank, dtype=np.float64)
        nonlinear = 0.0
        if overlap_rank:
            for degree in range(2, used + 1):
                directions = []
                for atom_ids in recombine_atomic_blocks(used, degree):
                    sample_count = int(np.sum(atom_counts[atom_ids]))
                    directions.append(np.sum(atom_covariance[atom_ids], axis=0) / sample_count - identity)
                nonlinear += evaluate_one_sided_mixed_taylor_contraction(
                    singular_values, directions, compensated_sum=bool(compensated_sum),
                    imag_tol=resolved_imag_tol,
                )

        total_covariance_b = np.sum(atom_covariance, axis=0)
        q_bar_b = total_covariance_b / n_correction - identity
        linear_b = (
            float(np.sum(atom_norm_sq)) / n_correction
            - float(np.trace(b0))
            - float(np.dot(singular_values, np.diag(q_bar_b)))
        )
        base_covariance = (
            float(np.trace(a0)) + float(np.trace(b0))
            - 2.0 * float(np.sum(singular_values))
        )
        covariance_estimate = base_covariance + linear_b + nonlinear
        mean_estimate = (0.0 if known_zero_mean else fixed_reference_mean_sq(
            resolved_reference.mean, generated
        ))
        raw = covariance_estimate + mean_estimate
        if not all(math.isfinite(float(value)) for value in [raw, covariance_estimate, mean_estimate]):
            raise FloatingPointError("one-sided RTD produced a non-finite output")
        return {
            "estimate_w2_sq_raw": float(raw),
            "covariance_w2_sq_estimate": float(covariance_estimate),
            "mean_w2_sq_estimate": float(mean_estimate),
            "status": "ok", "error_message": "", "dimension": dimension,
            "n_samples_A": resolved_reference.sample_count,
            "n_samples_B": int(generated.shape[0]),
            "known_zero_mean": bool(known_zero_mean),
            "covariance_normalization": resolved_reference.covariance_normalization,
            **metadata,
        }
    except _InsufficientSamplesError as error:
        return _failure_result(dimension, status="skipped_runtime_cap", error=error, metadata=metadata)
    except _PilotRankError as error:
        return _failure_result(dimension, status="pilot_rank_failure", error=error, metadata=metadata)
    except Exception as error:
        return _failure_result(
            dimension, status="numerical_failure", error=f"{type(error).__name__}: {error}",
            metadata=metadata,
        )


__all__ = [
    "FixedGaussianReference",
    "fixed_reference_empirical_fid",
    "rtd_fixed_reference",
    "fixed_reference_mean_sq",
    "helmert_contrasts",
    "make_fixed_reference",
    "make_fixed_reference_from_statistics",
]

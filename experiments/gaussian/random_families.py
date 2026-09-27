"""Dimension-stable covariance families for scaling experiments.

The families in this module keep their spectral parameters fixed as the
ambient dimension changes.  This makes a change in estimator error easier to
attribute to dimension, rather than to a changing condition-number regime.
All defaults produce positive-definite covariance matrices whose operator norm
is at most one.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from .sampling import RNGInput, make_rng, random_orthogonal
from gaussian_w2._core.linalg import symmetrize


FloatArray = NDArray[np.float64]
SeedInput = Optional[Union[int, np.integer, np.random.SeedSequence]]


def _validate_dimension(dimension: int) -> int:
    if dimension <= 0:
        raise ValueError("dimension must be positive")
    return dimension


def _covariance_from_eigenvalues(
    eigenvalues: FloatArray, generator: np.random.Generator
) -> FloatArray:
    dimension = int(eigenvalues.size)
    basis = random_orthogonal(dimension, generator)
    covariance = (basis * eigenvalues[np.newaxis, :]) @ basis.T
    return symmetrize(np.asarray(covariance, dtype=np.float64))


def _log_uniform_spectrum(
    dimension: int,
    generator: np.random.Generator,
    lambda_min: float,
    lambda_max: float,
) -> FloatArray:
    return np.asarray(
        np.exp(
            generator.uniform(
                np.log(lambda_min), np.log(lambda_max), size=dimension
            )
        ),
        dtype=np.float64,
    )


def generate_haar_log_uniform_pair(
    dimension: int,
    rng: RNGInput = None,
    *,
    seed: SeedInput = None,
    lambda_min: float = 0.05,
    lambda_max: float = 1.0,
) -> Tuple[FloatArray, FloatArray]:
    """Draw a pair with independent log-uniform spectra and Haar bases.

    The eigenvalue interval is independent of ``dimension``.  Independent
    bases make the two covariance matrices noncommuting with probability one
    when ``dimension > 1`` and neither spectrum is exactly flat.
    """

    dimension = _validate_dimension(dimension)
    lambda_min = float(lambda_min)
    lambda_max = float(lambda_max)
    if not (0.0 < lambda_min < lambda_max <= 1.0):
        raise ValueError("require 0 < lambda_min < lambda_max <= 1")

    generator = make_rng(rng, seed=seed)
    spectrum_a = _log_uniform_spectrum(
        dimension, generator, lambda_min, lambda_max
    )
    spectrum_b = _log_uniform_spectrum(
        dimension, generator, lambda_min, lambda_max
    )
    a = _covariance_from_eigenvalues(spectrum_a, generator)
    b = _covariance_from_eigenvalues(spectrum_b, generator)
    return a, b


def _spiked_spectrum(
    dimension: int,
    generator: np.random.Generator,
    *,
    spike_fraction: float,
    bulk_min: float,
    bulk_max: float,
    spike_min: float,
    spike_max: float,
) -> FloatArray:
    number_spikes = min(
        dimension, max(1, int(np.ceil(spike_fraction * dimension)))
    )
    number_bulk = dimension - number_spikes
    bulk = generator.uniform(bulk_min, bulk_max, size=number_bulk)
    spikes = generator.uniform(spike_min, spike_max, size=number_spikes)
    spectrum = np.concatenate((bulk, spikes)).astype(np.float64, copy=False)
    generator.shuffle(spectrum)
    return spectrum


def generate_spiked_bulk_pair(
    dimension: int,
    rng: RNGInput = None,
    *,
    seed: SeedInput = None,
    spike_fraction: float = 0.1,
    bulk_min: float = 0.1,
    bulk_max: float = 0.3,
    spike_min: float = 0.7,
    spike_max: float = 1.0,
) -> Tuple[FloatArray, FloatArray]:
    """Draw heterogeneous bulk-and-spike covariances in independent bases.

    The number of spikes is ``ceil(spike_fraction * dimension)`` (with at
    least one spike).  Both spectra and both Haar bases are drawn
    independently.
    """

    dimension = _validate_dimension(dimension)
    spike_fraction = float(spike_fraction)
    bulk_min = float(bulk_min)
    bulk_max = float(bulk_max)
    spike_min = float(spike_min)
    spike_max = float(spike_max)
    if not (0.0 < spike_fraction <= 1.0):
        raise ValueError("require 0 < spike_fraction <= 1")
    if not (
        0.0 < bulk_min < bulk_max < spike_min < spike_max <= 1.0
    ):
        raise ValueError(
            "require 0 < bulk_min < bulk_max < spike_min < spike_max <= 1"
        )

    generator = make_rng(rng, seed=seed)
    common = dict(
        spike_fraction=spike_fraction,
        bulk_min=bulk_min,
        bulk_max=bulk_max,
        spike_min=spike_min,
        spike_max=spike_max,
    )
    spectrum_a = _spiked_spectrum(dimension, generator, **common)
    spectrum_b = _spiked_spectrum(dimension, generator, **common)
    a = _covariance_from_eigenvalues(spectrum_a, generator)
    b = _covariance_from_eigenvalues(spectrum_b, generator)
    return a, b


def _toeplitz_correlation(dimension: int, rho: float) -> FloatArray:
    indices = np.arange(dimension, dtype=np.int64)
    distances = np.abs(indices[:, np.newaxis] - indices[np.newaxis, :])
    return np.asarray(np.power(rho, distances), dtype=np.float64)


def _normalize_operator_norm(matrix: FloatArray, target: float) -> FloatArray:
    largest = float(np.linalg.eigvalsh(matrix)[-1])
    if not np.isfinite(largest) or largest <= 0.0:  # defensive; AR(1) is PD
        raise ValueError("Toeplitz covariance has invalid operator norm")
    return np.asarray(matrix * (target / largest), dtype=np.float64)


def generate_rotated_toeplitz_pair(
    dimension: int,
    rng: RNGInput = None,
    *,
    seed: SeedInput = None,
    rho_a: float = 0.4,
    rho_b: float = 0.7,
    operator_norm: float = 1.0,
) -> Tuple[FloatArray, FloatArray]:
    """Draw independently rotated, normalized AR(1)/Toeplitz covariances.

    Before rotation, the entries are ``T(rho)[i,j] = rho**abs(i-j)``.
    Each matrix is normalized to the requested operator norm and then given an
    independent Haar basis.  Consequently the pair retains structured spectra
    without sharing an eigenbasis.
    """

    dimension = _validate_dimension(dimension)
    rho_a = float(rho_a)
    rho_b = float(rho_b)
    operator_norm = float(operator_norm)
    if not (0.0 < rho_a < 1.0 and 0.0 < rho_b < 1.0):
        raise ValueError("require 0 < rho_a < 1 and 0 < rho_b < 1")
    if not (0.0 < operator_norm <= 1.0):
        raise ValueError("require 0 < operator_norm <= 1")

    generator = make_rng(rng, seed=seed)
    toeplitz_a = _normalize_operator_norm(
        _toeplitz_correlation(dimension, rho_a), operator_norm
    )
    toeplitz_b = _normalize_operator_norm(
        _toeplitz_correlation(dimension, rho_b), operator_norm
    )
    basis_a = random_orthogonal(dimension, generator)
    basis_b = random_orthogonal(dimension, generator)
    a = symmetrize(basis_a @ toeplitz_a @ basis_a.T)
    b = symmetrize(basis_b @ toeplitz_b @ basis_b.T)
    return np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)


_FAMILY_GENERATORS = {
    "haar_log_uniform": generate_haar_log_uniform_pair,
    "spiked_bulk": generate_spiked_bulk_pair,
    "rotated_toeplitz": generate_rotated_toeplitz_pair,
}

_FAMILY_PARAMETERS = {
    "haar_log_uniform": {"lambda_min", "lambda_max"},
    "spiked_bulk": {"spike_fraction", "bulk_min", "bulk_max", "spike_min", "spike_max"},
    "rotated_toeplitz": {"rho_a", "rho_b", "operator_norm"},
}


def family_parameters(family: str, config: Optional[Mapping[str, Any]]) -> tuple[str, dict]:
    """Validate the three paper families and their editable spectral parameters."""
    name = family.strip().lower()
    if name not in _FAMILY_GENERATORS:
        raise ValueError(f"unsupported covariance family {family!r}")
    if config is not None and not isinstance(config, Mapping):
        raise ValueError("config must be a mapping or None")
    parameters = dict(config or {})
    unknown = set(parameters) - _FAMILY_PARAMETERS[name]
    if unknown:
        raise ValueError(f"unknown configuration keys for {name}: {sorted(unknown)}")
    return name, parameters


def generate_scaling_covariance_pair(
    family: str,
    dimension: int,
    rng: RNGInput = None,
    *,
    seed: SeedInput = None,
    config: Optional[Mapping[str, Any]] = None,
) -> Tuple[FloatArray, FloatArray]:
    """Generate one random pair with the paper's original draw order."""
    name, parameters = family_parameters(family, config)
    generator = make_rng(rng, seed=seed)
    return _FAMILY_GENERATORS[name](dimension, generator, **parameters)

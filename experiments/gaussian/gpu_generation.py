"""The three paper random families, preserving their CUDA random draw order."""
from __future__ import annotations
import math
from typing import Any, Mapping, Optional
from gaussian_w2._core.gpu_backend import require_cupy
from gaussian_w2._core.gpu_linalg import _symmetrize
from .random_families import family_parameters


def _random_orthogonal(dimension: int, rng: Any, cp: Any) -> Any:
    gaussian = rng.standard_normal((dimension, dimension), dtype=cp.float64)
    q, r = cp.linalg.qr(gaussian)
    signs = cp.where(cp.diag(r) < 0.0, -1.0, 1.0)
    return q * signs[None, :]


def _covariance_from_spectrum(spectrum: Any, rng: Any, cp: Any) -> Any:
    basis = _random_orthogonal(int(spectrum.size), rng, cp)
    return _symmetrize((basis * spectrum[None, :]) @ basis.T)


def gpu_generate_scaling_covariance_pair(
    family: str,
    dimension: int,
    *,
    seed: int,
    config: Optional[Mapping[str, Any]] = None,
    device: Optional[int] = None,
) -> tuple[Any, Any]:
    """Generate one of the three random covariance families directly on GPU."""

    cp = require_cupy()
    selected = int(cp.cuda.Device().id if device is None else device)
    d = int(dimension)
    if d < 1:
        raise ValueError("dimension must be a positive integer")
    name, parameters = family_parameters(family, config)
    with cp.cuda.Device(selected):
        rng = cp.random.RandomState(int(seed) % (2**32))
        if name == "haar_log_uniform":
            low = float(parameters.get("lambda_min", 0.05))
            high = float(parameters.get("lambda_max", 1.0))
            if not (0.0 < low < high <= 1.0):
                raise ValueError("require 0 < lambda_min < lambda_max <= 1")
            spectrum_a = cp.exp(rng.uniform(math.log(low), math.log(high), size=d)).astype(cp.float64)
            spectrum_b = cp.exp(rng.uniform(math.log(low), math.log(high), size=d)).astype(cp.float64)
            pair = (
                _covariance_from_spectrum(spectrum_a, rng, cp),
                _covariance_from_spectrum(spectrum_b, rng, cp),
            )
        elif name == "spiked_bulk":
            fraction = float(parameters.get("spike_fraction", 0.1))
            bulk_min = float(parameters.get("bulk_min", 0.1))
            bulk_max = float(parameters.get("bulk_max", 0.3))
            spike_min = float(parameters.get("spike_min", 0.7))
            spike_max = float(parameters.get("spike_max", 1.0))
            if not (0.0 < fraction <= 1.0):
                raise ValueError("require 0 < spike_fraction <= 1")
            if not (0.0 < bulk_min < bulk_max < spike_min < spike_max <= 1.0):
                raise ValueError("invalid bulk/spike spectral intervals")
            n_spikes = min(d, max(1, int(math.ceil(fraction * d))))
            n_bulk = d - n_spikes

            def spectrum() -> Any:
                values = cp.concatenate(
                    (
                        rng.uniform(bulk_min, bulk_max, size=n_bulk),
                        rng.uniform(spike_min, spike_max, size=n_spikes),
                    )
                ).astype(cp.float64)
                rng.shuffle(values)
                return values

            # Preserve the original spectrum-A, basis-A, spectrum-B, basis-B
            # order. The CPU family uses its own, different random stream.
            pair = (
                _covariance_from_spectrum(spectrum(), rng, cp),
                _covariance_from_spectrum(spectrum(), rng, cp),
            )
        else:
            rho_a = float(parameters.get("rho_a", 0.4))
            rho_b = float(parameters.get("rho_b", 0.7))
            operator_norm = float(parameters.get("operator_norm", 1.0))
            if not (0.0 < rho_a < 1.0 and 0.0 < rho_b < 1.0):
                raise ValueError("require 0 < rho_a,rho_b < 1")
            if not (0.0 < operator_norm <= 1.0):
                raise ValueError("require 0 < operator_norm <= 1")
            indices = cp.arange(d, dtype=cp.int64)
            distances = cp.abs(indices[:, None] - indices[None, :])

            def rotated(rho: float) -> Any:
                toeplitz = cp.power(cp.float64(rho), distances).astype(cp.float64)
                largest = cp.linalg.eigvalsh(toeplitz)[-1]
                toeplitz *= operator_norm / largest
                basis = _random_orthogonal(d, rng, cp)
                return _symmetrize(basis @ toeplitz @ basis.T)

            pair = rotated(rho_a), rotated(rho_b)
        return pair

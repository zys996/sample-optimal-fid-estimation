"""CPU entry points for valid real (n, d) samples and fixed reference moments."""
import time
import numpy as np
from .._core.empirical import empirical_covariance
from .._core.fixed_reference_fid import FixedGaussianReference, fixed_reference_empirical_fid
from .._core.linalg import psd_eigh, symmetrize
from .extrapolation import extrapolation_sample_sizes, fit_inverse_sample_polynomial


def empirical_fixed_reference(reference: FixedGaussianReference, samples, *,
                              known_zero_mean=False, rtol=1e-10, atol=1e-12):
    """Empirical FD, using centered 1/(N-1) or known-zero-mean 1/N moments.

    The reference is always fixed. ``known_zero_mean=True`` requires its mean
    to be zero and its covariance convention to be 1/N.
    """
    if not known_zero_mean:
        return fixed_reference_empirical_fid(reference, samples, rtol=rtol, atol=atol)
    if reference.covariance_normalization != '1/N' or np.any(reference.mean != 0):
        raise ValueError('known_zero_mean requires a zero-mean 1/N reference')
    covariance, _, _ = empirical_covariance(samples, centered=False, normalization='1/N')
    if covariance.shape != reference.covariance.shape:
        raise ValueError('reference and generated feature dimensions differ')
    middle = symmetrize(reference.sqrt_covariance @ covariance @ reference.sqrt_covariance)
    values, _, _ = psd_eigh(middle, rtol=rtol, atol=atol)
    raw = float(np.trace(reference.covariance) + np.trace(covariance) - 2 * np.sqrt(values).sum())
    if not np.isfinite(raw):
        raise FloatingPointError('empirical FD produced a non-finite output')
    return {'estimate_w2_sq_raw': raw, 'status': 'ok',
            'known_zero_mean': True, 'covariance_normalization': '1/N'}


def plugin_curve(reference, samples, *, minimum_samples=5000, num_points=15,
                 sample_schedule='uniform_n', regression_orders=(1, 2, 3),
                 regression_weightings=('ordinary_ols',), seed=0,
                 known_zero_mean=False, rtol=1e-10, atol=1e-12):
    """Share finite-sample FD values across all requested extrapolation fits.

    Each smaller node uses a fresh NumPy permutation of the full pool. The
    maximum node uses the full pool unchanged. This preserves the paper's
    shuffle-without-replacement seed protocol, including its RNG call order.
    """
    samples = np.asarray(samples, dtype=np.float64)
    maximum = len(samples)
    sizes = extrapolation_sample_sizes(maximum, minimum_samples=minimum_samples,
                                      num_points=num_points, schedule=sample_schedule)
    rng = np.random.default_rng(seed)
    values, times = [], []
    started = time.perf_counter()
    for n in sizes:
        subset = samples if n == maximum else samples[rng.permutation(maximum)[:n]]
        point_started = time.perf_counter()
        result = empirical_fixed_reference(reference, subset, known_zero_mean=known_zero_mean,
                                           rtol=rtol, atol=atol)
        values.append(result['estimate_w2_sq_raw'])
        times.append(time.perf_counter() - point_started)
    curve_seconds = time.perf_counter() - started
    fits = {weighting: {order: fit_inverse_sample_polynomial(sizes, values, order=order,
                                                            weighting=weighting)
                        for order in regression_orders} for weighting in regression_weightings}
    return {'sample_sizes': sizes, 'plugin_values': tuple(values), 'point_seconds': times,
            'endpoint_plugin_w2_sq': values[-1], 'fits_by_weighting': fits,
            'curve_seconds': curve_seconds, 'status': 'ok'}

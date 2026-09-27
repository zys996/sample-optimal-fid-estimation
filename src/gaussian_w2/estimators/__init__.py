"""Paper estimators against a fixed Gaussian reference.

ImageNet uses centered 1/(N-1) moments and independent Helmert contrasts for
RTD under the Gaussian model. Gaussian simulations instead use known zero
means and raw 1/N covariance observations. Both hold the reference fixed and
split generated-side observations into a pilot and fresh correction data.
All reported estimates use the raw squared-distance value.
"""
from .._core.empirical import empirical_covariance
from .._core.rtd_utils import choose_taylor_degree
from .._core.fixed_reference_fid import (
    FixedGaussianReference, make_fixed_reference, make_fixed_reference_from_statistics,
    fixed_reference_empirical_fid,
    rtd_fixed_reference, helmert_contrasts,
)
from .._core.truth import gaussian_w2_sq_truth
from .fixed_reference import empirical_fixed_reference, plugin_curve
from .extrapolation import (
    extrapolation_sample_sizes, fit_inverse_sample_polynomial,
    inverse_sample_extrapolation_weights, apply_extrapolation_weights,
)

__all__ = [
    'empirical_covariance', 'choose_taylor_degree',
    'FixedGaussianReference', 'make_fixed_reference', 'make_fixed_reference_from_statistics',
    'fixed_reference_empirical_fid',
    'rtd_fixed_reference', 'helmert_contrasts',
    'empirical_fixed_reference', 'plugin_curve', 'gaussian_w2_sq_truth',
    'extrapolation_sample_sizes', 'fit_inverse_sample_polynomial',
    'inverse_sample_extrapolation_weights', 'apply_extrapolation_weights',
]

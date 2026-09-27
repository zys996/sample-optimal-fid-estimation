"""Public API and the CPU known-zero extension, against independent identities."""
import importlib
import numpy as np
import pytest
from gaussian_w2.estimators import (
    empirical_fixed_reference, make_fixed_reference_from_statistics,
    rtd_fixed_reference, gaussian_w2_sq_truth, plugin_curve,
)


def test_zero_mean_empirical_matches_covariance_truth():
    samples = np.random.default_rng(72).normal(size=(25, 3))
    a = np.diag([.2, .4, .8])
    reference = make_fixed_reference_from_statistics(np.zeros(3), a, covariance_normalization='1/N')
    expected = gaussian_w2_sq_truth(None, a, None, samples.T @ samples / len(samples))
    result = empirical_fixed_reference(reference, samples, known_zero_mean=True)
    assert result['estimate_w2_sq_raw'] == pytest.approx(expected['w2_sq_true'], abs=1e-13)


@pytest.mark.parametrize('rank', [2, 3])
def test_raw_zero_mean_rtd_matches_equivalent_helmert_observations(rank):
    # H H^T=I and H 1=0. Feeding H^T X to the preserved centered path
    # reconstructs precisely X as its covariance observations.
    n, d = 49, 3
    x = np.random.default_rng(71).normal(size=(n, d))
    x[:, rank:] = 0
    h = np.zeros((n, n+1))
    for i in range(n):
        denominator = np.sqrt((i+1)*(i+2))
        h[i, :i+1] = 1 / denominator
        h[i, i+1] = -(i+1) / denominator
    raw = h.T @ x
    a = np.eye(d)
    zero_reference = make_fixed_reference_from_statistics(np.zeros(d), a, covariance_normalization='1/N')
    centered_reference = make_fixed_reference_from_statistics(np.zeros(d), a)
    kwargs = dict(m0=30, L=3)
    observed = rtd_fixed_reference(zero_reference, x, known_zero_mean=True, **kwargs)
    expected = rtd_fixed_reference(centered_reference, raw, **kwargs)
    assert observed['status'] == expected['status'] == 'ok'
    assert observed['estimate_w2_sq_raw'] == pytest.approx(expected['estimate_w2_sq_raw'], abs=2e-12)
    assert observed['n_correction'] == n - 30
    assert observed['mean_w2_sq_estimate'] == 0


def test_wrong_zero_mean_reference_convention_is_rejected():
    ref = make_fixed_reference_from_statistics(np.zeros(2), np.eye(2))
    x = np.ones((10, 2))
    with pytest.raises(ValueError, match='1/N'):
        empirical_fixed_reference(ref, x, known_zero_mean=True)
    assert rtd_fixed_reference(ref, x, known_zero_mean=True)['status'] == 'numerical_failure'


def test_plugin_curve_preserves_shuffle_draw_order_and_endpoint():
    x = np.random.default_rng(5).normal(size=(60, 3))
    ref = make_fixed_reference_from_statistics(np.zeros(3), np.eye(3), covariance_normalization='1/N')
    result = plugin_curve(ref, x, minimum_samples=10, num_points=5, seed=79,
                          known_zero_mean=True, regression_weightings=('ordinary_ols', 'variance_aware'))
    rng = np.random.default_rng(79)
    expected = [empirical_fixed_reference(ref, x[rng.permutation(len(x))[:n]], known_zero_mean=True)['estimate_w2_sq_raw']
                for n in result['sample_sizes'][:-1]]
    expected.append(empirical_fixed_reference(ref, x, known_zero_mean=True)['estimate_w2_sq_raw'])
    assert result['plugin_values'] == pytest.approx(expected, abs=1e-14)
    assert set(result['fits_by_weighting']) == {'ordinary_ols', 'variance_aware'}


def test_gpu_public_module_imports_without_loading_cupy():
    import sys
    before = 'cupy' in sys.modules
    importlib.import_module('gaussian_w2.estimators.gpu')
    assert ('cupy' in sys.modules) == before


@pytest.mark.parametrize('stage', ['import', 'device_probe'])
def test_gpu_checks_do_not_swallow_keyboard_interrupt(monkeypatch, stage):
    from types import SimpleNamespace
    from gaussian_w2._core import gpu_backend

    def interrupted(*args):
        raise KeyboardInterrupt

    fake_cupy = SimpleNamespace(cuda=SimpleNamespace(runtime=SimpleNamespace(getDeviceCount=interrupted)))
    monkeypatch.setattr(gpu_backend.importlib, 'import_module',
                        interrupted if stage == 'import' else lambda name: fake_cupy)
    with pytest.raises(KeyboardInterrupt):
        gpu_backend.gpu_available()


@pytest.mark.parametrize('field', ['eigenvalues', 'eigenvectors'])
def test_nonfinite_eigendecomposition_cannot_become_valid_support(monkeypatch, field):
    from gaussian_w2._core.linalg import psd_eigh
    values, vectors = np.ones(2), np.eye(2)
    if field == 'eigenvalues':
        values[0] = np.nan
    else:
        vectors[0, 0] = np.nan
    monkeypatch.setattr(np.linalg, 'eigh', lambda matrix: (values, vectors))
    with pytest.raises(FloatingPointError, match='eigendecomposition'):
        psd_eigh(np.eye(2))


def test_nonfinite_svd_cannot_become_zero_overlap(monkeypatch):
    from gaussian_w2._core.rtd_utils import _support_factors, _support_overlap_svd
    factors = _support_factors(np.eye(2), rtol=1e-10, atol=1e-12)
    monkeypatch.setattr(np.linalg, 'svd', lambda matrix, **kwargs: (np.eye(2), np.full(2, np.nan), np.eye(2)))
    with pytest.raises(FloatingPointError, match='SVD'):
        _support_overlap_svd(factors, factors, rtol=1e-10, atol=1e-12)


def test_finite_means_that_overflow_distance_are_rejected():
    with np.errstate(over='ignore'), pytest.raises(FloatingPointError, match='distance'):
        gaussian_w2_sq_truth([1e308], [[1.]], [0.], [[1.]])

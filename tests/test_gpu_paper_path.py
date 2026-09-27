"""Check streamed moments on CPU and the complete CUDA path when available."""
from types import SimpleNamespace

import numpy as np
import pytest

from gaussian_w2._core import fixed_reference_gpu as gpu
from gaussian_w2._core.gpu_backend import gpu_available, require_cupy
from gaussian_w2._core.gpu_linalg import _psd_eigh


@pytest.mark.parametrize('bad', [np.nan, np.inf])
def test_nonfinite_eigensolver_outputs_are_rejected(monkeypatch, bad):
    monkeypatch.setattr(np.linalg, 'eigh', lambda matrix: (np.array([1., bad]), np.eye(2)))
    with pytest.raises(FloatingPointError, match='non-finite eigenvalues'):
        _psd_eigh(np.eye(2), rtol=1e-10, atol=1e-12, cp=np)


def test_negative_covariance_spectrum_is_still_rejected():
    with pytest.raises(ValueError, match='positive semidefinite'):
        _psd_eigh(np.diag([1., -.1]), rtol=1e-10, atol=1e-12, cp=np)


@pytest.mark.parametrize('mean, trace', [(np.nan, 1.), (np.inf, 1.), (0., np.inf)])
def test_empirical_fd_does_not_report_nonfinite_results(mean, trace):
    reference = SimpleNamespace(mean=np.array([mean]), trace=trace,
                                sqrt_covariance=np.eye(1), dimension=1)
    samples = np.array([[-1.], [.5], [1.]])
    with pytest.raises(FloatingPointError, match='non-finite output'):
        gpu._generated_plugin_value(reference, samples, known_zero_mean=False,
            covariance_normalization='1/(N-1)', rtol=1e-10, atol=1e-12, cp=np)


@pytest.mark.parametrize('known_zero_mean', [False, True])
@pytest.mark.parametrize('chunk_size', [1, 3])
def test_streamed_covariance_matches_explicit_observations(known_zero_mean, chunk_size):
    # The accumulation helpers accept an array module, so their indexing can
    # be tested without a GPU; this does not stand in for CUDA parity below.
    x = np.random.default_rng(212).normal(size=(41, 3))
    observations = x if known_zero_mean else gpu._helmert_from_validated(x, cp=np)
    scatter = gpu._stream_covariance_scatter(
        x, observation_start=7, observation_stop=27,
        known_zero_mean=known_zero_mean, chunk_size=chunk_size, cp=np,
    )
    np.testing.assert_allclose(scatter, observations[7:27].T @ observations[7:27], atol=1e-13)


@pytest.mark.parametrize('known_zero_mean', [False, True])
@pytest.mark.parametrize('on_support', [False, True])
def test_streamed_atoms_match_explicit_group_moments_and_support(known_zero_mean, on_support):
    rng = np.random.default_rng(310)
    x = rng.normal(size=(47, 3))
    if on_support:
        x[:, 2] = 0
    observations = x if known_zero_mean else gpu._helmert_from_validated(x, cp=np)
    basis = np.eye(3)[:, :2]
    transform = np.diag([2., 0.5]) @ basis.T
    factors = SimpleNamespace(rank=2, support_vectors=basis)
    start, count = 11, len(observations) - 11
    covariance, norms, counts, spans, leakage = gpu._stream_one_sided_atoms(
        x, observation_start=start, observation_count=count,
        known_zero_mean=known_zero_mean, transform_b=transform, n_atoms=4,
        chunk_size=3, factors_b=factors, cp=np,
    )
    for atom, (a, b) in enumerate(spans):
        group = observations[start + a:start + b]
        projected = group @ transform.T
        np.testing.assert_allclose(covariance[atom], projected.T @ projected, atol=1e-12)
        np.testing.assert_allclose(norms[atom], np.sum(group ** 2), atol=1e-12)
        assert counts[atom] == len(group)
    fresh = observations[start:]
    residual = fresh - (fresh @ basis) @ basis.T
    expected = np.linalg.norm(residual) / max(1., np.linalg.norm(fresh))
    assert leakage == pytest.approx(expected, abs=1e-14)
    assert (leakage < 1e-12) == on_support


CUDA_AVAILABLE = gpu_available()


def test_cached_gpu_reference_uses_current_rtd_support_tolerance(monkeypatch):
    from gaussian_w2._core.gpu_linalg import _support_factors
    from gaussian_w2._core.rtd_utils import _support_overlap_svd
    covariance = np.diag([1., 1e-12])
    cached = _support_factors(covariance, rtol=1e-10, atol=1e-12, cp=np)
    reference = SimpleNamespace(covariance=covariance, factors=cached)
    calls = []

    def resolve(*args, **kwargs):
        calls.append(kwargs['rtol'])
        return _support_factors(*args, **kwargs)

    monkeypatch.setattr(gpu, '_support_factors', resolve)
    assert gpu._rtd_reference_factors(reference, rtol=1e-10, atol=1e-12, cp=np) is cached
    assert calls == []
    tight = gpu._rtd_reference_factors(reference, rtol=1e-14, atol=0, cp=np)
    unit = _support_factors(np.eye(2), rtol=1e-14, atol=0, cp=np)
    values, _ = _support_overlap_svd(tight, unit, rtol=1e-14, atol=0)
    np.testing.assert_allclose(values, [1., 1e-6], atol=1e-15, rtol=0)
    assert cached.rank == 1 and tight.rank == 2
    assert reference.factors is cached
    np.testing.assert_array_equal(cached.full_sqrt, np.diag([1., 1e-6]))
    assert calls == [1e-14]


@pytest.mark.skipif(not CUDA_AVAILABLE, reason='requires a real CuPy CUDA device')
@pytest.mark.parametrize('known_zero_mean', [False, True])
@pytest.mark.parametrize('rank', [2, 3])
def test_cuda_rtd_and_empirical_match_cpu_on_identical_data(known_zero_mean, rank):
    from gaussian_w2._core.fixed_reference_fid import (
        make_fixed_reference_from_statistics, rtd_fixed_reference,
    )
    from gaussian_w2.estimators.fixed_reference import empirical_fixed_reference
    cp = require_cupy()
    rng = np.random.default_rng(1701)
    x = rng.normal(size=(161, 3))
    x[:, rank:] = 0
    a = np.diag([1.2, 0.9, 0.7])
    normalization = '1/N' if known_zero_mean else '1/(N-1)'
    cpu_ref = make_fixed_reference_from_statistics(
        np.zeros(3), a, sample_count=200, covariance_normalization=normalization,
    )
    gpu_ref = gpu.prepare_gpu_fixed_reference_from_statistics(
        np.zeros(3), a, sample_count=200, covariance_normalization=normalization,
    )
    options = dict(m0=100, L=3, known_zero_mean=known_zero_mean, stream_chunk_size=2)
    expected = rtd_fixed_reference(cpu_ref, x, **options)
    observed = gpu.gpu_rtd_fixed_reference(
        gpu_ref, cp.asarray(x), covariance_normalization=normalization, **options,
    )
    assert expected['status'] == observed['status'] == 'ok'
    assert observed['estimate_w2_sq_raw'] == pytest.approx(expected['estimate_w2_sq_raw'], abs=1e-9)
    for key in ('m0', 'n_correction', 'L_used', 'L_requested'):
        assert observed[key] == expected[key]
    assert observed['gpu_method_seconds'] >= 0
    expected_plugin = empirical_fixed_reference(cpu_ref, x, known_zero_mean=known_zero_mean)
    observed_plugin = gpu.gpu_fixed_reference_empirical(
        gpu_ref, cp.asarray(x), covariance_normalization=normalization, known_zero_mean=known_zero_mean,
    )
    assert observed_plugin['estimate_w2_sq_raw'] == pytest.approx(expected_plugin['estimate_w2_sq_raw'], abs=1e-8)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason='requires a real CuPy CUDA device')
def test_cuda_reports_pilot_support_failure_and_insufficient_correction():
    x = np.random.default_rng(88).normal(size=(40, 5))
    ref = gpu.prepare_gpu_fixed_reference_from_statistics(
        np.zeros(5), np.eye(5), sample_count=None, reference_mode='exact_population_covariance',
        covariance_normalization='1/N',
    )
    result = gpu.gpu_rtd_fixed_reference(
        ref, x, known_zero_mean=True, covariance_normalization='1/N', m0=2, L=3,
    )
    assert result['status'] == 'pilot_rank_failure'
    assert np.isnan(result['estimate_w2_sq_raw'])
    result = gpu.gpu_rtd_fixed_reference(
        ref, x, known_zero_mean=True, covariance_normalization='1/N', m0=38, L=3,
    )
    assert result['status'] == 'skipped_runtime_cap'


@pytest.mark.skipif(not CUDA_AVAILABLE, reason='requires a real CuPy CUDA device')
def test_cuda_rotated_rank_one_support_and_rtd_match_analytic_value():
    from gaussian_w2._core.gpu_linalg import _support_factors
    from gaussian_w2._core.rtd_utils import _support_overlap_svd
    cp = require_cupy()
    direction = cp.asarray([1., 2., 3.]) / cp.sqrt(14.)
    projection = cp.outer(direction, direction)
    factors = _support_factors(projection, rtol=1e-10, atol=1e-12, cp=cp)
    identity = _support_factors(cp.eye(3), rtol=1e-10, atol=1e-12, cp=cp)
    values, vt = _support_overlap_svd(identity, factors, rtol=1e-10, atol=1e-12, xp=cp)
    assert factors.rank == len(values) == 1
    assert vt.shape == (1, 3)
    cp.testing.assert_allclose(values, cp.ones(1), atol=2e-14, rtol=0)
    reference = gpu.prepare_gpu_fixed_reference_from_statistics(
        cp.zeros(3), cp.eye(3), covariance_normalization='1/N',
    )
    result = gpu.gpu_rtd_fixed_reference(
        reference, cp.tile(direction, (20, 1)), covariance_normalization='1/N',
        known_zero_mean=True, m0=10, L=3,
    )
    assert result['status'] == 'ok'
    assert result['estimate_w2_sq_raw'] == pytest.approx(2., abs=5e-13, rel=0)

    # A real tiny eigenvalue still belongs to full roots, and to the RTD
    # support when an explicitly tighter numerical threshold retains it.
    small = cp.diag(cp.asarray([1., 1e-12]))
    for rtol, expected_rank in [(1e-10, 1), (1e-14, 2)]:
        small_factors = _support_factors(small, rtol=rtol, atol=0, cp=cp)
        unit_factors = _support_factors(cp.eye(2), rtol=rtol, atol=0, cp=cp)
        values, _ = _support_overlap_svd(unit_factors, small_factors, rtol=rtol, atol=0, xp=cp)
        assert len(values) == expected_rank
        cp.testing.assert_allclose(small_factors.full_sqrt, cp.diag(cp.asarray([1., 1e-6])),
                                   atol=1e-15, rtol=0)

    # Construct with the default cutoff, then request finer support at call
    # time. Both pilot and correction atoms have identity covariance exactly.
    reference = gpu.prepare_gpu_fixed_reference_from_statistics(
        cp.zeros(2), small, covariance_normalization='1/N',
    )
    observations = cp.tile(cp.sqrt(2.) * cp.eye(2), (12, 1))
    result = gpu.gpu_rtd_fixed_reference(
        reference, observations, covariance_normalization='1/N',
        known_zero_mean=True, m0=8, L=2, rtol=1e-14, atol=0,
    )
    assert result['status'] == 'ok'
    assert result['estimate_w2_sq_raw'] == pytest.approx((1 - 1e-6) ** 2, abs=5e-13, rel=0)

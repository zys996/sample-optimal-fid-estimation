"""Independent support and exact low-rank checks for one-sided RTD."""
import numpy as np
import pytest

from gaussian_w2._core.fixed_reference_fid import (
    make_fixed_reference_from_statistics, rtd_fixed_reference,
)
from gaussian_w2._core.gpu_linalg import _support_factors as gpu_support_factors
from gaussian_w2._core.linalg import psd_sqrt
from gaussian_w2._core.rtd_utils import _support_factors, _support_overlap_svd
from gaussian_w2._core.truth import gaussian_w2_sq_truth


@pytest.fixture(params=['cpu', 'gpu_arithmetic'])
def factors(request):
    # NumPy exercises the GPU factor formulas without claiming CUDA coverage.
    def build(matrix, *, rtol=1e-10, atol=1e-12):
        if request.param == 'cpu':
            return _support_factors(matrix, rtol=rtol, atol=atol)
        return gpu_support_factors(matrix, rtol=rtol, atol=atol, cp=np)
    return build


@pytest.mark.parametrize('pair', ['rank_one_left', 'rank_one_right', 'orthogonal', 'zero'])
def test_overlap_uses_only_retained_eigenspaces(factors, pair):
    v = np.array([1., 2., 3.]) / np.sqrt(14.)
    w = np.array([2., -1., 0.]) / np.sqrt(5.)
    projection = np.outer(v, v)
    a, b, expected_rank = {
        'rank_one_left': (projection, np.eye(3), 1),
        'rank_one_right': (np.eye(3), projection, 1),
        'orthogonal': (projection, np.outer(w, w), 0),
        'zero': (np.zeros((3, 3)), projection, 0),
    }[pair]
    fa, fb = factors(a), factors(b)
    singular_values, vt = _support_overlap_svd(fa, fb, rtol=1e-10, atol=1e-12)
    assert len(singular_values) == expected_rank
    assert len(singular_values) <= min(fa.rank, fb.rank)
    assert vt.shape == (expected_rank, 3)
    np.testing.assert_allclose(singular_values, np.ones(expected_rank), atol=2e-14, rtol=0)
    np.testing.assert_allclose(vt @ (fb.support_vectors @ fb.support_vectors.T), vt,
                               atol=2e-14, rtol=0)


@pytest.mark.parametrize('small, rtol, expected_rank', [
    (1e-8, 1e-10, 2), (1e-12, 1e-10, 1), (1e-12, 1e-14, 2),
])
def test_true_small_positive_support_follows_explicit_tolerance(factors, small, rtol, expected_rank):
    a = np.diag([1., small])
    fa, fb = factors(a, rtol=rtol, atol=0), factors(np.eye(2), rtol=rtol, atol=0)
    values, _ = _support_overlap_svd(fa, fb, rtol=rtol, atol=0)
    assert len(values) == expected_rank
    expected = [1., np.sqrt(small)] if expected_rank == 2 else [1.]
    np.testing.assert_allclose(values, expected, atol=1e-15, rtol=0)


def test_general_square_roots_and_truth_keep_true_tiny_positive_spectrum():
    small = 1e-12
    a = np.diag([1., small])
    expected_root = np.diag([1., np.sqrt(small)])
    reference = make_fixed_reference_from_statistics(np.zeros(2), a)
    gpu_factors = gpu_support_factors(a, rtol=1e-10, atol=1e-12, cp=np)
    for root in (psd_sqrt(a), reference.sqrt_covariance, gpu_factors.full_sqrt):
        np.testing.assert_allclose(root, expected_root, atol=1e-15, rtol=0)
    truth = gaussian_w2_sq_truth(None, a, None, np.eye(2))['w2_sq_true']
    assert truth == pytest.approx((1 - np.sqrt(small)) ** 2, abs=1e-14, rel=0)


@pytest.mark.parametrize('known_zero_mean', [False, True])
@pytest.mark.parametrize('degree', [2, 3, 4])
def test_rotated_rank_one_rtd_matches_exact_constant_covariance(known_zero_mean, degree):
    # Every covariance observation has outer product vv.T, in both pilot and
    # correction. All Taylor corrections therefore vanish and FD(I, vv.T)=2.
    normalization = '1/N' if known_zero_mean else '1/(N-1)'
    reference = make_fixed_reference_from_statistics(np.zeros(3), np.eye(3),
                                                    covariance_normalization=normalization)
    n = 20
    for direction in (np.array([1., 0., 0.]), np.array([1., 2., 3.]) / np.sqrt(14.)):
        observations = np.tile(direction, (n, 1))
        if known_zero_mean:
            samples = observations
        else:
            h = np.zeros((n, n + 1))
            for i in range(n):
                denominator = np.sqrt((i + 1) * (i + 2))
                h[i, :i + 1] = 1 / denominator
                h[i, i + 1] = -(i + 1) / denominator
            samples = h.T @ observations
        result = rtd_fixed_reference(reference, samples, known_zero_mean=known_zero_mean,
                                     m0=10, L=degree)
        assert result['status'] == 'ok'
        assert result['estimate_w2_sq_raw'] == pytest.approx(2., abs=5e-13, rel=0)

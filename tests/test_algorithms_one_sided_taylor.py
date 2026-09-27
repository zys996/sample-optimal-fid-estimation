"""Independent checks of the one-sided square-root expansion."""

import itertools
import math

import numpy as np
import pytest
from scipy.linalg import sqrtm

from gaussian_w2._core.gpu_backend import gpu_available, require_cupy
from gaussian_w2._core.one_sided_taylor import (
    evaluate_one_sided_homogeneous_taylor_coefficient as homogeneous,
    evaluate_one_sided_mixed_taylor_contraction as mixed,
    evaluate_one_sided_homogeneous_taylor_coefficient_gpu as homogeneous_gpu,
    evaluate_one_sided_mixed_taylor_contraction_gpu as mixed_gpu,
)


def half_binomial(degree):
    return math.prod(0.5 - j for j in range(degree)) / math.factorial(degree)


@pytest.mark.parametrize("degree", range(2, 7))
def test_diagonal_homogeneous_matches_scalar_binomial_series(degree):
    s = np.array([0.3, 0.8, 1.4])
    direction = np.array([0.2, -0.3, 0.1])
    expected = -2 * half_binomial(degree) * np.sum(s * direction ** degree)
    assert homogeneous(s, np.diag(direction), degree) == pytest.approx(expected, abs=2e-15)


@pytest.mark.parametrize("degree", range(2, 7))
def test_diagonal_mixed_matches_product_formula(degree):
    rng = np.random.default_rng(31 + degree)
    s = np.array([0.3, 0.8, 1.4])
    diagonals = rng.normal(scale=0.1, size=(degree, 3))
    expected = -2 * half_binomial(degree) * np.sum(s * np.prod(diagonals, axis=0))
    assert mixed(s, [np.diag(row) for row in diagonals]) == pytest.approx(expected, abs=3e-14)


@pytest.mark.parametrize("degree", [2, 3, 4])
def test_noncommuting_coefficient_matches_cauchy_integral_of_matrix_sqrt(degree):
    # An independent reference: extract a Taylor coefficient by integrating
    # the matrix square root around a small complex circle (no recurrence).
    s = np.array([0.7, 1.2, 1.9])
    v = np.array([[0.1, -0.06, 0.03], [-0.06, -0.08, 0.04], [0.03, 0.04, 0.05]])
    theta = 2 * np.pi * np.arange(128) / 128
    radius = 0.4
    values = []
    for z in radius * np.exp(1j * theta):
        matrix = s[:, None] * (np.eye(3) + z * v) * s[None, :]
        values.append(-2 * np.trace(sqrtm(matrix)))
    expected = np.mean(np.asarray(values) * np.exp(-1j * degree * theta)) / radius ** degree
    assert abs(expected.imag) < 1e-11
    assert homogeneous(s, v, degree) == pytest.approx(float(expected.real), abs=2e-11)


def test_mixed_contraction_is_permutation_symmetric():
    rng = np.random.default_rng(319)
    raw = rng.normal(scale=0.1, size=(3, 3, 3))
    directions = [(x + x.T) / 2 for x in raw]
    s = np.array([0.5, 0.9, 1.3])
    expected = mixed(s, directions)
    for order in itertools.permutations(directions):
        assert mixed(s, order) == pytest.approx(expected, abs=2e-14)


def test_zero_direction_returns_zero_without_diagnostic_failure():
    assert mixed([1.0], [np.zeros((1, 1)), np.zeros((1, 1))]) == 0.0


@pytest.mark.parametrize("s", [[0.0, 1.0], [-0.1, 1.0]])
def test_nonpositive_support_is_rejected(s):
    with pytest.raises(ValueError, match="positive"):
        homogeneous(s, np.eye(2), 2)


def test_nonfinite_arithmetic_is_rejected():
    with pytest.raises(FloatingPointError):
        homogeneous([1.0], [[np.inf]], 2)
    with pytest.raises(FloatingPointError):
        homogeneous([1.0], [[1e300]], 4)


@pytest.mark.skipif(not gpu_available(), reason="requires CUDA and CuPy")
@pytest.mark.parametrize("degree", [2, 3, 4])
def test_gpu_matches_cpu(degree):
    cp = require_cupy()
    rng = np.random.default_rng(510 + degree)
    s = np.array([0.4, 0.9, 1.3])
    raw = rng.normal(scale=0.1, size=(degree, 3, 3))
    directions = [(x + x.T) / 2 for x in raw]
    assert homogeneous_gpu(s, directions[0], degree, cp=cp) == pytest.approx(
        homogeneous(s, directions[0], degree), abs=3e-13
    )
    assert mixed_gpu(s, directions, cp=cp) == pytest.approx(mixed(s, directions), abs=3e-12)

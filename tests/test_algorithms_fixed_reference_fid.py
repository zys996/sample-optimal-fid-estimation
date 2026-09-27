"""Fixed-reference protocols, independent scalar formulas, and streaming."""

import math

import numpy as np
import pytest

import gaussian_w2._core.fixed_reference_fid as fixed_reference_module
from gaussian_w2._core.fixed_reference_fid import (
    fixed_reference_empirical_fid,
    rtd_fixed_reference as rtd,
    fixed_reference_mean_sq,
    helmert_contrasts,
    make_fixed_reference,
    make_fixed_reference_from_statistics,
)
from gaussian_w2.estimators.fixed_reference import plugin_curve


def test_fixed_reference_plugin_matches_scalar_gaussian_formula():
    real = np.random.default_rng(871).normal(loc=0.2, size=(45, 1))
    generated = np.random.default_rng(950).normal(loc=-0.1, scale=0.6, size=(31, 1))
    reference = make_fixed_reference(real)
    expected = (real.mean() - generated.mean()) ** 2 + (
        real.std(ddof=1) - generated.std(ddof=1)
    ) ** 2
    observed = fixed_reference_empirical_fid(reference, generated)
    assert observed["estimate_w2_sq_raw"] == pytest.approx(expected, abs=3e-14)
    assert observed["covariance_normalization"] == "1/(N-1)"
    assert not reference.mean.flags.writeable
    assert not reference.covariance.flags.writeable


def test_fid_infinity_matches_manual_subsamples_and_least_squares():
    generated = np.random.default_rng(950).normal(loc=0.1, scale=0.6, size=(40, 1))
    reference = make_fixed_reference_from_statistics([0.2], [[1.3]])
    sizes = np.array([10, 20, 30, 40])
    rng = np.random.default_rng(112)
    values = []
    for n in sizes:
        x = generated if n == 40 else generated[rng.permutation(40)[:n]]
        values.append((x.mean() - 0.2) ** 2 + (x.std(ddof=1) - np.sqrt(1.3)) ** 2)
    expected = np.linalg.lstsq(np.column_stack([np.ones(4), 1 / sizes]), values, rcond=None)[0][0]
    observed = plugin_curve(reference, generated, minimum_samples=10, num_points=4,
                            seed=112, regression_orders=(1,))
    assert observed["fits_by_weighting"]["ordinary_ols"][1]["intercept"] == pytest.approx(expected, abs=3e-14)
    assert list(observed["sample_sizes"]) == sizes.tolist()
    assert observed["plugin_values"] == pytest.approx(values, abs=3e-14)


def test_mean_plugin_matches_squared_sample_mean_difference():
    generated = np.array([[0.2, -0.3], [1.4, 0.1], [-0.5, 0.8], [0.7, -1.1]])
    reference_mean = np.array([0.1, -0.2])
    delta = generated.mean(axis=0) - reference_mean
    assert fixed_reference_mean_sq(reference_mean, generated) == pytest.approx(delta @ delta, abs=2e-15)


def test_helmert_contrasts_reproduce_centered_scatter():
    samples = np.random.default_rng(14).normal(size=(27, 3))
    contrasts = helmert_contrasts(samples)
    residuals = samples - samples.mean(axis=0)
    assert contrasts.shape == (26, 3)
    assert contrasts.T @ contrasts == pytest.approx(residuals.T @ residuals, abs=2e-14)


def test_default_pilot_and_determinism():
    ref = make_fixed_reference_from_statistics([0.2, -0.1], np.diag([0.7, 1.1]))
    x = np.random.default_rng(6002).normal(size=(80, 2))
    first, second = rtd(ref, x, L=3), rtd(ref, x, L=3)
    assert first["status"] == "ok", first["error_message"]
    assert first["m0"] == 39
    assert first["n_correction"] == 40
    assert first["covariance_observation_count"] == 79
    assert first["mean_w2_sq_estimate"] == fixed_reference_mean_sq(ref.mean, x)
    assert first["estimate_w2_sq_raw"] == second["estimate_w2_sq_raw"]


@pytest.mark.parametrize("known_zero_mean", [False, True])
def test_scalar_sample_pilot_matches_explicit_binomial_polynomial(known_zero_mean):
    # Evaluate the full RTD formula independently in one dimension; using
    # 31 correction observations also tests weighting of unequal atom sizes.
    x = np.random.default_rng(804).normal(scale=np.sqrt(2.2), size=(61, 1))
    a, m0, degree = 1.7, 29, 4
    ref = make_fixed_reference_from_statistics([0.0], [[a]], covariance_normalization="1/N" if known_zero_mean else "1/(N-1)")
    observations = x.ravel() if known_zero_mean else helmert_contrasts(x).ravel()
    b0 = np.mean(observations[:m0] ** 2)
    correction = observations[m0:]
    atoms = np.array_split(np.arange(len(correction)), degree)
    s = math.sqrt(a * b0)
    bhat = np.mean(correction ** 2)
    expected = a + b0 - 2 * s + bhat - b0 - s * (bhat / b0 - 1)
    for k in range(2, degree + 1):
        groups = np.array_split(np.arange(degree), k)
        directions = [np.mean(correction[np.concatenate([atoms[i] for i in group])] ** 2) / b0 - 1 for group in groups]
        half_binomial = math.prod(0.5 - j for j in range(k)) / math.factorial(k)
        expected += -2 * s * half_binomial * math.prod(directions)
    if not known_zero_mean:
        expected += x.mean() ** 2
    observed = rtd(ref, x, m0=m0, L=degree, known_zero_mean=known_zero_mean)
    assert observed["status"] == "ok", observed["error_message"]
    assert observed["estimate_w2_sq_raw"] == pytest.approx(expected, abs=3e-13)


def test_streaming_does_not_materialize_helmert_matrix(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("materialized Helmert rows")
    monkeypatch.setattr(fixed_reference_module, "helmert_contrasts", forbidden)
    result = rtd(make_fixed_reference_from_statistics(np.zeros(3), np.eye(3)), np.random.default_rng(6003).normal(size=(65, 3)), m0=24, L=3, stream_chunk_size=2)
    assert result["status"] == "ok", result["error_message"]


@pytest.mark.parametrize("known_zero_mean", [False, True])
@pytest.mark.parametrize("chunk_size", [1, 2, 7, 10_000])
def test_streaming_is_chunk_size_invariant(chunk_size, known_zero_mean):
    ref = make_fixed_reference_from_statistics(np.zeros(3), np.array([[1.1, 0.08, 0], [0.08, 0.9, 0.04], [0, 0.04, 0.7]]), covariance_normalization="1/N" if known_zero_mean else "1/(N-1)")
    x = np.random.default_rng(701).normal(size=(73, 3))
    expected = rtd(ref, x, m0=28, L=4, known_zero_mean=known_zero_mean, stream_chunk_size=3)
    observed = rtd(ref, x, m0=28, L=4, known_zero_mean=known_zero_mean, stream_chunk_size=chunk_size)
    assert expected["status"] == observed["status"] == "ok"
    assert observed["estimate_w2_sq_raw"] == pytest.approx(expected["estimate_w2_sq_raw"], abs=2e-11)


def test_point_mass_generated_distribution():
    ref = make_fixed_reference_from_statistics([0.3, -0.2], np.diag([0.7, 1.2]))
    x = np.tile([-0.1, 0.8], (16, 1))
    result = rtd(ref, x, L=2)
    assert result["status"] == "ok", result["error_message"]
    assert result["estimate_w2_sq_raw"] == pytest.approx(1.9 + 0.4 ** 2 + 1.0 ** 2, abs=1e-14)


def test_orthogonal_supports_keep_the_linear_covariance_trace():
    ref = make_fixed_reference_from_statistics([0, 0], np.diag([1, 0]), covariance_normalization="1/N")
    x = np.zeros((20, 2))
    x[:, 1] = np.random.default_rng(20).normal(size=20)
    result = rtd(ref, x, m0=10, L=3, known_zero_mean=True)
    assert result["status"] == "ok", result["error_message"]
    assert result["estimate_w2_sq_raw"] == pytest.approx(1 + np.mean(x[10:, 1] ** 2), abs=1e-14)


def test_correction_outside_pilot_support_is_an_explicit_failure():
    ref = make_fixed_reference_from_statistics([0, 0], np.eye(2), covariance_normalization="1/N")
    x = np.array([[1, 0]] * 4 + [[1, 1]] * 8, dtype=float)
    result = rtd(ref, x, m0=4, L=2, known_zero_mean=True)
    assert result["status"] == "pilot_rank_failure"
    assert math.isnan(result["estimate_w2_sq_raw"])


def test_small_budget_is_recorded_as_unavailable():
    result = rtd(make_fixed_reference_from_statistics([0], [[1]]), [[-1], [0.5], [1]], L=2)
    assert result["status"] == "skipped_runtime_cap"
    assert result["covariance_observation_count"] == 2
    assert math.isnan(result["estimate_w2_sq_raw"])


@pytest.mark.parametrize("call", [lambda: helmert_contrasts([[1, 2]]), lambda: make_fixed_reference([[1, 2]])])
def test_insufficient_samples_are_rejected(call):
    with pytest.raises(ValueError):
        call()


def test_finite_samples_with_overflow_cannot_return_valid_moments_or_fd():
    from gaussian_w2._core.empirical import empirical_covariance
    from gaussian_w2.estimators.fixed_reference import empirical_fixed_reference
    samples = np.full((10, 1), 1e200)
    reference = make_fixed_reference_from_statistics([0.], [[1.]])
    zero_reference = make_fixed_reference_from_statistics([0.], [[1.]], covariance_normalization='1/N')
    with np.errstate(over='ignore', invalid='ignore'):
        for call in (
            lambda: empirical_covariance(samples),
            lambda: fixed_reference_empirical_fid(reference, samples),
            lambda: empirical_fixed_reference(zero_reference, samples, known_zero_mean=True),
            lambda: fixed_reference_mean_sq(reference.mean, samples),
        ):
            with pytest.raises((FloatingPointError, ValueError, np.linalg.LinAlgError)):
                call()
        result = rtd(reference, samples, m0=4, L=2)
    assert result['status'] != 'ok'
    assert not np.isfinite(result['estimate_w2_sq_raw'])

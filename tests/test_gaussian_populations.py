"""Population geometry and random-stream regression checks for the paper families."""
import numpy as np
import pytest

from experiments.gaussian.random_families import generate_scaling_covariance_pair
from experiments.gaussian.stress_families import build_fixed_reference_pair
from gaussian_w2.estimators import gaussian_w2_sq_truth


# The first three entries of A[0] and B[0], from the pre-cleanup CPU generator.
# These anchors catch changes in the order of spectrum, shuffle, and Haar draws.
RANDOM_ANCHORS = {
    "haar_log_uniform": [
        .4136530926402184, .023499359482291836, -.0438689975814341,
        .19313947631400818, -.020330914170891168, .039584094477809625,
    ],
    "spiked_bulk": [
        .21252056283074958, .039192122171287605, -.025934485653190585,
        .31686501473572515, -.05495281682659859, .1360339406276045,
    ],
    "rotated_toeplitz": [
        .51440986187286, -.052858076135744586, .12425870721019781,
        .15293707356669026, .006487024259021732, .060044649397600125,
    ],
}


@pytest.mark.parametrize("family", RANDOM_ANCHORS)
def test_random_population_draw_order_and_spectrum(family):
    a, b = generate_scaling_covariance_pair(family, 8, seed=7)
    np.testing.assert_allclose(np.r_[a[0, :3], b[0, :3]], RANDOM_ANCHORS[family],
                               rtol=1e-12, atol=1e-14)
    for matrix in (a, b):
        np.testing.assert_allclose(matrix, matrix.T, atol=1e-14)
        eigenvalues = np.linalg.eigvalsh(matrix)
        assert np.all((eigenvalues > 0) & (eigenvalues <= 1 + 1e-12))
        if family == "haar_log_uniform":
            assert eigenvalues[0] >= .05
        elif family == "spiked_bulk":
            assert np.count_nonzero(eigenvalues > .7) == 1
            assert np.all((eigenvalues[:-1] >= .1) & (eigenvalues[:-1] <= .3))
        else:
            assert eigenvalues[-1] == pytest.approx(1.)
    assert np.linalg.norm(a @ b - b @ a) > 1e-3
    again = generate_scaling_covariance_pair(family, 8, seed=7)
    for first, repeated in zip((a, b), again):
        np.testing.assert_array_equal(first, repeated)
    changed, _ = generate_scaling_covariance_pair(family, 8, seed=8)
    assert not np.allclose(a, changed)


@pytest.mark.parametrize("fraction", [.25, .5, .75])
@pytest.mark.parametrize("randomized", [False, True])
def test_rank_deficient_populations_are_projectors_with_prescribed_distance(fraction, randomized):
    d, target = 32, 5.
    pair = build_fixed_reference_pair("rank_deficient", d, rank_fraction=fraction,
        target_w2_sq=target, randomize_angles=randomized,
        population_seed=123 if randomized else None)
    rank = int(d * fraction)
    for covariance in (pair.A, pair.B):
        np.testing.assert_allclose(covariance @ covariance, covariance, atol=2e-15)
        eigenvalues = np.linalg.eigvalsh(covariance)
        assert np.count_nonzero(eigenvalues > 1e-10) == rank
        assert eigenvalues[0] >= -1e-14
    # Fidelity of two projectors is the sum of singular values of their overlap.
    singular_values = np.linalg.svd(pair.A @ pair.B, compute_uv=False)
    assert 2 * rank - 2 * singular_values.sum() == pytest.approx(target, abs=1e-12)
    assert gaussian_w2_sq_truth(None, pair.A, None, pair.B)["w2_sq_true"] == pytest.approx(target, abs=1e-10)


@pytest.mark.parametrize("kappa", [100., 10000., 1000000.])
@pytest.mark.parametrize("randomized", [False, True])
def test_ill_conditioned_spectra_and_analytic_distance(kappa, randomized):
    d, target = 32, 5.
    pair = build_fixed_reference_pair("ill_conditioned", d, condition_number=kappa,
        target_w2_sq=target, randomize_angles=randomized,
        population_seed=123 if randomized else None)
    expected = np.r_[np.full(d // 2, 1 / kappa), np.ones(d // 2)]
    for covariance in (pair.A, pair.B):
        np.testing.assert_allclose(np.linalg.eigvalsh(covariance), expected, atol=2e-15)
    assert np.linalg.norm(pair.A @ pair.B - pair.B @ pair.A) > 0
    assert gaussian_w2_sq_truth(None, pair.A, None, pair.B)["w2_sq_true"] == pytest.approx(target, abs=1e-8)


@pytest.mark.parametrize("family, parameters", [
    ("rank_deficient", {"rank_fraction": .5}),
    ("ill_conditioned", {"condition_number": 1e4}),
])
def test_population_profiles_change_across_seeds_but_preserve_truth(family, parameters):
    pairs = [build_fixed_reference_pair(family, 32, target_w2_sq=5,
        randomize_angles=True, population_seed=seed, **parameters) for seed in (7, 8, 7)]
    np.testing.assert_array_equal(pairs[0].B, pairs[2].B)
    assert pairs[0].metadata == pairs[2].metadata
    assert not np.allclose(pairs[0].B, pairs[1].B)
    for pair in pairs:
        profile = pair.metadata['distance_contribution_profile']
        assert sum(profile) == pytest.approx(5.)
        assert np.ptp(profile) > 0


@pytest.mark.parametrize("scale", [.5, 1.])
def test_isotropic_distance_and_scale(scale):
    pair = build_fixed_reference_pair("isotropic", 32, target_w2_sq=5, generated_scale=scale)
    np.testing.assert_array_equal(pair.B, scale * np.eye(32))
    assert 32 * (np.sqrt(pair.A[0, 0]) - np.sqrt(scale)) ** 2 == pytest.approx(5.)
    assert pair.metadata['family'] == 'isotropic'


@pytest.mark.parametrize("family", ["haar_spectral_kappa_100", "haar_log_uniform__near"])
def test_only_configured_random_family_names_are_supported(family):
    with pytest.raises(ValueError, match="unsupported covariance family"):
        generate_scaling_covariance_pair(family, 8, seed=7)


def test_misspelled_parameters_and_infeasible_stress_populations_are_rejected():
    with pytest.raises(ValueError, match="unknown configuration keys"):
        generate_scaling_covariance_pair('haar_log_uniform', 8, config={'lamda_min': .2})
    with pytest.raises(ValueError, match="infeasible"):
        build_fixed_reference_pair('rank_deficient', 8, rank_fraction=.25, target_w2_sq=5)
    with pytest.raises(ValueError, match="does not accept"):
        build_fixed_reference_pair('ill_conditioned', 8, rank_fraction=.5)
    with pytest.raises(ValueError, match="unsupported fixed-reference family"):
        build_fixed_reference_pair('rank_050', 8)

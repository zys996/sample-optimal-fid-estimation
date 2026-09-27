"""Check extrapolation identities, archived estimates, and invalid designs."""
import csv
import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from gaussian_w2._core.fid_infinity import (
    CHEBYSHEV_INVERSE_N,
    ORDINARY_OLS,
    UNIFORM_INVERSE_N,
    UNIFORM_N,
    VARIANCE_AWARE,
    apply_extrapolation_weights,
    extrapolation_sample_sizes,
    fid_infinity_sample_sizes,
    fit_fid_infinity,
    fit_inverse_sample_polynomial,
    inverse_sample_extrapolation_weights,
)


@pytest.mark.parametrize(
    "schedule", [UNIFORM_N, UNIFORM_INVERSE_N, CHEBYSHEV_INVERSE_N]
)
def test_named_sample_schedules_include_unique_endpoints(schedule):
    sizes = extrapolation_sample_sizes(
        50_000,
        minimum_samples=5_000,
        num_points=15,
        schedule=schedule,
    )
    assert sizes[0] == 5_000
    assert sizes[-1] == 50_000
    assert len(sizes) == len(set(sizes)) == 15
    assert list(sizes) == sorted(sizes)


def test_uniform_n_named_schedule_preserves_fid_infinity_grid():
    assert extrapolation_sample_sizes(50_000) == fid_infinity_sample_sizes(50_000)


@pytest.mark.parametrize(
    ("maximum", "expected"),
    [
        (
            20_000,
            (
                5_000,
                6_071,
                7_142,
                8_214,
                9_285,
                10_357,
                11_428,
                12_500,
                13_571,
                14_642,
                15_714,
                16_785,
                17_857,
                18_928,
                20_000,
            ),
        ),
        (40_000, tuple(range(5_000, 40_001, 2_500))),
        (
            80_000,
            (
                5_000,
                10_357,
                15_714,
                21_071,
                26_428,
                31_785,
                37_142,
                42_500,
                47_857,
                53_214,
                58_571,
                63_928,
                69_285,
                74_642,
                80_000,
            ),
        ),
    ],
)
def test_reference_sample_size_grids(maximum, expected):
    assert fid_infinity_sample_sizes(maximum) == expected


@pytest.mark.parametrize("weighting", [ORDINARY_OLS, VARIANCE_AWARE])
@pytest.mark.parametrize("order", [1, 2, 3, 6])
def test_exact_polynomial_bias_is_removed(weighting, order):
    sizes = fid_infinity_sample_sizes(100_000, minimum_samples=30_000, num_points=20)
    inverse = 100_000.0 / np.asarray(sizes)
    coefficients = np.asarray([2.75, -.8, .19, -.035, .006, -.0009, .00012])[:order + 1]
    values = np.polynomial.polynomial.polyval(inverse, coefficients)
    fitted = fit_inverse_sample_polynomial(sizes, values, order=order, weighting=weighting)
    assert fitted["intercept"] == pytest.approx(coefficients[0], abs=3e-12, rel=0)


@pytest.mark.parametrize("weighting", [ORDINARY_OLS, VARIANCE_AWARE])
def test_intercept_equals_direct_weighted_regression(weighting):
    sizes = np.asarray(fid_infinity_sample_sizes(100_000, minimum_samples=30_000, num_points=20))
    values = np.random.default_rng(9917).normal(size=sizes.size)
    relative_sizes = sizes / np.max(sizes)
    design = np.polynomial.polynomial.polyvander(1 / relative_sizes, 4)
    scale = np.ones_like(relative_sizes) if weighting == ORDINARY_OLS else np.sqrt(relative_sizes)
    coefficients, _, rank, _ = np.linalg.lstsq(design * scale[:, None], values * scale, rcond=None)
    fitted = fit_inverse_sample_polynomial(sizes, values, order=4, weighting=weighting)
    assert rank == 5
    assert fitted["intercept"] == pytest.approx(coefficients[0], abs=2e-11, rel=2e-11)


@pytest.mark.parametrize("order", [1, 3, 6])
def test_weights_satisfy_bias_constraints_and_va_objective(order):
    sizes = np.asarray(fid_infinity_sample_sizes(100_000, minimum_samples=30_000, num_points=20))
    power_design = np.polynomial.polynomial.polyvander(np.max(sizes) / sizes, order)
    target = np.r_[1., np.zeros(order)]
    weights = {}
    for weighting in (ORDINARY_OLS, VARIANCE_AWARE):
        weights[weighting] = np.asarray(inverse_sample_extrapolation_weights(sizes, order=order, weighting=weighting)["weights"])
        assert power_design.T @ weights[weighting] == pytest.approx(target, abs=3e-10, rel=0)
    relative_sizes = sizes / np.max(sizes)
    ols_objective = np.sum(weights[ORDINARY_OLS]**2 / relative_sizes)
    va_objective = np.sum(weights[VARIANCE_AWARE]**2 / relative_sizes)
    assert va_objective <= ols_objective


@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_ols_and_va_coincide_with_order_plus_one_nodes(order):
    sizes = extrapolation_sample_sizes(50_000, num_points=order + 1, schedule=UNIFORM_INVERSE_N)
    ols = inverse_sample_extrapolation_weights(sizes, order=order)["weights"]
    va = inverse_sample_extrapolation_weights(sizes, order=order, weighting=VARIANCE_AWARE)["weights"]
    assert ols == pytest.approx(va, abs=2e-13, rel=2e-13)


def test_linear_baseline_keeps_negative_raw_intercepts():
    sizes = fid_infinity_sample_sizes(20_000)
    values = [-.12 - 8 / n for n in sizes]
    fitted = fit_fid_infinity(sizes, values)
    assert fitted["intercept"] == pytest.approx(-.12, abs=1e-12)
    assert set(fitted) == {"intercept", "extrapolation_weights"}


def test_compensated_application_keeps_small_terms_under_cancellation():
    assert apply_extrapolation_weights([1e16, 1., -1e16], [1., 1., 1.]) == 1.


def test_paper_nodes_reproduce_archived_development_estimates():
    data = Path(__file__).resolve().parents[1] / "reproduce" / "data"
    with gzip.open(data / "development_nodes.csv.gz", "rt") as handle:
        records = list(csv.DictReader(handle))
    first = records[0]
    key = first["case_id"], first["trial_id"]
    values = {int(row["sample_size"]): float(row["estimate"]) for row in records
              if (row["case_id"], row["trial_id"]) == key}
    specs = {row["configuration_id"]: row for row in json.loads((data / "development_plan.json").read_text())["configurations"]}
    checked = set()
    with gzip.open(data / "development_trials.csv.gz", "rt") as handle:
        for row in csv.DictReader(handle):
            if (row["case_id"], row["trial_id"]) != key:
                continue
            spec = specs.get(row["configuration_id"])
            if not spec or not spec.get("order"):
                continue
            sizes = spec["sample_sizes"]
            fitted = fit_inverse_sample_polynomial(sizes, [values[n] for n in sizes],
                                                  order=spec["order"], weighting=spec["weighting"])
            assert fitted["intercept"] == pytest.approx(float(row["estimate"]), abs=1e-8, rel=0)
            checked.add(spec["weighting"])
    assert checked == {ORDINARY_OLS, VARIANCE_AWARE}


@pytest.mark.parametrize(
    ("sizes", "values", "order"),
    [
        ([50_000, 75_000, 100_000], [1., 2., 3.], 3),
        ([50_000, 50_000, 100_000], [1., 2., 3.], 2),
        ([50_000, 75_000, 100_000], [1., 2.], 2),
    ],
)
def test_invalid_designs_and_misaligned_observations_are_rejected(sizes, values, order):
    with pytest.raises(ValueError):
        fit_inverse_sample_polynomial(sizes, values, order=order)


def test_numerically_rank_deficient_design_is_rejected():
    with pytest.raises(ValueError, match="rank deficient"):
        inverse_sample_extrapolation_weights([1, 10**12, 2*10**12, 3*10**12], order=3)


def test_unknown_weighting_is_rejected():
    with pytest.raises(ValueError, match="weighting"):
        inverse_sample_extrapolation_weights([10, 20], order=1, weighting="unknown")


@pytest.mark.parametrize("weights", [[1.], [[1., 2.]]])
def test_weight_shapes_cannot_broadcast_or_truncate(weights):
    with pytest.raises(ValueError):
        apply_extrapolation_weights([1., 2.], weights)


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_nonfinite_extrapolation_is_not_returned(value):
    with pytest.raises(FloatingPointError, match="non-finite output"):
        fit_inverse_sample_polynomial([50_000, 75_000, 100_000], [1., value, 3.], order=2)
    with pytest.raises(FloatingPointError, match="non-finite output"):
        apply_extrapolation_weights([1., 2.], [1., value])


def test_finite_values_that_overflow_cannot_produce_a_valid_estimate():
    with pytest.raises((FloatingPointError, OverflowError)):
        apply_extrapolation_weights([1e308, 1e308], [2., 2.])

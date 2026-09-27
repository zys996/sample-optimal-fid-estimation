import math

import pytest

from gaussian_w2.evaluation import across_cases, trial_metrics


def test_trial_sd_and_rmse_have_different_denominators():
    metrics = trial_metrics([1.0, 3.0, 5.0], target=2.0)
    assert metrics["mean_estimate"] == 3.0
    assert metrics["signed_error"] == metrics["center_error"] == 1.0
    assert metrics["sd"] == 2.0
    assert metrics["rmse"] == pytest.approx(math.sqrt(11.0 / 3.0))
    assert metrics["median_absolute_error"] == 1.0


def test_center_error_is_not_mean_absolute_error():
    assert trial_metrics([-2.0, 2.0], target=0.0)["center_error"] == 0.0


def test_cross_case_sd_is_not_standard_error():
    result = across_cases([1.0, 3.0, 5.0])
    assert result == {"n_cases": 3, "mean": 3.0, "sd": 2.0}


@pytest.mark.parametrize("values", [[], [1.0, float("nan")], [float("inf")]])
def test_failures_are_not_silently_removed(values):
    with pytest.raises(ValueError):
        trial_metrics(values, target=0.0)


def test_one_observation_does_not_estimate_variability():
    assert math.isnan(across_cases([2.0])["sd"])


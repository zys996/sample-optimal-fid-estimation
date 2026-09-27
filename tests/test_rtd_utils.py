"""Paper degree rule and disjoint regrouping used by one-sided RTD."""

import math

import numpy as np
import pytest

from gaussian_w2._core.rtd_utils import choose_taylor_degree, recombine_atomic_blocks


@pytest.mark.parametrize("degree", range(2, 7))
def test_degree_recombination_uses_every_atom_exactly_once(degree):
    groups = recombine_atomic_blocks(6, degree)
    assert np.concatenate(groups).tolist() == list(range(6))
    assert max(map(len, groups)) - min(map(len, groups)) <= 1


def test_paper_degree_formula_and_explicit_caps():
    epsilon = min(1.0, math.sqrt(5000 / 20000))
    expected = max(2, math.ceil(math.log(5000 / epsilon) / math.log(4)))
    requested, used, reason = choose_taylor_degree(5000, 20000)
    assert requested == used == expected
    assert reason == "none"
    assert choose_taylor_degree(5000, 20000, degree_cap=4) == (
        expected, 4, "runtime_dimension_cap"
    )
    assert choose_taylor_degree(20, 10, requested_degree=8) == (
        8, 5, "sample_feasibility_cap"
    )


def test_infeasible_pool_is_explicit():
    assert choose_taylor_degree(4, 3, requested_degree=3) == (
        3, 0, "insufficient_correction_samples"
    )


@pytest.mark.parametrize("kwargs", [{"rho_design": 1}, {"C_T": 0}, {"requested_degree": 1}, {"degree_cap": 1}])
def test_invalid_degree_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        choose_taylor_degree(4, 40, **kwargs)

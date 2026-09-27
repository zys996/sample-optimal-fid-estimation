"""Paper-path estimates and failure states recorded before simplification."""
import json
from pathlib import Path

import numpy as np
import pytest

from gaussian_w2.estimators import make_fixed_reference_from_statistics, rtd_fixed_reference


CASES = json.loads((Path(__file__).parent / 'fixtures/rtd_before_simplification.json').read_text())['cases']


@pytest.mark.parametrize('case', CASES, ids=[f"{c['spec']['label']}-zero={c['spec']['known_zero_mean']}" for c in CASES])
@pytest.mark.parametrize('chunk_size', [None, 7])
def test_one_sided_estimates_match_before_simplification(case, chunk_size):
    spec, expected = case['spec'], case['expected']
    d, known = spec['dimension'], spec['known_zero_mean']
    rng = np.random.default_rng(spec['seed'])
    a = rng.normal(size=(d, spec['reference_rank'])) / np.sqrt(d)
    b = rng.normal(size=(d, spec['generated_rank'])) / np.sqrt(d)
    reference_mean = np.zeros(d) if known else rng.normal(size=d) * 0.2
    generated_mean = np.zeros(d) if known else rng.normal(size=d) * 0.2
    samples = rng.normal(size=(spec['n'], spec['generated_rank'])) @ b.T + generated_mean
    reference = make_fixed_reference_from_statistics(
        reference_mean, a @ a.T, covariance_normalization='1/N' if known else '1/(N-1)')
    actual = rtd_fixed_reference(reference, samples, m0=spec['m0'], L=spec['L'],
                                 known_zero_mean=known, stream_chunk_size=chunk_size)
    assert actual['status'] == expected['status']
    for key in ('m0', 'L_used', 'n_correction'):
        assert actual[key] == expected[key]
    if expected['status'] == 'ok':
        # Historical floating-point outputs are compared at the agreed absolute
        # tolerance; independent identities and support tests remain strict.
        assert actual['estimate_w2_sq_raw'] == pytest.approx(expected['estimate_w2_sq_raw'], rel=0, abs=1e-6)
    else:
        assert not np.isfinite(actual['estimate_w2_sq_raw'])
        assert actual['error_message']

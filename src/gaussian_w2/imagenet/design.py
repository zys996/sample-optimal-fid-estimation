"""Study configurations and reproducible row sampling, independent of GPU code."""
from __future__ import annotations
import hashlib
import json
import re
import numpy as np
from gaussian_w2.estimators.extrapolation import extrapolation_sample_sizes, inverse_sample_extrapolation_weights


def stable_seed(master, generator, phase, repetition, n=0):
    """ImageNet seed: first 16 hex digits of canonical JSON SHA-256."""
    token = json.dumps([master, generator, phase, repetition, n], sort_keys=True, separators=(',', ':'), allow_nan=False)
    return int(hashlib.sha256(token.encode()).hexdigest()[:16], 16)


def outer_seed(master, generator, trial, development):
    code = int.from_bytes(hashlib.sha256(generator.encode()).digest()[:4], 'little')
    entropy = [master, code, trial] + ([0xAB1A] if development else [])
    return int(np.random.SeedSequence(entropy).generate_state(1, dtype=np.uint32)[0])


def outer_indices(config, case, trial, maximum):
    s = config['sampling']
    development = s['outer_mode'] == 'development'
    start, stop = s['development_range'] if development else s['heldout_blocks'][trial]
    seed = outer_seed(case['master_seed'], case['generator'], trial, development)
    return start + np.random.default_rng(seed).permutation(stop - start)[:maximum], seed, [start, stop]


def node_indices(master, generator, phase, trial, n, maximum):
    seed = stable_seed(master, generator, phase, trial, n)
    indices = None if n == maximum else np.random.default_rng(seed).permutation(maximum)[:n]
    return indices, seed


def validate_config(config):
    """Validate inputs and sampling; specifications validates each estimator grid."""
    if config['schema'] != 'imagenet-feature-study-v1' or config['backend'] not in ('numpy', 'cupy'):
        raise ValueError('unsupported schema or backend')
    s = config['sampling']
    if s['outer_mode'] not in ('development', 'heldout') or not s['num_trials'] > 0:
        raise ValueError('invalid sampling mode or trials')
    ranges = [s['development_range'], *s['heldout_blocks']]
    if any(not 0 <= start < stop for start, stop in ranges):
        raise ValueError('invalid half-open sampling range')
    if any(a[1] > b[0] for a, b in zip(ranges, ranges[1:])):
        raise ValueError('development and heldout blocks must be ordered and disjoint')
    if s['outer_mode'] == 'heldout' and len(s['heldout_blocks']) != s['num_trials']:
        raise ValueError('one heldout block is required per trial')
    available = s['development_range'][1] - s['development_range'][0] if s['outer_mode'] == 'development' else min(b-a for a,b in s['heldout_blocks'])
    budgets = s['sample_budgets']
    if not budgets or len(set(budgets)) != len(budgets) or not all(0 < s['minimum_samples'] < n <= available for n in budgets):
        raise ValueError('invalid or duplicate sample budgets')
    if not 0 < config['rtd']['m0_numerator'] < config['rtd']['m0_denominator']:
        raise ValueError('RTD pilot fraction must be between zero and one')
    if config['proxies']['aggregation'] != 'median' or not config['proxies']['repetitions'] > 0:
        raise ValueError('proxies require positive repetitions and median aggregation')
    ids = [c['case_id'] for c in config['cases']]
    if not ids or len(set(ids)) != len(ids) or any(not re.fullmatch(r'[A-Za-z0-9_.-]+', x) for x in ids):
        raise ValueError('invalid or duplicate case IDs')
    for c in config['cases']:
        if not c['dimension'] > 0 or c['reference_count'] < 2 or c['pool_size'] < max(stop for _, stop in ranges):
            raise ValueError('invalid feature dimension, reference count, or pool coverage')
    for g in {c['generator'] for c in config['cases']}:
        if len({c['master_seed'] for c in config['cases'] if c['generator'] == g}) != 1:
            raise ValueError('embeddings of a generator must share the same master seed')
    tolerances = [*config['numerics'].values(), config['rtd']['imag_tol']]
    if not all(np.isfinite(v) and v >= 0 for v in tolerances):
        raise ValueError('numerical tolerances must be finite and nonnegative')
    return config


def specifications(config, embedding, maximum):
    """Resolve every label to explicit nodes and intercept weights; share nodes at runtime."""
    result = []
    for spec in config['estimators'][embedding]:
        r = dict(spec)
        if r['kind'] == 'empirical':
            r.update(sample_sizes=[maximum], weights=[1.0], order=0, weighting='none')
        elif r['kind'] == 'rtd':
            r.update(sample_sizes=[], weights=[], order=None, weighting='none')
        elif r['kind'] == 'extrapolation':
            sizes = list(extrapolation_sample_sizes(maximum, minimum_samples=config['sampling']['minimum_samples'],
                         num_points=r['num_points'], schedule=r['sample_schedule']))
            weights = inverse_sample_extrapolation_weights(sizes, order=r['order'], weighting=r['weighting'])
            r.update(sample_sizes=sizes, weights=weights['weights'])
        else:
            raise ValueError(f"unknown estimator kind: {r['kind']}")
        result.append(r)
    if len({r['configuration_id'] for r in result}) != len(result):
        raise ValueError('duplicate configuration IDs')
    return result

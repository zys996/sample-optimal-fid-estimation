"""Configuration and original deterministic Gaussian seed derivation."""
from __future__ import annotations
import copy
import hashlib
import math
from pathlib import Path
import yaml
from gaussian_w2.estimators import choose_taylor_degree, extrapolation_sample_sizes


def stable_seed(master_seed, *components):
    """Original SHA-256 seed: decimal master, NUL-separated component strings."""
    digest = hashlib.sha256(str(int(master_seed)).encode('ascii'))
    for component in components:
        digest.update(b'\0')
        digest.update(str(component).encode('utf-8'))
    return int.from_bytes(digest.digest()[:4], byteorder='big', signed=False)


def load_config(value):
    """Load a custom YAML or mapping and validate the scientific design."""
    config = copy.deepcopy(value) if isinstance(value, dict) else yaml.safe_load(Path(value).read_text(encoding='utf-8'))
    backend = config.setdefault('backend', 'numpy')
    if backend not in ('numpy', 'cupy'):
        raise ValueError('backend must be numpy or cupy')
    config.setdefault('master_seed', 20260913)
    e = config['experiment']
    for key in ('dimensions', 'sample_budgets'):
        values = e[key]
        if not values or not all(n > 0 for n in values) or values != sorted(set(values)):
            raise ValueError(f'{key} must be positive, sorted, and unique')
    if not e.setdefault('known_zero_mean', True) or not e.setdefault('nested_sample_prefixes', True):
        raise ValueError('Gaussian suites use known zero means and nested sample prefixes')
    numerator = e.setdefault('m0_fraction_numerator', 4)
    denominator = e.setdefault('m0_fraction_denominator', 5)
    if not 0 < numerator < denominator:
        raise ValueError('pilot fraction must lie strictly between zero and one')
    f = config.setdefault('fid_infinity', {})
    f.setdefault('minimum_samples', 5000)
    f.setdefault('num_points', 15)
    f.setdefault('sample_schedule', 'uniform_n')
    f.setdefault('regression_orders', [1, 2, 3])
    f.setdefault('regression_weightings', ['ordinary_ols'])
    for key in ('regression_orders', 'regression_weightings'):
        if not f[key] or len(set(f[key])) != len(f[key]):
            raise ValueError(f'{key} must be nonempty and unique')
    if any(w not in ('ordinary_ols', 'variance_aware') for w in f['regression_weightings']):
        raise ValueError('unsupported extrapolation weighting')
    if not all(0 < o < f['num_points'] for o in f['regression_orders']):
        raise ValueError('orders must be positive and below num_points')
    specs = {method_name(w, o): (w, o) for w in f['regression_weightings'] for o in f['regression_orders']}
    expected = ['empirical_plugin', *specs, 'general_adaptive']
    e.setdefault('methods', expected)
    methods = e['methods']
    if not methods or len(set(methods)) != len(methods) or set(methods) - set(expected):
        raise ValueError('methods must be nonempty, unique, and belong to the estimator grid')
    g = config.setdefault('general_order', {})
    for k, v in [('rho_design', .25), ('C_T', 1.), ('degree_cap', None), ('compensated_sum', True)]:
        g.setdefault(k, v)
    for n in e['sample_budgets']:
        extrapolation_sample_sizes(n, minimum_samples=f['minimum_samples'],
                                   num_points=f['num_points'], schedule=f['sample_schedule'])
        for d in e['dimensions']:
            _, used, _ = choose_taylor_degree(d, n - n*numerator//denominator,
                                             rho_design=g['rho_design'], C_T=g['C_T'],
                                             adaptive_degree=True, degree_cap=g['degree_cap'])
            if used == 0:
                raise ValueError('pilot fraction leaves insufficient RTD correction observations')
    cases = config['covariance_cases']
    if not cases or len({c['case_label'] for c in cases}) != len(cases):
        raise ValueError('covariance_cases must have unique case_label values')
    for c in cases:
        if c['family'] not in ('isotropic', 'rank_deficient', 'ill_conditioned', 'haar_log_uniform', 'spiked_bulk', 'rotated_toeplitz'):
            raise ValueError(f"unsupported family {c['family']}")
        if not c['num_instances'] > 0 or not c['num_trials'] > 0:
            raise ValueError('case and trial counts must be positive')
        c.setdefault('parameters', {})
    config.setdefault('numerics', {}).setdefault('eig_rtol', 1e-10)
    config['numerics'].setdefault('eig_atol', 1e-12)
    config.setdefault('gpu', {}).setdefault('device', 0)
    config['gpu'].setdefault('sample_batch_size', 5000)
    config['gpu'].setdefault('synchronize_timing', True)
    if not config['gpu']['sample_batch_size'] > 0:
        raise ValueError('sample_batch_size must be positive')
    if not all(math.isfinite(v) and v >= 0 for v in config['numerics'].values()):
        raise ValueError('eigenvalue tolerances must be finite and nonnegative')
    config.setdefault('output', {}).setdefault('directory', 'outputs/gaussian')
    return config


def method_name(weighting, order):
    label = 'ols' if weighting == 'ordinary_ols' else 'variance_aware'
    return f'fid_infinity_{label}_order{order}'


def tasks(config):
    for case in config['covariance_cases']:
        for dimension in config['experiment']['dimensions']:
            for instance in range(case['num_instances']):
                yield {**case, 'dimension': dimension, 'instance_id': instance}

"""One feature-based runner for development ablations and heldout budget studies.

Each n gets its own permutation within the common outer pool. All estimator
labels consume the same in-memory n-node, including the empirical endpoint.
Each case runs from start to finish and saves its results for separate analysis.
"""
from __future__ import annotations
import argparse
import csv
import importlib.metadata
import platform
import math
import os
from pathlib import Path
import time
import numpy as np
from .design import node_indices, outer_indices, specifications, stable_seed, validate_config
from .io import atomic_json, binding, load_config, load_moments, open_pool, read_json
from gaussian_w2.estimators.extrapolation import extrapolation_sample_sizes, fit_inverse_sample_polynomial


class Backend:
    def __init__(self, config, case):
        mean, cov, count, metadata = load_moments(case['reference'])
        if count != case['reference_count'] or len(mean) != case['dimension']:
            raise ValueError('reference dimension/sample count differs from config')
        signature = case['extractor_signature']
        if signature is not None and metadata.get('extractor_signature') != signature:
            raise ValueError('reference extractor signature differs from config')
        self.options = {'rtol': config['numerics']['eig_rtol'], 'atol': config['numerics']['eig_atol']}
        self.gpu = config['backend'] == 'cupy'
        if self.gpu:
            import cupy as cp
            from gaussian_w2.estimators.gpu import prepare_gpu_fixed_reference_from_statistics, gpu_fixed_reference_empirical, gpu_rtd_fixed_reference
            self.xp = cp
            make, self.empirical_fn, self.rtd_fn = prepare_gpu_fixed_reference_from_statistics, gpu_fixed_reference_empirical, gpu_rtd_fixed_reference
        else:
            from gaussian_w2.estimators import make_fixed_reference_from_statistics, fixed_reference_empirical_fid, rtd_fixed_reference
            self.xp = np
            make, self.empirical_fn, self.rtd_fn = make_fixed_reference_from_statistics, fixed_reference_empirical_fid, rtd_fixed_reference
        self.reference = make(mean, cov, sample_count=count, covariance_normalization='1/(N-1)', **self.options)

    def upload(self, x):
        return self.xp.asarray(x, dtype=self.xp.float64, order='C')

    def subset(self, x, indices):
        return x[self.xp.asarray(indices, dtype=self.xp.int64)]

    def call(self, x, rtd=None):
        started = time.perf_counter()
        try:
            if rtd is None:
                value = self.empirical_fn(self.reference, x, **self.options)
            else:
                options = {k: rtd[k] for k in ('L', 'rho_design', 'C_T', 'degree_cap', 'compensated_sum', 'stream_chunk_size')}
                if not self.gpu:
                    options['imag_tol'] = rtd['imag_tol']
                value = self.rtd_fn(self.reference, x, m0=len(x)*rtd['m0_numerator']//rtd['m0_denominator'], **options, **self.options)
            estimate = float(value['estimate_w2_sq_raw'])
            if value.get('status') != 'ok' or not math.isfinite(estimate):
                raise FloatingPointError(f"{value.get('status')}: {value.get('error_message', 'nonfinite estimate')}")
            seconds = float(value.get('gpu_method_seconds', time.perf_counter() - started))
            record = {'status': 'ok', 'estimate': estimate, 'runtime_seconds': seconds}
            for k in ('L_used', 'm0', 'n_correction', 'mean_w2_sq_estimate', 'covariance_w2_sq_estimate'):
                if k in value and value[k] is not None and math.isfinite(float(value[k])):
                    record[k] = value[k]
            return record
        except Exception as error:
            if rtd is None:
                raise
            return {'status': 'failed', 'estimate': None, 'runtime_seconds': None,
                    'failure_type': type(error).__name__, 'error_message': str(error) or repr(error),
                    'm0': len(x)*rtd['m0_numerator']//rtd['m0_denominator'],
                    'n_correction': len(x)-1-len(x)*rtd['m0_numerator']//rtd['m0_denominator']}

    def release(self):
        if self.gpu:
            self.xp.get_default_memory_pool().free_all_blocks()


def case_binding(plan, case):
    """Store the case and measurement settings directly in saved records."""
    config = plan['config']
    return {'schema': plan['schema'], 'case': case, 'inputs': plan['inputs'][case['case_id']],
            'settings': {k: config[k] for k in ('backend', 'numerics', 'rtd', 'sampling', 'proxies')}}


def numerical_environment(backend):
    environment = {'python': platform.python_version(), 'platform': platform.system(),
                   'machine': platform.machine(), 'numpy': np.__version__,
                   'scipy': importlib.metadata.version('scipy')}
    if backend == 'cupy':
        import cupy as cp
        name = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)['name']
        environment.update(cupy=cp.__version__, cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),
                           cuda_driver=cp.cuda.runtime.driverGetVersion(),
                           gpu=name.decode('utf-8') if isinstance(name, bytes) else name)
    return environment


def validate_terminal(record, *, allow_failure=True):
    if record.get('status') == 'ok':
        if not math.isfinite(float(record['estimate'])) or not math.isfinite(float(record['runtime_seconds'])) or record['runtime_seconds'] < 0:
            raise ValueError('invalid finite estimate/runtime in result')
    elif not (allow_failure and record.get('status') == 'failed' and record.get('estimate') is None
              and record.get('runtime_seconds') is None and record.get('failure_type') and record.get('error_message')):
        raise ValueError('invalid terminal failure record')


def validate_trial(record, identity, specs):
    if record.get('binding') != identity:
        raise ValueError('trial result binding differs')
    if record.get('specifications') != specs:
        raise ValueError('trial estimator specifications changed')
    rows = record.get('rows', [])
    expected = {s['configuration_id']: s for s in specs}
    if len(rows) != len(expected) or {r['configuration_id'] for r in rows} != set(expected):
        raise ValueError('incomplete or duplicate trial estimator coverage')
    for row in rows:
        validate_terminal(row, allow_failure=expected[row['configuration_id']]['kind'] == 'rtd')
        if not math.isfinite(float(row['fit_seconds'])) or row['fit_seconds'] < 0:
            raise ValueError('invalid fit time')


def compute_nodes(backend, data, sizes, case, phase, trial, endpoint=None):
    """Compute each requested node once in memory for all methods in this trial."""
    points = {}
    for n in sorted(set(sizes)):
        indices, seed = node_indices(case['master_seed'], case['generator'], phase, trial, n, len(data))
        value = (endpoint if n == len(data) and endpoint is not None
                 else backend.call(data if indices is None else backend.subset(data, indices)))
        validate_terminal(value, allow_failure=False)
        points[n] = {**value, 'n': n, 'sampling_seed': seed, 'pool_size': len(data)}
    return points


def validate_proxy(record, config, identity):
    if record.get('binding') != identity:
        raise ValueError('proxy input binding changed')
    fid_runs, rtd_runs = record['fid_infinity_runs'], record['rtd_runs']
    repetitions = list(range(config['proxies']['repetitions']))
    if ([r['repetition'] for r in fid_runs] != repetitions or
            [r['binding']['repetition'] for r in rtd_runs] != repetitions):
        raise ValueError('incomplete or duplicate proxy repetitions')
    if not all(math.isfinite(float(r['estimate'])) for r in fid_runs):
        raise ValueError('nonfinite FID-infinity proxy estimate')
    for row in rtd_runs:
        validate_terminal(row)
    values = record['proxies']
    if not math.isfinite(float(values['plugin'])):
        raise ValueError('nonfinite empirical proxy estimate')
    fid_center = float(np.median([r['estimate'] for r in fid_runs]))
    rtd_center = (float(np.median([r['estimate'] for r in rtd_runs]))
                  if all(r['status'] == 'ok' for r in rtd_runs) else None)
    if values['fid_infinity'] != fid_center or values['rtd'] != rtd_center:
        raise ValueError('proxy centers differ from their repeated estimates')


def proxy_case(config, case, identity, backend, pool):
    """Compute development proxies, or import the explicitly configured source."""
    source = config['proxies']['source_run']
    if source is not None:
        source_path = Path(source) / 'cases' / case['case_id'] / 'proxies.json'
        saved = read_json(source_path)
        source_plan = read_json(Path(source) / 'manifest.json')
        old_config = source_plan['config']
        old_case = next(c for c in old_config['cases'] if c['case_id'] == case['case_id'])
        expected_proxy_binding = case_binding(source_plan, old_case)
        validate_proxy(saved, old_config, expected_proxy_binding)
        old = source_plan['inputs'][case['case_id']]
        if old != identity['inputs'] or source_plan['config']['rtd'] != config['rtd']:
            raise ValueError('proxy source uses different inputs or RTD settings')
        if old_case != case or old_config['numerics'] != config['numerics'] or old_config['backend'] != config['backend']:
            raise ValueError('proxy source uses different case parameters or numerics')
        if old_config['sampling']['development_range'] != config['sampling']['development_range']:
            raise ValueError('proxy source has a different development range')
        for k in config['proxies']:
            if k != 'source_run' and old_config['proxies'][k] != config['proxies'][k]:
                raise ValueError(f'proxy source differs at {k}')
        return {**saved, 'binding': identity, 'source': str(source_path)}
    p = config['proxies']
    start, stop = config['sampling']['development_range']
    data = backend.upload(pool[start:stop])
    points = compute_nodes(backend, data, [len(data)], case, p['plugin_seed_phase'], 0)
    endpoint = points[len(data)]
    sizes = list(extrapolation_sample_sizes(len(data), minimum_samples=p['minimum_samples'], num_points=p['num_points'], schedule=p['sample_schedule']))
    fid_runs, rtd_runs = [], []
    for repetition in range(p['repetitions']):
        nodes = compute_nodes(backend, data, sizes, case, p['node_seed_phase'], repetition, endpoint)
        fit = fit_inverse_sample_polynomial(sizes, [nodes[n]['estimate'] for n in sizes], order=1, weighting='ordinary_ols')
        fid_runs.append({'repetition': repetition, 'estimate': float(fit['intercept']),
                         'sample_sizes': sizes, 'nodes': list(nodes.values())})
        seed = stable_seed(case['master_seed'], case['generator'], p['rtd_seed_phase'], repetition)
        indices = np.random.default_rng(seed).permutation(len(data))
        rtd_binding = {**identity, 'repetition': repetition, 'sampling_seed': seed, 'pool_size': len(data)}
        record = {**backend.call(backend.subset(data, indices), config['rtd']), 'binding': rtd_binding}
        rtd_runs.append(record)
    values = {'plugin': endpoint['estimate'], 'fid_infinity': float(np.median([r['estimate'] for r in fid_runs])),
              'rtd': float(np.median([r['estimate'] for r in rtd_runs])) if all(r['status']=='ok' for r in rtd_runs) else None}
    result = {'binding': identity, 'proxies': values, 'fid_infinity_runs': fid_runs, 'rtd_runs': rtd_runs,
              'interpretation': 'conditional algorithm variability within one shared development pool'}
    validate_proxy(result, config, identity)
    del data
    backend.release()
    return result


def reuse_trial(config, case, trial, maximum, identity, specs):
    source = next((x for x in config['reuse'] if x['sample_budget'] == maximum), None)
    if source is None:
        return None
    root = Path(source['run'])
    plan = read_json(root/'manifest.json')
    if plan['inputs'][case['case_id']] != identity['inputs'] or plan['config']['rtd'] != config['rtd'] or plan['config']['numerics'] != config['numerics']:
        raise ValueError('reused trial has different features, reference, numerics, or RTD settings')
    path = root/'cases'/case['case_id']/f'n{maximum}'/f'trial_{trial:03d}.json'
    saved = read_json(path)
    source_case = next(c for c in plan['config']['cases'] if c['case_id'] == case['case_id'])
    if source_case != case or plan['config']['backend'] != config['backend']:
        raise ValueError('reused trial has different case or backend settings')
    _, source_seed, source_block = outer_indices(plan['config'], source_case, trial, maximum)
    source_identity = {**case_binding(plan, source_case), 'sampling_seed': source_seed,
                       'block': source_block, 'trial_id': trial, 'sample_budget': maximum}
    source_specs = plan['specifications'][case['embedding']][str(maximum)]
    validate_trial(saved, source_identity, source_specs)
    for k in ('sampling_seed', 'block', 'trial_id', 'sample_budget'):
        if source_identity[k] != identity[k]:
            raise ValueError(f'reused trial sampling differs: {k}')
    old_specs = {r['configuration_id']: r for r in plan['specifications'][case['embedding']][str(maximum)]}
    old_rows = {r['configuration_id']: r for r in saved['rows']}
    rows = []
    for spec in specs:
        if old_specs.get(spec['configuration_id']) != spec:
            raise ValueError('reused trial estimator settings differ')
        rows.append({**old_rows[spec['configuration_id']], 'source_record': str(path)})
    return {'binding': identity, 'rows': rows, 'nodes': saved['nodes'], 'specifications': specs, 'reused_from': str(path), 'numerical_environment': saved['numerical_environment'],
            'source_node_seed_phase': plan['config']['sampling']['node_seed_phase']}


def run(config_path, output, *, case_id=None):
    """Compute complete cases; saved cases are inputs to analysis, never resume state."""
    config = validate_config(load_config(config_path))
    cases = [c for c in config['cases'] if case_id is None or c['case_id'] == case_id]
    if not cases:
        raise ValueError('unknown case ID')
    output = Path(output)
    for case in cases:
        if (output/f"trials.{case['case_id']}.csv").exists():
            raise FileExistsError(f"case {case['case_id']} is already complete; select an unfinished case or a new output directory")
    resolved = {e: {str(n): specifications(config, e, n) for n in config['sampling']['sample_budgets']} for e in config['estimators']}
    output.mkdir(parents=True, exist_ok=True)
    inputs = {}
    pools = {}
    for case in config['cases']:
        pools[case['case_id']] = open_pool(case['features'], (case['pool_size'], case['dimension']))
        inputs[case['case_id']] = {'features': binding(case['features']), 'reference': binding(case['reference'])}
        if case['feature_metadata'] is not None:
            metadata = read_json(case['feature_metadata'])
            if metadata.get('extractor_signature') != case['extractor_signature'] or metadata.get('shape') != [case['pool_size'], case['dimension']]:
                raise ValueError('generated feature extractor signature or shape differs from config')
            inputs[case['case_id']]['feature_metadata'] = binding(case['feature_metadata'])
    environment = numerical_environment(config['backend'])
    manifest = {'schema': 'imagenet-run-v3', 'config': config, 'inputs': inputs,
                'specifications': resolved, 'numerical_environment': environment}
    manifest_path = output/'manifest.json'
    if manifest_path.exists():
        saved = read_json(manifest_path)
        if any(saved.get(k) != manifest[k] for k in ('schema', 'config', 'inputs', 'specifications')):
            raise ValueError('run configuration or inputs differ; choose a new output directory')
    else:
        atomic_json(manifest_path, manifest)
    for case in cases:
        folder = output/'cases'/case['case_id']
        identity = case_binding(manifest, case)
        backend = Backend(config, case)
        pool = pools[case['case_id']]
        proxies = proxy_case(config, case, identity, backend, pool)
        atomic_json(folder/'proxies.json', proxies)
        all_rows = []
        for trial in range(config['sampling']['num_trials']):
            for maximum in config['sampling']['sample_budgets']:
                indices, seed, block = outer_indices(config, case, trial, maximum)
                trial_binding = {**identity, 'trial_id': trial, 'sample_budget': maximum, 'sampling_seed': seed,
                                 'block': block}
                specs = resolved[case['embedding']][str(maximum)]
                record = reuse_trial(config, case, trial, maximum, trial_binding, specs)
                if record is None:
                    data = backend.upload(pool[indices])
                    sizes = sorted({n for spec in specs for n in spec['sample_sizes']})
                    nodes = compute_nodes(backend, data, sizes, case, config['sampling']['node_seed_phase'], trial)
                    rows = []
                    for spec in specs:
                        start = time.perf_counter()
                        if spec['kind'] == 'rtd':
                            row = {**backend.call(data, config['rtd']), 'fit_seconds': 0.0}
                        else:
                            values = [nodes[n]['estimate'] for n in spec['sample_sizes']]
                            value = values[0] if spec['kind'] == 'empirical' else fit_inverse_sample_polynomial(
                                spec['sample_sizes'], values, order=spec['order'], weighting=spec['weighting'])['intercept']
                            elapsed = time.perf_counter() - start
                            row = {'status': 'ok', 'estimate': float(value), 'fit_seconds': elapsed,
                                   'runtime_seconds': elapsed + sum(nodes[n]['runtime_seconds'] for n in spec['sample_sizes'])}
                        rows.append({**row, 'configuration_id': spec['configuration_id']})
                    record = {'binding': trial_binding, 'rows': rows, 'nodes': list(nodes.values()),
                              'specifications': specs, 'numerical_environment': environment}
                    del data
                    backend.release()
                validate_trial(record, trial_binding, specs)
                atomic_json(folder/f'n{maximum}'/f'trial_{trial:03d}.json', record)
                for row in record['rows']:
                    all_rows.append({**row, 'case_id': case['case_id'], 'generator': case['generator'], 'embedding': case['embedding'],
                                     'trial_id': trial, 'sample_budget': maximum, 'sampling_seed': seed,
                                     'block_start': block[0], 'block_stop': block[1], 'proxy_rtd': proxies['proxies']['rtd']})
                print(f"{case['case_id']} trial={trial} N={maximum} complete", flush=True)
        # The CSV is published only when this entire case is complete.
        path = output/f"trials.{case['case_id']}.csv"
        temporary = path.with_suffix('.csv.partial')
        with temporary.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=sorted({k for row in all_rows for k in row}))
            writer.writeheader()
            writer.writerows(all_rows)
        os.replace(temporary, path)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--case-id', help='Run one complete generator/embedding case, for job arrays')
    args = parser.parse_args()
    run(args.config, args.output, case_id=args.case_id)

if __name__ == '__main__':
    main()

"""Run complete isotropic, random, or repeated-stress Gaussian tasks.

Example: python -m experiments.gaussian.run --config configs/examples/gaussian.yaml
Each task computes every trial and saves one result file. Within a trial, all
budgets share a maximum-size sample pool and use the same budget prefixes.
"""
from __future__ import annotations
import argparse
import contextlib
import json
import math
from pathlib import Path
import time
import numpy as np
from gaussian_w2.estimators import (
    make_fixed_reference_from_statistics, empirical_fixed_reference,
    rtd_fixed_reference, plugin_curve, gaussian_w2_sq_truth,
)
from gaussian_w2._core.linalg import psd_eigh
from .config import load_config, method_name, stable_seed, tasks
from .records import atomic_json, make_manifest, read_manifest
from .stress_families import build_fixed_reference_pair
from .random_families import generate_scaling_covariance_pair


def population(task, seed, config, xp):
    params = dict(task['parameters'])
    if task['family'] in ('isotropic', 'rank_deficient', 'ill_conditioned'):
        if params.get('randomize_angles', False):
            if 'population_seed' in params:
                raise ValueError('population_seed is derived from case identity, not configured')
            params['population_seed'] = seed
        pair = build_fixed_reference_pair(task['family'], task['dimension'], xp=xp, **params)
        return pair.A, pair.B, dict(pair.metadata)
    if config['backend'] == 'cupy':
        from .gpu_generation import gpu_generate_scaling_covariance_pair
        a, b = gpu_generate_scaling_covariance_pair(task['family'], task['dimension'], seed=seed,
                                                   config=params, device=config['gpu']['device'])
    else:
        a, b = generate_scaling_covariance_pair(task['family'], task['dimension'], seed=seed, config=params)
    return a, b, {}


def sampling_factor(covariance, config, xp):
    """Cache the population support factor once for all conditional trials."""
    rtol, atol = config['numerics']['eig_rtol'], config['numerics']['eig_atol']
    if config['backend'] == 'cupy':
        from gaussian_w2._core.gpu_linalg import _support_factors
        factors = _support_factors(covariance, rtol=rtol, atol=atol, cp=xp)
        positive = factors.eigenvalues > factors.threshold
        return factors.support_vectors * xp.sqrt(factors.eigenvalues[positive])[None, :]
    values, vectors, threshold = psd_eigh(covariance, rtol=rtol, atol=atol)
    positive = values > threshold
    return vectors[:, positive] * np.sqrt(values[positive])[None, :]


def sample_pool(covariance, budget, seed, config, xp, *, root=None):
    """Threshold numerical support, then draw original batched Gaussian rows.

    GPU uses CuPy RandomState and its original draw order. NumPy uses its local
    Generator; same seed identities, but CPU and CUDA random arrays differ.
    """
    if root is None:
        root = sampling_factor(covariance, config, xp)
    rng = (xp.random.RandomState(seed % 2**32) if config['backend'] == 'cupy'
           else np.random.default_rng(seed))
    rank = root.shape[1]
    samples = xp.empty((budget, covariance.shape[0]), dtype=xp.float64)
    batch = config['gpu']['sample_batch_size']
    for start in range(0, budget, batch):
        stop = min(start + batch, budget)
        samples[start:stop] = rng.standard_normal((stop-start, rank), dtype=xp.float64) @ root.T
    return samples


def prepare_reference(a, config, xp):
    kwargs = dict(sample_count=None, covariance_normalization='1/N',
                  rtol=config['numerics']['eig_rtol'], atol=config['numerics']['eig_atol'])
    if config['backend'] == 'cupy':
        from gaussian_w2.estimators.gpu import prepare_gpu_fixed_reference_from_statistics
        return prepare_gpu_fixed_reference_from_statistics(xp.zeros(len(a)), a, **kwargs,
                    reference_mode='exact_population_covariance', device=config['gpu']['device'])
    return make_fixed_reference_from_statistics(np.zeros(len(a)), a, **kwargs)


def estimate_budget(reference, samples, sampling_seed, config):
    n = len(samples)
    f, e, g = config['fid_infinity'], config['experiment'], config['general_order']
    methods = e['methods']
    numerics = dict(rtol=config['numerics']['eig_rtol'], atol=config['numerics']['eig_atol'])
    use_gpu = config['backend'] == 'cupy'
    if use_gpu:
        from gaussian_w2.estimators.gpu import (gpu_fixed_reference_plugin_curve,
            gpu_rtd_fixed_reference, gpu_fixed_reference_empirical)
        curve_function, rtd_function, empirical_function = (gpu_fixed_reference_plugin_curve,
            gpu_rtd_fixed_reference, gpu_fixed_reference_empirical)
        kwargs = dict(device=config['gpu']['device'], covariance_normalization='1/N',
                      synchronize_timing=config['gpu']['synchronize_timing'])
    else:
        curve_function, rtd_function, empirical_function = plugin_curve, rtd_fixed_reference, empirical_fixed_reference
        kwargs = {}
    results, nodes = {}, {}
    if any(m != 'general_adaptive' for m in methods):
        try:
            curve = curve_function(reference, samples, minimum_samples=f['minimum_samples'],
                num_points=f['num_points'], sample_schedule=f['sample_schedule'],
                regression_orders=tuple(f['regression_orders']),
                regression_weightings=tuple(f['regression_weightings']),
                seed=stable_seed(sampling_seed, 'ols', n), known_zero_mean=True,
                **numerics, **kwargs)
            nodes = {'sample_sizes': curve['sample_sizes'], 'plugin_values': curve['plugin_values'],
                     'seed': stable_seed(sampling_seed, 'ols', n)}
            results['empirical_plugin'] = {'status': 'ok',
                'estimate_w2_sq_raw': curve['endpoint_plugin_w2_sq'],
                'runtime_seconds': curve['point_seconds'][-1]}
            for w, fits in curve['fits_by_weighting'].items():
                for order, fit in fits.items():
                    results[method_name(w, order)] = {'status': 'ok',
                        'estimate_w2_sq_raw': fit['intercept'],
                        'runtime_seconds': curve['curve_seconds']}
        except Exception as error:
            results.update({m: {'status': 'numerical_failure', 'error_message': str(error)}
                            for m in methods if m != 'general_adaptive'})
            # A failed small-node curve must not erase the valid full endpoint.
            if 'empirical_plugin' in methods:
                try:
                    results['empirical_plugin'] = empirical_function(reference, samples,
                        known_zero_mean=True, **numerics, **kwargs)
                except Exception as endpoint_error:
                    results['empirical_plugin']['error_message'] = str(endpoint_error)
    if 'general_adaptive' in methods:
        m0 = n * e['m0_fraction_numerator'] // e['m0_fraction_denominator']
        started = time.perf_counter()
        try:
            result = rtd_function(reference, samples, m0=m0, L=None, known_zero_mean=True,
                                  **g, **numerics, **kwargs)
        except Exception as error:
            result = {'status': 'numerical_failure', 'error_message': str(error)}
        result['runtime_seconds'] = time.perf_counter() - started
        result['m0'] = m0
        results['general_adaptive'] = result
    return {m: results[m] for m in methods}, nodes


def _trial_rows(task, trial_id, seed, covariance_seed, truth, samples, reference, config):
    rows, curves = [], {}
    for budget in config['experiment']['sample_budgets']:
        estimates, nodes = estimate_budget(reference, samples[:budget], seed, config)
        curves[str(budget)] = nodes
        for method, result in estimates.items():
            raw = result.get('estimate_w2_sq_raw')
            if raw is not None and not math.isfinite(raw):
                raw = None
            row = {k: task[k] for k in ('suite', 'family', 'case_label', 'dimension', 'instance_id')}
            row.update(N=budget, trial_id=trial_id, covariance_seed=covariance_seed,
                       sampling_seed=seed, method=method, true_w2_sq=truth,
                       estimate_w2_sq_raw=raw, status=result['status'],
                       error_message=result.get('error_message', ''), backend=config['backend'],
                       signed_error_w2_sq_raw=None if raw is None else raw-truth,
                       abs_error_w2_sq_raw=None if raw is None else abs(raw-truth),
                       m0=result.get('m0'), L_requested=result.get('L_requested'),
                       L_used=result.get('L_used'), n_correction=result.get('n_correction'),
                       runtime_seconds=result.get('runtime_seconds'))
            rows.append(row)
    return rows, curves


def run(config_or_path, output=None, *, task_index=None, overwrite=False):
    """Compute every trial in each selected task, then save its complete result."""
    config = load_config(config_or_path)
    output = Path(output or config['output']['directory']).expanduser().resolve()
    all_tasks = list(tasks(config))
    if task_index is not None and not 0 <= task_index < len(all_tasks):
        raise ValueError('task-index is outside configured tasks')
    if overwrite and task_index is None:
        raise ValueError('--overwrite requires --task-index')
    selected = [(i, task) for i, task in enumerate(all_tasks)
                if task_index is None or i == task_index]
    for index, _ in selected:
        path = output/'records'/f'task_{index:05d}.json'
        if path.exists() and not overwrite:
            raise FileExistsError(f'task result already exists: {path}; use --task-index and --overwrite to rerun it')
    output.mkdir(parents=True, exist_ok=True)
    frozen = output/'config.json'
    if frozen.exists():
        if json.loads(frozen.read_text(encoding='utf-8')) != config:
            raise ValueError('output directory contains a different frozen configuration')
    elif (output/'manifest.json').exists() or any((output/'records').glob('*.json')):
        raise ValueError('Gaussian run is missing its saved configuration; use a new output directory')
    xp, context = np, contextlib.nullcontext()
    if config['backend'] == 'cupy':
        from gaussian_w2.estimators.gpu import require_cupy
        xp = require_cupy()
        context = xp.cuda.Device(config['gpu']['device'])
    if (output/'manifest.json').exists():
        read_manifest(output)
    else:
        if any((output/'records').glob('*.json')):
            raise ValueError('Gaussian results require their saved manifest; use a new output directory')
        if not frozen.exists():
            atomic_json(frozen, config)
        atomic_json(output/'manifest.json', make_manifest(config, xp))
    completed_rows = []
    with context:
        for index, task in selected:
            covariance_seed = stable_seed(config['master_seed'], 'covariance', task['case_label'],
                                           task['dimension'], task['instance_id'])
            trial_records = []
            a, b, metadata = population(task, covariance_seed, config, xp)
            if config['backend'] == 'cupy':
                from gaussian_w2.estimators.gpu import gpu_gaussian_w2_sq_truth
                truth = gpu_gaussian_w2_sq_truth(None, a, None, b,
                    rtol=config['numerics']['eig_rtol'], atol=config['numerics']['eig_atol'],
                    device=config['gpu']['device'])['w2_sq_true']
            else:
                truth = gaussian_w2_sq_truth(None, a, None, b,
                    rtol=config['numerics']['eig_rtol'], atol=config['numerics']['eig_atol'])['w2_sq_true']
            if metadata.get('randomize_angles', False):
                if not math.isclose(truth, metadata['true_w2_sq'], abs_tol=5e-8, rel_tol=5e-9):
                    raise ValueError('analytic population truth disagrees with numerical truth')
                truth = metadata['true_w2_sq']
            reference = prepare_reference(a, config, xp)
            root = sampling_factor(b, config, xp)
            for trial in range(task['num_trials']):
                seed = stable_seed(config['master_seed'], 'sample', task['case_label'],
                                   task['dimension'], task['instance_id'], trial)
                samples = sample_pool(b, max(config['experiment']['sample_budgets']), seed, config, xp, root=root)
                rows, curves = _trial_rows(task, trial, seed, covariance_seed, truth, samples, reference, config)
                trial_records.append({'trial_id': trial, 'sampling_seed': seed,
                                      'curves': curves, 'rows': rows})
                completed_rows.extend(rows)
                print(f"Computed task {index+1}/{len(all_tasks)}, trial {trial+1}/{task['num_trials']}", flush=True)
                del samples
            atomic_json(output/'records'/f'task_{index:05d}.json', {
                'schema': 'gaussian-task-v1', 'config': config, 'task_index': index,
                'task': task, 'covariance_seed': covariance_seed,
                'population_metadata': metadata, 'trials': trial_records})
    return {'tasks': len(selected), 'rows': len(completed_rows),
            'ok': sum(r['status'] == 'ok' for r in completed_rows),
            'failed': sum(r['status'] != 'ok' for r in completed_rows), 'output': str(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, help='YAML experiment configuration')
    parser.add_argument('--output', help='separate output directory; overrides configured location')
    parser.add_argument('--backend', choices=['numpy', 'cupy'], help='override backend before freezing configuration')
    parser.add_argument('--task-index', type=int, help='run one deterministic case/dimension/instance task')
    parser.add_argument('--overwrite', action='store_true', help='replace the complete result of the selected --task-index')
    args = parser.parse_args()
    config = load_config(args.config)
    if args.backend:
        config['backend'] = args.backend
    print(json.dumps(run(config, args.output, task_index=args.task_index, overwrite=args.overwrite), indent=2))


if __name__ == '__main__':
    main()

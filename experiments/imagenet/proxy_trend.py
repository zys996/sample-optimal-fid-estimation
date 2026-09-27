"""Compute plug-in, FID-infinity, and RTD across development-pool budgets."""
from __future__ import annotations

import argparse
import copy
import csv
import os
from pathlib import Path

from gaussian_w2.imagenet.design import validate_config
from gaussian_w2.imagenet.io import atomic_json, binding, load_config, open_pool, read_json
from gaussian_w2.imagenet.runner import Backend, numerical_environment, proxy_case


def load_study(path, inputs_file=None):
    study = read_json(path)
    config = load_config(Path(path).parent / study['study_config'])
    if inputs_file is not None:
        inputs = load_config(inputs_file)
        config['cases'] = [{**case, 'extractor_signature': inputs['extractor_signature']}
                           for case in inputs['cases']]
    config['sampling']['sample_budgets'] = study['sample_budgets']
    config['proxies']['source_run'] = None
    return validate_config(config)


def result_rows(case, maximum, record):
    """Export each repetition; the plotter computes the same proxy medians."""
    common = {k: case[k] for k in ('case_id', 'generator', 'embedding', 'dimension')}
    common.update(sample_budget=maximum, runtime_seconds=None, failure_type='', error_message='')
    endpoint = next(n for n in record['fid_infinity_runs'][0]['nodes'] if n['n'] == maximum)
    yield {**common, 'proxy': 'plugin', 'repetition': 0, 'status': 'ok',
           'estimate': endpoint['estimate'], 'runtime_seconds': endpoint['runtime_seconds']}
    for run in record['fid_infinity_runs']:
        yield {**common, 'proxy': 'fid_infinity', 'repetition': run['repetition'],
               'status': 'ok', 'estimate': run['estimate']}
    for run in record['rtd_runs']:
        yield {**common, 'proxy': 'rtd', 'repetition': run['binding']['repetition'],
               **{k: v for k, v in run.items() if k != 'binding'}}


def run(config_path, output, *, inputs_file=None, case_id=None):
    config = load_study(config_path, inputs_file)
    cases = [c for c in config['cases'] if case_id is None or c['case_id'] == case_id]
    if not cases:
        raise ValueError('unknown case ID')
    output = Path(output)
    for case in cases:
        if (output / f"trials.{case['case_id']}.csv").exists():
            raise FileExistsError(f"case {case['case_id']} is already complete")
    budgets = config['sampling']['sample_budgets']
    environment = numerical_environment(config['backend'])
    manifest = {'schema': 'imagenet-proxy-trend-v1', 'config': config,
                'sample_budgets': budgets, 'case_ids': [c['case_id'] for c in config['cases']],
                'numerical_environment': environment}
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / 'manifest.json'
    if manifest_path.exists():
        saved = read_json(manifest_path)
        if any(saved.get(k) != manifest[k] for k in ('schema', 'config', 'sample_budgets', 'case_ids')):
            raise ValueError('run settings differ; use a new output directory')
    else:
        atomic_json(manifest_path, manifest)
    start = config['sampling']['development_range'][0]
    for case in cases:
        pool = open_pool(case['features'], (case['pool_size'], case['dimension']))
        inputs = {key: binding(case[key]) for key in ('features', 'reference')}
        if case['feature_metadata'] is not None:
            metadata = read_json(case['feature_metadata'])
            if (metadata.get('extractor_signature') != case['extractor_signature']
                    or metadata.get('shape') != [case['pool_size'], case['dimension']]):
                raise ValueError('feature metadata differs from the configured case')
        backend = Backend(config, case)
        rows = []
        for maximum in budgets:
            budget_config = copy.deepcopy(config)
            budget_config['sampling']['development_range'] = [start, start + maximum]
            identity = {'case': case, 'inputs': inputs, 'sample_budget': maximum,
                        'development_range': [start, start + maximum],
                        'numerical_environment': environment}
            print(f"{case['case_id']} N={maximum} started", flush=True)
            record = proxy_case(budget_config, case, identity, backend, pool)
            atomic_json(output / 'cases' / case['case_id'] / f'n{maximum}' / 'proxies.json', record)
            rows.extend(result_rows(case, maximum, record))
            failed = sum(r['status'] != 'ok' for r in record['rtd_runs'])
            print(f"{case['case_id']} N={maximum} complete; RTD failures={failed}", flush=True)
        path = output / f"trials.{case['case_id']}.csv"
        temporary = path.with_suffix('.csv.partial')
        with temporary.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted({k for row in rows for k in row}))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
        del backend, pool
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/paper/imagenet/proxy_trend.json')
    parser.add_argument('--output')
    parser.add_argument('--inputs-file', help='Feature paths in the shared inputs.json format')
    parser.add_argument('--case-id', help='Run one complete generator/embedding case')
    parser.add_argument('--list-cases', action='store_true')
    args = parser.parse_args()
    if args.list_cases:
        for case in load_study(args.config, args.inputs_file)['cases']:
            print(case['case_id'])
        return
    if args.output is None:
        parser.error('--output is required for computation')
    run(args.config, args.output, inputs_file=args.inputs_file, case_id=args.case_id)


if __name__ == '__main__':
    main()

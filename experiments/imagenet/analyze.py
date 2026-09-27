"""Summarize a new feature-study run without opening images or feature arrays.

Targets are saved development proxies (RTD by default), never population
truth. Development reporting emphasizes CE, trial SD and RMSE. Heldout
reporting emphasizes median absolute error within each case. Both sets of
columns are retained, with equal case weights in the outer summary.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from experiments.analysis_utils import check_row_keys, read_csvs, summarize_cell, summarize_cases, write_tables

KEY_COLUMNS = ('case_id', 'sample_budget', 'configuration_id', 'trial_id')
REQUIRED = (*KEY_COLUMNS, 'status', 'estimate')
PROXIES = ('rtd', 'plugin', 'fid_infinity')


def analyze(run_dir, output=None, proxy='rtd'):
    """Write complete-design per_case.csv and summary.csv and return both tables."""
    if proxy not in PROXIES:
        raise ValueError(f'proxy must be one of {PROXIES}')
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir/'manifest.json').read_text(encoding='utf-8'))
    config = manifest['config']
    sampling = config['sampling']
    count, budgets = sampling['num_trials'], sampling['sample_budgets']
    paths = sorted(run_dir.glob('trials.*.csv'))
    data = read_csvs(paths, REQUIRED, ('sample_budget', 'trial_id'))
    def specs(case, n):
        return manifest['specifications'][case['embedding']][str(n)]
    expected = {(case['case_id'], n, spec['configuration_id'], trial)
                for case in config['cases'] for n in budgets for spec in specs(case, n) for trial in range(count)}
    check_row_keys(data, KEY_COLUMNS, expected)
    results = []
    for case in config['cases']:
        rows_for_case = data[data['case_id'].eq(case['case_id'])]
        for key in ('embedding', 'generator'):
            if key in rows_for_case and not rows_for_case[key].eq(case[key]).all():
                raise ValueError(f'{key} differs from the manifest for {case["case_id"]}')
        path = run_dir/'cases'/case['case_id']/'proxies.json'
        target = json.loads(path.read_text(encoding='utf-8')).get('proxies', {}).get(proxy) if path.exists() else None
        if target is not None:
            try:
                target = float(target)
            except (TypeError, ValueError):
                target = np.nan
        for n in budgets:
            for spec in specs(case, n):
                rows = rows_for_case[rows_for_case['sample_budget'].eq(n)
                                     & rows_for_case['configuration_id'].eq(spec['configuration_id'])]
                results.append({k: case[k] for k in ('case_id', 'generator', 'embedding')}
                               | {'sample_budget': n, 'configuration_id': spec['configuration_id'],
                                  'proxy': proxy, 'target': target,
                                  **summarize_cell(rows, count, target, 'estimate')})
    per_case = pd.DataFrame(results)
    summary = summarize_cases(per_case, ('embedding', 'sample_budget', 'configuration_id', 'proxy'))
    mode = sampling['outer_mode']
    if mode not in ('development', 'heldout'):
        raise ValueError('sampling.outer_mode must be development or heldout')
    primary = 'median_absolute_error' if mode == 'heldout' else 'center_error,sd,rmse'
    per_case['evaluation_mode'] = mode
    summary['evaluation_mode'] = mode
    summary['primary_metric'] = primary
    summary['target_interpretation'] = 'saved_development_proxy;conditional_on_fixed_reference'
    write_tables(output or run_dir/'analysis', per_case, summary)
    return {'per_case': per_case, 'summary': summary}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, help='feature-study run containing manifest.json and completed case CSVs')
    parser.add_argument('--output', help='summary output directory (default: RUN/analysis)')
    parser.add_argument('--proxy', choices=PROXIES, default='rtd', help='saved development target (default: rtd)')
    args = parser.parse_args()
    tables = analyze(args.run, args.output, args.proxy)
    print(f"Wrote {len(tables['per_case'])} case cells and {len(tables['summary'])} summary cells.")


if __name__ == '__main__':
    main()

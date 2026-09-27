"""Summarize a new Gaussian run using its frozen design and raw.csv export.

Every configured trial must have status=ok and a finite estimate before a cell
has metrics. Outer means give each covariance instance equal weight. For one
isotropic instance, report trial SD; between-case SD is undefined and blank.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from experiments.analysis_utils import check_row_keys, read_csvs, summarize_cell, summarize_cases, write_tables
from .config import load_config, tasks

CASE_COLUMNS = ('suite', 'family', 'case_label', 'dimension', 'instance_id')
KEY_COLUMNS = (*CASE_COLUMNS, 'N', 'method', 'trial_id')
REQUIRED = (*KEY_COLUMNS, 'status', 'estimate_w2_sq_raw', 'true_w2_sq')


def analyze(run_dir, output=None):
    """Write per_case.csv and summary.csv; return the two DataFrames."""
    run_dir = Path(run_dir)
    config = load_config(json.loads((run_dir/'config.json').read_text(encoding='utf-8')))
    planned = list(tasks(config))
    budgets, methods = config['experiment']['sample_budgets'], config['experiment']['methods']
    data = read_csvs([run_dir/'raw.csv'], REQUIRED, ('dimension', 'instance_id', 'N', 'trial_id'))
    expected = {tuple(task[k] for k in CASE_COLUMNS)+(n, method, trial)
                for task in planned for n in budgets for method in methods for trial in range(task['num_trials'])}
    check_row_keys(data, KEY_COLUMNS, expected)
    grouped = {key: rows for key, rows in data.groupby(list(CASE_COLUMNS), sort=False, dropna=False)}
    empty = data.iloc[:0]
    results = []
    for task in planned:
        case = {k: task[k] for k in CASE_COLUMNS}
        case_rows = grouped.get(tuple(case.values()), empty)
        truths = pd.to_numeric(case_rows['true_w2_sq'], errors='coerce').to_numpy(dtype=float)
        if len(truths) and (not np.isfinite(truths).all() or not np.all(truths == truths[0])):
            raise ValueError(f'inconsistent or nonfinite population truth for {case}')
        target = float(truths[0]) if len(truths) else task['parameters'].get('target_w2_sq', np.nan)
        for n in budgets:
            for method in methods:
                rows = case_rows[case_rows['N'].eq(n) & case_rows['method'].eq(method)]
                results.append({**case, 'N': n, 'method': method, 'true_w2_sq': target,
                    **summarize_cell(rows, task['num_trials'], target, 'estimate_w2_sq_raw')})
    per_case = pd.DataFrame(results)
    summary = summarize_cases(per_case, ('suite', 'family', 'case_label', 'dimension', 'N', 'method'))
    summary['primary_metric'] = 'median_absolute_error'
    summary['uncertainty_basis'] = np.where(summary['expected_cases'].eq(1),
        'trial_sd;between_case_sd_undefined', 'sample_sd_across_case_median_absolute_errors')
    # Keep the within-population uncertainty visible in the one-case experiment.
    summary['trial_sd'] = np.where(summary['expected_cases'].eq(1), summary['sd_mean'], np.nan)
    write_tables(output or run_dir/'analysis', per_case, summary)
    return {'per_case': per_case, 'summary': summary}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, help='Gaussian run directory containing config.json and raw.csv')
    parser.add_argument('--output', help='summary output directory (default: RUN/analysis)')
    args = parser.parse_args()
    tables = analyze(args.run, args.output)
    print(f"Wrote {len(tables['per_case'])} case cells and {len(tables['summary'])} summary cells.")


if __name__ == '__main__':
    main()

"""Small reporting helpers: complete trial cells only, equal case weighting."""
from pathlib import Path
import numpy as np
import pandas as pd
from gaussian_w2.evaluation import across_cases, trial_metrics

METRICS = ('mean_estimate', 'signed_error', 'center_error', 'sd', 'rmse', 'median_absolute_error')


def read_csvs(paths, required, integer_columns):
    """Read full-precision exports without opening original feature arrays."""
    frames = []
    for path in paths:
        if not Path(path).exists():
            continue
        frame = pd.read_csv(path, float_precision='round_trip')
        missing = set(required) - set(frame.columns)
        if missing:
            raise ValueError(f'{path} is missing columns: {sorted(missing)}')
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=required)
    data = pd.concat(frames, ignore_index=True)
    for column in integer_columns:
        values = pd.to_numeric(data[column], errors='coerce')
        if not np.isfinite(values).all() or (values != np.floor(values)).any():
            raise ValueError(f'{column} must contain finite integer identifiers')
        data[column] = values.astype('int64')
    return data


def check_row_keys(frame, columns, expected):
    if frame.duplicated(list(columns)).any():
        raise ValueError('duplicate trial rows cannot be counted as independent observations')
    observed = set(frame.loc[:, list(columns)].itertuples(index=False, name=None))
    if observed - expected:
        raise ValueError(f'unknown trial rows outside the configured design: {list(observed-expected)[:3]}')


def summarize_cell(rows, expected_trials, target, estimate_column):
    """Make all metrics unavailable unless every configured trial succeeded."""
    values = pd.to_numeric(rows[estimate_column], errors='coerce').to_numpy(dtype=float)
    successful = rows['status'].eq('ok').to_numpy() & np.isfinite(values)
    attempted, ok = len(rows), int(successful.sum())
    reason = []
    if attempted != expected_trials:
        reason.append('incomplete_trials')
    if ok != attempted:
        reason.append('failed_or_nonfinite_trials')
    if target is None or not np.isfinite(target):
        reason.append('unavailable_target')
    result = dict(expected_trials=expected_trials, attempted_trials=attempted, ok_trials=ok,
                  failed_trials=attempted-ok, missing_trials=max(0, expected_trials-attempted),
                  available=not reason, unavailable_reason=';'.join(reason))
    result.update({name: np.nan for name in METRICS})
    if not reason:
        measured = trial_metrics(values, float(target))
        result.update({name: measured[name] for name in METRICS})
    return result


def summarize_cases(frame, group_columns):
    """Aggregate the complete configured case grid without dropping failed cases."""
    output = []
    for key, cases in frame.groupby(list(group_columns), sort=False, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        available = bool(cases['available'].all())
        row = dict(zip(group_columns, key))
        row.update(expected_cases=len(cases), attempted_cases=int((cases.attempted_trials > 0).sum()),
                   available_cases=int(cases.available.sum()), available=available,
                   unavailable_reason='' if available else 'unavailable_cases')
        for count in ('expected_trials', 'attempted_trials', 'ok_trials', 'failed_trials', 'missing_trials'):
            row[count] = int(cases[count].sum())
        for metric in METRICS:
            values = cases[metric].to_numpy(dtype=float)
            stats = across_cases(values) if available and np.isfinite(values).all() else {'mean': np.nan, 'sd': np.nan}
            row[f'{metric}_mean'] = stats['mean']
            row[f'{metric}_between_case_sd'] = stats['sd']
        output.append(row)
    return pd.DataFrame(output)


def write_tables(output, per_case, summary):
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    for name, frame in [('per_case', per_case), ('summary', summary)]:
        frame.to_csv(destination/f'{name}.csv', index=False, float_format='%.17g', na_rep='')

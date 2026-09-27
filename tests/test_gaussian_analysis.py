"""Reporting contracts for complete, failed, and absent Gaussian trial cells."""
import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from experiments.gaussian.analyze import analyze
from experiments.gaussian.config import load_config

ROOT = Path(__file__).resolve().parents[1]
METRICS = ('mean_estimate', 'signed_error', 'center_error', 'sd', 'rmse', 'median_absolute_error')


def write_run(tmp_path, *, instances=2, trials=3):
    config = load_config(ROOT/'configs/examples/gaussian.yaml')
    config['experiment']['sample_budgets'] = [160]
    config['experiment']['methods'] = ['empirical_plugin']
    case = copy.deepcopy(config['covariance_cases'][0])
    case.update(num_instances=instances, num_trials=trials)
    config['covariance_cases'] = [case]
    (tmp_path/'config.json').write_text(json.dumps(config), encoding='utf-8')
    rows = []
    for instance in range(instances):
        values = [1., 3., 5.] if instance == 0 else [4., 4., 4.]
        for trial, value in enumerate((values * max(1, trials))[:trials]):
            rows.append(dict(suite=case['suite'], family=case['family'], case_label=case['case_label'],
                dimension=4, instance_id=instance, N=160, method='empirical_plugin', trial_id=trial,
                status='ok', estimate_w2_sq_raw=value, true_w2_sq=2.))
    frame = pd.DataFrame(rows)
    frame.to_csv(tmp_path/'raw.csv', index=False)
    return frame


def test_exact_case_metrics_and_equal_case_outer_summary(tmp_path):
    write_run(tmp_path)
    tables = analyze(tmp_path)
    a, b = tables['per_case'].to_dict('records')
    assert (a['mean_estimate'], a['center_error'], a['sd'], a['median_absolute_error']) == (3., 1., 2., 1.)
    assert a['rmse'] == pytest.approx(np.sqrt(11/3))
    assert b['median_absolute_error'] == 2.
    outer = tables['summary'].iloc[0]
    assert outer['expected_cases'] == outer['available_cases'] == 2
    assert outer['expected_trials'] == outer['attempted_trials'] == outer['ok_trials'] == 6
    assert outer['median_absolute_error_mean'] == 1.5
    assert outer['median_absolute_error_between_case_sd'] == pytest.approx(np.sqrt(.5))
    assert np.isnan(outer['trial_sd'])
    assert (tmp_path/'analysis/per_case.csv').exists()
    assert (tmp_path/'analysis/summary.csv').exists()


def test_single_population_keeps_50_trial_sd_and_no_between_case_sd(tmp_path):
    frame = write_run(tmp_path, instances=1, trials=50)
    values = np.arange(50, dtype=float) / 7
    frame['estimate_w2_sq_raw'] = values
    frame.to_csv(tmp_path/'raw.csv', index=False, float_format='%.17g')
    table = analyze(tmp_path)['summary'].iloc[0]
    assert table['expected_trials'] == table['ok_trials'] == 50
    assert table['trial_sd'] == pytest.approx(values.std(ddof=1))
    assert np.isnan(table['median_absolute_error_between_case_sd'])
    assert table['uncertainty_basis'] == 'trial_sd;between_case_sd_undefined'


@pytest.mark.parametrize('kind', ['failed', 'nonfinite', 'missing'])
def test_failed_or_missing_trial_makes_whole_cell_and_outer_unavailable(tmp_path, kind):
    frame = write_run(tmp_path)
    if kind == 'missing':
        frame = frame.drop(index=0)
    elif kind == 'failed':
        frame.loc[0, ['status', 'estimate_w2_sq_raw']] = ['numerical_failure', np.nan]
    else:
        frame.loc[0, 'estimate_w2_sq_raw'] = np.inf
    frame.to_csv(tmp_path/'raw.csv', index=False)
    tables = analyze(tmp_path)
    first = tables['per_case'].iloc[0]
    assert not first['available']
    assert first['ok_trials'] == 2 and first['expected_trials'] == 3
    assert first['missing_trials'] == (1 if kind == 'missing' else 0)
    assert first['failed_trials'] == (0 if kind == 'missing' else 1)
    assert all(np.isnan(first[k]) for k in METRICS)
    outer = tables['summary'].iloc[0]
    assert not outer['available'] and outer['available_cases'] == 1
    assert np.isnan(outer['median_absolute_error_mean'])


def test_completely_absent_csv_preserves_expected_cells(tmp_path):
    write_run(tmp_path)
    (tmp_path/'raw.csv').unlink()
    tables = analyze(tmp_path)
    assert len(tables['per_case']) == 2
    assert tables['per_case']['attempted_trials'].eq(0).all()
    assert tables['per_case']['missing_trials'].eq(3).all()
    assert tables['summary'].iloc[0]['expected_cases'] == 2
    assert not tables['summary'].iloc[0]['available']


@pytest.mark.parametrize('problem', ['duplicate', 'unknown_trial', 'unknown_method', 'inconsistent_truth'])
def test_invalid_rows_are_rejected(tmp_path, problem):
    frame = write_run(tmp_path)
    if problem == 'duplicate':
        frame = pd.concat([frame, frame.iloc[:1]])
    elif problem == 'unknown_trial':
        frame.loc[0, 'trial_id'] = 99
    elif problem == 'unknown_method':
        frame.loc[0, 'method'] = 'unknown'
    else:
        frame.loc[0, 'true_w2_sq'] += 1
    frame.to_csv(tmp_path/'raw.csv', index=False)
    with pytest.raises(ValueError):
        analyze(tmp_path)


def test_csv_preserves_float64_summary_values(tmp_path):
    frame = write_run(tmp_path)
    frame.loc[0, 'estimate_w2_sq_raw'] = 1.2345678901234567
    frame.to_csv(tmp_path/'raw.csv', index=False, float_format='%.17g')
    result = analyze(tmp_path, tmp_path/'custom_summary')
    saved = pd.read_csv(tmp_path/'custom_summary/per_case.csv', float_precision='round_trip')
    assert saved['rmse'].tolist() == result['per_case']['rmse'].tolist()

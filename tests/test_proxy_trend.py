"""Proxy-budget alignment, repetition summaries, and terminal output handling."""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.imagenet.proxy_trend import load_study, result_rows
from experiments.imagenet.plot_proxy_trend import summarize, summarize_runs
from gaussian_w2.imagenet.io import atomic_json, load_config

ROOT = Path(__file__).resolve().parents[1]


def test_proxy_trend_preserves_development_proxy_settings():
    config = load_study(ROOT/'configs/paper/imagenet/proxy_trend.json')
    original = load_config(ROOT/'configs/paper/imagenet/ablation.json')
    assert config['sampling']['sample_budgets'] == list(range(30000, 300001, 30000))
    assert len(config['cases']) == 21
    for key in ('proxies', 'rtd', 'numerics', 'cases'):
        assert config[key] == original[key]


@pytest.fixture
def completed_case(tmp_path):
    config = load_study(ROOT/'configs/paper/imagenet/proxy_trend.json')
    case = config['cases'][0]
    budget = 300000
    config['cases'] = [case]
    atomic_json(tmp_path/'manifest.json', {'schema': 'imagenet-proxy-trend-v1',
                'config': config, 'sample_budgets': [budget], 'case_ids': [case['case_id']]})
    record = {'fid_infinity_runs': [], 'rtd_runs': []}
    for repetition, value in enumerate([1., 2., 3., 4., 90.]):
        record['fid_infinity_runs'].append({'repetition': repetition, 'estimate': value,
            'nodes': [{'n': budget, 'estimate': -2., 'runtime_seconds': 1.}]})
        record['rtd_runs'].append({'binding': {'repetition': repetition}, 'estimate': value,
            'status': 'ok', 'runtime_seconds': 1.})
    path = tmp_path/f"trials.{case['case_id']}.csv"
    pd.DataFrame(result_rows(case, budget, record)).to_csv(path, index=False)
    return tmp_path, path, case, record


def test_proxy_summary_uses_median_and_keeps_negative_values(completed_case):
    root, _, _, _ = completed_case
    summary = summarize(root).set_index('proxy')
    assert summary.loc['plugin', 'median_estimate'] == -2.
    assert summary.loc['fid_infinity', 'median_estimate'] == 3.
    assert summary.loc['rtd', 'median_estimate'] == 3.
    assert summary.loc['plugin', 'n_attempted'] == 1
    assert summary.loc['rtd', 'n_attempted'] == 5


def test_failed_proxy_does_not_report_a_success_only_median(completed_case):
    root, path, case, record = completed_case
    record = copy.deepcopy(record)
    record['rtd_runs'][0].update(status='failed', estimate=None, runtime_seconds=None,
                               failure_type='FloatingPointError', error_message='test failure')
    pd.DataFrame(result_rows(case, 300000, record)).to_csv(path, index=False)
    row = summarize(root).set_index('proxy').loc['rtd']
    assert np.isnan(row.median_estimate)
    assert row.n_ok == 4 and row.n_failed == 1 and not row.available


@pytest.mark.parametrize('problem', ['duplicate', 'missing'])
def test_proxy_summary_requires_all_recorded_repetitions(completed_case, problem):
    root, path, _, _ = completed_case
    data = pd.read_csv(path)
    data = pd.concat([data, data.iloc[:1]]) if problem == 'duplicate' else data.iloc[:-1]
    data.to_csv(path, index=False)
    with pytest.raises(ValueError, match='duplicate|missing'):
        summarize(root)


def test_50k_supplement_only_changes_budget(monkeypatch):
    monkeypatch.setenv('FD_DATA_ROOT', '/example/features')
    original = load_study(ROOT/'configs/paper/imagenet/proxy_trend.json')
    supplement = load_study(ROOT/'configs/paper/imagenet/proxy_trend_50k.json',
                            inputs_file=ROOT/'configs/paper/imagenet/inputs.json')
    assert supplement['sampling']['sample_budgets'] == [50000]
    original['sampling']['sample_budgets'] = [50000]
    assert supplement == original


def test_combine_supplement_preserves_existing_results(completed_case):
    root, path, _, _ = completed_case
    supplement = root/'supplement'
    manifest = json.loads((root/'manifest.json').read_text())
    manifest['sample_budgets'] = [50000]
    manifest['config']['sampling']['sample_budgets'] = [50000]
    atomic_json(supplement/'manifest.json', manifest)
    rows = pd.read_csv(path)
    rows['sample_budget'] = 50000
    rows['estimate'] += 10
    rows.to_csv(supplement/path.name, index=False)
    combined = summarize_runs([root, supplement])
    assert len(combined) == 6
    pd.testing.assert_frame_equal(combined.iloc[:3], summarize(root))
    pd.testing.assert_frame_equal(combined.iloc[3:].reset_index(drop=True), summarize(supplement))
    manifest['config']['rtd']['m0_numerator'] = 4
    atomic_json(supplement/'manifest.json', manifest)
    with pytest.raises(ValueError, match='different study settings'):
        summarize_runs([root, supplement])


def test_combine_rejects_overlapping_budget_cells(completed_case):
    root, _, _, _ = completed_case
    with pytest.raises(ValueError, match='overlapping'):
        summarize_runs([root, root])

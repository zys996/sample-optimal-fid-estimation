"""Audit the publication reconstruction against independent manuscript fixtures."""
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import pytest

from reproduce.__main__ import reproduce
from reproduce.audit_nodes import audit_nodes
from reproduce.metrics import DATA, development_statistics, gaussian_statistics, read, read_json


def test_all_paper_tables_and_final_budget_source(tmp_path):
    tables=reproduce(tmp_path,figures=False)
    expected=read_json('paper_table_values')
    assert len(tables)==13 and tables.keys()==expected.keys()
    for label,tex in tables.items():
        body=tex.split(r'\midrule',1)[1].split(r'\bottomrule')[0]
        actual=re.findall(r'-?\d+\.\d+',body)
        wanted=list(expected[label]['decimal_values'])
        assert actual==wanted,label
    for path in (tmp_path / 'tables').glob('*.tex'):
        tex_body = path.read_text(encoding="utf-8").split(r'\midrule', 1)[1].split(r'\bottomrule')[0]
        html_body = path.with_suffix('.html').read_text(encoding="utf-8").split('<tbody>', 1)[1]
        assert re.findall(r'-?\d+\.\d+', html_body) == re.findall(r'-?\d+\.\d+', tex_body)
        assert '\\' not in html_body
    # Independently recomputed raw-trial aggregates retain precision before display.
    ablation=pd.read_csv(tmp_path/'tables/ablation_summary.csv',float_precision='round_trip').set_index(['embedding','stage'])
    assert ablation.loc[('fid','selected_strategy'),'mean_rmse']==pytest.approx(.03075080241525349,abs=1e-15,rel=0)
    assert ablation.loc[('clip','ols'),'mean_rmse']==pytest.approx(.11206674785059163,abs=1e-15,rel=0)
    assert ablation.loc[('fid','vale_weighting'),'rmse_drop_percent']==pytest.approx(26.05123891661768,abs=1e-12,rel=0)
    chosen=json.loads((tmp_path/'analysis/selected_configurations.json').read_text(encoding="utf-8"))
    identifiers=lambda rows: {(r['embedding'],r['order']):r['configuration_id'] for r in rows}
    assert identifiers(chosen)==identifiers(read_json('expected_selection'))
    budget=pd.read_csv(tmp_path/'analysis/sample_budget_trials.csv',float_precision='round_trip')
    heldout=pd.read_csv(tmp_path/'analysis/selected_heldout_trials.csv',float_precision='round_trip')
    endpoint=budget[budget.max_samples.eq(50000)]
    assert endpoint.measurement_source.eq('heldout_trials').all()
    cols=['case_id','trial_id','configuration_id','estimate','runtime_seconds']
    pd.testing.assert_frame_equal(endpoint[cols].reset_index(drop=True),heldout[cols].reset_index(drop=True),check_exact=True)
    report=pd.read_csv(tmp_path/'tables/sample_budget.csv',float_precision='round_trip')
    heldout_report=pd.read_csv(tmp_path/'tables/heldout.csv',float_precision='round_trip')
    keys=['embedding','method','proxy']
    pd.testing.assert_frame_equal(
        report[report.max_samples.eq(50000)].sort_values(keys).reset_index(drop=True),
        heldout_report.sort_values(keys).reset_index(drop=True),
        check_exact=True,
    )


def test_randomized_proxy_values_are_medians_of_five_repetitions():
    repetitions=read('proxy_runs').groupby(['case_id','proxy']).estimate.agg(['median','count'])
    recorded=read('proxies')
    proxies=recorded[recorded.proxy.isin(['rtd','fid_infinity'])].set_index(['case_id','proxy']).reference_value
    assert repetitions.index.equals(proxies.sort_index().index)
    assert repetitions['count'].eq(5).all()
    pd.testing.assert_series_equal(repetitions['median'],proxies.sort_index(),check_names=False,check_exact=True)


def test_nodes_rebuild_archived_estimates():
    result=audit_nodes()
    assert sum(row['estimates_rebuilt'] for row in result)==94775
    assert max(row['max_absolute_difference'] for row in result)<2e-11


def test_gaussian_aggregation_is_median_absolute_trial_error():
    raw=read('gaussian_random')
    cases,summary=gaussian_statistics(raw)
    subset=raw[raw.family.eq('haar_log_uniform') & raw.dimension.eq(200) & raw.N.eq(50000) & raw.method.eq('fid_infinity_ols_order2')]
    # Independent direct calculation catches abs(median estimate - truth),
    # pooled-trial error, or accidental population SD substitutions.
    values=[float(np.median(np.abs(g.estimate_w2_sq_raw-g.true_w2_sq))) for _,g in subset.groupby('instance_id')]
    row=summary[summary.family.eq('haar_log_uniform') & summary.dimension.eq(200) & summary.N.eq(50000) & summary.method.eq('fid_infinity_ols_order2')].iloc[0]
    assert row.mean_case_median==pytest.approx(np.mean(values),abs=1e-15)
    assert row.sd_case_median==pytest.approx(np.std(values,ddof=1),abs=1e-15)


def test_failed_or_duplicate_records_are_rejected():
    raw=read('development_trials')
    proxies=read('proxies');plan=read_json('development_plan')
    broken=raw.copy();broken.loc[0,'status']='failed'
    with pytest.raises(ValueError,match='Failed'):
        development_statistics(broken,proxies,plan)
    with pytest.raises(ValueError,match='Duplicate'):
        development_statistics(pd.concat([raw,raw.iloc[:1]]),proxies,plan)


def test_bundle_contains_no_private_machine_metadata():
    import gzip
    for path in DATA.iterdir():
        text=gzip.decompress(path.read_bytes()).decode() if path.suffix=='.gz' else path.read_text(encoding="utf-8")
        for token in ['/Users/','/scratch/','git_commit','hostname','account_name']:
            assert token not in text,(path.name,token)


def test_executed_notebook_contains_every_paper_figure_and_table():
    path = Path(__file__).resolve().parents[1] / 'reproduce' / 'paper_results.ipynb'
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cells = [cell for cell in notebook['cells'] if cell['cell_type'] == 'code']
    assert all(cell['execution_count'] is not None for cell in cells)
    outputs = [output for cell in cells for output in cell['outputs']]
    assert all(output['output_type'] != 'error' for output in outputs)
    assert sum('image/png' in output.get('data', {}) for output in outputs) == 11
    tables = [''.join(output.get('data', {}).get('text/html', [])) for output in outputs]
    assert sum('<table' in table for table in tables) == 13


def test_proxy_trend_records_and_node_fits():
    from gaussian_w2.estimators.extrapolation import fit_inverse_sample_polynomial
    from reproduce.metrics import proxy_trend_statistics

    raw = read('proxy_trend_trials')
    nodes = read('proxy_trend_nodes')
    summary = proxy_trend_statistics(raw)
    regular = summary[summary.sample_budget.ne(50000)]
    assert len(regular) == 630
    assert regular.groupby(['embedding', 'sample_budget', 'proxy']).size().eq(7).all()
    assert set(regular.sample_budget) == set(range(30000, 300001, 30000))
    assert set(regular.proxy) == {'plugin', 'fid_infinity', 'rtd'}
    assert len(summary[summary.sample_budget.eq(50000)]) == 3
    keys = ['case_id', 'sample_budget', 'repetition']
    saved = raw[raw.proxy.eq('fid_infinity')].set_index(keys).estimate
    rebuilt = []
    for key, group in nodes.groupby(keys):
        assert len(group) == group.sample_size.nunique() == 15
        assert group.status.eq('ok').all()
        fit = fit_inverse_sample_polynomial(group.sample_size, group.estimate, order=1)
        rebuilt.append(abs(fit['intercept'] - saved[key]))
    assert len(rebuilt) == 1055
    assert max(rebuilt) < 1e-6


def test_intro_refits_match_published_curves():
    from reproduce.metrics import intro_statistics, proxy_trend_statistics

    trend = proxy_trend_statistics(read('proxy_trend_trials'))
    gaussian, imagenet = intro_statistics(read('gaussian_random'), read('gaussian_nodes'),
                                         trend, read('proxy_trend_nodes'))
    assert len(gaussian) == len(imagenet) == 30
    assert set(imagenet.sample_budget) == {50000, 60000, 90000, 120000, 150000,
                                          180000, 210000, 240000, 270000, 300000}
    # Endpoints read from the original figure data, before the reproduction port.
    for frame, coordinate, value, expected in [
        (gaussian, 'dimension', 3000, {'fid_infinity': 1.3516168182250965,
                                      'vale2': .10433050340149067, 'rtd': .10200832593162659}),
        (imagenet, 'sample_budget', 300000, {'fid_infinity': .6311319511359353,
                                            'vale2': .6272750558511081, 'rtd': .6407780583451685}),
    ]:
        metric = 'mean_case_median' if coordinate == 'dimension' else 'median_estimate'
        values = frame[frame[coordinate].eq(value)].set_index('method')[metric]
        for method, target in expected.items():
            assert values[method] == pytest.approx(target, abs=1e-6, rel=0)

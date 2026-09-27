from pathlib import Path
import json
import pytest
from experiments.gaussian.config import load_config, tasks
from experiments.gaussian.run import run
from experiments.gaussian.collect import collect
from experiments.gaussian.records import load_task

ROOT = Path(__file__).resolve().parents[1]


def test_paper_designs_preserve_final_grids():
    isotropic = load_config(ROOT/'configs/paper/gaussian/isotropic.yaml')
    assert isotropic['master_seed'] == 20260913
    assert isotropic['experiment']['dimensions'] == [5000]
    assert isotropic['experiment']['sample_budgets'] == [100000]
    assert isotropic['fid_infinity']['minimum_samples'] == 5000
    assert isotropic['covariance_cases'][0]['case_label'] == 'section6_isotropic_b05_D5'
    assert isotropic['covariance_cases'][0]['num_trials'] == 50
    for name, count, seed in [('random', 300, 20260812), ('stress', 600, 20260920)]:
        config = load_config(ROOT/f'configs/paper/gaussian/{name}.yaml')
        assert config['master_seed'] == seed
        assert len(list(tasks(config))) == count
        assert config['experiment']['sample_budgets'] == [50000, 60000, 80000, 100000]
        assert all(t['num_trials'] == 5 and t['num_instances'] == 10 for t in tasks(config))
        assert config['general_order']['degree_cap'] is None
        assert config['experiment']['m0_fraction_numerator'] == 4
        assert config['experiment']['m0_fraction_denominator'] == 5


def test_three_suite_cpu_example_and_independent_collection(tmp_path, monkeypatch):
    config = load_config(ROOT/'configs/examples/gaussian.yaml')
    result = run(config, tmp_path)
    assert result['rows'] == result['ok'] == 96
    assert result['failed'] == 0 and result['tasks'] == 3
    records = {p.name: p.read_bytes() for p in (tmp_path/'records').glob('*.json')}
    assert len(records) == 3
    assert not (tmp_path/'raw.csv').exists()
    from experiments.gaussian import run as module
    monkeypatch.setattr(module, 'population', lambda *a: pytest.fail('collection ran a population'))
    summary = collect(tmp_path)
    assert summary['tasks'] == summary['expected_tasks'] == 3
    assert summary['rows'] == summary['ok'] == 96
    with pytest.raises(FileExistsError, match='task result already exists'):
        run(config, tmp_path)
    assert records == {p.name: p.read_bytes() for p in (tmp_path/'records').glob('*.json')}
    example = json.loads(next(iter(records.values())))
    for trial in example['trials']:
        assert set(trial['curves']) == {'160', '200'}
        for row in trial['rows']:
            assert row['signed_error_w2_sq_raw'] == pytest.approx(row['estimate_w2_sq_raw']-row['true_w2_sq'])
            if row['method'] == 'general_adaptive':
                assert row['m0'] == row['N']*4//5


def test_independent_tasks_and_explicit_single_task_overwrite(tmp_path, monkeypatch):
    config = load_config(ROOT/'configs/examples/gaussian.yaml')
    run(config, tmp_path, task_index=0)
    first = (tmp_path/'records/task_00000.json').read_bytes()
    run(config, tmp_path, task_index=1)
    assert (tmp_path/'records/task_00000.json').read_bytes() == first
    assert collect(tmp_path)['tasks'] == 2
    other = (tmp_path/'records/task_00001.json').read_bytes()
    with pytest.raises(ValueError, match='requires --task-index'):
        run(config, tmp_path, overwrite=True)
    from experiments.gaussian import run as module
    original = module.sample_pool
    calls = []
    def sample(*args, **kwargs):
        calls.append(args[2])
        return original(*args, **kwargs)
    monkeypatch.setattr(module, 'sample_pool', sample)
    run(config, tmp_path, task_index=0, overwrite=True)
    assert len(calls) == 2
    assert (tmp_path/'records/task_00001.json').read_bytes() == other


def test_interrupted_task_has_no_partial_result(tmp_path, monkeypatch):
    from experiments.gaussian import run as module
    config = load_config(ROOT/'configs/examples/gaussian.yaml')
    original = module.sample_pool
    calls = []
    def interrupt_second_trial(*args, **kwargs):
        calls.append(args[2])
        if len(calls) == 2:
            raise RuntimeError('interrupted')
        return original(*args, **kwargs)
    monkeypatch.setattr(module, 'sample_pool', interrupt_second_trial)
    with pytest.raises(RuntimeError, match='interrupted'):
        run(config, tmp_path, task_index=0)
    assert not list((tmp_path/'records').glob('*.json'))
    monkeypatch.setattr(module, 'sample_pool', original)
    result = run(config, tmp_path, task_index=0)
    assert result['rows'] == result['ok'] == 32
    assert len(json.loads((tmp_path/'records/task_00000.json').read_text())['trials']) == 2


def test_additional_task_rejects_changed_design(tmp_path):
    config = load_config(ROOT/'configs/examples/gaussian.yaml')
    run(config, tmp_path, task_index=0)
    saved_config = (tmp_path/'config.json').read_bytes()
    config['master_seed'] += 1
    with pytest.raises(ValueError, match='different frozen configuration'):
        run(config, tmp_path, task_index=1)
    assert (tmp_path/'config.json').read_bytes() == saved_config


@pytest.mark.parametrize('problem,match', [
    ('missing_trial', 'trial coverage'), ('missing_row', 'method/budget coverage'),
    ('nonfinite', 'successful estimate is nonfinite'), ('wrong_seed', 'row identity'),
])
def test_collection_checks_saved_output(tmp_path, problem, match):
    config = load_config(ROOT/'configs/examples/gaussian.yaml')
    run(config, tmp_path, task_index=0)
    collect(tmp_path)
    original_export = (tmp_path/'raw.csv').read_bytes()
    path = tmp_path/'records/task_00000.json'
    record = json.loads(path.read_text(encoding='utf-8'))
    if problem == 'missing_trial':
        record['trials'].pop()
    elif problem == 'missing_row':
        record['trials'][0]['rows'].pop()
    elif problem == 'nonfinite':
        record['trials'][0]['rows'][0]['estimate_w2_sq_raw'] = None
    else:
        record['trials'][0]['rows'][0]['sampling_seed'] += 1
    path.write_text(json.dumps(record), encoding='utf-8')
    with pytest.raises(ValueError, match=match):
        collect(tmp_path)
    assert (tmp_path/'raw.csv').read_bytes() == original_export


def test_collection_recomputes_derived_errors(tmp_path):
    config = load_config(ROOT/'configs/examples/gaussian.yaml')
    run(config, tmp_path, task_index=0)
    path = tmp_path/'records/task_00000.json'
    record = json.loads(path.read_text(encoding='utf-8'))
    row = record['trials'][0]['rows'][0]
    row['signed_error_w2_sq_raw'] = row['abs_error_w2_sq_raw'] = 999.
    path.write_text(json.dumps(record), encoding='utf-8')
    exported = load_task(path, config, list(tasks(config))[0], 0)
    assert exported[0]['signed_error_w2_sq_raw'] == row['estimate_w2_sq_raw'] - row['true_w2_sq']
    assert exported[0]['abs_error_w2_sq_raw'] == abs(exported[0]['signed_error_w2_sq_raw'])


def test_numerical_failures_are_saved_and_counted(tmp_path, monkeypatch):
    from experiments.gaussian import run as module
    config = load_config(ROOT/'configs/examples/gaussian.yaml')
    config['covariance_cases'] = config['covariance_cases'][:1]
    config['experiment']['methods'] = ['empirical_plugin']
    monkeypatch.setattr(module, 'estimate_budget', lambda *args: (
        {'empirical_plugin': {'status': 'numerical_failure', 'error_message': 'test failure'}}, {}))
    result = run(config, tmp_path)
    assert result['failed'] == result['rows'] == 4
    summary = collect(tmp_path)
    assert summary['failed'] == summary['rows'] == 4
    record = json.loads((tmp_path/'records/task_00000.json').read_text())
    assert all(row['signed_error_w2_sq_raw'] is None
               for trial in record['trials'] for row in trial['rows'])

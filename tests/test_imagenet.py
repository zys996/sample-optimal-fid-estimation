"""Check the ImageNet paper configuration and sampling rules."""
import copy
from pathlib import Path

import numpy as np
import pytest

from gaussian_w2.imagenet.design import (
    node_indices, outer_indices, specifications, validate_config,
)
from gaussian_w2.imagenet.io import read_json, load_config

ROOT = Path(__file__).resolve().parents[1]


def test_paper_full_bank_and_frozen_selection():
    c=validate_config(load_config(ROOT/'configs/paper/imagenet/ablation.json'))
    assert len(c['cases'])==21
    assert {c['embedding'] for c in c['cases']}=={'fid','fd_dinov2','clip'}
    bank=specifications(c,'fid',50000)
    assert len(bank)==102
    for order in range(1,5):
        for schedule in ('uniform_n','uniform_inverse_n','chebyshev_inverse_n'):
            for ell in (order+1,10,15,20):
                assert any(s['configuration_id']==f'ols_p{order}_{schedule}_l{ell}' for s in bank)
                assert any(s['configuration_id']==f'va_p{order}_{schedule}_l{ell}' for s in bank)
    heldout=validate_config(load_config(ROOT/'configs/paper/imagenet/heldout.json'))
    selection=read_json(ROOT/'configs/paper/imagenet/development_selection.json')
    for e in heldout['estimators']:
        for spec in heldout['estimators'][e]:
            if spec['configuration_id'].startswith('va_order'):
                expected=next(x for x in selection['selected_by_order'] if x['embedding']==e and x['order']==spec['order'])
                assert (spec['num_points'],spec['sample_schedule'])==(expected['num_points'],expected['sample_schedule'])


def test_paper_generator_seeds_and_lane_ranges():
    c=read_json(ROOT/'configs/paper/imagenet/preparation.json')
    pools={p['pool_id']:p for p in c['pools']}
    assert len(pools)==7
    assert pools['stylegan_xl_imagenet512']['seed_start']==2026090301+5*550000
    assert pools['ddo_edm2_l_imagenet512']['seed_start']==2026090301+6*550000
    assert pools['var_d36_imagenet512']['seed_start']==2026090301+7*550000
    assert pools['var_d36_imagenet512']['lane_ranges']==[[0,184000],[184000,367000],[367000,550000]]


def test_paper_sampling_uses_fresh_nodes_and_disjoint_blocks():
    config=load_config(ROOT/'configs/paper/imagenet/ablation.json')
    case=config['cases'][0]
    master,generator=case['master_seed'],case['generator']
    settings=config['sampling']
    phase=settings['node_seed_phase']
    maximum=settings['sample_budgets'][0]
    minimum=settings['minimum_samples']
    trial=1

    idx,seed,block=outer_indices(config,case,trial,maximum)
    start,stop=settings['development_range']
    assert block==[start,stop]
    assert len(idx)==len(np.unique(idx))==maximum
    assert np.all((idx>=start)&(idx<stop))
    np.testing.assert_array_equal(idx,start+np.random.default_rng(seed).permutation(stop-start)[:maximum])

    small,_=node_indices(master,generator,phase,trial,minimum,maximum)
    larger,_=node_indices(master,generator,phase,trial,2*minimum,maximum)
    assert not np.array_equal(small,larger[:minimum])
    assert node_indices(master,generator,phase,trial,maximum,maximum)[0] is None

    heldout=load_config(ROOT/'configs/paper/imagenet/heldout.json')
    blocks=[outer_indices(heldout,case,t,maximum)[0] for t in range(heldout['sampling']['num_trials'])]
    assert len(np.unique(np.concatenate(blocks)))==sum(len(block) for block in blocks)


def _trial_record(identity, specs):
    return {'binding': identity, 'specifications': copy.deepcopy(specs), 'nodes': [],
            'rows': [{'configuration_id': s['configuration_id'], 'status': 'ok',
                      'estimate': float(i), 'runtime_seconds': 1., 'fit_seconds': 0.}
                     for i, s in enumerate(specs)],
            'numerical_environment': {'python': 'recorded version'}}


@pytest.mark.parametrize('problem', ['specifications', 'duplicate', 'nonfinite'])
def test_trial_record_checks_estimator_identity_and_complete_finite_outputs(problem):
    from gaussian_w2.imagenet.runner import validate_trial
    config = load_config(ROOT/'configs/paper/imagenet/heldout.json')
    specs = specifications(config, 'fid', 50000)
    identity = {'case_id': config['cases'][0]['case_id'], 'trial_id': 0}
    record = _trial_record(identity, specs)
    validate_trial(record, identity, specs)
    if problem == 'specifications':
        changed = copy.deepcopy(config)
        changed['estimators']['fid'][1]['num_points'] = 10
        specs = specifications(changed, 'fid', 50000)
    elif problem == 'duplicate':
        record['rows'][1] = record['rows'][0]
    else:
        record['rows'][0]['estimate'] = float('nan')
    with pytest.raises(ValueError):
        validate_trial(record, identity, specs)


def test_50k_reuse_preserves_heldout_rows_and_allows_different_node_phase(tmp_path):
    from gaussian_w2.imagenet.io import atomic_json
    from gaussian_w2.imagenet.runner import case_binding, reuse_trial, validate_trial
    source = load_config(ROOT/'configs/paper/imagenet/heldout.json')
    target = load_config(ROOT/'configs/paper/imagenet/budgets_reuse_heldout.json')
    target['reuse'] = [{'sample_budget': 50000, 'run': str(tmp_path)}]
    case = source['cases'][0]
    specs = specifications(source, case['embedding'], 50000)
    target_specs = specifications(target, case['embedding'], 50000)
    assert source['sampling']['node_seed_phase'] != target['sampling']['node_seed_phase']
    inputs = {case['case_id']: {'features': {'path': case['features']},
                              'reference': {'path': case['reference']}}}
    plan = {'schema': 'imagenet-run-v3', 'config': source, 'inputs': inputs,
            'specifications': {case['embedding']: {'50000': specs}}}

    def identity(config):
        _, seed, block = outer_indices(config, case, 0, 50000)
        return {**case_binding({**plan, 'config': config}, case),
                'sampling_seed': seed, 'block': block, 'trial_id': 0, 'sample_budget': 50000}

    original = _trial_record(identity(source), specs)
    atomic_json(tmp_path/'manifest.json', plan)
    atomic_json(tmp_path/'cases'/case['case_id']/'n50000'/'trial_000.json', original)
    reused = reuse_trial(target, case, 0, 50000, identity(target), target_specs)
    validate_trial(reused, identity(target), target_specs)
    assert [{k: v for k, v in row.items() if k != 'source_record'}
            for row in reused['rows']] == original['rows']
    assert reused['numerical_environment'] == original['numerical_environment']
    assert reused['source_node_seed_phase'] == source['sampling']['node_seed_phase']

    target['sampling']['outer_mode'] = 'development'
    with pytest.raises(ValueError, match='sampling differs'):
        reuse_trial(target, case, 0, 50000, identity(target), target_specs)


def test_proxy_repetitions_and_failure_preserving_medians():
    from gaussian_w2.imagenet.runner import validate_proxy
    config = load_config(ROOT/'configs/paper/imagenet/ablation.json')
    identity = {'case_id': config['cases'][0]['case_id']}
    count = config['proxies']['repetitions']
    center = float(np.median(range(count)))
    record = {'binding': identity, 'proxies': {'plugin': 3., 'fid_infinity': center, 'rtd': center},
              'fid_infinity_runs': [{'repetition': r, 'estimate': float(r)} for r in range(count)],
              'rtd_runs': [{'binding': {'repetition': r}, 'status': 'ok',
                            'estimate': float(r), 'runtime_seconds': 1.} for r in range(count)]}
    validate_proxy(record, config, identity)
    incomplete = copy.deepcopy(record)
    incomplete['fid_infinity_runs'].pop()
    with pytest.raises(ValueError, match='repetitions'):
        validate_proxy(incomplete, config, identity)
    nonfinite = copy.deepcopy(record)
    nonfinite['fid_infinity_runs'][0]['estimate'] = float('nan')
    with pytest.raises(ValueError, match='nonfinite'):
        validate_proxy(nonfinite, config, identity)
    record['rtd_runs'][0].update(status='failed', estimate=None, runtime_seconds=None,
                                 failure_type='FloatingPointError', error_message='nonfinite estimate')
    with pytest.raises(ValueError, match='centers differ'):
        validate_proxy(record, config, identity)
    record['proxies']['rtd'] = None
    validate_proxy(record, config, identity)


class _NodeBackend:
    """Record node evaluations without constructing images or feature pools."""
    def __init__(self):
        self.samples = []

    def subset(self, data, indices):
        return data[indices]

    def call(self, data):
        self.samples.append(data.copy())
        return {'status': 'ok', 'estimate': float(data.sum()), 'runtime_seconds': 1.0}


def test_node_computation_shares_duplicate_sizes_and_preserves_sampling():
    from gaussian_w2.imagenet.runner import compute_nodes
    case = load_config(ROOT/'configs/paper/imagenet/ablation.json')['cases'][0]
    data = np.arange(24).reshape(12, 2)
    backend = _NodeBackend()
    points = compute_nodes(backend, data, [8, 4, 12, 4], case, 'ablation_nodes', 2)
    assert list(points) == [4, 8, 12]
    assert len(backend.samples) == 3
    for n, observed in zip(points, backend.samples):
        indices, seed = node_indices(case['master_seed'], case['generator'], 'ablation_nodes', 2, n, len(data))
        np.testing.assert_array_equal(observed, data if indices is None else data[indices])
        assert points[n]['sampling_seed'] == seed
        assert points[n]['n'] == n and points[n]['pool_size'] == len(data)
        assert points[n]['estimate'] == float(observed.sum())


def test_proxy_repetitions_recompute_nodes_and_share_only_endpoint():
    from gaussian_w2.imagenet.runner import compute_nodes
    case = load_config(ROOT/'configs/paper/imagenet/ablation.json')['cases'][0]
    data = np.arange(40).reshape(20, 2)
    backend = _NodeBackend()
    endpoint = {'status': 'ok', 'estimate': 999.0, 'runtime_seconds': 2.0}
    original = endpoint.copy()
    results = [compute_nodes(backend, data, [6, 20], case, 'fid_infinity', repetition, endpoint)
               for repetition in range(2)]
    assert len(backend.samples) == 2
    assert endpoint == original
    for repetition, points in enumerate(results):
        indices, seed = node_indices(case['master_seed'], case['generator'], 'fid_infinity', repetition, 6, len(data))
        np.testing.assert_array_equal(backend.samples[repetition], data[indices])
        assert points[6]['sampling_seed'] == seed
        assert points[6]['estimate'] == float(data[indices].sum())
        assert points[20]['estimate'] == endpoint['estimate']
        assert points[20]['runtime_seconds'] == endpoint['runtime_seconds']
    assert results[0][6]['sampling_seed'] != results[1][6]['sampling_seed']
    assert not np.array_equal(backend.samples[0], backend.samples[1])

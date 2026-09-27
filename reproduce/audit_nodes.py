"""Rebuild archived extrapolation estimates from the bundled plug-in nodes."""
from __future__ import annotations

import json
from pathlib import Path


from gaussian_w2.estimators.extrapolation import (
    apply_extrapolation_weights, extrapolation_sample_sizes,
    inverse_sample_extrapolation_weights,
)
from .metrics import DATA, read, read_json


def audit_nodes(data=DATA):
    reports=[]
    for stage in ['development','heldout','budget']:
        nodes=read(stage+'_nodes',data)
        trials=read(stage+'_trials',data)
        if stage=='budget':
            configs=read_json('development_plan',data)['configurations']
            keys=['case_id','max_samples','trial_id']
        else:
            configs=read_json(stage+'_plan',data)['configurations']
            keys=['case_id','trial_id']
        configs={r['configuration_id']:r for r in configs}
        node_groups={key:group.set_index('sample_size').estimate for key,group in nodes.groupby(keys)}
        differences=[]; low_order=[]; rebuilt=0
        for key, group in trials.groupby(keys):
            curve=node_groups[key]
            for row in group.itertuples():
                identifier=row.configuration_id
                if identifier.startswith('rtd'):continue
                spec=configs[identifier]
                if identifier=='empirical_plugin':
                    n=int(row.max_samples) if stage=='budget' else 50000
                    estimate=float(curve[n])
                else:
                    if stage=='budget':
                        # Recompute nodes/weights from the development-selected
                        # schedule while changing only its upper sample budget.
                        sizes=extrapolation_sample_sizes(int(row.max_samples),minimum_samples=5000,
                                                        num_points=int(spec['num_points']),schedule=spec['sample_schedule'])
                        weights=inverse_sample_extrapolation_weights(sizes,order=int(spec['order']),weighting=spec['weighting'])['weights']
                    else:
                        sizes=spec['sample_sizes'];weights=spec['weights']
                    estimate=apply_extrapolation_weights([curve[n] for n in sizes],weights)
                difference=abs(estimate-row.estimate)
                differences.append(difference)
                if spec.get('order') is None or spec.get('order',0)<=4:low_order.append(difference)
                rebuilt+=1
        maximum=max(differences)
        if maximum>1e-5 or max(low_order)>1e-8:
            raise ValueError(f'{stage} node reconstruction disagrees with archived estimates')
        reports.append({'dataset':stage,'estimates_rebuilt':rebuilt,'max_absolute_difference':maximum,
                        'max_difference_orders_1_to_4_and_empirical':max(low_order)})
    nodes=read('gaussian_nodes',data)
    diffs=[];rebuilt=0;cache={}
    for stage in ['gaussian_isotropic','gaussian_random','gaussian_stress']:
        trials=read(stage,data)
        keys=['family','case_label','dimension','N','instance_id','trial_id']
        groups={key:group.sort_values('sample_size') for key,group in nodes[nodes.dataset.eq(stage)].groupby(keys)}
        for key,group in trials.groupby(keys):
            curve=groups[key]; sizes=tuple(int(x) for x in curve.sample_size)
            for row in group.itertuples():
                if row.method=='general_adaptive':continue
                if row.method=='empirical_plugin':estimate=float(curve.estimate.iloc[-1])
                else:
                    order=int(row.method[-1]);weight_key=(sizes,order)
                    if weight_key not in cache:cache[weight_key]=inverse_sample_extrapolation_weights(sizes,order=order)['weights']
                    estimate=apply_extrapolation_weights(curve.estimate,cache[weight_key])
                diffs.append(abs(estimate-row.estimate_w2_sq_raw));rebuilt+=1
    if max(diffs)>1e-8:raise ValueError('Gaussian node reconstruction disagrees with archived estimates')
    reports.append({'dataset':'gaussian','estimates_rebuilt':rebuilt,'max_absolute_difference':max(diffs)})
    nodes=read('proxy_nodes',data)
    targets=read('proxy_runs',data).query("proxy == 'fid_infinity'").set_index(['case_id','repetition']).estimate
    diffs=[]
    for key,group in nodes.groupby(['case_id','repetition']):
        group=group.sort_values('sample_size')
        weights=inverse_sample_extrapolation_weights(group.sample_size,order=1)['weights']
        diffs.append(abs(apply_extrapolation_weights(group.estimate,weights)-targets[key]))
    if max(diffs)>1e-9:raise ValueError('Full-pool FID-infinity proxy reconstruction disagrees')
    reports.append({'dataset':'proxy','estimates_rebuilt':len(diffs),'max_absolute_difference':max(diffs)})
    return reports


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('outputs/paper/node_audit.json'))
    args=parser.parse_args()
    reports=audit_nodes()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(reports,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(reports,indent=2))

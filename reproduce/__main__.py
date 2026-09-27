"""Reconstruct all experimental paper figures and tables on a CPU."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .metrics import (DATA, development_statistics, gaussian_statistics,
                      heldout_statistics, read, read_json, runtime_statistics,
                      selected_heldout, proxy_trend_statistics, intro_statistics)
from .tables import all_tables


def reproduce(output, *, data=DATA, figures=True):
    output=Path(output)
    if output.resolve()==Path(data).resolve() or Path(data).resolve() in output.resolve().parents:
        raise ValueError('Outputs must not overwrite bundled data')
    folders={name:output/name for name in ['figures','tables','analysis']}
    for folder in folders.values():folder.mkdir(parents=True,exist_ok=True)
    plan=read_json('development_plan',data)
    proxies=read('proxies',data)

    iso=read('gaussian_isotropic',data)
    random_cases,dimension=gaussian_statistics(read('gaussian_random',data))
    stress_cases,stress=gaussian_statistics(read('gaussian_stress',data))
    random_cases.to_csv(folders['analysis']/'gaussian_random_per_case.csv',index=False)
    stress_cases.to_csv(folders['analysis']/'gaussian_stress_per_case.csv',index=False)
    dimension.to_csv(folders['analysis']/'gaussian_dimension_aggregation.csv',index=False)
    stress.to_csv(folders['analysis']/'gaussian_stress_aggregation.csv',index=False)
    dev=read('development_trials',data)
    per_case,summary,ranking,chosen=development_statistics(dev,proxies,plan)
    per_case.to_csv(folders['analysis']/'development_per_case.csv',index=False)
    summary.to_csv(folders['analysis']/'development_by_embedding.csv',index=False)
    ranking.to_csv(folders['analysis']/'development_ranking.csv',index=False)
    chosen.to_json(folders['analysis']/'selected_configurations.json',orient='records',indent=2)
    full=read('heldout_trials',data)
    selected=selected_heldout(full,chosen)
    heldout_cases,heldout=heldout_statistics(selected,proxies)
    runtime=runtime_statistics(selected)
    b=read('budget_trials',data)
    b['measurement_source']='budget_trials'
    if not set(b.max_samples)=={10000,20000,30000,40000}:
        raise ValueError('The archived 50K budget rows must not replace final held-out measurements')
    if set(b.method)!=set(selected.method):
        raise ValueError('Budget and held-out methods differ')
    budget_raw=pd.concat([b,selected],ignore_index=True)
    budget_cases,budget=heldout_statistics(budget_raw,proxies)
    selected.to_csv(folders['analysis']/'selected_heldout_trials.csv',index=False)
    budget_raw.to_csv(folders['analysis']/'sample_budget_trials.csv',index=False)
    heldout_cases.to_csv(folders['analysis']/'heldout_per_case.csv',index=False)
    budget_cases.to_csv(folders['analysis']/'sample_budget_per_case.csv',index=False)
    trend = proxy_trend_statistics(read('proxy_trend_trials', data))
    trend.to_csv(folders['analysis']/'proxy_trend_summary.csv', index=False)
    tables=all_tables(folders['tables'],iso,proxies,per_case,summary,chosen,plan,heldout,runtime,budget,trend)
    print(f"Recomputed {len(tables)} tables and independently reselected VALE rules.",flush=True)
    if figures:
        from .figures import (CONDITION, RANDOM, RANK, order_scatter_figure, plot_grid,
                              intro_figure, proxy_figures)
        for family, title, stem in zip(
                RANDOM, ['Haar log-uniform', 'Spiked bulk', 'Rotated Toeplitz'],
                ['haar', 'spiked_bulk', 'rotated_toeplitz']):
            plot_grid(dimension[dimension.family.eq(family)], 'family', [family], [title],
                      folders['figures']/f'gaussian_dimension_{stem}.pdf', reference=True)
        gaussian, imagenet = intro_statistics(read('gaussian_random', data), read('gaussian_nodes', data),
                                               trend, read('proxy_trend_nodes', data))
        gaussian.to_csv(folders['analysis']/'intro_gaussian.csv', index=False)
        imagenet.to_csv(folders['analysis']/'intro_imagenet.csv', index=False)
        intro_figure(gaussian, imagenet, folders['figures']/'intro_gaussian_and_imagenet.pdf')
        proxy_figures(trend[trend.sample_budget.ne(50000)], folders['figures'])
        plot_grid(stress[stress.family.eq('rank_deficient')],'case_label',RANK,[r'$q=0.25$',r'$q=0.50$',r'$q=0.75$'],
                  folders['figures']/'gaussian_rank_error.pdf')
        plot_grid(stress[stress.family.eq('ill_conditioned')],'case_label',CONDITION,[r'$\kappa=10^2$',r'$\kappa=10^4$',r'$\kappa=10^6$'],
                  folders['figures']/'gaussian_condition_error.pdf')
        order_scatter_figure(per_case,folders['figures'],'rtd',max_order=6)
        print('Rendered all 11 paper figures.',flush=True)
    print(f"Output: {output}",flush=True)
    return tables


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('outputs/paper'))
    parser.add_argument('--data',type=Path,default=DATA,help='Directory of complete bundled publication records')
    parser.add_argument('--no-figures',action='store_true',help='Recompute tables and statistics without plotting')
    args=parser.parse_args()
    reproduce(args.output,data=args.data,figures=not args.no_figures)


if __name__=='__main__':
    main()

"""Summarize completed ImageNet proxy trials and plot medians over sample size."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROXIES = ('plugin', 'fid_infinity', 'rtd')
KEY_COLUMNS = ('sample_budget', 'proxy', 'repetition')
REQUIRED = ('case_id', 'generator', 'embedding', 'dimension', *KEY_COLUMNS,
            'status', 'estimate', 'runtime_seconds', 'failure_type', 'error_message')
SUMMARY_COLUMNS = ('case_id', 'generator', 'embedding', 'dimension', 'sample_budget',
                   'proxy', 'median_estimate', 'n_ok', 'n_failed', 'n_attempted', 'available')
EMBEDDINGS = {'fid': 'Inception', 'fd_dinov2': 'DINOv2', 'clip': 'CLIP'}
GENERATOR_LABELS = {
    'stylegan_xl_imagenet64': 'StyleGAN-XL (64)',
    'stylegan_xl_imagenet256': 'StyleGAN-XL (256)',
    'stylegan_xl_imagenet512': 'StyleGAN-XL (512)',
    'ddo_edm2_s_imagenet64': 'DDO EDM2-S (64)',
    'ddo_edm2_l_imagenet512': 'DDO EDM2-L (512)',
    'var_d30_imagenet256': 'VAR-d30 (256)',
    'var_d36_imagenet512': 'VAR-d36 (512)',
}
STYLES = {'plugin': ('Empirical', '#55616C', 's'),
          'fid_infinity': (r'OLS$_1$', '#0072B2', '^'),
          'rtd': ('RTD', '#009E73', 'o')}


def summarize(run):
    """Return one median per case, sample budget and proxy from completed CSVs.

    A completed case must contain every configured repetition. Any failed or
    nonfinite repetition leaves its median unavailable instead of being dropped.
    Cases with no completed CSV are omitted, so a running study can be reported.
    """
    run = Path(run)
    manifest = json.loads((run/'manifest.json').read_text(encoding='utf-8'))
    if manifest['schema'] != 'imagenet-proxy-trend-v1':
        raise ValueError('expected an imagenet-proxy-trend-v1 manifest')
    config = manifest['config']
    budgets = manifest['sample_budgets']
    repetitions = config['proxies']['repetitions']
    cases = {case['case_id']: case for case in config['cases']}
    chosen = set(manifest['case_ids'])
    if chosen - set(cases):
        raise ValueError('manifest case_ids contains unknown cases')
    expected = {(n, proxy, repetition) for n in budgets for proxy in PROXIES
                for repetition in range(1 if proxy == 'plugin' else repetitions)}
    results = []
    paths = sorted(run.glob('trials.*.csv'),
                   key=lambda path: list(cases).index(path.name[len('trials.'):-len('.csv')])
                   if path.name[len('trials.'):-len('.csv')] in cases else len(cases))
    for path in paths:
        case_id = path.name[len('trials.'):-len('.csv')]
        if case_id not in chosen:
            raise ValueError(f'{path.name} is outside the selected cases')
        case = cases[case_id]
        rows = pd.read_csv(path, float_precision='round_trip')
        missing_columns = set(REQUIRED) - set(rows.columns)
        if missing_columns:
            raise ValueError(f'{path.name} is missing columns: {sorted(missing_columns)}')
        for column in ('dimension', 'sample_budget', 'repetition'):
            values = pd.to_numeric(rows[column], errors='coerce')
            if not np.isfinite(values).all() or (values != np.floor(values)).any():
                raise ValueError(f'{path.name}: {column} must contain finite integers')
            rows[column] = values.astype('int64')
        for column in ('case_id', 'generator', 'embedding', 'dimension'):
            if not rows[column].eq(case[column]).all():
                raise ValueError(f'{path.name}: {column} differs from the manifest')
        if rows.duplicated(list(KEY_COLUMNS)).any():
            raise ValueError(f'{path.name} contains duplicate trial rows')
        observed = set(rows.loc[:, list(KEY_COLUMNS)].itertuples(index=False, name=None))
        if observed != expected:
            raise ValueError(f'{path.name} has missing or unexpected budget/proxy/repetition rows '
                             f'({len(expected-observed)} missing, {len(observed-expected)} unexpected)')
        for n in budgets:
            for proxy in PROXIES:
                cell = rows[rows['sample_budget'].eq(n) & rows['proxy'].eq(proxy)]
                values = pd.to_numeric(cell['estimate'], errors='coerce').to_numpy(dtype=float)
                successful = cell['status'].eq('ok').to_numpy() & np.isfinite(values)
                n_ok, n_attempted = int(successful.sum()), len(cell)
                available = n_ok == n_attempted
                results.append({column: case[column] for column in
                                ('case_id', 'generator', 'embedding', 'dimension')}
                               | {'sample_budget': n, 'proxy': proxy,
                                  'median_estimate': float(np.median(values)) if available else np.nan,
                                  'n_ok': n_ok, 'n_failed': n_attempted-n_ok,
                                  'n_attempted': n_attempted, 'available': available})
    return pd.DataFrame(results, columns=SUMMARY_COLUMNS)


def summarize_runs(runs):
    """Combine disjoint budgets measured with the same study settings."""
    summaries = []
    baseline = None
    for run in runs:
        config = json.loads((Path(run)/'manifest.json').read_text(encoding='utf-8'))['config']
        config['sampling'].pop('sample_budgets', None)
        if baseline is not None and config != baseline:
            raise ValueError('proxy-trend runs have different study settings')
        baseline = config
        summaries.append(summarize(run))
    summary = pd.concat(summaries, ignore_index=True)
    if summary.duplicated(['case_id', 'sample_budget', 'proxy']).any():
        raise ValueError('proxy-trend runs contain overlapping case/budget/proxy cells')
    return summary


def _label(proxy, embedding=None):
    label = STYLES[proxy][0]
    return f'FID-{label}' if embedding == 'fid' and proxy != 'plugin' else label


def _draw_case(ax, rows, budgets, embedding=None):
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    for proxy in PROXIES:
        _, color, marker = STYLES[proxy]
        label = _label(proxy, embedding)
        values = rows[rows['proxy'].eq(proxy)].set_index('sample_budget')['median_estimate']
        # Keep NaNs in the sequence so failed cells produce visible line gaps.
        ax.plot(budgets, values.reindex(budgets), label=label, color=color,
                marker=marker, markersize=4, linewidth=1.5)
    if 50000 in budgets:
        ax.set_xticks([50000] + [n for n in (100000, 200000, 300000)
                                  if min(budgets) <= n <= max(budgets)])
    else:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, position: f'{value/1000:g}'))
    ax.tick_params(labelsize=9)
    ax.grid(axis='both', alpha=0.18, linewidth=0.7)
    ax.spines[['top', 'right']].set_visible(False)
    ax.ticklabel_format(axis='y', style='plain', useOffset=False)
    ax.margins(x=0.04)


def _finish_figure(fig, output, name, title, embedding=None):
    from matplotlib.lines import Line2D

    handles = [Line2D([], [], color=color, marker=marker, label=_label(proxy, embedding),
                      linewidth=1.5, markersize=4)
               for proxy, (_, color, marker) in STYLES.items()]
    height = fig.get_figheight()
    fig.get_layout_engine().set(rect=(0, 0, 1, 1-0.7/height))
    heading = fig.suptitle(title, fontsize=14, y=1-0.04/height)
    heading.set_in_layout(False)
    xlabel = fig.supxlabel('Sample count (K)', fontsize=11)
    ylabel = fig.supylabel('FID estimate' if embedding == 'fid' else 'FD estimate', fontsize=11)
    legend = fig.legend(handles=handles, loc='upper center', ncol=3, frameon=False,
                        bbox_to_anchor=(0.5, 1-0.29/height))
    paths = [output/f'{name}.{extension}' for extension in ('pdf', 'png')]
    for path in paths:
        fig.savefig(path, dpi=180, bbox_inches='tight',
                    bbox_extra_artists=(heading, legend, xlabel, ylabel))
    return paths


def plot(summary, output):
    """Save the generator-by-embedding grid and embedding sheets under output/figures."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if summary.empty:
        raise ValueError('no completed case CSVs to plot')
    cases = summary[['case_id', 'generator', 'embedding']].drop_duplicates()
    if cases.duplicated(['generator', 'embedding']).any():
        raise ValueError('plot requires one case per generator and embedding')
    generators = list(dict.fromkeys(summary['generator']))
    embedding_order = ('fid', 'fd_dinov2', 'clip')
    embeddings = sorted(summary['embedding'].unique(),
                        key=lambda value: (embedding_order.index(value)
                                           if value in embedding_order else len(embedding_order), value))
    budgets = sorted(summary['sample_budget'].unique())
    destination = Path(output)/'figures'
    destination.mkdir(parents=True, exist_ok=True)
    paths = []
    fig, axes = plt.subplots(len(generators), len(embeddings), sharex=True, squeeze=False,
                             figsize=(5*len(embeddings), 2.5*len(generators)+1), layout='constrained')
    for row, generator in enumerate(generators):
        for column, embedding in enumerate(embeddings):
            ax = axes[row, column]
            cell = summary[summary['generator'].eq(generator) & summary['embedding'].eq(embedding)]
            _draw_case(ax, cell, budgets)
            if cell.empty:
                ax.text(0.5, 0.5, 'No completed result', transform=ax.transAxes,
                        ha='center', va='center', color='0.5', fontsize=9)
            if row == 0:
                ax.set_title(EMBEDDINGS.get(embedding, embedding), fontsize=12)
            if column == 0:
                ax.set_ylabel(GENERATOR_LABELS.get(generator, generator), fontsize=10)
    paths.extend(_finish_figure(fig, destination, 'proxy_trend', 'FD estimates across sample sizes'))
    plt.close(fig)

    if len(embeddings) > 1:
        for embedding in embeddings:
            data = summary[summary['embedding'].eq(embedding)]
            selected_generators = [g for g in generators if g in set(data['generator'])]
            columns = min(2, len(selected_generators))
            rows = (len(selected_generators)+columns-1)//columns
            fig, axes = plt.subplots(rows, columns, sharex=True, squeeze=False,
                                     figsize=(5.5*columns, 2.7*rows+1), layout='constrained')
            for index, (ax, generator) in enumerate(zip(axes.flat, selected_generators)):
                _draw_case(ax, data[data['generator'].eq(generator)], budgets, embedding)
                ax.set_title(GENERATOR_LABELS.get(generator, generator), fontsize=11)
                if index+columns >= len(selected_generators):
                    ax.tick_params(labelbottom=True)
            for ax in list(axes.flat)[len(selected_generators):]:
                ax.set_visible(False)
            paths.extend(_finish_figure(fig, destination, f'proxy_trend_{embedding}',
                                       f'{EMBEDDINGS.get(embedding, embedding)}: FD estimates across sample sizes',
                                       embedding))
            plt.close(fig)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, help='proxy-trend run containing manifest.json and completed case CSVs')
    parser.add_argument('--additional-run', action='append', default=[],
                        help='supplemental budgets to combine with RUN; can be repeated')
    parser.add_argument('--output', help='report directory (default: RUN)')
    args = parser.parse_args()
    output = Path(args.output or args.run)
    summary = summarize_runs([args.run, *args.additional_run])
    if summary.empty:
        parser.error('no completed case CSVs to report')
    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output/'summary.csv', index=False, float_format='%.17g', na_rep='')
    paths = plot(summary, output)
    print(f'Wrote {len(summary)} summary cells and {len(paths)} figure files to {output}.')
    print(f'{int(summary.n_failed.sum())} failed or nonfinite repetitions; '
          f'{int((~summary.available).sum())} unavailable plot points.')


if __name__ == '__main__':
    main()

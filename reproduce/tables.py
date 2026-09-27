"""Render every experimental table and retain full precision CSV values."""
from __future__ import annotations

from html import escape
import re

import pandas as pd

from gaussian_w2.evaluation import across_cases, trial_metrics
from .metrics import EMBEDDINGS, METHODS

LABELS = {"Empirical": "Empirical", "FID_infinity": r"FID$_\infty$", "VA-1": r"VALE$_1$", "VA-2": r"VALE$_2$", "VA-3": r"VALE$_3$", "RTD": "RTD"}
SCHEDULES = [("uniform_n", r"Uniform in $n$"), ("uniform_inverse_n", r"Uniform in $1/n$"), ("chebyshev_inverse_n", r"Chebyshev in $1/n$")]
MODELS = [
    ("stylegan_xl_imagenet64", "StyleGAN-XL", 64, "1.51"),
    ("ddo_edm2_s_imagenet64", "DDO/EDM2-S", 64, "0.97"),
    ("stylegan_xl_imagenet256", "StyleGAN-XL", 256, "2.30"),
    ("var_d30_imagenet256", "VAR-d30", 256, "1.73"),
    ("stylegan_xl_imagenet512", "StyleGAN-XL", 512, "2.41"),
    ("ddo_edm2_l_imagenet512", "DDO/EDM2-L", 512, "1.21"),
    ("var_d36_imagenet512", "VAR-d36", 512, "2.63"),
]


def cell(mean, sd, *, digits=4, bold=False):
    text = f"{mean:.{digits}f}\\pm{sd:.{digits}f}"
    return "$" + (r"\mathbf{" + text + "}" if bold else text) + "$"


def table(out, stem, label, headers, rows, caption, alignment=None, font_size="small", column_sep=None):
    rendered = []
    for row in rows:
        rendered.append(row if isinstance(row, str) else " & ".join(map(str, row)) + r" \\")
    spacing = [] if column_sep is None else [rf"\setlength{{\tabcolsep}}{{{column_sep}pt}}"]
    tex = "\n".join([r"\begin{table}[h]", r"\centering", "\\" + font_size, *spacing,
                     rf"\caption{{{caption}}}", rf"\label{{{label}}}",
                     r"\begin{tabular}{@{}" + (alignment or "l" + "r"*(len(headers)-1)) + "@{}}",
                     r"\toprule", " & ".join(headers) + r" \\", r"\midrule", *rendered,
                     r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    (out / f"{stem}.tex").write_text(tex, encoding="utf-8")
    (out / f"{stem}.html").write_text(table_html(headers, rows, caption), encoding="utf-8")
    return label, tex



def html_text(value):
    """Render the small set of inline LaTeX forms used by these tables."""
    text = escape(str(value)).replace("$", "").replace("{,}", ",")
    for command, replacement in [(r"\pm", " ± "), (r"\infty", "∞"),
                                 (r"\ell", "ℓ"), (r"\%", "%"),
                                 (r"\downarrow", "↓"), (r"\;", " ")]:
        text = text.replace(command, replacement)
    text = re.sub(r"\\(?:mathbf|textbf)\{([^{}]*)\}", r"<strong>\1</strong>", text)
    return re.sub(r"_\{([^{}]*)\}|_([A-Za-z0-9∞])",
                  lambda match: f"<sub>{match[1] or match[2]}</sub>", text)


def table_html(headers, rows, caption):
    """Build notebook-ready HTML from the same formatted cells as the LaTeX."""
    lines = ['<table>', f'<caption>{html_text(caption)}</caption>', '<thead><tr>']
    lines.extend(f'<th scope="col">{html_text(header)}</th>' for header in headers)
    lines.extend(['</tr></thead>', '<tbody>'])
    for row in rows:
        if isinstance(row, str):
            # The budget table separates embeddings with a full-width heading.
            section = re.fullmatch(r"\\multicolumn\{(\d+)\}\{[^}]*\}\{(.*)\}\s*\\\\", row)
            if section is None:
                raise ValueError(f"Unsupported table separator: {row}")
            lines.append(f'<tr><th colspan="{section[1]}" scope="rowgroup" style="text-align:left">'
                         f'{html_text(section[2])}</th></tr>')
        else:
            lines.append('<tr>' + ''.join(f'<td>{html_text(value)}</td>' for value in row) + '</tr>')
    lines.extend(['</tbody>', '</table>', ''])
    return '\n'.join(lines)


def isotropic_table(out, iso):
    methods = ["empirical_plugin", "fid_infinity_ols_order1", "fid_infinity_ols_order2",
               "fid_infinity_ols_order3", "general_adaptive"]
    stats = {method: trial_metrics(iso[iso.method.eq(method)].estimate_w2_sq_raw, 5.)
             for method in methods}
    pd.DataFrame([{"method": method, **stats[method]} for method in methods]).to_csv(
        out / "isotropic.csv", index=False)
    rows = [[label] + [f"{stats[method][metric]:.4f}" for method in methods]
            for label, metric in [("Mean estimate", "mean_estimate"),
                                  ("Signed error", "signed_error"),
                                  ("Standard Deviation", "sd")]]
    return table(out, "isotropic", "tab:hard-isotropic",
                 ["", "Empirical", r"OLS$_1$", r"OLS$_2$", r"OLS$_3$", "RTD"], rows,
                 "Isotropic experiment with $D_0=5$, $d_0=5000$, and $N=100{,}000$.")


def checkpoint_proxy_table(out, proxies):
    values = proxies[proxies.proxy.eq("rtd")].set_index(["generator", "embedding"]).reference_value
    rows = [[label, resolution] + [f"{values[generator, embedding]:.3f}" for embedding, _ in EMBEDDINGS]
            + [published] for generator, label, resolution, published in MODELS]
    records = [{"generator": generator, "resolution": resolution, "published_fid": published,
                **{embedding: values[generator, embedding] for embedding, _ in EMBEDDINGS}}
               for generator, label, resolution, published in MODELS]
    pd.DataFrame(records).to_csv(out / "checkpoint_proxies.csv", index=False)
    return table(out, "checkpoint_proxies", "tab:imagenet-model-detail",
                 ["Model", "Resolution", "Inception $Q_c$", "DINOv2 $Q_c$", "CLIP $Q_c$", "Published FID"],
                 rows, "Model checkpoints and full-300K RTD proxy values.")


def matched_ols_va_table(out, per_case):
    rows, records = [], []
    for embedding, display in EMBEDDINGS:
        for order in range(1, 5):
            group = per_case[per_case.embedding.eq(embedding) & per_case.order.eq(order)
                             & per_case.sample_schedule.eq("uniform_n") & per_case.num_points.eq(15)]
            ols = group[group.weighting.eq("ordinary_ols")].set_index("case_id")
            va = group[group.weighting.eq("variance_aware")].set_index("case_id").reindex(ols.index)
            record = {"embedding": embedding, "order": order}
            cells = []
            for name, frame, metric in [("ols_sd", ols, "trial_sd"), ("va_sd", va, "trial_sd"),
                                        ("ols_rmse", ols, "rmse"), ("va_rmse", va, "rmse")]:
                stats = across_cases(frame[metric])
                record[name] = stats['mean']
                record[name + "_sd"] = stats['sd']
                cells.append(cell(stats['mean'], stats['sd']))
            drop = across_cases(100 * (1 - va.rmse / ols.rmse))['mean']
            wins = int((va.rmse < ols.rmse).sum())
            record.update(rmse_drop_percent=drop, wins=wins)
            records.append(record)
            rows.append([display if order == 1 else "", order, *cells, f"{drop:.2f}", f"{wins}/7"])
    pd.DataFrame(records).to_csv(out / "matched_ols_va.csv", index=False)
    return table(out, "matched_ols_va", "tab:imagenet-VA-OLS",
                 ["Embedding", "Order $k$", r"OLS$_k$ SD", r"VALE$_k$ SD", r"OLS$_k$ RMSE", r"VALE$_k$ RMSE", r"RMSE drop (\%)", "Win rate"],
                 rows, "Matched OLS$_k$--VALE$_k$ comparison. RMSE drop is the mean casewise percentage reduction; win rate counts cases with lower RMSE.")


def selected_strategy_table(out, chosen):
    rows = []
    for embedding, display in EMBEDDINGS:
        for order in (1, 2, 3):
            result = chosen[chosen.embedding.eq(embedding) & chosen.order.eq(order)].iloc[0]
            schedule = ("Endpoints" if order == 1 and result.num_points == 2
                        else dict(SCHEDULES)[result.sample_schedule])
            rows.append([display if order == 1 else "", order, schedule, int(result.num_points),
                         cell(result.mean_rmse, result.std_rmse)])
    chosen.to_csv(out / "selected_strategies.csv", index=False)
    return table(out, "selected_strategies", "tab:imagenet-optimal-schedule",
                 ["Embedding", "Order $k$", "Schedule", r"Points $\ell$", "RMSE"],
                 rows, "Selected VALE$_k$ configurations.")


def va_grid_tables(out, summary, chosen, plan):
    lookup = summary.set_index(["embedding", "configuration_id"])
    best = chosen.set_index(["embedding", "order"]).configuration_id
    records, tables = [], {}
    for embedding, display in EMBEDDINGS:
        rows = []
        for order in range(1, 5):
            for points in (order + 1, 10, 15, 20):
                cells = []
                for schedule, _ in SCHEDULES:
                    alias = next(alias for alias in plan['strategy_grid']
                                 if alias['order'] == order and alias['num_points'] == points
                                 and alias['sample_schedule'] == schedule)
                    identifier = alias['canonical_configuration_id']
                    result = lookup.loc[embedding, identifier]
                    cells.append(cell(result.mean_rmse, result.std_rmse,
                                      bold=identifier == best[embedding, order]))
                    records.append({'embedding': embedding, 'order': order, 'num_points': points,
                                    'sample_schedule': schedule, 'configuration_id': identifier,
                                    'mean_rmse': result.mean_rmse, 'sd_rmse': result.std_rmse})
                rows.append([order, points, *cells])
        suffix = {'fid': '', 'fd_dinov2': '-dinov2', 'clip': '-clip'}[embedding]
        label, tex = table(out, f"va_grid_{embedding}", "tab:imagenet-va-full-grid" + suffix,
                           ["Order $k$", r"Points $\ell$", r"Uniform $n$", r"Uniform $1/n$", r"Chebyshev $1/n$"],
                           rows, f"VALE$_k$ regression-point ablation for {display}.")
        tables[label] = tex
    pd.DataFrame(records).to_csv(out / "va_grid.csv", index=False)
    return tables


def runtime_table(out, runtime):
    lookup = runtime.set_index(['embedding', 'method'])
    rows = []
    for embedding, display in EMBEDDINGS:
        cells = [cell(1000 * lookup.loc[(embedding, method), 'mean_runtime'],
                      1000 * lookup.loc[(embedding, method), 'sd_runtime'], digits=1) for method in METHODS]
        rows.append([display, *cells])
    runtime.to_csv(out / 'runtime.csv', index=False)
    return table(out, 'runtime', 'tab:imagenet-computation-time',
                 ['Embedding'] + [LABELS[method] for method in METHODS], rows,
                 r'Recorded estimator runtime (milliseconds) for a 50K trial. Mean $\pm$ sample SD over all 35 held-out runs.',
                 font_size='footnotesize', column_sep=3.5)


def proxy_sensitivity_table(out, heldout):
    value, spread = 'mean_median_absolute_error', 'std_median_absolute_error'
    lookup = heldout.set_index(['embedding', 'method', 'proxy'])
    proxies = ['plugin', 'fid_infinity', 'rtd']
    rows = []
    for embedding, display in EMBEDDINGS:
        best = {proxy: min(METHODS, key=lambda method: lookup.loc[(embedding, method, proxy), value])
                for proxy in proxies}
        for method in METHODS:
            cells = [cell(lookup.loc[(embedding, method, proxy), value],
                          lookup.loc[(embedding, method, proxy), spread], bold=best[proxy] == method)
                     for proxy in proxies]
            rows.append([display if method == 'Empirical' else '', LABELS[method], *cells])
    heldout.to_csv(out / 'heldout.csv', index=False)
    return table(out, 'proxy_sensitivity', 'tab:imagenet-proxy-median-error',
                 ['Embedding', 'Estimator', 'Empirical proxy', r'FID$_\infty$ proxy', 'RTD proxy'],
                 rows, 'Held-out utility and sensitivity to the proxy at $N=50$K. '
                 r'Each entry is the mean $\pm$ sample SD across seven checkpoints of '
                 'the median absolute error over five held-out trials.')



def ablation_summary_table(out, summary, chosen):
    """Average full-precision per-order RMSE means for the paper ablation."""
    matched = summary[summary.order.isin([1, 2, 3])
                      & summary.sample_schedule.eq("uniform_n") & summary.num_points.eq(15)]
    records, values = [], {}
    for embedding, _ in EMBEDDINGS:
        stages = [matched[matched.embedding.eq(embedding) & matched.weighting.eq(weighting)]
                  for weighting in ("ordinary_ols", "variance_aware")]
        stages.append(chosen[chosen.embedding.eq(embedding) & chosen.order.le(3)])
        previous = None
        for stage, frame in zip(("ols", "vale_weighting", "selected_strategy"), stages):
            mean = frame.mean_rmse.mean()
            drop = None if previous is None else 100 * (1 - mean / previous)
            values[embedding, stage] = (mean, drop)
            records.append({"embedding": embedding, "stage": stage, "mean_rmse": mean,
                            "rmse_drop_percent": drop})
            previous = mean
    rows = []
    for stage, label in [("ols", r"FID$_\infty$, OLS$_2$, OLS$_3$"),
                         ("vale_weighting", r"$+$ VALE weighting"),
                         ("selected_strategy", r"$+$ Regression strategy optimization")]:
        cells = []
        for embedding, _ in EMBEDDINGS:
            mean, drop = values[embedding, stage]
            text = f"{mean:.4f}"
            if stage == "selected_strategy":
                text = r"\mathbf{" + text + "}"
            if drop is not None:
                text += rf"\;(\downarrow{drop:.1f}\%)"
            cells.append("$" + text + "$")
        rows.append([label, *cells])
    pd.DataFrame(records).to_csv(out / "ablation_summary.csv", index=False)
    return table(out, "ablation_summary", "tab:imagenet-ablation-summary",
                 ["Configuration", "Inception", "DINOv2", "CLIP"], rows,
                 "Successive ablations on 50K development-pool trials. RMSE is averaged "
                 "over seven checkpoints and orders $k=1,2,3$; parentheses show reductions "
                 "from the preceding row.")


def proxy_agreement_table(out, trend):
    """Compare each proxy with RTD at the same budget, averaging across models."""
    lookup = trend.pivot(index=["embedding", "generator", "sample_budget"],
                         columns="proxy", values="median_estimate")
    rows, records = [], []
    for proxy, budget in [("fid_infinity", 30000), ("fid_infinity", 300000), ("plugin", 300000)]:
        cells = []
        for embedding, _ in EMBEDDINGS:
            group = lookup.xs((embedding, budget), level=("embedding", "sample_budget"))
            gap = (group[proxy] - group["rtd"]).abs().mean()
            records.append({"embedding": embedding, "proxy": proxy,
                            "sample_budget": budget, "mean_absolute_difference": gap})
            cells.append(f"{gap:.4f}")
        rows.append([r"FID$_\infty$" if proxy == "fid_infinity" else "Empirical",
                     f"{budget // 1000}K", *cells])
    pd.DataFrame(records).to_csv(out / "proxy_agreement.csv", index=False)
    return table(out, "proxy_agreement", "tab:imagenet-proxy-agreement",
                 ["Estimator", "$N$", "Inception", "DINOv2", "CLIP"], rows,
                 "Mean absolute differences from RTD at the same sample budget across seven generators.")


def budget_decimal(value):
    """Format a full-precision statistic to three decimals."""
    return f"{value:.3f}"


def sample_budget_tables(out, budget):
    value, spread = 'mean_median_absolute_error', 'std_median_absolute_error'
    lookup = budget[budget.proxy.eq('rtd')].set_index(['embedding', 'max_samples', 'method'])
    tables = {}
    for stem, label, embeddings, caption in [
        ('sample_budget', 'tab:imagenet-fd-sample-budget-heldout', EMBEDDINGS[:1],
         r'Sample-budget sensitivity for Inception. Entries report the mean $\pm$ sample SD '
         'across seven checkpoints of the median absolute error over five held-out trials against the RTD proxy.'),
        ('sample_budget_additional', 'tab:imagenet-sample-budget-additional', EMBEDDINGS[1:],
         'Sample-budget sensitivity for DINOv2 and CLIP, using the same error statistic and RTD reference proxy.')]:
        rows = []
        for embedding, display in embeddings:
            if len(embeddings) > 1:
                rows.append(r'\multicolumn{7}{l}{\textbf{' + display + r'}} \\')
            for sample_size in [10000, 20000, 30000, 40000, 50000]:
                best = min(METHODS, key=lambda method: lookup.loc[(embedding, sample_size, method), value])
                cells = []
                for method in METHODS:
                    result = lookup.loc[(embedding, sample_size, method)]
                    mean = budget_decimal(result[value])
                    if method == best:
                        mean = r"\mathbf{" + mean + "}"
                    cells.append("$" + mean + rf"\pm{budget_decimal(result[spread])}" + "$")
                rows.append([f'{sample_size // 1000}K', *cells])
        key, tex = table(out, stem, label, ['$N$'] + [LABELS[method] for method in METHODS],
                         rows, caption, font_size='footnotesize', column_sep=2)
        tables[key] = tex
    budget.to_csv(out / 'sample_budget.csv', index=False)
    return tables


def all_tables(out, iso, proxies, per_case, summary, chosen, plan, heldout, runtime, budget, trend):
    """Write the thirteen experimental tables, keyed by manuscript label."""
    budget_tables = sample_budget_tables(out, budget)
    tables = dict([
        isotropic_table(out, iso),
        proxy_sensitivity_table(out, heldout),
        ablation_summary_table(out, summary, chosen),
        runtime_table(out, runtime),
        ('tab:imagenet-fd-sample-budget-heldout', budget_tables['tab:imagenet-fd-sample-budget-heldout']),
        checkpoint_proxy_table(out, proxies),
        proxy_agreement_table(out, trend),
        matched_ols_va_table(out, per_case),
        selected_strategy_table(out, chosen),
    ])
    tables.update(va_grid_tables(out, summary, chosen, plan))
    tables['tab:imagenet-sample-budget-additional'] = budget_tables['tab:imagenet-sample-budget-additional']
    return tables

"""Grouping and selection for paper statistics; primitives are shared with runs."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from gaussian_w2.evaluation import across_cases, trial_metrics

DATA = Path(__file__).resolve().parent / "data"
EMBEDDINGS = [("fid", "Inception"), ("fd_dinov2", "DINOv2"), ("clip", "CLIP")]
METHODS = ["Empirical", "FID_infinity", "VA-1", "VA-2", "VA-3", "RTD"]


def read(name, data=DATA):
    path = Path(data) / f"{name}.csv.gz"
    return pd.read_csv(path, float_precision="round_trip")


def read_json(name, data=DATA):
    return json.loads((Path(data) / f"{name}.json").read_text(encoding='utf-8'))


def check_trials(frame, keys, trials):
    if not frame.status.eq("ok").all():
        raise ValueError("Failed records require explicit handling, never deletion")
    if frame.duplicated(keys + ["trial_id"]).any():
        raise ValueError("Duplicate trial record")
    for _, group in frame.groupby(keys, sort=False):
        if len(group) != trials or group.trial_id.nunique() != trials:
            raise ValueError("Incomplete trial cell")


def summarize(frame, keys, metrics, expected_cases=7):
    rows = []
    for values, group in frame.groupby(keys, sort=True):
        values = (values,) if len(keys) == 1 else values
        if len(group) != expected_cases:
            raise ValueError(f"Expected {expected_cases} equally weighted cases")
        row = dict(zip(keys, values))
        for metric in metrics:
            stats = across_cases(group[metric])
            row.update({f"mean_{metric}": stats["mean"], f"std_{metric}": stats["sd"]})
        rows.append(row)
    return pd.DataFrame(rows)


def gaussian_statistics(frame):
    keys = ["family", "case_label", "dimension", "N", "instance_id", "method"]
    check_trials(frame, keys, 5)
    rows = []
    for values, group in frame.groupby(keys, sort=True):
        if group.true_w2_sq.nunique() != 1:
            raise ValueError("Inconsistent population truth")
        stats = trial_metrics(group.estimate_w2_sq_raw, group.true_w2_sq.iloc[0])
        rows.append({**dict(zip(keys, values)), **stats})
    cases = pd.DataFrame(rows)
    outer = summarize(cases, ["family", "case_label", "dimension", "N", "method"],
                      ["median_absolute_error"], expected_cases=10)
    outer = outer.rename(columns={"mean_median_absolute_error": "mean_case_median", "std_median_absolute_error": "sd_case_median"})
    outer["lower_mean_minus_sd"] = outer.mean_case_median - outer.sd_case_median
    outer["upper_mean_plus_sd"] = outer.mean_case_median + outer.sd_case_median
    return cases, outer


def development_statistics(raw, proxies, plan):
    check_trials(raw, ["case_id", "configuration_id"], 10)
    config = {c["configuration_id"]: c for c in plan["configurations"]}
    targets = proxies[proxies.proxy.eq("rtd")].set_index("case_id").reference_value
    rows = []
    for (case, identifier), group in raw.groupby(["case_id", "configuration_id"], sort=True):
        result = trial_metrics(group.estimate, targets[case])
        rows.append({"case_id": case, "generator": group.generator.iloc[0], "embedding": group.embedding.iloc[0],
                     "proxy": "rtd", **{k:v for k,v in config[identifier].items() if k not in ["weights", "sample_sizes"]},
                     **result, "absolute_center_error": result["center_error"], "trial_sd": result["sd"], "trials": result["n_trials"]})
    per_case = pd.DataFrame(rows)
    keys = ["embedding", "configuration_id", "weighting", "order", "sample_schedule", "num_points", "order_sweep", "strategy"]
    summary = summarize(per_case, keys, ["absolute_center_error", "trial_sd", "rmse"])
    # All equivalent nominal schedules appear in the appendix, but only distinct
    # canonical configurations enter ranking and development selection.
    ranking = summary[summary.strategy].copy()
    medians = per_case.groupby(["embedding", "configuration_id"]).rmse.median()
    ranking["median_rmse"] = [medians[e, c] for e,c in zip(ranking.embedding,ranking.configuration_id)]
    ranking = ranking.sort_values(["embedding", "order", "mean_rmse", "median_rmse", "configuration_id"])
    chosen = ranking.groupby(["embedding", "order"], sort=False).head(1).reset_index(drop=True)
    return per_case, summary, ranking, chosen


def selected_heldout(raw, chosen):
    rows = []
    for embedding, _ in EMBEDDINGS:
        mapping = {"empirical_plugin": "Empirical", "ols_p1_uniform_n_l15": "FID_infinity", "rtd": "RTD"}
        for row in chosen[chosen.embedding.eq(embedding) & chosen.order.le(3)].itertuples():
            mapping[row.configuration_id] = f"VA-{row.order}"
        subset = raw[raw.embedding.eq(embedding) & raw.configuration_id.isin(mapping)].copy()
        subset["method"] = subset.configuration_id.map(mapping)
        subset["max_samples"] = 50000
        subset["measurement_source"] = "heldout_trials"
        rows.append(subset)
    result = pd.concat(rows, ignore_index=True)
    if len(result) != 630:
        raise ValueError("Selected held-out evaluation requires 630 trial estimates")
    return result


def heldout_statistics(raw, proxies):
    check_trials(raw, ["case_id", "max_samples", "method"], 5)
    targets = proxies.set_index(["case_id", "proxy"]).reference_value
    rows = []
    for (case, n, method), group in raw.groupby(["case_id", "max_samples", "method"], sort=True):
        for proxy in ("rtd", "plugin", "fid_infinity"):
            result = trial_metrics(group.estimate, targets[case, proxy])
            rows.append({"case_id": case, "generator": group.generator.iloc[0], "embedding": group.embedding.iloc[0],
                         "configuration_id": group.configuration_id.iloc[0], "method": method, "max_samples": n,
                         "proxy": proxy, **result})
    cases = pd.DataFrame(rows)
    outer = summarize(cases, ["embedding", "max_samples", "method", "proxy"], ["median_absolute_error"])
    return cases, outer


def runtime_statistics(raw):
    rows = []
    for (embedding, method), group in raw.groupby(["embedding", "method"], sort=True):
        if len(group) != 35:
            raise ValueError("Runtime aggregation requires all 35 held-out measurements")
        s = across_cases(group.runtime_seconds)
        rows.append({"embedding": embedding, "method": method, "mean_runtime": s["mean"], "sd_runtime": s["sd"], "runs": 35})
    return pd.DataFrame(rows)


def proxy_trend_statistics(raw):
    """Summarize the original repetitions at each development-pool budget."""
    for proxy, group in raw.groupby('proxy'):
        check_trials(group.rename(columns={'repetition': 'trial_id'}),
                     ['case_id', 'sample_budget', 'proxy'], 1 if proxy == 'plugin' else 5)
    rows = []
    keys = ['case_id', 'generator', 'embedding', 'dimension', 'sample_budget', 'proxy']
    for values, group in raw.groupby(keys, sort=True):
        if not np.isfinite(group.estimate).all():
            raise ValueError('Nonfinite proxy estimates require explicit handling')
        value = float(np.median(group.estimate))
        rows.append({**dict(zip(keys, values)), 'median_estimate': value})
    return pd.DataFrame(rows)


def refit_vale2(nodes, keys):
    """Apply second-order variance-aware weights to saved plug-in curves."""
    from gaussian_w2.estimators.extrapolation import fit_inverse_sample_polynomial

    rows = []
    for values, group in nodes.groupby(keys, sort=True):
        group = group.sort_values('sample_size')
        fit = fit_inverse_sample_polynomial(group.sample_size, group.estimate,
                                            order=2, weighting='variance_aware')
        rows.append({**dict(zip(keys, values)), 'estimate': fit['intercept']})
    return pd.DataFrame(rows)


def intro_statistics(random, gaussian_nodes, trend, trend_nodes):
    """Rebuild the two introductory comparisons on their shared 15-node curves."""
    keys = ['family', 'case_label', 'dimension', 'N', 'instance_id', 'trial_id']
    raw = random[random.family.eq('haar_log_uniform') & random.N.eq(50000)]
    nodes = gaussian_nodes[gaussian_nodes.dataset.eq('gaussian_random') &
                           gaussian_nodes.family.eq('haar_log_uniform') & gaussian_nodes.N.eq(50000)]
    fitted = refit_vale2(nodes, keys).rename(columns={'estimate': 'estimate_w2_sq_raw'})
    truth = raw[raw.method.eq('empirical_plugin')][keys + ['true_w2_sq']]
    fitted = fitted.merge(truth, on=keys, validate='one_to_one')
    fitted['method'], fitted['status'] = 'vale2', 'ok'
    baselines = raw[raw.method.isin(['fid_infinity_ols_order1', 'general_adaptive'])].copy()
    baselines['method'] = baselines.method.map({'fid_infinity_ols_order1': 'fid_infinity',
                                              'general_adaptive': 'rtd'})
    _, gaussian = gaussian_statistics(pd.concat([baselines, fitted], ignore_index=True))

    generator = 'ddo_edm2_l_imagenet512'
    selected = trend.generator.eq(generator) & trend.embedding.eq('fid') & trend.sample_budget.ge(50000)
    imagenet = trend[selected & trend.proxy.ne('plugin')].rename(columns={'proxy': 'method'})
    nodes = trend_nodes[trend_nodes.generator.eq(generator) & trend_nodes.embedding.eq('fid') &
                        trend_nodes.sample_budget.ge(50000)]
    keys = ['case_id', 'generator', 'embedding', 'dimension', 'sample_budget', 'repetition']
    fitted = refit_vale2(nodes, keys)
    fitted['proxy'], fitted['status'] = 'vale2', 'ok'
    vale2 = proxy_trend_statistics(fitted).rename(columns={'proxy': 'method'})
    imagenet = pd.concat([imagenet, vale2], ignore_index=True)
    return gaussian, imagenet

"""Post-run ImageNet statistics must retain the complete planned study grid."""
import csv
import json
import math

import numpy as np
import pandas as pd
import pytest

from experiments.imagenet.analyze import analyze


METRICS = (
    "mean_estimate", "signed_error", "center_error", "sd", "rmse",
    "median_absolute_error",
)
AGGREGATES = tuple(
    f"{metric}_{suffix}"
    for metric in METRICS
    for suffix in ("mean", "between_case_sd")
)


def make_run(tmp_path, *, case_ids=("case_a",), mode="development", trials=3,
             budgets=(100,), configurations=("empirical_plugin",)):
    """Only write the analysis inputs; the underlying feature data do not exist."""
    run = tmp_path / "run"
    run.mkdir()
    cases = [
        {"case_id": case_id, "generator": f"generator_{case_id}", "embedding": "fid"}
        for case_id in case_ids
    ]
    manifest = {
        "config": {
            "cases": cases,
            "sampling": {
                "num_trials": trials, "sample_budgets": list(budgets), "outer_mode": mode,
            },
        },
        "specifications": {
            "fid": {
                str(budget): [{"configuration_id": name} for name in configurations]
                for budget in budgets
            }
        },
    }
    (run / "manifest.json").write_text(json.dumps(manifest))
    for case_id in case_ids:
        write_proxy(run, case_id)
    return run


def write_proxy(run, case_id, *, rtd=2.0, plugin=11.0, fid_infinity=7.0):
    path = run / "cases" / case_id / "proxies.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "proxies": {"rtd": rtd, "plugin": plugin, "fid_infinity": fid_infinity},
    }))


def trial_rows(case_id="case_a", values=(1.0, 3.0, 5.0), *, budget=100,
               configuration="empirical_plugin"):
    return [
        {
            "case_id": case_id,
            "generator": f"generator_{case_id}",
            "embedding": "fid",
            "sample_budget": budget,
            "configuration_id": configuration,
            "trial_id": trial,
            "status": "ok",
            "estimate": value,
            # Analysis must use the selected proxies.json value, not this stale hint.
            "proxy_rtd": -999.0,
        }
        for trial, value in enumerate(values)
    ]


def write_trials(run, rows, filename=None):
    if filename is None:
        for case_id in dict.fromkeys(row["case_id"] for row in rows):
            write_trials(run, [row for row in rows if row["case_id"] == case_id],
                         f"trials.{case_id}.csv")
        return
    with (run / filename).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(trial_rows()[0]))
        writer.writeheader()
        writer.writerows(rows)


def assert_no_metrics(row, names=METRICS):
    assert all(pd.isna(row[name]) for name in names)


def test_trial_metrics_and_equal_case_aggregation(tmp_path):
    run = make_run(tmp_path, case_ids=("case_a", "case_b"))
    write_proxy(run, "case_b", rtd=5.0)
    write_trials(run, trial_rows() + trial_rows("case_b", (0.0, 4.0, 8.0)))

    result = analyze(run)
    assert isinstance(result["per_case"], pd.DataFrame)
    assert isinstance(result["summary"], pd.DataFrame)
    per_case = result["per_case"].set_index("case_id")
    assert len(per_case) == 2
    a, b = per_case.loc["case_a"], per_case.loc["case_b"]
    assert a["generator"] == "generator_case_a"
    assert a["embedding"] == "fid"
    assert a["sample_budget"] == 100
    assert a["configuration_id"] == "empirical_plugin"
    assert a["proxy"] == "rtd"
    assert a["target"] == 2.0
    assert a["expected_trials"] == a["attempted_trials"] == a["ok_trials"] == 3
    assert a["missing_trials"] == a["failed_trials"] == 0
    assert bool(a["available"])
    assert a["mean_estimate"] == 3.0
    assert a["signed_error"] == a["center_error"] == 1.0
    assert a["sd"] == 2.0  # Sample denominator T - 1.
    assert a["rmse"] == pytest.approx(math.sqrt(11.0 / 3.0))  # Denominator T.
    assert a["median_absolute_error"] == 1.0
    assert b["signed_error"] == -1.0
    assert b["sd"] == 4.0
    assert b["rmse"] == pytest.approx(math.sqrt(35.0 / 3.0))
    assert b["median_absolute_error"] == 3.0

    summary = result["summary"]
    assert len(summary) == 1
    group = summary.iloc[0]
    assert group["embedding"] == "fid"
    assert group["sample_budget"] == 100
    assert group["configuration_id"] == "empirical_plugin"
    assert group["proxy"] == "rtd"
    assert group["expected_cases"] == group["attempted_cases"] == group["available_cases"] == 2
    assert group["primary_metric"] == "center_error,sd,rmse"
    for metric in METRICS:
        values = np.array([a[metric], b[metric]])
        assert group[f"{metric}_mean"] == pytest.approx(values.mean())
        assert group[f"{metric}_between_case_sd"] == pytest.approx(values.std(ddof=1))
    assert group["center_error_mean"] == 1.0
    assert group["signed_error_mean"] == 0.0
    assert group["rmse_mean"] != pytest.approx(math.sqrt(46.0 / 6.0))
    assert (run / "analysis" / "per_case.csv").is_file()
    assert (run / "analysis" / "summary.csv").is_file()


@pytest.mark.parametrize("problem", ["failed", "incomplete", "missing", "nonfinite"])
def test_unavailable_case_invalidates_all_aggregate_metrics(tmp_path, problem):
    run = make_run(tmp_path, case_ids=("case_a", "case_b"))
    rows = trial_rows("case_b")
    expected_counts = (3, 3, 0, 0)
    if problem == "failed":
        rows[1].update(status="failed", estimate=None)
        expected_counts = (3, 2, 0, 1)
    elif problem == "incomplete":
        rows.pop()
        expected_counts = (2, 2, 1, 0)
    elif problem == "missing":
        rows = []
        expected_counts = (0, 0, 3, 0)
    else:
        rows[1]["estimate"] = float("inf")
        expected_counts = (3, 2, 0, 1)
    write_trials(run, trial_rows() + rows)

    result = analyze(run)
    per_case = result["per_case"].set_index("case_id")
    assert bool(per_case.loc["case_a", "available"])
    unavailable = per_case.loc["case_b"]
    assert not bool(unavailable["available"])
    assert isinstance(unavailable["unavailable_reason"], str)
    assert unavailable["unavailable_reason"]
    assert unavailable["expected_trials"] == 3
    assert tuple(unavailable[name] for name in (
        "attempted_trials", "ok_trials", "missing_trials", "failed_trials",
    )) == expected_counts
    assert_no_metrics(unavailable)
    summary = result["summary"].iloc[0]
    assert summary["expected_cases"] == 2
    assert summary["attempted_cases"] == (1 if problem == "missing" else 2)
    assert summary["available_cases"] == 1
    assert_no_metrics(summary, AGGREGATES)


@pytest.mark.parametrize("missing_proxy", ["null", "missing_file", "nonfinite"])
def test_unavailable_proxy_blocks_metrics_without_discarding_successful_trials(tmp_path, missing_proxy):
    run = make_run(tmp_path)
    write_trials(run, trial_rows())
    if missing_proxy == "missing_file":
        (run / "cases" / "case_a" / "proxies.json").unlink()
    else:
        write_proxy(run, "case_a", rtd=None if missing_proxy == "null" else float("inf"))
    result = analyze(run)
    row = result["per_case"].iloc[0]
    assert row["attempted_trials"] == row["ok_trials"] == 3
    assert row["missing_trials"] == row["failed_trials"] == 0
    assert not bool(row["available"])
    assert any(word in row["unavailable_reason"].lower() for word in ("proxy", "target"))
    assert_no_metrics(row)
    assert result["summary"].iloc[0]["available_cases"] == 0
    assert_no_metrics(result["summary"].iloc[0], AGGREGATES)


def test_no_trials_still_reports_every_planned_case_budget_and_configuration(tmp_path):
    run = make_run(tmp_path, case_ids=("case_a", "case_b"), budgets=(100, 200),
                   configurations=("empirical_plugin", "rtd"))
    result = analyze(run)
    assert len(result["per_case"]) == 8
    assert len(result["summary"]) == 4
    for _, row in result["per_case"].iterrows():
        assert row["expected_trials"] == row["missing_trials"] == 3
        assert row["attempted_trials"] == row["ok_trials"] == row["failed_trials"] == 0
        assert not bool(row["available"])
        assert_no_metrics(row)
    for _, row in result["summary"].iterrows():
        assert row["expected_cases"] == 2
        assert row["attempted_cases"] == row["available_cases"] == 0
        assert_no_metrics(row, AGGREGATES)


def test_case_csv_shards_and_heldout_primary_metric(tmp_path):
    run = make_run(tmp_path, case_ids=("case_a", "case_b"), mode="heldout")
    write_trials(run, trial_rows(), "trials.case_a.csv")
    write_trials(run, trial_rows("case_b"), "trials.case_b.csv")
    result = analyze(run)
    assert len(result["per_case"]) == 2
    assert result["summary"].iloc[0]["available_cases"] == 2
    assert result["summary"].iloc[0]["primary_metric"] == "median_absolute_error"


@pytest.mark.parametrize("proxy,target", [("plugin", 11.0), ("fid_infinity", 7.0)])
def test_explicit_proxy_selection(tmp_path, proxy, target):
    run = make_run(tmp_path)
    write_proxy(run, "case_a", rtd=None)
    write_trials(run, trial_rows())
    result = analyze(run, proxy=proxy)
    row = result["per_case"].iloc[0]
    assert bool(row["available"])
    assert row["proxy"] == result["summary"].iloc[0]["proxy"] == proxy
    assert row["target"] == target
    assert row["signed_error"] == 3.0 - target


@pytest.mark.parametrize("source", ["within_case", "across_files"])
def test_duplicate_trial_rows_rejected(tmp_path, source):
    run = make_run(tmp_path)
    rows = trial_rows()
    if source == "within_case":
        write_trials(run, rows + [rows[0]])
    else:
        write_trials(run, rows, "trials.case_a.csv")
        write_trials(run, [rows[0]], "trials.another.csv")
    with pytest.raises(ValueError):
        analyze(run)


def test_old_full_export_does_not_override_case_results(tmp_path):
    run = make_run(tmp_path)
    write_trials(run, trial_rows())
    write_trials(run, trial_rows(values=(20.0, 30.0)), "trials.csv")
    result = analyze(run)
    row = result["per_case"].iloc[0]
    assert row["attempted_trials"] == row["ok_trials"] == 3
    assert bool(row["available"])
    assert row["mean_estimate"] == 3.0


def test_summary_groups_use_only_the_cases_for_their_embedding(tmp_path):
    run = make_run(tmp_path, case_ids=("case_a", "case_b", "case_c"),
                   budgets=(100, 200), configurations=("empirical_plugin", "rtd"))
    path = run / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["config"]["cases"][2]["embedding"] = "clip"
    manifest["specifications"]["clip"] = manifest["specifications"]["fid"]
    path.write_text(json.dumps(manifest))
    rows = []
    for case_id in ("case_a", "case_b", "case_c"):
        for budget in (100, 200):
            for configuration in ("empirical_plugin", "rtd"):
                cell = trial_rows(case_id, budget=budget, configuration=configuration)
                if case_id == "case_c":
                    for row in cell:
                        row["embedding"] = "clip"
                if (case_id, budget, configuration) == ("case_a", 100, "rtd"):
                    cell[0].update(status="failed", estimate=None)
                rows.extend(cell)
    write_trials(run, rows)

    result = analyze(run)
    assert len(result["per_case"]) == 12
    assert len(result["summary"]) == 8
    for _, row in result["summary"].iterrows():
        expected = 2 if row["embedding"] == "fid" else 1
        assert row["expected_cases"] == row["attempted_cases"] == expected
        affected = (row["embedding"], row["sample_budget"], row["configuration_id"]) == ("fid", 100, "rtd")
        assert row["available_cases"] == expected - int(affected)
        if affected:
            assert_no_metrics(row, AGGREGATES)
        else:
            assert row["mean_estimate_mean"] == 3.0
            if expected == 1:
                assert all(pd.isna(row[f"{metric}_between_case_sd"]) for metric in METRICS)


@pytest.mark.parametrize("field,value", [
    ("case_id", "unplanned_case"),
    ("configuration_id", "unplanned_configuration"),
    ("sample_budget", 999),
    ("trial_id", -1),
    ("trial_id", 3),
])
def test_unplanned_trial_rows_rejected(tmp_path, field, value):
    run = make_run(tmp_path)
    rows = trial_rows()
    rows[0][field] = value
    write_trials(run, rows)
    with pytest.raises(ValueError):
        analyze(run)


def test_csv_outputs_retain_full_float_precision_and_custom_output(tmp_path):
    run = make_run(tmp_path)
    write_trials(run, trial_rows(values=(1.2345678901234567, 3.345678901234567, 5.456789012345678)))
    destination = tmp_path / "reports" / "imagenet"
    result = analyze(run, output=destination)
    for name, columns in (("per_case", METRICS), ("summary", AGGREGATES)):
        with (destination / f"{name}.csv").open(newline="") as stream:
            saved = list(csv.DictReader(stream))
        assert len(saved) == len(result[name])
        for written, (_, expected) in zip(saved, result[name].iterrows()):
            for metric in columns:
                if pd.isna(expected[metric]):
                    assert written[metric] == "" or math.isnan(float(written[metric]))
                else:
                    assert float(written[metric]) == float(expected[metric])

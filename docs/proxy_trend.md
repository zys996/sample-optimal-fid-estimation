# Proxy estimates across sample sizes

Compare plug-in, FID-infinity, and RTD on the 21 generator/embedding cases at
30K, 60K, ..., 300K samples. Each budget uses the first N rows of the same
300K development pool; the real reference stays fixed. The horizontal axis is
sample count and the vertical axis is the raw FD estimate.

The [paper-results notebook](../reproduce/paper_results.ipynb) reproduces the
final paper's proxy figures from bundled records: StyleGAN-XL-256 in the main
text and the other six generators in three appendix figures. These use the ten
budgets from 30K to 300K. The introduction's DDO/EDM2-L-512 example uses 50K,
60K, 90K, ..., 300K and adds VALE₂ by refitting the saved 15 uniform-in-n nodes
with second-order variance-aware weights. See [the reproduction guide](reproduction.md).

The settings come from `configs/paper/imagenet/ablation.json`:

- Plug-in uses all N rows once.
- FID-infinity fits order 1 with 15 nodes uniform in n, from 5K to N. Each smaller
  node is independently subsampled without replacement. Five fits give a median.
- RTD uses `m0=floor(0.6*N)` and the configured adaptive degree. Five permutations
  of the same N rows give a median.

The repetitions vary the algorithm within a fixed pool. The 300K endpoint uses
exactly the development proxy sampling rules. Raw estimates, including negative
values, are saved. If any RTD repetition fails, its curve has a gap at that budget;
the summary reports the failure count.

## Compute on saved features

Use the GPU measurement environment described in [environments.md](environments.md).
Set `FD_DATA_ROOT` to the feature/reference layout in [imagenet.md](imagenet.md),
then run from the repository root:

```sh
export FD_DATA_ROOT=/path/to/imagenet_features
export FD_PROXY_RUN=/path/to/results/proxy_trend
python -m experiments.imagenet.proxy_trend --config configs/paper/imagenet/proxy_trend.json --output "$FD_PROXY_RUN"
python -m experiments.imagenet.plot_proxy_trend --run "$FD_PROXY_RUN"
```

For features stored elsewhere, pass `--inputs-file /path/to/inputs.json` to the
compute command. It has the same `extractor_signature` and `cases` structure as
`configs/paper/imagenet/inputs.json`, with the desired feature/reference paths.
The study configuration can also point to a different `study_config` or budget list.

List case identifiers with:

```sh
python -m experiments.imagenet.proxy_trend --list-cases
```

Use `--case-id CASE_ID` to compute one complete generator/embedding case. There
are 21 cases, 210 case/budget combinations, and 2,310 reported estimates: 210
plug-in values, 1,050 FID-infinity fits, and 1,050 RTD runs. A case runs through
all ten budgets and publishes its CSV when complete.

## Plot saved results

Plotting only needs the CPU reporting dependencies and the completed result
files. It can be rerun locally:

```sh
python -m experiments.imagenet.plot_proxy_trend --run /path/to/results/proxy_trend
```

The output includes:

- `manifest.json`: resolved experiment settings and case identifiers.
- `cases/CASE/nN/proxies.json`: all repetitions, FID-infinity nodes, and RTD outcomes.
- `trials.CASE.csv`: individual estimates for a completed case.
- `summary.csv`: each proxy median, repetition count, and failure count.
- `figures/proxy_trend.pdf` and `.png`: generator rows and embedding columns.
- `figures/proxy_trend_fid`, `proxy_trend_fd_dinov2`, and `proxy_trend_clip`
  in PDF/PNG: one sheet per embedding.

`--output /path/to/figures_report` directs the summary and figures to a separate
folder. The plotter can display a subset of completed cases.


## Add the 50K point to an existing run

The supplemental config measures only 50K with the same 21 cases, feature prefix,
reference, seeds, and estimator settings. It adds 63 summary cells from 231
individual estimates (21 plug-in values and 105 repetitions of each other method).
Use the same input configuration and `FD_DATA_ROOT` as the original run.
The commands below use the paper's `inputs.json`; for a custom feature layout,
pass the original inputs file instead. Run from the repository root in the same
GPU environment:

```sh
FD_PROXY_BASE="$SCRATCH/gaussian_w2/runs/proxy_trend_30k_300k_v1"
FD_PROXY_EXTRA="$SCRATCH/gaussian_w2/runs/proxy_trend_50k_v1"
FD_PROXY_COMBINED="$SCRATCH/gaussian_w2/runs/proxy_trend_with_50k_v1"

python -m experiments.imagenet.proxy_trend --config configs/paper/imagenet/proxy_trend_50k.json --inputs-file configs/paper/imagenet/inputs.json --output "$FD_PROXY_EXTRA"
python -m experiments.imagenet.plot_proxy_trend --run "$FD_PROXY_BASE" --additional-run "$FD_PROXY_EXTRA" --output "$FD_PROXY_COMBINED"
```

The combined report has 693 summary cells at 30K, 50K, 60K, 90K, ..., 300K and
regenerates the overall grid and all three embedding sheets. The 50K measurement
has an explicit horizontal-axis tick. The original trial files stay in their
respective run directories. `--additional-run` can also combine further disjoint
budgets; all runs must use the same study settings.

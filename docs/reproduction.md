# Reconstruct the paper results

The [paper-results notebook](../reproduce/paper_results.ipynb) contains executed
outputs for all 11 figures and 13 empirical tables. It follows the introduction,
Gaussian experiments, ImageNet experiments, and appendix results in paper order.
Each figure or table has its own display cell.

After installing `requirements-report.txt` and `python -m pip install -e ".[notebook]"`,
open it with:

```bash
jupyter lab reproduce/paper_results.ipynb
```

Use **Run All Cells** to recompute and refresh the embedded outputs. The
notebook calls the same functions as the command-line entry point below;
statistics and strategy selection are defined once. Its HTML tables contain
the same formatted numbers as the LaTeX exports.

From the installed repository root:

```bash
python -m reproduce --output outputs/paper
```

The command uses the bundled numerical records on CPU to create all 11
figures (PDF and PNG), all 13 empirical tables (HTML, LaTeX, and
full-precision CSV records), intermediate case statistics, and fresh development selections.
Use `--no-figures` for numerical reconstruction alone.

For detailed numerical verification, run:

```bash
python -m pytest tests/test_reproduction.py -q
```

These tests compare the tables with the manuscript, check strategy selection
and proxy medians, and independently rebuild 94,775 recorded Empirical/OLS/VALE
estimates from the included plug-in nodes using the estimator implementation in
this repository, including the first-order full-pool proxy repetitions. The
proxy-trend records and node fits are checked separately.
To save the node comparison separately, use
`python -m reproduce.audit_nodes --output outputs/paper/node_audit.json`.
The bundle also includes RTD trial estimates and five full-pool proxy
repetitions. The experiment runners evaluate RTD on Gaussian observations
or ImageNet features.

## Reporting rules

- **Isotropic Gaussian:** mean estimate, signed error, and sample SD across 50
  trajectories, for each of five methods. The population target is exactly 5.
- **Random and stress Gaussian:** compute the median of five absolute trial
  errors for each population pair, then report the mean and sample SD across ten
  pairs. Both stress figures use the final repeated ten-pair/five-trial design.
- **ImageNet ablation:** within each case, compute center error
  `abs(mean(estimate) - proxy)`, sample trial SD, and RMSE with denominator 10.
  Each plotted mean and table entry gives equal weight to the seven generators
  within one embedding. Between-generator SD also uses `ddof=1`. The successive
  ablation summary averages the full-precision order-specific mean RMSEs over
  orders 1–3; percentage reductions compare consecutive configurations.
  Values are rounded only for display.
- **Selection:** recompute every canonical VALE candidate's case RMSE against the
  fixed RTD proxy. Within each embedding and order, minimize mean case RMSE,
  breaking ties by median case RMSE and then canonical configuration identifier.
  Selection uses development data only. The 48 nominal schedule combinations
  represent 45 distinct rules; aliases are restored in the appendix tables.
- **Held-out and budget:** median absolute error over the five disjoint held-out
  blocks per generator, then mean and sample SD across seven generators. Negative
  estimates remain raw; failed or incomplete cells cause an error.
- **Runtime:** directly aggregate all 35 recorded runtimes within an embedding
  and method, with sample SD. Node timings are charged to every selected rule
  using them and its recorded fit time is added. Timings cover estimator
  evaluation in the recorded GPU environment.
- **Proxy trends:** report medians of five FID-infinity and RTD estimates at
  each budget from 30K, 60K, ..., 300K. Proxy agreement is the mean absolute
  difference from RTD across seven generators at the same budget.
- **Introduction comparison:** the Gaussian panel uses the same ten-pair
  median-error aggregation at 50K. The ImageNet panel shows median estimates
  at 50K, 60K, 90K, ..., 300K for DDO/EDM2-L-512. Both extrapolators use 15
  sample sizes uniformly spaced from 5K to N; VALE₂ is refitted from the saved
  plug-in nodes with second-order variance-aware weights.

The RTD pilot uses `m0=0.8N` for Gaussian experiments and `m0=floor(0.6N)` for
ImageNet. ImageNet's covariance correction operates after removal of one mean
direction, so its recorded correction count is `N-1-m0`.

**Final 50K endpoint:** the sample-budget tables use the same selected trial rows
as the complete held-out comparison at 50K. The budget experiment contributes
the 10K–40K results.

## Figure and table map

| Paper result | Output under `outputs/paper` | Input records |
|---|---|---|
| Introduction comparison | `figures/intro_gaussian_and_imagenet.pdf` | `gaussian_random`, `gaussian_nodes`, `proxy_trend_trials`, `proxy_trend_nodes` |
| Haar Gaussian dimension figure | `figures/gaussian_dimension_haar.pdf` | `gaussian_random` |
| StyleGAN-XL-256 proxy example | `figures/imagenet_proxy_example.pdf` | `proxy_trend_trials` |
| Spiked-bulk dimension figure | `figures/gaussian_dimension_spiked_bulk.pdf` | `gaussian_random` |
| Rotated Toeplitz dimension figure | `figures/gaussian_dimension_rotated_toeplitz.pdf` | `gaussian_random` |
| Gaussian rank figure | `figures/gaussian_rank_error.pdf` | `gaussian_stress` |
| Gaussian condition figure | `figures/gaussian_condition_error.pdf` | `gaussian_stress` |
| Three additional-generator proxy figures | `figures/imagenet_proxy_consistency_{inception,dinov2,clip}_additional.pdf` | `proxy_trend_trials` |
| ImageNet OLS order figure | `figures/imagenet_ols_order_rtd_absolute.pdf` | `development_trials`, `proxies`, `development_plan` |
| Isotropic table | `tables/isotropic.tex` | `gaussian_isotropic` |
| Held-out comparison against three proxies | `tables/proxy_sensitivity.tex` | Selected `heldout_trials`, all three `proxies` |
| Successive ablation summary | `tables/ablation_summary.tex` | Development records and fresh selection |
| Runtime | `tables/runtime.tex` | Selected `heldout_trials` runtimes |
| Inception sample budgets | `tables/sample_budget.tex` | `budget_trials` 10K–40K plus selected `heldout_trials` 50K |
| Checkpoints and proxies | `tables/checkpoint_proxies.tex` | `proxies` |
| Proxy agreement | `tables/proxy_agreement.tex` | `proxy_trend_trials` |
| Matched OLS–VALE | `tables/matched_ols_va.tex` | Development records |
| Selected VALE schedules | `tables/selected_strategies.tex` | Development records and fresh selection |
| Three complete VALE grids | `tables/va_grid_{fid,fd_dinov2,clip}.tex` | Development records and nominal aliases |
| DINOv2 and CLIP sample budgets | `tables/sample_budget_additional.tex` | `budget_trials` 10K–40K plus selected `heldout_trials` 50K |

The ImageNet OLS order figure contains 378 hollow case markers: three metrics, three
embeddings, six orders, and seven generators. Solid curves show the means.
Orders 7–8 remain in the archived candidate bank but are excluded from this
final figure. The Gaussian figures preserve the paper's grids, colors, markers,
mean/SD bands, and dimension slope guides.

## Numerical correspondence

The bundled `paper_table_values.json` contains the manuscript's 805 decimal
entries from its 13 empirical tables. All entries agree at the displayed
precision. Runtime means and sample SDs are reported in milliseconds with one
decimal place; sample-budget tables use three decimal places. CSV records
retain full precision. The notebook, HTML tables, and LaTeX tables share the
same formatted values.

RTD numerical regression tests use an absolute tolerance of `1e-6` and zero
relative tolerance. Analytic support and low-rank tests use tighter tolerances.

## Bundled data and auditability

`reproduce/sources.json` records the input files, schemas, and row counts.
Compressed CSVs preserve original decimal numeric values without lossy rounding;
the reader uses round-trip binary64 parsing. The bundle includes:

| Records | Rows |
|---|---:|
| Isotropic Gaussian estimates | 250 |
| Random Gaussian estimates | 30,000 |
| Repeated stress Gaussian estimates | 60,000 |
| Shared Gaussian plug-in nodes | 270,750 |
| Development trial estimates | 11,340 |
| Development plug-in nodes | 24,990 |
| Full held-out trial bank | 9,135 |
| Held-out plug-in nodes | 12,495 |
| Budget 10K–40K trial estimates | 2,520 |
| Budget 10K–40K plug-in nodes | 21,490 |
| Reference proxy values | 63 |
| Randomized proxy repetitions | 210 |
| Full-pool first-order proxy nodes | 1,575 |
| Proxy-trend estimates | 2,321 |
| Proxy-trend plug-in nodes | 15,825 |

Plans include exact nodes, weights, canonical configuration identities, schedule
aliases, and scientific case identifiers. Tests cover the paper values,
strategy selection, aggregation, source mapping, and node reconstruction.

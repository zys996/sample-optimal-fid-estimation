# Gaussian experiments

The entry point is one small runner shared by three designs:

| Config | Population design | Dimensions / budgets | Repetitions |
|---|---|---|---|
| `configs/paper/gaussian/isotropic.yaml` | Isotropic, generated scale 0.5, squared FD 5 | d=5000, N=100000 | 50 trials |
| `configs/paper/gaussian/random.yaml` | Haar log-uniform, spiked bulk, rotated Toeplitz | 10 dimensions, four budgets 50K–100K | 10 covariance instances × 5 trials |
| `configs/paper/gaussian/stress.yaml` | Three ranks and three condition numbers; squared FD 5 | Same dimension/budget grid | 10 randomized angle profiles × 5 trials |

Run from the repository root after installation:

```sh
python -m experiments.gaussian.run --config configs/examples/gaussian.yaml --output outputs/gaussian_smoke
python -m experiments.gaussian.collect --run outputs/gaussian_smoke
python -m experiments.gaussian.run --config configs/paper/gaussian/isotropic.yaml --output outputs/isotropic
python -m experiments.gaussian.collect --run outputs/isotropic
```

Copy a config to change dimensions, budgets, repetitions, covariance parameters,
extrapolation orders/weightings/schedule, pilot fraction, or numerical tolerances.
The CPU example covers all three suites, both OLS and variance-aware fits, and a
small explicit Taylor-degree cap. Full paper configs use CuPy and the uncapped
adaptive rule. `--backend numpy` runs a custom configuration on CPU.

The reference covariance A is exact and fixed. Only B is sampled. All means are
known zero; empirical covariances are uncentered `X.T @ X / N`. This differs
from the centered `1/(N-1)` feature-statistic convention in ImageNet experiments.
Each conditional trial draws a single maximum-budget pool. All N values are
nested prefixes. RTD uses `m0 = floor(4N/5)` by default and fresh correction
observations after that pilot. With `n=N-m0`, its degree rule is

```
epsilon = min(1, sqrt(d/n))
L_requested = max(2, ceil(log(C_T*d/epsilon) / log(1/rho_design)))
L_used = min(L_requested, floor(n/2), optional_degree_cap)
```

The private numerical kernels implement the degree partition, support
thresholds, Sylvester recursion, and compensated polarization sums. The
known-zero-mean path streams raw rows and sets the mean term to zero; the
centered-feature path uses Helmert contrasts. Tests compare the two paths on
equivalent observations, including singular supports, and check recorded
estimates and pilot-rank failures.

Seeds are derived with SHA-256 from the master seed and case identity:

- Covariance: `(master, "covariance", case_label, d, instance_id)`.
- Sampling: `(master, "sample", case_label, d, instance_id, trial_id)`.
- Extrapolation subsets: `(sampling_seed, "ols", N)`.

The hash starts with the decimal master seed, appends each UTF-8 component with
a leading NUL byte, and interprets its first four digest bytes as a big-endian
integer. Case labels and generation batches are specified in the configs.
CUDA generation uses CuPy RandomState; CPU runs use NumPy Generator. These
backends use distinct random streams with the same seed identities. Smaller
extrapolation nodes use fresh NumPy permutations of the full N-prefix; the
endpoint uses all N rows.

Each task computes every configured trial and then atomically saves
`records/task_XXXXX.json`, including covariance/sample seeds, population metadata,
finite-sample curves, raw estimates, statuses, and RTD sample-split/degree metadata.
`manifest.json` records the resolved configuration and Python/NumPy/SciPy versions
(plus CuPy/CUDA/GPU details for CUDA runs).

`--task-index i` runs one case/dimension/instance task. Separate tasks can share
an output directory with the same configuration. A task has no intermediate
checkpoints: if interrupted, run that task again from the beginning. Existing
results are protected; `--task-index i --overwrite` explicitly recomputes and
replaces only that task after it finishes. Numerical estimator failures are
saved with a failure status. Population-construction and hardware-initialization
failures stop the task.

After the tasks finish, collect their results without running any estimators:

```sh
python -m experiments.gaussian.collect --run outputs/gaussian_smoke
```

Collection checks the configuration, trial identities, seeds, method/budget
coverage, and finite successful estimates, then writes `raw.csv` and
`run_summary.json`. The summary includes completed and expected task counts;
a partial collection contains only completed tasks. Repeat collection after
running or replacing additional tasks to refresh these exports.

## Analyze a new run

```sh
python -m experiments.gaussian.analyze --run outputs/gaussian_smoke --output outputs/gaussian_smoke/analysis
```

The analyzer reads `config.json` and `raw.csv` and writes full-precision
`per_case.csv` and `summary.csv`. Per-case columns include
mean estimate, absolute center error, trial sample SD, RMSE, and median absolute
error. Outer means and sample SD give every configured covariance instance equal
weight. For the isotropic experiment's one population, `trial_sd` retains its
50-trial sample SD and all between-case SD columns remain blank.

Each output cell includes trial counts and is reported when all configured
trials succeed. Otherwise its metrics and the corresponding outer summary are
blank. The analyzer verifies that rows are unique and match the configured
design.

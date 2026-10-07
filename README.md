# Sample-Optimal Estimation of the Fréchet Inception Distance

Code and experimental records for *Sample-Optimal Estimation of the Fréchet Inception Distance* by Ziyun Chen, Jerry Li, Kevin Tian, and Yusong Zhu.

The paper is available at: [arXiv:2610.07114](https://arxiv.org/abs/2610.07114).

We studied the finite-sample accuracy of FID estimation with a fixed Gaussian reference and  established tight bounds for the empirical and polynomial extrapolation estimators and introduces Relative Taylor Debiasing (RTD), which achieves optimal sample complexity. This repository implements the empirical estimator, FID-infinity and higher-order OLS extrapolation, variance-aware extrapolation (VALE), and RTD and tests on Gaussian and Imagenet environment.

All estimates are **squared** Fréchet distances. The reference statistics stay fixed, and RTD uses a pilot and fresh correction observations on the generated side.

## Repository layout

| Path | Contents |
|---|---|
| `src/gaussian_w2/estimators/` | Public estimator APIs; numerical routines are in `src/gaussian_w2/_core/` |
| `experiments/gaussian/` | Gaussian population generators, trial runner, and analysis |
| `experiments/imagenet/` | Image generation, feature preparation, estimator evaluation, and proxy studies |
| `configs/` | Paper experiment settings and a small Gaussian example |
| `reproduce/` | Recorded measurements, report generation, and the executed paper-results notebook |
| `docs/` | Reproduction instructions, data conventions, and environment details |
| `tests/` | Estimator, experiment, and reported-result checks |

See [the code guide](docs/layout.md) for implementation entry points.

## Installation

For CPU reporting, the estimator examples, and small Gaussian experiments, use Python 3.9 through 3.12 with the pinned dependencies below. Run from the repository root:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-report.txt
python -m pip install -e ".[notebook,test]"
```

On Windows PowerShell, activate the environment with `.venv\Scripts\Activate.ps1` instead. To use only the estimator library, install `python -m pip install -e .` in your environment.

The full paper experiments use CUDA. In a **separate environment** on a compatible Linux NVIDIA machine with Python 3.10 or newer, install:

```sh
python -m pip install -e ".[report,gpu]"
```

The GPU extra uses CuPy 14 and NumPy 2; the pinned CPU reporting environment uses NumPy 1.26. Image generation and feature extraction require an additional model stack. See [environments](docs/environments.md) and [ImageNet preparation](docs/imagenet.md) for those setups.

## Reconstruct the paper results

The executed [paper-results notebook](reproduce/paper_results.ipynb) contains the 11 figures and 13 empirical tables. Its saved outputs can be viewed without running code. To reconstruct them from the bundled numerical records on CPU:

```sh
python -m reproduce --output outputs/paper
```

This writes figures as PDF and PNG, tables as HTML and LaTeX, and full-precision CSV statistics under `outputs/paper/`. It also recomputes the development-set strategy selection. Use `--no-figures` to reconstruct only the numerical reports.

For interactive use, open the notebook and run its cells:

```sh
jupyter lab reproduce/paper_results.ipynb
```

The notebook and command-line report use the same calculations. The [reproduction guide](docs/reproduction.md) maps each paper result to its input records and documents the aggregation rules.

## Use the estimators

The public API accepts sample arrays and fixed reference statistics. This small CPU example evaluates empirical FID, first-order OLS (FID-infinity), second-order VALE, and RTD on the same generated samples:

```python
import numpy as np
from gaussian_w2.estimators import (
    make_fixed_reference_from_statistics,
    plugin_curve,
    rtd_fixed_reference,
)

rng = np.random.default_rng(7)
x = rng.normal(size=(200, 4))
reference = make_fixed_reference_from_statistics(np.zeros(4), np.eye(4))

curve = plugin_curve(
    reference,
    x,
    minimum_samples=50,
    num_points=15,
    sample_schedule="uniform_n",
    regression_orders=(1, 2),
    regression_weightings=("ordinary_ols", "variance_aware"),
    seed=7,
)
fits = curve["fits_by_weighting"]
rtd = rtd_fixed_reference(reference, x, m0=120, L=3)
if rtd["status"] != "ok":
    raise RuntimeError(rtd["error_message"])

print({
    "Empirical": curve["endpoint_plugin_w2_sq"],
    "FID-infinity": fits["ordinary_ols"][1]["intercept"],
    "VALE-2": fits["variance_aware"][2]["intercept"],
    "RTD": rtd["estimate_w2_sq_raw"],
})
```

The example uses 15 sample sizes from 50 to 200; the paper's FID-infinity baseline uses 15 uniform-in-sample-count nodes from 5K to the full budget. Optimized VALE uses the schedules selected on the development data. Fits return their intercept and extrapolation weights, and raw estimates may be negative.

By default, the API uses centered `1/(N-1)` covariance estimates. Gaussian paper experiments instead use known zero means and uncentered `1/N` covariance observations. RTD corrects the covariance term; for unknown means it retains the ordinary plug-in mean term. See [the code guide](docs/layout.md) for the CPU and GPU entry points.

## Run experiments

### Gaussian experiments

The small example generates its own observations and covers isotropic, random, and singular covariance cases:

```sh
python -m experiments.gaussian.run --config configs/examples/gaussian.yaml --output outputs/gaussian_smoke
python -m experiments.gaussian.collect --run outputs/gaussian_smoke
python -m experiments.gaussian.analyze --run outputs/gaussian_smoke
```

The paper configurations are `configs/paper/gaussian/isotropic.yaml`, `random.yaml`, and `stress.yaml`. Copy a configuration and choose a new output directory to change the study. See [Gaussian experiments](experiments/gaussian/README.md) for population definitions, sampling rules, and task-level execution.

### ImageNet experiments

ImageNet evaluation reads generated feature pools and frozen real-reference moments. After preparing the inputs and setting `FD_DATA_ROOT`, a development run and its analysis use:

```sh
python experiments/imagenet/run.py --config configs/paper/imagenet/ablation.json --output outputs/imagenet_ablation
python -m experiments.imagenet.analyze --run outputs/imagenet_ablation
```

The remaining stages use `heldout.json`, `heldout_full_grid.json`, and `budgets_reuse_heldout.json` under `configs/paper/imagenet/`. The [ImageNet guide](docs/imagenet.md) gives the full commands, shared input layout, and development/held-out split. The [proxy-trend guide](docs/proxy_trend.md) covers comparisons of empirical, FID-infinity, and RTD estimates from 30K to 300K samples.

## Data

`reproduce/data/` contains the trial estimates, plug-in nodes, proxy values, and experiment plans used to reconstruct the reported results. The [source inventory](reproduce/sources.json) records their schemas and row counts. Gaussian experiments can also be rerun from generated observations.

The repository does not include ImageNet images, full generated feature pools, or model weights. The [preparation instructions](docs/imagenet.md) describe how to generate features and reference statistics from the required external inputs. The [third-party source list](THIRD_PARTY.md) records the upstream implementations and checkpoints, which retain their own licenses.

## Tests

```sh
python -m pytest -q
```

The suite checks estimator calculations, experiment protocols, and reconstruction of the paper's tables from the recorded measurements. To run the report checks alone:

```sh
python -m pytest tests/test_reproduction.py -q
```

## AI disclosure

The authors initially formulated the FID estimation problem and derived an $O(\frac{d^2}{\epsilon^2})$ sample complexity bound for the empirical plug-in estimator.
We then consulted GPT 5.6 Sol Pro on whether the upper bound could be improved, and through interactive conversations, discovered the RTD strategy in [Section 5](https://arxiv.org/html/2610.07114v1#S5) with LLM assistance.
Motivated by prior work on $\mathrm{FID}_\infty$ ([Chong and Forsyth, 2020](https://arxiv.org/abs/1911.07023)), the authors in parallel proposed the extrapolation framework in [Section 4](https://arxiv.org/html/2610.07114v1#S4) (along with empirical improvements, e.g., the variance-aware choice of regression weights).
The authors finally developed a unified analysis establishing tight bounds for all estimators studied. The manuscript was written solely by the authors, who take full responsibility for the organization and presentation of all results.

## Acknowledgements

We thank the NSF AI Institute for Foundations of Machine Learning (IFML) for their support of this project, and the Texas Advanced Computing Center (TACC) for providing computing resources.

# Reading the code

Start with the task you want to carry out:

| Task | Entry point |
| --- | --- |
| Use an estimator on new data | `src/gaussian_w2/estimators/` |
| View and recompute paper tables and figures | `reproduce/paper_results.ipynb` |
| Generate Gaussian populations and run trials | `experiments/gaussian/` |
| Prepare image features or evaluate feature pools | `experiments/imagenet/` |
| Inspect exact paper parameters | `configs/paper/` |
| Try a small Gaussian experiment | `configs/examples/gaussian.yaml` |

The public estimator functions receive arrays and options. Their numerical
building blocks live in `_core/`. The estimator reading order is:

| Calculation | Implementation |
| --- | --- |
| OLS/VALE nodes and intercept weights | `src/gaussian_w2/_core/fid_infinity.py` |
| One-sided RTD pilot and correction | `src/gaussian_w2/_core/fixed_reference_fid.py` |
| Taylor/Sylvester recursion | `src/gaussian_w2/_core/one_sided_taylor.py` |
| RTD degree, support, and block helpers | `src/gaussian_w2/_core/rtd_utils.py` |
| GPU paper estimators | `src/gaussian_w2/_core/fixed_reference_gpu.py` |
| GPU support factors and population truth | `src/gaussian_w2/_core/gpu_linalg.py` |

RTD uses a generated-side sample pilot in both the known-zero-mean Gaussian
and centered-feature experiments. Extrapolation returns intercept weights and
the raw estimate.

The RTD entry points are `rtd_fixed_reference` (CPU) and
`gpu_rtd_fixed_reference` (GPU). Gaussian configs and recorded results use
`general_order` and `general_adaptive` as RTD identifiers.

Experiment runners own data generation, sampling, repetition, and result files.
Each task runs from start to finish. Gaussian results are collected with
`python -m experiments.gaussian.collect`; ImageNet analysis reads the completed
case CSVs directly.
Paper reproduction reads recorded trials and reconstructs the reported numbers.
The notebook calls the same reproduction functions as `python -m reproduce`,
then displays each figure and table in its own block. Table calculations are
grouped by experiment in `reproduce/tables.py`; HTML and LaTeX share the same
headers and numeric rows.
Both use the metric definitions in `gaussian_w2.evaluation`: trial SD and
between-case SD use `ddof=1`, while RMSE uses the mean squared error over trials.
Detailed comparisons with the manuscript live in `tests/test_reproduction.py`.

Paper configurations specify the study design. Copy a configuration to a new
filename to change an experiment, and use a separate output directory.
Keep the resolved configuration with its outputs; the configuration includes
the sample design, numerical options, and random-seed rules.

ImageNet generation adapters live in `imagenet/adapters.py`, embedding models
and preprocessing in `imagenet/extractors.py`, and streaming reference moments
in `imagenet/moments.py` (under `src/gaussian_w2/`).

`tools/make_submission_zip.py` exports the source, configurations, recorded
results, and documentation. It excludes the repository history and locally
generated outputs.

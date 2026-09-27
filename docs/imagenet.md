# ImageNet experiments

The evaluator reads feature matrices and frozen real-reference moments. Image generation and extraction use separate commands and require external images and model checkpoints. Prepare the features using the commands below, then run the paper studies.

## Data contract

Each generated pool is an uncompressed, memory-mappable `.npy` matrix with shape `(pool_size, dimension)`. Store generated features in float32; the estimator converts selected rows to float64. Each reference `.npz` contains `mean`, `covariance`, `sample_count`, `dimension`, the scalar string `covariance_normalization="1/(N-1)"`, and `metadata_json`. The last field is JSON and contains an `extractor_signature`. A generated `.metadata.json` sidecar records the same signature and matrix shape. Paper configurations require these signatures to agree.

Each run manifest records input file paths, sizes, and modification times.

For your own features, copy the study configuration and its `inputs.json`, then set the case paths, counts, and dimensions. A case accepts any embedding name. For custom features without extraction sidecars, set `extractor_signature: null` and `feature_metadata: null`.

The seven generators are StyleGAN-XL at 64, 256, and 512 pixels; DDO EDM2-S at 64 pixels and DDO EDM2-L at 512 pixels; VAR-d30 at 256 pixels and VAR-d36 at 512 pixels. Each has 550,000 generated images, encoded in Inception (2048 dimensions), DINOv2 (1024), and raw projected CLIP ViT-L/14 (768). DINOv2 and CLIP features are **not** L2-normalized. Each resolution uses moments from all 1,281,167 ImageNet training images cropped to that resolution. This is fixed-reference estimation: the real Gaussian is never resampled or bias-corrected.

The historical CLIP preprocessing converts uint8 RGB batches to float32, resizes the shorter side to 224 with antialiased tensor bicubic interpolation (`align_corners=False`), applies a center crop with floor offsets, divides by 255, and uses CLIP's channel means and standard deviations. Both generated features and real-reference moments use this path. It differs from the official PIL preprocessing returned by `clip.load`; numerical equivalence is not assumed. The archived signature's string `official CLIP resize/crop and channel normalization` remains an unchanged **legacy protocol identifier**, not a claim of equivalent preprocessing. The implementation and archived metadata are preserved; correcting this description does not change or regenerate the recorded features.

## Paper study commands

Set generic paths to your data and output locations:

```sh
export FD_DATA_ROOT=/path/to/imagenet_features
export FD_ABLATION_RUN=/path/to/results/imagenet_ablation
export FD_HELDOUT_RUN=/path/to/results/imagenet_heldout
export FD_BUDGET_RUN=/path/to/results/imagenet_budgets

python experiments/imagenet/run.py --config configs/paper/imagenet/ablation.json --output "$FD_ABLATION_RUN"
python experiments/imagenet/run.py --config configs/paper/imagenet/heldout.json --output "$FD_HELDOUT_RUN"
python experiments/imagenet/run.py --config configs/paper/imagenet/budgets_reuse_heldout.json --output "$FD_BUDGET_RUN"
```

Summarize a completed run with:

```sh
python -m experiments.imagenet.analyze --run "$FD_ABLATION_RUN"
python -m experiments.imagenet.analyze --run "$FD_HELDOUT_RUN"
python -m experiments.imagenet.analyze --run "$FD_BUDGET_RUN"
```

The analyzer collects the completed case CSVs and writes `analysis/per_case.csv` and `analysis/summary.csv` using the saved development proxies. Its default target is RTD; `--proxy plugin` or `--proxy fid_infinity` selects a secondary proxy. Run these commands from the repository root.

`inputs.json` is the only shared input description. Each stage names it through `inputs_file`; the loader resolves this one level and writes all cases and settings into `manifest.json`. `estimator_grid` expands the explicitly declared candidate grid into concrete configuration IDs, integer nodes, and weights in the manifest. Selected-stage JSONs specify every selected rule directly.

| Configuration | Sampling | Estimators |
|---|---|---|
| `ablation.json` | 10 draws of 50K without replacement from rows `[0,300000)` | Empirical, RTD, OLS/VA orders 1–4 under every listed grid, plus OLS orders 5–8 with 15 uniform-n nodes |
| `heldout.json` | Five disjoint 50K blocks after development | Empirical, FID-infinity, selected VA orders 1–3, RTD |
| `heldout_full_grid.json` | Same five heldout blocks | The complete 102-label bank |
| `budgets_reuse_heldout.json` | Heldout prefixes at 10K, 20K, 30K, 40K, and 50K | New 10K–40K records, exact imported 50K records from `FD_HELDOUT_RUN` |

The full bank has 102 labels: 96 OLS/VA combinations, four additional higher-order OLS rules, empirical, and RTD. For orders 1–4, the grid has all three schedules (`uniform_n`, `uniform_inverse_n`, `chebyshev_inverse_n`) and point counts `p+1`, 10, 15, 20. Equivalent labels share evaluated nodes and are grouped under canonical configurations in the archived reports. All stages use minimum sample size 5K. RTD uses `m0=floor(3*N/5)` pilot contrasts, where `N` is the number of generated features; its correction count is `N-1-m0` after the Helmert transform. Its adaptive degree uses `rho_design=0.25`, `C_T=1`, and no degree cap.

`development_selection.json` records the selected configurations, nodes, and weights. The rule minimizes mean absolute RMSE across seven development cases within each embedding/order; ties use median RMSE and canonical configuration ID. The selected VA rules are:

| Embedding | Order 1 | Order 2 | Order 3 |
|---|---|---|---|
| Inception | uniform n, 2 nodes | Chebyshev in 1/n, 15 | Chebyshev in 1/n, 20 |
| DINOv2 | uniform n, 2 nodes | Chebyshev in 1/n, 15 | Chebyshev in 1/n, 20 |
| CLIP | uniform 1/n, 15 | uniform 1/n, 20 | uniform n, 20 |

FID-infinity is OLS order 1 with 15 uniform-n nodes. Heldout evaluation uses the configurations selected on development data.

The development run computes three 300K proxy centers: empirical; the median of five randomized FID-infinity fits; and the median of five independently permuted full-pool RTD runs with a 180K pilot. The repeated proxies measure algorithm variability within one shared 300K pool. The primary target is the RTD median. Heldout and budget stages import these frozen development proxies. All five RTD proxy repetitions are required to compute their median.

`--case-id CASE_ID` runs one complete generator/embedding case, including its proxies, all budgets, and all trials, and is suitable for a job array. An interrupted case is rerun from the beginning with this option. Completed cases have a `trials.CASE_ID.csv` file and reject another run at the same output location. After all cases finish, use the analyzer above to summarize them.

## Sampling and saved results

Generated row `i` has `class_id=i%1000` and within-class sample number `i//1000`. Rows `[0,300000)` are development, followed by `[300000,350000)`, …, `[500000,550000)` for the five heldout blocks.

The estimator master seed for a resolution is `20260810 + resolution`. Development outer sampling uses NumPy `SeedSequence([master, code, trial, 0xAB1A])`; heldout uses `[master, code, trial]`. Here `code` is the first four bytes of SHA-256(generator name), interpreted little-endian; the derived seed is one uint32 state word. Budget pools are prefixes of the same permutation of the full 50K heldout block, not separately shuffled blocks.

The per-node seed is the first 16 hexadecimal digits of SHA-256 of compact, sorted-key JSON `[master,generator,phase,repetition,n]`, interpreted as an integer. Phases are explicit: `development_nodes`, `heldout_nodes`, and `heldout_budget_nodes`; proxy phases are `plugin`, `fid_infinity`, and `rtd`. Generator identity and seeds agree across embeddings. Every n below N uses a separate `default_rng(seed).permutation(N)[:n]`; these are **not nested prefixes** of one node permutation. The full endpoint uses all N rows in their original outer order. All orders, weighting rules, and schedules reuse the same node at a given case/trial/N/n. The empirical estimate uses the shared full endpoint. The 300K endpoint is also shared across proxy repetitions.

Each output includes:

- `manifest.json`: expanded configuration, concrete nodes/weights, input file metadata, and numerical environment.
- `cases/CASE/proxies.json`: frozen proxy values and repeated proxy records; `fid_infinity_runs[].nodes` stores the raw FID-infinity nodes.
- `cases/CASE/nN/trial_TRIAL.json`: estimator records, raw `nodes`, configuration, sample budget, outer block, sampling seed, and numerical environment. Each node includes its sample size, seed, runtime, and estimate.
- `trials.CASE.csv`: all estimates for one case, published atomically after its budgets and trials finish.

Saved records are matched to the experiment configuration and sample identity; trial coverage and numerical outputs are checked. Runtime for an extrapolator is the sum of its required node times plus its own fit time; nodes are shared computationally but charged to each method's standalone runtime. RTD is measured separately. Estimates are retained as computed. Numerical failures are recorded with their exception and a null estimate/runtime. Metrics require all trials in the corresponding cell to succeed.

The budget experiment evaluates 10K–40K and imports the 50K records from the heldout comparison after checking matching inputs and configuration. Both tables therefore share the same 50K results.

## Generate all features from images and models

Use a CUDA machine with sufficient memory and disk. The optional image stack is separate from the NumPy/SciPy estimator environment. The source generation setup used Python 3.12, PyTorch 2.7.1, torchvision 0.22.1, NumPy 1.26.4 and CUDA 12.6 wheels. Dependencies include Pillow, click, requests, tqdm, ninja, psutil, regex, ftfy, packaging, imageio, imageio-ffmpeg, dill, einops, absl-py, typed-argument-parser, timm 0.4.12, and huggingface_hub 0.34.4. Install official OpenAI CLIP from the pinned checkout below. Estimator GPU requirements and recorded measurement environments are described in `environments.md`.

```sh
export FD_UPSTREAM_ROOT=/path/to/upstream
export FD_MODEL_CACHE=/path/to/checkpoints
export FD_DATA_ROOT=/path/to/imagenet_features
export IMAGENET_TRAIN=/path/to/imagenet/train

python -m pip install -e ".[images]"
python experiments/imagenet/fetch_models.py --config configs/paper/imagenet/models.json --manifest "$FD_MODEL_CACHE/download_manifest.json"
python -m pip install --no-deps -e "$FD_UPSTREAM_ROOT/openai_clip"
```

`models.json` contains the exact source registry used here. The fetch command obtains repositories at the declared revisions, downloads missing checkpoints, and records model sources and local file metadata in a manifest. Upstream code/checkpoints retain their original licenses and are not redistributed or relicensed here.

| Upstream | Revision |
|---|---|
| `autonomousvision/stylegan_xl` | `4241ff9cfeb69d617427107a75d69e9d1c2d92f2` |
| `NVlabs/edm2` | `4bf8162f601bcc09472ce8a32dd0cbe8889dc8fc` |
| `FoundationVision/VAR` | `78b95394fc5896192e3a003e4b295f8ea743c48f` |
| `openai/CLIP` | `d50d76daa670286dd6cacf3bcd80b5e4823fc8e1` |
| `facebookresearch/dinov2` | `7764ea0f912e53c92e82eb78a2a1631e92725fc8` |

StyleGAN checkpoint URLs are the original `https://s3.eu-central-1.amazonaws.com/avg-projects/stylegan_xl/models/imagenet{64,256,512}.pkl`. DDO uses Hugging Face `nvidia/DirectDiscriminativeOptimization` revision `49257a0ed81726645e274e2fd5ac8e38a3ca7826`, files `edm2-img64-s-ddo.pkl` and `edm2-img512-l-ddo.pkl`. VAR uses Hugging Face `FoundationVision/var` revision `6d0ee6598a42f75079aa79e58b012b57b284c91b`, files `var_d30.pth`, `var_d36.pth`, and `vae_ch160v4096z32.pth`. The machine-readable registry provides complete URLs or repository/filename/revision triples.

The Inception detector downloads `https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl`. CLIP uses official `clip.load("ViT-L/14", jit=False)`. DINOv2 uses `dinov2_vitl14` from Torch Hub at `facebookresearch/dinov2:7764ea0f912e53c92e82eb78a2a1631e92725fc8`. `models.json` records this source version and the checkpoint URLs. To use a different DINO model source, set `dino_repository` to a GitHub repository and revision with `dino_source: "github"`, or to a local checkout with `dino_source: "local"`, and update the shared input signature to match the generated feature metadata.

Run every declared lane in its own process; each lane immediately extracts all three embeddings and does not retain images. For example:

```sh
python experiments/imagenet/prepare.py generate --config configs/paper/imagenet/preparation.json --pool stylegan_xl_imagenet64 --lane 0
python experiments/imagenet/prepare.py merge --config configs/paper/imagenet/preparation.json --pool stylegan_xl_imagenet64
```

Repeat generation for each lane in each pool's `lane_ranges`, then merge that pool. There are 22 lanes across the seven generators. The configuration specifies lane boundaries, row order, batch sizes, model presets, and pool seed starts. StyleGAN/EDM2 use per-image seeds. VAR seeds each batch with its first image seed; subsequent draws depend on position within that batch. Keep the configured VAR lane boundaries and batch sizes for reproduction. Each lane runs from its first row and publishes its feature arrays and `complete.json` only after all batches finish.

For the real references:

```sh
python experiments/imagenet/prepare.py reference --config configs/paper/imagenet/preparation.json --resolution 64
python experiments/imagenet/prepare.py reference --config configs/paper/imagenet/preparation.json --resolution 256
python experiments/imagenet/prepare.py reference --config configs/paper/imagenet/preparation.json --resolution 512
```

Supply the ImageNet-1K training data under its applicable access terms. A sorted directory tree or ZIP of images is supported. Raw images receive the EDM2 Dhariwal crop: repeated BOX halving, BICUBIC resize of the shorter side, then centered square crop. `source_format: "edm2"` instead accepts images already preprocessed at the requested resolution. Streaming Chan accumulators compute float64 sample moments without retaining feature rows. Run reference extraction separately from each generator process to avoid upstream Python module-name collisions.

Merging writes a `complete.json` marker for the pool; reference extraction writes `imagenetRESOLUTION.complete.json` after saving all three embeddings. An interrupted lane, merge, or reference extraction can be rerun from the beginning. Completed tasks reject overwrite.

After merging, generated matrices are under `FD_DATA_ROOT/features/GENERATOR/EMBEDDING.npy`; real moments are under `FD_DATA_ROOT/references/imagenetRESOLUTION.EMBEDDING.npz`, matching `inputs.json`. Reference counts, dimensions, and extractor signatures are checked before evaluation.

See [the environment guide](environments.md) for environment and validation details.

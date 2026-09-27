# Environments

`requirements-report.txt` pins the CPU reporting dependencies validated with
Python 3.9.6. Use this environment for paper reporting and the small Gaussian
experiment.
The optional `notebook` extra adds JupyterLab and the notebook execution client.

The final repeated-Gaussian and ImageNet RTD budget runs recorded this environment:

| Component | Recorded value |
| --- | --- |
| GPU | NVIDIA GH200 120GB |
| CPU | ARM Neoverse-V2, 72 logical CPUs |
| Architecture | Linux aarch64 |
| Python | 3.12.11 |
| NumPy | 2.3.5 |
| SciPy | 1.18.0 |
| pandas | 3.0.5 |
| PyYAML | 6.0.3 |
| CuPy | 14.1.1 (`cupy-cuda12x`) |
| CUDA runtime linked to CuPy | 12.9 |
| Local CUDA toolkit / NVRTC | 12.6 |

Repository validation covers the Gaussian CPU experiment, numerical tests, and
reconstruction of the paper results from bundled measurements. Image generation
uses a separate environment; see [the ImageNet guide](imagenet.md).

For a compatible Linux NVIDIA machine with Python 3.10 or newer, install the
`gpu` extra. CuPy additionally needs compatible CUDA libraries and a driver;
follow the [official installation instructions](https://docs.cupy.dev/en/stable/install.html)
for the machine being used. Use a separate environment for CuPy 14, which
requires NumPy 2; the CPU reporting environment uses NumPy 1.26.

The paper configurations specify the sampling rules, backend, generation batch
layout, and seeds. Use them with the recorded GPU environment for the paper
experiments. Bundled trial data provide the inputs for reconstructing the
reported statistics and timings.

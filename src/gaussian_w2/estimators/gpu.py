"""Optional float64 CuPy estimators; importing this module does not load CUDA."""
from .._core.gpu_backend import GPUUnavailableError, require_cupy, gpu_available
from .._core.gpu_linalg import gpu_gaussian_w2_sq_truth
from .._core.fixed_reference_gpu import (
    GPUFixedReferenceStatistics, prepare_gpu_fixed_reference,
    prepare_gpu_fixed_reference_from_statistics, gpu_fixed_reference_empirical,
    gpu_rtd_fixed_reference, gpu_fixed_reference_plugin_curve,
)
__all__ = [
    "GPUUnavailableError", "require_cupy", "gpu_available", "gpu_gaussian_w2_sq_truth",
    "GPUFixedReferenceStatistics", "prepare_gpu_fixed_reference",
    "prepare_gpu_fixed_reference_from_statistics", "gpu_fixed_reference_empirical",
    "gpu_rtd_fixed_reference", "gpu_fixed_reference_plugin_curve",
]

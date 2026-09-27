"""Lazy access to the optional CuPy/CUDA backend."""
from __future__ import annotations

import importlib
from typing import Any


class GPUUnavailableError(RuntimeError):
    """A CUDA operation was requested without a usable CuPy GPU."""


def require_cupy() -> Any:
    """Import CuPy on demand and require at least one visible CUDA device."""
    try:
        cp = importlib.import_module("cupy")
    except Exception as error:
        raise GPUUnavailableError("Install a CUDA-compatible CuPy wheel to use the GPU backend.") from error
    try:
        count = cp.cuda.runtime.getDeviceCount()
    except Exception as error:
        raise GPUUnavailableError("CuPy could not access the CUDA runtime.") from error
    if count < 1:
        raise GPUUnavailableError("No CUDA device is visible.")
    return cp


def gpu_available() -> bool:
    """Return whether CuPy and a visible CUDA device are available."""
    try:
        require_cupy()
    except GPUUnavailableError:
        return False
    return True

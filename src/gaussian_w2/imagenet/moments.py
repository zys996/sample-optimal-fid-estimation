"""Streaming reference moments accumulated on the Torch device in float64.

Torch is imported only when an accumulator is constructed.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _positive_dimension(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("dimension must be a positive integer")
    try:
        dimension = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("dimension must be a positive integer") from error
    if dimension < 1 or dimension != value:
        raise ValueError("dimension must be a positive integer")
    return dimension


class MomentState:
    """Count, float64 mean, and centered scatter returned by the accumulator.

    ``m2 = sum_i (x_i - mean) (x_i - mean)^T``. Reference extraction updates
    these quantities with :class:`TorchChanMoments` and stores them on CPU.
    """

    def __init__(
        self,
        dimension: int,
        *,
        count: int = 0,
        mean: Optional[np.ndarray] = None,
        m2: Optional[np.ndarray] = None,
    ) -> None:
        self.dimension = _positive_dimension(dimension)
        if isinstance(count, (bool, np.bool_)) or int(count) != count or int(count) < 0:
            raise ValueError("count must be a nonnegative integer")
        self.count = int(count)
        self.mean = (
            np.zeros(self.dimension, dtype=np.float64)
            if mean is None
            else np.array(mean, dtype=np.float64, copy=True)
        )
        self.m2 = (
            np.zeros((self.dimension, self.dimension), dtype=np.float64)
            if m2 is None
            else np.array(m2, dtype=np.float64, copy=True)
        )
        if self.mean.shape != (self.dimension,):
            raise ValueError("mean shape does not match dimension")
        if self.m2.shape != (self.dimension, self.dimension):
            raise ValueError("m2 shape does not match dimension")
        if not np.all(np.isfinite(self.mean)) or not np.all(np.isfinite(self.m2)):
            raise ValueError("moment state contains NaN or infinite values")
        if self.count == 0 and (
            np.any(self.mean != 0.0) or np.any(self.m2 != 0.0)
        ):
            raise ValueError("an empty moment state must have zero mean and m2")
        self.m2 = (self.m2 + self.m2.T) * 0.5

    def covariance(self, *, ddof: int = 1) -> FloatArray:
        if isinstance(ddof, (bool, np.bool_)) or int(ddof) != ddof or int(ddof) < 0:
            raise ValueError("ddof must be a nonnegative integer")
        denominator = self.count - int(ddof)
        if denominator < 1:
            raise ValueError("not enough samples for the requested covariance ddof")
        covariance = self.m2 / float(denominator)
        return np.asarray((covariance + covariance.T) * 0.5, dtype=np.float64)


class TorchChanMoments:
    """GPU-capable Chan accumulator with lazy Torch dependency.

    Production ImageNet preparation keeps the expensive Gram products on the
    GPU in float64.  :meth:`to_numpy` returns the exact same state type used by
    the NumPy covariance checks.
    """

    def __init__(self, dimension: int, *, device: Any = "cuda") -> None:
        try:
            import torch
        except ImportError as error:  # pragma: no cover - exercised on server
            raise RuntimeError("TorchChanMoments requires PyTorch") from error
        self._torch = torch
        self.dimension = _positive_dimension(dimension)
        self.device = torch.device(device)
        self.count = 0
        self.mean = torch.zeros(
            self.dimension, dtype=torch.float64, device=self.device
        )
        self.m2 = torch.zeros(
            (self.dimension, self.dimension), dtype=torch.float64, device=self.device
        )

    def update(self, values: Any) -> "TorchChanMoments":
        torch = self._torch
        if not isinstance(values, torch.Tensor):
            values = torch.as_tensor(values)
        if values.ndim != 2 or int(values.shape[1]) != self.dimension:
            raise ValueError(
                f"feature batch must have shape (n,{self.dimension}); "
                f"got {tuple(values.shape)}"
            )
        other_count = int(values.shape[0])
        if other_count == 0:
            return self
        batch = values.detach().to(
            device=self.device, dtype=torch.float64, non_blocking=True
        )
        if not bool(torch.isfinite(batch).all().item()):
            raise ValueError("feature batch contains NaN or infinite values")
        other_mean = batch.mean(dim=0)
        centered = batch - other_mean
        other_m2 = centered.T @ centered
        if self.count == 0:
            self.mean.copy_(other_mean)
            self.m2.copy_(other_m2)
            self.count = other_count
            return self
        total = self.count + other_count
        delta = other_mean - self.mean
        self.m2.add_(other_m2)
        self.m2.add_(
            torch.outer(delta, delta),
            alpha=float(self.count * other_count) / float(total),
        )
        self.mean.add_(delta, alpha=float(other_count) / float(total))
        self.count = total
        return self

    def to_numpy(self) -> MomentState:
        return MomentState(
            self.dimension,
            count=self.count,
            mean=self.mean.detach().cpu().numpy(),
            m2=self.m2.detach().cpu().numpy(),
        )


__all__ = ["MomentState", "TorchChanMoments"]

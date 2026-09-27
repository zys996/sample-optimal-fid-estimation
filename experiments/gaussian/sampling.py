"""Random streams and Haar bases for the three random Gaussian families."""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
RNGInput = Optional[Union[int, np.integer, np.random.Generator, np.random.SeedSequence]]


def make_rng(
    rng: RNGInput = None,
    *,
    seed: Optional[Union[int, np.integer, np.random.SeedSequence]] = None,
) -> np.random.Generator:
    """Resolve either an existing generator or a seed into a Generator."""

    if rng is not None and seed is not None:
        raise ValueError("pass either rng or seed, not both")
    source = seed if seed is not None else rng
    if isinstance(source, np.random.Generator):
        return source
    return np.random.default_rng(source)


def random_orthogonal(dimension: int, rng: RNGInput = None) -> FloatArray:
    """Draw a Haar-distributed orthogonal matrix via sign-corrected QR."""

    if not isinstance(dimension, (int, np.integer)) or int(dimension) <= 0:
        raise ValueError("dimension must be a positive integer")
    generator = make_rng(rng)
    gaussian = generator.standard_normal((int(dimension), int(dimension)))
    q, r = np.linalg.qr(gaussian)
    diagonal = np.diag(r)
    signs = np.where(diagonal < 0.0, -1.0, 1.0)
    return np.asarray(q * signs[np.newaxis, :], dtype=np.float64)



__all__ = ["make_rng", "random_orthogonal"]

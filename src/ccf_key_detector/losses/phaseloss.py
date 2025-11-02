"""Phase-alignment loss for interval torus representations."""

from __future__ import annotations

import numpy as np


def phase_loss(
    torus_a: np.ndarray,
    torus_b: np.ndarray,
    *,
    allow_inversion: bool = False,
) -> float:
    """Compute the minimum squared difference after circular alignment.

    Parameters
    ----------
    torus_a, torus_b:
        Square arrays representing interval torus surfaces. ``torus_b`` is
        shifted along the θ-axis to minimise the Frobenius norm.
    allow_inversion:
        If ``True``, additionally minimise over inverted copies of ``torus_b``.
    """

    a = np.asarray(torus_a, dtype=np.float64)
    b = np.asarray(torus_b, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or a.shape != b.shape or a.shape[0] != a.shape[1]:
        raise ValueError("Input torus arrays must be square and share the same shape")

    n = a.shape[0]
    best = np.inf

    for shift in range(n):
        rolled = np.roll(b, -shift, axis=0)
        diff = a - rolled
        best = min(best, float(np.sum(diff * diff)))

        if allow_inversion:
            flipped = np.flip(rolled, axis=0)
            diff_inv = a - flipped
            best = min(best, float(np.sum(diff_inv * diff_inv)))

    return best


__all__ = ["phase_loss"]

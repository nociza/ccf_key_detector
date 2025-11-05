"""Phase-alignment loss for interval torus representations."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch


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


def phase_loss_torch(
    torus_a: torch.Tensor,
    torus_b: torch.Tensor,
    *,
    allow_inversion: bool = False,
) -> torch.Tensor:
    """Torch-compatible version of :func:`phase_loss` supporting autograd."""

    if torus_a.shape != torus_b.shape:
        raise ValueError("torus_a and torus_b must share the same shape")
    if torus_a.ndim == 2:
        torus_a = torus_a.unsqueeze(0)
        torus_b = torus_b.unsqueeze(0)
    if torus_a.ndim != 3:
        raise ValueError("Torus tensors must be 2-D or 3-D")

    _, n_bins, _ = torus_a.shape
    losses: Sequence[torch.Tensor] = []

    for shift in range(n_bins):
        rolled = torch.roll(torus_b, shifts=-shift, dims=1)
        diff = (torus_a - rolled) ** 2
        losses.append(diff.sum(dim=(1, 2)))
        if allow_inversion:
            flipped = torch.flip(rolled, dims=[1])
            diff_inv = (torus_a - flipped) ** 2
            losses.append(diff_inv.sum(dim=(1, 2)))

    stacked = torch.stack(losses, dim=0)
    return stacked.min(dim=0).values.mean()


__all__ = ["phase_loss", "phase_loss_torch"]

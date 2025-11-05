"""Interval torus utilities."""

from __future__ import annotations

from typing import Sequence, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    import torch

def build_torus(pdf: np.ndarray) -> np.ndarray:
    """Construct the interval torus ``M(θ, δ) = p(θ) p(θ ⊕ δ)``."""

    pdf = np.asarray(pdf, dtype=np.float64)
    if pdf.ndim != 1:
        raise ValueError("pdf must be a 1-D array")
    if pdf.size == 0:
        raise ValueError("pdf cannot be empty")

    n_bins = pdf.size
    torus = np.empty((n_bins, n_bins), dtype=np.float64)
    for delta in range(n_bins):
        torus[:, delta] = pdf * np.roll(pdf, -delta)
    return torus


def column_sums(torus: np.ndarray) -> np.ndarray:
    """Return the interval marginal ``g(δ)`` from the torus."""

    torus = np.asarray(torus, dtype=np.float64)
    if torus.ndim != 2 or torus.shape[0] != torus.shape[1]:
        raise ValueError("torus must be a square matrix")
    return torus.sum(axis=0)


def rotate_torus(torus: np.ndarray, shift: int) -> np.ndarray:
    """Rotate the torus along the θ-axis by ``shift`` bins."""

    torus = np.asarray(torus, dtype=np.float64)
    if torus.ndim != 2 or torus.shape[0] != torus.shape[1]:
        raise ValueError("torus must be a square matrix")
    return np.roll(torus, -shift % torus.shape[0], axis=0)


def build_torus_torch(pdf: "torch.Tensor") -> "torch.Tensor":
    """Torch variant of :func:`build_torus` retaining gradients."""

    import torch  # Lazy import to keep numpy-only environments lightweight

    if pdf.ndim == 1:
        pdf = pdf.unsqueeze(0)
    if pdf.ndim != 2:
        raise ValueError("pdf must be 1-D or 2-D tensor")
    batch, n_bins = pdf.shape
    torus_columns: Sequence[torch.Tensor] = [
        torch.roll(pdf, shifts=-delta, dims=1) for delta in range(n_bins)
    ]
    rolled = torch.stack(torus_columns, dim=1)  # (batch, n_bins, n_bins)
    torus = pdf.unsqueeze(1) * rolled
    return torus

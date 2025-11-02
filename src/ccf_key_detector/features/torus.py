"""Interval torus utilities."""

from __future__ import annotations

import numpy as np


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

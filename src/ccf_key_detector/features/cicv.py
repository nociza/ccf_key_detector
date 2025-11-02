"""Circular interval content (CICV) feature utilities."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def compute_cicv(pdf: np.ndarray, *, normalize: bool = True) -> np.ndarray:
    """Compute the directed CICV from a chroma PDF.

    Parameters
    ----------
    pdf:
        1-D probability mass function defined on the circular log-chroma grid.
    normalize:
        If ``True`` (default) the output is scaled to sum to 1.
    """

    pdf = np.asarray(pdf, dtype=np.float64)
    if pdf.ndim != 1:
        raise ValueError("pdf must be a 1-D array")
    if pdf.size == 0:
        raise ValueError("pdf cannot be empty")

    # Wiener–Khinchin on the circle: g = ifft(|FFT(p)|^2).
    spectrum = np.fft.fft(pdf)
    power = np.abs(spectrum) ** 2
    cicv = np.fft.ifft(power).real
    cicv = np.clip(cicv, 0.0, None)

    if normalize:
        total = cicv.sum()
        if total > 0.0:
            cicv = cicv / total
    return cicv


def fold_cicv(cicv: np.ndarray, *, normalize: bool = True) -> np.ndarray:
    """Fold the directed CICV onto the unsigned interval domain.

    The folded representation resides on ``[0, 1/2]`` (discrete bins).
    """

    cicv = np.asarray(cicv, dtype=np.float64)
    if cicv.ndim != 1:
        raise ValueError("cicv must be a 1-D array")
    n_bins = cicv.size
    if n_bins == 0:
        raise ValueError("cicv cannot be empty")

    half = n_bins // 2
    folded = np.zeros(half + 1, dtype=np.float64)
    folded[0] = cicv[0]
    for k in range(1, half + 1):
        partner = n_bins - k
        if k == partner:  # Only occurs for even ``n_bins`` at Nyquist.
            folded[k] = cicv[k]
        else:
            folded[k] = cicv[k] + cicv[partner]

    if normalize:
        total = folded.sum()
        if total > 0.0:
            folded = folded / total
    return folded


def rotate_pdf(pdf: np.ndarray, shift: int) -> np.ndarray:
    """Rotate a circular PDF by ``shift`` bins upward."""

    pdf = np.asarray(pdf, dtype=np.float64)
    if pdf.ndim != 1:
        raise ValueError("pdf must be a 1-D array")
    return np.roll(pdf, -shift)


def cicv_pair(pdf: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convenience helper returning both directed and folded CICVs."""

    directed = compute_cicv(pdf)
    folded = fold_cicv(directed)
    return directed, folded


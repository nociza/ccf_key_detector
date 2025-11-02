"""Tonic saliency (center-field) utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np

_DEFAULT_INTERVALS: Tuple[Tuple[float, float], ...] = (
    (0.0, 1.0),  # unison
    (np.log2(3 / 2) % 1.0, 0.8),  # perfect fifth
    ((1.0 - np.log2(3 / 2)) % 1.0, 0.6),  # perfect fourth
    (np.log2(5 / 4) % 1.0, 0.35),  # major third
    (np.log2(6 / 5) % 1.0, 0.3),  # minor third
)


@dataclass(frozen=True)
class CenterFieldConfig:
    """Configuration for center-field computation."""

    intervals: Tuple[Tuple[float, float], ...] = _DEFAULT_INTERVALS
    kernel_sigma_bins: float = 3.0


def center_field(pdf: np.ndarray, config: CenterFieldConfig | None = None) -> Tuple[np.ndarray, float, float]:
    """Compute the center-field ``C(θ) = p(θ) (W * p)(θ)`` and circular stats.

    Returns
    -------
    center:
        Center-field curve on the unit log-chroma circle.
    mu:
        Circular mean (fraction of an octave).
    rho:
        Circular concentration in [0, 1].
    """

    pdf = np.asarray(pdf, dtype=np.float64)
    if pdf.ndim != 1:
        raise ValueError("pdf must be a 1-D array")
    if pdf.size == 0:
        raise ValueError("pdf cannot be empty")

    if config is None:
        config = CenterFieldConfig()

    kernel = _build_kernel(pdf.size, config)
    # Circular convolution via FFT.
    neighborhood = np.fft.ifft(np.fft.fft(kernel) * np.fft.fft(pdf)).real
    neighborhood = np.clip(neighborhood, 0.0, None)

    center = pdf * neighborhood
    center_sum = center.sum()

    theta = np.linspace(0.0, 1.0, num=pdf.size, endpoint=False)
    if center_sum > 0.0:
        first_moment = np.sum(center * np.exp(2j * np.pi * theta)) / center_sum
        mu = (np.angle(first_moment) / (2 * np.pi)) % 1.0
        rho = float(abs(first_moment))
    else:
        mu = 0.0
        rho = 0.0

    return center, float(mu), rho


def _build_kernel(n_bins: int, config: CenterFieldConfig) -> np.ndarray:
    """Construct the tonic-support kernel ``W`` on the circle."""

    x = np.linspace(0.0, 1.0, num=n_bins, endpoint=False)
    kernel = np.zeros(n_bins, dtype=np.float64)
    sigma = max(config.kernel_sigma_bins, 1e-6)
    for interval, weight in config.intervals:
        distances = np.abs(x - (interval % 1.0))
        distances = np.minimum(distances, 1.0 - distances)
        kernel += weight * np.exp(-0.5 * (distances * n_bins / sigma) ** 2)

    total = kernel.sum()
    if total > 0.0:
        kernel /= total
    else:
        kernel[:] = 1.0 / n_bins
    return kernel

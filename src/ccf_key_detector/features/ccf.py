"""Continuous chroma feature extraction utilities."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Protocol, Sequence, Tuple

import numpy as np

_NUMERIC_EPS = 1e-12


@dataclass(frozen=True)
class CCFConfig:
    """Configuration for continuous chroma extraction."""

    n_bins: int = 120
    f_ref_hz: float = 220.0
    smoothing_sigma_bins: float = 1.5
    window: str = "hann"
    silence_rms_floor: float = 1e-6

    def __post_init__(self) -> None:
        if self.n_bins <= 0:
            raise ValueError("n_bins must be positive")
        if self.f_ref_hz <= 0:
            raise ValueError("f_ref_hz must be positive")
        if self.smoothing_sigma_bins < 0:
            raise ValueError("smoothing_sigma_bins cannot be negative")
        if not self.window:
            raise ValueError("window name must be non-empty")
        if self.silence_rms_floor <= 0:
            raise ValueError("silence_rms_floor must be positive")

    def to_dict(self) -> Dict[str, float | int | str]:
        return {
            "n_bins": self.n_bins,
            "f_ref_hz": self.f_ref_hz,
            "smoothing_sigma_bins": self.smoothing_sigma_bins,
            "window": self.window,
            "silence_rms_floor": self.silence_rms_floor,
        }


class CCFExtractor(Protocol):
    """Protocol describing the continuous chroma feature extractor."""

    config: CCFConfig

    def compute(self, frame: Sequence[float] | memoryview | np.ndarray, sample_rate: int) -> Tuple[list[float], Dict[str, float]]:
        """Return a PDF on the unit log-frequency cycle plus diagnostics."""


class _CCFExtractor:
    """Concrete extractor implementing the protocol."""

    def __init__(self, config: CCFConfig) -> None:
        self.config = config

    def compute(self, frame: Sequence[float] | memoryview | np.ndarray, sample_rate: int) -> Tuple[list[float], Dict[str, float]]:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")

        samples = _as_float32(frame)
        if samples.ndim != 1:
            raise ValueError("CCF extractor expects a 1-D mono frame")

        mono = samples.astype(np.float64, copy=False)
        fft_size = mono.size
        if fft_size == 0:
            return _uniform_pdf(self.config.n_bins), self._diagnostics(0.0, 1.0, 0.0, fft_size)

        window = _window(self.config.window, fft_size)
        windowed = mono * window
        spectrum = np.fft.rfft(windowed)
        magnitudes = np.abs(spectrum)
        power = magnitudes**2

        if power.size:
            power[0] = 0.0

        rms = float(np.sqrt(np.mean(mono**2)))
        spectral_flatness = _spectral_flatness(power)
        energy_sum = float(np.sum(power))

        pdf = _project_to_cycle(
            power=power,
            sample_rate=sample_rate,
            fft_size=fft_size,
            n_bins=self.config.n_bins,
            f_ref_hz=self.config.f_ref_hz,
        )

        if self.config.smoothing_sigma_bins > 0.0:
            pdf = _circular_gaussian_smooth(pdf, self.config.smoothing_sigma_bins)

        total = float(np.sum(pdf))
        if not np.isfinite(total) or total <= _NUMERIC_EPS:
            pdf = _uniform_pdf(self.config.n_bins)
        else:
            pdf = pdf / total

        diagnostics = self._diagnostics(rms, spectral_flatness, energy_sum, fft_size)
        return pdf.astype(np.float64).tolist(), diagnostics

    def _diagnostics(
        self,
        rms: float,
        spectral_flatness: float,
        energy_sum: float,
        fft_size: int,
    ) -> Dict[str, float]:
        return {
            "rms": float(rms),
            "spectral_flatness": float(spectral_flatness),
            "raw_energy": float(energy_sum),
            "fft_size": float(fft_size),
            "n_bins": float(self.config.n_bins),
        }


def _as_float32(frame: Sequence[float] | memoryview | np.ndarray) -> np.ndarray:
    """Convert an arbitrary buffer into a 1-D float32 NumPy array."""

    if isinstance(frame, np.ndarray):
        return frame.astype(np.float32, copy=False)

    if isinstance(frame, memoryview):
        is_contiguous = bool(getattr(frame, "c_contiguous", getattr(frame, "contiguous", True)))
        if frame.format == "f" and is_contiguous:
            return np.frombuffer(frame, dtype=np.float32)
        if frame.format in {"d", "g"} and is_contiguous:
            return np.frombuffer(frame, dtype=np.float64).astype(np.float32)
        return np.array(frame.tolist(), dtype=np.float32)

    return np.asarray(frame, dtype=np.float32)


@lru_cache(maxsize=8)
def _window(name: str, size: int) -> np.ndarray:
    """Retrieve a deterministic analysis window."""

    name_lc = name.lower()
    if name_lc == "hann":
        win = np.hanning(size)
    elif name_lc == "blackman":
        win = np.blackman(size)
    elif name_lc in {"rect", "rectangular"}:
        win = np.ones(size, dtype=np.float64)
    else:
        raise ValueError(f"Unsupported window: {name}")
    return win.astype(np.float64)


def _spectral_flatness(power: np.ndarray) -> float:
    """Compute spectral flatness, guarding against underflow."""

    eps = _NUMERIC_EPS
    positive = power[power > 0.0]
    if positive.size == 0:
        return 1.0
    log_mean = float(np.mean(np.log(positive + eps)))
    arithmetic_mean = float(np.mean(positive))
    flatness = np.exp(log_mean) / (arithmetic_mean + eps)
    return float(np.clip(flatness, 0.0, 1.0))


def _project_to_cycle(
    *,
    power: np.ndarray,
    sample_rate: int,
    fft_size: int,
    n_bins: int,
    f_ref_hz: float,
) -> np.ndarray:
    """Wrap spectral power onto the unit log-frequency cycle."""

    freqs = np.fft.rfftfreq(fft_size, d=1.0 / sample_rate)
    if freqs.size != power.size:
        raise ValueError("Frequency and power arrays must align in size")

    positive_mask = freqs > 0.0
    if not np.any(positive_mask):
        return _uniform_pdf(n_bins)

    freqs = freqs[positive_mask]
    power = power[positive_mask]

    log_coords = np.log2(freqs / f_ref_hz)
    fractional = np.mod(log_coords, 1.0)
    positions = fractional * n_bins

    lower_idx = np.floor(positions).astype(int) % n_bins
    fractions = positions - np.floor(positions)
    upper_idx = (lower_idx + 1) % n_bins

    pdf = np.zeros(n_bins, dtype=np.float64)
    np.add.at(pdf, lower_idx, power * (1.0 - fractions))
    np.add.at(pdf, upper_idx, power * fractions)

    if np.allclose(pdf, 0.0):
        return _uniform_pdf(n_bins)
    return pdf


def _circular_gaussian_smooth(pdf: np.ndarray, sigma_bins: float) -> np.ndarray:
    """Apply circular Gaussian smoothing using FFT convolution."""

    if sigma_bins <= 0.0:
        return pdf

    n_bins = pdf.size
    kernel = _circular_gaussian_kernel(n_bins, sigma_bins)
    smoothed = np.fft.ifft(np.fft.fft(pdf) * np.fft.fft(kernel)).real
    smoothed[smoothed < 0.0] = 0.0
    return smoothed


def _circular_gaussian_kernel(n_bins: int, sigma_bins: float) -> np.ndarray:
    """Construct a circularly wrapped Gaussian kernel with unit area."""

    indices = np.arange(n_bins, dtype=np.float64)
    wrapped_distance = np.minimum(indices, n_bins - indices)
    kernel = np.exp(-0.5 * (wrapped_distance / sigma_bins) ** 2)
    kernel_sum = np.sum(kernel)
    if kernel_sum <= _NUMERIC_EPS:
        return np.ones(n_bins, dtype=np.float64) / float(n_bins)
    return kernel / kernel_sum


def _uniform_pdf(n_bins: int) -> np.ndarray:
    return np.full(n_bins, 1.0 / float(n_bins), dtype=np.float64)


def create_extractor(config: CCFConfig) -> CCFExtractor:
    """Factory for the continuous chroma extractor."""

    return _CCFExtractor(config)


__all__ = [
    "CCFConfig",
    "CCFExtractor",
    "create_extractor",
]

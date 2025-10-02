"""Single-key evaluator implementations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, Sequence

import numpy as np

from .kk_major import KK_MAJOR_PROFILE, cosine_similarity


class SKEBackend(str, Enum):
    KK_MAJOR = "kk_major"


@dataclass(frozen=True)
class SKEConfig:
    mode: SKEBackend = SKEBackend.KK_MAJOR
    reference_bins: int = 12

    def __post_init__(self) -> None:
        if self.reference_bins <= 0:
            raise ValueError("reference_bins must be positive")


class SKEvaluator(Protocol):
    config: SKEConfig

    def evaluate(self, pdf: Sequence[float]) -> float:
        """Return the likelihood that ``pdf`` matches the configured tonal center."""


class _KKMajorEvaluator:
    def __init__(self, config: SKEConfig) -> None:
        self.config = config
        raw_profile = np.asarray(KK_MAJOR_PROFILE, dtype=np.float64)
        if raw_profile.size != self.config.reference_bins:
            raise ValueError("KK_MAJOR profile expects 12 reference bins")
        profile_pdf = raw_profile / raw_profile.sum()
        profile_centered = profile_pdf - np.mean(profile_pdf)
        norm = np.linalg.norm(profile_centered)
        if norm == 0.0:
            raise ValueError("Centered KK profile has zero norm")
        self._profile_centered = profile_centered

    def evaluate(self, pdf: Sequence[float]) -> float:
        reduced = reduce_cycle(pdf, self.config.reference_bins)
        total = float(np.sum(reduced))
        if total <= 0.0:
            return 0.0
        reduced_pdf = reduced / total
        reduced_centered = reduced_pdf - np.mean(reduced_pdf)
        score = cosine_similarity(self._profile_centered, reduced_centered)
        return float(max(0.0, score))


def reduce_cycle(pdf: Sequence[float], reference_bins: int) -> np.ndarray:
    """Reduce an arbitrary-length PDF on [0,1) to ``reference_bins`` bins.

    The reduction respects the cyclic topology: each input sample is treated as a
    box filter centred at ``(i + 0.5) / N`` and distributes its probability onto the
    nearest reference bins using linear interpolation. This ensures octave-equivalent
    energy reinforces corresponding pitch classes even when ``len(pdf)`` is not a
    multiple of ``reference_bins``.
    """

    source = np.asarray(pdf, dtype=np.float64)
    if source.ndim != 1:
        raise ValueError("pdf must be a 1-D sequence")
    if source.size == 0:
        return np.zeros(reference_bins, dtype=np.float64)

    n_source = source.size
    target = np.zeros(reference_bins, dtype=np.float64)

    positions = (np.arange(n_source, dtype=np.float64) + 0.5) / n_source
    scaled = positions * reference_bins - 0.5
    lower = np.floor(scaled).astype(int)
    frac = scaled - np.floor(scaled)
    upper = (lower + 1) % reference_bins
    lower %= reference_bins

    np.add.at(target, lower, source * (1.0 - frac))
    np.add.at(target, upper, source * frac)

    return target


def expand_profile(profile: Sequence[float], target_bins: int) -> np.ndarray:
    """Expand a discrete pitch-class profile onto ``target_bins`` samples over the cycle."""

    if target_bins <= 0:
        raise ValueError("target_bins must be positive")

    reference = np.asarray(profile, dtype=np.float64)
    ref_bins = reference.size
    if ref_bins == 0:
        return np.zeros(target_bins, dtype=np.float64)

    centers = np.arange(ref_bins, dtype=np.float64) / ref_bins
    centers_extended = np.concatenate([centers, [1.0]])
    values_extended = np.concatenate([reference, [reference[0]]])

    positions = (np.arange(target_bins, dtype=np.float64) + 0.5) / target_bins
    positions %= 1.0

    expanded = np.interp(positions, centers_extended, values_extended)
    return expanded


def create_ske(config: SKEConfig) -> SKEvaluator:
    """Factory placeholder that will dispatch to backend-specific evaluators."""

    if config.mode == SKEBackend.KK_MAJOR:
        return _KKMajorEvaluator(config)
    raise ValueError(f"Unsupported SKE backend: {config.mode}")


__all__ = [
    "SKEConfig",
    "SKEBackend",
    "SKEvaluator",
    "create_ske",
    "reduce_cycle",
    "expand_profile",
]

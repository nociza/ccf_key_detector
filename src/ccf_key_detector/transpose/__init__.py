"""Circular transposition operators for the continuous chroma PDF."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol, Sequence

import numpy as np

from ccf_key_detector.ske import SKEvaluator


@dataclass(frozen=True)
class TranspositionConfig:
    """Configuration for the circular transposition operator."""

    n_bins: int

    def __post_init__(self) -> None:
        if self.n_bins <= 0:
            raise ValueError("n_bins must be positive")


class TranspositionOperator(Protocol):
    """Protocol describing circular permutation behaviour."""

    config: TranspositionConfig

    def apply(self, vector: Sequence[float], shift: int) -> list[float]:
        """Return ``vector`` rotated by ``shift`` bins (positive == upward)."""

    def matrix(self) -> list[list[int]]:
        """Return the explicit permutation matrix ``T``."""

    def powers(self, max_power: int) -> list[list[list[int]]]:  # pragma: no cover - rarely used
        """Return matrices for ``T^k`` up to ``max_power`` (inclusive)."""


class _TranspositionOperator:
    def __init__(self, config: TranspositionConfig) -> None:
        self.config = config
        self._matrix: np.ndarray | None = None

    def apply(self, vector: Sequence[float], shift: int) -> list[float]:
        if shift == 0:
            return list(vector)
        array = np.asarray(vector, dtype=np.float64)
        if array.ndim != 1:
            raise ValueError("Transposition operator expects 1-D vectors")
        if array.size != self.config.n_bins:
            raise ValueError(
                f"Vector length {array.size} does not match n_bins={self.config.n_bins}"
            )
        shift_mod = shift % self.config.n_bins
        rotated = np.roll(array, -shift_mod)
        return rotated.tolist()

    def matrix(self) -> list[list[int]]:
        if self._matrix is None:
            eye = np.eye(self.config.n_bins, dtype=int)
            self._matrix = np.roll(eye, -1, axis=1)
        return self._matrix.tolist()

    def powers(self, max_power: int) -> list[list[list[int]]]:  # pragma: no cover - optional
        matrices = []
        base = np.array(self.matrix(), dtype=int)
        power = np.eye(self.config.n_bins, dtype=int)
        for _ in range(max_power + 1):
            matrices.append(power.tolist())
            power = power @ base
        return matrices


def create_operator(config: TranspositionConfig) -> TranspositionOperator:
    """Factory for the circular transposition operator."""

    return _TranspositionOperator(config)


__all__ = [
    "TranspositionConfig",
    "TranspositionOperator",
    "create_operator",
    "scan_distribution",
]


def scan_distribution(
    pdf: Sequence[float],
    ske: SKEvaluator,
    operator: TranspositionOperator,
) -> np.ndarray:
    """Evaluate SKE across all circular shifts of ``pdf`` and normalise."""

    vector = np.asarray(pdf, dtype=np.float64)
    if vector.ndim != 1:
        raise ValueError("pdf must be 1-D for tonal scan")
    if vector.size != operator.config.n_bins:
        raise ValueError(
            f"pdf length {vector.size} does not match operator n_bins={operator.config.n_bins}"
        )

    scores = np.empty(operator.config.n_bins, dtype=np.float64)
    for shift in range(operator.config.n_bins):
        rotated = operator.apply(vector, shift)
        scores[shift] = ske.evaluate(rotated)

    total = scores.sum()
    if not np.isfinite(total) or total <= 0.0:
        return np.full(operator.config.n_bins, 1.0 / operator.config.n_bins)
    return scores / total

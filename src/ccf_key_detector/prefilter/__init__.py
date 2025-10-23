"""Deterministic pre-filter implementations for mono signals."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import numpy as np


class PreFilterMode(str, Enum):
    SPECTRAL_GATE = "spectral_gate"
    HPF_ONLY = "hpf_only"
    NONE = "none"


@dataclass(frozen=True)
class PreFilterConfig:
    enable: bool = True
    mode: PreFilterMode = PreFilterMode.HPF_ONLY
    hpf_cutoff_hz: float = 80.0

    def __post_init__(self) -> None:
        if self.hpf_cutoff_hz <= 0.0:
            raise ValueError("hpf_cutoff_hz must be positive")


class PreFilter(Protocol):
    config: PreFilterConfig

    def process(self, frame: memoryview, sample_rate: int) -> memoryview:
        """Process and return a mono frame according to the chosen mode."""


class _BypassPreFilter:
    def __init__(self, config: PreFilterConfig) -> None:
        self.config = config

    def process(self, frame: memoryview, sample_rate: int) -> memoryview:  # noqa: D401
        """Return the buffer unchanged."""

        return frame


class _HighPassPreFilter:
    """Minimal first-order high-pass filter for plosive suppression."""

    def __init__(self, config: PreFilterConfig) -> None:
        self.config = config
        self._prev_input = 0.0
        self._prev_output = 0.0

    def process(self, frame: memoryview, sample_rate: int) -> memoryview:
        samples = _to_float_array(frame)
        rc = 1.0 / (2 * np.pi * self.config.hpf_cutoff_hz)
        dt = 1.0 / sample_rate
        alpha = rc / (rc + dt)
        output = np.empty_like(samples)
        prev_output = self._prev_output
        prev_input = self._prev_input
        for idx, current in enumerate(samples):
            prev_output = alpha * (prev_output + current - prev_input)
            output[idx] = prev_output
            prev_input = current
        self._prev_output = float(prev_output)
        self._prev_input = float(prev_input)
        return memoryview(output.astype(np.float32, copy=False))


def _to_float_array(frame: memoryview) -> np.ndarray:
    if isinstance(frame, np.ndarray):
        return frame.astype(np.float32, copy=False)
    if isinstance(frame, memoryview):
        if frame.format == "f" and bool(getattr(frame, "c_contiguous", True)):
            return np.frombuffer(frame, dtype=np.float32)
        return np.array(frame.tolist(), dtype=np.float32)
    return np.asarray(frame, dtype=np.float32)


def create_prefilter(config: PreFilterConfig) -> PreFilter:
    """Return a deterministic pre-filter instance."""

    if not config.enable or config.mode == PreFilterMode.NONE:
        return _BypassPreFilter(config)
    if config.mode == PreFilterMode.HPF_ONLY:
        return _HighPassPreFilter(config)
    raise NotImplementedError("Spectral gate mode is not yet implemented")


__all__ = [
    "PreFilterMode",
    "PreFilterConfig",
    "PreFilter",
    "create_prefilter",
]

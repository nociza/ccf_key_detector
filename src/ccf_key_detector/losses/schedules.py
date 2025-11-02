"""KL warm-up and free-bits helpers."""

from __future__ import annotations

import numpy as np


def kl_warmup(step: int, total_steps: int, target_beta: float) -> float:
    """Linearly ramp ``β`` from 0 to ``target_beta`` over ``total_steps``."""

    if total_steps <= 0:
        return float(target_beta)
    ratio = min(max(step, 0), total_steps) / total_steps
    return float(target_beta * ratio)


def apply_free_bits(kl_values: np.ndarray, free_bits: float) -> np.ndarray:
    """Subtract ``free_bits`` per dimension, clamping at zero."""

    kl = np.asarray(kl_values, dtype=np.float64)
    if free_bits <= 0:
        return kl
    adjusted = kl - free_bits
    adjusted[adjusted < 0.0] = 0.0
    return adjusted


__all__ = ["kl_warmup", "apply_free_bits"]

"""Interval-content (z-) loss utilities."""

from __future__ import annotations

import numpy as np

_EPS = 1e-12


def l2_cicv_loss(g_p: np.ndarray, g_q: np.ndarray) -> float:
    """Return the L2 loss between two CICV distributions."""

    g_p = np.asarray(g_p, dtype=np.float64)
    g_q = np.asarray(g_q, dtype=np.float64)
    if g_p.shape != g_q.shape:
        raise ValueError("CICV arrays must share the same shape")
    return float(np.sum((g_p - g_q) ** 2))


def kl_cicv_loss(g_p: np.ndarray, g_q: np.ndarray) -> float:
    """Return KL(g_p || g_q) with small epsilon stabilisation."""

    g_p = np.asarray(g_p, dtype=np.float64)
    g_q = np.asarray(g_q, dtype=np.float64)
    if g_p.shape != g_q.shape:
        raise ValueError("CICV arrays must share the same shape")
    p = g_p + _EPS
    q = g_q + _EPS
    return float(np.sum(p * (np.log(p) - np.log(q))))


def js_cicv_loss(g_p: np.ndarray, g_q: np.ndarray) -> float:
    """Return the Jensen–Shannon divergence between two CICVs."""

    g_p = np.asarray(g_p, dtype=np.float64)
    g_q = np.asarray(g_q, dtype=np.float64)
    if g_p.shape != g_q.shape:
        raise ValueError("CICV arrays must share the same shape")
    m = 0.5 * (g_p + g_q)
    return 0.5 * (kl_cicv_loss(g_p, m) + kl_cicv_loss(g_q, m))


__all__ = [
    "l2_cicv_loss",
    "kl_cicv_loss",
    "js_cicv_loss",
]

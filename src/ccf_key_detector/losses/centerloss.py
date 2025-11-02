"""Center geometry loss."""

from __future__ import annotations

import math


def circular_distance(mu_p: float, mu_q: float) -> float:
    """Return ``1 - cos(2π(μp - μq))`` clipped to [0, 2]."""

    delta = (mu_p - mu_q) % 1.0
    return 1.0 - math.cos(2.0 * math.pi * delta)


def center_loss(mu_p: float, rho_p: float, mu_q: float, rho_q: float) -> float:
    """Compute the center-field loss weighted by concentration."""

    confidence = min(max(rho_p, 0.0), max(rho_q, 0.0))
    return confidence * circular_distance(mu_p, mu_q)


__all__ = ["center_loss", "circular_distance"]

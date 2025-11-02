"""Latent head utilities."""

from __future__ import annotations

import math
from typing import Tuple

import torch


def split_center_texture(latent: torch.Tensor, center_dim: int = 2) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split latent into centre (first ``center_dim``) and texture components."""

    if latent.shape[-1] < center_dim:
        raise ValueError("latent dimensionality is smaller than centre dimension")
    return latent[..., :center_dim], latent[..., center_dim:]


def center_to_polar(z_center: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert 2-D center vector to (radius, angle) with angle ∈ [0,1)."""

    if z_center.shape[-1] != 2:
        raise ValueError("centre latent must have dimension 2")
    x = z_center[..., 0]
    y = z_center[..., 1]
    radius = torch.sqrt(x ** 2 + y ** 2)
    angle = torch.atan2(y, x) / (2 * math.pi)
    angle = torch.remainder(angle, 1.0)
    return radius, angle


__all__ = ["split_center_texture", "center_to_polar"]

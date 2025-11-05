"""Latent head utilities."""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn


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


class AuxiliaryHeads(nn.Module):
    """Predict feature-level targets (CICV, centre-field) from latent codes."""

    def __init__(self, latent_dim: int, n_bins: int, hidden_multiplier: int = 2) -> None:
        super().__init__()
        hidden_dim = max(latent_dim * hidden_multiplier, 32)
        self.cicv_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, n_bins),
        )
        self.center_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, n_bins),
        )

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        cicv_logits = self.cicv_head(z)
        center_logits = self.center_head(z)
        return cicv_logits, center_logits


__all__ = ["split_center_texture", "center_to_polar", "AuxiliaryHeads"]

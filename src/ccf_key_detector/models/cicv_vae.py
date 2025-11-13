"""Lightweight VAE that operates directly on CICV vectors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class CICVVAEConfig:
    input_dim: int = 180
    hidden_dims: Tuple[int, ...] = (256, 128)
    latent_dim: int = 16

    def __post_init__(self) -> None:
        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if self.latent_dim <= 0:
            raise ValueError("latent_dim must be positive")


class CICVVAE(nn.Module):
    """Feed-forward VAE that reconstructs CICV vectors."""

    def __init__(self, config: CICVVAEConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = self._build_mlp(config.input_dim, config.hidden_dims)
        last_dim = config.hidden_dims[-1] if config.hidden_dims else config.input_dim
        self.fc_mu = nn.Linear(last_dim, config.latent_dim)
        self.fc_logvar = nn.Linear(last_dim, config.latent_dim)
        decoder_hidden = tuple(reversed(config.hidden_dims))
        first_decode_dim = decoder_hidden[0] if decoder_hidden else config.input_dim
        self.fc_decode = nn.Linear(config.latent_dim, first_decode_dim)
        decoder_tail: tuple[int, ...]
        if decoder_hidden:
            decoder_tail = decoder_hidden[1:] + (config.input_dim,)
        else:
            decoder_tail = (config.input_dim,)
        self.decoder = self._build_mlp(
            first_decode_dim,
            decoder_tail,
            final_activation=nn.Sigmoid(),
        )

    @staticmethod
    def _build_mlp(input_dim: int, hidden_dims: Sequence[int], final_activation: nn.Module | None = None) -> nn.Sequential:
        layers: list[nn.Module] = []
        in_dim = input_dim
        for idx, dim in enumerate(hidden_dims):
            layers.append(nn.Linear(in_dim, dim))
            if idx < len(hidden_dims) - 1:
                layers.append(nn.ReLU(inplace=True))
            in_dim = dim
        if final_activation is not None:
            layers.append(final_activation)
        return nn.Sequential(*layers)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_decode(z)
        recon = self.decoder(h)
        return recon

    def forward(self, x: torch.Tensor, *, sample: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar) if sample else mu
        recon = self.decode(z)
        return recon, mu, logvar, z


__all__ = ["CICVVAE", "CICVVAEConfig"]

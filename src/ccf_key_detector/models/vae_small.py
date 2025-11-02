"""Small convolutional VAE for feature tensors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class SmallVAEConfig:
    input_channels: int = 1
    input_height: int = 1
    input_width: int = 180
    hidden_channels: Tuple[int, ...] = (32, 64)
    latent_dim: int = 16
    use_batch_norm: bool = False


class SmallVAE(nn.Module):
    """Minimal convolutional VAE for 1-D/2-D feature grids."""

    def __init__(self, config: SmallVAEConfig) -> None:
        super().__init__()
        if config.latent_dim < 2:
            raise ValueError("latent_dim must be >= 2 to support centre/texture split")
        if config.input_channels <= 0:
            raise ValueError("input_channels must be positive")
        if config.input_height <= 0 or config.input_width <= 0:
            raise ValueError("input dimensions must be positive")
        if len(config.hidden_channels) == 0:
            raise ValueError("hidden_channels must contain at least one entry")

        self.config = config
        self.encoder = self._build_encoder()

        with torch.no_grad():
            dummy = torch.zeros(
                1,
                config.input_channels,
                config.input_height,
                config.input_width,
            )
            encoded = self.encoder(dummy)
        self._encoded_shape = encoded.shape[1:]
        self._encoded_dim = int(np.prod(self._encoded_shape))

        self.fc_mu = nn.Linear(self._encoded_dim, config.latent_dim)
        self.fc_logvar = nn.Linear(self._encoded_dim, config.latent_dim)
        self.fc_decode = nn.Linear(config.latent_dim, self._encoded_dim)

        self.decoder = self._build_decoder()

    # ------------------------------------------------------------------ encoder
    def _conv_args(self) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
        if self.config.input_height == 1:
            return (1, 4), (1, 2), (0, 1)
        return (3, 3), (2, 2), (1, 1)

    def _build_encoder(self) -> nn.Sequential:
        kernel, stride, padding = self._conv_args()
        layers: list[nn.Module] = []
        in_c = self.config.input_channels
        for out_c in self.config.hidden_channels:
            layers.append(nn.Conv2d(in_c, out_c, kernel_size=kernel, stride=stride, padding=padding))
            if self.config.use_batch_norm:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.ReLU(inplace=True))
            in_c = out_c
        return nn.Sequential(*layers)

    # ------------------------------------------------------------------ decoder
    def _build_decoder(self) -> nn.Sequential:
        kernel, stride, padding = self._conv_args()
        layers: list[nn.Module] = []
        channels = list(self.config.hidden_channels)[::-1]
        in_c = channels[0]
        for out_c in channels[1:]:
            layers.append(
                nn.ConvTranspose2d(
                    in_c,
                    out_c,
                    kernel_size=kernel,
                    stride=stride,
                    padding=padding,
                )
            )
            if self.config.use_batch_norm:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.ReLU(inplace=True))
            in_c = out_c

        layers.append(
            nn.ConvTranspose2d(
                in_c,
                self.config.input_channels,
                kernel_size=kernel,
                stride=stride,
                padding=padding,
            )
        )
        layers.append(nn.Sigmoid())
        return nn.Sequential(*layers)

    # ------------------------------------------------------------------ api
    @property
    def encoded_shape(self) -> Tuple[int, ...]:
        return self._encoded_shape

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        h = h.view(x.size(0), -1)
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
        h = h.view(z.size(0), *self._encoded_shape)
        recon = self.decoder(h)
        recon = self._match_input_dims(recon)
        return recon

    def forward(self, x: torch.Tensor, *, sample: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar) if sample else mu
        recon = self.decode(z)
        return recon, mu, logvar, z

    # ------------------------------------------------------------------ helpers
    def _match_input_dims(self, recon: torch.Tensor) -> torch.Tensor:
        _, _, H, W = recon.shape
        target_h, target_w = self.config.input_height, self.config.input_width
        if H != target_h or W != target_w:
            recon = recon[..., :target_h, :target_w]
            if recon.shape[-2] != target_h or recon.shape[-1] != target_w:
                recon = F.interpolate(
                    recon,
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=False,
                )
        return recon


__all__ = ["SmallVAE", "SmallVAEConfig"]

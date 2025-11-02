"""Minimal training loop for the feature-level VAE.

This script is intentionally lightweight—run small synthetic or curated
datasets on CPU to validate the plumbing before moving training to the T4
server. The curriculum from the build plan (PT1–PT3) can be layered on top of
``train_small`` by adding auxiliary heads/losses inside the loop.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ccf_key_detector.data import FeatureDatasetConfig, VocalFeatureDataset
from ccf_key_detector.features import CCFConfig
from ccf_key_detector.losses import kl_warmup
from ccf_key_detector.models import SmallVAE, SmallVAEConfig


@dataclass
class TrainingConfig:
    dataset_root: Path
    feature: FeatureDatasetConfig = field(default_factory=FeatureDatasetConfig)
    model: SmallVAEConfig = field(default_factory=SmallVAEConfig)
    batch_size: int = 8
    num_epochs: int = 1
    lr: float = 1e-3
    beta_target: float = 1.0
    kl_warmup_steps: int = 1000
    device: Optional[str] = None
    max_steps_per_epoch: Optional[int] = None


def train_small(config: TrainingConfig) -> None:
    device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dataset = VocalFeatureDataset(config.dataset_root, config.feature)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    model = SmallVAE(config.model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    global_step = 0
    model.train()
    for epoch in range(config.num_epochs):
        for step, batch in enumerate(dataloader):
            x = batch["ccf"].to(device)
            x = x.unsqueeze(1).unsqueeze(1)

            recon, mu, logvar, _ = model(x)
            recon_loss = F.mse_loss(recon, x)
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            beta = kl_warmup(global_step, config.kl_warmup_steps, config.beta_target)
            loss = recon_loss + beta * kl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1
            if config.max_steps_per_epoch is not None and step + 1 >= config.max_steps_per_epoch:
                break

        print(
            f"Epoch {epoch+1}/{config.num_epochs} | loss={loss.item():.4f} "
            f"recon={recon_loss.item():.4f} kl={kl.item():.4f} beta={beta:.3f}"
        )


def _parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path, help="Root directory containing audio files")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--frame-length", type=float, default=10.0, help="Frame length (sec)")
    parser.add_argument("--hop-length", type=float, default=1.0, help="Hop length (sec)")
    parser.add_argument("--n-bins", type=int, default=180, help="Chroma resolution")
    parser.add_argument("--device", type=str, default=None, help="Device override (cpu/cuda)")
    parser.add_argument("--max-steps", type=int, default=None, help="Max steps per epoch for debugging")

    args = parser.parse_args()
    feature_cfg = FeatureDatasetConfig(
        sample_rate=48_000,
        frame_length_sec=args.frame_length,
        hop_length_sec=args.hop_length,
        ccf=CCFConfig(n_bins=args.n_bins),
    )
    model_cfg = SmallVAEConfig(
        input_channels=1,
        input_height=1,
        input_width=args.n_bins,
    )
    return TrainingConfig(
        dataset_root=args.dataset_root,
        feature=feature_cfg,
        model=model_cfg,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        max_steps_per_epoch=args.max_steps,
    )


def main() -> None:
    config = _parse_args()
    train_small(config)


if __name__ == "__main__":
    main()

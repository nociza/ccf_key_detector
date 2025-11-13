"""Training loop for the feature-level VAE with checkpointing and AMP support."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from ccf_key_detector.data import (
    FeatureDatasetConfig,
    PrecomputedFeatureDataset,
    VocalFeatureDataset,
)
from ccf_key_detector.features import CCFConfig, build_torus_torch
from ccf_key_detector.losses import kl_warmup, phase_loss_torch
from ccf_key_detector.models import AuxiliaryHeads, SmallVAE, SmallVAEConfig


@dataclass
class LossWeights:
    lambda_recon: float = 1.0
    lambda_z: float = 1.0
    lambda_center: float = 0.5
    lambda_phase: float = 0.1
    beta_target: float = 1.0
    kl_warmup_steps: int = 1000
    free_bits: float = 0.0


@dataclass
class TrainingConfig:
    dataset_root: Path
    feature: FeatureDatasetConfig = field(default_factory=FeatureDatasetConfig)
    model: SmallVAEConfig = field(default_factory=SmallVAEConfig)
    batch_size: int = 8
    num_epochs: int = 1
    lr: float = 1e-3
    loss: LossWeights = field(default_factory=LossWeights)
    device: Optional[str] = None
    max_steps_per_epoch: Optional[int] = None
    precomputed_manifest: Optional[Path] = None
    checkpoint_dir: Optional[Path] = None
    save_every: int = 1
    resume_from: Optional[Path] = None
    amp: bool = False
    grad_clip: Optional[float] = None
    log_file: Optional[Path] = None
    num_workers: int = 0
    pin_memory: bool = False


@dataclass
class TrainState:
    model: SmallVAE
    aux_heads: AuxiliaryHeads


@dataclass
class TrainArtifacts:
    state: TrainState
    history: Optional[List[Dict[str, float]]] = None


def train_small(
    config: TrainingConfig,
    *,
    state: Optional[TrainState] = None,
    return_state: bool = False,
    record_history: bool = False,
) -> Optional[TrainArtifacts]:
    device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if config.precomputed_manifest is not None:
        dataset = PrecomputedFeatureDataset(config.precomputed_manifest)
    else:
        feature_cfg = replace(config.feature, include_torus=True)
        dataset = VocalFeatureDataset(config.dataset_root, feature_cfg)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory and device.startswith("cuda"),
    )

    if state is not None:
        model = state.model.to(device)
        aux_heads = state.aux_heads.to(device)
    else:
        model = SmallVAE(config.model).to(device)
        aux_heads = AuxiliaryHeads(config.model.latent_dim, config.model.input_width).to(device)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(aux_heads.parameters()), lr=config.lr
    )
    scaler = GradScaler(enabled=config.amp and device.startswith("cuda"))

    start_epoch = 0
    global_step = 0
    if config.resume_from is not None:
        start_epoch, global_step = _load_checkpoint(
            config.resume_from, model, aux_heads, optimizer, scaler, device
        )

    logger = _TrainLogger(config.log_file)
    model.train()
    history: Optional[List[Dict[str, float]]] = [] if record_history else None
    for epoch in range(start_epoch, config.num_epochs):
        for step, batch in enumerate(dataloader):
            x = batch["ccf"].to(device).unsqueeze(1).unsqueeze(1)

            with autocast(enabled=scaler.is_enabled()):
                recon, mu, logvar, z = model(x)
                recon_loss = F.mse_loss(recon, x)
                kl = _kl_with_free_bits(mu, logvar, config.loss.free_bits)
                beta = kl_warmup(global_step, config.loss.kl_warmup_steps, config.loss.beta_target)

                cicv_logits, center_logits = aux_heads(z)
                cicv_pred = torch.softmax(cicv_logits, dim=1)
                cicv_target = batch["cicv"].to(device)
                z_loss = F.mse_loss(cicv_pred, cicv_target)

                center_pred = torch.relu(center_logits)
                center_target = batch["center_field"].to(device)
                center_recon = F.mse_loss(center_pred, center_target)
                mu_pred, rho_pred = _center_stats(center_pred)
                mu_target = batch["mu"].to(device)
                rho_target = batch["rho"].to(device)
                center_geom = _center_geometry(mu_pred, rho_pred, mu_target, rho_target)
                center_loss = center_recon + center_geom

                phase_component = torch.tensor(0.0, device=device)
                if config.loss.lambda_phase > 0.0 and "torus" in batch:
                    ccf_pred = recon.view(recon.size(0), -1)
                    ccf_pred = torch.relu(ccf_pred)
                    ccf_pred = _normalize_pdf(ccf_pred)
                    torus_pred = build_torus_torch(ccf_pred)
                    torus_target = batch["torus"].to(device)
                    if torus_target.dim() == 2:
                        torus_target = torus_target.unsqueeze(1)
                    phase_component = phase_loss_torch(torus_target, torus_pred, allow_inversion=True)

                loss = (
                    config.loss.lambda_recon * recon_loss
                    + beta * kl
                    + config.loss.lambda_z * z_loss
                    + config.loss.lambda_center * center_loss
                    + config.loss.lambda_phase * phase_component
                )

            optimizer.zero_grad()
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                if config.grad_clip is not None:
                    scaler.unscale_(optimizer)
                    clip_grad_norm_(list(model.parameters()) + list(aux_heads.parameters()), config.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if config.grad_clip is not None:
                    clip_grad_norm_(list(model.parameters()) + list(aux_heads.parameters()), config.grad_clip)
                optimizer.step()

            global_step += 1
            if config.max_steps_per_epoch is not None and step + 1 >= config.max_steps_per_epoch:
                break

        epoch_idx = epoch + 1
        metrics = {
            "epoch": float(epoch_idx),
            "loss": float(loss.item()),
            "recon": float(recon_loss.item()),
            "kl": float(kl.item()),
            "beta": float(beta),
            "z": float(z_loss.item()),
            "center": float(center_loss.item()),
            "phase": float(phase_component.item()),
        }
        if history is not None:
            history.append(metrics)
        logger.log(metrics)

        print(
            f"Epoch {epoch_idx}/{config.num_epochs} | loss={loss.item():.4f} "
            f"recon={recon_loss.item():.4f} kl={kl.item():.4f} beta={beta:.3f} "
            f"z={z_loss.item():.4f} center={center_loss.item():.4f} phase={phase_component.item():.4f}"
        )

        if config.checkpoint_dir is not None and (
            epoch_idx % config.save_every == 0 or epoch_idx == config.num_epochs
        ):
            _save_checkpoint(
                config.checkpoint_dir,
                epoch_idx,
                global_step,
                model,
                aux_heads,
                optimizer,
                scaler,
                config,
            )

    if return_state:
        model_cpu = model.to("cpu")
        aux_cpu = aux_heads.to("cpu")
        logger.close()
        return TrainArtifacts(state=TrainState(model_cpu, aux_cpu), history=history)

    logger.close()
    return None


def _parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset_root",
        type=Path,
        help="Root directory containing audio files (ignored if --precomputed-manifest is set)",
    )
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--frame-length", type=float, default=10.0, help="Frame length (sec)")
    parser.add_argument("--hop-length", type=float, default=1.0, help="Hop length (sec)")
    parser.add_argument("--n-bins", type=int, default=180, help="Chroma resolution")
    parser.add_argument("--device", type=str, default=None, help="Device override (cpu/cuda)")
    parser.add_argument("--max-steps", type=int, default=None, help="Max steps per epoch for debugging")
    parser.add_argument("--lambda-z", type=float, default=1.0, help="Weight for CICV loss")
    parser.add_argument("--lambda-center", type=float, default=0.5, help="Weight for center-field loss")
    parser.add_argument("--lambda-phase", type=float, default=0.1, help="Weight for phase loss")
    parser.add_argument("--lambda-recon", type=float, default=1.0, help="Weight for reconstruction loss")
    parser.add_argument("--beta-target", type=float, default=1.0, help="Target KL weight")
    parser.add_argument("--kl-warmup-steps", type=int, default=1000, help="Number of warm-up steps for KL weight")
    parser.add_argument("--free-bits", type=float, default=0.0, help="Free bits per latent dimension")
    parser.add_argument(
        "--precomputed-manifest",
        type=Path,
        default=None,
        help="Optional path to a manifest.json produced by scripts/precompute_features.py",
    )
    parser.add_argument("--latent-dim", type=int, default=16, help="Latent dimensionality")
    parser.add_argument(
        "--hidden-channels",
        type=int,
        nargs="+",
        default=(32, 64),
        help="Encoder channel widths (space separated)",
    )
    parser.add_argument("--use-batch-norm", action="store_true", help="Enable BatchNorm in encoder/decoder")
    parser.add_argument("--checkpoint-dir", type=Path, default=None, help="Directory to store checkpoints")
    parser.add_argument("--save-every", type=int, default=1, help="Checkpoint frequency (epochs)")
    parser.add_argument("--resume-from", type=Path, default=None, help="Path to a checkpoint to resume from")
    parser.add_argument("--amp", action="store_true", help="Enable automatic mixed precision (CUDA only)")
    parser.add_argument("--grad-clip", type=float, default=None, help="Gradient clipping value (L2 norm)")
    parser.add_argument("--log-file", type=Path, default=None, help="Optional CSV log file for metrics")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count")
    parser.add_argument("--pin-memory", action="store_true", help="Enable pinned memory for DataLoader")

    args = parser.parse_args()
    feature_cfg = FeatureDatasetConfig(
        sample_rate=48_000,
        frame_length_sec=args.frame_length,
        hop_length_sec=args.hop_length,
        ccf=CCFConfig(n_bins=args.n_bins),
        include_torus=True,
    )
    model_cfg = SmallVAEConfig(
        input_channels=1,
        input_height=1,
        input_width=args.n_bins,
        hidden_channels=tuple(args.hidden_channels),
        latent_dim=args.latent_dim,
        use_batch_norm=args.use_batch_norm,
    )
    loss_cfg = LossWeights(
        lambda_recon=args.lambda_recon,
        lambda_z=args.lambda_z,
        lambda_center=args.lambda_center,
        lambda_phase=args.lambda_phase,
        beta_target=args.beta_target,
        kl_warmup_steps=args.kl_warmup_steps,
        free_bits=args.free_bits,
    )
    return TrainingConfig(
        dataset_root=args.dataset_root,
        feature=feature_cfg,
        model=model_cfg,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        lr=args.lr,
        loss=loss_cfg,
        device=args.device,
        max_steps_per_epoch=args.max_steps,
        precomputed_manifest=args.precomputed_manifest,
        checkpoint_dir=args.checkpoint_dir,
        save_every=args.save_every,
        resume_from=args.resume_from,
        amp=args.amp,
        grad_clip=args.grad_clip,
        log_file=args.log_file,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )


def main() -> None:
    config = _parse_args()
    train_small(config)


def _normalize_pdf(pdf: torch.Tensor) -> torch.Tensor:
    denom = pdf.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return pdf / denom


def _kl_with_free_bits(mu: torch.Tensor, logvar: torch.Tensor, free_bits: float) -> torch.Tensor:
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    if free_bits > 0.0:
        kl_per_dim = torch.clamp(kl_per_dim - free_bits, min=0.0)
    return kl_per_dim.mean()


def _center_stats(center_map: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    n_bins = center_map.shape[1]
    normed = _normalize_pdf(center_map)
    device = center_map.device
    theta = torch.linspace(0.0, 1.0, steps=n_bins, device=device)
    two_pi = 2.0 * math.pi
    cos_t = torch.cos(two_pi * theta)
    sin_t = torch.sin(two_pi * theta)
    mean_cos = torch.sum(normed * cos_t, dim=1)
    mean_sin = torch.sum(normed * sin_t, dim=1)
    mu = torch.atan2(mean_sin, mean_cos) / two_pi
    mu = torch.remainder(mu, 1.0)
    rho = torch.sqrt(mean_cos**2 + mean_sin**2)
    return mu, rho


def _center_geometry(
    mu_pred: torch.Tensor,
    rho_pred: torch.Tensor,
    mu_target: torch.Tensor,
    rho_target: torch.Tensor,
) -> torch.Tensor:
    distance = 1.0 - torch.cos(2.0 * math.pi * (mu_pred - mu_target))
    weight = torch.minimum(rho_pred, rho_target)
    return torch.mean(weight * distance)


def _save_checkpoint(
    checkpoint_dir: Path,
    epoch: int,
    global_step: int,
    model: SmallVAE,
    aux_heads: AuxiliaryHeads,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    config: TrainingConfig,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "global_step": global_step,
        "model": model.state_dict(),
        "aux_heads": aux_heads.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler.is_enabled() else None,
        "model_config": asdict(config.model),
        "feature_config": asdict(config.feature),
        "loss_config": asdict(config.loss),
        "precomputed_manifest": str(config.precomputed_manifest) if config.precomputed_manifest else None,
    }
    path = checkpoint_dir / f"epoch_{epoch:04d}.pt"
    torch.save(payload, path)


def _load_checkpoint(
    path: Path,
    model: SmallVAE,
    aux_heads: AuxiliaryHeads,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: str,
) -> Tuple[int, int]:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    aux_heads.load_state_dict(ckpt["aux_heads"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if scaler.is_enabled() and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    epoch = int(ckpt.get("epoch", 0))
    global_step = int(ckpt.get("global_step", 0))
    return epoch, global_step


class _TrainLogger:
    def __init__(self, path: Optional[Path]) -> None:
        self.path = path
        self._file = None
        self._writer: Optional[csv.DictWriter] = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._file = path.open("w", newline="")
            fieldnames = ["epoch", "loss", "recon", "kl", "beta", "z", "center", "phase"]
            self._writer = csv.DictWriter(self._file, fieldnames=fieldnames)
            self._writer.writeheader()

    def log(self, metrics: Dict[str, float]) -> None:
        if self._writer is not None:
            self._writer.writerow(metrics)
            self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


if __name__ == "__main__":
    main()

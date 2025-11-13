"""Evaluate a trained VAE checkpoint on a precomputed manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import torch

from ccf_key_detector.data import PrecomputedFeatureDataset
from ccf_key_detector.models import AuxiliaryHeads, SmallVAE, SmallVAEConfig


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Path to manifest.json")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default=None, help="cpu or cuda")
    parser.add_argument("--max-batches", type=int, default=None, help="Optional cap for quick tests")
    parser.add_argument("--json-out", type=Path, default=None, help="Optional JSON report path")
    return parser.parse_args()


def _load_model(checkpoint_path: Path, device: torch.device) -> tuple[SmallVAE, AuxiliaryHeads]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_cfg = SmallVAEConfig(**checkpoint["model_config"])
    model = SmallVAE(model_cfg).to(device)
    model.load_state_dict(checkpoint["model"])
    aux = AuxiliaryHeads(model_cfg.latent_dim, model_cfg.input_width).to(device)
    aux.load_state_dict(checkpoint["aux_heads"])
    model.eval()
    aux.eval()
    return model, aux


def _center_stats(center_map: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    center_map = torch.relu(center_map)
    pdf = center_map / center_map.sum(dim=1, keepdim=True).clamp_min(1e-8)
    n_bins = center_map.shape[1]
    theta = torch.linspace(0.0, 1.0, steps=n_bins, device=center_map.device)
    cos_t = torch.cos(2.0 * torch.pi * theta)
    sin_t = torch.sin(2.0 * torch.pi * theta)
    mean_cos = torch.sum(pdf * cos_t, dim=1)
    mean_sin = torch.sum(pdf * sin_t, dim=1)
    mu = torch.atan2(mean_sin, mean_cos) / (2.0 * torch.pi)
    mu = torch.remainder(mu, 1.0)
    rho = torch.sqrt(mean_cos ** 2 + mean_sin ** 2)
    return mu, rho


def evaluate(manifest: Path, checkpoint: Path, batch_size: int, device: str, max_batches: int | None) -> Dict[str, float]:
    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dataset = PrecomputedFeatureDataset(manifest)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model, aux = _load_model(checkpoint, device_t)

    angle_errors = []
    rho_errors = []
    center_geom = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            x = batch["ccf"].to(device_t).unsqueeze(1).unsqueeze(1)
            _, _, _, z = model(x)
            _, center_logits = aux(z)
            mu_pred, rho_pred = _center_stats(center_logits)
            mu_target = batch["mu"].to(device_t)
            rho_target = batch["rho"].to(device_t)

            angle_delta = torch.remainder(mu_pred - mu_target + 0.5, 1.0) - 0.5
            angle_errors.extend((angle_delta.abs() * 1200.0).cpu().tolist())
            rho_errors.extend(torch.abs(rho_pred - rho_target).cpu().tolist())

            geom = torch.minimum(rho_pred, rho_target) * (
                1.0 - torch.cos(2.0 * torch.pi * (mu_pred - mu_target))
            )
            center_geom.extend(geom.cpu().tolist())

            if max_batches is not None and batch_idx + 1 >= max_batches:
                break

    results = {
        "mean_circular_error_cents": float(torch.tensor(angle_errors).mean()),
        "median_circular_error_cents": float(torch.tensor(angle_errors).median()),
        "rho_mae": float(torch.tensor(rho_errors).mean()),
        "center_geom_mean": float(torch.tensor(center_geom).mean()),
        "num_samples": len(angle_errors),
    }
    return results


def main() -> None:
    args = _parse_args()
    metrics = evaluate(args.manifest, args.checkpoint, args.batch_size, args.device, args.max_batches)

    print("== Evaluation Metrics ==")
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with args.json_out.open("w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2)


if __name__ == "__main__":
    main()

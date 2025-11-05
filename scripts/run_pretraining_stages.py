"""Run Stage S1→S2 pretraining and report centre metrics."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional

import torch

from ccf_key_detector.data import FeatureDatasetConfig, VocalFeatureDataset
from ccf_key_detector.features import CCFConfig
from ccf_key_detector.models import SmallVAEConfig
from ccf_key_detector.train.train_small import LossWeights, TrainingConfig, TrainState, train_small


@dataclass
class StageResult:
    name: str
    history: Optional[List[Dict[str, float]]]


def _build_training_config(
    dataset_root: Path,
    n_bins: int,
    frame_length: float,
    hop_length: float,
    epochs: int,
    batch_size: int,
    lr: float,
    loss: LossWeights,
    device: Optional[str],
) -> TrainingConfig:
    feature_cfg = FeatureDatasetConfig(
        sample_rate=48_000,
        frame_length_sec=frame_length,
        hop_length_sec=hop_length,
        ccf=CCFConfig(n_bins=n_bins),
    )
    model_cfg = SmallVAEConfig(
        input_channels=1,
        input_height=1,
        input_width=n_bins,
    )
    return TrainingConfig(
        dataset_root=dataset_root,
        feature=feature_cfg,
        model=model_cfg,
        batch_size=batch_size,
        num_epochs=epochs,
        lr=lr,
        loss=loss,
        device=device,
    )


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
    rho = torch.sqrt(mean_cos**2 + mean_sin**2)
    return mu, rho


def _evaluate_center_metrics(state: TrainState, config: TrainingConfig, batch_size: int) -> Dict[str, float]:
    dataset = VocalFeatureDataset(config.dataset_root, replace(config.feature, include_torus=True))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    device = torch.device("cpu")
    model = state.model.to(device)
    aux = state.aux_heads.to(device)
    model.eval()
    aux.eval()

    angle_errors: List[float] = []
    rho_errors: List[float] = []
    center_geom: List[float] = []

    with torch.no_grad():
        for batch in dataloader:
            x = batch["ccf"].to(device).unsqueeze(1).unsqueeze(1)
            _, _, _, z = model(x)
            cicv_logits, center_logits = aux(z)
            mu_pred, rho_pred = _center_stats(center_logits)
            mu_target = batch["mu"].to(device)
            rho_target = batch["rho"].to(device)

            angle_delta = torch.remainder(mu_pred - mu_target + 0.5, 1.0) - 0.5
            angle_errors.extend((angle_delta.abs() * 1200.0).tolist())  # cents
            rho_errors.extend(torch.abs(rho_pred - rho_target).tolist())

            geom = torch.minimum(rho_pred, rho_target) * (
                1.0 - torch.cos(2.0 * torch.pi * (mu_pred - mu_target))
            )
            center_geom.extend(geom.tolist())

    return {
        "mean_circular_error_cents": float(torch.tensor(angle_errors).mean()),
        "median_circular_error_cents": float(torch.tensor(angle_errors).median()),
        "rho_mae": float(torch.tensor(rho_errors).mean()),
        "center_geom_mean": float(torch.tensor(center_geom).mean()),
    }


def run_stages(args: argparse.Namespace) -> Dict[str, object]:
    dataset_root = args.dataset_root.resolve()
    stage1_loss = LossWeights(
        lambda_recon=args.lambda_recon,
        lambda_z=args.lambda_z,
        lambda_center=args.lambda_center,
        lambda_phase=0.0,
        beta_target=args.beta_target,
        kl_warmup_steps=args.kl_warmup_steps,
        free_bits=args.free_bits,
    )
    stage1_cfg = _build_training_config(
        dataset_root,
        args.n_bins,
        args.frame_length,
        args.hop_length,
        args.epochs_stage1,
        args.batch_size,
        args.lr,
        stage1_loss,
        args.device,
    )
    stage1_artifacts = train_small(
        stage1_cfg,
        return_state=True,
        record_history=True,
    )
    assert stage1_artifacts is not None

    stage2_loss = LossWeights(
        lambda_recon=args.lambda_recon,
        lambda_z=args.lambda_z,
        lambda_center=args.lambda_center,
        lambda_phase=args.lambda_phase,
        beta_target=args.beta_target,
        kl_warmup_steps=args.kl_warmup_steps,
        free_bits=args.free_bits,
    )
    stage2_cfg = replace(stage1_cfg, num_epochs=args.epochs_stage2, loss=stage2_loss)
    stage2_artifacts = train_small(
        stage2_cfg,
        state=stage1_artifacts.state,
        return_state=True,
        record_history=True,
    )
    assert stage2_artifacts is not None

    metrics = _evaluate_center_metrics(stage2_artifacts.state, stage2_cfg, args.eval_batch_size)

    return StageResult("stage1", stage1_artifacts.history), StageResult("stage2", stage2_artifacts.history), metrics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path, help="Synthetic dataset directory")
    parser.add_argument("--epochs-stage1", type=int, default=5)
    parser.add_argument("--epochs-stage2", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--frame-length", type=float, default=1.5)
    parser.add_argument("--hop-length", type=float, default=1.5)
    parser.add_argument("--n-bins", type=int, default=180)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--lambda-recon", type=float, default=1.0)
    parser.add_argument("--lambda-z", type=float, default=1.0)
    parser.add_argument("--lambda-center", type=float, default=0.7)
    parser.add_argument("--lambda-phase", type=float, default=0.05)
    parser.add_argument("--beta-target", type=float, default=0.5)
    parser.add_argument("--kl-warmup-steps", type=int, default=200)
    parser.add_argument("--free-bits", type=float, default=0.0)
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    stage1_res, stage2_res, metrics = run_stages(args)

    for stage_res in (stage1_res, stage2_res):
        print(f"== {stage_res.name.upper()} ==")
        if stage_res.history:
            for epoch_metrics in stage_res.history:
                print(
                    f"  epoch={epoch_metrics['epoch']:.0f} "
                    f"loss={epoch_metrics['loss']:.4f} "
                    f"recon={epoch_metrics['recon']:.4f} "
                    f"center={epoch_metrics['center']:.4f} "
                    f"phase={epoch_metrics['phase']:.4f}"
                )

    print("== EVAL ==")
    for key, value in metrics.items():
        print(f"  {key}: {value:.6f}")

    if args.json_out is not None:
        payload = {
            "stage1": stage1_res.history,
            "stage2": stage2_res.history,
            "metrics": metrics,
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with args.json_out.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)


if __name__ == "__main__":
    main()

"""Train a CICV-only VAE on precomputed feature manifests."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from ccf_key_detector.data import PrecomputedFeatureDataset
from ccf_key_detector.models.cicv_vae import CICVVAE, CICVVAEConfig


class CICVDataset(Dataset):
    def __init__(self, manifest: Path) -> None:
        self.dataset = PrecomputedFeatureDataset(manifest)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> torch.Tensor:
        sample = self.dataset[idx]
        cicv = sample["cicv"].float()
        return cicv


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Path to precomputed manifest.json")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=(256, 128))
    parser.add_argument("--beta", type=float, default=0.1, help="KL weight")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Where to save the trained model (.pt)")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional cap on steps per epoch")
    return parser.parse_args()


def train(args: argparse.Namespace) -> Dict[str, float]:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dataset = CICVDataset(args.manifest)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model_cfg = CICVVAEConfig(
        input_dim=dataset[0].numel(),
        hidden_dims=tuple(args.hidden_dims),
        latent_dim=args.latent_dim,
    )
    model = CICVVAE(model_cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = []
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        steps = 0
        for batch_idx, cicv in enumerate(dataloader):
            cicv = cicv.to(device)
            recon, mu, logvar, _ = model(cicv)
            recon_loss = F.mse_loss(recon, cicv)
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + args.beta * kl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            steps += 1
            if args.max_steps is not None and steps >= args.max_steps:
                break

        avg_loss = running_loss / max(1, steps)
        history.append({"epoch": epoch + 1, "loss": avg_loss})
        print(f"Epoch {epoch+1}/{args.epochs} | loss={avg_loss:.6f}")

    checkpoint = {
        "model": model.state_dict(),
        "config": model_cfg.__dict__,
        "train_history": history,
        "beta": args.beta,
        "manifest": str(args.manifest),
    }
    checkpoint_path = args.checkpoint
    if checkpoint_path.suffix == "":
        checkpoint_path = checkpoint_path.with_suffix(".pt")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")
    return history[-1]


def main() -> None:
    args = _parse_args()
    train(args)


if __name__ == "__main__":
    main()

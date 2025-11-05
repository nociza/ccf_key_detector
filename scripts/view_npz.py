"""Quick inspector for precomputed .npz feature bundles."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("npz_path", type=Path, help="Path to the .npz file to inspect")
    parser.add_argument("--frame", type=int, default=0, help="Frame index to visualise")
    parser.add_argument(
        "--show-torus",
        action="store_true",
        help="Render the torus as an image if present",
    )
    parser.add_argument("--headless", action="store_true", help="Print stats only; no plots")
    return parser.parse_args()


def _describe(data: np.lib.npyio.NpzFile) -> None:
    print("== arrays ==")
    for key in sorted(data.files):
        arr = data[key]
        print(f"{key:16s} shape={arr.shape} dtype={arr.dtype}")


def _plot_cycle(ax: plt.Axes, values: np.ndarray, title: str) -> None:
    x = np.linspace(0.0, 1.0, num=values.size, endpoint=False)
    ax.plot(x, values, lw=1.5)
    ax.set_xlim(0.0, 1.0)
    ax.set_title(title)
    ax.set_xlabel("cycle")


def main() -> None:
    args = _parse_args()
    npz_path = args.npz_path.resolve()
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)

    with np.load(npz_path, allow_pickle=False) as data:
        _describe(data)
        frame_idx = args.frame

        if args.headless:
            return

        ccf = data["ccf"][frame_idx]
        cicv = data["cicv"][frame_idx]
        center = data["center"][frame_idx]
        mu = float(data["mu"][frame_idx])
        rho = float(data["rho"][frame_idx])

        fig, axes = plt.subplots(3, 1, figsize=(8, 8), constrained_layout=True)
        _plot_cycle(axes[0], ccf, "Continuous Chroma (PDF)")
        _plot_cycle(axes[1], cicv, "CICV")
        _plot_cycle(axes[2], center, f"Center Field (μ={mu:.3f}, ρ={rho:.3f})")

        if args.show_torus and "torus" in data.files:
            torus = data["torus"][frame_idx]
            fig_torus, ax_torus = plt.subplots(figsize=(6, 5))
            im = ax_torus.imshow(torus, origin="lower", aspect="auto", cmap="viridis")
            ax_torus.set_title("Interval Torus")
            ax_torus.set_xlabel("δ bins")
            ax_torus.set_ylabel("θ bins")
            fig_torus.colorbar(im, ax=ax_torus, shrink=0.8)

        plt.show()


if __name__ == "__main__":
    main()

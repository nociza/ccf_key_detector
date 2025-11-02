#!/usr/bin/env python3
"""Visualise feature responses for a half-diminished seventh chord."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ccf_key_detector.features import (
    CCFConfig,
    build_torus,
    center_field,
    compute_cicv,
    create_extractor,
)

SAMPLE_RATE = 48_000
DURATION_SEC = 0.5
F_REF = 220.0

HALF_DIM7_INTERVALS = (0, 3, 6, 10)  # semitones above the root


def synth_half_dim7(root_hz: float = F_REF, amplitude: float = 0.6) -> np.ndarray:
    t = np.arange(int(SAMPLE_RATE * DURATION_SEC), dtype=np.float64) / SAMPLE_RATE
    waveform = np.zeros_like(t)
    for semitone in HALF_DIM7_INTERVALS:
        freq = root_hz * (2.0 ** (semitone / 12.0))
        waveform += amplitude * np.sin(2.0 * np.pi * freq * t)
    waveform = np.clip(waveform, -1.0, 1.0)
    return waveform.astype(np.float32)


def main() -> None:
    config = CCFConfig(n_bins=180, smoothing_sigma_bins=1.0)
    extractor = create_extractor(config)
    waveform = synth_half_dim7()
    pdf, diagnostics = extractor.compute(memoryview(waveform), SAMPLE_RATE)
    pdf_np = np.array(pdf)

    cicv = compute_cicv(pdf_np)
    torus = build_torus(pdf_np)
    center, mu, rho = center_field(pdf_np)

    theta = np.linspace(0.0, 1.0, num=pdf_np.size, endpoint=False)
    delta = np.linspace(0.0, 1.0, num=cicv.size, endpoint=False)

    print("Diagnostics:")
    for key, value in diagnostics.items():
        print(f"  {key}: {value:.6f}")
    print(f"Center-field mean μ: {mu:.4f} (fraction of octave)")
    print(f"Center-field concentration ρ: {rho:.4f}")

    fig = plt.figure(figsize=(12, 8))

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(theta, pdf_np, color="#1f77b4")
    ax1.set_title("log-CCF p(θ)")
    ax1.set_xlabel("θ (fraction of octave)")
    ax1.set_ylabel("Density")

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(delta, cicv, color="#ff7f0e")
    ax2.set_title("CICV g(δ)")
    ax2.set_xlabel("δ (interval fraction)")
    ax2.set_ylabel("Probability")

    ax3 = fig.add_subplot(2, 2, 3)
    im = ax3.pcolormesh(delta, theta, torus, shading="auto", cmap="magma")
    ax3.set_title("Interval Torus M(θ, δ)")
    ax3.set_xlabel("δ")
    ax3.set_ylabel("θ")
    fig.colorbar(im, ax=ax3, label="Intensity")

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(theta, center, color="#2ca02c")
    ax4.axvline(mu % 1.0, color="red", linestyle="--", label=f"μ={mu:.2f}")
    ax4.set_title("Center-field C(θ)")
    ax4.set_xlabel("θ")
    ax4.set_ylabel("Saliency")
    ax4.legend()

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

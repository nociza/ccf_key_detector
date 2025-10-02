#!/usr/bin/env python3
"""Generate a three-tone test clip and visualise its continuous chroma PDF.

The script produces a mono WAV file containing the simultaneous sine tones at
250 Hz, 300 Hz, and 440 Hz. It then feeds the audio through the continuous
chroma extractor and writes a diagnostic plot showing how the spectral energy
wraps around the unit log-frequency cycle.

Usage (from repository root):

    PYTHONPATH=src python scripts/generate_mixed_tones.py \
        --output-wav build/mixed_tones.wav \
        --output-plot build/mixed_tones_ccf.png
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ccf_key_detector.features_ccf import CCFConfig, create_extractor


DEFAULT_TONES = (250.0, 300.0, 440.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-wav",
        type=Path,
        default=Path("build/mixed_tones.wav"),
        help="Destination WAV file (16-bit PCM)",
    )
    parser.add_argument(
        "--output-plot",
        type=Path,
        default=Path("build/mixed_tones_ccf.png"),
        help="Destination path for the CCF plot",
    )
    parser.add_argument("--duration", type=float, default=5.0, help="Clip length in seconds")
    parser.add_argument("--sample-rate", type=int, default=48_000, help="Target sampling rate")
    parser.add_argument("--amplitude", type=float, default=0.4, help="Peak amplitude per tone")
    parser.add_argument("--n-bins", type=int, default=180, help="Number of CCF bins")
    parser.add_argument(
        "--smoothing",
        type=float,
        default=1.0,
        help="Circular Gaussian smoothing sigma (bins)",
    )
    parser.add_argument(
        "--frame-length",
        type=float,
        default=10.0,
        help="Frame length in seconds for the CCF (will zero-pad if needed)",
    )
    return parser.parse_args()


def synthesise_tones(
    *,
    tones: tuple[float, ...],
    sample_rate: int,
    duration: float,
    amplitude: float,
) -> np.ndarray:
    t = np.arange(int(sample_rate * duration), dtype=np.float64) / sample_rate
    waveform = np.zeros_like(t)
    for freq in tones:
        waveform += amplitude * np.sin(2.0 * math.pi * freq * t)
    waveform = np.clip(waveform, -1.0, 1.0)
    return waveform.astype(np.float32)


def write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(samples, -1.0, 0.999969482421875)
    pcm = (clipped * 32768.0).astype(np.int16)
    import wave

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


def prepare_frame(samples: np.ndarray, frame_length_sec: float, sample_rate: int) -> np.ndarray:
    target_length = int(round(frame_length_sec * sample_rate))
    if target_length <= len(samples):
        return samples[:target_length]
    return np.pad(samples, (0, target_length - len(samples)))


def compute_ccf(samples: np.ndarray, sample_rate: int, config: CCFConfig) -> np.ndarray:
    extractor = create_extractor(config)
    pdf, _ = extractor.compute(memoryview(samples), sample_rate)
    return np.array(pdf)


def plot_ccf(pdf: np.ndarray, config: CCFConfig, tones: tuple[float, ...], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    positions = np.linspace(0.0, 1.0, num=pdf.size, endpoint=False)

    plt.figure(figsize=(10, 5))
    plt.plot(positions, pdf, label="CCF PDF", linewidth=2)
    for freq in tones:
        coord = math.fmod(math.log2(freq / config.f_ref_hz), 1.0)
        coord = coord if coord >= 0 else coord + 1.0
        plt.axvline(coord, color="red", linestyle="--", alpha=0.6, label=f"{freq:.0f} Hz")
    plt.xlabel("Log-cycle position (fraction)")
    plt.ylabel("Probability density")
    plt.title("Continuous chroma PDF for mixed tones")
    handles, labels = plt.gca().get_legend_handles_labels()
    # Deduplicate legend entries for repeated labels.
    unique = dict(zip(labels, handles))
    plt.legend(unique.values(), unique.keys(), loc="upper right")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def main() -> None:
    args = parse_args()
    tones = DEFAULT_TONES
    samples = synthesise_tones(
        tones=tones,
        sample_rate=args.sample_rate,
        duration=args.duration,
        amplitude=args.amplitude,
    )
    write_wav(args.output_wav, samples, args.sample_rate)

    frame = prepare_frame(samples, args.frame_length, args.sample_rate)
    ccf_config = CCFConfig(n_bins=args.n_bins, smoothing_sigma_bins=args.smoothing)
    pdf = compute_ccf(frame, args.sample_rate, ccf_config)
    plot_ccf(pdf, ccf_config, tones, args.output_plot)

    print(f"Wrote WAV file to {args.output_wav}")
    print(f"Wrote CCF plot to {args.output_plot}")


if __name__ == "__main__":
    main()

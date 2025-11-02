#!/usr/bin/env python3
"""Plot the CCF and SKE tonal distribution for a given audio file."""

from __future__ import annotations

import argparse
import wave
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ccf_key_detector.features import CCFConfig, create_extractor
from ccf_key_detector.ske import SKEBackend, SKEConfig, create_ske, expand_profile
from ccf_key_detector.ske.kk_major import KK_MAJOR_PROFILE
from ccf_key_detector.transpose import TranspositionConfig, create_operator, scan_distribution


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "audio_path",
        type=Path,
        help="Path to the audio file to analyse (mono WAV recommended)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("build/ske_distribution.png"),
        help="Destination image path for the plot",
    )
    parser.add_argument("--n-bins", type=int, default=180, help="Number of CCF bins")
    parser.add_argument("--smoothing", type=float, default=1.0, help="CCF smoothing sigma (bins)")
    parser.add_argument(
        "--frame-length",
        type=float,
        default=10.0,
        help="Frame length in seconds (audio will be padded or truncated)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=48_000,
        help="Target sample rate; audio will be resampled if needed",
    )
    return parser.parse_args()


def load_audio(path: Path, sample_rate: int) -> np.ndarray:
    with wave.open(str(path), "rb") as wf:
        src_rate = wf.getframerate()
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        n_frames = wf.getnframes()
        pcm = wf.readframes(n_frames)

    if sample_width == 2:
        dtype = np.int16
        norm = 32768.0
    elif sample_width == 4:
        dtype = np.int32
        norm = 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sample_width}")

    samples = np.frombuffer(pcm, dtype=dtype).astype(np.float32) / norm
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)

    if src_rate == sample_rate:
        return samples

    duration = len(samples) / src_rate
    target_length = int(round(duration * sample_rate))
    if target_length <= 0:
        return np.zeros(1, dtype=np.float32)
    src_times = np.linspace(0.0, duration, num=len(samples), endpoint=False)
    dst_times = np.linspace(0.0, duration, num=target_length, endpoint=False)
    resampled = np.interp(dst_times, src_times, samples)
    return resampled.astype(np.float32)


def prepare_frame(samples: np.ndarray, frame_length_sec: float, sample_rate: int) -> np.ndarray:
    target_len = int(round(frame_length_sec * sample_rate))
    if target_len <= len(samples):
        return samples[:target_len]
    return np.pad(samples, (0, target_len - len(samples)))


def main() -> None:
    args = parse_args()

    samples = load_audio(args.audio_path, args.sample_rate)
    frame = prepare_frame(samples, args.frame_length, args.sample_rate)

    extractor = create_extractor(CCFConfig(n_bins=args.n_bins, smoothing_sigma_bins=args.smoothing))
    ccf_pdf, _ = extractor.compute(frame, args.sample_rate)
    ccf_pdf = np.asarray(ccf_pdf, dtype=np.float64)

    operator = create_operator(TranspositionConfig(n_bins=args.n_bins))
    ske = create_ske(SKEConfig(mode=SKEBackend.KK_MAJOR))
    tonal_pdf = scan_distribution(ccf_pdf, ske, operator)

    kk_profile = expand_profile(KK_MAJOR_PROFILE, args.n_bins)
    if kk_profile.sum() > 0:
        kk_profile = kk_profile / kk_profile.sum()

    x = np.linspace(0.0, 1.0, num=args.n_bins, endpoint=False)

    plt.figure(figsize=(12, 5))
    plt.plot(x, ccf_pdf, label="CCF PDF", color="#1f77b4")
    plt.plot(x, tonal_pdf, label="SKE distribution", color="#2ca02c")
    plt.plot(x, kk_profile, label="KK profile", color="#ff7f0e", linestyle="--")
    plt.xlabel("Cycle position (fraction)")
    plt.ylabel("Density")
    plt.title(f"CCF and Tonal Distribution for {args.audio_path}")
    plt.legend()
    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output)
    plt.close()

    peak_idx = int(np.argmax(tonal_pdf))
    print(f"Saved plot to {args.output}")
    print(f"Peak tonal bin: {peak_idx} with probability {tonal_pdf[peak_idx]:.3f}")


if __name__ == "__main__":
    main()

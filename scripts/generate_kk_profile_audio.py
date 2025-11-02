#!/usr/bin/env python3
"""Synthesize a KK-major-profile-matched waveform and report SKE score."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ccf_key_detector.features import CCFConfig, create_extractor
from ccf_key_detector.ske import SKEBackend, SKEConfig, create_ske, reduce_cycle
from ccf_key_detector.ske.kk_major import KK_MAJOR_PROFILE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=5.0, help="Duration of the waveform (s)")
    parser.add_argument("--sample-rate", type=int, default=48_000, help="Sample rate (Hz)")
    parser.add_argument("--output", type=Path, default=Path("build/kk_profile.wav"), help="WAV output path")
    return parser.parse_args()


def synthesise(duration: float, sample_rate: int) -> np.ndarray:
    """Synthesize a normalised waveform that follows the KK major amplitudes."""

    amplitudes = compute_amplitudes(sample_rate, duration)
    t = np.arange(int(sample_rate * duration), dtype=np.float64) / sample_rate
    waveform = np.zeros_like(t)
    for idx, amplitude in enumerate(amplitudes):
        freq = 220.0 * (2.0 ** (idx / 12.0))
        waveform += amplitude * np.sin(2.0 * np.pi * freq * t)
    waveform /= np.max(np.abs(waveform))
    return waveform.astype(np.float32)


def write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(samples, -1.0, 0.999969482421875)
    pcm = (pcm * 32768.0).astype(np.int16)
    import wave

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


def main() -> None:
    args = parse_args()
    waveform = synthesise(args.duration, args.sample_rate)
    write_wav(args.output, waveform, args.sample_rate)

    extractor = create_extractor(CCFConfig())
    pdf, _ = extractor.compute(memoryview(waveform), args.sample_rate)

    ske = create_ske(SKEConfig(mode=SKEBackend.KK_MAJOR))
    score = ske.evaluate(pdf)

    print(f"Wrote {args.output}")
    print(f"SKE score: {score:.4f}")


def compute_amplitudes(sample_rate: int, duration: float) -> np.ndarray:
    """Solve for partial amplitudes whose CCF matches the KK major profile."""

    extractor = create_extractor(CCFConfig(smoothing_sigma_bins=0.0))
    t = np.arange(int(sample_rate * duration), dtype=np.float64) / sample_rate
    basis = []
    for idx in range(12):
        freq = 220.0 * (2.0 ** (idx / 12.0))
        sine = np.sin(2.0 * np.pi * freq * t).astype(np.float32)
        pdf, _ = extractor.compute(memoryview(sine), sample_rate)
        reduced = reduce_cycle(pdf, 12)
        basis.append(reduced)
    basis_matrix = np.stack(basis, axis=1)
    target = np.array(KK_MAJOR_PROFILE, dtype=np.float64)
    target /= target.sum()
    active = np.ones(12, dtype=bool)
    weights = np.zeros(12, dtype=np.float64)
    for _ in range(12):
        # Impose a simplex constraint by solving for non-negative weights whose sum is one.
        B = basis_matrix[:, active]
        A = np.vstack([B, np.ones((1, B.shape[1]), dtype=np.float64)])
        b = np.concatenate([target, [1.0]])
        solution, *_ = np.linalg.lstsq(A, b, rcond=None)
        weights[active] = solution
        if np.all(weights[active] >= -1e-9):
            weights = np.clip(weights, 0.0, None)
            break
        worst = np.argmin(np.where(active, weights, np.inf))
        active[worst] = False
    weights = np.clip(weights, 0.0, None)
    total = weights.sum()
    if total == 0.0:
        weights = np.full_like(weights, 1.0 / weights.size)
    else:
        weights /= total
    amplitudes = np.sqrt(weights)
    return amplitudes


if __name__ == "__main__":
    main()

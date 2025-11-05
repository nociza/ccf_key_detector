"""Synthesize a tiny dataset of tonal snippets for Stage S1 experiments.

Each item is a short mono waveform designed to exercise different tonal
centers and interval structures. The script also performs lightweight
sanity checks on the resulting continuous chroma PDFs so the dataset can be
used immediately for PT1–PT2 pretraining.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import soundfile as sf

from ccf_key_detector.features.ccf import CCFConfig, create_extractor


DEFAULT_SAMPLE_RATE = 48_000
DEFAULT_FRAME_LENGTH = 1.5  # seconds


@dataclass(frozen=True)
class SnippetSpec:
    """Parameterisation of a synthetic snippet."""

    label: str
    tones: Sequence[float]
    detune_cents: Sequence[float] = ()
    add_noise: bool = False
    amplitude: float = 0.3


def _tone(
    freq_hz: float,
    amplitude: float,
    duration: float,
    sample_rate: int,
) -> np.ndarray:
    t = np.arange(int(duration * sample_rate), dtype=np.float64) / sample_rate
    return (amplitude * np.sin(2.0 * math.pi * freq_hz * t)).astype(np.float64)


def _render_snippet(
    spec: SnippetSpec,
    sample_rate: int,
    frame_length_sec: float,
) -> np.ndarray:
    frame = np.zeros(int(frame_length_sec * sample_rate), dtype=np.float64)
    detunes = list(spec.detune_cents) or [0.0 for _ in spec.tones]
    for freq_hz, cents in zip(spec.tones, detunes):
        detuned = freq_hz * (2.0 ** (cents / 1200.0))
        frame += _tone(detuned, spec.amplitude / max(len(spec.tones), 1), frame_length_sec, sample_rate)
    if spec.add_noise:
        rng = np.random.default_rng()
        frame += rng.normal(0.0, spec.amplitude / 4.0, frame.shape)
    frame = np.clip(frame, -0.95, 0.95)
    return frame.astype(np.float32)


def _snippet_library(f_ref: float) -> List[SnippetSpec]:
    fifth = f_ref * (3.0 / 2.0)
    fourth = f_ref * (4.0 / 3.0)
    major_third = f_ref * (5.0 / 4.0)
    minor_third = f_ref * (6.0 / 5.0)
    major_seventh = f_ref * (15.0 / 8.0)
    diminished_fifth = f_ref * math.sqrt(2.0)
    return [
        SnippetSpec("single_A3", (f_ref,)),
        SnippetSpec("octave_A4", (f_ref, 2 * f_ref)),
        SnippetSpec("fifth", (f_ref, fifth)),
        SnippetSpec("fourth", (f_ref, fourth)),
        SnippetSpec("major_tri", (f_ref, major_third, fifth)),
        SnippetSpec("minor_tri", (f_ref, minor_third, fifth)),
        SnippetSpec("dom7", (f_ref, major_third, fifth, major_seventh)),
        SnippetSpec("half_dim7", (f_ref, minor_third, diminished_fifth, major_seventh)),
        SnippetSpec("detuned_fifth", (f_ref, fifth), detune_cents=(0.0, 35.0)),
        SnippetSpec("noisy_major", (f_ref, major_third, fifth), add_noise=True),
    ]


def _repeat_snippets(
    base_specs: Sequence[SnippetSpec],
    n_samples: int,
    rng: np.random.Generator,
) -> List[SnippetSpec]:
    specs: List[SnippetSpec] = []
    while len(specs) < n_samples:
        base = rng.choice(base_specs)
        label = f"{base.label}_{len(specs):03d}"
        specs.append(SnippetSpec(label, base.tones, base.detune_cents, base.add_noise, base.amplitude))
    return specs


def _write_waveforms(
    specs: Sequence[SnippetSpec],
    output_dir: Path,
    sample_rate: int,
    frame_length_sec: float,
) -> List[Dict[str, str]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata: List[Dict[str, str]] = []
    for spec in specs:
        frame = _render_snippet(spec, sample_rate, frame_length_sec)
        path = output_dir / f"{spec.label}.wav"
        sf.write(str(path), frame, sample_rate)
        metadata.append({"path": str(path), "label": spec.label})
    return metadata


def _sanity_checks(
    files: Sequence[Dict[str, str]],
    ccf_config: CCFConfig,
    sample_rate: int,
) -> Dict[str, float]:
    extractor = create_extractor(ccf_config)
    sums: List[float] = []
    rms_values: List[float] = []
    concentrations: List[float] = []
    for entry in files:
        waveform, _ = sf.read(entry["path"], dtype="float32")
        pdf, diagnostics = extractor.compute(memoryview(waveform), sample_rate)
        sums.append(float(np.sum(pdf)))
        rms_values.append(float(diagnostics["rms"]))
        concentrations.append(float(np.sqrt(np.sum(np.asarray(pdf) ** 2))))
    return {
        "pdf_sum_mean": float(np.mean(sums)),
        "pdf_sum_std": float(np.std(sums)),
        "rms_mean": float(np.mean(rms_values)),
        "rms_std": float(np.std(rms_values)),
        "l2_concentration_mean": float(np.mean(concentrations)),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path, help="Directory to write the synthetic dataset")
    parser.add_argument("--n-samples", type=int, default=100, help="Number of snippets to generate")
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE, help="Sample rate (Hz)")
    parser.add_argument(
        "--frame-length",
        type=float,
        default=DEFAULT_FRAME_LENGTH,
        help="Length of each snippet in seconds",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for snippet shuffling",
    )
    parser.add_argument(
        "--save-metadata",
        action="store_true",
        help="Write a metadata.json file with snippet descriptions",
    )
    parser.add_argument(
        "--ccf-bins",
        type=int,
        default=120,
        help="Number of bins for sanity-check computations",
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=1.0,
        help="Circular smoothing sigma (bins) for sanity checks",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    rng = np.random.default_rng(args.seed)
    base_specs = _snippet_library(CCFConfig().f_ref_hz)
    specs = _repeat_snippets(base_specs, args.n_samples, rng)
    files = _write_waveforms(specs, args.output_dir, args.sample_rate, args.frame_length)
    stats = _sanity_checks(
        files,
        CCFConfig(n_bins=args.ccf_bins, smoothing_sigma_bins=args.smoothing),
        args.sample_rate,
    )

    print(f"Generated {len(files)} snippets in {args.output_dir}")
    for key, value in stats.items():
        print(f"{key}: {value:.6f}")

    if args.save_metadata:
        metadata_path = args.output_dir / "metadata.json"
        with metadata_path.open("w", encoding="utf-8") as fh:
            json.dump({"snippets": [asdict(spec) for spec in specs], "stats": stats}, fh, indent=2)
        print(f"Wrote metadata to {metadata_path}")


if __name__ == "__main__":
    main()

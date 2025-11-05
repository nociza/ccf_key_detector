"""Precompute continuous-chroma, CICV, and center-field features for a dataset."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import numpy as np

from ccf_key_detector.data import FeatureDatasetConfig, VocalFeatureDataset
from ccf_key_detector.data.precomputed import PrecomputeConfig
from ccf_key_detector.features import CCFConfig


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio_root", type=Path, help="Directory containing audio files")
    parser.add_argument("--output-dir", type=Path, default=Path("build/precomputed"))
    parser.add_argument("--sample-rate", type=int, default=48_000)
    parser.add_argument("--frame-length", type=float, default=1.5, help="Frame length in seconds")
    parser.add_argument("--hop-length", type=float, default=1.5, help="Hop length in seconds")
    parser.add_argument("--n-bins", type=int, default=180, help="Number of chroma bins")
    parser.add_argument("--smoothing", type=float, default=1.0, help="Circular smoothing (bins)")
    parser.add_argument("--include-torus", action="store_true", help="Store interval torus per frame")
    parser.add_argument("--skip-folded-cicv", action="store_true", help="Do not store folded CICV")
    parser.add_argument("--normalize-audio", action="store_true", help="Normalize audio per frame before feature extraction")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    parser.add_argument("--manifest-name", type=str, default="manifest.json")
    parser.add_argument("--max-files", type=int, default=None, help="Process at most N audio files")
    return parser.parse_args()


def _ensure_relative(path: Path, root: Path) -> Path:
    try:
        return path.resolve().relative_to(root.resolve())
    except ValueError as exc:  # pragma: no cover - guardrail
        raise RuntimeError(f"{path} is outside dataset root {root}") from exc


def main() -> None:
    args = _parse_args()
    audio_root = args.audio_root.resolve()
    if not audio_root.exists():
        raise FileNotFoundError(audio_root)

    ccf_cfg = CCFConfig(n_bins=args.n_bins, smoothing_sigma_bins=args.smoothing)
    pre_cfg = PrecomputeConfig(
        sample_rate=args.sample_rate,
        frame_length_sec=args.frame_length,
        hop_length_sec=args.hop_length,
        ccf=ccf_cfg,
        include_torus=args.include_torus,
        include_folded_cicv=not args.skip_folded_cicv,
    )
    feature_cfg = FeatureDatasetConfig(
        sample_rate=args.sample_rate,
        frame_length_sec=args.frame_length,
        hop_length_sec=args.hop_length,
        ccf=ccf_cfg,
        normalize_audio=args.normalize_audio,
        include_torus=args.include_torus,
    )

    dataset = VocalFeatureDataset(audio_root, config=feature_cfg)
    target_root = args.output_dir.resolve() / f"cfg_{pre_cfg.cfg_id()}"
    manifest_path = target_root / args.manifest_name
    if manifest_path.exists() and not args.overwrite:
        raise RuntimeError(f"{manifest_path} already exists; use --overwrite to replace it")

    aggregation: Dict[Path, Dict[str, List[np.ndarray]]] = {}
    diag_buffers: Dict[Path, Dict[str, List[float]]] = {}
    starts: Dict[Path, List[int]] = {}
    counts: Dict[Path, int] = defaultdict(int)

    for idx in range(len(dataset)):
        sample = dataset[idx]
        metadata = sample["metadata"]
        audio_path = Path(metadata["path"]).resolve()
        rel_path = _ensure_relative(audio_path, audio_root)

        if args.max_files is not None and len(aggregation) >= args.max_files and rel_path not in aggregation:
            break

        record = aggregation.setdefault(
            rel_path,
            {
                "ccf": [],
                "cicv": [],
                "center": [],
                "mu": [],
                "rho": [],
            },
        )
        record["ccf"].append(sample["ccf"].numpy())
        record["cicv"].append(sample["cicv"].numpy())
        record["center"].append(sample["center_field"].numpy())
        record["mu"].append(float(sample["mu"]))
        record["rho"].append(float(sample["rho"]))

        if not args.skip_folded_cicv:
            record.setdefault("cicv_folded", []).append(sample["cicv_folded"].numpy())
        if args.include_torus and "torus" in sample:
            record.setdefault("torus", []).append(sample["torus"].numpy())

        starts.setdefault(rel_path, []).append(int(metadata["start"]))
        diags = diag_buffers.setdefault(rel_path, defaultdict(list))
        for key, value in sample["diagnostics"].items():
            diags[key].append(float(value))
        counts[rel_path] += 1

    target_root.mkdir(parents=True, exist_ok=True)
    entries: List[Dict[str, object]] = []

    for rel_path, data in aggregation.items():
        out_path = target_root / rel_path.with_suffix(".npz")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        arrays = {name: np.stack(values).astype(np.float32) for name, values in data.items() if name not in {"mu", "rho"}}
        arrays["mu"] = np.asarray(data["mu"], dtype=np.float32)
        arrays["rho"] = np.asarray(data["rho"], dtype=np.float32)
        arrays["starts"] = np.asarray(starts[rel_path], dtype=np.int64)

        diag = diag_buffers.get(rel_path, {})
        for key, values in diag.items():
            arrays[f"diag_{key}"] = np.asarray(values, dtype=np.float32)

        np.savez_compressed(out_path, **arrays)

        entries.append(
            {
                "audio": str(rel_path),
                "features": str(out_path.relative_to(target_root)),
                "n_frames": int(arrays["mu"].shape[0]),
                "sample_rate": args.sample_rate,
                "frame_length_sec": args.frame_length,
                "hop_length_sec": args.hop_length,
                "starts": arrays["starts"].tolist(),
            }
        )

    manifest = {
        "cfg_id": pre_cfg.cfg_id(),
        "config": pre_cfg.to_dict(),
        "audio_root": str(audio_root),
        "entries": entries,
    }
    manifest_path = target_root / args.manifest_name
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"Wrote {len(entries)} feature files to {target_root}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()

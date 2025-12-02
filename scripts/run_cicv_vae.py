#!/usr/bin/env python3
"""Realtime CICV-VAE scoring pipeline (no SKE scan)."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import numpy as np

from ccf_key_detector.audio_io import AudioStreamConfig, create_audio_stream
from ccf_key_detector.features import CCFConfig, compute_cicv, create_extractor
from ccf_key_detector.models.cicv_vae import CICVVAE, CICVVAEConfig
from ccf_key_detector.prefilter import PreFilterConfig, create_prefilter


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-file", type=Path, help="Optional WAV file for offline playback")
    parser.add_argument("--sample-rate", type=int, default=48_000)
    parser.add_argument("--frame-length", type=float, default=1.0)
    parser.add_argument("--hop-length", type=float, default=0.5)
    parser.add_argument("--n-bins", type=int, default=720)
    parser.add_argument("--checkpoint", type=Path, required=True, help="CICV-VAE checkpoint (.pt)")
    parser.add_argument("--temperature", type=float, default=35.0, help="Score temperature")
    parser.add_argument("--device", type=str, default=None, help="Override torch device (cpu/cuda)")
    parser.add_argument("--max-hops", type=int, default=None, help="Optional cap on hop count")
    parser.add_argument("--print-stats", action="store_true", help="Print diagnostics per hop")
    return parser.parse_args()


def _load_model(checkpoint_path: Path, device: torch.device) -> CICVVAE:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_cfg = CICVVAEConfig(**checkpoint["config"])
    model = CICVVAE(model_cfg).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = _load_model(args.checkpoint, device)

    prefilter = create_prefilter(PreFilterConfig(enable=True))
    ccf_config = CCFConfig(n_bins=args.n_bins)
    extractor = create_extractor(ccf_config)
    stream_cfg = AudioStreamConfig(
        sample_rate=args.sample_rate,
        frame_length_sec=args.frame_length,
        hop_length_sec=args.hop_length,
        use_live_input=args.audio_file is None,
        file_path=str(args.audio_file) if args.audio_file else None,
    )
    audio_stream = create_audio_stream(stream_cfg)

    print(f"Running CICV-VAE scoring ({'live input' if args.audio_file is None else args.audio_file})")
    hop_count = 0
    for frame_idx, frame in audio_stream.frames():
        processed = prefilter.process(frame, args.sample_rate)
        pdf, diagnostics = extractor.compute(processed, args.sample_rate)
        pdf_arr = np.asarray(pdf, dtype=np.float32)
        cicv = compute_cicv(pdf_arr).astype(np.float32)
        tensor = torch.from_numpy(cicv).unsqueeze(0).to(device)
        with torch.no_grad():
            recon, _, _, _ = model(tensor)
            loss = F.mse_loss(recon, tensor).item()
        score = math.exp(-args.temperature * loss)

        if args.print_stats:
            rms = diagnostics.get("rms", 0.0)
            flatness = diagnostics.get("spectral_flatness", 0.0)
            print(
                f"hop={frame_idx:06d} loss={loss:.6e} score={score:.4f} "
                f"rms={rms:.4f} flatness={flatness:.4f}"
            )

        hop_count += 1
        if args.max_hops is not None and hop_count >= args.max_hops:
            break

    audio_stream.stop()  # type: ignore[union-attr]


if __name__ == "__main__":
    main()

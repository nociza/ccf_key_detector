#!/usr/bin/env python3
"""Run the realtime CCF pipeline with optional live audio and visualization."""

from __future__ import annotations

import argparse
import signal
from dataclasses import replace
from pathlib import Path
from typing import Optional

import numpy as np

from ccf_key_detector.app import FrameResult, RunnerConfig, create_runner
from ccf_key_detector.prefilter import PreFilterConfig
from ccf_key_detector.rt_viz import (
    VisualizationConfig,
    VisualizationFrame,
    VisualizationMode,
    create_visualizer,
)


def _parse_device(raw: Optional[str]) -> Optional[int | str]:
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return raw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-file", type=Path, help="Optional WAV file for offline playback")
    parser.add_argument("--sample-rate", type=int, default=48_000, help="Processing sample rate (Hz)")
    parser.add_argument("--frame-length", type=float, default=10.0, help="Frame length in seconds")
    parser.add_argument("--hop-length", type=float, default=1.0, help="Hop length in seconds")
    parser.add_argument(
        "--n-bins", type=int, default=180, help="Number of bins for the continuous chroma PDF"
    )
    parser.add_argument(
        "--viz-mode",
        choices=[mode.value for mode in VisualizationMode],
        default=VisualizationMode.LINEAR.value,
        help="Visualization mode",
    )
    parser.add_argument(
        "--window-hops",
        type=int,
        default=30,
        help="Number of hops retained in the visualizer ring buffer",
    )
    parser.add_argument(
        "--tail-length",
        type=int,
        default=3,
        help="Tail length for polar visualization (ignored in cylinder mode)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional PortAudio device identifier for live capture (index or name)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without rendering (useful for dry runs/logging)",
    )
    parser.add_argument(
        "--normalize-ske-dist",
        action="store_true",
        help="Normalize SKE distribution (default: raw scores)",
    )
    parser.add_argument(
        "--print-stats",
        action="store_true",
        help="Print diagnostics (including SKE score) to stdout each hop",
    )
    parser.add_argument(
        "--osc-host",
        type=str,
        default="127.0.0.1",
        help="Destination host for OSC streaming (requires an OSC viz-mode)",
    )
    parser.add_argument(
        "--osc-port",
        type=int,
        default=7400,
        help="Destination UDP port for OSC streaming",
    )
    parser.add_argument(
        "--osc-address",
        type=str,
        default="/ccf/pdf",
        help="OSC address pattern used for streamed packets",
    )
    parser.add_argument(
        "--osc-skip-frame-index",
        action="store_true",
        help="Omit the frame index from OSC payloads",
    )
    parser.add_argument(
        "--osc-max-packet-size",
        type=int,
        default=65_507,
        help="Maximum OSC datagram payload size before raising an error",
    )
    return parser.parse_args()


def low_confidence_heuristic(result: FrameResult) -> bool:
    diagnostics = result.diagnostics
    rms = diagnostics.get("rms", 0.0)
    flatness = diagnostics.get("spectral_flatness", 1.0)
    return rms < 1e-3 or flatness > 0.6


def main() -> None:
    args = parse_args()

    use_live_input = args.audio_file is None
    pre_filter = PreFilterConfig(enable=True)

    runner_config = RunnerConfig(
        sample_rate=args.sample_rate,
        frame_length_sec=args.frame_length,
        hop_length_sec=args.hop_length,
        n_bins=args.n_bins,
        use_live_input=use_live_input,
        audio_file_path=str(args.audio_file) if args.audio_file else None,
        pre_filter=pre_filter,
        audio_input_device=_parse_device(args.device),
    )
    runner_config.ccf = replace(runner_config.ccf, n_bins=args.n_bins)

    runner = create_runner(runner_config)

    viz_config = VisualizationConfig(
        mode=VisualizationMode(args.viz_mode),
        window_hops=args.window_hops,
        tail_length=args.tail_length,
        render=not args.headless,
        normalize_ske_dist=args.normalize_ske_dist,
        osc_host=args.osc_host,
        osc_port=args.osc_port,
        osc_address=args.osc_address,
        osc_include_frame_index=not args.osc_skip_frame_index,
        osc_max_packet_size=args.osc_max_packet_size,
    )
    visualizer = create_visualizer(viz_config)

    def shutdown_handler(signum, frame):  # noqa: ANN001
        runner.close()
        visualizer.close()

    signal.signal(signal.SIGINT, shutdown_handler)

    try:
        for result in runner.iterate():
            frame = VisualizationFrame(
                pdf=result.pdf,
                low_confidence=low_confidence_heuristic(result),
                diagnostics=result.diagnostics,
            )
            visualizer.push(result.frame_index, frame)
            if args.print_stats:
                ske_score = result.diagnostics.get("ske_score", 0.0)
                distribution = result.diagnostics.get("ske_distribution")
                top_desc = ""
                if isinstance(distribution, list):
                    dist = np.array(distribution, dtype=np.float64)
                    top_indices = dist.argsort()[::-1][:3]
                    top_desc = ", ".join(
                        f"{idx}:{dist[idx]:.3f}" for idx in top_indices
                    )
                    top_desc = f" top[{top_desc}]"
                print(
                    f"hop={result.frame_index} ske={ske_score:.3f} "
                    f"rms={result.diagnostics.get('rms', 0.0):.4f} "
                    f"flat={result.diagnostics.get('spectral_flatness', 0.0):.3f}{top_desc}"
                )
    except KeyboardInterrupt:
        pass
    finally:
        visualizer.close()
        runner.close()


if __name__ == "__main__":
    main()

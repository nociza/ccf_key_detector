from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Deque, Optional

import numpy as np

if TYPE_CHECKING:
    from . import VisualizationConfig, VisualizationFrame

from ccf_key_detector.ske import expand_profile
from ccf_key_detector.ske.kk_major import KK_MAJOR_PROFILE


@dataclass
class _LineState:
    tail: Deque[np.ndarray]
    tail_indices: Deque[int]
    figure: Optional["matplotlib.figure.Figure"] = None
    axis: Optional["matplotlib.axes.Axes"] = None
    main_line: Optional["matplotlib.lines.Line2D"] = None
    tail_lines: list["matplotlib.lines.Line2D"] | None = None
    profile_line: Optional["matplotlib.lines.Line2D"] = None
    ske_line: Optional["matplotlib.lines.Line2D"] = None
    profile_cache: tuple[int, np.ndarray] | None = None
    diagnostics_text: Optional["matplotlib.text.Text"] = None
    plt: Optional["module"] = None


class LineVisualizer:
    """Linear (cartesian) visualizer mirroring the PNG diagnostic plot."""

    def __init__(self, config: "VisualizationConfig") -> None:
        self.config = config
        self._state = _LineState(
            tail=deque(maxlen=max(1, config.tail_length)),
            tail_indices=deque(maxlen=max(1, config.tail_length)),
        )
        if self.config.render:
            self._init_matplotlib()

    def _init_matplotlib(self) -> None:
        import matplotlib
        import matplotlib.pyplot as plt

        matplotlib.rcParams["toolbar"] = "toolmanager"
        plt.ion()
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.set_xlabel("Cycle position (fraction)")
        ax.set_ylabel("Probability density")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        (line,) = ax.plot([], [], color="#1f77b4", linewidth=2)
        (profile_line,) = ax.plot([], [], color="#ff7f0e", linestyle="--", linewidth=1.5)
        (ske_line,) = ax.plot([], [], color="#2ca02c", linewidth=1.5, alpha=0.7)
        tail_lines = [
            ax.plot([], [], color="#1f77b4", linewidth=1, alpha=max(0.1, 0.5 - i * 0.15))[0]
            for i in range(self.config.tail_length)
        ]
        diagnostics_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, ha="left", va="top")
        self._state.figure = fig
        self._state.axis = ax
        self._state.main_line = line
        self._state.tail_lines = tail_lines
        self._state.profile_line = profile_line
        self._state.ske_line = ske_line
        self._state.diagnostics_text = diagnostics_text
        self._state.plt = plt

    def push(self, frame_index: int, frame: "VisualizationFrame") -> None:
        pdf = np.asarray(frame.pdf, dtype=np.float64)
        if pdf.ndim != 1:
            raise ValueError("LineVisualizer expects 1-D pdf vectors")

        self._state.tail.append(pdf)
        self._state.tail_indices.append(frame_index)

        if not self.config.render:
            return

        if self._state.axis is None:
            self._init_matplotlib()
            if self._state.axis is None:
                return

        self._update_plot(frame_index, frame)

    def _update_plot(self, frame_index: int, frame: "VisualizationFrame") -> None:
        assert self._state.axis is not None
        assert self._state.main_line is not None
        assert self._state.tail_lines is not None
        assert self._state.plt is not None
        assert self._state.diagnostics_text is not None
        assert self._state.figure is not None
        assert self._state.profile_line is not None
        assert self._state.ske_line is not None

        pdf = self._state.tail[-1]
        n_bins = pdf.size
        x = np.linspace(0.0, 1.0, num=n_bins, endpoint=False)

        profile = self._ensure_profile(n_bins)
        ske_curve = self._ske_curve(frame.diagnostics.get("ske_distribution"), n_bins, pdf.max())

        max_candidates = [pdf.max(), profile.max()]
        if ske_curve.size:
            max_candidates.append(ske_curve.max())
        max_val = float(max(max_candidates)) if pdf.size else 0.0
        y_max = max(0.05, max_val * 1.05)
        self._state.axis.set_ylim(0.0, y_max)

        color = "#d62728" if frame.low_confidence else "#1f77b4"
        self._state.main_line.set_data(x, pdf)
        self._state.main_line.set_color(color)
        self._state.profile_line.set_data(x, profile)
        if ske_curve.size:
            self._state.ske_line.set_data(x, ske_curve)
            self._state.ske_line.set_alpha(0.7)
        else:
            self._state.ske_line.set_data([], [])

        for idx, line in enumerate(self._state.tail_lines):
            if idx >= len(self._state.tail) - 1:
                line.set_data([], [])
                continue
            tail_pdf = self._state.tail[-(idx + 2)]
            tail_x = np.linspace(0.0, 1.0, num=tail_pdf.size, endpoint=False)
            line.set_data(tail_x, tail_pdf)

        rms = frame.diagnostics.get("rms")
        flatness = frame.diagnostics.get("spectral_flatness")
        latency_ms = frame.diagnostics.get("processing_latency_ms")
        status = frame.diagnostics.get("audio_status")
        ske_score = frame.diagnostics.get("ske_score")
        lines = [f"frame {frame_index}"]
        if isinstance(rms, (int, float)):
            lines.append(f"RMS {float(rms):.4f}")
        if isinstance(flatness, (int, float)):
            lines.append(f"Flat {float(flatness):.3f}")
        if isinstance(latency_ms, (int, float)):
            lines.append(f"Latency {float(latency_ms):.1f} ms")
        if isinstance(ske_score, (int, float)):
            lines.append(f"SKE {float(ske_score):.3f}")
        if status:
            lines.append(status)
        self._state.diagnostics_text.set_text(" | ".join(lines))

        self._state.figure.canvas.draw_idle()
        self._state.plt.pause(max(1.0 / self.config.fps, 0.001))

    def close(self) -> None:
        if not self.config.render:
            return
        if self._state.plt is None:
            return
        try:
            self._state.plt.ioff()
            if self._state.figure is not None:
                self._state.plt.close(self._state.figure)
        finally:
            self._state.figure = None
            self._state.axis = None
            self._state.main_line = None
            self._state.tail_lines = None
            self._state.diagnostics_text = None
            self._state.plt = None
            self._state.profile_line = None
            self._state.ske_line = None
            self._state.profile_cache = None

    def _ensure_profile(self, n_bins: int) -> np.ndarray:
        if self._state.profile_cache is None or self._state.profile_cache[0] != n_bins:
            expanded = expand_profile(KK_MAJOR_PROFILE, n_bins)
            if expanded.sum() > 0.0:
                expanded = expanded / expanded.sum()
            self._state.profile_cache = (n_bins, expanded)
        profile = self._state.profile_cache[1]
        profile = profile.astype(np.float64, copy=True)
        return profile

    def _ske_curve(self, distribution: object, n_bins: int, peak: float) -> np.ndarray:
        if not isinstance(distribution, (list, tuple, np.ndarray)):
            return np.array([], dtype=np.float64)
        curve = np.asarray(distribution, dtype=np.float64)
        if curve.size != n_bins:
            return np.array([], dtype=np.float64)
        if peak > 0.0:
            curve = curve * peak
        return curve

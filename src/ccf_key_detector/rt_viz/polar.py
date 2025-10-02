from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Deque, Optional

import numpy as np

if TYPE_CHECKING:
    from . import VisualizationConfig, VisualizationFrame


@dataclass
class _PolarState:
    tail: Deque[np.ndarray]
    tail_indices: Deque[int]
    figure: Optional["matplotlib.figure.Figure"] = None
    axis: Optional["matplotlib.axes.Axes"] = None
    main_line: Optional["matplotlib.lines.Line2D"] = None
    tail_lines: list["matplotlib.lines.Line2D"] | None = None
    diagnostics_text: Optional["matplotlib.text.Text"] = None
    plt: Optional["module"] = None


class PolarVisualizer:
    """Polar plot visualizer for tonal-center PDFs."""

    def __init__(self, config: "VisualizationConfig") -> None:
        self.config = config
        self._state = _PolarState(
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
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="polar")
        ax.set_ylim(0.0, 1.0)
        ax.set_theta_direction(-1)
        ax.set_theta_offset(math.pi / 2)
        ax.grid(True, linestyle=":", alpha=0.4)
        (line,) = ax.plot([], [], color="#1f77b4", linewidth=2)
        tail_lines = [
            ax.plot([], [], color="#1f77b4", linewidth=1, alpha=max(0.1, 0.5 - i * 0.15))[0]
            for i in range(self.config.tail_length)
        ]
        diagnostics_text = ax.text(0.02, 0.02, "", transform=ax.transAxes, ha="left", va="bottom")

        self._state.figure = fig
        self._state.axis = ax
        self._state.main_line = line
        self._state.tail_lines = tail_lines
        self._state.diagnostics_text = diagnostics_text
        self._state.plt = plt

    def push(self, frame_index: int, frame: "VisualizationFrame") -> None:
        pdf = np.asarray(frame.pdf, dtype=np.float64)
        if pdf.ndim != 1:
            raise ValueError("PolarVisualizer expects 1-D pdf vectors")

        self._state.tail.append(pdf)
        self._state.tail_indices.append(frame_index)

        if not self.config.render:
            return

        self._update_plot(frame_index, frame)

    def _update_plot(self, frame_index: int, frame: VisualizationFrame) -> None:
        assert self._state.axis is not None
        assert self._state.main_line is not None
        assert self._state.tail_lines is not None
        assert self._state.plt is not None
        assert self._state.diagnostics_text is not None

        pdf = self._state.tail[-1]
        n_bins = pdf.size
        theta = np.linspace(0.0, 2 * np.pi, num=n_bins + 1)
        r = np.concatenate([pdf, pdf[:1]])

        color = "#d62728" if frame.low_confidence else "#1f77b4"
        self._state.main_line.set_data(theta, r)
        self._state.main_line.set_color(color)

        for idx, line in enumerate(self._state.tail_lines):
            if idx >= len(self._state.tail) - 1:
                line.set_data([], [])
                continue
            tail_pdf = self._state.tail[-(idx + 2)]
            tail_theta = np.linspace(0.0, 2 * np.pi, num=tail_pdf.size + 1)
            tail_r = np.concatenate([tail_pdf, tail_pdf[:1]])
            line.set_data(tail_theta, tail_r)

        rms = frame.diagnostics.get("rms")
        flatness = frame.diagnostics.get("spectral_flatness")
        latency_ms = frame.diagnostics.get("processing_latency_ms")
        ske_score = frame.diagnostics.get("ske_score")
        text_lines = [f"frame {frame_index}"]
        if isinstance(rms, (int, float)):
            text_lines.append(f"RMS {float(rms):.4f}")
        if isinstance(flatness, (int, float)):
            text_lines.append(f"Flat {float(flatness):.3f}")
        if isinstance(latency_ms, (int, float)):
            text_lines.append(f"Latency {float(latency_ms):.1f} ms")
        if isinstance(ske_score, (int, float)):
            text_lines.append(f"SKE {float(ske_score):.3f}")
        status = frame.diagnostics.get("audio_status")
        if status:
            text_lines.append(status)
        self._state.diagnostics_text.set_text("\n".join(text_lines))

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

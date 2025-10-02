from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Deque, Optional

import numpy as np

if TYPE_CHECKING:
    from . import VisualizationConfig, VisualizationFrame


@dataclass
class _CylinderState:
    ring_buffer: Deque[np.ndarray]
    indices: Deque[int]
    figure: Optional["matplotlib.figure.Figure"] = None
    axis: Optional["matplotlib.axes.Axes"] = None
    image: Optional["matplotlib.image.AxesImage"] = None
    plt: Optional["module"] = None


class CylinderVisualizer:
    """Scrolling image visualizer for tonal-center PDFs over time."""

    def __init__(self, config: "VisualizationConfig") -> None:
        self.config = config
        self._state = _CylinderState(
            ring_buffer=deque(maxlen=max(1, config.window_hops)),
            indices=deque(maxlen=max(1, config.window_hops)),
        )
        if self.config.render:
            self._init_matplotlib()

    def _init_matplotlib(self) -> None:
        import matplotlib
        import matplotlib.pyplot as plt

        matplotlib.rcParams["toolbar"] = "toolmanager"
        plt.ion()
        fig, ax = plt.subplots(figsize=(8, 4))
        data = np.zeros((self.config.window_hops, 1))
        image = ax.imshow(
            data,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            vmin=0.0,
            vmax=1.0,
            extent=[0.0, 1.0, 0, self.config.window_hops],
            cmap="viridis",
        )
        ax.set_xlabel("Cycle position")
        ax.set_ylabel("Hop")
        ax.set_title("Tonal-center probability surface")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="PDF")

        self._state.figure = fig
        self._state.axis = ax
        self._state.image = image
        self._state.plt = plt

    def push(self, frame_index: int, frame: "VisualizationFrame") -> None:
        pdf = np.asarray(frame.pdf, dtype=np.float64)
        if pdf.ndim != 1:
            raise ValueError("CylinderVisualizer expects 1-D pdf vectors")
        self._state.ring_buffer.append(pdf)
        self._state.indices.append(frame_index)

        if not self.config.render:
            return

        self._update_plot(frame_index, frame)

    def _update_plot(self, frame_index: int, frame: VisualizationFrame) -> None:
        assert self._state.image is not None
        assert self._state.axis is not None
        assert self._state.figure is not None
        assert self._state.plt is not None

        data = np.vstack(self._state.ring_buffer)
        # Normalize color range for better contrast.
        vmax = max(1e-6, float(data.max()))
        self._state.image.set_data(data)
        self._state.image.set_extent([0.0, 1.0, frame_index - data.shape[0] + 1, frame_index + 1])
        self._state.image.set_clim(0.0, vmax)

        status = frame.diagnostics.get("audio_status")
        latency_ms = frame.diagnostics.get("processing_latency_ms")
        ske_score = frame.diagnostics.get("ske_score")
        title_parts = [f"Hop {frame_index}"]
        if isinstance(latency_ms, (int, float)):
            title_parts.append(f"{float(latency_ms):.1f} ms")
        if isinstance(ske_score, (int, float)):
            title_parts.append(f"SKE {float(ske_score):.3f}")
        if status:
            title_parts.append(status)
        title = " — ".join(title_parts)
        self._state.axis.set_title(title)

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
            self._state.image = None
            self._state.plt = None

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Deque, Optional

import numpy as np

if TYPE_CHECKING:
    from . import VisualizationConfig, VisualizationFrame


@dataclass
class _SurfaceState:
    ccf_buffer: Deque[np.ndarray]
    ske_buffer: Deque[np.ndarray]
    indices: Deque[int]
    figure: Optional["matplotlib.figure.Figure"] = None
    ccf_axis: Optional["matplotlib.axes._axes.Axes"] = None
    ske_axis: Optional["matplotlib.axes._axes.Axes"] = None
    plt: Optional["module"] = None


class SurfaceVisualizer:
    """Matplotlib-based 3D surfaces for CCF and SKE distributions."""

    def __init__(self, config: VisualizationConfig) -> None:
        self.config = config
        maxlen = max(1, config.window_hops)
        self._state = _SurfaceState(
            ccf_buffer=deque(maxlen=maxlen),
            ske_buffer=deque(maxlen=maxlen),
            indices=deque(maxlen=maxlen),
        )
        self._normalize = getattr(config, "normalize_ske_dist", True)
        if self.config.render:
            self._init_matplotlib()

    def _init_matplotlib(self) -> None:
        import matplotlib
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        matplotlib.rcParams["toolbar"] = "toolmanager"
        plt.ion()
        fig = plt.figure(figsize=(12, 6))
        ax_ccf = fig.add_subplot(1, 2, 1, projection="3d")
        ax_ske = fig.add_subplot(1, 2, 2, projection="3d")

        ax_ccf.set_xlabel("Cycle position")
        ax_ccf.set_ylabel("Hop")
        ax_ccf.set_zlabel("CCF density")
        ax_ccf.set_title("Continuous Chroma Surface")

        ax_ske.set_xlabel("Cycle position")
        ax_ske.set_ylabel("Hop")
        ax_ske.set_zlabel("SKE value")
        ax_ske.set_title("SKE Distribution Surface")
        ax_ske.set_zlim(0.0, 1.0)

        self._state.figure = fig
        self._state.ccf_axis = ax_ccf
        self._state.ske_axis = ax_ske
        self._state.plt = plt

    def push(self, frame_index: int, frame: VisualizationFrame) -> None:
        ccf_pdf = np.asarray(frame.pdf, dtype=np.float64)
        if ccf_pdf.ndim != 1:
            raise ValueError("SurfaceVisualizer expects 1-D pdf vectors")
        ske_distribution = frame.diagnostics.get("ske_distribution")
        if not isinstance(ske_distribution, (list, tuple, np.ndarray)):
            return
        ske_pdf = np.asarray(ske_distribution, dtype=np.float64)
        if ske_pdf.shape != ccf_pdf.shape:
            return

        self._state.ccf_buffer.append(ccf_pdf)
        self._state.ske_buffer.append(ske_pdf)
        self._state.indices.append(frame_index)

        if not self.config.render:
            return
        if self._state.ccf_axis is None:
            self._init_matplotlib()
            if self._state.ccf_axis is None:
                return
        self._update_plot()

    def _update_plot(self) -> None:
        assert self._state.ccf_axis is not None
        assert self._state.ske_axis is not None
        assert self._state.figure is not None
        assert self._state.plt is not None

        ccf_matrix = np.vstack(self._state.ccf_buffer)
        ske_matrix = np.vstack(self._state.ske_buffer)
        if self._normalize:
            row_sums = ske_matrix.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            ske_matrix = ske_matrix / row_sums

        hop_count, n_bins = ccf_matrix.shape
        y_values = np.array(self._state.indices, dtype=np.float64)
        x_values = np.linspace(0.0, 1.0, num=n_bins, endpoint=False)
        X_grid, Y_grid = np.meshgrid(x_values, y_values)

        ax_ccf = self._state.ccf_axis
        ax_ske = self._state.ske_axis

        ax_ccf.clear()
        ax_ccf.plot_surface(X_grid, Y_grid, ccf_matrix, cmap="magma", linewidth=0, antialiased=False)
        ax_ccf.set_xlabel("Cycle position")
        ax_ccf.set_ylabel("Hop")
        ax_ccf.set_zlabel("CCF density")
        ax_ccf.set_title("Continuous Chroma Surface")

        ax_ske.clear()
        ax_ske.plot_surface(X_grid, Y_grid, ske_matrix, cmap="viridis", linewidth=0, antialiased=False)
        peak_bins = np.argmax(ske_matrix, axis=1)
        peak_heights = ske_matrix[np.arange(hop_count), peak_bins]
        peak_xs = (peak_bins + 0.5) / n_bins
        ax_ske.scatter(peak_xs, y_values, peak_heights, color="red", s=10)
        ax_ske.set_xlabel("Cycle position")
        ax_ske.set_ylabel("Hop")
        ax_ske.set_zlabel("SKE value")
        ax_ske.set_title("SKE Distribution Surface")
        ax_ske.set_zlim(0.0, 1.0)

        self._state.figure.canvas.draw_idle()
        self._state.plt.pause(max(1.0 / self.config.fps, 0.001))

    def close(self) -> None:
        if not self.config.render or self._state.plt is None:
            return
        try:
            self._state.plt.ioff()
            if self._state.figure is not None:
                self._state.plt.close(self._state.figure)
        finally:
            self._state.figure = None
            self._state.ccf_axis = None
            self._state.ske_axis = None
            self._state.plt = None
